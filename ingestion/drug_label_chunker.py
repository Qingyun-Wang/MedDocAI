"""
Drug Label Chunker — ingestion/drug_label_chunker.py

Reads all 13 FDA drug label zip files, deduplicates to one label per generic
drug name (prescription drugs only), chunks by section, embeds with OpenAI,
and upserts into Qdrant drug_labels collection.

Key design decisions:
  - Prescription-only: only records with openfda.generic_name field (~26K unique drugs)
  - OTC labels (no openfda field) are skipped — covered by live openFDA API
  - Dedup: most complete label per generic (highest filled-section count, then newest)
  - Qdrant: local path mode (data/qdrant_storage/) — no Docker required
  - Truncation: sections capped at 8,000 tokens before embedding

Cost: ~$1.92 (OpenAI text-embedding-3-small at $0.02/M tokens)

Usage:
    python -m ingestion.drug_label_chunker
    python -m ingestion.drug_label_chunker --dry-run   # count chunks, no API calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import uuid
import zipfile
from datetime import datetime
from typing import Optional

import time
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# Tokeniser for text-embedding-3-small (uses cl100k_base encoding)
_TOKENISER = tiktoken.get_encoding("cl100k_base")

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LABEL_DIR      = "data/openfda/drug/label"
QDRANT_PATH    = "data/qdrant_storage"
COLLECTION     = "drug_labels"
EMBED_MODEL    = "text-embedding-3-small"
EMBED_DIM      = 1536
BATCH_SIZE     = 100          # chunks per OpenAI embedding call
MAX_TOKENS     = 8_000        # hard token limit for OpenAI text-embedding-3-small (max 8,192)
# Using tiktoken for precise counting — medical text is denser than typical English
# (~3.5 chars/token vs ~4), so char estimates are unreliable for edge cases.

# Sections to extract, in priority order
# For warnings: use warnings_and_cautions if present, else warnings (rarely both exist)
TARGET_SECTIONS = [
    "indications_and_usage",
    "warnings_and_cautions",
    "warnings",
    "contraindications",
    "adverse_reactions",
    "dosage_and_administration",
]

# Only one warnings field per label — if warnings_and_cautions present, skip warnings
WARNINGS_PAIR = {"warnings_and_cautions", "warnings"}


# ---------------------------------------------------------------------------
# Step 1: Deduplication pass (streams all 13 zips in memory-efficient way)
# ---------------------------------------------------------------------------

def _section_score(record: dict) -> int:
    """Count how many key sections have non-empty content."""
    return sum(
        1 for s in TARGET_SECTIONS
        if record.get(s) and isinstance(record[s], list) and record[s][0].strip()
    )


def _effective_date(record: dict) -> str:
    """Return effective_time as a sortable string, empty string if missing."""
    return record.get("effective_time", "") or ""


def deduplicate_labels(label_dir: str = LABEL_DIR) -> dict[str, dict]:
    """Stream all zip files and return one best label per generic drug name.

    'Best' = highest section completeness score, tie-broken by most recent date.
    Only prescription drugs (records with openfda.generic_name) are kept.

    Returns: {generic_name_lower: record_dict}
    """
    best: dict[str, dict] = {}   # generic_name → best record so far
    zip_files = sorted(
        f for f in os.listdir(label_dir) if f.endswith(".json.zip")
    )
    logger.info("Deduplicating across %d zip files ...", len(zip_files))

    total_scanned = 0
    total_eligible = 0

    for zip_fname in zip_files:
        zip_path = os.path.join(label_dir, zip_fname)
        logger.info("  Reading %s ...", zip_fname)

        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(zf.namelist()[0]) as f:
                records = json.loads(f.read())["results"]

        total_scanned += len(records)

        for record in records:
            openfda = record.get("openfda", {})
            generic_names = openfda.get("generic_name", [])
            if not generic_names:
                continue   # skip OTC / non-openfda records

            total_eligible += 1
            primary = generic_names[0].lower().strip()

            score = _section_score(record)
            date  = _effective_date(record)

            existing = best.get(primary)
            if existing is None:
                best[primary] = record
            else:
                existing_score = _section_score(existing)
                existing_date  = _effective_date(existing)
                # Keep if more complete; break ties by newer date
                if score > existing_score or (
                    score == existing_score and date > existing_date
                ):
                    best[primary] = record

    logger.info(
        "Scanned %d total records → %d eligible → %d unique generics kept",
        total_scanned, total_eligible, len(best),
    )
    return best


# ---------------------------------------------------------------------------
# Step 2: Chunk each deduplicated label by section
# ---------------------------------------------------------------------------

def _extract_sections(record: dict) -> list[tuple[str, str]]:
    """Return list of (section_type, text) for non-empty sections.

    Deduplication rule: if warnings_and_cautions present, skip warnings.
    Sections are truncated to MAX_CHARS.
    """
    sections = []
    warnings_and_cautions_present = bool(
        record.get("warnings_and_cautions") and
        record["warnings_and_cautions"][0].strip()
    )

    for section in TARGET_SECTIONS:
        # Skip redundant 'warnings' if 'warnings_and_cautions' already added
        if section == "warnings" and warnings_and_cautions_present:
            continue

        val = record.get(section)
        if not val or not isinstance(val, list):
            continue
        text = val[0].strip()
        if not text:
            continue

        # Truncate to embedding token limit using precise tiktoken counting
        tokens = _TOKENISER.encode(text)
        if len(tokens) > MAX_TOKENS:
            text = _TOKENISER.decode(tokens[:MAX_TOKENS])

        sections.append((section, text))

    return sections


def build_chunks(best_labels: dict[str, dict]) -> list[dict]:
    """Convert deduplicated label records into flat chunk dicts.

    Each chunk has: id, text, metadata (drug_name, brand_names, section_type,
    ndc_list, source, effective_time).
    """
    chunks = []
    for generic_name, record in best_labels.items():
        openfda = record.get("openfda", {})
        brand_names = openfda.get("brand_name", [])
        ndc_list    = openfda.get("product_ndc", []) or openfda.get("package_ndc", [])
        eff_time    = record.get("effective_time", "")

        sections = _extract_sections(record)
        if not sections:
            continue

        for section_type, text in sections:
            # Deterministic ID: hash of (generic_name, section_type)
            chunk_id_str = f"{generic_name}::{section_type}"
            chunk_uuid   = str(uuid.UUID(bytes=hashlib.md5(
                chunk_id_str.encode()
            ).digest()))

            chunks.append({
                "id":           chunk_uuid,
                "text":         text,
                "drug_name":    generic_name,
                "brand_names":  brand_names[:5],    # cap list size
                "section_type": section_type,
                "ndc_list":     ndc_list[:10],
                "source":       "fda_label",
                "effective_time": eff_time,
            })

    logger.info("Built %d chunks from %d unique labels", len(chunks), len(best_labels))
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Embed + upsert into Qdrant
# ---------------------------------------------------------------------------

def _setup_collection(client: QdrantClient) -> None:
    """Create drug_labels collection if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", COLLECTION)
    else:
        logger.info("Collection '%s' already exists", COLLECTION)


def embed_and_upsert(
    chunks: list[dict],
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed chunks in batches and upsert into Qdrant. Returns points upserted."""
    _setup_collection(qdrant_client)

    total_upserted = 0
    n_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Embed — with exponential backoff on rate limit errors
        texts = [c["text"] for c in batch]
        for attempt in range(6):
            try:
                response = openai_client.embeddings.create(
                    model=EMBED_MODEL,
                    input=texts,
                )
                vectors = [e.embedding for e in response.data]
                break
            except OpenAIRateLimitError as e:
                if attempt == 5:
                    raise
                wait = min(2 ** attempt * 10, 120)  # 10s, 20s, 40s, 80s, 120s
                logger.warning("Rate limit hit — waiting %ds (attempt %d/5)...", wait, attempt + 1)
                time.sleep(wait)

        # Build Qdrant points
        points = [
            PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={
                    "drug_name":    chunk["drug_name"],
                    "brand_names":  chunk["brand_names"],
                    "section_type": chunk["section_type"],
                    "ndc_list":     chunk["ndc_list"],
                    "source":       chunk["source"],
                    "effective_time": chunk["effective_time"],
                    "text":         chunk["text"],
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]

        qdrant_client.upsert(collection_name=COLLECTION, points=points)
        total_upserted += len(points)

        if batch_num % 10 == 0 or batch_num == n_batches:
            logger.info(
                "  Batch %d/%d — upserted %d/%d chunks",
                batch_num, n_batches, total_upserted, len(chunks),
            )

    return total_upserted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(label_dir: str = LABEL_DIR, dry_run: bool = False) -> dict:
    """Full pipeline: deduplicate → chunk → embed → upsert.

    Args:
        label_dir: Directory containing drug-label-*.json.zip files.
        dry_run:   If True, count chunks without making any API calls.

    Returns:
        Stats dict: {unique_labels, total_chunks, upserted, estimated_cost_usd}
    """
    # Deduplication pass
    best_labels = deduplicate_labels(label_dir)

    # Build chunks
    chunks = build_chunks(best_labels)

    # Cost estimate
    total_chars = sum(len(c["text"]) for c in chunks)
    est_tokens  = total_chars / 4
    est_cost    = est_tokens * 0.02 / 1_000_000

    stats = {
        "unique_labels":      len(best_labels),
        "total_chunks":       len(chunks),
        "estimated_tokens":   int(est_tokens),
        "estimated_cost_usd": round(est_cost, 4),
        "upserted":           0,
    }

    logger.info(
        "Plan: %d unique labels → %d chunks | ~%.0fM tokens | est. $%.2f",
        len(best_labels), len(chunks), est_tokens / 1_000_000, est_cost,
    )

    if dry_run:
        logger.info("Dry run — stopping before API calls")
        return stats

    # Validate API key
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file."
        )

    openai_client = OpenAI(api_key=openai_key)
    from vector_store.client import get_qdrant_client
    qdrant_client = get_qdrant_client()

    logger.info("Embedding and upserting %d chunks ...", len(chunks))
    upserted = embed_and_upsert(chunks, openai_client, qdrant_client)
    stats["upserted"] = upserted

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Ingest FDA drug labels into Qdrant")
    parser.add_argument("--label-dir", default=LABEL_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Count chunks without making API calls")
    args = parser.parse_args()

    stats = run(label_dir=args.label_dir, dry_run=args.dry_run)
    print("\nResults:", json.dumps(stats, indent=2))

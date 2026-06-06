"""
MedlinePlus Ingester — ingestion/medlineplus_ingester.py

Parses MedlinePlus XML files and upserts into Qdrant `medlineplus` collection.

Sources:
  - mplus_topics_*.xml     : 1,017 English health topic articles
  - *definitions.xml       : health term definitions (vitamins, minerals,
                             nutrition, fitness, general health)

Chunking strategy: one chunk per topic / per definition term.
Each chunk embeds: title + synonyms + categories + full summary text.

ID strategy:
  - Health topics:  MD5("medlineplus::health_topic::{url}")   -- URL is unique
  - Definitions:    MD5("medlineplus::definition::{title}")   -- title is unique
    Definitions are globally deduplicated by title across all files before
    chunk IDs are generated (some terms appear in multiple definition files).

Cost: ~$0.01 (OpenAI text-embedding-3-small, ~600K tokens total)

Usage:
    python -m ingestion.medlineplus_ingester
    python -m ingestion.medlineplus_ingester --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import uuid
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MEDLINEPLUS_DIR = "data/medlineplus"
QDRANT_PATH     = "data/qdrant_storage"
COLLECTION      = "medlineplus"
EMBED_MODEL     = "text-embedding-3-small"
EMBED_DIM       = 1536
BATCH_SIZE      = 200
LANGUAGE        = "English"

DEFINITION_FILES = [
    "fitnessdefinitions.xml",
    "generalhealthdefinitions.xml",
    "mineralsdefinitions.xml",
    "nutritiondefinitions.xml",
    "vitaminsdefinitions.xml",
]


# ---------------------------------------------------------------------------
# Step 1: Parse health topics XML
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_health_topics(xml_path: str, language: str = LANGUAGE) -> list[dict]:
    """Parse MedlinePlus health topics XML. Returns list of topic dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    topics = []

    for topic_el in root.findall("health-topic"):
        if topic_el.get("language") != language:
            continue

        title     = topic_el.get("title", "").strip()
        url       = topic_el.get("url", "").strip()
        meta_desc = topic_el.get("meta-desc", "").strip()

        raw_summary  = topic_el.findtext("full-summary") or ""
        full_summary = _strip_html(raw_summary)

        also_called = [
            el.text.strip()
            for el in topic_el.findall("also-called")
            if el.text and el.text.strip()
        ]
        groups = [
            el.text.strip()
            for el in topic_el.findall("group")
            if el.text and el.text.strip()
        ]
        institute = (topic_el.findtext("primary-institute") or "").strip()

        if not full_summary and not meta_desc:
            continue

        topics.append({
            "title":        title,
            "url":          url,
            "meta_desc":    meta_desc,
            "full_summary": full_summary,
            "also_called":  also_called,
            "groups":       groups,
            "institute":    institute,
            "source_type":  "health_topic",
        })

    logger.info("Parsed %d %s health topics", len(topics), language)
    return topics


# ---------------------------------------------------------------------------
# Step 2: Parse definition XML files
# ---------------------------------------------------------------------------

def parse_definitions(definitions_dir: str = MEDLINEPLUS_DIR) -> list[dict]:
    """Parse all *definitions.xml files.

    Deduplicates globally by title across all files so that terms appearing
    in multiple definition files (e.g., 'Protein' in nutrition + fitness)
    appear only once.
    """
    all_definitions: list[dict] = []
    seen_titles: set[str] = set()   # global deduplication across all files

    for fname in DEFINITION_FILES:
        path = os.path.join(definitions_dir, fname)
        if not os.path.exists(path):
            logger.warning("Definition file not found: %s", path)
            continue

        category = fname.replace("definitions.xml", "").capitalize()
        tree     = ET.parse(path)
        root     = tree.getroot()

        for group_el in root.findall(".//term-group"):
            term_el = group_el.find("term")
            defn_el = group_el.find("definition")
            if term_el is None or defn_el is None:
                continue

            term = (term_el.text or "").lstrip(">").strip()
            defn = (defn_el.text or "").lstrip(">").strip()
            ref_url = group_el.get("reference-url", "")

            if not term or not defn:
                continue

            # Skip if we've already seen this exact title (dedup across files)
            if term in seen_titles:
                logger.debug("Skipping duplicate definition: '%s' (already from another file)", term)
                continue
            seen_titles.add(term)

            all_definitions.append({
                "title":        term,
                "url":          ref_url,
                "meta_desc":    f"{category} definition: {term}",
                "full_summary": defn,
                "also_called":  [],
                "groups":       [category],
                "institute":    group_el.get("reference", ""),
                "source_type":  "definition",
            })

    logger.info("Parsed %d unique health term definitions", len(all_definitions))
    return all_definitions


# ---------------------------------------------------------------------------
# Step 3: Build chunks
# ---------------------------------------------------------------------------

def _make_embed_text(topic: dict) -> str:
    """Combine title + synonyms + categories + summary for embedding."""
    parts = [topic["title"]]
    if topic["also_called"]:
        parts.append("Also called: " + "; ".join(topic["also_called"]))
    if topic["groups"]:
        parts.append("Category: " + "; ".join(topic["groups"]))
    if topic["full_summary"]:
        parts.append(topic["full_summary"])
    elif topic["meta_desc"]:
        parts.append(topic["meta_desc"])
    return "\n\n".join(parts)


def build_chunks(topics: list[dict]) -> list[dict]:
    """Convert topic/definition dicts into chunk dicts ready for embedding."""
    chunks = []
    for topic in topics:
        embed_text = _make_embed_text(topic)
        if not embed_text.strip():
            continue

        source_type = topic["source_type"]

        # ID rules:
        #   health_topic → URL is always unique per MedlinePlus page
        #   definition   → title is unique (globally deduplicated at parse time)
        if source_type == "health_topic":
            id_source = topic["url"] or topic["title"]
        else:
            id_source = topic["title"]

        chunk_uuid = str(uuid.UUID(bytes=hashlib.md5(
            f"medlineplus::{source_type}::{id_source}".encode()
        ).digest()))

        chunks.append({
            "id":           chunk_uuid,
            "embed_text":   embed_text,
            "title":        topic["title"],
            "url":          topic["url"],
            "meta_desc":    topic["meta_desc"],
            "full_summary": topic["full_summary"],
            "also_called":  topic["also_called"],
            "groups":       topic["groups"],
            "institute":    topic["institute"],
            "source_type":  source_type,
            "source":       "medlineplus",
            "language":     LANGUAGE,
        })

    # Final safety check for duplicate IDs
    ids = [c["id"] for c in chunks]
    n_unique = len(set(ids))
    if n_unique != len(chunks):
        logger.warning(
            "Still %d duplicate IDs after deduplication (%d chunks, %d unique)",
            len(chunks) - n_unique, len(chunks), n_unique,
        )

    logger.info("Built %d chunks (%d unique IDs)", len(chunks), n_unique)
    return chunks


# ---------------------------------------------------------------------------
# Step 4: Embed + upsert into Qdrant
# ---------------------------------------------------------------------------

def _setup_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", COLLECTION)
    else:
        logger.info("Collection '%s' already exists — will upsert", COLLECTION)


def embed_and_upsert(
    chunks: list[dict],
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    batch_size: int = BATCH_SIZE,
) -> int:
    _setup_collection(qdrant_client)
    total_upserted = 0
    n_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch    = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=[c["embed_text"] for c in batch],
        )
        vectors = [e.embedding for e in response.data]

        points = [
            PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={
                    "title":        chunk["title"],
                    "url":          chunk["url"],
                    "meta_desc":    chunk["meta_desc"],
                    "full_summary": chunk["full_summary"],
                    "also_called":  chunk["also_called"],
                    "groups":       chunk["groups"],
                    "institute":    chunk["institute"],
                    "source_type":  chunk["source_type"],
                    "source":       chunk["source"],
                    "language":     chunk["language"],
                    "text":         chunk["embed_text"],
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]

        qdrant_client.upsert(collection_name=COLLECTION, points=points)
        total_upserted += len(points)
        logger.info("Batch %d/%d — upserted %d/%d", batch_num, n_batches,
                    total_upserted, len(chunks))

    return total_upserted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(medlineplus_dir: str = MEDLINEPLUS_DIR, dry_run: bool = False) -> dict:
    """Full pipeline: parse → chunk → embed → upsert."""
    xml_files = sorted(
        f for f in os.listdir(medlineplus_dir)
        if f.startswith("mplus_topics_") and f.endswith(".xml")
        and "compressed" not in f and "group" not in f
    )
    if not xml_files:
        raise FileNotFoundError(f"No mplus_topics_*.xml in {medlineplus_dir}")
    xml_path = os.path.join(medlineplus_dir, xml_files[-1])
    logger.info("Using: %s", xml_files[-1])

    topics      = parse_health_topics(xml_path)
    definitions = parse_definitions(medlineplus_dir)
    chunks      = build_chunks(topics + definitions)

    total_chars = sum(len(c["embed_text"]) for c in chunks)
    est_tokens  = total_chars / 4
    est_cost    = est_tokens * 0.02 / 1_000_000

    stats = {
        "health_topics":      len(topics),
        "definitions":        len(definitions),
        "total_chunks":       len(chunks),
        "estimated_tokens":   int(est_tokens),
        "estimated_cost_usd": round(est_cost, 4),
        "upserted":           0,
    }

    logger.info(
        "Plan: %d topics + %d definitions = %d chunks | ~%.0fK tokens | est. $%.4f",
        len(topics), len(definitions), len(chunks), est_tokens / 1_000, est_cost,
    )

    if dry_run:
        logger.info("Dry run — stopping before API calls")
        return stats

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    upserted = embed_and_upsert(
        chunks,
        OpenAI(api_key=openai_key),
        QdrantClient(path=QDRANT_PATH),
    )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=MEDLINEPLUS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = run(medlineplus_dir=args.dir, dry_run=args.dry_run)
    print("\nResults:", json.dumps(stats, indent=2))

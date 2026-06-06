"""
Qdrant Retrieval Tool — tools/qdrant_tool.py

Semantic search over the two Qdrant collections built in Phase 1:
  - drug_labels  : FDA drug label sections (48,639 chunks)
  - medlineplus  : patient education topics + definitions (1,102 chunks)

Returns a list of Evidence objects (the common retrieval currency).

Design:
  - A single shared OpenAI client + Qdrant client (lazy-initialised singletons)
    so agents don't re-open connections on every call.
  - Optional metadata filtering (e.g., restrict to a drug_name or section_type).

Usage:
    from tools.qdrant_tool import search_drug_labels, search_medlineplus

    evidence = search_drug_labels("metformin kidney warnings", limit=5)
    evidence = search_medlineplus("what is high blood pressure", limit=3)
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from models.schemas import Evidence

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_PATH   = "data/qdrant_storage"
EMBED_MODEL   = "text-embedding-3-small"
DRUG_LABELS   = "drug_labels"
MEDLINEPLUS   = "medlineplus"

# ---------------------------------------------------------------------------
# Lazy singletons — opened once, reused across calls
# ---------------------------------------------------------------------------

_openai_client: Optional[OpenAI] = None
_qdrant_client: Optional[QdrantClient] = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def _embed(text: str) -> list[float]:
    """Embed a query string with the same model used for ingestion."""
    response = _get_openai().embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding


def _build_filter(filters: Optional[dict]) -> Optional[Filter]:
    """Convert a simple {field: value} dict into a Qdrant Filter (AND of matches)."""
    if not filters:
        return None
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filters.items()
    ]
    return Filter(must=conditions)


# ---------------------------------------------------------------------------
# Public search functions
# ---------------------------------------------------------------------------

def search_drug_labels(
    query: str,
    limit: int = 5,
    filters: Optional[dict] = None,
) -> list[Evidence]:
    """Semantic search over FDA drug label sections.

    Args:
        query:   Natural-language query (e.g., "metformin renal contraindications").
        limit:   Max results to return.
        filters: Optional metadata filter, e.g. {"section_type": "contraindications"}
                 or {"drug_name": "metformin hydrochloride"}.

    Returns:
        List of Evidence objects sorted by relevance (highest score first).
    """
    client = _get_qdrant()
    vector = _embed(query)

    results = client.query_points(
        collection_name=DRUG_LABELS,
        query=vector,
        limit=limit,
        query_filter=_build_filter(filters),
        with_payload=True,
    ).points

    evidence = []
    for r in results:
        p = r.payload
        drug    = p.get("drug_name", "unknown drug")
        section = p.get("section_type", "")
        brands  = p.get("brand_names", [])
        brand_str = f" ({brands[0]})" if brands else ""

        evidence.append(Evidence(
            source="fda_label",
            title=f"{drug}{brand_str} — {section.replace('_', ' ')}",
            text=p.get("text", ""),
            score=r.score,
            metadata={
                "drug_name":    drug,
                "brand_names":  brands,
                "section_type": section,
                "ndc_list":     p.get("ndc_list", []),
                "effective_time": p.get("effective_time", ""),
            },
            citation=f"FDA Label — {drug}, {section.replace('_', ' ').title()}",
        ))
    return evidence


def search_medlineplus(
    query: str,
    limit: int = 5,
    filters: Optional[dict] = None,
) -> list[Evidence]:
    """Semantic search over MedlinePlus patient education topics + definitions.

    Args:
        query:   Natural-language query (e.g., "what is chronic kidney disease").
        limit:   Max results to return.
        filters: Optional metadata filter, e.g. {"source_type": "health_topic"}.

    Returns:
        List of Evidence objects sorted by relevance.
    """
    client = _get_qdrant()
    vector = _embed(query)

    results = client.query_points(
        collection_name=MEDLINEPLUS,
        query=vector,
        limit=limit,
        query_filter=_build_filter(filters),
        with_payload=True,
    ).points

    evidence = []
    for r in results:
        p = r.payload
        title = p.get("title", "Health Topic")
        url   = p.get("url", "")
        # Prefer the full_summary for the answer text; fall back to embed text
        text = p.get("full_summary") or p.get("text", "")

        evidence.append(Evidence(
            source="medlineplus",
            title=title,
            text=text,
            score=r.score,
            metadata={
                "url":         url,
                "groups":      p.get("groups", []),
                "also_called": p.get("also_called", []),
                "source_type": p.get("source_type", ""),
                "institute":   p.get("institute", ""),
            },
            citation=f"MedlinePlus — {title}",
        ))
    return evidence


def close() -> None:
    """Close the Qdrant client (call at shutdown if needed)."""
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass   # ignore errors during shutdown
        _qdrant_client = None


# Close the client cleanly before interpreter shutdown so QdrantClient.__del__
# doesn't raise "sys.meta_path is None" ImportErrors during module teardown.
atexit.register(close)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Drug label search: 'metformin kidney disease' ===")
    for e in search_drug_labels("metformin kidney disease warnings", limit=3):
        print(f"  [{e.score:.3f}] {e.title}")
        print(f"          {e.text[:100].strip()}...")

    print("\n=== MedlinePlus search: 'high blood pressure' ===")
    for e in search_medlineplus("what is high blood pressure", limit=3):
        print(f"  [{e.score:.3f}] {e.title}")
        print(f"          {e.text[:100].strip()}...")

    print("\n=== Filtered: only contraindication sections for metformin ===")
    for e in search_drug_labels("metformin", limit=3,
                                filters={"section_type": "contraindications"}):
        print(f"  [{e.score:.3f}] {e.title}")

    close()

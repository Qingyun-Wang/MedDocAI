"""
Agent 3 — Evidence Filter — agents/evidence_filter.py

Cleans up raw evidence before it reaches the Answer Generator, using a
retrieve-then-rerank strategy:

  1. Deduplicate near-identical items FIRST (cheap) — same source+drug+section,
     or highly overlapping text. Done before reranking so we don't pay to score
     redundant pairs.
  2. RERANK the survivors with a cross-encoder (tools/reranker.py) against the
     user's query. This reads (query, passage) jointly and gives every item —
     vector hits, live-API results, SQL rows — one comparable relevance score.
  3. Drop items below a relevance floor, then cap to MAX_EVIDENCE.

If the cross-encoder is unavailable (no torch / model), it falls back to the
previous bi-encoder ordering (score threshold + a mid-priority constant for
non-scored items) so the pipeline still works.
"""

from __future__ import annotations

import logging
import re

from agents.state import PipelineState
from models.schemas import Evidence
from tools import reranker

logger = logging.getLogger(__name__)

# --- Reranked path ---
RERANK_MIN_SCORE = 0.02   # drop items the cross-encoder scores as clearly irrelevant
MAX_EVIDENCE = 8          # final cap

# --- Fallback (bi-encoder) path ---
BI_ENCODER_THRESHOLD = 0.30
NONSCORED_PRIORITY = 0.55


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _dedup_key(e: Evidence) -> str:
    drug    = e.metadata.get("drug_name", "")
    section = e.metadata.get("section_type", "")
    if drug and section:
        return f"{e.source}:{drug}:{section}"
    return f"{e.source}:{_normalize(e.text)[:80]}"


def _is_near_duplicate(a: Evidence, b: Evidence) -> bool:
    ta = set(_normalize(a.text).split())
    tb = set(_normalize(b.text).split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) > 0.85


def _dedup(evidence: list[Evidence]) -> list[Evidence]:
    """Exact-key dedup followed by near-duplicate (token-Jaccard) dedup."""
    seen_keys: set[str] = set()
    unique: list[Evidence] = []
    for e in evidence:
        key = _dedup_key(e)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(e)

    deduped: list[Evidence] = []
    for e in unique:
        if any(_is_near_duplicate(e, kept) for kept in deduped):
            continue
        deduped.append(e)
    return deduped


def _fallback_order(evidence: list[Evidence]) -> list[Evidence]:
    """Bi-encoder ordering used when the cross-encoder is unavailable."""
    kept = [e for e in evidence
            if e.score is None or e.score >= BI_ENCODER_THRESHOLD]
    kept.sort(
        key=lambda e: e.score if e.score is not None else NONSCORED_PRIORITY,
        reverse=True,
    )
    return kept[:MAX_EVIDENCE]


def evidence_filter_node(state: PipelineState) -> dict:
    """LangGraph node: dedup → cross-encoder rerank → trim."""
    raw = state.get("raw_evidence", [])

    # Step 1: dedup first (cheap — fewer pairs to rerank)
    deduped = _dedup(raw)

    # Step 2: rerank (or fall back to bi-encoder order)
    if reranker.is_available() and deduped:
        # Rerank against the ORIGINAL user query — cross-encoders judge relevance
        # to what the user actually asked, not the keyword-enriched shaped query.
        query = state.get("query", "")
        reranked = reranker.rerank(query, deduped, top_k=None)
        # Drop clearly-irrelevant items, then cap
        filtered = [e for e in reranked if (e.score or 0.0) >= RERANK_MIN_SCORE]
        filtered = filtered[:MAX_EVIDENCE]
        method = "cross-encoder"
    else:
        filtered = _fallback_order(deduped)
        method = "bi-encoder-fallback"

    trace = state.get("trace", [])
    trace = trace + [
        f"Filter: {len(raw)} raw -> {len(deduped)} deduped -> "
        f"{len(filtered)} kept ({method}, cap={MAX_EVIDENCE})"
    ]

    return {
        "filtered_evidence": filtered,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agents.retrieval import retrieval_node
    from agents.router import router_node
    from agents.state import new_state

    state = new_state("What should a kidney patient know about metformin?", None, "patient")
    state.update(router_node(state))
    state.update(retrieval_node(state))
    result = evidence_filter_node(state)

    print(f"\nRaw: {len(state['raw_evidence'])} -> Filtered: {len(result['filtered_evidence'])}")
    for e in result["filtered_evidence"]:
        score = f"{e.score:.3f}" if e.score is not None else "  -  "
        print(f"  [{score}] ({e.source}) {e.title[:55]}")

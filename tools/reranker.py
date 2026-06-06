"""
Cross-Encoder Reranker — tools/reranker.py

Second-stage reranking for the retrieve-then-rerank pattern.

The bi-encoder (Qdrant / OpenAI embeddings) retrieves a large candidate pool fast
but scores query and document SEPARATELY — imprecise. This cross-encoder reads
(query, document) JOINTLY with full cross-attention and produces a far more
accurate relevance score. We use it in the Evidence Filter to reorder candidates
and keep the best few.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free, ~80MB, CPU-friendly).

Key benefit for this project: the cross-encoder scores EVERY evidence item on one
comparable scale — vector hits, live-API results, and SQL rows alike — replacing
the arbitrary 0.55 constant the filter previously used for non-scored evidence.

Graceful fallback: if sentence-transformers / the model can't load (e.g. CI without
torch), is_available() returns False and callers keep the bi-encoder ordering.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from models.schemas import Evidence

logger = logging.getLogger(__name__)

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy singletons
_model = None
_load_attempted = False


def _get_model():
    """Lazily load the cross-encoder. Returns None if unavailable (fallback)."""
    global _model, _load_attempted
    if _model is None and not _load_attempted:
        _load_attempted = True
        try:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(RERANK_MODEL)
            logger.info("Loaded cross-encoder reranker: %s", RERANK_MODEL)
        except Exception as e:
            logger.warning(
                "Cross-encoder unavailable (%s) — falling back to bi-encoder order", e
            )
            _model = None
    return _model


def is_available() -> bool:
    """True if the cross-encoder model is loadable."""
    return _get_model() is not None


def _sigmoid(x: float) -> float:
    """Map a raw logit to a 0-1 relevance probability for interpretability."""
    return 1.0 / (1.0 + math.exp(-x))


def rerank(
    query: str,
    evidence: list[Evidence],
    top_k: Optional[int] = None,
) -> list[Evidence]:
    """Rerank evidence by cross-encoder relevance to the query.

    - Overwrites each Evidence.score with a sigmoid-normalised rerank score (0-1),
      preserving the original bi-encoder score in metadata['bi_encoder_score'].
    - Returns evidence sorted by rerank score (descending), capped at top_k.
    - If the model is unavailable, returns the input unchanged (capped at top_k),
      so the pipeline still works without torch.
    """
    if not evidence:
        return []

    model = _get_model()
    if model is None:
        # Fallback: preserve incoming order
        return evidence if top_k is None else evidence[:top_k]

    # Score every (query, document) pair jointly
    pairs = [(query, e.text) for e in evidence]
    raw_scores = model.predict(pairs)   # numpy array of logits

    for e, raw in zip(evidence, raw_scores):
        if e.score is not None:
            e.metadata["bi_encoder_score"] = round(float(e.score), 4)
        e.score = round(_sigmoid(float(raw)), 4)
        e.metadata["reranked"] = True

    ranked = sorted(evidence, key=lambda e: e.score or 0.0, reverse=True)
    return ranked if top_k is None else ranked[:top_k]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def ev(title, text):
        return Evidence(source="test", title=title, text=text, score=0.7,
                        citation=title)

    query = "What should a kidney disease patient know about metformin?"
    candidates = [
        ev("relevant", "Metformin is contraindicated in severe renal impairment "
                        "(eGFR below 30). Use with caution in moderate kidney disease."),
        ev("somewhat", "Metformin is used to treat type 2 diabetes by lowering "
                       "blood sugar."),
        ev("irrelevant", "Aspirin is a common pain reliever and blood thinner."),
        ev("very_relevant", "In patients with chronic kidney disease, metformin "
                            "carries a risk of lactic acidosis and requires renal "
                            "function monitoring."),
    ]

    print(f"Reranker available: {is_available()}\n")
    print(f"Query: {query}\n")
    ranked = rerank(query, candidates)
    for e in ranked:
        print(f"  [{e.score:.3f}] {e.title}: {e.text[:60]}...")

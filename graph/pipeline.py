"""
LangGraph Pipeline — graph/pipeline.py

Wires the 6 agents into a conditional state machine with a corrective-RAG loop.

Flow:
    router -> retrieval -> evidence_filter -> answer_generator -> reviewer
                                                                     │
              ┌──────────────────────────────────────────────────────┤
              │ (review failed AND iteration < max)                   │
              ▼                                                       │
   route_back_to == "router"          -> back to router              │
   route_back_to == "answer_generator"-> back to answer_generator    │
              │                                                       │
              └─────────────── (passed OR iteration >= max) ─────────┘
                                          │
                                          ▼
                                       safety -> END

Loop control: the reviewer increments `iteration`. The conditional edge after the
reviewer checks both review_passed and iteration vs max_iterations, guaranteeing
termination.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.answer_generator import answer_generator_node
from agents.evidence_filter import evidence_filter_node
from agents.patient_summary import patient_summary_node
from agents.retrieval import retrieval_node
from agents.reviewer import reviewer_node
from agents.router import router_node
from agents.safety_agent import safety_node
from agents.observability import format_summary, finish_query, instrument_node, start_query
from agents.state import PipelineState, new_state
from agents.tracing import traced

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge after the reviewer
# ---------------------------------------------------------------------------

def _route_after_router(
    state: PipelineState,
) -> Literal["patient_summary", "answer_generator", "retrieval"]:
    """Branch after the Router:
    - patient_summary intent -> dedicated node (serves the pre-computed summary)
    - direct_answer intent   -> FAST PATH straight to the Answer Generator
      (conversational/meta questions need no retrieval — skips Agents 2 & 3)
    - everything else        -> the normal retrieval pipeline
    """
    intent = state.get("intent")
    if intent == "patient_summary":
        return "patient_summary"
    if intent == "direct_answer":
        return "answer_generator"
    return "retrieval"


def _route_after_answer(
    state: PipelineState,
) -> Literal["reviewer", "safety"]:
    """Branch after the Answer Generator:
    - direct_answer fast path -> straight to Safety. There is no retrieved evidence
      to verify faithfulness against, so the Reviewer would false-fail these
      conversational answers (the Challenge-#12 failure mode) and waste retries.
    - normal path -> the Reviewer (corrective-RAG quality gate).
    """
    if state.get("intent") == "direct_answer":
        return "safety"
    return "reviewer"


def _route_after_review(
    state: PipelineState,
) -> Literal["router", "answer_generator", "safety"]:
    """Decide where to go after the reviewer.

    - If the review passed -> safety (done).
    - If the iteration cap is reached -> safety (graceful degradation).
    - Otherwise route back to the stage the reviewer chose.
    """
    if state.get("review_passed"):
        return "safety"

    if state.get("iteration", 0) >= state.get("max_iterations", 2):
        logger.info("Reviewer retry budget exhausted -> forwarding to safety")
        return "safety"

    target = state.get("route_back_to")
    if target == "router":
        return "router"
    # default / "answer_generator"
    return "answer_generator"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_pipeline():
    """Construct and compile the LangGraph pipeline. Returns a compiled graph."""
    graph = StateGraph(PipelineState)

    # Nodes
    graph.add_node("router", instrument_node("router", router_node))
    graph.add_node("patient_summary", instrument_node("patient_summary", patient_summary_node))
    graph.add_node("retrieval", instrument_node("retrieval", retrieval_node))
    graph.add_node("evidence_filter", instrument_node("evidence_filter", evidence_filter_node))
    graph.add_node("answer_generator", instrument_node("answer_generator", answer_generator_node))
    graph.add_node("reviewer", instrument_node("reviewer", reviewer_node))
    graph.add_node("safety", instrument_node("safety", safety_node))

    # Entry: router, then branch — patient_summary serves the stored summary,
    # direct_answer fast-paths to the Answer Generator (no retrieval),
    # everything else goes through the normal retrieval pipeline.
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "patient_summary":  "patient_summary",
            "answer_generator": "answer_generator",   # direct_answer fast path
            "retrieval":        "retrieval",
        },
    )
    graph.add_edge("patient_summary", "safety")
    graph.add_edge("retrieval", "evidence_filter")
    graph.add_edge("evidence_filter", "answer_generator")

    # After the Answer Generator: direct answers skip the reviewer (no evidence
    # to verify against); normal answers go through the corrective-RAG review.
    graph.add_conditional_edges(
        "answer_generator",
        _route_after_answer,
        {
            "reviewer": "reviewer",
            "safety":   "safety",
        },
    )

    # Conditional edge after reviewer (the corrective loop)
    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {
            "router":           "router",
            "answer_generator": "answer_generator",
            "safety":           "safety",
        },
    )

    graph.add_edge("safety", END)

    return graph.compile()


# Module-level compiled singleton (built once on first import)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def _persist_metrics(state: PipelineState, metrics: dict, patient_context: dict | None) -> None:
    """Best-effort write of one row to SQLite query_metrics. Never raises.

    Skipped when MEDDOCAI_METRICS != "1" or the DB file is absent — that guard is
    what stops CI / eval-gate runners from creating a stray data/meddocai.db.
    """
    import os
    if os.getenv("MEDDOCAI_METRICS", "1") != "1":
        return
    db_path = os.getenv("MEDDOCAI_DB", "data/meddocai.db")
    if not os.path.exists(db_path):
        return
    try:
        from ingestion.sqlite_loader import MedDocDB
        MedDocDB(db_path).save_query_metrics(
            query_id=metrics.get("query_id", ""),
            query=state.get("query", ""),
            intent=state.get("intent", ""),
            user_role=state.get("user_role", ""),
            patient_id=(patient_context or {}).get("patient_id"),
            iterations=state.get("iteration", 0),
            review_passed=bool(state.get("review_passed", False)),
            n_evidence=len(state.get("filtered_evidence", []) or []),
            metrics=metrics,
        )
    except Exception as exc:   # noqa: BLE001 — metrics must never break an answer
        logger.debug("metrics persist skipped: %s", exc)


@traced("answer_query", run_type="chain")
def answer_query(
    query: str,
    patient_context: dict | None = None,
    user_role: str = "anonymous",
    max_iterations: int = 2,
    conversation_history: list[dict] | None = None,
) -> PipelineState:
    """Run the full pipeline for a query. Returns the final PipelineState.

    conversation_history: recent prior turns [{role, content}, ...] for follow-up
    resolution and coherent answers.
    """
    pipeline = get_pipeline()
    initial = new_state(query, patient_context, user_role, max_iterations,
                        conversation_history=conversation_history)
    start_query(initial["query_id"])
    try:
        # recursion_limit guards against any unexpected cycling beyond our logic
        final = pipeline.invoke(initial, config={"recursion_limit": 25})
    finally:
        metrics = finish_query(initial["query_id"])
    final["metrics"] = metrics
    _persist_metrics(final, metrics, patient_context)
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)

    query = sys.argv[1] if len(sys.argv) > 1 else \
        "What are the key warnings for metformin?"

    print(f"\nQuery: {query}\n" + "=" * 60)
    final = answer_query(query)

    print("\n--- TRACE ---")
    for step in final.get("trace", []):
        print(f"  {step}")

    print("\n--- METRICS ---")
    m = final.get("metrics", {})
    print(f"  {format_summary(m)}")
    for c in m.get("calls", []):
        print(f"    {c['caller']:<18} in={c['input_tokens']:>6} "
              f"out={c['output_tokens']:>5} {c['latency_ms']:>8.0f}ms")
    if m.get("by_node"):
        print("  nodes: " + ", ".join(f"{k}={v:.0f}ms" for k, v in m["by_node"].items()))

    print("\n--- FINAL ANSWER ---")
    print(final.get("final_answer", "(no answer)"))

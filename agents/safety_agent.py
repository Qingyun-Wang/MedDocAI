"""
Agent 6 — Safety Agent — agents/safety_agent.py

The final stage. Assembles the user-facing answer with:
  - The reviewed answer text + citations.
  - A standard "informational, not medical advice" disclaimer.
  - An extra transparency note if the Reviewer exhausted its retries without
    passing (graceful degradation).
  - A clinical-decision flag if the query asked for a diagnosis/treatment decision.

Rule-based (no LLM call) — disclaimers should be deterministic and consistent.
"""

from __future__ import annotations

import logging
import re

from agents.state import PipelineState

logger = logging.getLogger(__name__)

STANDARD_DISCLAIMER = (
    "ℹ️ This is general health information from public sources (FDA, MedlinePlus, "
    "CMS), not medical advice. Always consult a qualified healthcare professional "
    "for decisions about your care."
)

UNVERIFIED_NOTE = (
    "⚠️ Note: I could not fully verify this answer against my sources within the "
    "allowed attempts. Please treat it with extra caution and confirm with a "
    "healthcare professional."
)

# Phrases suggesting the user wants a clinical decision (not just information)
_CLINICAL_DECISION_PATTERNS = [
    r"\bshould i (take|stop|start|change|increase|decrease)\b",
    r"\bdiagnos(e|is|ed)\b",
    r"\bwhat('?s| is) wrong with me\b",
    r"\bdo i have\b",
    r"\bis it safe for me to\b",
    r"\bcan i (take|stop|combine)\b",
]

CLINICAL_FLAG = (
    "🩺 This question may involve a clinical decision. Information here can help you "
    "prepare, but the decision should be made with your prescriber or doctor."
)


def _wants_clinical_decision(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _CLINICAL_DECISION_PATTERNS)


def safety_node(state: PipelineState) -> dict:
    """LangGraph node: assemble final answer with disclaimers."""
    answer    = state.get("answer", "")
    citations = state.get("citations", [])
    passed    = state.get("review_passed", False)

    disclaimers = [STANDARD_DISCLAIMER]

    # Graceful degradation: reviewer never passed within retry budget
    if not passed:
        disclaimers.insert(0, UNVERIFIED_NOTE)

    # Clinical-decision flag
    if _wants_clinical_decision(state.get("query", "")):
        disclaimers.append(CLINICAL_FLAG)

    # Assemble final answer
    parts = [answer]
    if citations:
        parts.append("\n**Sources:**")
        parts.extend(citations)
    parts.append("\n---")
    parts.extend(disclaimers)
    final_answer = "\n".join(parts)

    trace = state.get("trace", [])
    trace = trace + [
        f"Safety: review_passed={passed}, "
        f"disclaimers={len(disclaimers)}, "
        f"clinical_flag={_wants_clinical_decision(state.get('query', ''))}"
    ]

    return {
        "final_answer": final_answer,
        "disclaimers": disclaimers,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agents.state import new_state

    # Simulate a completed pipeline state
    state = new_state("Should I take metformin if I have kidney disease?", None, "patient")
    state["answer"] = "Metformin is contraindicated in severe renal impairment [1]."
    state["citations"] = ["[1] FDA Label — Metformin, Contraindications"]
    state["review_passed"] = True
    result = safety_node(state)

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])

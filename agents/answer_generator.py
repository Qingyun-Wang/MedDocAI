"""
Agent 4 — Answer Generator — agents/answer_generator.py

Synthesises a grounded answer from the filtered evidence using Claude.
- Cites sources inline with [n] markers that map to the evidence list.
- Prompt tone adapts to user role: patient_friendly vs care_manager.
- On a retry (reviewer routed back here), incorporates the reviewer's feedback.
"""

from __future__ import annotations

import logging

from agents.llm import call_claude_text
from agents.state import PipelineState
from models.schemas import Evidence

logger = logging.getLogger(__name__)

_SYSTEM_PATIENT = """You are a healthcare information assistant speaking directly to a patient.
Explain clearly in plain, friendly language a non-medical person can understand.
Ground every statement in the provided evidence. Use inline citation markers like [1], [2]
that refer to the numbered evidence. Do not invent facts not in the evidence.
Never give a diagnosis or tell the patient to start/stop/change a medication."""

_SYSTEM_CARE_MANAGER = """You are a healthcare information assistant supporting a care manager.
Use precise, clinical-but-accessible language. Ground every statement in the provided
evidence with inline citation markers like [1], [2]. Surface drug-condition interactions
and monitoring considerations. Do not invent facts not in the evidence. Frame everything
as informational support for care coordination, not clinical decision-making."""


def _system_for_role(role: str) -> str:
    return _SYSTEM_CARE_MANAGER if role == "care_manager" else _SYSTEM_PATIENT


def _format_evidence(evidence: list[Evidence]) -> str:
    """Number the evidence so the model can cite [n]."""
    lines = []
    for i, e in enumerate(evidence, 1):
        lines.append(f"[{i}] ({e.source}) {e.title}\n    {e.text}")
    return "\n\n".join(lines)


def _build_user_prompt(state: PipelineState) -> str:
    evidence = state.get("filtered_evidence", [])
    lines = [f"User question: {state['query']}", ""]

    ctx = state.get("patient_context")
    if ctx:
        conditions = ctx.get("conditions_json") or ctx.get("conditions") or []
        cond_names = [c.get("display", "") for c in conditions][:8]
        lines.append(f"This question is about a patient: {ctx.get('name','')}, "
                     f"{ctx.get('age','?')}yo {ctx.get('gender','')}.")
        if cond_names:
            lines.append(f"Patient's active conditions: {', '.join(cond_names)}")
        lines.append("Tailor the answer to this patient's situation where relevant.")
        lines.append("")

    lines.append("EVIDENCE (cite these with [n] markers):")
    lines.append(_format_evidence(evidence) if evidence
                 else "(no evidence retrieved)")

    feedback = state.get("reviewer_feedback")
    if feedback and state.get("route_back_to") == "answer_generator":
        lines.append("")
        lines.append("REVIEWER FEEDBACK ON YOUR PREVIOUS ANSWER (fix these issues):")
        lines.append(f"  {feedback}")

    lines.append("")
    lines.append("Write a clear, well-structured answer grounded in the evidence above. "
                 "Use [n] citation markers. If the evidence does not cover the question, "
                 "say so honestly rather than guessing.")
    return "\n".join(lines)


def answer_generator_node(state: PipelineState) -> dict:
    """LangGraph node: generate a grounded answer with citation markers."""
    role = state.get("user_role", "anonymous")
    answer = call_claude_text(
        system=_system_for_role(role),
        user=_build_user_prompt(state),
        max_tokens=1200,
    )

    # Build the citation list from the evidence the model was given
    evidence = state.get("filtered_evidence", [])
    citations = [f"[{i}] {e.citation}" for i, e in enumerate(evidence, 1)]

    trace = state.get("trace", [])
    trace = trace + [f"AnswerGen: produced {len(answer)} chars, {len(citations)} citations"]

    return {
        "answer": answer,
        "citations": citations,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agents.evidence_filter import evidence_filter_node
    from agents.retrieval import retrieval_node
    from agents.router import router_node
    from agents.state import new_state

    state = new_state("What are the key warnings for metformin?", None, "patient")
    state.update(router_node(state))
    state.update(retrieval_node(state))
    state.update(evidence_filter_node(state))
    result = answer_generator_node(state)

    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== CITATIONS ===")
    for c in result["citations"]:
        print(f"  {c}")

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
from agents.state import PipelineState, format_conversation_history
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


def _build_user_prompt(state: PipelineState, direct_mode: bool = False) -> str:
    evidence = state.get("filtered_evidence", [])
    lines = []

    # Recent conversation for coherent follow-ups (grounding still comes from evidence)
    convo = format_conversation_history(state.get("conversation_history", []))
    if convo:
        lines.append("RECENT CONVERSATION (for continuity — do not treat as evidence; "
                     "ground all clinical claims in the EVIDENCE below):")
        lines.append(convo)
        lines.append("")

    lines.append(f"User question: {state['query']}")
    lines.append("")

    # If a patient is selected, their record is included in the evidence list as a
    # 'Patient Record' item (added by the Evidence Filter). Point the model at it so
    # it tailors the answer — no need to duplicate the patient facts here.
    ctx = state.get("patient_context")
    has_patient_evidence = any(
        e.source == "patient_record" for e in evidence
    )
    if ctx and has_patient_evidence:
        lines.append("This question concerns a specific patient. One of the evidence "
                     "items below is their Patient Record — tailor your answer to their "
                     "conditions, medications, and lab results where relevant, and cite "
                     "it like any other source.")
        lines.append("")

    if evidence or not direct_mode:
        lines.append("EVIDENCE (cite these with [n] markers):")
        lines.append(_format_evidence(evidence) if evidence
                     else "(no evidence retrieved)")

    feedback = state.get("reviewer_feedback")
    if feedback and state.get("route_back_to") == "answer_generator":
        lines.append("")
        lines.append("REVIEWER FEEDBACK ON YOUR PREVIOUS ANSWER (fix these issues):")
        lines.append(f"  {feedback}")

    lines.append("")
    if direct_mode:
        lines.append(
            "This is a conversational or meta question — no document retrieval was "
            "performed. Answer naturally and briefly from the conversation history "
            "and the patient context above (if any). Do NOT invent clinical facts, "
            "statistics, or sources. If the user asks for medical/policy specifics "
            "you don't have here, suggest they ask it as a direct question so you "
            "can look it up properly."
        )
    else:
        # Structured-lookup intents (eligibility / pricing) return exact rows. The
        # model tends to "help" by adding worked examples (e.g. inventing an income
        # figure and computing the dollar amount) — none of which is in the evidence,
        # which tanks faithfulness. Forbid that explicitly for these intents.
        if state.get("intent") == "policy_eligibility":
            lines.append(
                "This is a structured data lookup (Medicaid eligibility / drug pricing). "
                "The evidence contains EXACT values. State only the figures and facts "
                "present in the evidence. Do NOT compute, extrapolate, or invent "
                "illustrative examples, dollar amounts, income scenarios, or any numbers "
                "not explicitly in the evidence — a worked example the evidence does not "
                "contain is a hallucination. Answer the specific question directly from "
                "the evidence, then STOP. Do NOT append generic procedural advice (e.g. "
                "'contact your local office', 'visit the website', 'compare your income to "
                "the FPL guidelines', 'your actual copay may vary') or other statements not "
                "grounded in the evidence — each ungrounded sentence counts against you. "
                "Required safety disclaimers are added separately by a later step, so you "
                "do not need to add them. Keep the answer concise and factual."
            )
            lines.append("")
        lines.append("Write a clear, well-structured answer grounded in the evidence above. "
                     "Use [n] citation markers. If the evidence does not cover the question, "
                     "say so honestly rather than guessing.")
    return "\n".join(lines)


def answer_generator_node(state: PipelineState) -> dict:
    """LangGraph node: generate a grounded answer with citation markers.

    Two modes:
    - Normal: answers from the retrieved+filtered evidence (then goes to the Reviewer).
    - direct_answer intent (fast path): retrieval/filter were skipped — answers from
      conversation history + patient context. The patient record is injected here as
      evidence [1] (normally the Evidence Filter does this). Sets review_passed=True
      because there is no retrieved evidence to verify faithfulness against — running
      the Reviewer would false-fail conversational answers (cf. Challenge #12).
    """
    role = state.get("user_role", "anonymous")
    direct_mode = state.get("intent") == "direct_answer"

    updates: dict = {}
    if direct_mode:
        # Filter was skipped — inject the patient record ourselves (if available)
        from agents.evidence_filter import build_patient_evidence
        ctx = state.get("patient_context")
        evidence = []
        if ctx:
            patient_ev = build_patient_evidence(ctx)
            if patient_ev is not None:
                evidence = [patient_ev]
        state["filtered_evidence"] = evidence
        updates["filtered_evidence"] = evidence
        updates["review_passed"] = True   # no evidence to verify — skip the review loop

    answer = call_claude_text(
        system=_system_for_role(role),
        user=_build_user_prompt(state, direct_mode=direct_mode),
        max_tokens=600 if direct_mode else 1200,
        caller="answer_generator",
    )

    # Build the citation list from the evidence the model was given
    evidence = state.get("filtered_evidence", [])
    citations = [f"[{i}] {e.citation}" for i, e in enumerate(evidence, 1)]

    trace = state.get("trace", [])
    trace = trace + [
        f"AnswerGen: produced {len(answer)} chars, {len(citations)} citations"
        + (" [DIRECT: no retrieval, skipping reviewer]" if direct_mode else "")
    ]

    updates.update({
        "answer": answer,
        "citations": citations,
        "trace": trace,
    })
    return updates


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

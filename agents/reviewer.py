"""
Agent 5 — Reviewer (Corrective RAG) — agents/reviewer.py

The quality gate. Uses Claude as an LLM judge to check the generated answer on
two axes:
  1. RELEVANCE   — does the answer actually address the user's query?
  2. FAITHFULNESS — is every claim grounded in the retrieved evidence?

On PASS  -> forward to Safety.
On FAIL  -> produce structured feedback and a route-back target:
   - evidence insufficient/wrong  -> route back to ROUTER
   - evidence good but answer bad  -> route back to ANSWER GENERATOR

Loop control lives in the graph (graph/pipeline.py): the reviewer increments the
iteration counter; when iteration >= max_iterations the graph stops looping and
forwards the best-attempt answer to Safety regardless.
"""

from __future__ import annotations

import logging

from agents.llm import call_claude_structured
from agents.state import PipelineState, ReviewResult
from models.schemas import Evidence

logger = logging.getLogger(__name__)

_SYSTEM = """You are a quality reviewer for a healthcare information assistant.
You evaluate whether an AI-generated answer is (a) RELEVANT to the user's question and
(b) FAITHFUL to the supplied evidence.

Calibrate your strictness carefully — be rigorous about real problems, but do NOT fail
answers over technicalities:

RELEVANT means the answer addresses what was actually asked (not a different topic).

FAITHFUL means the substantive clinical claims are supported by the evidence. Apply
these calibration rules:
- A specific fact (a dose, an eGFR threshold, a price, a contraindication, a statistic)
  must appear in the evidence. If it does, the claim is faithful.
- Evidence drawn from a product that CONTAINS the drug in question counts as evidence
  about that drug. Example: a boxed warning on a 'sitagliptin-metformin' label about
  lactic acidosis DOES support a statement about metformin's lactic-acidosis risk.
- General, widely-known medical framing and standard safety advice ("talk to your
  doctor", "watch for symptoms") does NOT need a citation.
- Minor paraphrase or reasonable synthesis of the evidence is faithful. Only fail
  faithfulness if the answer states a specific clinical fact that is absent from, or
  contradicts, the evidence.

PASS the answer if it is relevant and its specific claims are supported. Only FAIL when
there is a genuine problem a clinician would object to.

If it fails, decide WHERE to send it back:
- 'router': the EVIDENCE genuinely lacked what was needed — we must retrieve different
  evidence. In your feedback, suggest specific search terms or a tool to call.
- 'answer_generator': the evidence was adequate but the ANSWER invented a fact, missed
  key evidence, or didn't address the question — we need a better-written answer.

Provide concise, actionable feedback for the next attempt (empty if passed)."""

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {
            "type": "boolean",
            "description": "Does the answer address the user's query?",
        },
        "faithful": {
            "type": "boolean",
            "description": "Is every substantive claim grounded in the evidence?",
        },
        "passed": {
            "type": "boolean",
            "description": "True only if BOTH relevant and faithful",
        },
        "route_back_to": {
            "type": ["string", "null"],
            "enum": ["router", "answer_generator", None],
            "description": "Where to send back on failure; null if passed",
        },
        "feedback": {
            "type": "string",
            "description": "Actionable feedback for the next attempt (empty if passed)",
        },
    },
    "required": ["relevant", "faithful", "passed", "feedback"],
}


def _format_evidence(evidence: list[Evidence]) -> str:
    lines = []
    for i, e in enumerate(evidence, 1):
        lines.append(f"[{i}] ({e.source}) {e.title}\n    {e.text}")
    return "\n\n".join(lines) if lines else "(no evidence)"


def _build_user_prompt(state: PipelineState) -> str:
    return (
        f"USER QUERY:\n{state['query']}\n\n"
        f"EVIDENCE PROVIDED TO THE ANSWER GENERATOR:\n"
        f"{_format_evidence(state.get('filtered_evidence', []))}\n\n"
        f"GENERATED ANSWER:\n{state.get('answer', '')}\n\n"
        f"Evaluate relevance and faithfulness. If it fails, choose route_back_to "
        f"('router' for bad evidence, 'answer_generator' for bad answer) and give "
        f"actionable feedback."
    )


def reviewer_node(state: PipelineState) -> dict:
    """LangGraph node: judge the answer, decide pass / route-back, bump iteration."""
    raw = call_claude_structured(
        system=_SYSTEM,
        user=_build_user_prompt(state),
        tool_name="review_answer",
        tool_description="Judge the answer's relevance and faithfulness",
        input_schema=_INPUT_SCHEMA,
        max_tokens=700,
    )

    result = ReviewResult(
        relevant=raw.get("relevant", False),
        faithful=raw.get("faithful", False),
        passed=raw.get("passed", False),
        route_back_to=raw.get("route_back_to"),
        feedback=raw.get("feedback", ""),
    )

    # Enforce internal consistency: passed iff relevant AND faithful
    passed = bool(result.relevant and result.faithful)

    # Determine route-back target on failure; default to answer_generator
    route_back = None
    if not passed:
        route_back = result.route_back_to
        if route_back not in ("router", "answer_generator"):
            route_back = "answer_generator"

    iteration = state.get("iteration", 0) + 1

    trace = state.get("trace", [])
    trace = trace + [
        f"Reviewer: relevant={result.relevant}, faithful={result.faithful}, "
        f"passed={passed}, route_back={route_back}, iteration={iteration}"
    ]

    return {
        "review_passed": passed,
        "route_back_to": route_back if not passed else None,
        "reviewer_feedback": result.feedback if not passed else None,
        "iteration": iteration,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agents.answer_generator import answer_generator_node
    from agents.evidence_filter import evidence_filter_node
    from agents.retrieval import retrieval_node
    from agents.router import router_node
    from agents.state import new_state

    state = new_state("What are the key warnings for metformin?", None, "patient")
    state.update(router_node(state))
    state.update(retrieval_node(state))
    state.update(evidence_filter_node(state))
    state.update(answer_generator_node(state))
    result = reviewer_node(state)

    print("\n=== REVIEW RESULT ===")
    print(f"  passed:        {result['review_passed']}")
    print(f"  route_back_to: {result['route_back_to']}")
    print(f"  iteration:     {result['iteration']}")
    print(f"  feedback:      {result['reviewer_feedback']}")

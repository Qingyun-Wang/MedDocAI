"""
Agent 1 — Router — agents/router.py

Classifies the query intent, extracts entities (drug name, state), shapes the
query for retrieval (enriched with patient context), and selects which tools
the Retrieval agent should call.

On a retry (reviewer routed back here), it incorporates the reviewer's feedback
to shape a better query / tool selection.
"""

from __future__ import annotations

import logging

from agents.llm import call_claude_structured
from agents.state import INTENTS, PipelineState, RouterDecision

logger = logging.getLogger(__name__)

# Map each intent to the tools the Retrieval agent should call
INTENT_TOOL_MAP: dict[str, list[str]] = {
    "medication_info":     ["search_drug_labels", "fetch_drug_label"],
    "drug_recall":         ["check_drug_recalls", "check_drug_shortages"],
    "policy_eligibility":  ["lookup_state_eligibility", "lookup_drug_price"],
    "condition_education": ["search_medlineplus", "explain_condition_snomed"],
    "patient_summary":     [],   # handled by patient_summary_node (serves stored summary)
    "general":             ["search_medlineplus", "search_drug_labels"],
}

_SYSTEM = """You are the Router for a healthcare document intelligence assistant.
Your job is to classify a user's query and prepare it for retrieval.

Intent definitions:
- medication_info: questions about a drug (what it treats, side effects, warnings,
  dosing, interactions, safety for a patient)
- drug_recall: questions about recalls, shortages, or safety alerts for a drug
- policy_eligibility: questions about Medicaid eligibility, income limits, drug
  pricing/cost, or coverage policy
- condition_education: questions asking to explain a health condition or topic in
  plain language
- patient_summary: requests to summarise or review a specific patient's situation
- general: anything else, or unclear

Rules:
- Extract the drug name if the query mentions a specific medication.
- Extract the US state name if the query mentions one.
- Build a 'shaped_query': a concise search query optimised for semantic retrieval.
  If patient context is provided, enrich the shaped_query with the patient's
  relevant conditions (e.g., 'metformin safety' + patient has CKD ->
  'metformin safety renal impairment chronic kidney disease').
- If reviewer feedback is provided, use it to improve the shaped_query and intent."""

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": INTENTS,
            "description": "The classified intent",
        },
        "shaped_query": {
            "type": "string",
            "description": "Query reshaped for semantic retrieval, enriched with "
                           "patient conditions if context present",
        },
        "drug_name": {
            "type": ["string", "null"],
            "description": "Drug name from the query, or null",
        },
        "state_name": {
            "type": ["string", "null"],
            "description": "US state name from the query, or null",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the routing decision",
        },
    },
    "required": ["intent", "shaped_query", "reasoning"],
}


def _build_user_prompt(state: PipelineState) -> str:
    lines = [f"User query: {state['query']}"]
    lines.append(f"User role: {state.get('user_role', 'anonymous')}")

    ctx = state.get("patient_context")
    if ctx:
        conditions = ctx.get("conditions_json") or ctx.get("conditions") or []
        meds = ctx.get("medications_json") or ctx.get("medications") or []
        cond_names = [c.get("display", "") for c in conditions][:8]
        med_names = [m.get("display", "") for m in meds
                     if m.get("status") == "active"][:8]
        lines.append("")
        lines.append("PATIENT CONTEXT:")
        lines.append(f"  Name: {ctx.get('name', 'unknown')}, "
                     f"Age: {ctx.get('age', '?')}, Gender: {ctx.get('gender', '?')}")
        if cond_names:
            lines.append(f"  Active conditions: {', '.join(cond_names)}")
        if med_names:
            lines.append(f"  Active medications: {', '.join(med_names)}")

    feedback = state.get("reviewer_feedback")
    if feedback and state.get("route_back_to") == "router":
        lines.append("")
        lines.append("REVIEWER FEEDBACK FROM PREVIOUS ATTEMPT (improve on this):")
        lines.append(f"  {feedback}")

    return "\n".join(lines)


def router_node(state: PipelineState) -> dict:
    """LangGraph node: classify intent, shape query, select tools."""
    decision_raw = call_claude_structured(
        system=_SYSTEM,
        user=_build_user_prompt(state),
        tool_name="route_query",
        tool_description="Classify and route the healthcare query",
        input_schema=_INPUT_SCHEMA,
        max_tokens=600,
    )

    # Validate / normalise via Pydantic
    decision = RouterDecision(
        intent=decision_raw.get("intent", "general"),
        shaped_query=decision_raw.get("shaped_query", state["query"]),
        drug_name=decision_raw.get("drug_name"),
        state_name=decision_raw.get("state_name"),
        reasoning=decision_raw.get("reasoning", ""),
    )

    intent = decision.intent if decision.intent in INTENTS else "general"
    tools = INTENT_TOOL_MAP.get(intent, INTENT_TOOL_MAP["general"])

    trace = state.get("trace", [])
    trace = trace + [
        f"Router: intent={intent}, drug={decision.drug_name}, "
        f"state={decision.state_name}, tools={tools}"
    ]

    return {
        "intent": intent,
        "shaped_query": decision.shaped_query,
        "drug_name": decision.drug_name,
        "state_name": decision.state_name,
        "tools_to_call": tools,
        "trace": trace,
        # Clear any prior route-back signal now that we've re-routed
        "route_back_to": None,
        "reviewer_feedback": None,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agents.state import new_state

    tests = [
        ("What are the side effects of metformin?", None, "anonymous"),
        ("Is there a recall on semaglutide?", None, "anonymous"),
        ("What is the Medicaid income limit for pregnant women in Texas?", None, "anonymous"),
        ("Explain high blood pressure in simple terms", None, "patient"),
    ]
    for query, ctx, role in tests:
        state = new_state(query, ctx, role)
        result = router_node(state)
        print(f"\nQuery: {query}")
        print(f"  intent:       {result['intent']}")
        print(f"  shaped_query: {result['shaped_query']}")
        print(f"  drug_name:    {result['drug_name']}")
        print(f"  state_name:   {result['state_name']}")
        print(f"  tools:        {result['tools_to_call']}")

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
from agents.state import (
    INTENTS,
    PipelineState,
    RouterDecision,
    format_conversation_history,
)

logger = logging.getLogger(__name__)

# Map each intent to the tools the Retrieval agent should call
INTENT_TOOL_MAP: dict[str, list[str]] = {
    "medication_info":     ["search_drug_labels", "fetch_drug_label", "explain_drug_by_name"],
    "drug_recall":         ["check_drug_recalls", "check_drug_shortages"],
    "policy_eligibility":  ["lookup_state_eligibility", "lookup_drug_price"],
    "condition_education": ["search_medlineplus", "explain_condition_snomed",
                            "explain_lab_loinc", "explain_procedure_snomed"],
    "patient_summary":     [],   # handled by patient_summary_node (serves stored summary)
    "direct_answer":       [],   # fast path — no retrieval; answered from chat history
                                 # + patient context by the Answer Generator directly
    "general":             ["search_medlineplus", "search_drug_labels"],
}

# Extra tools added ON RETRY (when the reviewer routed back here for insufficient
# evidence) to cast a wider net — a genuinely different candidate pool, not a repeat.
INTENT_RETRY_EXTRA_TOOLS: dict[str, list[str]] = {
    "medication_info":     ["search_medlineplus", "check_drug_recalls"],
    "drug_recall":         ["search_drug_labels", "fetch_drug_label"],
    "policy_eligibility":  ["search_medlineplus"],
    # Live MedlinePlus health-topics search is a RETRY-ONLY fallback for topic
    # questions the frozen Qdrant index missed (health topics only — not drugs).
    "condition_education": ["search_drug_labels", "search_medlineplus_live"],
    "general":             ["fetch_drug_label", "check_drug_recalls", "search_medlineplus_live"],
}

_SYSTEM = """You are the Router for a healthcare document intelligence assistant.
Your job is to classify a user's query and prepare it for retrieval.

Intent definitions:
- medication_info: questions about a drug OR a patient's medications — what a drug
  treats, side effects, warnings, dosing, interactions, whether a drug is safe for a
  patient, what to monitor, or the most important drug-related concerns for a specific
  patient. A clinical question about a patient's medications/monitoring/safety belongs
  here (it needs evidence retrieval) EVEN IF no specific drug is named.
- drug_recall: questions about recalls, shortages, or safety alerts for a drug
- policy_eligibility: questions about Medicaid eligibility, income limits, drug
  pricing/cost, or coverage policy
- condition_education: questions asking to explain a health condition, lab value, or
  topic in plain language
- patient_summary: ONLY broad requests for an overall overview/summary of the WHOLE
  patient (e.g., "summarize this patient", "give me an overview of this member", "tell
  me about this patient", "review this patient's chart"). A question about a SPECIFIC
  clinical aspect — drug concerns, what to monitor, is a drug safe, what a lab means —
  is NOT patient_summary; classify those as medication_info (or condition_education)
  so they retrieve evidence and answer the specific question.
- direct_answer: conversational or meta questions that need NO database/API retrieval —
  they can be answered entirely from the conversation history and/or the patient context
  already provided. Examples: "what was my last question?", "can you repeat that?",
  "what did you just tell me?", greetings ("hi", "thanks", "goodbye"), "who are you?",
  "what can you help me with?". IMPORTANT: any question that asks for medical facts,
  drug information, policy data, or anything verifiable in our sources is NOT
  direct_answer — even if it sounds casual.
- general: anything else, or unclear

Rules:
- Extract the drug name if the query mentions a specific medication.
- Extract the US state name if the query mentions one.
- Build a 'shaped_query': a concise search query optimised for semantic retrieval.
  If patient context is provided, enrich the shaped_query with the patient's
  relevant conditions (e.g., 'metformin safety' + patient has CKD ->
  'metformin safety renal impairment chronic kidney disease').
- If reviewer feedback is provided, use it to improve the shaped_query and intent.
- If a RECENT CONVERSATION is shown, use it to resolve pronouns/references in the
  current query (e.g., "what about its side effects?" after discussing metformin means
  the shaped_query should be about metformin's side effects, and drug_name = metformin).
- IF this is a RETRY and previous queries are shown: the earlier search did NOT find
  adequate evidence. Produce a MEANINGFULLY DIFFERENT shaped_query — do not repeat a
  previous one. Broaden it, use clinical synonyms, rephrase, or target different
  aspects (e.g., switch from 'side effects' to 'adverse reactions contraindications
  monitoring'). A near-identical query will just return the same evidence and fail again."""

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
    lines = []

    # Recent conversation — lets the Router resolve references in the current query
    # ("what about its side effects?" -> the drug from the previous turn).
    convo = format_conversation_history(state.get("conversation_history", []))
    if convo:
        lines.append("RECENT CONVERSATION (resolve references like 'it'/'that drug' "
                     "from this when shaping the query):")
        lines.append(convo)
        lines.append("")

    lines.append(f"User query: {state['query']}")
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

    is_retry = state.get("route_back_to") == "router"
    if is_retry:
        attempted = state.get("attempted_queries", [])
        if attempted:
            lines.append("")
            lines.append("PREVIOUS SEARCHES THAT DID NOT FIND ADEQUATE EVIDENCE "
                         "(do NOT repeat — try a different angle):")
            for q in attempted:
                lines.append(f"  - {q}")

        feedback = state.get("reviewer_feedback")
        if feedback:
            lines.append("")
            lines.append("REVIEWER FEEDBACK FROM PREVIOUS ATTEMPT (address this):")
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
    tools = list(INTENT_TOOL_MAP.get(intent, INTENT_TOOL_MAP["general"]))

    # On a retry-to-router, broaden the tool set so retrieval casts a wider net
    # (a genuinely different candidate pool, not a repeat of the failed attempt).
    is_retry = state.get("route_back_to") == "router"
    if is_retry:
        extra = INTENT_RETRY_EXTRA_TOOLS.get(intent, [])
        # Union, preserving order and de-duplicating
        tools = list(dict.fromkeys(tools + extra))

    # Track every shaped query so future retries can avoid repeating them
    attempted = state.get("attempted_queries", [])
    attempted = attempted + [decision.shaped_query]

    trace = state.get("trace", [])
    trace = trace + [
        f"Router: intent={intent}, drug={decision.drug_name}, "
        f"state={decision.state_name}, tools={tools}"
        + (f" [RETRY: broadened tools, query diversified]" if is_retry else "")
    ]

    return {
        "intent": intent,
        "attempted_queries": attempted,
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

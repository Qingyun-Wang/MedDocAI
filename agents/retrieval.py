"""
Agent 2 — Retrieval — agents/retrieval.py

Calls the tools selected by the Router and collects their Evidence into
state['raw_evidence']. Pure orchestration — no LLM call here.

Tool dispatch uses the shaped_query and extracted entities (drug_name, state_name)
plus, where useful, the patient's clinical codes from patient_context.
"""

from __future__ import annotations

import logging

from agents.state import PipelineState
from models.schemas import Evidence
from tools import medlineplus_tool, openfda_tool, qdrant_tool, sqlite_tool

logger = logging.getLogger(__name__)


def _call_tool(name: str, state: PipelineState) -> list[Evidence]:
    """Dispatch a single tool by name using the current state."""
    query     = state.get("shaped_query") or state["query"]
    drug      = state.get("drug_name")
    state_nm  = state.get("state_name")
    ctx       = state.get("patient_context")

    try:
        if name == "search_drug_labels":
            # Retrieve a large candidate pool; the Evidence Filter's cross-encoder
            # reranker narrows it down to the most relevant few (retrieve-then-rerank).
            return qdrant_tool.search_drug_labels(query, limit=15)

        if name == "search_medlineplus":
            return qdrant_tool.search_medlineplus(query, limit=10)

        if name == "fetch_drug_label":
            if drug:
                return openfda_tool.fetch_drug_label(drug)
            return []

        if name == "check_drug_recalls":
            if drug:
                return openfda_tool.check_drug_recalls(drug)
            return []

        if name == "check_drug_shortages":
            if drug:
                return openfda_tool.check_drug_shortages(drug)
            return []

        if name == "lookup_drug_price":
            if drug:
                return sqlite_tool.lookup_drug_price(drug, limit=3)
            return []

        if name == "lookup_state_eligibility":
            if state_nm:
                return sqlite_tool.lookup_state_eligibility(state_nm)
            return []

        if name == "explain_condition_snomed":
            # Use patient's conditions if available; else skip (needs a code)
            results: list[Evidence] = []
            if ctx:
                conditions = ctx.get("conditions_json") or ctx.get("conditions") or []
                for c in conditions[:3]:
                    code = c.get("snomed_code", "")
                    if code:
                        results.extend(
                            medlineplus_tool.explain_condition_snomed(
                                code, c.get("display", ""), limit=1
                            )
                        )
            return results

        if name == "explain_drug_rxnorm":
            results = []
            if ctx:
                meds = ctx.get("medications_json") or ctx.get("medications") or []
                for m in meds[:3]:
                    if m.get("status") == "active" and m.get("rxnorm_code"):
                        results.extend(
                            medlineplus_tool.explain_drug_rxnorm(
                                m["rxnorm_code"], m.get("display", ""), limit=1
                            )
                        )
            return results

    except Exception as e:
        logger.warning("Tool %s failed: %s", name, e)
        return []

    logger.warning("Unknown tool requested: %s", name)
    return []


def retrieval_node(state: PipelineState) -> dict:
    """LangGraph node: call all selected tools, gather raw evidence."""
    tools = state.get("tools_to_call", [])
    all_evidence: list[Evidence] = []
    per_tool_counts = {}

    for tool_name in tools:
        evidence = _call_tool(tool_name, state)
        per_tool_counts[tool_name] = len(evidence)
        all_evidence.extend(evidence)

    trace = state.get("trace", [])
    trace = trace + [
        f"Retrieval: {sum(per_tool_counts.values())} evidence items "
        f"from {dict(per_tool_counts)}"
    ]

    return {
        "raw_evidence": all_evidence,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agents.router import router_node
    from agents.state import new_state

    state = new_state("What are the warnings for metformin?", None, "anonymous")
    state.update(router_node(state))
    result = retrieval_node(state)

    print(f"\nRetrieved {len(result['raw_evidence'])} evidence items:")
    for e in result["raw_evidence"][:6]:
        score = f"{e.score:.3f}" if e.score is not None else "  -  "
        print(f"  [{score}] ({e.source}) {e.title[:55]}")

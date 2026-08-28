"""
Agent 2 — Retrieval — agents/retrieval.py

Calls the tools selected by the Router and collects their Evidence into
state['raw_evidence']. Pure orchestration — no LLM call here.

Tool dispatch uses the shaped_query and extracted entities (drug_name, state_name)
plus, where useful, the patient's clinical codes from patient_context.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from agents.state import PipelineState
from models.schemas import Evidence
from tools import medlineplus_tool, openfda_tool, qdrant_tool, sqlite_tool

logger = logging.getLogger(__name__)

# --- R9 fan-out -----------------------------------------------------------
# Only these tools are driven by the free-text query, so only these are worth
# re-running per sub-query. The entity-driven tools (fetch_drug_label, the
# explain_* Connect lookups, the SQL lookups) ignore the query text entirely and
# would just return the same rows N times.
QUERY_DRIVEN_TOOLS = {"search_drug_labels", "search_medlineplus", "search_medlineplus_live"}

# Per-sub-query limits. Deliberately smaller than the single-query path: with N
# sub-queries the candidate pool is N x this, and the cross-encoder has to score
# every one of them.
FANOUT_LIMITS = {"search_drug_labels": 6, "search_medlineplus": 3, "search_medlineplus_live": 3}

FANOUT_MAX_WORKERS = 6   # sub-queries are independent network calls


def _call_tool(name: str, state: PipelineState,
               query_override: str | None = None,
               limit_override: int | None = None) -> list[Evidence]:
    """Dispatch a single tool by name using the current state.

    query_override / limit_override let the R9 fan-out path reuse this dispatcher
    for a focused sub-query without duplicating the tool table.
    """
    query     = query_override or state.get("shaped_query") or state["query"]
    drug      = state.get("drug_name")
    state_nm  = state.get("state_name")
    ctx       = state.get("patient_context")

    try:
        if name == "search_drug_labels":
            # Retrieve a large candidate pool; the Evidence Filter's cross-encoder
            # reranker narrows it down to the most relevant few (retrieve-then-rerank).
            return qdrant_tool.search_drug_labels(query, limit=limit_override or 15)

        if name == "search_medlineplus":
            return qdrant_tool.search_medlineplus(query, limit=limit_override or 10)

        if name == "search_medlineplus_live":
            # Live health-topics keyword search — corrective-RAG fallback used only
            # on a reviewer-triggered retry (see INTENT_RETRY_EXTRA_TOOLS). This is a
            # LEXICAL search, so feed it the ORIGINAL user question, not the keyword-
            # enriched shaped_query (which is tuned for vector retrieval and matches
            # the lexical index poorly).
            return medlineplus_tool.search_health_topics_live(
                query_override or state.get("query") or query, limit=limit_override or 5)

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

        if name == "explain_drug_by_name":
            # Name-based MedlinePlus drug monograph (Connect .v.dn fallback). Works
            # with OR without a selected patient — uses the Router's extracted drug name.
            return medlineplus_tool.explain_drug_by_name(drug) if drug else []

        if name == "explain_condition_snomed":
            # Explain the patient's conditions (needs SNOMED codes from their record).
            # Connect lookups run concurrently — N round-trips collapse to ~one.
            if not ctx:
                return []
            conditions = ctx.get("conditions_json") or ctx.get("conditions") or []
            jobs = [
                (lambda c=c: medlineplus_tool.explain_condition_snomed(
                    c["snomed_code"], c.get("display", ""), limit=1))
                for c in conditions[:3] if c.get("snomed_code")
            ]
            return medlineplus_tool.explain_batch(jobs)

        if name == "explain_drug_rxnorm":
            # Explain the patient's active meds by RxNorm code (code + name = most robust).
            if not ctx:
                return []
            meds = ctx.get("medications_json") or ctx.get("medications") or []
            jobs = [
                (lambda m=m: medlineplus_tool.explain_drug_rxnorm(
                    m["rxnorm_code"], m.get("display", ""), limit=1))
                for m in meds[:3]
                if m.get("status") == "active" and m.get("rxnorm_code")
            ]
            return medlineplus_tool.explain_batch(jobs)

        if name == "explain_lab_loinc":
            # Explain the patient's ABNORMAL labs by LOINC code (concurrently).
            if not ctx:
                return []
            labs = ctx.get("labs_json") or ctx.get("labs") or []
            jobs = [
                (lambda l=l: medlineplus_tool.explain_lab_loinc(
                    l["loinc_code"], l.get("display", ""), limit=1))
                for l in labs if l.get("is_abnormal") and l.get("loinc_code")
            ][:3]
            return medlineplus_tool.explain_batch(jobs)

        if name == "explain_procedure_snomed":
            # Explain the patient's recent procedures by SNOMED code (concurrently).
            if not ctx:
                return []
            procedures = ctx.get("procedures_json") or ctx.get("procedures") or []
            jobs = [
                (lambda p=p: medlineplus_tool.explain_procedure_snomed(
                    p["code"], p.get("display", ""), limit=1))
                for p in procedures[:3] if p.get("code")
            ]
            return medlineplus_tool.explain_batch(jobs)

    except Exception as e:
        logger.warning("Tool %s failed: %s", name, e)
        return []

    logger.warning("Unknown tool requested: %s", name)
    return []


def _fan_out(state: PipelineState, tools: list[str],
             sub_queries: list[str]) -> tuple[list[Evidence], dict]:
    """Run the query-driven tools once PER sub-query, concurrently.

    Every returned Evidence is tagged with metadata['sub_query'] so the Evidence
    Filter can rerank it against the query that actually fetched it — that tagging
    is what fixes the rerank mismatch, not the extra retrieval on its own.
    """
    query_tools = [t for t in tools if t in QUERY_DRIVEN_TOOLS]
    counts: dict[str, int] = {}
    collected: list[Evidence] = []

    def _one(sq: str) -> list[Evidence]:
        out: list[Evidence] = []
        for tool_name in query_tools:
            try:
                items = _call_tool(tool_name, state, query_override=sq,
                                   limit_override=FANOUT_LIMITS.get(tool_name))
            except Exception as e:                     # one bad sub-query must not
                logger.warning("fan-out %s/%s failed: %s", tool_name, sq, e)
                continue                               # sink the whole fan-out
            for ev in items:
                ev.metadata = {**(ev.metadata or {}), "sub_query": sq}
            out.extend(items)
        return out

    if sub_queries:
        with ThreadPoolExecutor(max_workers=min(FANOUT_MAX_WORKERS, len(sub_queries))) as pool:
            for sq, items in zip(sub_queries, pool.map(_one, sub_queries)):
                counts[f"fanout::{sq[:28]}"] = len(items)
                collected.extend(items)

    # Entity-driven tools ignore the query text, so run them once on the base state.
    for tool_name in [t for t in tools if t not in QUERY_DRIVEN_TOOLS]:
        items = _call_tool(tool_name, state)
        counts[tool_name] = len(items)
        collected.extend(items)

    return collected, counts


def retrieval_node(state: PipelineState) -> dict:
    """LangGraph node: call all selected tools, gather raw evidence.

    Two modes: the normal single-query path, and the R9 fan-out path when the
    Router flagged a diffuse multi-entity question.
    """
    tools = state.get("tools_to_call", [])
    sub_queries = state.get("sub_queries") or []

    if sub_queries:
        all_evidence, per_tool_counts = _fan_out(state, tools, sub_queries)
        mode = f" [FAN-OUT x{len(sub_queries)}]"
    else:
        all_evidence = []
        per_tool_counts = {}
        for tool_name in tools:
            evidence = _call_tool(tool_name, state)
            per_tool_counts[tool_name] = len(evidence)
            all_evidence.extend(evidence)
        mode = ""

    trace = state.get("trace", [])
    trace = trace + [
        f"Retrieval: {sum(per_tool_counts.values())} evidence items "
        f"from {dict(per_tool_counts)}{mode}"
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

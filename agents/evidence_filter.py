"""
Agent 3 — Evidence Filter — agents/evidence_filter.py

Cleans up raw evidence before it reaches the Answer Generator, using a
retrieve-then-rerank strategy:

  1. Deduplicate near-identical items FIRST (cheap) — same source+drug+section,
     or highly overlapping text. Done before reranking so we don't pay to score
     redundant pairs.
  2. RERANK the survivors with a cross-encoder (tools/reranker.py) against the
     user's query. This reads (query, passage) jointly and gives every item —
     vector hits, live-API results, SQL rows — one comparable relevance score.
  3. Drop items below a relevance floor, then cap to MAX_EVIDENCE.

If the cross-encoder is unavailable (no torch / model), it falls back to the
previous bi-encoder ordering (score threshold + a mid-priority constant for
non-scored items) so the pipeline still works.
"""

from __future__ import annotations

import logging
import re

from agents.state import PipelineState
from models.schemas import Evidence
from tools import reranker

logger = logging.getLogger(__name__)

# --- Reranked path ---
RERANK_MIN_SCORE = 0.02   # drop items the cross-encoder scores as clearly irrelevant
MAX_EVIDENCE = 8          # final cap on RETRIEVED evidence (patient record is extra)

# --- Fallback (bi-encoder) path ---
BI_ENCODER_THRESHOLD = 0.30
NONSCORED_PRIORITY = 0.55

# --- R9 fan-out path ---
# Each sub-query's chunks are reranked against THAT sub-query, then merged
# round-robin so every drug gets its best chunk before any drug gets a second.
PER_SUBQUERY_KEEP = 2
FANOUT_MAX_EVIDENCE = 10   # only modestly above MAX_EVIDENCE: instrumentation showed
                           # the Reviewer re-reads the whole evidence block and is the
                           # pipeline's largest input consumer, so each extra chunk is
                           # paid for twice (generate + review).

# Intents for which the patient's own record is relevant grounding.
# (Policy/eligibility and general questions don't need patient clinical context.)
CLINICAL_INTENTS = {"medication_info", "condition_education", "drug_recall"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _dedup_key(e: Evidence) -> str:
    drug    = e.metadata.get("drug_name", "")
    section = e.metadata.get("section_type", "")
    if drug and section:
        return f"{e.source}:{drug}:{section}"
    return f"{e.source}:{_normalize(e.text)[:80]}"


def _is_near_duplicate(a: Evidence, b: Evidence) -> bool:
    """True if a and b carry essentially the same content. Two signals (either fires):

    - Jaccard > 0.85: classic near-identical text of similar length.
    - Containment > 0.90: one text is almost entirely a SUBSET of the other. This
      catches the same source article reached two ways — e.g. MedlinePlus Connect's
      600-char truncated summary vs the full topic text from the Qdrant index — where
      the length gap crushes Jaccard (intersection ÷ a much larger union) even though
      the shorter side is ~fully contained in the longer. The containment rule requires
      the smaller side to have >= 8 tokens so a short generic line ("no recalls found")
      can't be spuriously "contained" in an unrelated long passage.
    """
    ta = set(_normalize(a.text).split())
    tb = set(_normalize(b.text).split())
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    jaccard = inter / len(ta | tb)
    smaller = min(len(ta), len(tb))
    containment = inter / smaller
    return jaccard > 0.85 or (smaller >= 8 and containment > 0.90)


def _dedup(evidence: list[Evidence]) -> list[Evidence]:
    """Exact-key dedup followed by near-duplicate (token-Jaccard) dedup."""
    seen_keys: set[str] = set()
    unique: list[Evidence] = []
    for e in evidence:
        key = _dedup_key(e)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(e)

    deduped: list[Evidence] = []
    for e in unique:
        if any(_is_near_duplicate(e, kept) for kept in deduped):
            continue
        deduped.append(e)
    return deduped


def build_patient_evidence(ctx: dict) -> Evidence | None:
    """Build a compact patient-record Evidence from the selected patient's data.

    Returns None if the patient has no usable clinical facts. This becomes a
    first-class evidence item (source='patient_record') so the Answer Generator
    can cite it and the Reviewer's faithfulness check counts patient facts as
    valid grounding — rather than treating patient data as a side-channel the
    review machinery doesn't know about.
    """
    conditions = ctx.get("conditions_json") or ctx.get("conditions") or []
    meds       = ctx.get("medications_json") or ctx.get("medications") or []
    labs       = ctx.get("labs_json") or ctx.get("labs") or []
    procedures = ctx.get("procedures_json") or ctx.get("procedures") or []

    cond_names = [c.get("display", "") for c in conditions if c.get("display")][:10]
    active_meds = [m.get("display", "") for m in meds
                   if m.get("status") == "active" and m.get("display")][:10]
    abnormal_labs = [
        f"{l.get('display', '')}: {l.get('value', '')} {l.get('unit', '') or ''}".strip()
        for l in labs if l.get("is_abnormal")
    ][:10]
    proc_names = [p.get("display", "") for p in procedures if p.get("display")][:8]

    if not (cond_names or active_meds or abnormal_labs or proc_names):
        return None

    parts = [f"Patient: {ctx.get('name', 'unknown')}, "
             f"{ctx.get('age', '?')}yo {ctx.get('gender', '')}"]
    if cond_names:
        parts.append("Active conditions: " + "; ".join(cond_names))
    if active_meds:
        parts.append("Active medications: " + "; ".join(active_meds))
    if abnormal_labs:
        parts.append("Abnormal lab results: " + "; ".join(abnormal_labs))
    if proc_names:
        parts.append("Recent procedures: " + "; ".join(proc_names))

    return Evidence(
        source="patient_record",
        title=f"Patient Record — {ctx.get('name', 'patient')}",
        text="\n".join(parts),
        # score=None: the patient record bypasses the cross-encoder (it's definitional
        # context, not a candidate competing for relevance). It is pinned to position [1]
        # by explicit prepending, NOT by a score. Using None — rather than a fake 1.0 —
        # keeps it honest: it has no rerank score, like the SQL/API evidence items.
        score=None,
        metadata={
            "patient_id": ctx.get("patient_id", ""),
            "name":       ctx.get("name", ""),
            "age":        ctx.get("age"),
            "gender":     ctx.get("gender", ""),
        },
        citation="Patient Record (selected patient's FHIR data)",
    )


def _fallback_order(evidence: list[Evidence]) -> list[Evidence]:
    """Bi-encoder ordering used when the cross-encoder is unavailable."""
    kept = [e for e in evidence
            if e.score is None or e.score >= BI_ENCODER_THRESHOLD]
    kept.sort(
        key=lambda e: e.score if e.score is not None else NONSCORED_PRIORITY,
        reverse=True,
    )
    return kept[:MAX_EVIDENCE]


def _rerank_by_sub_query(evidence: list[Evidence], fallback_query: str) -> list[Evidence]:
    """Rerank each sub-query's evidence against ITS OWN sub-query, then interleave.

    This is the actual R9 fix. Reranking a per-drug label section against the diffuse
    question ("what should I watch for across their meds?") scores it low and drops it;
    reranking it against "metformin adverse reactions" scores it correctly.

    Untagged items (from entity-driven tools, which ignore the query text) are graded
    against the original user query as before.

    Merge is ROUND-ROBIN across groups, not global-best: a global sort would let one
    drug with strong lexical overlap take every slot under the cap, which is exactly
    the coverage failure we are fixing.
    """
    groups: dict[str, list[Evidence]] = {}
    for e in evidence:
        groups.setdefault((e.metadata or {}).get("sub_query") or "", []).append(e)

    ranked: dict[str, list[Evidence]] = {}
    for sub_query, items in groups.items():
        query = sub_query or fallback_query
        scored = reranker.rerank(query, items, top_k=None)
        keep = [e for e in scored if (e.score or 0.0) >= RERANK_MIN_SCORE]
        # An untagged group is not a per-drug bucket, so it gets no per-drug quota.
        ranked[sub_query] = keep if sub_query == "" else keep[:PER_SUBQUERY_KEEP]

    merged: list[Evidence] = []
    for rank in range(max((len(v) for v in ranked.values()), default=0)):
        for sub_query in groups:                       # dict preserves insertion order
            bucket = ranked.get(sub_query, [])
            if rank < len(bucket):
                merged.append(bucket[rank])
    return merged


def evidence_filter_node(state: PipelineState) -> dict:
    """LangGraph node: dedup → cross-encoder rerank → trim."""
    raw = state.get("raw_evidence", [])

    # Step 1: dedup first (cheap — fewer pairs to rerank)
    deduped = _dedup(raw)

    # Step 2: rerank (or fall back to bi-encoder order)
    fanned_out = any((e.metadata or {}).get("sub_query") for e in deduped)

    if reranker.is_available() and deduped:
        # Rerank against the ORIGINAL user query — cross-encoders judge relevance
        # to what the user actually asked, not the keyword-enriched shaped query.
        query = state.get("query", "")
        if fanned_out:
            filtered = _rerank_by_sub_query(deduped, query)[:FANOUT_MAX_EVIDENCE]
            method = "cross-encoder/fan-out"
        else:
            reranked = reranker.rerank(query, deduped, top_k=None)
            # Drop clearly-irrelevant items, then cap
            filtered = [e for e in reranked if (e.score or 0.0) >= RERANK_MIN_SCORE]
            filtered = filtered[:MAX_EVIDENCE]
            method = "cross-encoder"
    else:
        filtered = _fallback_order(deduped)
        if fanned_out:
            filtered = filtered[:FANOUT_MAX_EVIDENCE]
        method = "bi-encoder-fallback"

    # Inject the patient record as a first-class evidence item for clinical intents.
    # Added AFTER ranking/cap so it's always present (it's contextual grounding, not
    # competing for relevance) and placed first so the answer cites it as [1].
    patient_injected = False
    ctx = state.get("patient_context")
    if ctx and state.get("intent") in CLINICAL_INTENTS:
        patient_ev = build_patient_evidence(ctx)
        if patient_ev is not None:
            filtered = [patient_ev] + filtered
            patient_injected = True

    trace = state.get("trace", [])
    trace = trace + [
        f"Filter: {len(raw)} raw -> {len(deduped)} deduped -> "
        f"{len(filtered)} kept ({method}, "
        f"cap={FANOUT_MAX_EVIDENCE if fanned_out else MAX_EVIDENCE}, "
        f"patient_record={'yes' if patient_injected else 'no'})"
    ]

    return {
        "filtered_evidence": filtered,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agents.retrieval import retrieval_node
    from agents.router import router_node
    from agents.state import new_state

    state = new_state("What should a kidney patient know about metformin?", None, "patient")
    state.update(router_node(state))
    state.update(retrieval_node(state))
    result = evidence_filter_node(state)

    print(f"\nRaw: {len(state['raw_evidence'])} -> Filtered: {len(result['filtered_evidence'])}")
    for e in result["filtered_evidence"]:
        score = f"{e.score:.3f}" if e.score is not None else "  -  "
        print(f"  [{score}] ({e.source}) {e.title[:55]}")

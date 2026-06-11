"""
Tests for the patient_summary intent path (Option B: serve the pre-computed summary).

Covers:
  - tools.sqlite_tool.get_patient_summary
  - agents.patient_summary.patient_summary_node (with/without context, no summary)
  - graph routing: _route_after_router sends patient_summary intent to the right node

Run with:  python -m pytest tests/test_patient_summary.py -v
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
            override=True)

from agents.patient_summary import (
    patient_summary_node,
    _NO_PATIENT_MSG,
    _NO_DATA_MSG,
    _BASIC_SUMMARY_HEADER,
)
from agents.state import new_state
from ingestion.sqlite_loader import MedDocDB
from tools.sqlite_tool import get_patient_summary

DB_PATH = "data/meddocai.db"


def _a_patient_with_summary() -> dict:
    db = MedDocDB(DB_PATH)
    for p in db.list_patients():
        if p["has_summary"]:
            return db.get_patient(p["patient_id"])
    return None


# ---------------------------------------------------------------------------
# get_patient_summary tool
# ---------------------------------------------------------------------------

class TestGetPatientSummaryTool:
    def test_returns_evidence_for_patient_with_summary(self):
        patient = _a_patient_with_summary()
        assert patient is not None, "No patient with a summary in DB (run batch pipeline)"
        ev = get_patient_summary(patient["patient_id"])
        assert len(ev) == 1
        assert ev[0].source == "patient_summary"
        assert ev[0].text == patient["summary_md"]
        assert ev[0].metadata["patient_id"] == patient["patient_id"]

    def test_returns_empty_for_unknown_patient(self):
        assert get_patient_summary("nonexistent-uuid-000") == []


# ---------------------------------------------------------------------------
# patient_summary_node
# ---------------------------------------------------------------------------

class TestPatientSummaryNode:
    def test_serves_stored_summary(self):
        patient = _a_patient_with_summary()
        state = new_state("summarize this patient", patient_context=patient,
                          user_role="care_manager")
        out = patient_summary_node(state)
        assert out["answer"] == patient["summary_md"]
        assert out["review_passed"] is True          # skips review loop
        assert len(out["citations"]) == 1
        assert len(out["filtered_evidence"]) == 1

    def test_no_patient_context_prompts_selection(self):
        state = new_state("summarize this patient", None, "anonymous")
        out = patient_summary_node(state)
        assert out["answer"] == _NO_PATIENT_MSG
        assert out["review_passed"] is True
        assert out["citations"] == []

    def test_no_summary_but_structured_data_serves_basic_profile(self):
        # Patient with FHIR data but no narrative summary (un-batched / new)
        state = new_state("summarize", patient_context={
            "patient_id": "no-such-id",   # not in DB -> no stored summary
            "name": "New Patient", "age": 60, "gender": "female",
            "conditions_json": [{"display": "Type 2 Diabetes", "snomed_code": "44054006"}],
            "medications_json": [{"display": "Metformin", "status": "active"}],
            "labs_json": [{"display": "eGFR", "value": 45, "is_abnormal": True}],
        }, user_role="care_manager")
        out = patient_summary_node(state)
        assert _BASIC_SUMMARY_HEADER.strip()[:20] in out["answer"]
        assert "Type 2 Diabetes" in out["answer"]
        assert out["review_passed"] is True

    def test_no_summary_no_data_returns_no_data_msg(self):
        # Patient id but zero clinical facts
        state = new_state("summarize", patient_context={
            "patient_id": "empty", "name": "Empty", "age": 30, "gender": "male",
            "conditions_json": [], "medications_json": [], "labs_json": [],
        }, user_role="care_manager")
        out = patient_summary_node(state)
        assert out["answer"] == _NO_DATA_MSG
        assert out["review_passed"] is True

    def test_context_missing_patient_id(self):
        state = new_state("summarize", patient_context={"name": "X"},  # no patient_id
                          user_role="care_manager")
        out = patient_summary_node(state)
        assert out["answer"] == _NO_PATIENT_MSG


# ---------------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------------

class TestRouterBranch:
    def test_patient_summary_intent_routes_to_summary_node(self):
        from graph.pipeline import _route_after_router
        state = new_state("q")
        state["intent"] = "patient_summary"
        assert _route_after_router(state) == "patient_summary"

    def test_other_intents_route_to_retrieval(self):
        from graph.pipeline import _route_after_router
        for intent in ["medication_info", "drug_recall", "policy_eligibility",
                       "condition_education", "general"]:
            state = new_state("q")
            state["intent"] = intent
            assert _route_after_router(state) == "retrieval"

    def test_pipeline_builds_with_summary_node(self):
        from graph.pipeline import build_pipeline
        assert build_pipeline() is not None


# ---------------------------------------------------------------------------
# Full pipeline integration (live LLM — router classification)
# ---------------------------------------------------------------------------

@pytest.mark.llm
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestPatientSummaryIntegration:
    def test_summary_query_serves_stored_summary(self):
        from graph.pipeline import answer_query
        patient = _a_patient_with_summary()
        final = answer_query("Summarize this patient for me",
                             patient_context=patient, user_role="care_manager")
        assert final["intent"] == "patient_summary"
        # The served answer should be the stored summary (contained in final_answer)
        assert patient["summary_md"][:50] in final["final_answer"]

    def test_summary_query_no_patient(self):
        from graph.pipeline import answer_query
        final = answer_query("Summarize this patient for me", None, "anonymous")
        assert final["intent"] == "patient_summary"
        assert "select" in final["final_answer"].lower()

"""
Tests for patient-record-as-evidence injection (Option B).

Covers:
  - build_patient_evidence() constructs a valid Evidence from patient context
  - evidence_filter injects it for clinical intents, skips it otherwise
  - the patient record is always present (not dropped by the rerank cut)

Run with:  python -m pytest tests/test_patient_evidence.py -v
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
            override=True)

from agents.evidence_filter import (
    build_patient_evidence,
    evidence_filter_node,
    CLINICAL_INTENTS,
)
from agents.state import new_state
from models.schemas import Evidence


def _patient_ctx():
    return {
        "patient_id": "p1",
        "name": "Test Patient",
        "age": 67,
        "gender": "male",
        "conditions_json": [
            {"display": "Type 2 Diabetes", "snomed_code": "44054006"},
            {"display": "Chronic Kidney Disease", "snomed_code": "709044004"},
        ],
        "medications_json": [
            {"display": "Metformin 500 MG", "status": "active", "rxnorm_code": "860975"},
            {"display": "Old Drug", "status": "stopped", "rxnorm_code": "111"},
        ],
        "labs_json": [
            {"display": "eGFR", "value": 42, "unit": "mL/min", "is_abnormal": True},
            {"display": "Sodium", "value": 140, "unit": "mEq/L", "is_abnormal": False},
        ],
    }


def _ev(title, text, score=0.8, **md):
    return Evidence(source="fda_label", title=title, text=text, score=score,
                    metadata=md, citation=title)


# ---------------------------------------------------------------------------
# build_patient_evidence
# ---------------------------------------------------------------------------

class TestBuildPatientEvidence:
    def test_builds_evidence(self):
        ev = build_patient_evidence(_patient_ctx())
        assert ev is not None
        assert ev.source == "patient_record"
        # score=None: it bypasses the reranker and is pinned by position, not score
        assert ev.score is None

    def test_includes_conditions(self):
        ev = build_patient_evidence(_patient_ctx())
        assert "Type 2 Diabetes" in ev.text
        assert "Chronic Kidney Disease" in ev.text

    def test_only_active_meds(self):
        ev = build_patient_evidence(_patient_ctx())
        assert "Metformin" in ev.text
        assert "Old Drug" not in ev.text   # stopped med excluded

    def test_only_abnormal_labs(self):
        ev = build_patient_evidence(_patient_ctx())
        assert "eGFR" in ev.text           # abnormal -> included
        assert "Sodium" not in ev.text     # normal -> excluded

    def test_empty_context_returns_none(self):
        ev = build_patient_evidence({"name": "X", "conditions_json": [],
                                     "medications_json": [], "labs_json": []})
        assert ev is None

    def test_citation_set(self):
        ev = build_patient_evidence(_patient_ctx())
        assert "Patient Record" in ev.citation


# ---------------------------------------------------------------------------
# evidence_filter injection
# ---------------------------------------------------------------------------

class TestPatientEvidenceInjection:
    def test_injected_for_clinical_intent(self):
        state = new_state("is metformin safe?", patient_context=_patient_ctx(),
                          user_role="care_manager")
        state["intent"] = "medication_info"
        state["raw_evidence"] = [
            _ev("a", "metformin contraindications renal impairment",
                drug_name="metformin", section_type="contraindications")
        ]
        out = evidence_filter_node(state)
        sources = [e.source for e in out["filtered_evidence"]]
        assert "patient_record" in sources
        # Patient record should be first
        assert out["filtered_evidence"][0].source == "patient_record"

    def test_skipped_for_policy_intent(self):
        state = new_state("medicaid limit in texas?", patient_context=_patient_ctx(),
                          user_role="care_manager")
        state["intent"] = "policy_eligibility"
        state["raw_evidence"] = [
            Evidence(source="eligibility", title="TX", text="income limits",
                     score=None, citation="CMS")
        ]
        out = evidence_filter_node(state)
        sources = [e.source for e in out["filtered_evidence"]]
        assert "patient_record" not in sources

    def test_skipped_when_no_patient(self):
        state = new_state("is metformin safe?", patient_context=None,
                          user_role="anonymous")
        state["intent"] = "medication_info"
        state["raw_evidence"] = [
            _ev("a", "metformin info", drug_name="metformin", section_type="warnings")
        ]
        out = evidence_filter_node(state)
        sources = [e.source for e in out["filtered_evidence"]]
        assert "patient_record" not in sources

    def test_clinical_intents_set(self):
        assert "medication_info" in CLINICAL_INTENTS
        assert "condition_education" in CLINICAL_INTENTS
        assert "drug_recall" in CLINICAL_INTENTS
        assert "policy_eligibility" not in CLINICAL_INTENTS
        assert "general" not in CLINICAL_INTENTS

    def test_patient_record_survives_when_many_candidates(self):
        """Patient record is added after the cap, so it's never dropped."""
        state = new_state("metformin warnings", patient_context=_patient_ctx(),
                          user_role="care_manager")
        state["intent"] = "medication_info"
        # 20 retrieved candidates (more than MAX_EVIDENCE)
        state["raw_evidence"] = [
            _ev(f"c{i}", f"distinct metformin warning text number {i}",
                score=0.8, drug_name=f"d{i}", section_type="warnings")
            for i in range(20)
        ]
        out = evidence_filter_node(state)
        assert out["filtered_evidence"][0].source == "patient_record"

"""
Tests for the tools layer (tools/).

Network-dependent tests (openFDA, MedlinePlus Connect) and embedding-dependent
tests (Qdrant) are marked and will be skipped if keys/services are unavailable.

Run with:  python -m pytest tests/test_tools.py -v
"""

import os
import pytest
from dotenv import load_dotenv

# Load .env so OPENAI_API_KEY is available at collection time (for skipif checks)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
            override=True)

from models.schemas import Evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _assert_valid_evidence(e: Evidence):
    """Every Evidence must have the core fields populated."""
    assert isinstance(e, Evidence)
    assert e.source
    assert e.title
    assert isinstance(e.text, str)
    assert isinstance(e.metadata, dict)
    assert e.citation


# ---------------------------------------------------------------------------
# SQLite tool (no network, no embeddings — always runs)
# ---------------------------------------------------------------------------

class TestSqliteTool:
    def test_drug_price_returns_evidence(self):
        from tools.sqlite_tool import lookup_drug_price
        results = lookup_drug_price("metformin", limit=3)
        assert len(results) > 0
        for e in results:
            _assert_valid_evidence(e)
            assert e.source == "nadac"
            assert e.metadata.get("ndc")

    def test_drug_price_unknown_drug(self):
        from tools.sqlite_tool import lookup_drug_price
        results = lookup_drug_price("zzznotreal999", limit=3)
        assert results == []

    def test_eligibility_returns_evidence(self):
        from tools.sqlite_tool import lookup_state_eligibility
        results = lookup_state_eligibility("Texas")
        assert len(results) == 1
        e = results[0]
        _assert_valid_evidence(e)
        assert e.source == "eligibility"
        assert e.metadata["state"] == "Texas"
        assert "%" in e.text

    def test_eligibility_case_insensitive(self):
        from tools.sqlite_tool import lookup_state_eligibility
        lower = lookup_state_eligibility("texas")
        upper = lookup_state_eligibility("TEXAS")
        assert len(lower) == 1 and len(upper) == 1
        assert lower[0].metadata["state"] == upper[0].metadata["state"]

    def test_eligibility_unknown_state(self):
        from tools.sqlite_tool import lookup_state_eligibility
        assert lookup_state_eligibility("Narnia") == []

    def test_list_states(self):
        from tools.sqlite_tool import list_available_states
        states = list_available_states()
        assert len(states) == 51
        assert "Texas" in states


# ---------------------------------------------------------------------------
# Qdrant tool (needs OPENAI_API_KEY for embeddings)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_openai(), reason="OPENAI_API_KEY not set")
class TestQdrantTool:
    def test_drug_label_search(self):
        from tools.qdrant_tool import search_drug_labels
        results = search_drug_labels("metformin kidney warnings", limit=3)
        assert len(results) > 0
        for e in results:
            _assert_valid_evidence(e)
            assert e.source == "fda_label"
            assert e.score is not None
            assert 0 <= e.score <= 1

    def test_drug_label_relevance(self):
        """Metformin renal query should surface metformin."""
        from tools.qdrant_tool import search_drug_labels
        results = search_drug_labels("metformin contraindications renal", limit=3)
        assert any("metformin" in e.metadata.get("drug_name", "").lower()
                   for e in results)

    def test_drug_label_section_filter(self):
        from tools.qdrant_tool import search_drug_labels
        results = search_drug_labels("metformin", limit=3,
                                     filters={"section_type": "contraindications"})
        assert len(results) > 0
        for e in results:
            assert e.metadata["section_type"] == "contraindications"

    def test_medlineplus_search(self):
        from tools.qdrant_tool import search_medlineplus
        results = search_medlineplus("high blood pressure", limit=3)
        assert len(results) > 0
        for e in results:
            _assert_valid_evidence(e)
            assert e.source == "medlineplus"

    def test_medlineplus_relevance(self):
        from tools.qdrant_tool import search_medlineplus
        results = search_medlineplus("what is high blood pressure hypertension", limit=3)
        titles = " ".join(e.title.lower() for e in results)
        assert "blood pressure" in titles or "hypertension" in titles

    def test_scores_descending(self):
        from tools.qdrant_tool import search_drug_labels
        results = search_drug_labels("diabetes medication", limit=5)
        scores = [e.score for e in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# openFDA tool (needs network)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestOpenFDATool:
    def test_fetch_drug_label(self):
        from tools.openfda_tool import fetch_drug_label
        results = fetch_drug_label("metformin")
        assert len(results) > 0
        for e in results:
            _assert_valid_evidence(e)
            assert e.source == "openfda_api"

    def test_fetch_drug_label_has_safety_sections(self):
        from tools.openfda_tool import fetch_drug_label
        results = fetch_drug_label("metformin")
        sections = {e.metadata.get("section_type") for e in results}
        # Metformin should have contraindications and/or warnings
        assert sections & {"contraindications", "warnings_and_cautions",
                           "boxed_warning", "warnings"}

    def test_otc_label_has_consumer_sections(self):
        from tools.openfda_tool import fetch_drug_label
        results = fetch_drug_label("ibuprofen", prefer_otc=True)
        sections = {e.metadata.get("section_type") for e in results}
        # OTC labels should have consumer-friendly sections
        assert sections & {"purpose", "ask_doctor", "when_using",
                           "stop_use", "warnings", "indications_and_usage"}

    def test_recalls_returns_evidence(self):
        from tools.openfda_tool import check_drug_recalls
        results = check_drug_recalls("semaglutide")
        assert len(results) > 0
        for e in results:
            _assert_valid_evidence(e)

    def test_recalls_unknown_drug_returns_no_recalls_msg(self):
        from tools.openfda_tool import check_drug_recalls
        results = check_drug_recalls("zzznotadrug999")
        assert len(results) == 1
        assert results[0].metadata.get("recall_count") == 0

    def test_shortages_returns_evidence(self):
        from tools.openfda_tool import check_drug_shortages
        results = check_drug_shortages("methotrexate")
        assert len(results) > 0
        _assert_valid_evidence(results[0])


# ---------------------------------------------------------------------------
# MedlinePlus Connect tool (needs network)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestMedlinePlusTool:
    def test_explain_condition_snomed(self):
        from tools.medlineplus_tool import explain_condition_snomed
        results = explain_condition_snomed("38341003")  # Hypertension
        assert len(results) > 0
        for e in results:
            _assert_valid_evidence(e)
            assert e.source == "medlineplus_connect"

    def test_condition_relevance(self):
        from tools.medlineplus_tool import explain_condition_snomed
        results = explain_condition_snomed("38341003")
        titles = " ".join(e.title.lower() for e in results)
        assert "blood pressure" in titles or "hypertension" in titles

    def test_explain_drug_rxnorm(self):
        from tools.medlineplus_tool import explain_drug_rxnorm
        results = explain_drug_rxnorm("860975")  # Metformin
        assert len(results) > 0
        titles = " ".join(e.title.lower() for e in results)
        assert "metformin" in titles or "diabetes" in titles

    def test_explain_lab_loinc(self):
        from tools.medlineplus_tool import explain_lab_loinc
        results = explain_lab_loinc("4548-4")  # HbA1c
        assert len(results) > 0
        _assert_valid_evidence(results[0])

    def test_unknown_code_returns_empty(self):
        from tools.medlineplus_tool import explain_condition_snomed
        results = explain_condition_snomed("000000000")
        assert isinstance(results, list)  # may be empty, must not raise


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------

class TestEvidenceModel:
    def test_minimal_evidence(self):
        e = Evidence(source="test", title="T", text="content", citation="cite")
        assert e.score is None
        assert e.metadata == {}

    def test_full_evidence(self):
        e = Evidence(
            source="fda_label", title="Metformin warnings",
            text="...", score=0.85,
            metadata={"drug_name": "metformin"},
            citation="FDA Label — Metformin",
        )
        assert e.score == 0.85
        assert e.metadata["drug_name"] == "metformin"

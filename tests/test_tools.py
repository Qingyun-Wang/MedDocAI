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


# The SQLite tool reads data/meddocai.db (gitignored). Skip those tests when it is
# absent (e.g. in CI) so the suite stays green without the local data.
_DB_PATH = "data/meddocai.db"
_needs_db = pytest.mark.skipif(
    not os.path.isfile(_DB_PATH),
    reason="requires data/meddocai.db (gitignored); skipped in CI",
)


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

@_needs_db
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

    def test_explain_drug_by_name(self):
        """Name-based Connect lookup (.v.dn fallback) reaches the drug monograph
        with NO code — proving the path the keyword search cannot take."""
        from tools.medlineplus_tool import explain_drug_by_name
        results = explain_drug_by_name("metformin")
        assert len(results) > 0
        for e in results:
            _assert_valid_evidence(e)
            assert e.source == "medlineplus_connect"
        titles = " ".join(e.title.lower() for e in results)
        assert "metformin" in titles or "diabetes" in titles

    def test_unknown_code_returns_empty(self):
        from tools.medlineplus_tool import explain_condition_snomed
        results = explain_condition_snomed("000000000")
        assert isinstance(results, list)  # may be empty, must not raise

    def test_search_health_topics_live(self):
        """Live Web Service keyword search over health topics."""
        from tools.medlineplus_tool import search_health_topics_live
        results = search_health_topics_live("high blood pressure", limit=3)
        assert len(results) > 0
        for e in results:
            _assert_valid_evidence(e)
            assert e.source == "medlineplus_web"
        titles = " ".join(e.title.lower() for e in results)
        assert "blood pressure" in titles or "hypertension" in titles

    def test_explain_procedure_snomed(self):
        """Connect procedure lookup by SNOMED code (Colonoscopy)."""
        from tools.medlineplus_tool import explain_procedure_snomed
        results = explain_procedure_snomed("73761001", "Colonoscopy")
        assert isinstance(results, list)  # Connect may lack a page; must not raise
        for e in results:
            _assert_valid_evidence(e)
            assert e.source == "medlineplus_connect"


# ---------------------------------------------------------------------------
# MedlinePlus batch runner (no network — uses fake jobs)
# ---------------------------------------------------------------------------

class TestExplainBatch:
    def test_empty_jobs(self):
        from tools.medlineplus_tool import explain_batch
        assert explain_batch([]) == []

    def test_flattens_results(self):
        from tools.medlineplus_tool import explain_batch
        e1 = Evidence(source="medlineplus_connect", title="A", text="x", citation="c")
        e2 = Evidence(source="medlineplus_connect", title="B", text="y", citation="c")
        jobs = [lambda: [e1], lambda: [e2], lambda: []]
        out = explain_batch(jobs)
        assert {e.title for e in out} == {"A", "B"}

    def test_failed_job_is_skipped(self):
        """One raising job must not sink the batch."""
        from tools.medlineplus_tool import explain_batch
        good = Evidence(source="medlineplus_connect", title="ok", text="x", citation="c")

        def boom():
            raise RuntimeError("connect failed")

        out = explain_batch([boom, lambda: [good]])
        assert len(out) == 1 and out[0].title == "ok"

    def test_no_drug_name_returns_empty(self):
        from tools.medlineplus_tool import explain_drug_by_name
        assert explain_drug_by_name("") == []
        assert explain_drug_by_name("   ") == []


# ---------------------------------------------------------------------------
# Live health-topics keyword search — XML parsing (no network)
# ---------------------------------------------------------------------------

_SAMPLE_WSEARCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult>
  <term>high blood pressure</term>
  <count>2</count>
  <list num="2" start="0" per="2">
    <document rank="0" url="https://medlineplus.gov/highbloodpressure.html">
      <content name="healthTopic">
        <health-topic title="High Blood Pressure" url="https://medlineplus.gov/highbloodpressure.html" meta-desc="HBP meta" language="English">
          <also-called>HBP</also-called>
          <also-called>Hypertension</also-called>
          <full-summary>&lt;p&gt;Blood pressure is the &lt;span class="qt0"&gt;force&lt;/span&gt; of your blood pushing against artery walls.&lt;/p&gt;</full-summary>
        </health-topic>
      </content>
    </document>
    <document rank="1" url="https://medlineplus.gov/heartdiseases.html">
      <content name="healthTopic">
        <health-topic title="Heart Diseases" url="https://medlineplus.gov/heartdiseases.html" meta-desc="heart meta" language="English">
          <full-summary>&lt;p&gt;Your heart is a muscle.&lt;/p&gt;</full-summary>
        </health-topic>
      </content>
    </document>
  </list>
</nlmSearchResult>"""


class TestWsearchParse:
    def test_parses_documents_to_evidence(self):
        from tools.medlineplus_tool import _parse_wsearch
        out = _parse_wsearch(_SAMPLE_WSEARCH_XML, limit=5)
        assert len(out) == 2
        e = out[0]
        _assert_valid_evidence(e)
        assert e.source == "medlineplus_web"
        assert e.title == "High Blood Pressure"
        # HTML/span highlight tags stripped from the summary
        assert "Blood pressure is the force of your blood" in e.text
        assert "<" not in e.text
        assert e.metadata["url"] == "https://medlineplus.gov/highbloodpressure.html"
        assert "Hypertension" in e.metadata["also_called"]
        assert "live search" in e.citation

    def test_limit_respected(self):
        from tools.medlineplus_tool import _parse_wsearch
        assert len(_parse_wsearch(_SAMPLE_WSEARCH_XML, limit=1)) == 1

    def test_zero_results(self):
        from tools.medlineplus_tool import _parse_wsearch
        xml = "<nlmSearchResult><count>0</count></nlmSearchResult>"
        assert _parse_wsearch(xml, limit=5) == []

    def test_malformed_xml_returns_empty(self):
        from tools.medlineplus_tool import _parse_wsearch
        assert _parse_wsearch("this is not xml <<<", limit=5) == []

    def test_empty_query_no_network(self):
        from tools.medlineplus_tool import search_health_topics_live
        assert search_health_topics_live("") == []
        assert search_health_topics_live("   ") == []


# ---------------------------------------------------------------------------
# Router → tool wiring (no network — guards the Fix 1 intent map edits)
# ---------------------------------------------------------------------------

class TestIntentToolWiring:
    def test_medication_info_includes_name_lookup(self):
        from agents.router import INTENT_TOOL_MAP
        assert "explain_drug_by_name" in INTENT_TOOL_MAP["medication_info"]

    def test_condition_education_includes_lab_explainer(self):
        from agents.router import INTENT_TOOL_MAP
        assert "explain_lab_loinc" in INTENT_TOOL_MAP["condition_education"]

    def test_condition_education_includes_procedure_explainer(self):
        from agents.router import INTENT_TOOL_MAP
        assert "explain_procedure_snomed" in INTENT_TOOL_MAP["condition_education"]

    def test_retrieval_procedure_dispatch_without_patient(self):
        from agents.retrieval import _call_tool
        from agents.state import new_state
        st = new_state("what procedures have I had", None, "anonymous")
        assert _call_tool("explain_procedure_snomed", st) == []

    def test_retrieval_dispatches_new_tools_without_patient(self):
        """The new tools must no-op (return []) when no patient context / drug."""
        from agents.retrieval import _call_tool
        from agents.state import new_state
        st = new_state("what is a lab test", None, "anonymous")
        st["drug_name"] = None
        assert _call_tool("explain_lab_loinc", st) == []
        assert _call_tool("explain_drug_by_name", st) == []

    def test_live_search_is_retry_only(self):
        """Live health-topics search is in the retry map, NOT the first-pass map."""
        from agents.router import INTENT_TOOL_MAP, INTENT_RETRY_EXTRA_TOOLS
        assert "search_medlineplus_live" not in INTENT_TOOL_MAP["condition_education"]
        assert "search_medlineplus_live" in INTENT_RETRY_EXTRA_TOOLS["condition_education"]
        assert "search_medlineplus_live" in INTENT_RETRY_EXTRA_TOOLS["general"]


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

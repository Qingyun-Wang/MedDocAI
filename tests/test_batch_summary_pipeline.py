"""
Unit tests for ingestion/batch_summary_pipeline.py

All external calls (MedlinePlus, openFDA, Claude) are mocked so
tests run without API keys or network access.

Run with:  python -m pytest tests/test_batch_summary_pipeline.py -v
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.batch_summary_pipeline import (
    EnrichedCondition,
    EnrichedLab,
    EnrichedMedication,
    EnrichedPatient,
    _build_user_prompt,
    _medlineplus_connect,
    _openfda_label,
    enrich_patient,
    generate_summary,
)
from ingestion.fhir_parser import parse_patient_file
from ingestion.sqlite_loader import MedDocDB

FHIR_DIR = "data/synthea/output_1/fhir"
SAMPLE_FILE = os.path.join(FHIR_DIR, "Abbott509_Aaron203_44.json")
RICH_FILE   = os.path.join(FHIR_DIR, "Abbott509_Evan454_49.json")   # F,58yo,13 conditions,6 abnormal labs

# Although the external calls are mocked, these tests still parse real Synthea FHIR
# files under data/ (gitignored). Skip the whole module when that data is absent
# (e.g. in CI) so the suite stays green without it.
pytestmark = pytest.mark.skipif(
    not os.path.isdir(FHIR_DIR),
    reason="requires local data/ (gitignored); skipped in CI",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_medlineplus_response(title: str, summary: str) -> dict:
    """Build a minimal mock MedlinePlus Connect JSON response."""
    return {
        "feed": {
            "entry": [{
                "title": {"_value": title},
                "summary": {"_value": f"<p>{summary}</p>"},
            }]
        }
    }


def _make_openfda_response(boxed: str = "", contra: str = "", warnings: str = "") -> dict:
    """Build a minimal mock openFDA label JSON response."""
    result = {}
    if boxed:
        result["boxed_warning"] = [boxed]
    if contra:
        result["contraindications"] = [contra]
    if warnings:
        result["warnings_and_cautions"] = [warnings]
    return {"results": [result]}


def _run(coro):
    """Run a coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# _medlineplus_connect
# ---------------------------------------------------------------------------

class TestMedlineplusConnect:

    def setup_method(self):
        import ingestion.batch_summary_pipeline as bp
        bp._SEM_MEDLINE = asyncio.Semaphore(5)

    def test_returns_title_and_plain_text(self):
        mock_response = _make_medlineplus_response(
            "High Blood Pressure",
            "High blood pressure is when the force of blood against artery walls is too high."
        )
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        title, summary = _run(_medlineplus_connect(
            mock_session, "38341003", "2.16.840.1.113883.6.96"
        ))
        assert title == "High Blood Pressure"
        assert "blood pressure" in summary.lower()
        assert "<p>" not in summary  # HTML stripped

    def test_returns_empty_on_404(self):
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        title, summary = _run(_medlineplus_connect(
            mock_session, "99999999", "2.16.840.1.113883.6.96"
        ))
        assert title == ""
        assert summary == ""

    def test_returns_empty_on_empty_feed(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"feed": {"entry": []}})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        title, summary = _run(_medlineplus_connect(
            mock_session, "38341003", "2.16.840.1.113883.6.96"
        ))
        assert title == ""
        assert summary == ""

    def test_summary_truncated_at_400_chars(self):
        long_summary = "A" * 600
        mock_response = _make_medlineplus_response("Title", long_summary)
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        _, summary = _run(_medlineplus_connect(
            mock_session, "38341003", "2.16.840.1.113883.6.96"
        ))
        assert len(summary) <= 400


# ---------------------------------------------------------------------------
# _openfda_label
# ---------------------------------------------------------------------------

class TestOpenFDALabel:

    def setup_method(self):
        import ingestion.batch_summary_pipeline as bp
        bp._SEM_FDA = asyncio.Semaphore(8)

    def test_returns_key_sections(self):
        mock_data = _make_openfda_response(
            boxed="WARNING: LACTIC ACIDOSIS",
            contra="Contraindicated in eGFR < 30",
            warnings="Monitor renal function",
        )
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        result = _run(_openfda_label(mock_session, "Metformin 500 MG"))
        assert "LACTIC ACIDOSIS" in result["boxed_warning"]
        assert "eGFR" in result["contraindications"]
        assert "renal" in result["warnings"].lower()

    def test_returns_empty_on_no_results(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"results": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        result = _run(_openfda_label(mock_session, "UnknownDrug 999 MG"))
        assert result == {"boxed_warning": "", "contraindications": "", "warnings": ""}

    def test_handles_network_error_gracefully(self):
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("connection refused"))

        result = _run(_openfda_label(mock_session, "Metformin"))
        assert result["boxed_warning"] == ""
        assert result["contraindications"] == ""


# ---------------------------------------------------------------------------
# enrich_patient (integration of all enrichment)
# ---------------------------------------------------------------------------

class TestEnrichPatient:

    def setup_method(self):
        import ingestion.batch_summary_pipeline as bp
        bp._SEM_MEDLINE = asyncio.Semaphore(5)
        bp._SEM_FDA = asyncio.Semaphore(8)

    def _make_mock_session(self):
        """Session whose calls return sensible mock data.

        aiohttp's session.get() returns a context manager directly (not a coroutine),
        so fake_get must be a plain function returning the mock response object.
        """
        ml_response = _make_medlineplus_response("Test Condition", "Plain English explanation.")
        fda_response = _make_openfda_response(contra="Test contraindication")

        def fake_get(url, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.status = 200
            if "medlineplus" in url:
                mock_resp.json = AsyncMock(return_value=ml_response)
            else:
                mock_resp.json = AsyncMock(return_value=fda_response)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            return mock_resp   # context manager, not a coroutine

        session = MagicMock()
        session.get = fake_get
        return session

    def test_enrich_returns_enriched_patient(self, tmp_path):
        patient = parse_patient_file(RICH_FILE)
        db = MedDocDB(str(tmp_path / "test.db"))
        db.create_tables()
        db.load_nadac("data/cms_medicaid/nadac_2026.csv")

        session = self._make_mock_session()
        result = _run(enrich_patient(patient, session, db))

        assert isinstance(result, EnrichedPatient)
        assert result.patient_id == patient.patient_id
        assert result.name == patient.name

    def test_enrich_conditions_populated(self, tmp_path):
        patient = parse_patient_file(RICH_FILE)
        db = MedDocDB(str(tmp_path / "test.db"))
        db.create_tables()
        db.load_nadac("data/cms_medicaid/nadac_2026.csv")

        session = self._make_mock_session()
        result = _run(enrich_patient(patient, session, db))

        assert len(result.conditions) == len(patient.conditions)
        for ec in result.conditions:
            assert ec.snomed_code != ""
            assert ec.display != ""
            # mock returns non-empty results
            assert ec.medlineplus_title == "Test Condition"

    def test_enrich_only_active_medications(self, tmp_path):
        patient = parse_patient_file(RICH_FILE)
        active_count = sum(1 for m in patient.medications if m.status == "active")
        db = MedDocDB(str(tmp_path / "test.db"))
        db.create_tables()
        db.load_nadac("data/cms_medicaid/nadac_2026.csv")

        session = self._make_mock_session()
        result = _run(enrich_patient(patient, session, db))

        assert len(result.medications) == active_count

    def test_enrich_patient_no_conditions(self, tmp_path):
        """Patient with zero conditions should not raise."""
        # Abbott509_Aleen583_8.json (age 17) is a known no-condition patient
        files = sorted(f for f in os.listdir(FHIR_DIR) if f.endswith(".json"))
        no_condition_patient = None
        for fname in files:
            p = parse_patient_file(os.path.join(FHIR_DIR, fname))
            if p and not p.conditions:
                no_condition_patient = p
                break
        if not no_condition_patient:
            pytest.skip("No patient without conditions found in first 100 files")

        db = MedDocDB(str(tmp_path / "test.db"))
        db.create_tables()
        db.load_nadac("data/cms_medicaid/nadac_2026.csv")

        session = self._make_mock_session()
        result = _run(enrich_patient(no_condition_patient, session, db))
        assert result.conditions == []


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:

    def _make_enriched_patient(self) -> EnrichedPatient:
        return EnrichedPatient(
            patient_id="test-123",
            name="John Test",
            age=67,
            gender="male",
            conditions=[
                EnrichedCondition("38341003", "Hypertension",
                                  "High Blood Pressure",
                                  "High blood pressure explanation.")
            ],
            medications=[
                EnrichedMedication(
                    "860975", "Metformin 500 MG", "active",
                    medlineplus_title="Metformin",
                    medlineplus_summary="Used to treat type 2 diabetes.",
                    fda_boxed_warning="WARNING: LACTIC ACIDOSIS",
                    fda_contraindications="Contraindicated in eGFR < 30",
                    fda_warnings="Monitor renal function",
                    nadac_price="$0.0300/EA",
                )
            ],
            labs=[
                EnrichedLab("4548-4", "HbA1c", 8.2, "%", True,
                            "A1C Test", "Measures blood sugar over 3 months.")
            ],
        )

    def test_prompt_contains_patient_name(self):
        ep = self._make_enriched_patient()
        prompt = _build_user_prompt(ep)
        assert "John Test" in prompt

    def test_prompt_contains_condition(self):
        ep = self._make_enriched_patient()
        prompt = _build_user_prompt(ep)
        assert "Hypertension" in prompt

    def test_prompt_contains_fda_warning(self):
        ep = self._make_enriched_patient()
        prompt = _build_user_prompt(ep)
        assert "LACTIC ACIDOSIS" in prompt

    def test_prompt_contains_nadac_price(self):
        ep = self._make_enriched_patient()
        prompt = _build_user_prompt(ep)
        assert "$0.0300/EA" in prompt

    def test_prompt_contains_abnormal_lab(self):
        ep = self._make_enriched_patient()
        prompt = _build_user_prompt(ep)
        assert "HbA1c" in prompt
        assert "ABNORMAL" in prompt

    def test_prompt_contains_summary_instructions(self):
        ep = self._make_enriched_patient()
        prompt = _build_user_prompt(ep)
        assert "Patient Overview" in prompt
        assert "Current Medications" in prompt
        assert "Care Considerations" in prompt


# ---------------------------------------------------------------------------
# generate_summary (mocked Claude)
# ---------------------------------------------------------------------------

class TestGenerateSummary:

    def setup_method(self):
        import ingestion.batch_summary_pipeline as bp
        bp._SEM_CLAUDE = asyncio.Semaphore(3)

    def _make_ep(self) -> EnrichedPatient:
        return EnrichedPatient("id-1", "Jane Doe", 55, "female")

    def test_returns_string_from_claude(self):
        mock_claude = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="## Patient Overview\nTest summary.")]
        mock_claude.messages.create = AsyncMock(return_value=mock_response)

        result = _run(generate_summary(self._make_ep(), mock_claude))
        assert "Patient Overview" in result
        assert "Test summary" in result

    def test_returns_error_string_on_exception(self):
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(side_effect=Exception("API error"))

        result = _run(generate_summary(self._make_ep(), mock_claude))
        assert "failed" in result.lower() or "error" in result.lower()

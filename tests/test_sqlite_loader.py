"""
Unit tests for ingestion/sqlite_loader.py

Run with:  python -m pytest tests/test_sqlite_loader.py -v
"""

import os
import pytest
from ingestion.sqlite_loader import MedDocDB
from ingestion.fhir_parser import parse_patient_file

FHIR_DIR = "data/synthea/output_1/fhir"
NADAC_CSV = "data/cms_medicaid/nadac_2026.csv"
ELIGIBILITY_CSV = "data/cms_medicaid/eligibility_levels.csv"
SAMPLE_FHIR = os.path.join(FHIR_DIR, "Abbott509_Aaron203_44.json")


@pytest.fixture
def db(tmp_path):
    """Create a fresh in-memory-like DB in a temp directory for each test."""
    db_path = str(tmp_path / "test.db")
    d = MedDocDB(db_path)
    d.create_tables()
    return d


@pytest.fixture
def loaded_db(db):
    """DB with NADAC and eligibility loaded."""
    db.load_nadac(NADAC_CSV)
    db.load_eligibility(ELIGIBILITY_CSV)
    return db


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def test_create_tables_idempotent(db):
    """Calling create_tables twice should not raise."""
    db.create_tables()  # second call
    db.create_tables()  # third call — still fine


# ---------------------------------------------------------------------------
# NADAC
# ---------------------------------------------------------------------------

def test_load_nadac_returns_count(db):
    count = db.load_nadac(NADAC_CSV)
    assert count == 31516

def test_load_nadac_deduplicates(db):
    """Each NDC should appear exactly once (most recent date kept)."""
    db.load_nadac(NADAC_CSV)
    with db._conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM nadac_pricing").fetchone()[0]
        unique = conn.execute(
            "SELECT COUNT(DISTINCT ndc) FROM nadac_pricing"
        ).fetchone()[0]
    assert total == unique == 31516

def test_load_nadac_idempotent(db):
    """Loading twice should not duplicate rows."""
    db.load_nadac(NADAC_CSV)
    db.load_nadac(NADAC_CSV)
    with db._conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM nadac_pricing").fetchone()[0]
    assert count == 31516

def test_get_drug_price_known_ndc(loaded_db):
    """A known NDC should return a pricing record."""
    # Use an NDC we know is in the dataset from smoke test
    results = loaded_db.search_drug_price_by_name("metformin", limit=1)
    assert results, "Expected at least one metformin result"
    ndc = results[0]["ndc"]
    row = loaded_db.get_drug_price(ndc)
    assert row is not None
    assert row["ndc"] == ndc
    assert isinstance(row["price_per_unit"], float)
    assert row["price_per_unit"] > 0

def test_get_drug_price_unknown_ndc(loaded_db):
    result = loaded_db.get_drug_price("00000000000")
    assert result is None

def test_search_drug_price_by_name(loaded_db):
    results = loaded_db.search_drug_price_by_name("metformin", limit=5)
    assert len(results) > 0
    assert all("metformin" in r["drug_name"].lower() for r in results)

def test_search_drug_price_no_match(loaded_db):
    results = loaded_db.search_drug_price_by_name("zzznodrug999")
    assert results == []

def test_nadac_dates_are_iso_format(loaded_db):
    """Dates should be stored as YYYY-MM-DD, not MM/DD/YYYY."""
    with loaded_db._conn() as conn:
        row = conn.execute(
            "SELECT effective_date FROM nadac_pricing LIMIT 1"
        ).fetchone()
    date_str = row[0]
    # ISO format: length 10, dashes at positions 4 and 7
    assert len(date_str) == 10
    assert date_str[4] == "-" and date_str[7] == "-"


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def test_load_eligibility_returns_count(db):
    count = db.load_eligibility(ELIGIBILITY_CSV)
    assert count == 51

def test_load_eligibility_idempotent(db):
    db.load_eligibility(ELIGIBILITY_CSV)
    db.load_eligibility(ELIGIBILITY_CSV)
    with db._conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM eligibility").fetchone()[0]
    assert count == 51

def test_get_state_eligibility_known_state(loaded_db):
    result = loaded_db.get_state_eligibility("Texas")
    assert result is not None
    assert result["state"] == "Texas"
    assert result["expansion_adults"] == "No"
    assert "%" in result["medicaid_0_1"]

def test_get_state_eligibility_case_insensitive(loaded_db):
    result_lower = loaded_db.get_state_eligibility("texas")
    result_upper = loaded_db.get_state_eligibility("TEXAS")
    result_mixed = loaded_db.get_state_eligibility("Texas")
    assert result_lower is not None
    assert result_lower["state"] == result_upper["state"] == result_mixed["state"]

def test_get_state_eligibility_expansion_state(loaded_db):
    result = loaded_db.get_state_eligibility("California")
    assert result is not None
    assert result["expansion_adults"] not in ("No", "", None)

def test_get_state_eligibility_unknown_state(loaded_db):
    result = loaded_db.get_state_eligibility("Narnia")
    assert result is None

def test_list_states_returns_all(loaded_db):
    states = loaded_db.list_states()
    assert len(states) == 51
    assert "Texas" in states
    assert "California" in states
    assert "District of Columbia" in states


# ---------------------------------------------------------------------------
# Patient summaries
# ---------------------------------------------------------------------------

def test_save_and_retrieve_patient(db):
    patient = parse_patient_file(SAMPLE_FHIR)
    assert patient is not None
    db.save_patient(patient)
    result = db.get_patient(patient.patient_id)
    assert result is not None
    assert result["patient_id"] == patient.patient_id
    assert result["name"] == patient.name
    assert result["age"] == patient.age
    assert result["summary_md"] is None  # not generated yet

def test_save_patient_with_summary(db):
    patient = parse_patient_file(SAMPLE_FHIR)
    db.save_patient(patient, summary_md="## Summary\nThis is a test summary.")
    result = db.get_patient(patient.patient_id)
    assert result["summary_md"] == "## Summary\nThis is a test summary."
    assert result["generated_at"] is not None

def test_save_patient_idempotent(db):
    patient = parse_patient_file(SAMPLE_FHIR)
    db.save_patient(patient)
    db.save_patient(patient)   # second save — should not raise or duplicate
    assert db.count_patients() == 1

def test_patient_json_fields_deserialized(db):
    patient = parse_patient_file(SAMPLE_FHIR)
    db.save_patient(patient)
    result = db.get_patient(patient.patient_id)
    assert isinstance(result["conditions_json"], list)
    assert isinstance(result["medications_json"], list)
    assert isinstance(result["labs_json"], list)
    assert isinstance(result["encounters_json"], list)

def test_get_patient_unknown_id(db):
    result = db.get_patient("nonexistent-uuid")
    assert result is None

def test_list_patients(db):
    patient = parse_patient_file(SAMPLE_FHIR)
    db.save_patient(patient)
    patients = db.list_patients()
    assert len(patients) == 1
    assert patients[0]["patient_id"] == patient.patient_id
    assert "has_summary" in patients[0]
    assert patients[0]["has_summary"] == 0   # no summary yet

def test_count_patients(db):
    assert db.count_patients() == 0
    patient = parse_patient_file(SAMPLE_FHIR)
    db.save_patient(patient)
    assert db.count_patients() == 1

def test_count_patients_with_summary(db):
    patient = parse_patient_file(SAMPLE_FHIR)
    db.save_patient(patient)
    assert db.count_patients_with_summary() == 0
    db.save_patient(patient, summary_md="test")
    assert db.count_patients_with_summary() == 1


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

def test_save_and_retrieve_chat(db):
    db.save_chat_message("sess-1", "user", "What is metformin?")
    db.save_chat_message("sess-1", "assistant", "Metformin is...",
                         sources=[{"title": "FDA Label", "url": "http://example.com"}])
    history = db.get_chat_history("sess-1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert isinstance(history[1]["sources"], list)
    assert history[1]["sources"][0]["title"] == "FDA Label"

def test_chat_history_session_isolation(db):
    db.save_chat_message("sess-A", "user", "Question A")
    db.save_chat_message("sess-B", "user", "Question B")
    history_a = db.get_chat_history("sess-A")
    history_b = db.get_chat_history("sess-B")
    assert len(history_a) == 1
    assert len(history_b) == 1
    assert history_a[0]["content"] == "Question A"
    assert history_b[0]["content"] == "Question B"

def test_chat_history_empty_session(db):
    history = db.get_chat_history("nonexistent-session")
    assert history == []

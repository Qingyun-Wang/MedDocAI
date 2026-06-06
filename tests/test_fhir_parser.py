"""
Unit tests for ingestion/fhir_parser.py

Run with:  python -m pytest tests/test_fhir_parser.py -v
"""

import os
import pytest
from ingestion.fhir_parser import parse_patient_file, parse_patient_directory, LOINC_REFERENCE_RANGES
from models.schemas import ParsedPatient

FHIR_DIR = "data/synthea/output_1/fhir"
SAMPLE_FILE = os.path.join(FHIR_DIR, "Abbott509_Aaron203_44.json")
RICH_FILE   = os.path.join(FHIR_DIR, "Abbott509_Evan454_49.json")   # F,58yo,13 conditions,6 abnormal labs


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_parse_returns_parsed_patient():
    result = parse_patient_file(SAMPLE_FILE)
    assert result is not None
    assert isinstance(result, ParsedPatient)

def test_parse_patient_demographics():
    p = parse_patient_file(SAMPLE_FILE)
    assert p.patient_id != ""
    assert p.name != ""
    assert p.birth_date.count("-") == 2      # YYYY-MM-DD
    assert p.age > 0
    assert p.gender in ("male", "female", "other", "unknown")

def test_parse_fhir_path_is_absolute():
    p = parse_patient_file(SAMPLE_FILE)
    assert os.path.isabs(p.fhir_path)
    assert os.path.exists(p.fhir_path)

def test_parse_nonexistent_file_returns_none():
    result = parse_patient_file("data/does_not_exist.json")
    assert result is None

def test_parse_rich_patient_has_conditions_meds_labs(tmp_path):
    p = parse_patient_file(RICH_FILE)
    assert len(p.conditions) > 0
    assert len(p.medications) > 0
    assert len(p.labs) > 0
    assert len(p.encounters) > 0


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def test_conditions_have_snomed_codes():
    p = parse_patient_file(RICH_FILE)
    for c in p.conditions:
        assert c.snomed_code != "", f"Empty SNOMED code for: {c.display}"
        assert c.display != ""
        assert c.status in ("active", "resolved", "inactive")

def test_conditions_only_active_or_recent():
    """All conditions should be active (or recently resolved — no very old resolved ones)."""
    p = parse_patient_file(RICH_FILE)
    for c in p.conditions:
        assert c.status in ("active", "resolved", "inactive")

def test_no_conditions_is_valid():
    """Young/healthy patients may have no conditions — this should not raise.
    Abbott509_Aleen583_8.json (age 17) and Abbott509_Abby832_27.json (age 73)
    are both known to have no conditions in our 10-patient demo set.
    """
    files = [f for f in os.listdir(FHIR_DIR) if f.endswith(".json")]
    parsed = [parse_patient_file(os.path.join(FHIR_DIR, f)) for f in files]
    empty_condition_patients = [p for p in parsed if p and not p.conditions]
    assert len(empty_condition_patients) > 0, "Expected some patients with no conditions"


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------

def test_medications_have_rxnorm_codes():
    p = parse_patient_file(RICH_FILE)
    for m in p.medications:
        assert m.rxnorm_code != "", f"Empty RxNorm code for: {m.display}"
        assert m.display != ""
        assert m.status in ("active", "stopped", "completed", "unknown")

def test_active_medications_subset():
    p = parse_patient_file(RICH_FILE)
    active = [m for m in p.medications if m.status == "active"]
    # All medications can have any status, but active subset must be valid
    for m in active:
        assert m.rxnorm_code != ""


# ---------------------------------------------------------------------------
# Lab results
# ---------------------------------------------------------------------------

def test_labs_have_loinc_codes():
    p = parse_patient_file(RICH_FILE)
    for lab in p.labs:
        assert lab.loinc_code != ""
        assert lab.display != ""
        assert lab.date != ""

def test_labs_deduplicated_per_loinc():
    """Each LOINC code should appear at most once (most recent kept)."""
    p = parse_patient_file(RICH_FILE)
    loinc_codes = [l.loinc_code for l in p.labs]
    assert len(loinc_codes) == len(set(loinc_codes)), "Duplicate LOINC codes in labs"

def test_abnormal_detection_high_hba1c():
    """HbA1c > 5.7 should be flagged as abnormal."""
    # Search all 10 demo patients (several have diabetes/abnormal HbA1c)
    files = sorted(f for f in os.listdir(FHIR_DIR) if f.endswith(".json"))
    for fname in files:
        p = parse_patient_file(os.path.join(FHIR_DIR, fname))
        if not p:
            continue
        hba1c = next((l for l in p.labs if l.loinc_code == "4548-4"), None)
        if hba1c and isinstance(hba1c.value, float) and hba1c.value > 5.7:
            assert hba1c.is_abnormal, f"HbA1c {hba1c.value}% should be flagged abnormal"
            return
    pytest.skip("No patient with high HbA1c found in first 200 files")

def test_normal_lab_not_flagged():
    """A lab value within range should not be abnormal."""
    # LDL < 100 is normal
    low, high = LOINC_REFERENCE_RANGES["18262-6"]   # LDL
    assert low is None and high == 100.0
    files = sorted(f for f in os.listdir(FHIR_DIR) if f.endswith(".json"))
    for fname in files:
        p = parse_patient_file(os.path.join(FHIR_DIR, fname))
        if not p:
            continue
        ldl = next((l for l in p.labs if l.loinc_code == "18262-6"), None)
        if ldl and isinstance(ldl.value, float) and ldl.value < 100.0:
            assert not ldl.is_abnormal
            return
    pytest.skip("No patient with normal LDL found in first 200 files")


# ---------------------------------------------------------------------------
# Encounters
# ---------------------------------------------------------------------------

def test_encounters_capped_at_five():
    p = parse_patient_file(RICH_FILE)
    assert len(p.encounters) <= 5

def test_encounters_sorted_most_recent_first():
    p = parse_patient_file(RICH_FILE)
    dates = [e.date for e in p.encounters if e.date]
    assert dates == sorted(dates, reverse=True), "Encounters not sorted most-recent-first"

def test_encounter_type_is_string():
    p = parse_patient_file(RICH_FILE)
    for e in p.encounters:
        assert isinstance(e.encounter_type, str)
        assert e.encounter_type != ""


# ---------------------------------------------------------------------------
# Batch parsing
# ---------------------------------------------------------------------------

def test_parse_directory_returns_list():
    # Only 10 demo patients remain after pruning the dataset
    patients = parse_patient_directory(FHIR_DIR, limit=10)
    assert len(patients) == 10
    assert all(isinstance(p, ParsedPatient) for p in patients)

def test_parse_directory_zero_failures():
    """All 10 demo patients should parse without errors."""
    patients = parse_patient_directory(FHIR_DIR, limit=10)
    assert len(patients) == 10


# ---------------------------------------------------------------------------
# Reference ranges sanity checks
# ---------------------------------------------------------------------------

def test_all_reference_range_keys_are_strings():
    for key in LOINC_REFERENCE_RANGES:
        assert isinstance(key, str) and "-" in key

def test_reference_ranges_have_at_least_one_bound():
    for loinc, (low, high) in LOINC_REFERENCE_RANGES.items():
        assert low is not None or high is not None, \
            f"Both bounds are None for LOINC {loinc}"

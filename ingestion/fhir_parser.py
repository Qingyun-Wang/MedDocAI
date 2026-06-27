"""
FHIR Parser — ingestion/fhir_parser.py

Parses Synthea-generated FHIR STU3 Bundle JSON files into ParsedPatient objects.

Key facts about the Synthea STU3 data (confirmed by inspection):
- clinicalStatus on Condition is a plain string ("active"), not a CodeableConcept
- Conditions use SNOMED CT codes only (no ICD-10) — MedlinePlus Connect accepts SNOMED
- Medication + MedicationRequest appear as consecutive pairs in the bundle;
  medicationReference.reference is always empty so we pair by position
- Observation resources do NOT include referenceRange — we use LOINC_REFERENCE_RANGES
- Encounter.class is {"code": "ambulatory"}, not a full CodeableConcept

Usage:
    from ingestion.fhir_parser import parse_patient_file

    patient = parse_patient_file("data/synthea/output_1/fhir/Abbott509_Aaron203_44.json")
    print(patient.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Optional

from models.schemas import (
    Condition, Encounter, LabResult, Medication, ParsedPatient, Procedure,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference ranges for the 15 clinically significant LOINC codes
# present in the Synthea dataset (confirmed by sampling 500 patients).
# Tuple: (low, high) — None means no bound on that side.
#   eGFR:  low=60, high=None  → value < 60 is abnormal
#   HbA1c: low=None, high=5.7 → value > 5.7 is abnormal
# ---------------------------------------------------------------------------
LOINC_REFERENCE_RANGES: dict[str, tuple[Optional[float], Optional[float]]] = {
    "4548-4":   (None,  5.7),    # HbA1c (%)              >5.7 = prediabetes/diabetes
    "2339-0":   (70.0,  100.0),  # Glucose (mg/dL)        fasting normal
    "6299-2":   (7.0,   20.0),   # Urea Nitrogen/BUN (mg/dL)
    "38483-4":  (0.6,   1.3),    # Creatinine (mg/dL)
    "49765-1":  (8.5,   10.5),   # Calcium (mg/dL)
    "2947-0":   (136.0, 145.0),  # Sodium (mEq/L)
    "6298-4":   (3.5,   5.0),    # Potassium (mEq/L)
    "2069-3":   (98.0,  107.0),  # Chloride (mEq/L)
    "20565-8":  (22.0,  29.0),   # Carbon Dioxide (mEq/L)
    "33914-3":  (60.0,  None),   # eGFR (mL/min/1.73m2)  <60 = CKD
    "2093-3":   (None,  200.0),  # Total Cholesterol (mg/dL)
    "2571-8":   (None,  150.0),  # Triglycerides (mg/dL)
    "18262-6":  (None,  100.0),  # LDL Cholesterol (mg/dL)
    "2085-9":   (40.0,  None),   # HDL Cholesterol (mg/dL) <40 = low
    "14959-1":  (None,  30.0),   # Microalbumin/Creatinine Ratio (mg/g)
}

# Conditions that are noise in a care summary — historical/administrative
_EXCLUDE_SNOMED_DISPLAYS = {
    "history of appendectomy",
    "received certificate of high school equivalency",
    "part-time employment",
    "full-time employment",
    "not in labor force",
    "stress (finding)",
    "social isolation (finding)",
}

# Administrative / documentation "procedures" that are noise in a care summary
# (Synthea emits these as Procedure resources — they carry no clinical meaning).
_EXCLUDE_PROCEDURE_DISPLAYS = {
    "documentation of current medications",
    "information gathering (procedure)",
    "medication reconciliation (procedure)",
    "assessment of health and social care needs (procedure)",
    "notification of treatment (procedure)",
    "referral to specialist (procedure)",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_patient_file(fhir_path: str) -> Optional[ParsedPatient]:
    """Parse a single Synthea FHIR STU3 Bundle JSON file.

    Returns a ParsedPatient or None if the file is unreadable / has no Patient resource.
    """
    try:
        with open(fhir_path, encoding="utf-8") as f:
            bundle = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", fhir_path, exc)
        return None

    entries = bundle.get("entry", [])
    if not entries:
        logger.warning("Empty bundle: %s", fhir_path)
        return None

    # Group resources by type and collect ordered (resource, index) tuples
    # — order matters for Medication/MedicationRequest pairing
    grouped: dict[str, list[tuple[dict, int]]] = {}
    for idx, entry in enumerate(entries):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "Unknown")
        grouped.setdefault(rtype, []).append((resource, idx))

    # ── Patient ──────────────────────────────────────────────────────────
    patient_resources = grouped.get("Patient", [])
    if not patient_resources:
        logger.warning("No Patient resource in %s", fhir_path)
        return None

    patient_resource = patient_resources[0][0]
    patient_id, name, birth_date, age, gender = _extract_patient(patient_resource)

    # ── Clinical data ────────────────────────────────────────────────────
    conditions = _extract_conditions(grouped.get("Condition", []))
    medications = _extract_medications(
        grouped.get("Medication", []),
        grouped.get("MedicationRequest", []),
    )
    labs = _extract_labs(grouped.get("Observation", []))
    encounters = _extract_encounters(grouped.get("Encounter", []))
    procedures = _extract_procedures(grouped.get("Procedure", []))

    return ParsedPatient(
        patient_id=patient_id,
        name=name,
        birth_date=birth_date,
        age=age,
        gender=gender,
        conditions=conditions,
        medications=medications,
        labs=labs,
        encounters=encounters,
        procedures=procedures,
        fhir_path=os.path.abspath(fhir_path),
    )


def parse_patient_directory(
    fhir_dir: str,
    limit: Optional[int] = None,
) -> list[ParsedPatient]:
    """Parse all FHIR JSON files in a directory.

    Args:
        fhir_dir: Path to directory containing *.json FHIR bundle files.
        limit: If set, stop after parsing this many files (useful for testing).

    Returns:
        List of successfully parsed ParsedPatient objects.
    """
    patients = []
    files = sorted(f for f in os.listdir(fhir_dir) if f.endswith(".json"))
    if limit:
        files = files[:limit]

    for i, fname in enumerate(files):
        path = os.path.join(fhir_dir, fname)
        patient = parse_patient_file(path)
        if patient:
            patients.append(patient)
        if (i + 1) % 500 == 0:
            logger.info("Parsed %d / %d files", i + 1, len(files))

    logger.info("Parsed %d patients from %s", len(patients), fhir_dir)
    return patients


# ---------------------------------------------------------------------------
# Private extraction helpers
# ---------------------------------------------------------------------------

def _extract_patient(
    resource: dict,
) -> tuple[str, str, str, int, str]:
    """Return (patient_id, name, birth_date, age, gender)."""
    patient_id = resource.get("id", "unknown")

    # Name — prefer "official" use; fall back to first entry
    name = "Unknown Patient"
    for name_obj in resource.get("name", []):
        given = " ".join(name_obj.get("given", []))
        family = name_obj.get("family", "")
        if given or family:
            name = f"{given} {family}".strip()
            if name_obj.get("use") == "official":
                break

    birth_date = resource.get("birthDate", "")
    age = _compute_age(birth_date)
    gender = resource.get("gender", "unknown")

    return patient_id, name, birth_date, age, gender


def _extract_conditions(
    condition_tuples: list[tuple[dict, int]],
) -> list[Condition]:
    """Extract active (and recently resolved) conditions from Condition resources.

    Filters:
    - Keep clinicalStatus == "active"
    - Keep clinicalStatus == "resolved" only if resolved within last 2 years
    - Skip noise entries (administrative/social determinants of health)
    """
    conditions = []
    cutoff_year = date.today().year - 2

    for resource, _ in condition_tuples:
        status = resource.get("clinicalStatus", "")
        if status not in ("active", "resolved", "inactive"):
            continue
        if status == "resolved":
            # Only keep recently resolved conditions for context
            abatement = resource.get("abatementDateTime", "") or resource.get("abatementPeriod", {}).get("end", "")
            if abatement:
                try:
                    resolved_year = int(abatement[:4])
                    if resolved_year < cutoff_year:
                        continue
                except ValueError:
                    pass

        # Extract SNOMED code
        coding = _find_coding(resource.get("code", {}), "snomed")
        if not coding:
            # Fall back to any available coding
            codings = resource.get("code", {}).get("coding", [])
            coding = codings[0] if codings else {}

        snomed_code = coding.get("code", "")
        display = coding.get("display") or resource.get("code", {}).get("text", "")

        if not snomed_code or not display:
            logger.debug("Skipping condition with no code/display")
            continue

        # Skip noise entries
        if display.lower().strip() in _EXCLUDE_SNOMED_DISPLAYS:
            continue

        onset = (
            resource.get("onsetDateTime", "")
            or resource.get("onsetPeriod", {}).get("start", "")
        )
        onset_date = onset[:10] if onset else None  # trim to YYYY-MM-DD

        conditions.append(Condition(
            snomed_code=snomed_code,
            display=display,
            onset_date=onset_date,
            status=status,
        ))

    return conditions


def _extract_medications(
    medication_tuples: list[tuple[dict, int]],
    med_request_tuples: list[tuple[dict, int]],
) -> list[Medication]:
    """Extract medications by pairing Medication + MedicationRequest resources.

    Synthea STU3 quirk: medicationReference.reference is always empty ("").
    Medication and MedicationRequest are emitted as consecutive pairs in the bundle.
    We pair them by sort order (bundle index) — Nth Medication → Nth MedicationRequest.
    """
    # Sort both lists by their position in the bundle
    meds = sorted(medication_tuples, key=lambda x: x[1])
    reqs = sorted(med_request_tuples, key=lambda x: x[1])

    medications = []
    for i, (med_resource, _) in enumerate(meds):
        # Get status from the paired MedicationRequest (if it exists)
        status = "unknown"
        dosage = None
        if i < len(reqs):
            req_resource = reqs[i][0]
            status = req_resource.get("status", "unknown")
            instructions = req_resource.get("dosageInstruction", [{}])
            dosage = instructions[0].get("text") if instructions else None

        # Extract RxNorm code from Medication resource
        coding = _find_coding(med_resource.get("code", {}), "rxnorm")
        if not coding:
            codings = med_resource.get("code", {}).get("coding", [])
            coding = codings[0] if codings else {}

        rxnorm_code = coding.get("code", "")
        display = coding.get("display") or med_resource.get("code", {}).get("text", "")

        if not rxnorm_code or not display:
            logger.debug("Skipping medication with no RxNorm code or display")
            continue

        medications.append(Medication(
            rxnorm_code=rxnorm_code,
            display=display,
            status=status,
            dosage=dosage,
        ))

    return medications


def _extract_labs(
    observation_tuples: list[tuple[dict, int]],
) -> list[LabResult]:
    """Extract lab results from Observation resources.

    Rules:
    - Only keep observations with category = 'laboratory'
    - Keep only the most recent result per LOINC code
    - Look up reference ranges from LOINC_REFERENCE_RANGES
    - Compute is_abnormal based on reference range
    """
    # Collect all lab observations keyed by LOINC code
    # {loinc_code: (resource, date_str)}
    latest_by_loinc: dict[str, tuple[dict, str]] = {}

    for resource, _ in observation_tuples:
        # Filter to laboratory category only
        is_lab = any(
            c.get("code") == "laboratory"
            for cat in resource.get("category", [])
            for c in cat.get("coding", [])
        )
        if not is_lab:
            continue

        coding = _find_coding(resource.get("code", {}), "loinc")
        if not coding:
            continue

        loinc_code = coding.get("code", "")
        if not loinc_code:
            continue

        obs_date = resource.get("effectiveDateTime", "")[:10]

        # Keep only the most recent result per LOINC code
        existing_date = latest_by_loinc.get(loinc_code, (None, ""))[1]
        if obs_date >= existing_date:
            latest_by_loinc[loinc_code] = (resource, obs_date)

    # Convert to LabResult models
    labs = []
    for loinc_code, (resource, obs_date) in latest_by_loinc.items():
        coding = _find_coding(resource.get("code", {}), "loinc") or {}
        display = coding.get("display") or resource.get("code", {}).get("text", loinc_code)

        # Extract value — try numeric first, fall back to string
        value: float | str
        val_quantity = resource.get("valueQuantity", {})
        if "value" in val_quantity:
            try:
                value = float(val_quantity["value"])
            except (TypeError, ValueError):
                value = str(val_quantity["value"])
        elif "valueString" in resource:
            value = resource["valueString"]
        elif "valueCodeableConcept" in resource:
            value = (
                resource["valueCodeableConcept"].get("text")
                or resource["valueCodeableConcept"].get("coding", [{}])[0].get("display", "")
            )
        else:
            logger.debug("Observation %s has no extractable value", loinc_code)
            continue

        unit = val_quantity.get("unit")

        # Look up reference range
        ref = LOINC_REFERENCE_RANGES.get(loinc_code, (None, None))
        ref_low, ref_high = ref

        # Compute is_abnormal (only meaningful for numeric values)
        is_abnormal = False
        if isinstance(value, float):
            if ref_low is not None and value < ref_low:
                is_abnormal = True
            if ref_high is not None and value > ref_high:
                is_abnormal = True

        labs.append(LabResult(
            loinc_code=loinc_code,
            display=display,
            value=value,
            unit=unit,
            reference_low=ref_low,
            reference_high=ref_high,
            is_abnormal=is_abnormal,
            date=obs_date,
        ))

    # Sort by clinical importance: abnormal first, then by LOINC code
    labs.sort(key=lambda x: (not x.is_abnormal, x.loinc_code))
    return labs


def _extract_encounters(
    encounter_tuples: list[tuple[dict, int]],
) -> list[Encounter]:
    """Extract the 5 most recent encounters from Encounter resources."""
    encounters = []

    for resource, _ in encounter_tuples:
        period = resource.get("period", {})
        start = period.get("start", "")
        enc_date = start[:10] if start else ""

        # class is {"code": "ambulatory"} in STU3
        enc_class = resource.get("class", {})
        enc_type = enc_class.get("code", "unknown") if isinstance(enc_class, dict) else "unknown"

        # Reason from type[].text
        reason = None
        for t in resource.get("type", []):
            reason = t.get("text") or next(
                (c.get("display") for c in t.get("coding", [])), None
            )
            if reason:
                break

        encounters.append(Encounter(
            encounter_type=enc_type,
            reason=reason,
            date=enc_date,
        ))

    # Sort descending by date, keep most recent 5
    encounters.sort(key=lambda x: x.date, reverse=True)
    return encounters[:5]


def _extract_procedures(
    procedure_tuples: list[tuple[dict, int]],
) -> list[Procedure]:
    """Extract the most recent procedures from Procedure resources.

    Synthea STU3 codes procedures with SNOMED CT (code.coding[].system = snomed),
    with a date in performedDateTime or performedPeriod.start. Keeps the 8 most
    recent, skipping entries with no code/display.
    """
    procedures = []

    for resource, _ in procedure_tuples:
        coding = _find_coding(resource.get("code", {}), "snomed")
        if not coding:
            codings = resource.get("code", {}).get("coding", [])
            coding = codings[0] if codings else {}

        code = coding.get("code", "")
        display = coding.get("display") or resource.get("code", {}).get("text", "")
        if not code or not display:
            continue

        # Skip administrative/documentation noise
        if display.lower().strip() in _EXCLUDE_PROCEDURE_DISPLAYS:
            continue

        performed = (
            resource.get("performedDateTime", "")
            or resource.get("performedPeriod", {}).get("start", "")
        )
        proc_date = performed[:10] if performed else None

        procedures.append(Procedure(
            code=code,
            code_system="snomed",
            display=display,
            date=proc_date,
            status=resource.get("status", "completed"),
        ))

    # Sort descending by date, keep most recent 8
    procedures.sort(key=lambda p: p.date or "", reverse=True)
    return procedures[:8]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _find_coding(codeable_concept: dict, system_keyword: str) -> Optional[dict]:
    """Return the first coding whose system URL contains system_keyword (case-insensitive)."""
    for coding in codeable_concept.get("coding", []):
        if system_keyword.lower() in coding.get("system", "").lower():
            return coding
    return None


def _compute_age(birth_date_str: str) -> int:
    """Compute age in years from an ISO date string. Returns 0 on parse failure."""
    if not birth_date_str:
        return 0
    try:
        bd = date.fromisoformat(birth_date_str[:10])
        today = date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except ValueError:
        logger.warning("Could not parse birthDate: %s", birth_date_str)
        return 0


# ---------------------------------------------------------------------------
# CLI entry point — for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    fhir_dir = "data/synthea/output_1/fhir"

    # If a specific file is passed, parse just that one
    if len(sys.argv) > 1:
        result = parse_patient_file(sys.argv[1])
        if result:
            print(result.model_dump_json(indent=2))
        else:
            print("Failed to parse file.")
        sys.exit(0)

    # Otherwise run a quick validation on 10 patients
    print("Running validation on 10 sample patients...\n")
    files = sorted(f for f in os.listdir(fhir_dir) if f.endswith(".json"))[:10]

    for fname in files:
        path = os.path.join(fhir_dir, fname)
        patient = parse_patient_file(path)
        if not patient:
            print(f"FAILED: {fname}")
            continue
        abnormal = [l for l in patient.labs if l.is_abnormal]
        active_meds = [m for m in patient.medications if m.status == "active"]
        print(
            f"OK  {patient.name:<35} "
            f"age={patient.age:>3}  "
            f"conditions={len(patient.conditions):>2}  "
            f"active_meds={len(active_meds):>2}  "
            f"labs={len(patient.labs):>2}  "
            f"abnormal={len(abnormal):>2}  "
            f"encounters={len(patient.encounters):>2}"
        )

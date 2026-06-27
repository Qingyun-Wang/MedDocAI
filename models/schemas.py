"""
Pydantic output models for the MedDocAI project.

ParsedPatient is the central output of the FHIR parser and the
input to the batch summary pipeline.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Condition(BaseModel):
    """A clinical condition extracted from a FHIR Condition resource.

    Synthea STU3 uses SNOMED CT codes exclusively.
    MedlinePlus Connect accepts SNOMED CT, so no ICD-10 mapping is needed.
    """
    snomed_code: str = Field(description="SNOMED CT concept code")
    display: str = Field(description="Human-readable condition name")
    onset_date: Optional[str] = Field(None, description="ISO date string, e.g. '1991-04-11'")
    status: str = Field(description="'active' | 'resolved' | 'inactive'")


class Medication(BaseModel):
    """A medication extracted from paired Medication + MedicationRequest resources.

    Synthea STU3 stores RxNorm codes on the Medication resource.
    The paired MedicationRequest carries the status (active/stopped/completed).
    """
    rxnorm_code: str = Field(description="RxNorm concept code")
    display: str = Field(description="Drug name and strength, e.g. 'Metformin 500 MG'")
    status: str = Field(description="'active' | 'stopped' | 'completed'")
    dosage: Optional[str] = Field(None, description="Free-text dosage instruction if present")


class LabResult(BaseModel):
    """A lab observation extracted from a FHIR Observation resource.

    Synthea STU3 does not populate referenceRange on Observation resources.
    Reference ranges are looked up from LOINC_REFERENCE_RANGES in fhir_parser.py.
    Only the most recent result per LOINC code is kept.
    """
    loinc_code: str = Field(description="LOINC code for the test")
    display: str = Field(description="Human-readable test name")
    value: float | str = Field(description="Numeric value or coded string result")
    unit: Optional[str] = Field(None, description="Unit of measure, e.g. 'mg/dL'")
    reference_low: Optional[float] = Field(None, description="Lower bound of normal range")
    reference_high: Optional[float] = Field(None, description="Upper bound of normal range")
    is_abnormal: bool = Field(description="True if value is outside reference range")
    date: str = Field(description="ISO date string of the observation")


class Encounter(BaseModel):
    """A clinical encounter (visit) extracted from a FHIR Encounter resource.

    Only the 5 most recent encounters are kept per patient.
    """
    encounter_type: str = Field(description="Class code: 'ambulatory' | 'inpatient' | 'emergency' | etc.")
    reason: Optional[str] = Field(None, description="Reason for visit if documented")
    date: str = Field(description="ISO date string of encounter start")


class Procedure(BaseModel):
    """A clinical procedure extracted from a FHIR Procedure resource.

    Synthea STU3 codes procedures with SNOMED CT; some real EHRs use CPT.
    Only the most recent few procedures are kept per patient.
    """
    code: str = Field(description="Procedure code (SNOMED CT in Synthea; CPT in some EHRs)")
    code_system: str = Field(description="'snomed' | 'cpt'")
    display: str = Field(description="Human-readable procedure name")
    date: Optional[str] = Field(None, description="ISO date string of the procedure")
    status: str = Field(default="completed", description="'completed' | 'in-progress' | etc.")


class ParsedPatient(BaseModel):
    """Complete structured patient record produced by the FHIR parser.

    This is stored in SQLite (patient_summaries table) and used as
    context for the batch summary pipeline and the RAG agent pipeline.
    """
    patient_id: str = Field(description="FHIR Patient resource id (UUID)")
    name: str = Field(description="Full name: 'Given Family'")
    birth_date: str = Field(description="ISO date string, e.g. '1972-08-29'")
    age: int = Field(description="Age in years computed from birth_date")
    gender: str = Field(description="'male' | 'female' | 'other' | 'unknown'")
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    encounters: list[Encounter] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    fhir_path: str = Field(description="Absolute path to the source FHIR JSON file")


# ---------------------------------------------------------------------------
# Retrieval models — the common currency between tools and agents
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """A single piece of retrieved evidence from any tool.

    All four retrieval tools (Qdrant, openFDA, MedlinePlus, SQLite) return
    a list of Evidence objects. This unified shape lets the Evidence Filter,
    Answer Generator, and Citation Verifier work the same way regardless of
    which tool produced the evidence.
    """
    source: str = Field(
        description="Tool/source name: 'fda_label' | 'openfda_api' | "
                    "'medlineplus' | 'medlineplus_connect' | 'nadac' | 'eligibility'"
    )
    title: str = Field(description="Short human-readable label for this evidence")
    text: str = Field(description="The actual content used for grounding the answer")
    score: Optional[float] = Field(
        None, description="Relevance score (vector similarity) if applicable"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Source-specific fields: drug_name, section_type, url, "
                    "ndc, snomed_code, loinc_code, state, etc.",
    )
    citation: str = Field(
        default="",
        description="Formatted citation string shown to the user, e.g. "
                    "'FDA Label — Metformin, Contraindications'",
    )

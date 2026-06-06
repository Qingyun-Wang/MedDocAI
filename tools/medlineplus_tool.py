"""
MedlinePlus Connect Tool — tools/medlineplus_tool.py

Live lookups against the MedlinePlus Connect API for patient-friendly
explanations keyed by clinical codes:
  - SNOMED CT  -> condition explanation     (from FHIR Condition codes)
  - RxNorm     -> drug explanation          (from FHIR Medication codes)
  - LOINC      -> lab test explanation      (from FHIR Observation codes)
  - ICD-10     -> diagnosis explanation

Returns Evidence objects. Use this when you have a clinical code (e.g. from a
patient's FHIR record) and want the plain-English patient education content.

Usage:
    from tools.medlineplus_tool import (
        explain_condition_snomed, explain_drug_rxnorm, explain_lab_loinc
    )

    evidence = explain_condition_snomed("38341003")   # Hypertension
    evidence = explain_drug_rxnorm("860975")          # Metformin
    evidence = explain_lab_loinc("4548-4")            # HbA1c
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import urllib.parse
import urllib.request

from dotenv import load_dotenv

from models.schemas import Evidence

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — MedlinePlus Connect code system OIDs
# ---------------------------------------------------------------------------

CONNECT_URL = "https://connect.medlineplus.gov/service"

CODE_SYSTEMS = {
    "snomed": "2.16.840.1.113883.6.96",
    "rxnorm": "2.16.840.1.113883.6.88",
    "loinc":  "2.16.840.1.113883.6.1",
    "icd10":  "2.16.840.1.113883.6.90",
}

MAX_SUMMARY_CHARS = 600


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _clean_html(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _connect(code: str, code_system: str, display_name: str = "") -> list[dict]:
    """Call MedlinePlus Connect for a code. Returns list of raw entries."""
    cs = CODE_SYSTEMS.get(code_system)
    if not cs:
        raise ValueError(f"Unknown code system: {code_system}")

    params = {
        "mainSearchCriteria.v.cs": cs,
        "mainSearchCriteria.v.c":  code,
        "knowledgeResponseType":   "application/json",
    }
    if display_name:
        params["mainSearchCriteria.v.dn"] = display_name

    query    = urllib.parse.urlencode(params)
    full_url = f"{CONNECT_URL}?{query}"

    try:
        req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=10) as r:
            data = json.loads(r.read())
    except Exception as e:
        logger.debug("MedlinePlus Connect error for %s/%s: %s", code_system, code, e)
        return []

    return data.get("feed", {}).get("entry", [])


def _entries_to_evidence(
    entries: list[dict],
    code: str,
    code_system: str,
    code_kind: str,
    limit: int = 2,
) -> list[Evidence]:
    """Convert raw Connect entries into Evidence objects."""
    evidence = []
    for entry in entries[:limit]:
        title_obj = entry.get("title", {})
        title = title_obj.get("_value", "") if isinstance(title_obj, dict) else str(title_obj)

        summary_obj = entry.get("summary", {})
        raw_summary = summary_obj.get("_value", "") if isinstance(summary_obj, dict) else ""
        summary = _clean_html(raw_summary)

        if not title and not summary:
            continue

        evidence.append(Evidence(
            source="medlineplus_connect",
            title=title or f"{code_kind} {code}",
            text=summary,
            score=None,
            metadata={
                "code":        code,
                "code_system": code_system,
                "code_kind":   code_kind,
            },
            citation=f"MedlinePlus Connect — {title} ({code_kind} {code})",
        ))
    return evidence


# ---------------------------------------------------------------------------
# Public lookup functions
# ---------------------------------------------------------------------------

def explain_condition_snomed(snomed_code: str, display_name: str = "",
                             limit: int = 2) -> list[Evidence]:
    """Patient-friendly explanation of a condition by SNOMED CT code."""
    entries = _connect(snomed_code, "snomed", display_name)
    return _entries_to_evidence(entries, snomed_code, "snomed", "SNOMED", limit)


def explain_drug_rxnorm(rxnorm_code: str, display_name: str = "",
                        limit: int = 2) -> list[Evidence]:
    """Patient-friendly explanation of a drug by RxNorm code."""
    entries = _connect(rxnorm_code, "rxnorm", display_name)
    return _entries_to_evidence(entries, rxnorm_code, "rxnorm", "RxNorm", limit)


def explain_lab_loinc(loinc_code: str, display_name: str = "",
                      limit: int = 2) -> list[Evidence]:
    """Patient-friendly explanation of a lab test by LOINC code."""
    entries = _connect(loinc_code, "loinc", display_name)
    return _entries_to_evidence(entries, loinc_code, "loinc", "LOINC", limit)


def explain_diagnosis_icd10(icd10_code: str, display_name: str = "",
                            limit: int = 2) -> list[Evidence]:
    """Patient-friendly explanation of a diagnosis by ICD-10 code."""
    entries = _connect(icd10_code, "icd10", display_name)
    return _entries_to_evidence(entries, icd10_code, "icd10", "ICD-10", limit)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Condition (SNOMED 38341003 = Hypertension) ===")
    for e in explain_condition_snomed("38341003"):
        print(f"  {e.title}")
        print(f"     {e.text[:100]}...")

    print("\n=== Drug (RxNorm 860975 = Metformin) ===")
    for e in explain_drug_rxnorm("860975"):
        print(f"  {e.title}")
        print(f"     {e.text[:100]}...")

    print("\n=== Lab (LOINC 4548-4 = HbA1c) ===")
    for e in explain_lab_loinc("4548-4"):
        print(f"  {e.title}")
        print(f"     {e.text[:100]}...")

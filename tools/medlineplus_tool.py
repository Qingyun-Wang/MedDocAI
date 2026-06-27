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
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from models.schemas import Evidence

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — MedlinePlus Connect code system OIDs
# ---------------------------------------------------------------------------

CONNECT_URL = "https://connect.medlineplus.gov/service"

# Live keyword search over health topics (separate API from Connect). Health topics
# ONLY — cannot reach drug monographs or the ADAM encyclopedia (licensed content not
# exposed by this API). Used as a corrective-RAG fallback for condition_education.
WSEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"

CODE_SYSTEMS = {
    "snomed": "2.16.840.1.113883.6.96",
    "rxnorm": "2.16.840.1.113883.6.88",
    "loinc":  "2.16.840.1.113883.6.1",
    "icd10":  "2.16.840.1.113883.6.90",
    "cpt":    "2.16.840.1.113883.6.12",
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


def explain_procedure_snomed(snomed_code: str, display_name: str = "",
                             limit: int = 2) -> list[Evidence]:
    """Patient-friendly explanation of a procedure by SNOMED CT code.

    Synthea codes procedures with SNOMED CT, which shares the Connect OID with
    SNOMED conditions. (For CPT-coded procedures use code_system='cpt'.)
    """
    entries = _connect(snomed_code, "snomed", display_name)
    return _entries_to_evidence(entries, snomed_code, "snomed", "SNOMED", limit)


def explain_drug_by_name(drug_name: str, limit: int = 2) -> list[Evidence]:
    """Patient-friendly drug explanation by NAME — no code required.

    Uses MedlinePlus Connect's display-name matching for medications: the RxNorm
    code system with an EMPTY code plus mainSearchCriteria.v.dn=<name> resolves to
    the MedlinePlus drug monograph. This is the only programmatic path to those
    (ASHP-licensed) drug pages — the Web Service keyword search cannot reach them —
    and it lets drug education work for ANY drug question, with or without a selected
    patient. Pass a clean base name ("metformin"), not a raw FHIR display string.
    """
    if not drug_name or not drug_name.strip():
        return []
    name = drug_name.strip()
    entries = _connect("", "rxnorm", name)
    return _entries_to_evidence(entries, name, "rxnorm", "RxNorm", limit)


# ---------------------------------------------------------------------------
# Concurrent batch runner — collapse N Connect round-trips into ~one
# ---------------------------------------------------------------------------

def explain_batch(jobs: list, max_workers: int = 5) -> list[Evidence]:
    """Run several Connect lookups concurrently and flatten their Evidence.

    Args:
        jobs: a list of zero-arg callables, each returning list[Evidence].
        max_workers: thread-pool size (Connect calls are blocking I/O).

    A failed job is skipped (its evidence omitted) so one bad lookup can't sink the
    batch. Returns [] for an empty job list.
    """
    if not jobs:
        return []
    results: list[Evidence] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for future in [ex.submit(job) for job in jobs]:
            try:
                results.extend(future.result())
            except Exception as exc:
                logger.debug("Connect batch job failed: %s", exc)
    return results


# ---------------------------------------------------------------------------
# Live health-topics keyword search (MedlinePlus Web Service)
# ---------------------------------------------------------------------------

def _parse_wsearch(xml_bytes: bytes | str, limit: int) -> list[Evidence]:
    """Parse a MedlinePlus Web Service (wsearch, rettype=topic) XML response.

    Structure: document / content[@name='healthTopic'] / health-topic[@title,
    @meta-desc] / full-summary. Title/url are ATTRIBUTES (clean); full-summary is
    HTML (matched terms wrapped in <span> highlights) → strip with _clean_html.
    Factored out of the HTTP call so it is unit-testable without network.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.debug("wsearch XML parse error: %s", exc)
        return []

    evidence: list[Evidence] = []
    for doc in root.findall(".//document"):
        ht = doc.find(".//health-topic")
        if ht is None:
            continue
        title = (ht.get("title") or "").strip()
        topic_url = (ht.get("url") or doc.get("url") or "").strip()
        summary = _clean_html(ht.findtext("full-summary", "") or ht.get("meta-desc", ""))
        also = [e.text.strip() for e in ht.findall("also-called")
                if e.text and e.text.strip()]
        if not title and not summary:
            continue
        evidence.append(Evidence(
            source="medlineplus_web",
            title=title or "Health Topic",
            text=summary,
            score=None,
            metadata={"url": topic_url, "also_called": also,
                      "source_type": "health_topic_live"},
            citation=f"MedlinePlus (live search) — {title}",
        ))
        if len(evidence) >= limit:
            break
    return evidence


def search_health_topics_live(query: str, limit: int = 5) -> list[Evidence]:
    """Live keyword search over MedlinePlus HEALTH TOPICS (the Web Service API).

    Used as a corrective-RAG fallback for condition_education when the frozen Qdrant
    `medlineplus` index misses (sparse coverage, or a topic added after our snapshot).

    Scope: HEALTH TOPICS ONLY. This API cannot reach drug monographs or the ADAM
    encyclopedia (licensed; not exposed). Do NOT rely on it for drug questions —
    searching a drug name returns tangential condition topics, not the drug page.
    Returns Evidence tagged source='medlineplus_web' to distinguish live hits from
    the indexed `medlineplus` collection.
    """
    if not query or not query.strip():
        return []
    params = {
        "db": "healthTopics",
        "term": query.strip(),
        "retmax": str(limit),
        "rettype": "topic",
    }
    url = f"{WSEARCH_URL}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=10) as r:
            xml_bytes = r.read()
    except Exception as exc:
        logger.debug("MedlinePlus Web Service error for %r: %s", query, exc)
        return []
    return _parse_wsearch(xml_bytes, limit)


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

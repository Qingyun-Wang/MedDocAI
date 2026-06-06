"""
Batch Summary Pipeline — ingestion/batch_summary_pipeline.py

For each Synthea patient:
  1. Parse FHIR → ParsedPatient (via fhir_parser)
  2. Enrich via async API calls:
       - MedlinePlus Connect (SNOMED) → plain-English condition explanations
       - MedlinePlus Connect (RXCUI)  → plain-English drug explanations
       - openFDA label API            → clinical warnings + contraindications
       - MedlinePlus Connect (LOINC)  → abnormal lab explanations
       - SQLite NADAC                 → drug pricing
  3. Claude generates structured markdown care summary
  4. Store in SQLite patient_summaries.summary_md

Design choices (decided by user):
  - Both openFDA + MedlinePlus for each medication (richer summaries)
  - asyncio with aiohttp for concurrency (performant, shows Python expertise)
  - Resume-safe: patients with existing summaries are skipped

Usage:
    # Set ANTHROPIC_API_KEY in .env first
    python -m ingestion.batch_summary_pipeline --limit 200
    python -m ingestion.batch_summary_pipeline --limit 10 --db data/test.db
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import anthropic
from dotenv import load_dotenv

from ingestion.fhir_parser import parse_patient_directory
from ingestion.sqlite_loader import MedDocDB
from models.schemas import ParsedPatient

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FHIR_DIR = "data/synthea/output_1/fhir"
DB_PATH = "data/meddocai.db"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# MedlinePlus Connect code system URIs
_SNOMED_SYSTEM = "2.16.840.1.113883.6.96"
_RXCUI_SYSTEM = "2.16.840.1.113883.6.88"
_LOINC_SYSTEM = "2.16.840.1.113883.6.1"
_MEDLINEPLUS_BASE = "https://connect.medlineplus.gov/service"
_OPENFDA_BASE = "https://api.fda.gov/drug/label.json"

# Asyncio semaphores — conservative limits to respect API rate limits
# MedlinePlus: 100 req/min → max 5 concurrent gives breathing room
# openFDA: 240 req/min with key → max 8 concurrent
# Claude: depends on tier → max 3 concurrent
_SEM_MEDLINE = None   # initialised in run_batch (must be in event loop)
_SEM_FDA = None
_SEM_CLAUDE = None


# ---------------------------------------------------------------------------
# Enriched data structures (intermediate, not stored in DB directly)
# ---------------------------------------------------------------------------

@dataclass
class EnrichedCondition:
    snomed_code: str
    display: str
    medlineplus_title: str = ""
    medlineplus_summary: str = ""


@dataclass
class EnrichedMedication:
    rxnorm_code: str
    display: str
    status: str
    medlineplus_title: str = ""
    medlineplus_summary: str = ""
    fda_boxed_warning: str = ""
    fda_contraindications: str = ""
    fda_warnings: str = ""
    nadac_price: str = "not found"


@dataclass
class EnrichedLab:
    loinc_code: str
    display: str
    value: float | str
    unit: str
    is_abnormal: bool
    medlineplus_title: str = ""
    medlineplus_summary: str = ""


@dataclass
class EnrichedPatient:
    patient_id: str
    name: str
    age: int
    gender: str
    conditions: list[EnrichedCondition] = field(default_factory=list)
    medications: list[EnrichedMedication] = field(default_factory=list)
    labs: list[EnrichedLab] = field(default_factory=list)
    fhir_path: str = ""


# ---------------------------------------------------------------------------
# Async API helpers
# ---------------------------------------------------------------------------

def _ssl_context() -> ssl.SSLContext:
    """Create SSL context with verification disabled for corporate proxies."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _medlineplus_connect(
    session: aiohttp.ClientSession,
    code: str,
    code_system: str,
    display_name: str = "",
) -> tuple[str, str]:
    """Call MedlinePlus Connect API. Returns (title, plain_text_summary)."""
    global _SEM_MEDLINE
    params = {
        "mainSearchCriteria.v.cs": code_system,
        "mainSearchCriteria.v.c": code,
        "knowledgeResponseType": "application/json",
    }
    if display_name:
        params["mainSearchCriteria.v.dn"] = display_name

    async with _SEM_MEDLINE:
        try:
            async with session.get(
                _MEDLINEPLUS_BASE, params=params, ssl=_ssl_context(), timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return "", ""
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug("MedlinePlus error for %s/%s: %s", code_system, code, exc)
            return "", ""

    entries = data.get("feed", {}).get("entry", [])
    if not entries:
        return "", ""

    # Take the first (best-match) entry
    entry = entries[0]
    title_obj = entry.get("title", {})
    title = title_obj.get("_value", "") if isinstance(title_obj, dict) else str(title_obj)

    summary_obj = entry.get("summary", {})
    raw_summary = summary_obj.get("_value", "") if isinstance(summary_obj, dict) else str(summary_obj)
    # Strip HTML tags
    summary = re.sub(r"<[^>]+>", " ", raw_summary).strip()
    # Collapse whitespace
    summary = re.sub(r"\s+", " ", summary)
    # Truncate to keep prompts manageable
    summary = summary[:400] if len(summary) > 400 else summary

    return title.strip(), summary


async def _openfda_label(
    session: aiohttp.ClientSession,
    drug_display: str,
) -> dict[str, str]:
    """Fetch key safety sections from openFDA drug label API.

    Returns dict with keys: boxed_warning, contraindications, warnings.
    Empty strings if not found.
    """
    global _SEM_FDA

    # Extract base drug name — strip strength/form ("Metformin HCl 500 MG" → "Metformin")
    base_name = re.split(r"\s+\d", drug_display)[0].strip()
    if not base_name:
        return {"boxed_warning": "", "contraindications": "", "warnings": ""}

    api_key = os.getenv("OPENFDA_API_KEY", "")
    url = _OPENFDA_BASE
    params = {
        "search": f'openfda.generic_name:"{base_name}"',
        "limit": "1",
    }
    if api_key:
        params["api_key"] = api_key

    async with _SEM_FDA:
        try:
            async with session.get(
                url, params=params, ssl=_ssl_context(), timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return {"boxed_warning": "", "contraindications": "", "warnings": ""}
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug("openFDA error for %s: %s", drug_display, exc)
            return {"boxed_warning": "", "contraindications": "", "warnings": ""}

    results = data.get("results", [])
    if not results:
        return {"boxed_warning": "", "contraindications": "", "warnings": ""}

    label = results[0]

    def _get_section(key: str) -> str:
        text = label.get(key, [""])[0]
        text = re.sub(r"\s+", " ", text).strip()
        return text[:350] if len(text) > 350 else text

    return {
        "boxed_warning": _get_section("boxed_warning"),
        "contraindications": _get_section("contraindications"),
        "warnings": _get_section("warnings_and_cautions") or _get_section("warnings"),
    }


# ---------------------------------------------------------------------------
# Per-patient enrichment
# ---------------------------------------------------------------------------

async def enrich_patient(
    patient: ParsedPatient,
    session: aiohttp.ClientSession,
    db: MedDocDB,
) -> EnrichedPatient:
    """Enrich a ParsedPatient with API data. Returns EnrichedPatient."""

    enriched = EnrichedPatient(
        patient_id=patient.patient_id,
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        fhir_path=patient.fhir_path,
    )

    # ── Conditions: MedlinePlus SNOMED lookup ─────────────────────────────
    condition_tasks = [
        _medlineplus_connect(session, c.snomed_code, _SNOMED_SYSTEM, c.display)
        for c in patient.conditions
    ]
    condition_results = await asyncio.gather(*condition_tasks, return_exceptions=True)

    for condition, result in zip(patient.conditions, condition_results):
        if isinstance(result, Exception):
            title, summary = "", ""
        else:
            title, summary = result
        enriched.conditions.append(EnrichedCondition(
            snomed_code=condition.snomed_code,
            display=condition.display,
            medlineplus_title=title,
            medlineplus_summary=summary,
        ))

    # ── Medications: MedlinePlus RXCUI + openFDA (active meds only) ───────
    active_meds = [m for m in patient.medications if m.status == "active"]

    # Launch MedlinePlus and openFDA calls concurrently per medication
    medline_tasks = [
        _medlineplus_connect(session, m.rxnorm_code, _RXCUI_SYSTEM, m.display)
        for m in active_meds
    ]
    fda_tasks = [
        _openfda_label(session, m.display)
        for m in active_meds
    ]

    medline_results = await asyncio.gather(*medline_tasks, return_exceptions=True)
    fda_results = await asyncio.gather(*fda_tasks, return_exceptions=True)

    for med, ml_res, fda_res in zip(active_meds, medline_results, fda_results):
        ml_title, ml_summary = ml_res if not isinstance(ml_res, Exception) else ("", "")
        fda_data = fda_res if not isinstance(fda_res, Exception) else {}

        # NADAC pricing — synchronous SQLite (fast, no I/O wait)
        price_row = db.search_drug_price_by_name(med.display.split()[0], limit=1)
        if price_row:
            p = price_row[0]
            nadac_price = f"${p['price_per_unit']:.4f}/{p['pricing_unit']}"
        else:
            nadac_price = "not found in NADAC"

        enriched.medications.append(EnrichedMedication(
            rxnorm_code=med.rxnorm_code,
            display=med.display,
            status=med.status,
            medlineplus_title=ml_title,
            medlineplus_summary=ml_summary,
            fda_boxed_warning=fda_data.get("boxed_warning", ""),
            fda_contraindications=fda_data.get("contraindications", ""),
            fda_warnings=fda_data.get("warnings", ""),
            nadac_price=nadac_price,
        ))

    # ── Abnormal labs: MedlinePlus LOINC lookup ───────────────────────────
    abnormal_labs = [l for l in patient.labs if l.is_abnormal]
    lab_tasks = [
        _medlineplus_connect(session, l.loinc_code, _LOINC_SYSTEM, l.display)
        for l in abnormal_labs
    ]
    lab_results = await asyncio.gather(*lab_tasks, return_exceptions=True)

    # Include all labs in output, only abnormal ones get MedlinePlus info
    abnormal_dict = {}
    for lab, result in zip(abnormal_labs, lab_results):
        title, summary = result if not isinstance(result, Exception) else ("", "")
        abnormal_dict[lab.loinc_code] = (title, summary)

    for lab in patient.labs:
        title, summary = abnormal_dict.get(lab.loinc_code, ("", ""))
        enriched.labs.append(EnrichedLab(
            loinc_code=lab.loinc_code,
            display=lab.display,
            value=lab.value,
            unit=lab.unit or "",
            is_abnormal=lab.is_abnormal,
            medlineplus_title=title,
            medlineplus_summary=summary,
        ))

    return enriched


# ---------------------------------------------------------------------------
# Claude summary generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a healthcare document intelligence assistant that generates
structured care summaries from patient data and trusted medical sources.

Your summaries must:
- Use plain English accessible to both patients and care managers
- Cite sources in brackets: [FDA Label], [MedlinePlus], [NADAC 2026]
- Mark drug-condition interactions with ⚠️
- Never make diagnoses or treatment recommendations
- End with a safety disclaimer"""


def _build_user_prompt(ep: EnrichedPatient) -> str:
    """Build the Claude user prompt from enriched patient data."""
    lines = [
        f"Patient: {ep.name}, {ep.age}yo {ep.gender}",
        "",
    ]

    # Conditions
    if ep.conditions:
        lines.append("ACTIVE CONDITIONS:")
        for c in ep.conditions:
            lines.append(f"  - {c.display} (SNOMED: {c.snomed_code})")
            if c.medlineplus_summary:
                lines.append(f"    MedlinePlus: {c.medlineplus_summary}")
        lines.append("")

    # Active medications
    active_meds = [m for m in ep.medications if m.status == "active"]
    if active_meds:
        lines.append("CURRENT MEDICATIONS:")
        for m in active_meds:
            lines.append(f"  - {m.display} (RxNorm: {m.rxnorm_code})")
            if m.medlineplus_summary:
                lines.append(f"    MedlinePlus: {m.medlineplus_summary}")
            if m.fda_boxed_warning:
                lines.append(f"    FDA BOXED WARNING: {m.fda_boxed_warning}")
            if m.fda_contraindications:
                lines.append(f"    FDA Contraindications: {m.fda_contraindications}")
            if m.fda_warnings:
                lines.append(f"    FDA Warnings: {m.fda_warnings}")
            lines.append(f"    NADAC Price: {m.nadac_price}")
        lines.append("")

    # Lab results
    if ep.labs:
        lines.append("LAB RESULTS:")
        for l in ep.labs:
            flag = " ⚠️ ABNORMAL" if l.is_abnormal else ""
            lines.append(f"  - {l.display}: {l.value} {l.unit}{flag}")
            if l.is_abnormal and l.medlineplus_summary:
                lines.append(f"    MedlinePlus: {l.medlineplus_summary}")
        lines.append("")

    lines.append("""Generate a structured markdown care summary with these sections:

## Patient Overview
(2-3 sentences summarising the patient's key health situation)

## Active Conditions
(bullet list — condition name + brief plain-English explanation + source)

## Current Medications
(for each active med: what it treats, key warnings relevant to this patient, Medicaid cost)
(use ⚠️ for any drug-condition interaction or safety concern)

## Lab Flags
(only include abnormal labs — value, what it means, why it matters for this patient)
(omit this section entirely if no abnormal labs)

## Care Considerations
(2-3 suggested follow-up items based on the above data — NOT medical advice)

---
⚕️ This summary is generated from public data sources for informational purposes only.
It is not medical advice. Always consult a qualified healthcare provider.""")

    return "\n".join(lines)


async def generate_summary(
    ep: EnrichedPatient,
    claude_client: anthropic.AsyncAnthropic,
) -> str:
    """Call Claude to generate a markdown care summary from enriched patient data."""
    global _SEM_CLAUDE

    async with _SEM_CLAUDE:
        try:
            response = await claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1500,
                system=_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": _build_user_prompt(ep)}
                ],
            )
            return response.content[0].text
        except Exception as exc:
            logger.error("Claude error for patient %s: %s", ep.patient_id, exc)
            return f"_Summary generation failed: {exc}_"


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    fhir_dir: str = FHIR_DIR,
    db_path: str = DB_PATH,
    limit: int = 200,
    skip_existing: bool = True,
) -> dict:
    """Parse, enrich, and summarise up to `limit` patients.

    Args:
        fhir_dir:      Directory of Synthea FHIR JSON files.
        db_path:       SQLite database path.
        limit:         Max number of patients to process.
        skip_existing: If True, skip patients who already have a summary.

    Returns:
        Stats dict: {processed, skipped, failed, duration_s}
    """
    global _SEM_MEDLINE, _SEM_FDA, _SEM_CLAUDE

    # ── Validate API key ──────────────────────────────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
        )

    # ── Initialise semaphores inside event loop ───────────────────────────
    _SEM_MEDLINE = asyncio.Semaphore(5)
    _SEM_FDA = asyncio.Semaphore(8)
    _SEM_CLAUDE = asyncio.Semaphore(3)

    # ── Setup DB and Claude client ────────────────────────────────────────
    db = MedDocDB(db_path)
    db.create_tables()
    db.load_nadac("data/cms_medicaid/nadac_2026.csv")
    db.load_eligibility("data/cms_medicaid/eligibility_levels.csv")

    claude_client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Parse patients ────────────────────────────────────────────────────
    logger.info("Parsing up to %d patients from %s ...", limit, fhir_dir)
    patients = parse_patient_directory(fhir_dir, limit=limit)
    logger.info("Parsed %d patients", len(patients))

    # ── Filter out already-processed if resume mode ───────────────────────
    if skip_existing:
        existing = {p["patient_id"] for p in db.list_patients(limit=10000)
                    if p["has_summary"]}
        to_process = [p for p in patients if p.patient_id not in existing]
        skipped_count = len(patients) - len(to_process)
        logger.info(
            "Skipping %d already-summarised patients. Processing %d.",
            skipped_count, len(to_process),
        )
    else:
        to_process = patients
        skipped_count = 0

    if not to_process:
        logger.info("Nothing to do.")
        return {"processed": 0, "skipped": skipped_count, "failed": 0, "duration_s": 0}

    # ── Cost estimate ────────────────────────────────────────────────────
    # ~800 input tokens + ~600 output tokens per patient at current Claude pricing
    est_cost = len(to_process) * (800 * 3 + 600 * 15) / 1_000_000
    print(f"\nBatch plan: {len(to_process)} patients | est. cost ≈ ${est_cost:.2f}")
    print("Starting in 3 seconds... (Ctrl-C to cancel)\n")
    await asyncio.sleep(3)

    # ── Process patients ──────────────────────────────────────────────────
    start = time.monotonic()
    processed = 0
    failed = 0

    # Save parsed (pre-summary) patients to DB first so list_patients() works
    for p in to_process:
        db.save_patient(p)

    connector = aiohttp.TCPConnector(ssl=_ssl_context())
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, patient in enumerate(to_process, 1):
            try:
                enriched = await enrich_patient(patient, session, db)
                summary_md = await generate_summary(enriched, claude_client)
                db.save_patient(patient, summary_md=summary_md)
                processed += 1
                elapsed = time.monotonic() - start
                rate = processed / elapsed * 60
                logger.info(
                    "[%d/%d] %-30s | %.0f patients/min",
                    i, len(to_process), patient.name[:30], rate,
                )
            except KeyboardInterrupt:
                logger.warning("Interrupted by user at patient %d/%d", i, len(to_process))
                break
            except Exception as exc:
                logger.error("Failed patient %s: %s", patient.patient_id, exc)
                failed += 1

    duration = time.monotonic() - start
    stats = {
        "processed": processed,
        "skipped": skipped_count,
        "failed": failed,
        "duration_s": round(duration, 1),
    }
    logger.info(
        "Done. processed=%d skipped=%d failed=%d duration=%.0fs",
        processed, skipped_count, failed, duration,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run MedDocAI batch summary pipeline")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max patients to process (default: 200)")
    parser.add_argument("--fhir-dir", default=FHIR_DIR,
                        help="Path to Synthea FHIR directory")
    parser.add_argument("--db", default=DB_PATH,
                        help="SQLite database path")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-process patients who already have a summary")
    args = parser.parse_args()

    stats = asyncio.run(run_batch(
        fhir_dir=args.fhir_dir,
        db_path=args.db,
        limit=args.limit,
        skip_existing=not args.no_skip,
    ))
    print(f"\nResults: {stats}")

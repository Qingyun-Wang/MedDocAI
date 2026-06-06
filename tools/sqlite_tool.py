"""
SQLite Retrieval Tool — tools/sqlite_tool.py

Structured-data lookups that return Evidence objects:
  - Drug pricing (NADAC 2026)      -- "how much does metformin cost?"
  - Medicaid eligibility by state  -- "what is the income limit in Texas?"

Wraps the MedDocDB class from ingestion/sqlite_loader.py and adapts its
dict results into the common Evidence shape.

Usage:
    from tools.sqlite_tool import lookup_drug_price, lookup_state_eligibility

    evidence = lookup_drug_price("metformin")
    evidence = lookup_state_eligibility("Texas")
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from ingestion.sqlite_loader import MedDocDB
from models.schemas import Evidence

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            override=True)
logger = logging.getLogger(__name__)

DB_PATH = "data/meddocai.db"

# Lazy singleton
_db: MedDocDB | None = None


def _get_db() -> MedDocDB:
    global _db
    if _db is None:
        _db = MedDocDB(DB_PATH)
    return _db


# ---------------------------------------------------------------------------
# Drug pricing
# ---------------------------------------------------------------------------

def lookup_drug_price(drug_name: str, limit: int = 5) -> list[Evidence]:
    """Look up Medicaid drug pricing (NADAC) by drug name.

    Returns one Evidence object per matching NDC product (up to `limit`).
    """
    db = _get_db()
    rows = db.search_drug_price_by_name(drug_name, limit=limit)

    evidence = []
    for row in rows:
        price = row.get("price_per_unit")
        unit  = row.get("pricing_unit", "")
        name  = row.get("drug_name", drug_name)
        ndc   = row.get("ndc", "")
        otc   = "OTC" if row.get("otc") == "Y" else "Prescription"
        eff   = row.get("effective_date", "")

        price_str = f"${price:.4f} per {unit}" if price is not None else "price unavailable"
        text = (f"{name} (NDC {ndc}, {otc}) has a Medicaid NADAC price of "
                f"{price_str}, effective {eff}.")

        evidence.append(Evidence(
            source="nadac",
            title=f"{name} — {price_str}",
            text=text,
            score=None,
            metadata={
                "drug_name":      name,
                "ndc":            ndc,
                "price_per_unit": price,
                "pricing_unit":   unit,
                "otc":            row.get("otc", ""),
                "effective_date": eff,
            },
            citation=f"NADAC 2026 — {name} (NDC {ndc})",
        ))
    return evidence


# ---------------------------------------------------------------------------
# State eligibility
# ---------------------------------------------------------------------------

def lookup_state_eligibility(state: str) -> list[Evidence]:
    """Look up Medicaid income eligibility for a state.

    Returns a single Evidence object (or empty list if state not found).
    """
    db = _get_db()
    row = db.get_state_eligibility(state)
    if not row:
        return []

    expansion = row.get("expansion_adults", "") or "No"
    expanded  = expansion not in ("No", "")

    text = (
        f"Medicaid income eligibility in {row['state']} (as % of Federal Poverty Level):\n"
        f"- Children ages 0-1: {row.get('medicaid_0_1', 'N/A')}\n"
        f"- Children ages 1-5: {row.get('medicaid_1_5', 'N/A')}\n"
        f"- Children ages 6-18: {row.get('medicaid_6_18', 'N/A')}\n"
        f"- Pregnant women: {row.get('pregnant_medicaid', 'N/A')}\n"
        f"- Parents/caretakers: {row.get('parent_caretaker', 'N/A')}\n"
        f"- Adult expansion: {expansion if expanded else 'Not adopted'}\n"
        f"- Separate CHIP: {row.get('chip_separate', 'N/A')}"
    )

    return [Evidence(
        source="eligibility",
        title=f"Medicaid Eligibility — {row['state']}",
        text=text,
        score=None,
        metadata={
            "state":            row["state"],
            "medicaid_0_1":     row.get("medicaid_0_1", ""),
            "pregnant_medicaid": row.get("pregnant_medicaid", ""),
            "expansion_adults": expansion,
            "expanded":         expanded,
        },
        citation=f"CMS Medicaid Eligibility Levels — {row['state']}",
    )]


def list_available_states() -> list[str]:
    """Return all states with eligibility data (for query validation)."""
    return _get_db().list_states()


# ---------------------------------------------------------------------------
# Pre-computed patient summary
# ---------------------------------------------------------------------------

def get_patient_summary(patient_id: str) -> list[Evidence]:
    """Fetch the pre-computed care summary for a patient (from the batch pipeline).

    Returns a single Evidence wrapping the stored markdown summary, or an empty
    list if the patient has no summary yet. This connects the offline batch layer
    (Step 1.3) to the chat pipeline so 'summarise this patient' serves the rich
    pre-computed summary instead of a generic vector search.
    """
    db = _get_db()
    row = db.get_patient(patient_id)
    if not row or not row.get("summary_md"):
        return []

    name = row.get("name", "patient")
    return [Evidence(
        source="patient_summary",
        title=f"Care Summary — {name}",
        text=row["summary_md"],
        score=None,
        metadata={
            "patient_id":   patient_id,
            "name":         name,
            "age":          row.get("age"),
            "gender":       row.get("gender"),
            "generated_at": row.get("generated_at", ""),
        },
        citation="Pre-computed care summary (FDA labels + MedlinePlus + NADAC)",
    )]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Drug price: metformin ===")
    for e in lookup_drug_price("metformin", limit=3):
        print(f"  {e.title}")
        print(f"     citation: {e.citation}")

    print("\n=== Eligibility: Texas ===")
    for e in lookup_state_eligibility("Texas"):
        print(f"  {e.title}")
        print(e.text)

    print("\n=== Eligibility: unknown state ===")
    result = lookup_state_eligibility("Narnia")
    print(f"  Returned {len(result)} results (expected 0)")

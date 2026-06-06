"""
openFDA Live API Tool — tools/openfda_tool.py

Real-time lookups against the openFDA API (always current, no local index):
  - Drug label by name      -- key safety sections for a specific drug
  - Drug recalls            -- current enforcement actions
  - Drug shortages          -- current supply shortages

Returns Evidence objects. Use this for:
  - Single known-drug clinical lookups (more current than the vector index)
  - Recall checks (must be real-time)
  - Shortage checks (must be real-time)

Usage:
    from tools.openfda_tool import (
        fetch_drug_label, check_drug_recalls, check_drug_shortages
    )

    evidence = fetch_drug_label("metformin")
    evidence = check_drug_recalls("semaglutide")
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
# Config
# ---------------------------------------------------------------------------

LABEL_URL       = "https://api.fda.gov/drug/label.json"
ENFORCEMENT_URL = "https://api.fda.gov/drug/enforcement.json"
SHORTAGE_URL    = "https://api.fda.gov/drug/shortages.json"

# Key clinical sections to extract from a label, in priority order
LABEL_SECTIONS = [
    "boxed_warning",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "indications_and_usage",
    "dosage_and_administration",
    "adverse_reactions",
]

# Consumer-friendly OTC sections (different from prescription labels)
OTC_SECTIONS = [
    "purpose",
    "indications_and_usage",
    "warnings",
    "ask_doctor",
    "when_using",
    "stop_use",
]

MAX_SECTION_CHARS = 600   # keep evidence snippets readable


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _api_get(url: str, params: dict) -> dict | None:
    """GET an openFDA endpoint with params. Returns parsed JSON or None."""
    api_key = os.getenv("OPENFDA_API_KEY", "")
    if api_key:
        params["api_key"] = api_key
    query = urllib.parse.urlencode(params, safe='":+')
    full_url = f"{url}?{query}"
    try:
        req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=12) as r:
            return json.loads(r.read())
    except Exception as e:
        logger.debug("openFDA API error: %s", e)
        return None


def _clean(text: str, limit: int = MAX_SECTION_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


# ---------------------------------------------------------------------------
# Drug label lookup
# ---------------------------------------------------------------------------

def fetch_drug_label(drug_name: str, prefer_otc: bool = False) -> list[Evidence]:
    """Fetch key clinical sections from the current FDA label for a drug.

    Args:
        drug_name:  Generic or brand drug name.
        prefer_otc: If True, prefer the OTC (consumer-friendly) label and
                    extract ask_doctor / when_using / stop_use sections.

    Returns:
        One Evidence object per non-empty key section (newest label).
    """
    product_type = "HUMAN OTC DRUG" if prefer_otc else "HUMAN PRESCRIPTION DRUG"
    search = (f'openfda.generic_name:"{drug_name}"'
              f'+AND+openfda.product_type:"{product_type}"')

    data = _api_get(LABEL_URL, {"search": search, "limit": "1"})

    # Fall back to any product type if the preferred one isn't found
    if not data or not data.get("results"):
        data = _api_get(LABEL_URL,
                        {"search": f'openfda.generic_name:"{drug_name}"', "limit": "1"})

    if not data or not data.get("results"):
        return []

    label   = data["results"][0]
    openfda = label.get("openfda", {})
    generic = openfda.get("generic_name", [drug_name])[0]
    brand   = openfda.get("brand_name", [""])[0]
    ptype   = openfda.get("product_type", [""])[0]
    is_otc  = ptype == "HUMAN OTC DRUG"

    sections = OTC_SECTIONS if is_otc else LABEL_SECTIONS
    brand_str = f" ({brand})" if brand else ""

    evidence = []
    for section in sections:
        val = label.get(section)
        if not val or not isinstance(val, list) or not val[0].strip():
            continue
        text = _clean(val[0])
        section_label = section.replace("_", " ").title()

        evidence.append(Evidence(
            source="openfda_api",
            title=f"{generic}{brand_str} — {section_label}",
            text=text,
            score=None,
            metadata={
                "drug_name":    generic,
                "brand_name":   brand,
                "section_type": section,
                "product_type": ptype,
                "is_otc":       is_otc,
            },
            citation=f"FDA Label (live) — {generic}, {section_label}",
        ))
    return evidence


# ---------------------------------------------------------------------------
# Drug recalls
# ---------------------------------------------------------------------------

def check_drug_recalls(drug_name: str, limit: int = 5) -> list[Evidence]:
    """Check for current/ongoing FDA recalls for a drug.

    Returns one Evidence per recall, or a single 'no recalls' Evidence if none.
    """
    search = f'openfda.generic_name:"{drug_name}"+AND+status:"Ongoing"'
    data = _api_get(ENFORCEMENT_URL, {"search": search, "limit": str(limit)})

    if not data or not data.get("results"):
        # Also try a broader product_description search
        search2 = f'product_description:"{drug_name}"+AND+status:"Ongoing"'
        data = _api_get(ENFORCEMENT_URL, {"search": search2, "limit": str(limit)})

    if not data or not data.get("results"):
        return [Evidence(
            source="openfda_api",
            title=f"No current recalls — {drug_name}",
            text=f"No ongoing FDA recalls found for {drug_name} as of the latest data.",
            score=None,
            metadata={"drug_name": drug_name, "recall_count": 0},
            citation="FDA Enforcement Reports (live)",
        )]

    evidence = []
    for r in data["results"]:
        desc   = _clean(r.get("product_description", ""), 200)
        reason = _clean(r.get("reason_for_recall", ""), 200)
        cls    = r.get("classification", "")
        status = r.get("status", "")
        firm   = r.get("recalling_firm", "")

        text = (f"RECALL ({cls}, {status}): {desc}. "
                f"Reason: {reason}. Recalling firm: {firm}.")

        evidence.append(Evidence(
            source="openfda_api",
            title=f"Recall [{cls}] — {drug_name}",
            text=text,
            score=None,
            metadata={
                "drug_name":      drug_name,
                "classification": cls,
                "status":         status,
                "reason":         reason,
                "recalling_firm": firm,
            },
            citation=f"FDA Enforcement Report (live) — {cls}",
        ))
    return evidence


# ---------------------------------------------------------------------------
# Drug shortages
# ---------------------------------------------------------------------------

def check_drug_shortages(drug_name: str, limit: int = 5) -> list[Evidence]:
    """Check for current FDA drug shortages by name.

    Returns one Evidence per shortage record, or a 'no shortages' Evidence.
    """
    search = f'generic_name:"{drug_name}"'
    data = _api_get(SHORTAGE_URL, {"search": search, "limit": str(limit)})

    if not data or not data.get("results"):
        return [Evidence(
            source="openfda_api",
            title=f"No current shortages — {drug_name}",
            text=f"No FDA shortage records found for {drug_name} as of the latest data.",
            score=None,
            metadata={"drug_name": drug_name, "shortage_count": 0},
            citation="FDA Drug Shortages (live)",
        )]

    evidence = []
    for r in data["results"]:
        name   = r.get("generic_name", drug_name)
        status = r.get("status", "")
        reason = _clean(r.get("shortage_reason", ""), 150)

        text = f"SHORTAGE: {name} — status: {status}."
        if reason:
            text += f" Reason: {reason}."

        evidence.append(Evidence(
            source="openfda_api",
            title=f"Shortage [{status}] — {name}",
            text=text,
            score=None,
            metadata={"drug_name": name, "status": status, "reason": reason},
            citation="FDA Drug Shortages (live)",
        ))
    return evidence


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Drug label: metformin (prescription) ===")
    for e in fetch_drug_label("metformin")[:3]:
        print(f"  {e.title}")
        print(f"     {e.text[:90]}...")

    print("\n=== Drug label: ibuprofen (prefer OTC) ===")
    for e in fetch_drug_label("ibuprofen", prefer_otc=True)[:3]:
        print(f"  {e.title}")

    print("\n=== Recalls: semaglutide ===")
    for e in check_drug_recalls("semaglutide")[:2]:
        print(f"  {e.title}")
        print(f"     {e.text[:90]}...")

    print("\n=== Shortages: methotrexate ===")
    for e in check_drug_shortages("methotrexate")[:2]:
        print(f"  {e.title}")

"""
Phase 1 Validation Script — scripts/validate_phase1.py

Runs end-to-end checks across every Phase 1 component:
  1. Data files present
  2. FHIR parser (all 10 demo patients)
  3. SQLite queries (NADAC pricing, eligibility)
  4. Qdrant ingestion (embeds if collections empty, requires OPENAI_API_KEY)
  5. Qdrant semantic search (verifies retrieval quality)
  6. Live APIs (openFDA, MedlinePlus Connect)

Prints a PASS / FAIL / SKIP summary at the end.

Usage:
    python scripts/validate_phase1.py
    python scripts/validate_phase1.py --skip-embedding   # skip Qdrant population
"""

from __future__ import annotations

import argparse
import os
import sys
import ssl
import time
import urllib.request
import json
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), override=True)

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during validation

# ---------------------------------------------------------------------------
# Result tracker
# ---------------------------------------------------------------------------

RESULTS: list[tuple[str, str, str]] = []  # (section, check, status)

def check(section: str, name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    RESULTS.append((section, name, status))
    icon = "[PASS]" if passed else "[FAIL]"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  {icon} {name}{detail_str}")
    return passed

def skip(section: str, name: str, reason: str = "") -> None:
    RESULTS.append((section, name, "SKIP"))
    print(f"  [SKIP] {name}  -- {reason}")

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

# ---------------------------------------------------------------------------
# Section 1 — Data files
# ---------------------------------------------------------------------------

def validate_data_files():
    section("1. Data Files")

    # openFDA drug labels
    label_dir = "data/openfda/drug/label"
    zips = [f for f in os.listdir(label_dir) if f.endswith(".json.zip")] if os.path.isdir(label_dir) else []
    check("Data", "openFDA drug label zips (13 parts)", len(zips) == 13, f"{len(zips)} found")

    # Other openFDA
    for path, name in [
        ("data/openfda/drug/ndc/drug-ndc-0001-of-0001.json.zip", "openFDA NDC"),
        ("data/openfda/drug/enforcement/drug-enforcement-0001-of-0001.json.zip", "openFDA enforcement"),
    ]:
        check("Data", name, os.path.exists(path))

    # MedlinePlus
    ml_dir = "data/medlineplus"
    ml_files = [f for f in os.listdir(ml_dir) if f.endswith(".xml")] if os.path.isdir(ml_dir) else []
    check("Data", "MedlinePlus XML files", len(ml_files) >= 6, f"{len(ml_files)} XML files")

    # CMS
    for fname, name in [
        ("data/cms_medicaid/nadac_2026.csv", "NADAC pricing CSV"),
        ("data/cms_medicaid/eligibility_levels.csv", "Eligibility CSV"),
    ]:
        size_mb = os.path.getsize(fname) / 1024 / 1024 if os.path.exists(fname) else 0
        check("Data", name, os.path.exists(fname), f"{size_mb:.0f} MB")

    # Synthea FHIR
    fhir_dir = "data/synthea/output_1/fhir"
    fhir_files = [f for f in os.listdir(fhir_dir) if f.endswith(".json")] if os.path.isdir(fhir_dir) else []
    check("Data", "Synthea FHIR demo patients", len(fhir_files) == 10, f"{len(fhir_files)} files")

# ---------------------------------------------------------------------------
# Section 2 — FHIR parser
# ---------------------------------------------------------------------------

def validate_fhir_parser():
    section("2. FHIR Parser")
    from ingestion.fhir_parser import parse_patient_directory

    fhir_dir = "data/synthea/output_1/fhir"
    patients = parse_patient_directory(fhir_dir, limit=10)

    check("FHIR", "All 10 patients parse without error", len(patients) == 10, f"{len(patients)} parsed")

    with_conditions = [p for p in patients if p.conditions]
    check("FHIR", "Patients with active conditions", len(with_conditions) >= 5,
          f"{len(with_conditions)}/10")

    with_meds = [p for p in patients if any(m.status == "active" for m in p.medications)]
    check("FHIR", "Patients with active medications", len(with_meds) >= 5,
          f"{len(with_meds)}/10")

    with_abnormal = [p for p in patients if any(l.is_abnormal for l in p.labs)]
    check("FHIR", "Patients with abnormal labs detected", len(with_abnormal) >= 2,
          f"{len(with_abnormal)}/10")

    ages = [p.age for p in patients]
    check("FHIR", "Age range covers paediatric + elderly",
          min(ages) < 18 and max(ages) > 80,
          f"min={min(ages)}, max={max(ages)}")

    return patients

# ---------------------------------------------------------------------------
# Section 3 — SQLite
# ---------------------------------------------------------------------------

def validate_sqlite():
    section("3. SQLite Database")
    from ingestion.sqlite_loader import MedDocDB

    db = MedDocDB("data/meddocai.db")

    nadac_count = db.count_patients()  # just to open db
    import sqlite3
    conn = sqlite3.connect("data/meddocai.db")

    nadac_rows = conn.execute("SELECT COUNT(*) FROM nadac_pricing").fetchone()[0]
    check("SQLite", "nadac_pricing populated", nadac_rows >= 30000, f"{nadac_rows:,} rows")

    elig_rows = conn.execute("SELECT COUNT(*) FROM eligibility").fetchone()[0]
    check("SQLite", "eligibility populated", elig_rows == 51, f"{elig_rows} rows")
    conn.close()

    results = db.search_drug_price_by_name("metformin", limit=3)
    check("SQLite", "Drug price search: metformin", len(results) > 0,
          f"{len(results)} results, first: ${results[0]['price_per_unit']:.4f}/{results[0]['pricing_unit']}" if results else "no results")

    texas = db.get_state_eligibility("Texas")
    check("SQLite", "Eligibility lookup: Texas", texas is not None and texas.get("state") == "Texas",
          f"children 0-1: {texas.get('medicaid_0_1','?')}" if texas else "not found")

    virginia = db.get_state_eligibility("virginia")  # test case-insensitive
    check("SQLite", "Eligibility lookup case-insensitive", virginia is not None,
          "lowercase 'virginia' works" if virginia else "failed")

    return db

# ---------------------------------------------------------------------------
# Section 4 — Qdrant ingestion
# ---------------------------------------------------------------------------

def validate_qdrant_ingestion(skip_embedding: bool = False):
    section("4. Qdrant Ingestion")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key or openai_key == "your_key_here":
        skip("Qdrant", "drug_labels collection", "OPENAI_API_KEY not set")
        skip("Qdrant", "medlineplus collection", "OPENAI_API_KEY not set")
        return False

    from qdrant_client import QdrantClient
    client = QdrantClient(path="data/qdrant_storage")
    existing = {c.name for c in client.get_collections().collections}
    client.close()

    # Drug labels
    if "drug_labels" not in existing:
        if skip_embedding:
            skip("Qdrant", "drug_labels collection", "--skip-embedding flag set")
        else:
            print("  → drug_labels collection empty — running ingestion (~$0.35, ~15 min)...")
            print("     Press Ctrl-C within 5s to cancel")
            time.sleep(5)
            from ingestion.drug_label_chunker import run as run_drug
            stats = run_drug()
            check("Qdrant", "drug_labels ingestion", stats["upserted"] > 40000,
                  f"{stats['upserted']:,} points upserted, cost ~${stats['estimated_cost_usd']:.2f}")
    else:
        client2 = QdrantClient(path="data/qdrant_storage")
        count = client2.count("drug_labels").count
        client2.close()
        if count == 0 and not skip_embedding:
            # Collection exists but is empty (previous run crashed) — re-run ingestion
            print("  → drug_labels collection exists but is empty — re-running ingestion...")
            from ingestion.drug_label_chunker import run as run_drug
            stats = run_drug()
            check("Qdrant", "drug_labels ingestion (retry)", stats["upserted"] > 40000,
                  f"{stats['upserted']:,} points, cost ~${stats['estimated_cost_usd']:.2f}")
        else:
            check("Qdrant", "drug_labels collection populated", count > 40000, f"{count:,} points")

    # MedlinePlus
    client3 = QdrantClient(path="data/qdrant_storage")
    existing2 = {c.name for c in client3.get_collections().collections}
    client3.close()

    if "medlineplus" not in existing2:
        if skip_embedding:
            skip("Qdrant", "medlineplus collection", "--skip-embedding flag set")
        else:
            print("  → medlineplus collection empty — running ingestion (~$0.01, ~1 min)...")
            from ingestion.medlineplus_ingester import run as run_ml
            stats = run_ml()
            check("Qdrant", "medlineplus ingestion", stats["upserted"] == 1102,
                  f"{stats['upserted']:,} points upserted")
    else:
        client4 = QdrantClient(path="data/qdrant_storage")
        count = client4.count("medlineplus").count
        client4.close()
        check("Qdrant", "medlineplus collection populated", count == 1102, f"{count:,} points")

    return True

# ---------------------------------------------------------------------------
# Section 5 — Qdrant semantic search
# ---------------------------------------------------------------------------

def validate_qdrant_search():
    section("5. Qdrant Semantic Search")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key or openai_key == "your_key_here":
        skip("Search", "drug_labels semantic search", "OPENAI_API_KEY not set")
        skip("Search", "medlineplus semantic search", "OPENAI_API_KEY not set")
        return

    from qdrant_client import QdrantClient
    from openai import OpenAI

    client = QdrantClient(path="data/qdrant_storage")
    existing = {c.name for c in client.get_collections().collections}

    if "drug_labels" not in existing or "medlineplus" not in existing:
        skip("Search", "Qdrant collections not populated yet", "run without --skip-embedding first")
        client.close()
        return

    oai = OpenAI(api_key=openai_key)

    def embed(text: str) -> list[float]:
        return oai.embeddings.create(model="text-embedding-3-small", input=[text]).data[0].embedding

    # Drug label search
    q1 = "metformin contraindications kidney disease renal impairment"
    vec1 = embed(q1)
    results1 = client.query_points("drug_labels", query=vec1, limit=3,
                                   with_payload=True).points
    top_drug = results1[0].payload.get("drug_name", "") if results1 else ""
    top_section = results1[0].payload.get("section_type", "") if results1 else ""
    check("Search", "drug_labels: metformin renal query",
          "metformin" in top_drug.lower(),
          f"top: '{top_drug}' / {top_section} (score {results1[0].score:.3f})" if results1 else "no results")

    # Check section type precision
    sections_returned = [r.payload.get("section_type", "") for r in results1]
    relevant_sections = {"contraindications", "warnings_and_cautions", "warnings"}
    check("Search", "drug_labels: section precision",
          any(s in relevant_sections for s in sections_returned),
          f"sections: {sections_returned}")

    # MedlinePlus search
    q2 = "high blood pressure hypertension"
    vec2 = embed(q2)
    results2 = client.query_points("medlineplus", query=vec2, limit=3,
                                   with_payload=True).points
    top_title = results2[0].payload.get("title", "") if results2 else ""
    check("Search", "medlineplus: hypertension query",
          any(word in top_title.lower() for word in ["blood pressure", "hypertension"]),
          f"top: '{top_title}' (score {results2[0].score:.3f})" if results2 else "no results")

    # Cross-collection: condition + drug
    q3 = "diabetes blood sugar medication treatment"
    vec3 = embed(q3)
    results_drug = client.query_points("drug_labels", query=vec3, limit=1,
                                       with_payload=True).points
    results_ml   = client.query_points("medlineplus",  query=vec3, limit=1,
                                       with_payload=True).points
    check("Search", "Cross-collection: diabetes query returns results in both",
          bool(results_drug) and bool(results_ml),
          f"drug: '{results_drug[0].payload.get('drug_name','')}', "
          f"ml: '{results_ml[0].payload.get('title','')}'")

    client.close()

# ---------------------------------------------------------------------------
# Section 6 — Live APIs
# ---------------------------------------------------------------------------

def validate_live_apis():
    section("6. Live APIs")
    ctx = ssl_ctx()

    def api_get(url: str) -> dict | None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
                return json.loads(r.read())
        except Exception as e:
            return None

    # openFDA label API
    data = api_get('https://api.fda.gov/drug/label.json?search=openfda.generic_name:"metformin"&limit=1')
    check("API", "openFDA label API: metformin lookup",
          bool(data and data.get("results")),
          f"{data['meta']['results']['total']:,} records" if data else "failed")

    # openFDA enforcement API
    data2 = api_get('https://api.fda.gov/drug/enforcement.json?search=status:"Ongoing"&limit=1')
    check("API", "openFDA enforcement API: active recalls",
          bool(data2 and data2.get("results")),
          f"{data2['meta']['results']['total']:,} active recalls" if data2 else "failed")

    # MedlinePlus Connect — SNOMED (Hypertension)
    data3 = api_get(
        "https://connect.medlineplus.gov/service"
        "?mainSearchCriteria.v.cs=2.16.840.1.113883.6.96"
        "&mainSearchCriteria.v.c=38341003"
        "&knowledgeResponseType=application/json"
    )
    entries = (data3 or {}).get("feed", {}).get("entry", []) if data3 else []
    check("API", "MedlinePlus Connect: SNOMED hypertension",
          len(entries) > 0,
          f"{len(entries)} entries returned" if entries else "failed")

    # MedlinePlus Connect — RXCUI (Metformin)
    data4 = api_get(
        "https://connect.medlineplus.gov/service"
        "?mainSearchCriteria.v.cs=2.16.840.1.113883.6.88"
        "&mainSearchCriteria.v.c=860975"
        "&knowledgeResponseType=application/json"
    )
    entries4 = (data4 or {}).get("feed", {}).get("entry", []) if data4 else []
    check("API", "MedlinePlus Connect: RXCUI metformin",
          len(entries4) > 0,
          f"{len(entries4)} entries" if entries4 else "failed")

# ---------------------------------------------------------------------------
# Section 7 — Summary
# ---------------------------------------------------------------------------

def print_summary():
    section("SUMMARY")
    passed = sum(1 for _, _, s in RESULTS if s == "PASS")
    failed = sum(1 for _, _, s in RESULTS if s == "FAIL")
    skipped = sum(1 for _, _, s in RESULTS if s == "SKIP")
    total = len(RESULTS)

    print(f"  Total checks:  {total}")
    print(f"  Passed:   {passed:>3}")
    print(f"  Failed:   {failed:>3}")
    print(f"  Skipped:  {skipped:>3}")
    print()

    if failed > 0:
        print("  FAILED CHECKS:")
        for sec, name, status in RESULTS:
            if status == "FAIL":
                print(f"    [FAIL] [{sec}] {name}")
        print()

    if skipped > 0:
        print("  SKIPPED (need API keys or --skip-embedding):")
        for sec, name, status in RESULTS:
            if status == "SKIP":
                print(f"    [SKIP] [{sec}] {name}")
        print()

    phase1_ready = failed == 0
    print(f"  Phase 1 status: {'READY for Phase 2' if phase1_ready else 'NOT READY — fix failures above'}")
    return phase1_ready


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Phase 1 components")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip Qdrant ingestion even if collections are empty")
    args = parser.parse_args()

    print("\nMedDocAI — Phase 1 Validation")
    print("=" * 60)

    validate_data_files()
    validate_fhir_parser()
    validate_sqlite()
    validate_qdrant_ingestion(skip_embedding=args.skip_embedding)
    validate_qdrant_search()
    validate_live_apis()

    ready = print_summary()
    sys.exit(0 if ready else 1)

"""
Phase 1 Retrieval Demo — scripts/demo_retrieval.py

Demonstrates and manually tests every data source built in Phase 1.
Run this to see what the system can actually retrieve before building agents.

Sections:
  1. FHIR Patient Data       -- parse a demo patient, show their clinical profile
  2. SQLite Drug Pricing     -- look up real drug costs from NADAC 2026
  3. SQLite Eligibility      -- look up Medicaid rules by state
  4. Qdrant Drug Labels      -- semantic search over FDA label sections
  5. Qdrant MedlinePlus      -- semantic search over patient education topics
  6. openFDA Live API        -- fetch current drug label + recall check
  7. MedlinePlus Connect API -- get patient-friendly explanations by clinical code
  8. End-to-end patient demo -- full retrieval for one demo patient

Usage:
    python scripts/demo_retrieval.py               # run all sections
    python scripts/demo_retrieval.py --section 4   # run only section 4
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), override=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "=" * 65

def header(n: int, title: str) -> None:
    print(f"\n{SEP}")
    print(f"  Section {n}: {title}")
    print(SEP)

def subheader(title: str) -> None:
    print(f"\n  --- {title} ---")

def ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def api_get(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx(), timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  API error: {e}")
        return None

# ---------------------------------------------------------------------------
# Section 1: FHIR Patient Data
# ---------------------------------------------------------------------------

def demo_fhir():
    header(1, "FHIR Patient Data")
    from ingestion.fhir_parser import parse_patient_file, parse_patient_directory

    fhir_dir = "data/synthea/output_1/fhir"
    patients = parse_patient_directory(fhir_dir)

    print(f"\n  Loaded {len(patients)} demo patients:\n")
    print(f"  {'Name':<35} {'Age':>4} {'Sex':>4}  {'Cond':>5}  {'Meds':>5}  {'Labs':>5}  {'Abn':>4}")
    print(f"  {'-'*65}")
    for p in sorted(patients, key=lambda x: -len(x.conditions)):
        active = sum(1 for m in p.medications if m.status == "active")
        abnormal = sum(1 for l in p.labs if l.is_abnormal)
        print(f"  {p.name:<35} {p.age:>4} {p.gender[0].upper():>4}  "
              f"{len(p.conditions):>5}  {active:>5}  {len(p.labs):>5}  {abnormal:>4}")

    # Show full detail for the richest patient
    richest = max(patients, key=lambda p: len(p.conditions) * 3 +
                  sum(1 for m in p.medications if m.status == "active") * 3 +
                  sum(1 for l in p.labs if l.is_abnormal) * 5)

    subheader(f"Detailed view: {richest.name}, {richest.age}yo {richest.gender}")

    if richest.conditions:
        print("\n  CONDITIONS:")
        for c in richest.conditions[:5]:
            print(f"    [{c.status}] {c.display} (SNOMED {c.snomed_code})")

    active_meds = [m for m in richest.medications if m.status == "active"]
    if active_meds:
        print("\n  ACTIVE MEDICATIONS:")
        for m in active_meds:
            print(f"    {m.display} (RxNorm {m.rxnorm_code})")

    abnormal_labs = [l for l in richest.labs if l.is_abnormal]
    if abnormal_labs:
        print("\n  ABNORMAL LABS:")
        for l in abnormal_labs:
            ref = f"(normal: {l.reference_low}–{l.reference_high})"
            print(f"    {l.display}: {l.value} {l.unit or ''} {ref} [ABNORMAL]")

    return richest


# ---------------------------------------------------------------------------
# Section 2: SQLite Drug Pricing
# ---------------------------------------------------------------------------

def demo_sqlite_pricing():
    header(2, "SQLite Drug Pricing (NADAC 2026)")
    from ingestion.sqlite_loader import MedDocDB

    db = MedDocDB("data/meddocai.db")

    drugs_to_check = [
        "metformin", "lisinopril", "atorvastatin", "insulin",
        "amoxicillin", "clopidogrel", "ibuprofen", "acetaminophen",
    ]

    print(f"\n  {'Drug search':<25} {'Results':>3}  {'Cheapest option':<45} {'Price':>10}")
    print(f"  {'-'*90}")

    for drug in drugs_to_check:
        results = db.search_drug_price_by_name(drug, limit=5)
        if results:
            cheapest = min(results, key=lambda r: r["price_per_unit"] or 999)
            price_str = f"${cheapest['price_per_unit']:.4f}/{cheapest['pricing_unit']}"
            name_trunc = cheapest["drug_name"][:44]
            print(f"  {drug:<25} {len(results):>3}  {name_trunc:<45} {price_str:>10}")
        else:
            print(f"  {drug:<25}   0  not found in NADAC")


# ---------------------------------------------------------------------------
# Section 3: SQLite Eligibility
# ---------------------------------------------------------------------------

def demo_sqlite_eligibility():
    header(3, "SQLite Medicaid Eligibility")
    from ingestion.sqlite_loader import MedDocDB

    db = MedDocDB("data/meddocai.db")

    states_to_check = ["Texas", "California", "Virginia", "Florida", "New York"]

    print(f"\n  {'State':<20} {'Children 0-1':>13} {'Pregnant':>10} {'Adults':>12} {'Expansion':>10}")
    print(f"  {'-'*70}")

    for state in states_to_check:
        e = db.get_state_eligibility(state)
        if e:
            exp = e.get("expansion_adults", "?") or "No"
            expanded = "YES" if exp not in ("No", "") else "No"
            print(f"  {state:<20} {e.get('medicaid_0_1','?'):>13} "
                  f"{e.get('pregnant_medicaid','?'):>10} "
                  f"{exp:>12} "
                  f"{expanded:>10}")

    subheader("States WITHOUT Medicaid expansion")
    no_exp = [s for s in db.list_states()
              if (db.get_state_eligibility(s) or {}).get("expansion_adults", "No") in ("No", "")]
    print(f"  {len(no_exp)} states: {', '.join(sorted(no_exp))}")


# ---------------------------------------------------------------------------
# Section 4: Qdrant Drug Label Semantic Search
# ---------------------------------------------------------------------------

def demo_qdrant_drug_labels():
    header(4, "Qdrant Semantic Search — Drug Labels")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("  SKIPPED: OPENAI_API_KEY not set")
        return

    from openai import OpenAI
    from qdrant_client import QdrantClient

    oai    = OpenAI(api_key=openai_key)
    client = QdrantClient(path="data/qdrant_storage")

    def search(query: str, limit: int = 3):
        vec = oai.embeddings.create(
            model="text-embedding-3-small", input=[query]
        ).data[0].embedding
        return client.query_points(
            "drug_labels", query=vec, limit=limit, with_payload=True
        ).points

    test_queries = [
        ("metformin kidney disease warnings",
         "Expected: metformin contraindications/warnings"),
        ("blood pressure medication side effects",
         "Expected: antihypertensive drug sections"),
        ("antibiotic allergy penicillin reaction",
         "Expected: penicillin adverse reactions or contraindications"),
        ("diabetes insulin dosage administration",
         "Expected: insulin dosage section"),
        ("pain reliever ibuprofen stomach risk",
         "Expected: ibuprofen warnings or contraindications"),
    ]

    for query, note in test_queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  Note:  {note}")
        results = search(query)
        for i, r in enumerate(results, 1):
            drug    = r.payload.get("drug_name", "?")
            section = r.payload.get("section_type", "?")
            score   = r.score
            snippet = r.payload.get("text", "")[:120].replace("\n", " ")
            snippet = snippet.encode("ascii", errors="replace").decode("ascii")
            print(f"    {i}. [{score:.3f}] {drug} / {section}")
            print(f"       \"{snippet}...\"")

    client.close()


# ---------------------------------------------------------------------------
# Section 5: Qdrant MedlinePlus Semantic Search
# ---------------------------------------------------------------------------

def demo_qdrant_medlineplus():
    header(5, "Qdrant Semantic Search — MedlinePlus")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("  SKIPPED: OPENAI_API_KEY not set")
        return

    from openai import OpenAI
    from qdrant_client import QdrantClient

    oai    = OpenAI(api_key=openai_key)
    client = QdrantClient(path="data/qdrant_storage")

    def search(query: str, limit: int = 3):
        vec = oai.embeddings.create(
            model="text-embedding-3-small", input=[query]
        ).data[0].embedding
        return client.query_points(
            "medlineplus", query=vec, limit=limit, with_payload=True
        ).points

    test_queries = [
        ("what is high blood pressure and why is it dangerous",
         "Expected: Hypertension / High Blood Pressure"),
        ("how do kidneys filter blood and what is kidney failure",
         "Expected: Kidney disease / CKD topic"),
        ("what does HbA1c test measure for diabetes",
         "Expected: A1C test topic"),
        ("explain what cholesterol is and LDL HDL difference",
         "Expected: Cholesterol topic or Cholesterol definition"),
        ("how does insulin work for type 2 diabetes",
         "Expected: Diabetes or Insulin topic"),
    ]

    for query, note in test_queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  Note:  {note}")
        results = search(query)
        for i, r in enumerate(results, 1):
            title   = r.payload.get("title", "?")
            groups  = r.payload.get("groups", [])
            score   = r.score
            snippet = r.payload.get("full_summary", "")[:120].replace("\n", " ")
            print(f"    {i}. [{score:.3f}] \"{title}\"  [{', '.join(groups[:2])}]")
            print(f"       \"{snippet}...\"")

    client.close()


# ---------------------------------------------------------------------------
# Section 6: openFDA Live API
# ---------------------------------------------------------------------------

def demo_openfda_api():
    header(6, "openFDA Live API")

    api_key = os.getenv("OPENFDA_API_KEY", "")
    key_param = f"&api_key={api_key}" if api_key else ""

    # Drug label lookup
    subheader("Drug label: metformin (key sections)")
    data = api_get(
        f"https://api.fda.gov/drug/label.json"
        f"?search=openfda.generic_name:\"metformin\""
        f"&limit=1{key_param}"
    )
    if data and data.get("results"):
        r = data["results"][0]
        openfda = r.get("openfda", {})
        print(f"  Drug:         {openfda.get('generic_name', ['?'])[0]}")
        print(f"  Brand:        {openfda.get('brand_name', ['?'])[0]}")
        print(f"  Manufacturer: {openfda.get('manufacturer_name', ['?'])[0]}")
        print(f"  Product type: {openfda.get('product_type', ['?'])[0]}")
        for section in ["boxed_warning", "contraindications", "warnings_and_cautions"]:
            val = r.get(section, [""])[0][:200]
            if val:
                print(f"\n  [{section}]:\n  {val[:180]}...")

    # Current recalls
    subheader("Current drug recalls (top 5)")
    data2 = api_get(
        f"https://api.fda.gov/drug/enforcement.json"
        f"?search=status:\"Ongoing\"&limit=5{key_param}"
    )
    if data2 and data2.get("results"):
        total = data2["meta"]["results"]["total"]
        print(f"  Total ongoing recalls: {total:,}")
        for r in data2["results"]:
            drug  = r.get("product_description", "")[:55]
            cls   = r.get("classification", "?")
            reason = r.get("reason_for_recall", "")[:60]
            print(f"  [{cls}] {drug}")
            print(f"         Reason: {reason}")

    # Drug shortages
    subheader("Current drug shortages (top 5)")
    data3 = api_get(
        f"https://api.fda.gov/drug/shortages.json?limit=5{key_param}"
    )
    if data3 and data3.get("results"):
        total = data3["meta"]["results"]["total"]
        print(f"  Total shortage records: {total:,}")
        for r in data3["results"][:5]:
            print(f"  [{r.get('status','?')}] {r.get('generic_name','?')}")


# ---------------------------------------------------------------------------
# Section 7: MedlinePlus Connect API
# ---------------------------------------------------------------------------

def demo_medlineplus_api():
    header(7, "MedlinePlus Connect API")

    base = "https://connect.medlineplus.gov/service"
    ctx  = ssl_ctx()

    def connect(params: str) -> list[dict]:
        url  = f"{base}?{params}&knowledgeResponseType=application/json"
        data = api_get(url)
        return (data or {}).get("feed", {}).get("entry", []) if data else []

    import re
    def clean(html_text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_text)).strip()[:200]

    # SNOMED conditions
    subheader("Condition lookup via SNOMED CT")
    snomed_tests = [
        ("38341003", "Hypertension"),
        ("73211009", "Diabetes mellitus type 2"),
        ("709044004", "Chronic kidney disease"),
    ]
    for code, display in snomed_tests:
        entries = connect(
            f"mainSearchCriteria.v.cs=2.16.840.1.113883.6.96"
            f"&mainSearchCriteria.v.c={code}"
        )
        if entries:
            title   = entries[0].get("title", {})
            title   = title.get("_value", "") if isinstance(title, dict) else str(title)
            summary = entries[0].get("summary", {})
            summary = clean(summary.get("_value", "") if isinstance(summary, dict) else "")
            print(f"\n  SNOMED {code} ({display}):")
            print(f"  -> Title: {title}")
            print(f"  -> {summary[:150]}...")

    # RXCUI drug lookup
    subheader("Drug lookup via RxNorm RXCUI")
    rxcui_tests = [
        ("860975", "Metformin"),
        ("29046", "Lisinopril"),
        ("83367",  "Atorvastatin"),
    ]
    for rxcui, name in rxcui_tests:
        entries = connect(
            f"mainSearchCriteria.v.cs=2.16.840.1.113883.6.88"
            f"&mainSearchCriteria.v.c={rxcui}"
        )
        if entries:
            title = entries[0].get("title", {})
            title = title.get("_value", "") if isinstance(title, dict) else str(title)
            print(f"  RXCUI {rxcui} ({name}) -> \"{title}\"  ({len(entries)} entries)")

    # LOINC lab lookup
    subheader("Lab test lookup via LOINC")
    loinc_tests = [
        ("4548-4",  "HbA1c"),
        ("33914-3", "eGFR"),
        ("2093-3",  "Total Cholesterol"),
    ]
    for loinc, name in loinc_tests:
        entries = connect(
            f"mainSearchCriteria.v.cs=2.16.840.1.113883.6.1"
            f"&mainSearchCriteria.v.c={loinc}"
        )
        if entries:
            title = entries[0].get("title", {})
            title = title.get("_value", "") if isinstance(title, dict) else str(title)
            print(f"  LOINC {loinc} ({name}) -> \"{title}\"  ({len(entries)} entries)")


# ---------------------------------------------------------------------------
# Section 8: End-to-end patient retrieval demo
# ---------------------------------------------------------------------------

def demo_end_to_end():
    header(8, "End-to-End Patient Retrieval Demo")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    api_key    = os.getenv("OPENFDA_API_KEY", "")
    key_param  = f"&api_key={api_key}" if api_key else ""

    # Load the richest demo patient
    from ingestion.fhir_parser import parse_patient_directory
    from ingestion.sqlite_loader import MedDocDB
    from openai import OpenAI
    from qdrant_client import QdrantClient
    import re

    patients = parse_patient_directory("data/synthea/output_1/fhir")
    patient  = max(patients, key=lambda p:
                   len(p.conditions) + sum(1 for m in p.medications if m.status == "active"))
    db       = MedDocDB("data/meddocai.db")

    print(f"\n  Patient: {patient.name}, {patient.age}yo {patient.gender}")
    print(f"  Source:  {os.path.basename(patient.fhir_path)}")

    active_meds = [m for m in patient.medications if m.status == "active"]
    print(f"\n  {len(patient.conditions)} conditions, {len(active_meds)} active medications")

    if not openai_key:
        print("\n  (Skipping vector search — OPENAI_API_KEY not set)")
        return

    oai    = OpenAI(api_key=openai_key)
    client = QdrantClient(path="data/qdrant_storage")

    def embed(text: str) -> list[float]:
        return oai.embeddings.create(
            model="text-embedding-3-small", input=[text]
        ).data[0].embedding

    def clean_html(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

    # For each active medication:
    # 1. Vector search for drug warnings (patient condition aware)
    # 2. Pricing from SQLite
    # 3. MedlinePlus Connect lookup
    print("\n  ====  MEDICATION ANALYSIS  ====")

    for med in active_meds[:3]:   # limit to 3 for demo
        print(f"\n  Drug: {med.display} (RxNorm {med.rxnorm_code})")

        # Build condition-aware query
        condition_terms = " ".join(c.display for c in patient.conditions[:3])
        query = f"{med.display.split()[0]} warnings {condition_terms}"

        results = client.query_points("drug_labels", query=embed(query),
                                      limit=2, with_payload=True).points
        if results:
            top = results[0]
            print(f"  FDA label [{top.score:.3f}] {top.payload.get('drug_name','')} / "
                  f"{top.payload.get('section_type','')}")
            snippet = top.payload.get("text", "")[:200].replace("\n", " ")
            print(f"    \"{snippet}...\"")

        # NADAC pricing
        base_name = med.display.split()[0]
        prices = db.search_drug_price_by_name(base_name, limit=1)
        if prices:
            p = prices[0]
            print(f"  NADAC cost: ${p['price_per_unit']:.4f}/{p['pricing_unit']}")

    # For each condition: MedlinePlus Connect
    print("\n  ====  CONDITION EDUCATION  ====")

    connect_base = "https://connect.medlineplus.gov/service"
    for cond in patient.conditions[:3]:
        params = (
            f"mainSearchCriteria.v.cs=2.16.840.1.113883.6.96"
            f"&mainSearchCriteria.v.c={cond.snomed_code}"
            f"&knowledgeResponseType=application/json"
        )
        data = api_get(f"{connect_base}?{params}")
        entries = (data or {}).get("feed", {}).get("entry", []) if data else []
        if entries:
            title   = entries[0].get("title", {})
            title   = title.get("_value", "") if isinstance(title, dict) else str(title)
            summary = entries[0].get("summary", {})
            summary = clean_html(summary.get("_value", "") if isinstance(summary, dict) else "")
            print(f"\n  {cond.display}:")
            print(f"  -> MedlinePlus: \"{title}\"")
            print(f"     {summary[:160]}...")

    # For abnormal labs: MedlinePlus Connect LOINC
    abnormal_labs = [l for l in patient.labs if l.is_abnormal]
    if abnormal_labs:
        print("\n  ====  ABNORMAL LAB FLAGS  ====")
        for lab in abnormal_labs[:3]:
            params = (
                f"mainSearchCriteria.v.cs=2.16.840.1.113883.6.1"
                f"&mainSearchCriteria.v.c={lab.loinc_code}"
                f"&knowledgeResponseType=application/json"
            )
            data = api_get(f"{connect_base}?{params}")
            entries = (data or {}).get("feed", {}).get("entry", []) if data else []
            ref = f"(normal: {lab.reference_low}–{lab.reference_high})"
            flag = "[ABNORMAL]"
            print(f"\n  {lab.display}: {lab.value} {lab.unit or ''} {ref} {flag}")
            if entries:
                title = entries[0].get("title", {})
                title = title.get("_value", "") if isinstance(title, dict) else str(title)
                print(f"  -> MedlinePlus: \"{title}\"")

    client.close()
    print(f"\n  This is what the batch pipeline and agent will assemble into")
    print(f"  a structured care summary per patient.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS = {
    1: ("FHIR Patient Data",               demo_fhir),
    2: ("SQLite Drug Pricing",             demo_sqlite_pricing),
    3: ("SQLite Eligibility",              demo_sqlite_eligibility),
    4: ("Qdrant Drug Label Search",        demo_qdrant_drug_labels),
    5: ("Qdrant MedlinePlus Search",       demo_qdrant_medlineplus),
    6: ("openFDA Live API",                demo_openfda_api),
    7: ("MedlinePlus Connect API",         demo_medlineplus_api),
    8: ("End-to-End Patient Demo",         demo_end_to_end),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 retrieval demo")
    parser.add_argument("--section", type=int, choices=SECTIONS.keys(),
                        help="Run only this section number")
    args = parser.parse_args()

    print("\nMedDocAI — Phase 1 Retrieval Demo")
    print("Demonstrates what each data source can actually return")

    sections_to_run = [args.section] if args.section else list(SECTIONS.keys())

    for n in sections_to_run:
        title, fn = SECTIONS[n]
        try:
            fn()
        except Exception as e:
            print(f"\n  [ERROR in section {n}]: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{SEP}")
    print("  Demo complete.")
    print(SEP)

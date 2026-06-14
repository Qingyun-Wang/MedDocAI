"""
Reference Verifier — evaluation/verify_references.py

The eval references are the "answer key" — every RAGAS score depends on them
being accurate. This tool shows each reference SIDE-BY-SIDE with its source of
truth so you can manually confirm it, source by source:

  - policy questions  -> the exact CMS eligibility / NADAC row from SQLite
  - patient questions -> the patient's real conditions/meds/labs from SQLite
  - recall questions  -> the live openFDA enforcement / shortage data
  - drug questions    -> the authoritative FDA label sections (live openFDA),
                         AND the chunks our own pipeline actually retrieved
                         (read from the cached answers_*.json, no rerun needed)

Usage:
    python -m evaluation.verify_references                 # all 30
    python -m evaluation.verify_references --id policy_01
    python -m evaluation.verify_references --category policy
    python -m evaluation.verify_references --id med_anon_02 --show-retrieved
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            override=True)

import logging, warnings
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.join(EVAL_DIR, "eval_dataset.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# Per-question hints: what entity each reference is about (so we can fetch its
# source). Drugs use the FDA label; states use eligibility; pricing uses NADAC.
DRUG_OF = {
    "med_anon_01": "warfarin", "med_anon_02": "metformin", "med_anon_03": "lisinopril",
    "med_anon_04": "ibuprofen", "med_anon_05": "atorvastatin", "med_anon_06": "amoxicillin",
    "med_anon_07": "clopidogrel", "med_anon_08": "insulin", "med_anon_09": "albuterol",
    "med_anon_10": "simvastatin",
    "med_pat_01": "metformin", "med_pat_02": "clopidogrel", "med_pat_03": "albuterol",
    "med_pat_04": "hydrocodone", "med_pat_05": "metformin", "med_pat_06": "nitroglycerin",
    "med_pat_07": None, "med_pat_08": "clopidogrel", "med_pat_09": None,
    "med_pat_10": "amoxicillin",
}
STATE_OF   = {"policy_01": "Texas", "policy_02": "California",
              "policy_03": "Virginia", "policy_04": "New York"}
PRICE_OF   = {"policy_05": "lisinopril"}
RECALL_OF  = {"recall_01": ("enforcement", "semaglutide"),
              "recall_02": ("shortage", "methotrexate"),
              "recall_03": ("enforcement", "alprazolam")}


def _ssl_ctx():
    c = ssl.create_default_context(); c.check_hostname = False; c.verify_mode = ssl.CERT_NONE
    return c

def _api(url, params):
    key = os.getenv("OPENFDA_API_KEY", "")
    if key:
        params["api_key"] = key
    query = urllib.parse.urlencode(params, safe='":+')
    full = f"{url}?{query}"
    try:
        req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=12) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)[:120]}

def _clean(t, n=500):
    return re.sub(r"\s+", " ", t).strip()[:n]


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------

def show_drug_source(drug: str):
    data = _api("https://api.fda.gov/drug/label.json",
                {"search": f'openfda.generic_name:"{drug}"', "limit": "1"})
    if data.get("_error") or not data.get("results"):
        print(f"    (could not fetch FDA label for {drug}: {data.get('_error','no results')})")
        return
    label = data["results"][0]
    for section in ["boxed_warning", "contraindications", "warnings_and_cautions", "warnings"]:
        val = label.get(section)
        if val and val[0].strip():
            print(f"    [FDA {section}]: {_clean(val[0], 350)}")


def show_state_source(state: str):
    from ingestion.sqlite_loader import MedDocDB
    row = MedDocDB("data/meddocai.db").get_state_eligibility(state)
    if not row:
        print(f"    (state {state} not found)")
        return
    keys = ["medicaid_0_1", "medicaid_1_5", "medicaid_6_18", "pregnant_medicaid",
            "parent_caretaker", "expansion_adults", "chip_separate"]
    print(f"    [SQLite eligibility — {row['state']}]")
    for k in keys:
        print(f"       {k}: {row.get(k)}")


def show_price_source(drug: str):
    from ingestion.sqlite_loader import MedDocDB
    rows = MedDocDB("data/meddocai.db").search_drug_price_by_name(drug, limit=5)
    print(f"    [SQLite NADAC — '{drug}' matches]")
    for r in rows:
        print(f"       {r['drug_name']}: ${r['price_per_unit']:.4f}/{r['pricing_unit']}")


def show_recall_source(kind: str, drug: str):
    if kind == "enforcement":
        data = _api("https://api.fda.gov/drug/enforcement.json",
                    {"search": f'openfda.generic_name:"{drug}"+AND+status:"Ongoing"', "limit": "3"})
        if data.get("_error") or not data.get("results"):
            data = _api("https://api.fda.gov/drug/enforcement.json",
                        {"search": f'product_description:"{drug}"+AND+status:"Ongoing"', "limit": "3"})
        print(f"    [openFDA enforcement — {drug}]")
        for r in (data.get("results") or [])[:3]:
            print(f"       [{r.get('classification')}] {_clean(r.get('product_description',''),80)}")
            print(f"          reason: {_clean(r.get('reason_for_recall',''),100)}")
    else:
        data = _api("https://api.fda.gov/drug/shortages.json",
                    {"search": f'generic_name:"{drug}"', "limit": "3"})
        print(f"    [openFDA shortages — {drug}]")
        for r in (data.get("results") or [])[:3]:
            print(f"       [{r.get('status')}] {r.get('generic_name','')}")


def show_patient_source(patient_prefix: str):
    from ingestion.sqlite_loader import MedDocDB
    db = MedDocDB("data/meddocai.db")
    full = None
    for p in db.list_patients():
        if p["name"].startswith(patient_prefix):
            full = db.get_patient(p["patient_id"]); break
    if not full:
        print(f"    (patient {patient_prefix} not found)"); return
    conds = [c["display"] for c in full.get("conditions_json", [])]
    meds  = [m["display"] for m in full.get("medications_json", []) if m.get("status") == "active"]
    abn   = [f"{l['display']}={l['value']}{l.get('unit') or ''}"
             for l in full.get("labs_json", []) if l.get("is_abnormal")]
    print(f"    [Patient record — {full['name']}, {full['age']}{full['gender'][0].upper()}]")
    print(f"       conditions: {'; '.join(conds[:8])}")
    print(f"       active meds: {'; '.join(meds)}")
    print(f"       abnormal labs: {'; '.join(abn[:6])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def latest_answers():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "answers_*.json")))
    if not files:
        return {}
    rows = json.load(open(files[-1], encoding="utf-8"))
    return {r["id"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=str, default=None)
    ap.add_argument("--category", type=str, default=None)
    ap.add_argument("--show-retrieved", action="store_true",
                    help="Also show the chunks our pipeline retrieved (from cache)")
    args = ap.parse_args()

    ds = json.load(open(DATASET, encoding="utf-8"))["questions"]
    if args.id:
        ds = [q for q in ds if q["id"] == args.id]
    if args.category:
        ds = [q for q in ds if q["category"] == args.category]

    answers = latest_answers()

    for q in ds:
        qid = q["id"]
        print("=" * 78)
        print(f"{qid}  [{q['category']}]")
        print(f"Q: {q['question']}")
        print(f"\nREFERENCE (the answer key — verify the claims below appear in the source):")
        print(f"  {q['reference']}")
        print(f"\nSOURCE OF TRUTH:")

        if qid in STATE_OF:
            show_state_source(STATE_OF[qid])
        elif qid in PRICE_OF:
            show_price_source(PRICE_OF[qid])
        elif qid in RECALL_OF:
            show_recall_source(*RECALL_OF[qid])
        elif q.get("patient") and (DRUG_OF.get(qid) is None or q["category"] == "care_mgr"):
            show_patient_source(q["patient"])
        elif DRUG_OF.get(qid):
            show_drug_source(DRUG_OF[qid])
            if q.get("patient"):
                show_patient_source(q["patient"])
        else:
            print("    (no automated source mapping — verify manually)")

        if args.show_retrieved and qid in answers:
            print(f"\n  WHAT OUR PIPELINE RETRIEVED ({len(answers[qid]['contexts'])} chunks):")
            for i, c in enumerate(answers[qid]["contexts"][:4], 1):
                print(f"    ({i}) {_clean(c, 160)}")
        print()


if __name__ == "__main__":
    main()

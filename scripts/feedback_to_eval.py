"""
Feedback -> eval candidates — scripts/feedback_to_eval.py

Closes the improvement loop: turn thumbs-down answers from real usage into
candidate questions for the RAGAS eval set, so the test set grows from observed
failures instead of only from questions I imagined in advance.

WHAT THIS DOES NOT DO — and why
-------------------------------
It does NOT write into evaluation/eval_dataset.json, and it does NOT invent a
`reference` answer.

A thumbs-down means the answer was wrong. Using that answer (or asking the model
to fix it) as the answer key would bake the failure into the thing that grades
future failures — the eval set would drift toward whatever the system already
does. References in this project are hand-written from the SOURCE (an FDA label
section, a CMS row, the patient's record); that discipline is what makes the
scores mean anything.

So this emits a *candidates* file with `reference` left empty, alongside the
context a human needs to write one: the failed answer, the user's note, and the
routing metadata. Reviewing and writing the reference stays a human step; merging
into the eval set is a deliberate, separate act.

Usage:
    python scripts/feedback_to_eval.py --stats
    python scripts/feedback_to_eval.py                      # -> evaluation/feedback_candidates.json
    python scripts/feedback_to_eval.py --rating up          # inspect what worked
    python scripts/feedback_to_eval.py --limit 20 --out /tmp/cand.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.sqlite_loader import MedDocDB

DB_PATH = "data/meddocai.db"
DEFAULT_OUT = os.path.join("evaluation", "feedback_candidates.json")

# Map a pipeline intent onto the eval set's category vocabulary. `general` is a
# deliberate fallback: an unmapped intent should be reviewed, not silently binned.
INTENT_TO_CATEGORY = {
    "medication_info":     "med_patient",     # refined to med_anon below if no patient
    "condition_education": "med_patient",
    "drug_recall":         "recall",
    "policy_eligibility":  "policy",
    "patient_summary":     "care_mgr",
    "direct_answer":       "general",
    "general":             "general",
}


def _patient_prefix(db: MedDocDB, patient_id: str | None) -> str | None:
    """eval_dataset.json identifies patients by name prefix ('Cali421'), not uuid."""
    if not patient_id:
        return None
    row = db.get_patient(patient_id)
    if not row or not row.get("name"):
        return None
    return row["name"].split()[0]


def build_candidates(rows: list[dict], db: MedDocDB) -> list[dict]:
    out = []
    for i, r in enumerate(rows, 1):
        patient = _patient_prefix(db, r.get("patient_id"))
        category = INTENT_TO_CATEGORY.get(r.get("intent") or "", "general")
        if category == "med_patient" and not patient:
            category = "med_anon"

        out.append({
            "id": f"fb_{i:03d}",
            "category": category,
            "question": r.get("question") or "",
            "user_role": r.get("user_role") or "anonymous",
            "patient": patient,
            # Left EMPTY on purpose — a human writes this from the source.
            "reference": "",
            "_review": {
                "why_flagged": r.get("comment") or "(no note given)",
                "rejected_answer": (r.get("answer") or "")[:1500],
                "intent_routed_to": r.get("intent"),
                "passed_internal_review": bool(r.get("review_passed")),
                "n_evidence": r.get("n_evidence"),
                "rated_at": r.get("created_at"),
                "query_id": r.get("query_id"),
            },
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Turn user feedback into eval candidates")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--rating", default="down", choices=["down", "up"],
                    help="Which verdict to export (default: down)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--stats", action="store_true", help="Just print counts and exit")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"No database at {args.db}", file=sys.stderr)
        return 1

    db = MedDocDB(args.db)

    counts = db.feedback_counts()
    total = counts["up"] + counts["down"]
    print(f"Feedback so far: {counts['up']} up, {counts['down']} down"
          + (f"  ({counts['up'] / total:.0%} positive)" if total else ""))
    if args.stats:
        for r in db.get_feedback(rating="down", limit=args.limit):
            note = r.get("comment") or "(no note)"
            print(f"  [down] {(r.get('question') or '')[:60]:<62} {note[:50]}")
        return 0

    rows = db.get_feedback(rating=args.rating, limit=args.limit)
    if not rows:
        print(f"No '{args.rating}' feedback recorded yet — nothing to export.")
        return 0

    candidates = build_candidates(rows, db)
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "source": f"user feedback (rating={args.rating})",
        "note": ("References are intentionally EMPTY. Write each one from the "
                 "underlying source (FDA label section / CMS row / patient record) "
                 "before merging into evaluation/eval_dataset.json."),
        "candidates": candidates,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(candidates)} candidate(s) -> {args.out}")
    print("Next: write a reference for each from its source, then merge the ones "
          "worth keeping into evaluation/eval_dataset.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

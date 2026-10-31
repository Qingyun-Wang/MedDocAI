"""
Build the tiny SQLite fixture the CI eval gate needs — scripts/build_eval_fixture_db.py

The gate currently runs 10 of 44 eval questions. Not by choice: 26 of them need
`data/meddocai.db`, which is gitignored and absent on a runner, so patient
personalisation, Medicaid policy, the care-manager persona and fan-out retrieval are
all COMPLETELY UNGATED — break any of them and the PR still goes green.

The gate does not need the 5.7 MB production database. It needs exactly enough to
answer the questions it runs:

  - only the patients the dataset actually pins (by name prefix)
  - the whole eligibility table (51 rows, tiny)
  - only the NADAC rows for drugs the policy questions ask about, not all 31,516

That lands around 200 KB — small enough to commit as a plain file, so no Git LFS, no
pointer, and no interaction with the Space's `.gitattributes` (the one file deploy.yml
must never touch).

Everything here is derived from the real database, so the fixture cannot drift into
fiction: regenerate it and the patients are byte-identical to production.

Usage:
    python -m scripts.build_eval_fixture_db                    # report only
    python -m scripts.build_eval_fixture_db --apply
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import sys

SRC = os.path.join("data", "meddocai.db")
DEST = os.path.join("evaluation", "fixtures", "meddocai_eval.db")
DATASET = os.path.join("evaluation", "eval_dataset.json")

# Drugs named by the policy/pricing questions. Keeping the NADAC slice question-driven
# rather than guessing is what keeps the fixture at ~200 KB instead of ~4.5 MB.
NADAC_DRUGS = ["lisinopril"]

TABLES_DDL_FROM_SRC = ("patient_summaries", "nadac_pricing", "eligibility")


def pinned_prefixes(dataset: str) -> list[str]:
    d = json.load(io.open(dataset, encoding="utf-8"))
    qs = d if isinstance(d, list) else d["questions"]
    return sorted({q["patient"] for q in qs if q.get("patient")})


def build(src: str, dest: str, prefixes: list[str], apply: bool) -> int:
    s = sqlite3.connect(src)
    s.row_factory = sqlite3.Row

    # Resolve each prefix to exactly one patient. Ambiguity here would hand an eval
    # question the wrong record with a hand-verified reference — the silent failure
    # resolve_patients() now raises on. Refuse to build a fixture that reproduces it.
    chosen, problems = {}, []
    for p in prefixes:
        rows = s.execute("SELECT * FROM patient_summaries WHERE name LIKE ?",
                         (p + "%",)).fetchall()
        if len(rows) != 1:
            problems.append(f"{p}: {len(rows)} matches")
        else:
            chosen[p] = rows[0]
    if problems:
        print("REFUSING to build — ambiguous or missing patient prefixes:", file=sys.stderr)
        for x in problems:
            print("   ", x, file=sys.stderr)
        return 2

    nadac = []
    for drug in NADAC_DRUGS:
        nadac += s.execute(
            "SELECT * FROM nadac_pricing WHERE LOWER(drug_name) LIKE ?",
            (f"%{drug.lower()}%",)).fetchall()
    elig = s.execute("SELECT * FROM eligibility").fetchall()

    print(f"fixture contents:")
    print(f"  patients      : {len(chosen)}  ({', '.join(sorted(chosen))})")
    print(f"  nadac_pricing : {len(nadac)}  (drugs: {', '.join(NADAC_DRUGS)})")
    print(f"  eligibility   : {len(elig)}")

    if not apply:
        print("\nreport only — re-run with --apply to write the fixture.")
        return 0

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        os.remove(dest)
    d = sqlite3.connect(dest)

    # Copy the real schema rather than restating it, so the fixture cannot drift from
    # production DDL.
    for t in TABLES_DDL_FROM_SRC:
        ddl = s.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)
        ).fetchone()
        if ddl and ddl[0]:
            d.execute(ddl[0])

    def insert(table, rows):
        if not rows:
            return
        cols = rows[0].keys()
        d.executemany(
            f"INSERT INTO {table} ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            [tuple(r[c] for c in cols) for r in rows],
        )

    insert("patient_summaries", list(chosen.values()))
    insert("nadac_pricing", nadac)
    insert("eligibility", elig)
    d.commit()
    d.execute("VACUUM")
    d.commit()
    d.close()

    size = os.path.getsize(dest)
    print(f"\nwrote {dest}  ({size/1024:.0f} KB)")
    if size > 1_000_000:
        print("WARNING: over 1 MB — check the NADAC slice before committing.",
              file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the CI eval fixture database")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.src):
        print(f"source database not found: {a.src}", file=sys.stderr)
        return 2
    prefixes = pinned_prefixes(a.dataset)
    print(f"{len(prefixes)} patient prefixes pinned by {a.dataset}")
    return build(a.src, a.dest, prefixes, a.apply)


if __name__ == "__main__":
    sys.exit(main())

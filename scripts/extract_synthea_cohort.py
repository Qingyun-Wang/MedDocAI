"""
Select a demo cohort from the Synthea 1M FHIR archives — scripts/extract_synthea_cohort.py

The project ships a handful of Synthea patients in data/synthea/output_1/fhir/. The
original ten were taken ALPHABETICALLY from the top of the archive, which is why they
all share the surname Abbott509 and why two of them have zero conditions and zero
medications. This script replaces that with a sampled, clinically-screened cohort.

Source: the Synthea 1M FHIR release (12 tar.gz archives, ~22 GB). `output_1` alone
holds 132,842 patient bundles with a full A-Z surname spread, so one archive is more
than enough and one streaming pass is all it costs.

Two cheap filters before any parsing:
  - bundle size >= --min-kb. Size tracks clinical richness closely (the empty patient
    Abby832 is 6 KB; a well-populated one is 76 KB+). The median bundle is 89 KB.
  - a uniform random sample with a PINNED SEED, so the cohort is reproducible and so
    surnames are drawn in proportion to the pool rather than from the letter A.

Then the candidates are parsed and ranked on what actually matters for this app:
a patient with no conditions or no active medications cannot answer a clinical
question, and the fan-out retrieval path is only exercised by patients on several
distinct drugs.

Existing files in the output directory are NEVER overwritten — 17 evaluation
questions are pinned to the original patients by name prefix.

Usage:
    python -m scripts.extract_synthea_cohort                   # report only, copies nothing
    python -m scripts.extract_synthea_cohort --n 90 --apply
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
from collections import Counter

DEFAULT_ARCHIVE = (r"C:\Users\User\Projects\damps\synthea_1m_fhir_3_0_May_24"
                   r"\synthea_1m_fhir_3_0_May_24\output_1_20170524T232103.tar.gz")
MEMBER_PREFIX = "output_1/fhir/"
DEST = os.path.join("data", "synthea", "output_1", "fhir")
SEED = 20260917                      # pinned: the cohort must be reproducible

AGE_BANDS = [(0, 17), (18, 39), (40, 59), (60, 74), (75, 200)]


def _band(age: int) -> tuple[int, int]:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return (lo, hi)
    return AGE_BANDS[-1]


def index_archive(archive: str, min_kb: int) -> list[tuple[str, int]]:
    """One streaming pass: every patient bundle at or above the size floor."""
    print(f"indexing {os.path.basename(archive)} (one pass over ~1.8 GB)…", flush=True)
    out = []
    with tarfile.open(archive, "r:gz") as t:
        for m in t:
            if (m.isfile() and m.name.startswith(MEMBER_PREFIX)
                    and m.name.endswith(".json") and m.size >= min_kb * 1024):
                out.append((m.name, m.size))
    print(f"  {len(out):,} bundles at >= {min_kb} KB", flush=True)
    return out


def extract(archive: str, members: list[str], dest_dir: str) -> list[str]:
    """Second pass: pull just the chosen members out to a directory."""
    want = set(members)
    os.makedirs(dest_dir, exist_ok=True)
    written = []
    print(f"extracting {len(want)} candidates…", flush=True)
    with tarfile.open(archive, "r:gz") as t:
        for m in t:
            if m.name in want:
                src = t.extractfile(m)
                if src is None:
                    continue
                path = os.path.join(dest_dir, os.path.basename(m.name))
                with open(path, "wb") as f:
                    shutil.copyfileobj(src, f)
                written.append(path)
                if len(written) == len(want):
                    break
    print(f"  wrote {len(written)} files to {dest_dir}", flush=True)
    return written


def profile(paths: list[str]) -> list[dict]:
    """Parse candidates and score them on what this app actually needs."""
    from ingestion.fhir_parser import parse_patient_file

    rows = []
    for i, p in enumerate(paths, 1):
        if i % 50 == 0:
            print(f"  parsed {i}/{len(paths)}", flush=True)
        pat = parse_patient_file(p)
        if pat is None:
            continue
        active = [m for m in pat.medications if m.status == "active"]
        rows.append({
            "file": os.path.basename(p),
            "name": pat.name,
            "age": pat.age or 0,
            "gender": pat.gender,
            "n_conditions": len(pat.conditions),
            "n_active_meds": len(active),
            "n_abnormal_labs": sum(1 for l in pat.labs if l.is_abnormal),
            "n_procedures": len(pat.procedures),
        })
    return rows


def choose(rows: list[dict], n: int, min_multidrug: int) -> list[dict]:
    """Pick the final cohort: usable first, fan-out coverage guaranteed, ages spread.

    Ranking purely by "most medications" would yield a cohort of uniformly very sick
    patients, which is its own kind of unrepresentative. So: reserve a quota for the
    multi-drug patients the fan-out path needs, then fill the rest evenly across age
    bands.
    """
    usable = [r for r in rows if r["n_conditions"] >= 1 and r["n_active_meds"] >= 1]
    rng = random.Random(SEED)

    multi = sorted([r for r in usable if r["n_active_meds"] >= 4],
                   key=lambda r: -r["n_active_meds"])
    picked = multi[:min_multidrug]
    picked_files = {r["file"] for r in picked}

    rest = [r for r in usable if r["file"] not in picked_files]
    by_band: dict[tuple, list] = {}
    for r in rest:
        by_band.setdefault(_band(r["age"]), []).append(r)
    for v in by_band.values():
        rng.shuffle(v)

    bands = [b for b in AGE_BANDS if by_band.get(b)]
    while len(picked) < n and bands:
        for b in list(bands):
            if len(picked) >= n:
                break
            if by_band[b]:
                picked.append(by_band[b].pop())
            else:
                bands.remove(b)
    return picked


def report(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"{label}: (none)")
        return
    md = sum(1 for r in rows if r["n_active_meds"] >= 4)
    print(f"\n{label}: {len(rows)} patients")
    print(f"  with >=4 active meds : {md}")
    print(f"  mean conditions      : {sum(r['n_conditions'] for r in rows)/len(rows):.1f}")
    print(f"  mean active meds     : {sum(r['n_active_meds'] for r in rows)/len(rows):.1f}")
    print(f"  mean abnormal labs   : {sum(r['n_abnormal_labs'] for r in rows)/len(rows):.1f}")
    bands = Counter(f"{lo}-{hi if hi<200 else '+'}" for lo, hi in (_band(r["age"]) for r in rows))
    print(f"  age bands            : {dict(sorted(bands.items()))}")
    print(f"  distinct surnames    : {len({r['name'].split()[-1] for r in rows})}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Select a Synthea demo cohort")
    ap.add_argument("--archive", default=DEFAULT_ARCHIVE)
    ap.add_argument("--n", type=int, default=90, help="patients to ADD")
    ap.add_argument("--candidates", type=int, default=400)
    ap.add_argument("--min-kb", type=int, default=80)
    ap.add_argument("--min-multidrug", type=int, default=20,
                    help="guaranteed patients on >=4 active meds (fan-out coverage)")
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--apply", action="store_true",
                    help="without this, nothing is copied into --dest")
    ap.add_argument("--report", default=None, help="write the cohort profile here as JSON")
    a = ap.parse_args()

    if not os.path.exists(a.archive):
        print(f"archive not found: {a.archive}", file=sys.stderr)
        return 2

    existing = set(os.listdir(a.dest)) if os.path.isdir(a.dest) else set()
    print(f"{len(existing)} bundles already in {a.dest} — these are never overwritten")

    # Guard against GIVEN-NAME collisions, not just filename collisions. The eval
    # resolves patients by name PREFIX (resolve_patients does name.startswith(...)),
    # so a new "Bogan983_Arlie395" would make the prefix "Arlie395" ambiguous with the
    # pinned "Abbott509_Arlie395" and silently hand an eval question the wrong patient
    # against a hand-verified reference. This actually happened on the first run.
    existing_given = {f[:-5].split("_")[1] for f in existing
                      if f.endswith(".json") and len(f[:-5].split("_")) >= 2}

    idx = index_archive(a.archive, a.min_kb)
    idx = [(n, s) for n, s in idx
           if os.path.basename(n) not in existing
           and os.path.basename(n)[:-5].split("_")[1] not in existing_given]
    rng = random.Random(SEED)
    rng.shuffle(idx)
    cand = [n for n, _ in idx[:a.candidates]]

    staging = tempfile.mkdtemp(prefix="synthea_cand_")
    try:
        paths = extract(a.archive, cand, staging)
        print("parsing candidates…", flush=True)
        rows = profile(paths)
        report(rows, "CANDIDATE POOL")
        usable = [r for r in rows if r["n_conditions"] >= 1 and r["n_active_meds"] >= 1]
        print(f"  clinically usable    : {len(usable)}/{len(rows)}")

        picked = choose(rows, a.n, a.min_multidrug)
        report(picked, "SELECTED COHORT")

        if a.report:
            io.open(a.report, "w", encoding="utf-8").write(
                json.dumps({"candidates": rows, "selected": picked}, indent=2))
            print(f"\nprofile written to {a.report}")

        if not a.apply:
            print("\n--plan mode: nothing copied. Re-run with --apply to install.")
            return 0

        os.makedirs(a.dest, exist_ok=True)
        for r in picked:
            shutil.copy2(os.path.join(staging, r["file"]), os.path.join(a.dest, r["file"]))
        print(f"\ninstalled {len(picked)} bundles into {a.dest} "
              f"(now {len(os.listdir(a.dest))} total)")
        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

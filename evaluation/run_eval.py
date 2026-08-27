"""
RAGAS Evaluation Harness — evaluation/run_eval.py

Runs the curated eval set (evaluation/eval_dataset.json) through the full
agent pipeline, then scores each answer with RAGAS metrics:

  - faithfulness       (answer grounded in retrieved evidence)   [no reference needed]
  - answer_relevancy   (answer addresses the question)           [no reference needed]
  - context_precision  (retrieved chunks are relevant)           [uses reference]
  - context_recall     (retrieval captured what reference needs) [uses reference]

Judge: OpenAI gpt-4o-mini — deliberately a DIFFERENT model family from the
pipeline's Claude, to avoid self-grading bias.

Two phases, with the pipeline outputs cached to disk between them, so a RAGAS
failure never loses the (expensive) pipeline answers:
  Phase A: pipeline run  -> results/answers_<ts>.json
  Phase B: RAGAS scoring -> results/eval_<ts>.json + results/eval_<ts>.md

Usage:
    python -m evaluation.run_eval                  # full 30-question run
    python -m evaluation.run_eval --limit 3        # cheap smoke test
    python -m evaluation.run_eval --ids med_anon_02,policy_01
    python -m evaluation.run_eval --skip-pipeline results/answers_X.json
                                                   # re-score cached answers only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            override=True)

# MUST precede any ragas import (see module docstring there)
import evaluation.ragas_compat  # noqa: F401

import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
for noisy in ["httpx", "qdrant_client", "sentence_transformers", "openai", "urllib3",
              "instructor", "ragas"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.join(EVAL_DIR, "eval_dataset.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_EMBED = "text-embedding-3-small"
CONCURRENCY = 4          # parallel RAGAS scorings

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
# Dataset loading + patient resolution
# ---------------------------------------------------------------------------

def load_dataset(limit: int | None, ids: list[str] | None) -> list[dict]:
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    if limit:
        questions = questions[:limit]
    return questions


def resolve_patients(questions: list[dict]) -> dict[str, dict]:
    """Map patient name-prefixes used in the dataset to full patient contexts.

    Short-circuits when no selected question needs a patient. This matters for CI:
    data/ is gitignored, so opening MedDocDB on a runner would create an empty
    SQLite file and then fail on `no such table: patient_summaries` — even for a
    subset (e.g. med_anon) that never touches patient data.
    """
    needed = {q["patient"] for q in questions if q.get("patient")}
    if not needed:
        return {}

    from ingestion.sqlite_loader import MedDocDB
    db = MedDocDB("data/meddocai.db")
    resolved: dict[str, dict] = {}
    for p in db.list_patients():
        for prefix in needed:
            if p["name"].startswith(prefix):
                resolved[prefix] = db.get_patient(p["patient_id"])
    missing = needed - set(resolved)
    if missing:
        raise RuntimeError(f"Patients not found in DB: {missing}")
    return resolved


# ---------------------------------------------------------------------------
# Phase A — run the pipeline
# ---------------------------------------------------------------------------

def run_pipeline_phase(questions: list[dict], ts: str,
                       results_dir: str = RESULTS_DIR) -> list[dict]:
    """Run each question through the agent pipeline. Caches results to disk."""
    from graph.pipeline import answer_query

    patients = resolve_patients(questions)
    rows: list[dict] = []
    t_total = time.time()

    print(f"\nPhase A — pipeline run ({len(questions)} questions)")
    print("-" * 60)

    for i, q in enumerate(questions, 1):
        ctx = patients.get(q["patient"]) if q.get("patient") else None
        t0 = time.time()
        try:
            final = answer_query(
                q["question"],
                patient_context=ctx,
                user_role=q.get("user_role", "anonymous"),
            )
            row = {
                "id":        q["id"],
                "category":  q["category"],
                "question":  q["question"],
                "reference": q["reference"],
                "answer":    final.get("answer", ""),
                "contexts":  [e.text for e in final.get("filtered_evidence", [])],
                "intent":    final.get("intent", ""),
                "iterations": final.get("iteration", 0),
                "review_passed": final.get("review_passed", False),
                "latency_s": round(time.time() - t0, 1),
                "error":     None,
            }
        except Exception as e:
            row = {
                "id": q["id"], "category": q["category"], "question": q["question"],
                "reference": q["reference"], "answer": "", "contexts": [],
                "intent": "", "iterations": 0, "review_passed": False,
                "latency_s": round(time.time() - t0, 1), "error": str(e)[:200],
            }
        rows.append(row)
        status = "ERR" if row["error"] else f"{row['latency_s']:>5.1f}s  iter={row['iterations']}"
        print(f"  [{i:>2}/{len(questions)}] {q['id']:<14} {status}")

    os.makedirs(results_dir, exist_ok=True)
    cache_path = os.path.join(results_dir, f"answers_{ts}.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nPipeline outputs cached -> {cache_path}")
    print(f"Phase A total: {time.time() - t_total:.0f}s")
    return rows


# ---------------------------------------------------------------------------
# Phase B — RAGAS scoring
# ---------------------------------------------------------------------------

async def score_phase(rows: list[dict]) -> list[dict]:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory
    from ragas.metrics.collections import (
        AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,
    )

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # max_tokens raised: faithfulness decomposes long answers into many claims and
    # the default output budget truncated on big drug-label questions.
    llm = llm_factory(JUDGE_MODEL, client=client, max_tokens=8192)
    emb = embedding_factory("openai", model=JUDGE_EMBED, client=client)

    m_faith = Faithfulness(llm=llm)
    m_rel   = AnswerRelevancy(llm=llm, embeddings=emb)
    m_prec  = ContextPrecision(llm=llm)
    m_rec   = ContextRecall(llm=llm)

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _safe(coro_fn):
        """coro_fn: zero-arg callable returning a fresh coroutine (enables retry)."""
        last_err = ""
        for attempt in range(2):           # one retry on transient judge errors
            try:
                res = await coro_fn()
                return round(float(res.value), 4)
            except Exception as e:
                last_err = str(e)[:80]
                await asyncio.sleep(2)
        return f"error: {last_err}"

    async def score_row(row: dict) -> None:
        async with sem:
            q, ans, ctxs, ref = (row["question"], row["answer"],
                                 row["contexts"], row["reference"])
            scores: dict = {}
            if row["error"] or not ans:
                row["scores"] = {m: None for m in METRICS}
                return
            if ctxs:
                scores["faithfulness"] = await _safe(
                    lambda: m_faith.ascore(user_input=q, response=ans, retrieved_contexts=ctxs))
                scores["context_precision"] = await _safe(
                    lambda: m_prec.ascore(user_input=q, reference=ref, retrieved_contexts=ctxs))
                scores["context_recall"] = await _safe(
                    lambda: m_rec.ascore(user_input=q, retrieved_contexts=ctxs, reference=ref))
            else:
                scores["faithfulness"] = None
                scores["context_precision"] = None
                scores["context_recall"] = None
            scores["answer_relevancy"] = await _safe(
                lambda: m_rel.ascore(user_input=q, response=ans))
            row["scores"] = scores

    print(f"\nPhase B — RAGAS scoring (judge={JUDGE_MODEL}, "
          f"concurrency={CONCURRENCY})")
    print("-" * 60)
    t0 = time.time()
    await asyncio.gather(*(score_row(r) for r in rows))
    print(f"Phase B total: {time.time() - t0:.0f}s")
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _numeric(values: list) -> list[float]:
    return [v for v in values if isinstance(v, (int, float))]


def build_report(rows: list[dict], ts: str,
                 results_dir: str = RESULTS_DIR) -> tuple[str, dict]:
    """Write JSON + markdown reports. Returns (markdown, summary dict)."""
    # Aggregates: overall + per category
    def agg(subset: list[dict]) -> dict:
        out = {}
        for m in METRICS:
            vals = _numeric([r.get("scores", {}).get(m) for r in subset])
            out[m] = round(sum(vals) / len(vals), 4) if vals else None
        return out

    categories = sorted({r["category"] for r in rows})
    summary = {
        "timestamp": ts,
        "judge_model": JUDGE_MODEL,
        "n_questions": len(rows),
        "overall": agg(rows),
        "by_category": {c: agg([r for r in rows if r["category"] == c])
                        for c in categories},
    }

    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"eval_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)

    # Markdown report
    lines = [
        f"# MedDocAI RAGAS Evaluation — {ts}",
        "",
        f"- Questions: **{len(rows)}**   Judge: **{JUDGE_MODEL}**   "
        f"Pipeline model: claude-sonnet-4-5",
        "",
        "## Overall scores",
        "",
        "| Metric | Score |",
        "|--------|------:|",
    ]
    for m in METRICS:
        v = summary["overall"][m]
        lines.append(f"| {m} | {v if v is not None else 'n/a'} |")

    lines += ["", "## By category", "",
              "| Category | " + " | ".join(METRICS) + " |",
              "|----------|" + "------:|" * len(METRICS)]
    for c in categories:
        vals = [summary["by_category"][c][m] for m in METRICS]
        lines.append(f"| {c} | " + " | ".join(
            str(v) if v is not None else "n/a" for v in vals) + " |")

    lines += ["", "## Per-question", "",
              "| id | faith | relev | prec | recall | iter | latency |",
              "|----|------:|------:|-----:|-------:|-----:|--------:|"]
    for r in rows:
        s = r.get("scores", {})
        def fmt(m):
            v = s.get(m)
            return f"{v:.2f}" if isinstance(v, (int, float)) else "—"
        lines.append(
            f"| {r['id']} | {fmt('faithfulness')} | {fmt('answer_relevancy')} | "
            f"{fmt('context_precision')} | {fmt('context_recall')} | "
            f"{r['iterations']} | {r['latency_s']}s |")

    md = "\n".join(lines)
    md_path = os.path.join(results_dir, f"eval_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nReports written:\n  {json_path}\n  {md_path}")
    return md, summary



# ---------------------------------------------------------------------------
# Quality gate (used by .github/workflows/eval-gate.yml)
# ---------------------------------------------------------------------------

def check_gate(
    summary: dict,
    rows: list[dict],
    thresholds: dict,
    min_scored_frac: float = 0.8,
    max_pipeline_errors: int = 0,
) -> tuple[bool, list[str]]:
    """Decide whether this run clears the quality bar. FAIL-CLOSED.

    A score can be a float, None, or the string "error: ...". `_numeric()` silently
    drops the last two when averaging, which means a run where 8/10 rows failed
    would report the mean of the 2 survivors and look great. So three separate
    rules guard that, and a MISSING aggregate is a failure, never a pass.

    Returns (passed, reasons).
    """
    reasons: list[str] = []
    n = len(rows)

    n_err = sum(1 for r in rows if r.get("error"))
    if n_err > max_pipeline_errors:
        reasons.append(f"{n_err} pipeline error(s) > allowed {max_pipeline_errors}")

    overall = summary.get("overall", {}) or {}
    for metric, floor in thresholds.items():
        value = overall.get(metric)

        # Rule 1 — a missing aggregate is a failure, not a pass.
        if value is None:
            reasons.append(f"{metric}: no numeric score produced (fail-closed)")
            continue

        # Rule 2 — enough rows must actually have been scored.
        scored = len(_numeric([r.get("scores", {}).get(metric) for r in rows]))
        frac = (scored / n) if n else 0.0
        if frac < min_scored_frac:
            reasons.append(
                f"{metric}: only {scored}/{n} rows scored "
                f"({frac:.0%} < {min_scored_frac:.0%}) — judge errors suspected"
            )
            continue

        # Rule 3 — the actual quality bar.
        if value < floor:
            reasons.append(f"{metric} {value:.4f} < {floor:.2f}")

    return (not reasons), reasons


def _parse_thresholds(pairs: list[str] | None) -> dict:
    """Parse repeated --fail-under METRIC=FLOAT into {metric: float}."""
    out: dict[str, float] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise SystemExit(f"--fail-under expects METRIC=FLOAT, got: {raw}")
        metric, _, value = raw.partition("=")
        metric = metric.strip()
        if metric not in METRICS:
            raise SystemExit(f"unknown metric {metric!r}; choose from {METRICS}")
        out[metric] = float(value)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions (smoke test)")
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated question ids to run")
    parser.add_argument("--skip-pipeline", type=str, default=None,
                        help="Path to a cached answers_*.json — score it without "
                             "re-running the pipeline")
    parser.add_argument("--out", type=str, default=RESULTS_DIR,
                        help="Directory for the report files (CI writes outside the repo)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Use this instead of a timestamp in filenames, so CI "
                             "knows the path without globbing (e.g. --tag ci)")
    parser.add_argument("--fail-under", action="append", metavar="METRIC=FLOAT",
                        help="Fail (exit 1) if the overall METRIC is below FLOAT. "
                             "Repeatable. Passing any of these enables the gate.")
    parser.add_argument("--min-scored-frac", type=float, default=0.8,
                        help="Per gated metric, the minimum fraction of rows that must "
                             "have a numeric score (guards against judge errors)")
    parser.add_argument("--max-pipeline-errors", type=int, default=0,
                        help="Maximum rows allowed to have a pipeline error")
    args = parser.parse_args()

    ts = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.skip_pipeline:
        with open(args.skip_pipeline, encoding="utf-8") as f:
            rows = json.load(f)
        if args.ids:
            wanted = set(args.ids.split(","))
            rows = [r for r in rows if r["id"] in wanted]
        if args.limit:
            rows = rows[:args.limit]
    else:
        ids = args.ids.split(",") if args.ids else None
        questions = load_dataset(args.limit, ids)
        rows = run_pipeline_phase(questions, ts, results_dir=args.out)

    rows = asyncio.run(score_phase(rows))
    md, summary = build_report(rows, ts, results_dir=args.out)

    # Console summary (top section of the markdown)
    print("\n" + "\n".join(md.split("\n## Per-question")[0].splitlines()))

    # ---- Quality gate (only active when --fail-under was passed) ----
    thresholds = _parse_thresholds(args.fail_under)
    if not thresholds:
        return 0

    passed, reasons = check_gate(
        summary, rows, thresholds,
        min_scored_frac=args.min_scored_frac,
        max_pipeline_errors=args.max_pipeline_errors,
    )
    print("\n" + "=" * 60)
    if passed:
        print("QUALITY GATE: PASS")
        for metric, floor in thresholds.items():
            print(f"  OK  {metric}: {summary['overall'][metric]:.4f} >= {floor:.2f}")
        return 0
    for reason in reasons:
        print(f"GATE FAIL: {reason}")
    print("QUALITY GATE: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())

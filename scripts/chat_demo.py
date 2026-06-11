"""
Interactive Chat Demo — scripts/chat_demo.py

A terminal REPL to test the full MedDocAI agent pipeline by hand.

Run:
    python scripts/chat_demo.py            # interactive
    python scripts/chat_demo.py --demo     # run scripted example queries (non-interactive)
    python scripts/chat_demo.py --trace    # start with agent trace visible

In the REPL, type a question or a command:
    /help       show commands
    /who        show current context (role + selected patient)
    /context    re-select role / patient
    /patients   list available patients
    /trace      toggle showing the agent trace
    /evidence   toggle showing the evidence sources used
    /examples   show example questions to try
    /quit       exit

Requires .env with ANTHROPIC_API_KEY and OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make stdout UTF-8 so Claude's output (m², ≥, —, emoji) prints on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            override=True)

# Quiet the noisy library logs so the demo output is clean
import logging
logging.basicConfig(level=logging.ERROR)
for noisy in ["httpx", "qdrant_client", "sentence_transformers", "openai", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

SEP = "=" * 70


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_answer(final: dict, show_trace: bool, show_evidence: bool) -> None:
    if show_trace:
        print("\n  ── agent trace ─────────────────────────────────────────")
        for step in final.get("trace", []):
            print(f"    {step}")

    if show_evidence:
        print("\n  ── evidence used ───────────────────────────────────────")
        for i, e in enumerate(final.get("filtered_evidence", []), 1):
            score = f"{e.score:.3f}" if e.score is not None else "pinned"
            print(f"    [{i}] ({score:>6}) {e.source}: {e.title[:55]}")

    print("\n" + SEP)
    print(final.get("final_answer", "(no answer)"))
    print(SEP)


# ---------------------------------------------------------------------------
# Context selection
# ---------------------------------------------------------------------------

def load_patients():
    from ingestion.sqlite_loader import MedDocDB
    db = MedDocDB("data/meddocai.db")
    rows = db.list_patients()
    return db, rows


def choose_context(db, patient_rows):
    """Interactive: pick anonymous / patient / care-manager + which patient.

    Returns (role, patient_context_dict_or_None, label).
    """
    print("\nWho are you? Choose a context:")
    print("  [0] Anonymous visitor (no patient record)")
    print("  [1] Patient   — answers in plain, patient-friendly language")
    print("  [2] Care manager — clinical tone, reviewing a patient")
    choice = input("Select 0/1/2: ").strip()

    if choice == "0" or choice not in ("1", "2"):
        return "anonymous", None, "Anonymous"

    role = "patient" if choice == "1" else "care_manager"

    # Pick a patient
    print(f"\nSelect a patient ({len(patient_rows)} available):")
    for i, p in enumerate(patient_rows):
        flag = "summary" if p["has_summary"] else "no summary"
        print(f"  [{i}] {p['name']}  ({p['age']}{p['gender'][0].upper()}, {flag})")
    idx = input(f"Select 0-{len(patient_rows)-1}: ").strip()

    try:
        patient = db.get_patient(patient_rows[int(idx)]["patient_id"])
    except (ValueError, IndexError):
        print("Invalid selection — defaulting to anonymous.")
        return "anonymous", None, "Anonymous"

    label = f"{role} viewing {patient['name']}"
    return role, patient, label


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

EXAMPLES = [
    ("Medication info", "What are the main warnings for metformin?"),
    ("Drug recall (live)", "Is there a current recall on semaglutide?"),
    ("Policy / eligibility", "What is the Medicaid income limit for pregnant women in Texas?"),
    ("Condition education", "Explain high blood pressure in simple terms"),
    ("Personalized (pick a patient first)", "Is metformin safe for this patient?"),
    ("Patient summary (pick a patient first)", "Summarize this patient for me"),
]


def show_examples():
    print("\nExample questions to try:")
    for cat, q in EXAMPLES:
        print(f"  • [{cat}] {q}")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_interactive(show_trace: bool):
    print(SEP)
    print("  MedDocAI — Interactive Chat Demo")
    print(SEP)
    print("Loading pipeline (first question is slower — loads the reranker model)...")

    from graph.pipeline import answer_query   # triggers model loads
    db, patient_rows = load_patients()

    role, patient_ctx, label = choose_context(db, patient_rows)
    show_evidence = False

    print("\nReady. Type a question, or /help for commands. /quit to exit.")
    print(f"Context: {label}")

    while True:
        try:
            user_in = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_in:
            continue

        # Commands
        if user_in.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye.")
            break
        if user_in.lower() in ("/help", "/h", "?"):
            print("\nCommands: /who  /context  /patients  /trace  /evidence  /examples  /quit")
            continue
        if user_in.lower() == "/who":
            print(f"  Context: {label}")
            print(f"  trace={'on' if show_trace else 'off'}, "
                  f"evidence={'on' if show_evidence else 'off'}")
            continue
        if user_in.lower() == "/context":
            role, patient_ctx, label = choose_context(db, patient_rows)
            print(f"Context: {label}")
            continue
        if user_in.lower() == "/patients":
            for i, p in enumerate(patient_rows):
                print(f"  [{i}] {p['name']} ({p['age']}{p['gender'][0].upper()})")
            continue
        if user_in.lower() == "/trace":
            show_trace = not show_trace
            print(f"  trace {'on' if show_trace else 'off'}")
            continue
        if user_in.lower() == "/evidence":
            show_evidence = not show_evidence
            print(f"  evidence {'on' if show_evidence else 'off'}")
            continue
        if user_in.lower() == "/examples":
            show_examples()
            continue
        if user_in.startswith("/"):
            print("  Unknown command. /help for options.")
            continue

        # It's a question — run the pipeline
        print("  ...thinking...")
        try:
            final = answer_query(user_in, patient_context=patient_ctx, user_role=role)
            print_answer(final, show_trace, show_evidence)
        except Exception as e:
            print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Scripted demo (non-interactive)
# ---------------------------------------------------------------------------

def run_demo():
    print(SEP)
    print("  MedDocAI — Scripted Demo (non-interactive)")
    print(SEP)
    print("Loading pipeline...\n")

    from graph.pipeline import answer_query
    db, patient_rows = load_patients()

    # Pick the richest patient for the personalized examples
    richest = max(
        (db.get_patient(p["patient_id"]) for p in patient_rows),
        key=lambda p: len(p.get("conditions_json", [])),
    )

    scripted = [
        ("Anonymous · medication", "What are the main warnings for metformin?", None, "anonymous"),
        ("Anonymous · policy", "Medicaid income limit for pregnant women in Texas?", None, "anonymous"),
        ("Care manager · personalized", "Is metformin safe for this patient?", richest, "care_manager"),
        (f"Care manager · summary", "Summarize this patient", richest, "care_manager"),
    ]

    for label, query, ctx, role in scripted:
        print("\n" + SEP)
        print(f"  [{label}]")
        print(f"  Q: {query}")
        if ctx:
            print(f"  Patient: {ctx['name']}")
        print(SEP)
        final = answer_query(query, patient_context=ctx, user_role=role)
        # show compact trace
        for step in final.get("trace", []):
            if step.startswith(("Router:", "Filter:", "Reviewer:", "PatientSummary:")):
                print(f"    {step}")
        print("\n  ANSWER:")
        ans = final.get("final_answer", "")
        # indent the answer for readability
        for line in ans.split("\n")[:18]:
            print(f"    {line}")
        if len(ans.split("\n")) > 18:
            print("    ... (truncated)")

    print("\n" + SEP)
    print("  Demo complete.")
    print(SEP)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedDocAI interactive chat demo")
    parser.add_argument("--demo", action="store_true",
                        help="Run scripted example queries non-interactively")
    parser.add_argument("--trace", action="store_true",
                        help="Start with the agent trace visible")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in .env")
        sys.exit(1)

    if args.demo:
        run_demo()
    else:
        run_interactive(show_trace=args.trace)

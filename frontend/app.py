"""
MedDocAI — Streamlit UI — frontend/app.py

A clickable chat interface over the multi-agent pipeline.

Run:
    streamlit run frontend/app.py

Layout:
  - Sidebar: context selector (anonymous / patient / care manager) + rich patient panel
  - Main:    chat history with grounded answers, collapsible Sources + agent-trace panels

Requires .env with ANTHROPIC_API_KEY and OPENAI_API_KEY.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            override=True)

import logging
logging.basicConfig(level=logging.ERROR)
for noisy in ["httpx", "qdrant_client", "sentence_transformers", "openai", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(page_title="MedDocAI", page_icon="🏥", layout="wide")


# ---------------------------------------------------------------------------
# Cached heavy resources (loaded once, reused across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_db():
    from ingestion.sqlite_loader import MedDocDB
    return MedDocDB("data/meddocai.db")


@st.cache_resource(show_spinner=False)
def _get_pipeline():
    # Building the pipeline + warming the model singletons happens once.
    from graph.pipeline import get_pipeline
    return get_pipeline()


@st.cache_data(show_spinner=False)
def _patient_list():
    return _get_db().list_patients()


def _run_query(query: str, patient_context: dict | None, role: str,
               conversation_history: list[dict]) -> dict:
    """Invoke the compiled pipeline (mirrors graph.pipeline.answer_query)."""
    from agents.state import new_state
    pipeline = _get_pipeline()
    state = new_state(query, patient_context, role, max_iterations=2,
                      conversation_history=conversation_history)
    return pipeline.invoke(state, config={"recursion_limit": 25})


# ---------------------------------------------------------------------------
# Persistence helpers (per-patient chat history in SQLite)
# ---------------------------------------------------------------------------

def _load_history(patient_id: str, persona: str) -> list[dict]:
    """Load a patient's stored history (for this persona) into app message format."""
    rows = _get_db().get_patient_chat_history(patient_id, persona=persona)
    messages = []
    for r in rows:
        if r["role"] == "user":
            messages.append({"role": "user", "content": r["content"]})
        else:
            blob = r.get("sources") or {}
            messages.append({
                "role": "assistant",
                "answer": r["content"],
                "disclaimers": blob.get("disclaimers", []),
                "evidence": blob.get("evidence", []),
                "trace": blob.get("trace", []),
            })
    return messages


def _save_user_message(patient_id: str, persona: str, content: str) -> None:
    _get_db().save_chat_message(
        session_id=st.session_state.session_id,
        role="user", content=content,
        patient_id=patient_id, persona=persona,
    )


def _save_assistant_message(patient_id: str, persona: str, msg: dict) -> None:
    _get_db().save_chat_message(
        session_id=st.session_state.session_id,
        role="assistant", content=msg["answer"],
        patient_id=patient_id, persona=persona,
        sources={
            "evidence": msg.get("evidence", []),
            "disclaimers": msg.get("disclaimers", []),
            "trace": msg.get("trace", []),
        },
    )


def _conversation_for_pipeline() -> list[dict]:
    """Build {role, content} turns from session messages for the pipeline."""
    turns = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            turns.append({"role": "user", "content": m["content"]})
        else:
            turns.append({"role": "assistant", "content": m.get("answer", "")})
    return turns


# ---------------------------------------------------------------------------
# Sidebar — context selector + patient panel
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, dict | None]:
    """Returns (role, patient_context)."""
    st.sidebar.title("🏥 MedDocAI")
    st.sidebar.caption("Healthcare document intelligence — source-grounded answers.")

    st.sidebar.markdown("### Who are you?")
    role_label = st.sidebar.radio(
        "Context",
        ["Anonymous visitor", "Patient", "Care manager"],
        label_visibility="collapsed",
    )

    role_map = {
        "Anonymous visitor": "anonymous",
        "Patient": "patient",
        "Care manager": "care_manager",
    }
    role = role_map[role_label]

    patient_context = None
    if role in ("patient", "care_manager"):
        patients = _patient_list()
        options = {f"{p['name']}  ({p['age']}{p['gender'][0].upper()})": p["patient_id"]
                   for p in patients}
        label = "Select your record:" if role == "patient" else "Select a patient:"
        picked = st.sidebar.selectbox(label, list(options.keys()))
        if picked:
            patient_context = _get_db().get_patient(options[picked])

    # Rich active-context panel
    if patient_context:
        _render_context_panel(patient_context)
    else:
        st.sidebar.markdown("---")
        st.sidebar.info("No patient selected — answers will be general (not personalized).")

    st.sidebar.markdown("---")
    if st.sidebar.button(
        "🗑️ Clear view",
        help="Clears the on-screen chat only. Your saved history is kept and "
             "reloads next time you select this patient.",
    ):
        # View-only clear — does NOT touch the persisted history in SQLite.
        st.session_state.messages = []
        st.rerun()

    st.sidebar.caption("Developer options")
    st.session_state.show_sources = st.sidebar.checkbox("Show sources by default", value=False)
    st.session_state.show_trace = st.sidebar.checkbox("Show agent trace by default", value=False)

    return role, patient_context


def _render_context_panel(ctx: dict) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Active context")
    st.sidebar.markdown(f"**{ctx.get('name','')}** · {ctx.get('age','?')}"
                        f"{ctx.get('gender','')[0].upper() if ctx.get('gender') else ''}")

    conditions = ctx.get("conditions_json") or []
    meds = [m for m in (ctx.get("medications_json") or []) if m.get("status") == "active"]
    abnormal = [l for l in (ctx.get("labs_json") or []) if l.get("is_abnormal")]

    if conditions:
        names = [c.get("display", "") for c in conditions][:6]
        st.sidebar.markdown("**📋 Conditions**")
        st.sidebar.caption("; ".join(names))
    if meds:
        names = [m.get("display", "") for m in meds][:6]
        st.sidebar.markdown("**💊 Active medications**")
        st.sidebar.caption("; ".join(names))
    if abnormal:
        flags = [f"⚠️ {l.get('display','')}: {l.get('value','')}" for l in abnormal][:5]
        st.sidebar.markdown("**🔬 Abnormal labs**")
        st.sidebar.caption("  \n".join(flags))

    has_summary = bool(ctx.get("summary_md"))
    st.sidebar.caption(f"Care summary: {'available' if has_summary else 'not generated'}")


# ---------------------------------------------------------------------------
# Rendering an assistant message (answer + expanders)
# ---------------------------------------------------------------------------

def _render_assistant(msg: dict) -> None:
    st.markdown(msg["answer"])

    # Disclaimers
    for d in msg.get("disclaimers", []):
        st.caption(d)

    evidence = msg.get("evidence", [])
    trace = msg.get("trace", [])

    if evidence:
        with st.expander(f"📚 Sources ({len(evidence)})",
                         expanded=st.session_state.get("show_sources", False)):
            for i, e in enumerate(evidence, 1):
                score = f"`{e['score']:.3f}`" if e["score"] is not None else "`pinned`"
                st.markdown(f"**[{i}]** {score} · *{e['source']}* — {e['title']}")
                if e.get("text"):
                    st.caption(e["text"][:240] + ("…" if len(e["text"]) > 240 else ""))

    if trace:
        with st.expander("🔍 How I answered this (agent trace)",
                         expanded=st.session_state.get("show_trace", False)):
            for step in trace:
                st.text(step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import uuid

    # Guard: keys present
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        st.error("ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in .env")
        st.stop()

    # Per-app-session id (groups this run's messages)
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "loaded_key" not in st.session_state:
        st.session_state.loaded_key = None

    role, patient_context = render_sidebar()
    patient_id = patient_context["patient_id"] if patient_context else None
    persona = role if role in ("patient", "care_manager") else None

    # Load persisted history when the selected (patient, persona) changes.
    # Anonymous (no patient) gets a fresh, non-persisted, session-only chat.
    current_key = (patient_id, persona)
    if current_key != st.session_state.loaded_key:
        if patient_id:
            st.session_state.messages = _load_history(patient_id, persona)
        else:
            st.session_state.messages = []
        st.session_state.loaded_key = current_key

    # Header
    ctx_label = ("Anonymous" if not patient_context
                 else f"{role.replace('_',' ').title()} · {patient_context['name']}")
    st.title("MedDocAI")
    st.caption(f"Context: **{ctx_label}** — ask about medications, conditions, "
               f"drug recalls, or Medicaid policy. Answers are source-grounded with citations.")
    if patient_id and st.session_state.messages:
        st.caption(f"💾 Loaded {len(st.session_state.messages)} message(s) from this "
                   f"patient's history.")
    elif not patient_id:
        st.caption("ℹ️ Anonymous chat is not saved. Select a patient to keep history "
                   "across sessions.")

    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant(msg)
            else:
                st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input("Ask a question…")
    if prompt:
        # Conversation history for the pipeline (built BEFORE adding the new turn)
        convo = _conversation_for_pipeline()

        # Show + store the user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        if patient_id:
            _save_user_message(patient_id, persona, prompt)

        # Run the pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking… (routing → retrieving → reranking → answering → reviewing)"):
                try:
                    final = _run_query(prompt, patient_context, role, convo)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.stop()

            evidence = [{
                "score": e.score,
                "source": e.source,
                "title": e.title,
                "text": e.text,
            } for e in final.get("filtered_evidence", [])]

            assistant_msg = {
                "role": "assistant",
                "answer": final.get("answer", "(no answer)"),
                "disclaimers": final.get("disclaimers", []),
                "evidence": evidence,
                "trace": final.get("trace", []),
            }
            _render_assistant(assistant_msg)
            st.session_state.messages.append(assistant_msg)
            if patient_id:
                _save_assistant_message(patient_id, persona, assistant_msg)


if __name__ == "__main__":
    main()

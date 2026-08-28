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


@st.cache_data(show_spinner=False)
def _patient_list():
    return _get_db().list_patients()


def _run_query(query: str, patient_context: dict | None, role: str,
               conversation_history: list[dict]) -> dict:
    """Run one query through the pipeline.

    Delegates to graph.pipeline.answer_query rather than re-implementing it: that
    is the single instrumented entry point (observability + tracing), so a local
    copy of the invoke would silently bypass all per-query metrics. The compiled
    graph is already a module-level singleton inside get_pipeline().
    """
    from graph.pipeline import answer_query
    return answer_query(query, patient_context, role, max_iterations=2,
                        conversation_history=conversation_history)


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
                "metrics": blob.get("metrics", {}),
                # query_id lets a rating survive a page reload (and links the
                # feedback row back to its query_metrics row).
                "query_id": blob.get("query_id"),
                "question": blob.get("question", ""),
                "intent": blob.get("intent", ""),
            })
    return messages


def _hydrate_feedback(messages: list[dict]) -> None:
    """Load any ratings already stored for these messages into session state."""
    ids = [m["query_id"] for m in messages if m.get("query_id")]
    if not ids:
        return
    for row in _get_db().get_feedback(query_ids=ids):
        st.session_state.feedback[row["query_id"]] = row["rating"]


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
            "metrics": msg.get("metrics", {}),
            "query_id": msg.get("query_id"),
            "question": msg.get("question", ""),
            "intent": msg.get("intent", ""),
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

    # Feedback tally — makes the improvement loop visible, and thumbs-down answers
    # are exported into eval candidates by scripts/feedback_to_eval.py.
    try:
        counts = _get_db().feedback_counts()
        if counts["up"] or counts["down"]:
            st.sidebar.caption(
                f"📊 Feedback: 👍 {counts['up']} · 👎 {counts['down']}")
    except Exception:
        pass                       # a stats caption must never break the sidebar

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

def _record_feedback(msg: dict, rating: str, comment: str = "") -> None:
    """Persist one thumbs verdict and mirror it into session state."""
    qid = msg.get("query_id")
    if not qid:
        return
    ctx = st.session_state.get("active_patient_id")
    try:
        _get_db().save_feedback(
            query_id=qid,
            rating=rating,
            comment=comment,
            question=msg.get("question", ""),
            answer=msg.get("answer", ""),
            intent=msg.get("intent", ""),
            user_role=st.session_state.get("active_role", ""),
            patient_id=ctx,
            persona=st.session_state.get("active_persona"),
            session_id=st.session_state.session_id,
            n_evidence=len(msg.get("evidence", [])),
        )
    except Exception as e:                      # never break the chat over feedback
        st.warning(f"Could not save feedback: {e}")
        return
    st.session_state.feedback[qid] = rating


def _render_feedback_controls(msg: dict) -> None:
    """Thumbs up/down under an answer, plus a note box on thumbs-down.

    The note is the highest-value signal in the whole loop: 'wrong drug' or
    'ignored her kidney disease' is what turns a bad answer into a test case.
    """
    qid = msg.get("query_id")
    if not qid:
        return                                   # replayed pre-feedback history
    current = st.session_state.feedback.get(qid)

    up_col, down_col, msg_col = st.columns([1, 1, 8])
    with up_col:
        if st.button("👍", key=f"fb_up_{qid}",
                     type="primary" if current == "up" else "secondary",
                     help="This answer was helpful"):
            _record_feedback(msg, "up")
            st.rerun()
    with down_col:
        if st.button("👎", key=f"fb_down_{qid}",
                     type="primary" if current == "down" else "secondary",
                     help="Something was wrong with this answer"):
            _record_feedback(msg, "down")
            st.rerun()
    with msg_col:
        if current == "up":
            st.caption("Thanks — logged as helpful.")
        elif current == "down":
            st.caption("Thanks — logged for review; it becomes an eval candidate.")

    if current == "down":
        with st.expander("What was wrong? (optional — this becomes the test case)"):
            note = st.text_area(
                "Note", key=f"fb_note_{qid}", label_visibility="collapsed",
                placeholder="e.g. missed her kidney disease / cited the wrong drug / "
                            "answered a different question",
            )
            if st.button("Save note", key=f"fb_save_{qid}"):
                _record_feedback(msg, "down", comment=note)
                st.success("Saved.")


def _render_assistant(msg: dict) -> None:
    st.markdown(msg["answer"])

    # Disclaimers
    for d in msg.get("disclaimers", []):
        st.caption(d)

    # Per-query cost/latency/token summary (observability)
    metrics = msg.get("metrics") or {}
    if metrics:
        from agents.observability import format_summary
        st.caption(f"⏱ {format_summary(metrics)}")

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

    _render_feedback_controls(msg)


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
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}          # {query_id: 'up'|'down'}

    role, patient_context = render_sidebar()
    patient_id = patient_context["patient_id"] if patient_context else None
    persona = role if role in ("patient", "care_manager") else None

    # Stashed so the feedback writer can label rows without threading args through
    # every render call.
    st.session_state.active_patient_id = patient_id
    st.session_state.active_persona = persona
    st.session_state.active_role = role

    # Load persisted history when the selected (patient, persona) changes.
    # Anonymous (no patient) gets a fresh, non-persisted, session-only chat.
    current_key = (patient_id, persona)
    if current_key != st.session_state.loaded_key:
        if patient_id:
            st.session_state.messages = _load_history(patient_id, persona)
            _hydrate_feedback(st.session_state.messages)
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
                "metrics": final.get("metrics", {}),
                # Carried so a thumbs verdict can be attributed and, later,
                # rebuilt into an eval question without a chat_history join.
                "query_id": final.get("query_id"),
                "question": prompt,
                "intent": final.get("intent", ""),
            }
            _render_assistant(assistant_msg)
            st.session_state.messages.append(assistant_msg)
            if patient_id:
                _save_assistant_message(patient_id, persona, assistant_msg)


if __name__ == "__main__":
    main()

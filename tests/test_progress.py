"""Tests for live pipeline progress — tests/test_progress.py

Covers the `on_progress` callback on graph.pipeline.answer_query and the
Streamlit reporter built on top of it.

The load-bearing property is NOT that progress lines look nice: it is that
switching to the streaming code path does not change the answer. `answer_query`
is the single instrumented entry point (observability + tracing + metrics
persistence), so the streaming branch must return exactly what `.invoke()`
returns and must still record metrics.
"""

from typing import TypedDict

import pytest
from langgraph.graph import END, StateGraph

import graph.pipeline as P


class _S(TypedDict, total=False):
    # NOTE: LangGraph DROPS any key a node returns that is not declared here —
    # silently, with no error. An under-declared fake state makes a node's
    # update look empty and the test fail for the wrong reason.
    query: str
    query_id: str
    iteration: int
    trace: list
    intent: str
    sub_queries: list
    raw_evidence: list
    filtered_evidence: list
    answer: str
    final_answer: str
    disclaimers: list
    review_passed: bool
    route_back_to: str


def _fake_pipeline(fail_review_once: bool = False):
    """A miniature graph with the same shape as the real one (incl. the retry loop)."""
    def router(s):
        return {"intent": "medication_info", "sub_queries": ["a", "b"],
                "trace": s.get("trace", []) + ["router"]}

    def retrieval(s):
        return {"raw_evidence": [1, 2, 3], "trace": s["trace"] + ["retrieval"]}

    def evidence_filter(s):
        return {"filtered_evidence": [1, 2], "trace": s["trace"] + ["evidence_filter"]}

    def answer_generator(s):
        return {"answer": "draft", "trace": s["trace"] + ["answer_generator"]}

    def reviewer(s):
        it = s.get("iteration", 0) + 1
        passed = not (fail_review_once and it == 1)
        return {"review_passed": passed, "iteration": it,
                "route_back_to": None if passed else "answer_generator",
                "trace": s["trace"] + ["reviewer"]}

    def safety(s):
        return {"disclaimers": ["d1"], "final_answer": "final",
                "trace": s["trace"] + ["safety"]}

    g = StateGraph(_S)
    for name, fn in [("router", router), ("retrieval", retrieval),
                     ("evidence_filter", evidence_filter),
                     ("answer_generator", answer_generator),
                     ("reviewer", reviewer), ("safety", safety)]:
        g.add_node(name, fn)
    g.set_entry_point("router")
    g.add_edge("router", "retrieval")
    g.add_edge("retrieval", "evidence_filter")
    g.add_edge("evidence_filter", "answer_generator")
    g.add_edge("answer_generator", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        lambda s: "safety" if s.get("review_passed") else "answer_generator",
        {"safety": "safety", "answer_generator": "answer_generator"},
    )
    g.add_edge("safety", END)
    return g.compile()


@pytest.fixture
def patched(monkeypatch):
    """Point answer_query at the fake graph; never touch SQLite or the network."""
    def _install(fail_review_once=False):
        monkeypatch.setattr(P, "get_pipeline", lambda: _fake_pipeline(fail_review_once))
        monkeypatch.setattr(P, "new_state",
                            lambda *a, **k: {"query": "q", "query_id": "qid", "trace": []})
        monkeypatch.setattr(P, "_persist_metrics", lambda *a, **k: None)
    return _install


# ---------------------------------------------------------------------------
# The callback must not change the answer
# ---------------------------------------------------------------------------

def test_streaming_path_returns_identical_state(patched):
    """The whole point: opting into progress must not alter the result."""
    patched()
    base = P.answer_query("q")
    streamed = P.answer_query("q", on_progress=lambda node, upd: None)

    for key in ("answer", "final_answer", "trace", "disclaimers", "iteration"):
        assert base.get(key) == streamed.get(key), f"{key} diverged"


def test_metrics_still_recorded_on_streaming_path(patched):
    """Streaming must not bypass observability — that bug already happened once."""
    patched()
    final = P.answer_query("q", on_progress=lambda node, upd: None)
    assert "metrics" in final
    assert final["metrics"].get("query_id")


def test_default_is_unchanged_invoke_path(patched, monkeypatch):
    """With no callback the graph is .invoke()d, never .stream()ed."""
    patched()
    real = P.get_pipeline()
    seen = {"invoke": 0, "stream": 0}

    class _Spy:
        def invoke(self, *a, **k):
            seen["invoke"] += 1
            return real.invoke(*a, **k)

        def stream(self, *a, **k):
            seen["stream"] += 1
            return real.stream(*a, **k)

    monkeypatch.setattr(P, "get_pipeline", lambda: _Spy())

    P.answer_query("q")
    assert seen == {"invoke": 1, "stream": 0}

    P.answer_query("q", on_progress=lambda n, u: None)
    assert seen == {"invoke": 1, "stream": 1}


# ---------------------------------------------------------------------------
# What the callback reports
# ---------------------------------------------------------------------------

def test_every_node_is_reported_in_order(patched):
    patched()
    seen = []
    P.answer_query("q", on_progress=lambda node, upd: seen.append(node))
    assert seen == ["router", "retrieval", "evidence_filter",
                    "answer_generator", "reviewer", "safety"]


def test_retry_reports_the_repeated_nodes(patched):
    """A corrective-RAG retry must be VISIBLE, not silently collapsed."""
    patched(fail_review_once=True)
    seen = []
    P.answer_query("q", on_progress=lambda node, upd: seen.append(node))
    assert seen.count("answer_generator") == 2
    assert seen.count("reviewer") == 2


def test_callback_receives_the_node_update(patched):
    patched()
    got = {}
    P.answer_query("q", on_progress=lambda node, upd: got.setdefault(node, upd))
    assert got["router"]["intent"] == "medication_info"
    assert len(got["router"]["sub_queries"]) == 2
    assert got["evidence_filter"]["filtered_evidence"] == [1, 2]


# ---------------------------------------------------------------------------
# The callback must never break the answer
# ---------------------------------------------------------------------------

def test_raising_callback_does_not_break_the_answer(patched):
    """UI telemetry is best-effort; the same rule as _persist_metrics."""
    patched()
    def boom(node, upd):
        raise RuntimeError("UI exploded")

    final = P.answer_query("q", on_progress=boom)
    assert final.get("final_answer") == "final"
    assert "metrics" in final


# ---------------------------------------------------------------------------
# The Streamlit reporter
# ---------------------------------------------------------------------------

class _FakeStatus:
    def __init__(self):
        self.lines = []

    def write(self, msg):
        self.lines.append(msg)


def _reporter():
    from frontend.app import _progress_reporter
    st = _FakeStatus()
    return st, _progress_reporter(st)


def test_reporter_names_the_intent_and_fanout():
    st, report = _reporter()
    report("router", {"intent": "medication_info", "sub_queries": ["a", "b", "c"]})
    assert "medication_info" in st.lines[0]
    assert "×3" in st.lines[0]


def test_reporter_omits_fanout_when_absent():
    st, report = _reporter()
    report("router", {"intent": "policy_eligibility", "sub_queries": []})
    assert "fan-out" not in st.lines[0]


def test_reporter_distinguishes_review_pass_from_retry():
    st, report = _reporter()
    report("reviewer", {"review_passed": True})
    report("reviewer", {"review_passed": False, "route_back_to": "router"})
    assert "passed" in st.lines[0]
    assert "router" in st.lines[1]


def test_reporter_counts_evidence():
    st, report = _reporter()
    report("retrieval", {"raw_evidence": [1, 2, 3, 4]})
    report("evidence_filter", {"filtered_evidence": [1, 2]})
    assert "4" in st.lines[0]
    assert "2" in st.lines[1]


def test_reporter_tolerates_missing_keys():
    """Node updates are partial by nature; a missing key must not raise."""
    st, report = _reporter()
    for node in ("router", "retrieval", "evidence_filter", "answer_generator",
                 "reviewer", "safety", "patient_summary"):
        report(node, {})
    assert len(st.lines) == 7


def test_reporter_ignores_unknown_nodes():
    st, report = _reporter()
    report("some_future_node", {})
    assert st.lines == []

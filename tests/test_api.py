"""
Unit tests for the FastAPI service layer (api/main.py).

The pipeline and the DB are monkeypatched, so these run with no API keys, no
data/, and no network — the point is to test the HTTP contract (status codes,
validation, response shape), not to re-test the pipeline.

Run with:  python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from models.schemas import Evidence

client = TestClient(app)


def _fake_final(**over):
    """A minimal PipelineState-shaped dict, as answer_query would return."""
    out = {
        "query_id": "qid-123",
        "answer": "Metformin carries a boxed warning for lactic acidosis [1].",
        "final_answer": "Metformin ... [1]\n\n---\nnot medical advice",
        "citations": ["[1] FDA Label — Metformin, Warnings"],
        "filtered_evidence": [
            Evidence(source="fda_label", title="metformin — warnings",
                     text="Lactic acidosis...", score=0.93,
                     citation="FDA Label — Metformin, Warnings"),
        ],
        "disclaimers": ["not medical advice"],
        "intent": "medication_info",
        "iteration": 1,
        "review_passed": True,
        "sub_queries": [],
        "trace": ["Router: intent=medication_info"],
        "metrics": {"llm_calls": 3, "total_tokens": 11859,
                    # floats on purpose: the pipeline records 0.1ms precision, and a
        # tidy int here previously hid a schema mismatch.
        "cost_usd": 0.0451, "total_latency_ms": 31200.7},
    }
    out.update(over)
    return out


class _FakeDB:
    """Stands in for MedDocDB."""
    def __init__(self, patients=None, raise_on=None):
        self._patients = patients if patients is not None else [{
            "patient_id": "p1", "name": "Cali421 Abernathy746", "age": 90, "gender": "male",
            "conditions_json": [{"display": "Hypertension"}],
            "medications_json": [{"display": "metformin", "status": "active"},
                                 {"display": "aspirin", "status": "stopped"}],
            "labs_json": [{"display": "LDL", "value": 163, "is_abnormal": True},
                          {"display": "Na", "value": 140, "is_abnormal": False}],
            "summary_md": "# Care Summary",
        }]
        self._raise_on = raise_on or set()
        self.saved = []

    def _boom(self, name):
        if name in self._raise_on:
            raise RuntimeError("db is down")

    def count_patients(self):
        self._boom("count_patients"); return len(self._patients)

    def list_patients(self, limit=100):
        """Mirrors the REAL method, which selects ONLY the roster columns.

        The fake originally returned the full patient dict — richer than reality —
        which hid a bug where the endpoint computed counts from clinical JSON that
        list_patients never returns (they were silently always 0). A fake that is
        more generous than the thing it replaces tests nothing.
        """
        self._boom("list_patients")
        return [{"patient_id": p["patient_id"], "name": p["name"], "age": p["age"],
                 "gender": p["gender"], "has_summary": bool(p.get("summary_md"))}
                for p in self._patients[:limit]]

    def get_patient(self, pid):
        self._boom("get_patient")
        return next((p for p in self._patients if p["patient_id"] == pid), None)

    def save_feedback(self, query_id, rating, comment="", **kw):
        self._boom("save_feedback")
        if rating not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        self.saved.append((query_id, rating, comment))
        self.saved_kwargs = kw

    def feedback_counts(self):
        self._boom("feedback_counts"); return {"up": 3, "down": 1}

    def get_query_metrics(self, query_id):
        self._boom("get_query_metrics")
        if query_id != "qid-123":
            return None
        return {"query_id": query_id, "query": "what should I watch for?",
                "intent": "medication_info", "user_role": "care_manager",
                "patient_id": "p1", "session_id": "s1",
                "review_passed": 1, "n_evidence": 11}


@pytest.fixture
def fake_db(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr("api.main._db", lambda: db)
    return db


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = {}

    def _answer_query(question, patient_context=None, user_role="anonymous",
                      max_iterations=2, conversation_history=None):
        calls.update(question=question, patient_context=patient_context,
                     user_role=user_role, max_iterations=max_iterations,
                     conversation_history=conversation_history)
        return _fake_final()

    import graph.pipeline as gp
    monkeypatch.setattr(gp, "answer_query", _answer_query)
    return calls


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_reports_degraded_instead_of_failing(self, monkeypatch):
        """A dependency outage must be visible, not a 500 — an orchestrator needs
        to tell 'process up, Qdrant down' from 'process dead'."""
        monkeypatch.setattr("api.main._db", lambda: _FakeDB(raise_on={"count_patients"}))
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"
        assert "error" in str(r.json()["checks"]["database"])

    def test_reports_per_dependency_checks(self, fake_db):
        body = client.get("/health").json()
        for key in ("anthropic_key", "openai_key", "database", "vector_store"):
            assert key in body["checks"]

    def test_every_response_carries_a_timing_header(self, fake_db):
        assert "X-Process-Time-Ms" in client.get("/health").headers


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

class TestPatients:

    def test_roster_is_thin_and_reports_summary_availability(self, fake_db):
        """The roster must expose only what list_patients actually selects."""
        p = client.get("/patients").json()[0]
        assert p["name"] == "Cali421 Abernathy746"
        assert p["has_summary"] is True
        # Counts belong to the detail endpoint; publishing them here would be a lie.
        assert "n_active_medications" not in p

    def test_detail_counts_only_active_meds_and_abnormal_labs(self, fake_db):
        b = client.get("/patients/p1").json()
        assert b["n_active_medications"] == 1     # the 'stopped' one is excluded
        assert b["n_abnormal_labs"] == 1          # the normal lab is excluded
        assert b["n_conditions"] == 1

    def test_detail_returns_the_precomputed_summary(self, fake_db):
        body = client.get("/patients/p1").json()
        assert body["summary_md"] == "# Care Summary"
        assert [m["display"] for m in body["medications"]] == ["metformin"]

    def test_unknown_patient_is_404(self, fake_db):
        assert client.get("/patients/nope").status_code == 404

    def test_db_outage_is_503_not_500(self, monkeypatch):
        monkeypatch.setattr("api.main._db", lambda: _FakeDB(raise_on={"list_patients"}))
        assert client.get("/patients").status_code == 503


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQuery:

    def test_happy_path_shape(self, fake_db, fake_pipeline):
        r = client.post("/query", json={"question": "What are metformin's warnings?"})
        assert r.status_code == 200
        b = r.json()
        assert b["query_id"] == "qid-123"
        assert b["intent"] == "medication_info"
        assert b["review_passed"] is True
        assert b["evidence"][0]["source"] == "fda_label"
        assert b["metrics"]["cost_usd"] == 0.0451
        assert b["metrics"]["total_latency_ms"] == 31200.7   # fractional ms preserved

    def test_answer_and_final_answer_are_both_returned(self, fake_db, fake_pipeline):
        """Scoring clients want `answer`; user-facing clients want `final_answer`."""
        b = client.post("/query", json={"question": "q"}).json()
        assert "not medical advice" in b["final_answer"]
        assert "not medical advice" not in b["answer"]

    def test_patient_id_is_resolved_to_full_context(self, fake_db, fake_pipeline):
        client.post("/query", json={"question": "q", "patient_id": "p1"})
        assert fake_pipeline["patient_context"]["name"] == "Cali421 Abernathy746"

    def test_unknown_patient_is_404_before_any_llm_call(self, fake_db, fake_pipeline):
        r = client.post("/query", json={"question": "q", "patient_id": "ghost"})
        assert r.status_code == 404
        assert fake_pipeline == {}            # pipeline never invoked — no wasted spend

    def test_conversation_history_is_forwarded(self, fake_db, fake_pipeline):
        client.post("/query", json={
            "question": "what about its side effects?",
            "conversation_history": [{"role": "user", "content": "tell me about metformin"}],
        })
        assert fake_pipeline["conversation_history"][0]["content"] == "tell me about metformin"

    def test_pipeline_failure_is_502(self, fake_db, monkeypatch):
        import graph.pipeline as gp
        def boom(*a, **k):
            raise RuntimeError("anthropic timeout")
        monkeypatch.setattr(gp, "answer_query", boom)
        assert client.post("/query", json={"question": "q"}).status_code == 502

    def test_null_cost_is_preserved_not_zeroed(self, fake_db, monkeypatch):
        """An unpriced model must surface as null, never a misleading 0.00."""
        import graph.pipeline as gp
        monkeypatch.setattr(gp, "answer_query",
                            lambda *a, **k: _fake_final(metrics={"llm_calls": 1,
                                                                 "total_tokens": 10,
                                                                 "cost_usd": None,
                                                                 "total_latency_ms": 5.4}))
        assert client.post("/query", json={"question": "q"}).json()["metrics"]["cost_usd"] is None


class TestQueryValidation:

    def test_empty_question_rejected(self, fake_db):
        assert client.post("/query", json={"question": ""}).status_code == 422

    def test_missing_question_rejected(self, fake_db):
        assert client.post("/query", json={}).status_code == 422

    def test_overlong_question_rejected(self, fake_db):
        assert client.post("/query", json={"question": "x" * 2001}).status_code == 422

    def test_invalid_role_rejected(self, fake_db):
        r = client.post("/query", json={"question": "q", "user_role": "admin"})
        assert r.status_code == 422

    def test_out_of_range_iterations_rejected(self, fake_db):
        assert client.post("/query", json={"question": "q", "max_iterations": 99}).status_code == 422


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class TestFeedback:

    def test_records_a_rating(self, fake_db):
        r = client.post("/feedback", json={"query_id": "qid-1", "rating": "down",
                                           "comment": "cited the wrong drug"})
        assert r.status_code == 201
        assert fake_db.saved == [("qid-1", "down", "cited the wrong drug")]

    def test_invalid_rating_rejected_by_schema(self, fake_db):
        assert client.post("/feedback", json={"query_id": "q", "rating": "meh"}).status_code == 422

    def test_context_is_backfilled_from_query_metrics(self, fake_db):
        """An HTTP client sends only a query_id.

        Without this join the feedback row would have no question text, and the
        eval exporter could not build a candidate from it — which is the entire
        point of collecting the rating.
        """
        client.post("/feedback", json={"query_id": "qid-123", "rating": "down"})
        kw = fake_db.saved_kwargs
        assert kw["question"] == "what should I watch for?"
        assert kw["intent"] == "medication_info"
        assert kw["patient_id"] == "p1"
        assert kw["n_evidence"] == 11

    def test_rating_is_still_recorded_when_no_metrics_row_exists(self, fake_db):
        """Metrics may be disabled (MEDDOCAI_METRICS=0); the verdict still counts."""
        r = client.post("/feedback", json={"query_id": "unknown-id", "rating": "up"})
        assert r.status_code == 201
        assert fake_db.saved[0][0] == "unknown-id"

    def test_stats_include_a_positive_rate(self, fake_db):
        b = client.get("/feedback/stats").json()
        assert b["up"] == 3 and b["down"] == 1
        assert b["positive_rate"] == 0.75

    def test_db_outage_is_503(self, monkeypatch):
        monkeypatch.setattr("api.main._db", lambda: _FakeDB(raise_on={"save_feedback"}))
        r = client.post("/feedback", json={"query_id": "q", "rating": "up"})
        assert r.status_code == 503


class TestOpenAPI:

    def test_schema_is_generated_and_documents_every_endpoint(self):
        """The service should be self-describing — /docs is part of the deliverable."""
        spec = client.get("/openapi.json").json()
        for path in ("/query", "/patients", "/feedback", "/health"):
            assert path in spec["paths"]
        assert spec["info"]["title"] == "MedDocAI API"

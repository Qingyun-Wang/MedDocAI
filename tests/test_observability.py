"""
Unit tests for agents/observability.py and agents/tracing.py.

These must run WITHOUT data/, without API keys, and without torch — they are
part of the free CI gate.

Run with:  python -m pytest tests/test_observability.py -v
"""

from __future__ import annotations

import inspect
import sqlite3
import threading

import pytest

from agents import observability as obs
from agents import tracing


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:

    def test_unknown_model_returns_none_not_zero(self):
        """None (not 0.0) so the UI shows 'n/a' rather than implying it was free."""
        assert obs.estimate_cost_usd("some-unpriced-model", 1000, 1000) is None

    def test_known_model_computes_from_price_table(self):
        # claude-sonnet-4-5: $3/MTok in, $15/MTok out
        cost = obs.estimate_cost_usd("claude-sonnet-4-5-20250929", 1_000_000, 1_000_000)
        assert cost == pytest.approx(18.00)

    def test_cache_tokens_priced_separately(self):
        # cache read $0.30/MTok, cache write $3.75/MTok
        cost = obs.estimate_cost_usd(
            "claude-sonnet-4-5-20250929", 0, 0,
            cache_read_tokens=1_000_000, cache_write_tokens=1_000_000,
        )
        assert cost == pytest.approx(4.05)

    def test_pinned_model_is_priced(self):
        """The model agents/llm.py actually uses must have a price entry."""
        from agents.llm import CLAUDE_MODEL
        assert CLAUDE_MODEL in obs.MODEL_PRICING_USD_PER_MTOK


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

class TestAccumulation:

    def test_sums_tokens_and_counts_calls(self):
        obs.start_query("q1")
        obs.record_llm_call(model="claude-sonnet-4-5", caller="router",
                            input_tokens=100, output_tokens=20, latency_ms=500)
        obs.record_llm_call(model="claude-sonnet-4-5", caller="reviewer",
                            input_tokens=300, output_tokens=50, latency_ms=700)
        m = obs.finish_query("q1")
        assert m["llm_calls"] == 2
        assert m["input_tokens"] == 400
        assert m["output_tokens"] == 70
        assert m["total_tokens"] == 470
        assert m["cost_usd"] > 0
        assert m["query_id"] == "q1"

    def test_unpriced_call_makes_total_cost_none(self):
        """A partial sum would understate the true cost — so the total is None."""
        obs.start_query("q2")
        obs.record_llm_call(model="claude-sonnet-4-5", input_tokens=10, output_tokens=10)
        obs.record_llm_call(model="mystery-model", input_tokens=10, output_tokens=10)
        assert obs.finish_query("q2")["cost_usd"] is None

    def test_record_outside_query_is_noop(self):
        """Unit tests that call agents directly must not blow up."""
        obs.finish_query()          # ensure cleared
        obs.record_llm_call(model="claude-sonnet-4-5", input_tokens=1, output_tokens=1)
        obs.record_node("router", 5.0)   # no exception == pass

    def test_node_latency_accumulates_across_retries(self):
        obs.start_query("q3")
        obs.record_node("answer_generator", 100.0)
        obs.record_node("answer_generator", 50.0)   # corrective-loop retry
        m = obs.finish_query("q3")
        assert m["by_node"]["answer_generator"] == pytest.approx(150.0)

    def test_threads_are_isolated(self):
        """ContextVars must not leak between Streamlit ScriptRunner threads."""
        results = {}

        def worker(name, n_calls):
            obs.start_query(name)
            for _ in range(n_calls):
                obs.record_llm_call(model="claude-sonnet-4-5",
                                    input_tokens=10, output_tokens=1)
            results[name] = obs.finish_query(name)["llm_calls"]

        t1 = threading.Thread(target=worker, args=("a", 2))
        t2 = threading.Thread(target=worker, args=("b", 5))
        t1.start(); t2.start(); t1.join(); t2.join()
        assert results == {"a": 2, "b": 5}


# ---------------------------------------------------------------------------
# Node instrumentation
# ---------------------------------------------------------------------------

class TestInstrumentNode:

    def test_preserves_return_value(self):
        wrapped = obs.instrument_node("demo", lambda state: {"answer": state["q"]})
        assert wrapped({"q": "hi"}) == {"answer": "hi"}

    def test_records_latency(self):
        obs.start_query("q4")
        obs.instrument_node("demo", lambda state: {})({})
        assert "demo" in obs.finish_query("q4")["by_node"]

    def test_signature_has_exactly_one_parameter(self):
        """LangGraph inspects the signature to decide whether to inject `config`.

        A wrapper accepting (*args, **kwargs) would change that behaviour.
        """
        def node(state):
            return {}
        wrapped = obs.instrument_node("demo", node)
        assert len(inspect.signature(wrapped).parameters) == 1

    def test_latency_recorded_even_when_node_raises(self):
        def boom(state):
            raise ValueError("node failed")
        obs.start_query("q5")
        with pytest.raises(ValueError):
            obs.instrument_node("boom", boom)({})
        assert "boom" in obs.finish_query("q5")["by_node"]


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:

    def test_shows_cost_when_known(self):
        out = obs.format_summary({"total_latency_ms": 18400, "llm_calls": 3,
                                  "total_tokens": 12481, "cost_usd": 0.0421})
        assert "18.4s" in out and "3 LLM calls" in out and "12,481 tok" in out and "$0.0421" in out

    def test_shows_na_when_cost_unknown(self):
        out = obs.format_summary({"total_latency_ms": 1000, "llm_calls": 1,
                                  "total_tokens": 10, "cost_usd": None})
        assert "cost n/a" in out and "$" not in out

    def test_empty_metrics_is_empty_string(self):
        assert obs.format_summary({}) == ""


# ---------------------------------------------------------------------------
# Tracing guard — must be OFF unless explicitly enabled
# ---------------------------------------------------------------------------

class TestTracingGuard:

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2",
                    "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
            monkeypatch.delenv(var, raising=False)

    def test_disabled_when_nothing_set(self):
        assert tracing.tracing_enabled() is False

    def test_key_alone_is_not_enough(self, monkeypatch):
        """A stray key must never silently ship prompts to a SaaS."""
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_fake")
        assert tracing.tracing_enabled() is False

    def test_flag_alone_is_not_enough(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        assert tracing.tracing_enabled() is False

    def test_enabled_only_with_both(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
        assert tracing.tracing_enabled() is True

    def test_wrap_is_identity_when_disabled(self):
        sentinel = object()
        assert tracing.maybe_wrap_anthropic(sentinel) is sentinel

    def test_traced_is_identity_when_disabled(self):
        def fn():
            return 42
        assert tracing.traced("x")(fn) is fn


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestQueryMetricsPersistence:

    def test_writes_to_db_where_create_tables_was_never_called(self, tmp_path):
        """The deployed Space DB predates this table — first write must create it."""
        from ingestion.sqlite_loader import MedDocDB
        path = str(tmp_path / "fresh.db")
        MedDocDB(path).save_query_metrics(
            query_id="q-1", query="metformin?", intent="medication_info",
            user_role="anonymous", patient_id=None, iterations=1,
            review_passed=True, n_evidence=5,
            metrics={"llm_calls": 3, "input_tokens": 1200, "output_tokens": 400,
                     "total_tokens": 1600, "cost_usd": 0.0096,
                     "total_latency_ms": 18400.0, "by_node": {"router": 900.0},
                     "calls": [{"caller": "router"}]},
        )
        row = sqlite3.connect(path).execute(
            "SELECT query_id, intent, patient_id, llm_calls, total_tokens, cost_usd "
            "FROM query_metrics"
        ).fetchone()
        assert row == ("q-1", "medication_info", None, 3, 1600, 0.0096)

    def test_anonymous_queries_are_recorded(self, tmp_path):
        """The blob-based alternative dropped these entirely — regression guard."""
        from ingestion.sqlite_loader import MedDocDB
        path = str(tmp_path / "anon.db")
        db = MedDocDB(path)
        db.save_query_metrics(query_id="a1", metrics={}, patient_id=None)
        db.save_query_metrics(query_id="a2", metrics={}, patient_id=None)
        n = sqlite3.connect(path).execute(
            "SELECT COUNT(*) FROM query_metrics WHERE patient_id IS NULL"
        ).fetchone()[0]
        assert n == 2

    def test_null_cost_persists_as_null(self, tmp_path):
        from ingestion.sqlite_loader import MedDocDB
        path = str(tmp_path / "nullcost.db")
        MedDocDB(path).save_query_metrics(
            query_id="q-n", metrics={"cost_usd": None, "llm_calls": 1},
        )
        cost = sqlite3.connect(path).execute(
            "SELECT cost_usd FROM query_metrics WHERE query_id='q-n'"
        ).fetchone()[0]
        assert cost is None

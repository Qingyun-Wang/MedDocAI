"""
Unit tests for the CI quality gate in evaluation/run_eval.py.

Pure functions over synthetic summaries — no API keys, no data/, no network — so
these run in the free unit-test CI job and guard the gate that guards everything else.

Run with:  python -m pytest tests/test_eval_gate.py -v
"""

from __future__ import annotations

import pytest

from evaluation.run_eval import _parse_thresholds, check_gate


def _rows(scores: list, metric: str = "faithfulness", error: str | None = None) -> list[dict]:
    """Build synthetic result rows carrying one metric each."""
    return [
        {"id": f"q{i}", "category": "med_anon", "error": error,
         "scores": {metric: s}}
        for i, s in enumerate(scores)
    ]


def _summary(value, metric: str = "faithfulness") -> dict:
    return {"overall": {metric: value}}


class TestPassingRuns:

    def test_all_green_passes(self):
        rows = _rows([0.9] * 10)
        passed, reasons = check_gate(_summary(0.9), rows, {"faithfulness": 0.85})
        assert passed and reasons == []

    def test_value_exactly_at_floor_passes(self):
        rows = _rows([0.85] * 10)
        passed, _ = check_gate(_summary(0.85), rows, {"faithfulness": 0.85})
        assert passed

    def test_multiple_metrics_all_above(self):
        rows = [{"id": "q", "category": "c", "error": None,
                 "scores": {"faithfulness": 0.9, "context_recall": 0.9}}]
        passed, _ = check_gate(
            {"overall": {"faithfulness": 0.9, "context_recall": 0.9}},
            rows, {"faithfulness": 0.85, "context_recall": 0.82},
        )
        assert passed


class TestQualityFailures:

    def test_below_floor_fails(self):
        rows = _rows([0.5] * 10)
        passed, reasons = check_gate(_summary(0.5), rows, {"faithfulness": 0.85})
        assert not passed
        assert "faithfulness" in reasons[0] and "0.85" in reasons[0]

    def test_just_below_floor_fails(self):
        rows = _rows([0.849] * 10)
        passed, _ = check_gate(_summary(0.8499), rows, {"faithfulness": 0.85})
        assert not passed

    def test_one_bad_metric_fails_the_whole_gate(self):
        rows = [{"id": "q", "category": "c", "error": None,
                 "scores": {"faithfulness": 0.95, "context_recall": 0.10}}]
        passed, reasons = check_gate(
            {"overall": {"faithfulness": 0.95, "context_recall": 0.10}},
            rows, {"faithfulness": 0.85, "context_recall": 0.82},
        )
        assert not passed
        assert any("context_recall" in r for r in reasons)


class TestFailClosed:
    """The rules that stop a broken run from looking like a good one."""

    def test_none_aggregate_fails(self):
        """A missing score must never be treated as a pass."""
        rows = _rows([None] * 10)
        passed, reasons = check_gate(_summary(None), rows, {"faithfulness": 0.85})
        assert not passed
        assert "fail-closed" in reasons[0]

    def test_mostly_errored_rows_fail_despite_high_mean(self):
        """8/10 rows errored; the 2 survivors average 1.0.

        Averaging drops the error strings, so the summary would read 1.0 and sail
        through a naive threshold check. min_scored_frac is what catches it.
        """
        scores = ["error: rate limit"] * 8 + [1.0, 1.0]
        rows = _rows(scores)
        passed, reasons = check_gate(
            _summary(1.0), rows, {"faithfulness": 0.85}, min_scored_frac=0.8
        )
        assert not passed
        assert "only 2/10 rows scored" in reasons[0]

    def test_enough_scored_rows_still_passes(self):
        scores = ["error: blip"] + [0.9] * 9      # 90% scored, above the 80% floor
        rows = _rows(scores)
        passed, _ = check_gate(
            _summary(0.9), rows, {"faithfulness": 0.85}, min_scored_frac=0.8
        )
        assert passed

    def test_pipeline_error_fails_by_default(self):
        rows = _rows([0.9] * 10)
        rows[0]["error"] = "connection reset"
        passed, reasons = check_gate(_summary(0.9), rows, {"faithfulness": 0.85})
        assert not passed
        assert "pipeline error" in reasons[0]

    def test_pipeline_errors_allowed_when_budgeted(self):
        rows = _rows([0.9] * 10)
        rows[0]["error"] = "connection reset"
        passed, _ = check_gate(
            _summary(0.9), rows, {"faithfulness": 0.85}, max_pipeline_errors=1
        )
        assert passed


class TestThresholdParsing:

    def test_parses_repeated_pairs(self):
        assert _parse_thresholds(["faithfulness=0.85", "context_recall=0.8"]) == {
            "faithfulness": 0.85, "context_recall": 0.8,
        }

    def test_empty_means_gate_disabled(self):
        assert _parse_thresholds(None) == {}
        assert _parse_thresholds([]) == {}

    def test_rejects_unknown_metric(self):
        with pytest.raises(SystemExit):
            _parse_thresholds(["made_up_metric=0.5"])

    def test_rejects_malformed_pair(self):
        with pytest.raises(SystemExit):
            _parse_thresholds(["faithfulness"])

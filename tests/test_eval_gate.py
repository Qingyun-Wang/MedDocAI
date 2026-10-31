"""
Unit tests for the CI quality gate in evaluation/run_eval.py.

Pure functions over synthetic summaries — no API keys, no data/, no network — so
these run in the free unit-test CI job and guard the gate that guards everything else.

Run with:  python -m pytest tests/test_eval_gate.py -v
"""

from __future__ import annotations

import pytest

from evaluation.run_eval import (_parse_category_thresholds, _parse_thresholds,
                                 check_gate)


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


# ---------------------------------------------------------------------------
# Per-category floors
# ---------------------------------------------------------------------------

class TestPerCategoryGate:
    """One aggregate cannot gate a heterogeneous question set.

    The categories legitimately score very differently on answer_relevancy —
    med_anon ~0.93, care_mgr ~0.62 — because RAGAS generates questions from the
    answer and a deliberately broad care-manager ask produces a broad answer whose
    generated questions sit far from the vague original.

    Averaging them fails two ways: widen the gate and it reddens on a healthy run,
    or lower the floor to compensate and med_anon could fall 0.93 -> 0.80 unnoticed.
    """

    def _rows(self, spec):
        """spec: {category: (n_rows, score)} -> rows carrying that score."""
        out = []
        for cat, (n, score) in spec.items():
            for i in range(n):
                out.append({"id": f"{cat}_{i}", "category": cat,
                            "scores": {"answer_relevancy": score}})
        return out

    def _summary(self, spec):
        rows = self._rows(spec)
        overall = sum(r["scores"]["answer_relevancy"] for r in rows) / len(rows)
        return {"overall": {"answer_relevancy": round(overall, 4)},
                "by_category": {c: {"answer_relevancy": s} for c, (n, s) in spec.items()}}

    def test_the_problem_this_solves(self):
        """A single aggregate floor reddens on a run where nothing regressed."""
        spec = {"med_anon": (10, 0.93), "care_mgr": (8, 0.62), "diffuse": (8, 0.75)}
        ok, why = check_gate(self._summary(spec), self._rows(spec),
                             {"answer_relevancy": 0.82})
        assert not ok and "answer_relevancy" in why[0], \
            "aggregate floor should fail here — that is the motivation"

    def test_per_category_floors_pass_the_same_healthy_run(self):
        spec = {"med_anon": (10, 0.93), "care_mgr": (8, 0.62), "diffuse": (8, 0.75)}
        ok, why = check_gate(
            self._summary(spec), self._rows(spec), {},
            category_thresholds={"med_anon": {"answer_relevancy": 0.85},
                                 "care_mgr": {"answer_relevancy": 0.50},
                                 "diffuse":  {"answer_relevancy": 0.65}})
        assert ok, why

    def test_a_strong_category_regressing_is_still_caught(self):
        """The point of per-category: med_anon falling 0.93 -> 0.80 must FAIL, even
        though a lowered aggregate floor would have let it through."""
        spec = {"med_anon": (10, 0.80), "care_mgr": (8, 0.62)}
        ok, why = check_gate(
            self._summary(spec), self._rows(spec), {},
            category_thresholds={"med_anon": {"answer_relevancy": 0.85}})
        assert not ok
        assert any("med_anon/answer_relevancy" in r for r in why)

    def test_failure_names_the_category(self):
        """'care_mgr relevancy dropped' is a lead; 'overall dipped' is not."""
        spec = {"care_mgr": (4, 0.40)}
        _, why = check_gate(self._summary(spec), self._rows(spec), {},
                            category_thresholds={"care_mgr": {"answer_relevancy": 0.60}})
        assert why and why[0].startswith("care_mgr/answer_relevancy")

    def test_configured_category_absent_from_run_fails_closed(self):
        """Silently skipping it is how a gate stops guarding something while green."""
        spec = {"med_anon": (4, 0.93)}
        ok, why = check_gate(self._summary(spec), self._rows(spec), {},
                             category_thresholds={"care_mgr": {"answer_relevancy": 0.5}})
        assert not ok
        assert any("no rows in this run" in r for r in why)

    def test_too_few_rows_fails_closed_rather_than_reporting_noise(self):
        """A 2-question category moves 0.25 on one bad answer."""
        spec = {"care_mgr": (2, 0.95)}
        ok, why = check_gate(self._summary(spec), self._rows(spec), {},
                             category_thresholds={"care_mgr": {"answer_relevancy": 0.5}},
                             min_category_rows=3)
        assert not ok, "a 2-row category is not readable even when it scores well"
        assert any("readable" in r for r in why)

    def test_unscored_category_rows_fail_closed(self):
        """Judge errors inside one category must not pass on the survivors."""
        rows = [{"id": "c0", "category": "care_mgr", "scores": {"answer_relevancy": 0.9}},
                {"id": "c1", "category": "care_mgr", "scores": {"answer_relevancy": None}},
                {"id": "c2", "category": "care_mgr",
                 "scores": {"answer_relevancy": "error: judge"}},
                {"id": "c3", "category": "care_mgr", "scores": {"answer_relevancy": None}}]
        summary = {"overall": {"answer_relevancy": 0.9},
                   "by_category": {"care_mgr": {"answer_relevancy": 0.9}}}
        ok, why = check_gate(summary, rows, {},
                             category_thresholds={"care_mgr": {"answer_relevancy": 0.5}})
        assert not ok
        assert any("rows scored" in r for r in why)

    def test_overall_and_category_floors_compose(self):
        spec = {"med_anon": (10, 0.93), "care_mgr": (4, 0.62)}
        ok, why = check_gate(
            self._summary(spec), self._rows(spec),
            {"faithfulness": 0.85},        # missing from summary -> fail-closed
            category_thresholds={"care_mgr": {"answer_relevancy": 0.50}})
        assert not ok
        assert any("faithfulness" in r for r in why)


class TestParseCategoryThresholds:
    def test_parses_category_metric_float(self):
        out = _parse_category_thresholds(["care_mgr:faithfulness=0.90",
                                          "care_mgr:answer_relevancy=0.50",
                                          "med_anon:faithfulness=0.95"])
        assert out == {"care_mgr": {"faithfulness": 0.90, "answer_relevancy": 0.50},
                       "med_anon": {"faithfulness": 0.95}}

    def test_empty_is_empty(self):
        assert _parse_category_thresholds(None) == {}

    @pytest.mark.parametrize("bad", ["care_mgr", "care_mgr:faithfulness",
                                     "faithfulness=0.9", "care_mgr:bogus=0.9"])
    def test_malformed_raises(self, bad):
        with pytest.raises(SystemExit):
            _parse_category_thresholds([bad])

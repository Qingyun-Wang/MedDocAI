"""
Tests for tools/reranker.py and the reranked Evidence Filter.

The cross-encoder tests need sentence-transformers + the model; they're marked
@pytest.mark.rerank and skipped if unavailable. The fallback-path tests run
always (they verify the pipeline still works WITHOUT the model).

Run with:  python -m pytest tests/test_reranker.py -v
"""

import pytest

from models.schemas import Evidence
from tools import reranker
from agents.evidence_filter import evidence_filter_node, _dedup
from agents.state import new_state


def _ev(title, text, score=0.7, source="fda_label", **md):
    return Evidence(source=source, title=title, text=text, score=score,
                    metadata=md, citation=title)


# ---------------------------------------------------------------------------
# Reranker — availability + core behaviour (needs model)
# ---------------------------------------------------------------------------

@pytest.mark.rerank
@pytest.mark.skipif(not reranker.is_available(), reason="cross-encoder not available")
class TestReranker:
    def test_orders_by_relevance(self):
        query = "metformin risks in kidney disease"
        cands = [
            _ev("irrelevant", "Aspirin is a common pain reliever."),
            _ev("relevant", "Metformin is contraindicated in severe renal "
                            "impairment and risks lactic acidosis in kidney disease."),
            _ev("weak", "Metformin treats type 2 diabetes."),
        ]
        ranked = reranker.rerank(query, cands)
        assert ranked[0].title == "relevant"
        assert ranked[-1].title == "irrelevant"

    def test_scores_are_normalised_0_1(self):
        ranked = reranker.rerank("diabetes drug", [_ev("a", "metformin for diabetes")])
        assert 0.0 <= ranked[0].score <= 1.0

    def test_preserves_bi_encoder_score(self):
        cands = [_ev("a", "metformin", score=0.71)]
        ranked = reranker.rerank("metformin", cands)
        assert ranked[0].metadata.get("bi_encoder_score") == 0.71
        assert ranked[0].metadata.get("reranked") is True

    def test_top_k_caps(self):
        cands = [_ev(f"c{i}", f"text about topic {i}") for i in range(10)]
        ranked = reranker.rerank("topic", cands, top_k=3)
        assert len(ranked) == 3

    def test_scores_everything_including_nonscored(self):
        """Cross-encoder gives a real score even to items that had score=None."""
        cands = [
            _ev("vec", "metformin contraindications", score=0.7),
            _ev("api", "metformin recall notice", score=None, source="openfda_api"),
        ]
        ranked = reranker.rerank("metformin recall", cands)
        assert all(e.score is not None for e in ranked)


# ---------------------------------------------------------------------------
# Reranker — empty / edge cases (no model needed)
# ---------------------------------------------------------------------------

class TestRerankerEdgeCases:
    def test_empty_returns_empty(self):
        assert reranker.rerank("q", []) == []

    def test_is_available_returns_bool(self):
        assert isinstance(reranker.is_available(), bool)


# ---------------------------------------------------------------------------
# Dedup (deterministic, no model)
# ---------------------------------------------------------------------------

class TestDedup:
    def test_exact_dedup(self):
        items = [
            _ev("a", "x", drug_name="metformin", section_type="warnings"),
            _ev("b", "y", drug_name="metformin", section_type="warnings"),
        ]
        assert len(_dedup(items)) == 1

    def test_near_dup_dedup(self):
        # Long shared text + one extra token => Jaccard > 0.85 => treated as duplicate
        txt = ("metformin is contraindicated in patients with severe renal "
               "impairment defined as estimated glomerular filtration rate below "
               "thirty milliliters per minute per body surface area")
        items = [
            _ev("a", txt, drug_name="x", section_type="s1"),
            _ev("b", txt + " value", drug_name="y", section_type="s2"),
        ]
        assert len(_dedup(items)) == 1

    def test_distinct_kept(self):
        items = [
            _ev("a", "metformin contraindications renal", drug_name="m", section_type="c"),
            _ev("b", "aspirin bleeding risk warning", drug_name="a", section_type="w"),
        ]
        assert len(_dedup(items)) == 2

    def test_cross_source_containment_dedup(self):
        """The same MedlinePlus article reached two ways — a truncated Connect summary
        (source=medlineplus_connect) and the full topic from the Qdrant index
        (source=medlineplus) — must collapse to one, even though the length gap makes
        Jaccard low. Regression for the CABG double-source bug."""
        full = ("coronary artery bypass surgery improves blood flow to the heart muscle "
                "by using a healthy blood vessel from another part of your body to route "
                "around a blocked artery this restores normal circulation reduces chest "
                "pain and lowers the risk of heart attack in coronary artery disease")
        truncated = ("coronary artery bypass surgery improves blood flow to the heart "
                     "muscle by using a healthy blood vessel")  # subset of `full`
        items = [
            _ev("Coronary Artery Bypass Surgery", full, source="medlineplus"),
            _ev("Coronary Artery Bypass Surgery", truncated, source="medlineplus_connect"),
        ]
        kept = _dedup(items)
        assert len(kept) == 1
        # keeps the fuller (first-seen) version
        assert kept[0].source == "medlineplus" and kept[0].text == full

    def test_short_generic_not_over_deduped(self):
        """A short generic line must NOT be treated as a duplicate of an unrelated long
        passage just because its few tokens happen to appear in it (min-token guard)."""
        items = [
            _ev("recall", "no current recalls found", source="openfda_api"),
            _ev("label", "metformin is contraindicated in severe renal impairment and "
                         "may cause lactic acidosis in patients with kidney disease",
                source="fda_label"),
        ]
        assert len(_dedup(items)) == 2


# ---------------------------------------------------------------------------
# Evidence Filter — both paths
# ---------------------------------------------------------------------------

class TestEvidenceFilterNode:
    def test_caps_at_max(self):
        from agents.evidence_filter import MAX_EVIDENCE
        state = new_state("metformin warnings")
        state["raw_evidence"] = [
            _ev(f"d{i}", f"metformin warning variant number {i} with distinct text",
                score=0.8, drug_name=f"drug{i}", section_type="warnings")
            for i in range(20)
        ]
        out = evidence_filter_node(state)
        assert len(out["filtered_evidence"]) <= MAX_EVIDENCE

    def test_dedup_runs_before_cap(self):
        # 5 exact duplicates + 1 distinct, BOTH relevant to the query so the
        # rerank relevance floor doesn't drop either -> tests dedup specifically.
        state = new_state("metformin warnings and side effects")
        dupes = [_ev("d", "metformin warnings include lactic acidosis risk",
                     drug_name="metformin", section_type="warnings")
                 for _ in range(5)]
        distinct = [_ev("x", "metformin common side effects include nausea and diarrhea",
                        drug_name="metformin", section_type="adverse_reactions")]
        state["raw_evidence"] = dupes + distinct
        out = evidence_filter_node(state)
        # 5 exact dupes collapse to 1, plus the 1 distinct = 2
        assert len(out["filtered_evidence"]) == 2

    def test_empty_evidence(self):
        state = new_state("q")
        state["raw_evidence"] = []
        out = evidence_filter_node(state)
        assert out["filtered_evidence"] == []

    def test_fallback_path_when_reranker_unavailable(self, monkeypatch):
        """Force reranker unavailable -> bi-encoder fallback path runs."""
        import agents.evidence_filter as ef
        monkeypatch.setattr(ef.reranker, "is_available", lambda: False)
        state = new_state("metformin")
        state["raw_evidence"] = [
            _ev("high", "metformin renal", score=0.9, drug_name="a", section_type="c"),
            _ev("low", "unrelated text", score=0.1, drug_name="b", section_type="w"),
        ]
        out = evidence_filter_node(state)
        titles = [e.title for e in out["filtered_evidence"]]
        # low-score item dropped by bi-encoder threshold
        assert "high" in titles
        assert "low" not in titles

    def test_trace_records_method(self):
        state = new_state("metformin warnings")
        state["raw_evidence"] = [
            _ev("a", "metformin contraindications", score=0.8,
                drug_name="m", section_type="c")
        ]
        out = evidence_filter_node(state)
        assert any("Filter:" in t for t in out["trace"])

"""
Unit tests for R9 fan-out retrieval (diffuse multi-entity queries).

Covers the three moving parts, all mocked — no API keys, no data/, no network:
  - agents/router.py          : the deterministic gate + sub-query synthesis
  - agents/retrieval.py       : per-sub-query retrieval and sub_query tagging
  - agents/evidence_filter.py : per-sub-query rerank + round-robin merge

Run with:  python -m pytest tests/test_fanout.py -v
"""

from __future__ import annotations

from agents import evidence_filter as ef_mod
from agents import retrieval as ret_mod
from agents import router as router_mod
from agents.router import (
    MAX_SUB_QUERIES,
    _active_meds,
    _fallback_sub_queries,
    should_fan_out,
)
from agents.state import new_state
from models.schemas import Evidence


def _ev(text="some text", score=None, **md):
    return Evidence(source="fda_label", title="t", text=text, score=score, metadata=md)


def _ctx(*med_names, inactive=()):
    """A patient context with the given ACTIVE meds (plus optional inactive ones)."""
    meds = [{"display": m, "status": "active", "rxnorm_code": "1"} for m in med_names]
    meds += [{"display": m, "status": "stopped", "rxnorm_code": "2"} for m in inactive]
    return {"name": "Test Patient", "age": 70, "gender": "male",
            "medications_json": meds, "conditions_json": [], "labs_json": []}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

class TestGate:

    def test_fires_for_diffuse_patient_medication_question(self):
        st = new_state("what should I watch for?", _ctx("metformin", "clopidogrel"), "care_manager")
        assert should_fan_out(st, "medication_info", None) is True

    def test_does_not_fire_when_a_specific_drug_is_named(self):
        """A named drug means the question is not diffuse — the normal path is better."""
        st = new_state("is metformin safe?", _ctx("metformin", "clopidogrel"), "care_manager")
        assert should_fan_out(st, "medication_info", "metformin") is False

    def test_does_not_fire_without_a_patient(self):
        st = new_state("what are common side effects?", None, "anonymous")
        assert should_fan_out(st, "medication_info", None) is False

    def test_does_not_fire_with_a_single_active_med(self):
        st = new_state("what should I watch for?", _ctx("metformin"), "patient")
        assert should_fan_out(st, "medication_info", None) is False

    def test_does_not_fire_for_non_clinical_intent(self):
        st = new_state("income limit in Texas?", _ctx("metformin", "clopidogrel"), "care_manager")
        assert should_fan_out(st, "policy_eligibility", None) is False

    def test_inactive_meds_do_not_count(self):
        st = new_state("what should I watch for?",
                       _ctx("metformin", inactive=("aspirin", "ibuprofen")), "patient")
        assert _active_meds(st["patient_context"]) == ["metformin"]
        assert should_fan_out(st, "medication_info", None) is False


class TestFallbackSubQueries:

    def test_one_query_per_active_med(self):
        st = new_state("what side effects?", _ctx("metformin", "clopidogrel"), "care_manager")
        subs = _fallback_sub_queries(st)
        assert len(subs) == 2
        assert subs[0].startswith("metformin") and subs[1].startswith("clopidogrel")

    def test_interaction_questions_get_interaction_phrasing(self):
        st = new_state("any overlapping risks or interactions?",
                       _ctx("metformin", "captopril"), "care_manager")
        assert all("interactions" in q for q in _fallback_sub_queries(st))

    def test_side_effect_questions_get_adverse_reaction_phrasing(self):
        st = new_state("what side effects should I watch for?",
                       _ctx("metformin", "captopril"), "care_manager")
        assert all("adverse reactions" in q for q in _fallback_sub_queries(st))


class TestRouterNodeIntegration:

    def _mock_llm(self, monkeypatch, **overrides):
        payload = {"intent": "medication_info", "shaped_query": "sq",
                   "drug_name": None, "state_name": None, "reasoning": "r"}
        payload.update(overrides)
        monkeypatch.setattr(router_mod, "call_claude_structured", lambda **kw: payload)

    def test_llm_sub_queries_are_used_when_gate_fires(self, monkeypatch):
        self._mock_llm(monkeypatch, sub_queries=["metformin adverse reactions",
                                                 "clopidogrel bleeding"])
        st = new_state("what to watch for?", _ctx("metformin", "clopidogrel"), "care_manager")
        out = router_mod.router_node(st)
        assert out["sub_queries"] == ["metformin adverse reactions", "clopidogrel bleeding"]

    def test_gate_closed_discards_llm_sub_queries(self, monkeypatch):
        """The model must not be able to switch fan-out on for a single-drug question."""
        self._mock_llm(monkeypatch, drug_name="metformin",
                       sub_queries=["metformin a", "metformin b"])
        st = new_state("is metformin safe?", _ctx("metformin", "clopidogrel"), "care_manager")
        assert router_mod.router_node(st)["sub_queries"] == []

    def test_missing_llm_sub_queries_fall_back_to_deterministic(self, monkeypatch):
        """A forgetful model must not silently disable the feature."""
        self._mock_llm(monkeypatch)                      # no sub_queries key at all
        st = new_state("what to watch for?", _ctx("metformin", "clopidogrel"), "care_manager")
        subs = router_mod.router_node(st)["sub_queries"]
        assert len(subs) == 2 and subs[0].startswith("metformin")

    def test_sub_queries_are_capped(self, monkeypatch):
        self._mock_llm(monkeypatch)
        many = _ctx(*[f"drug{i}" for i in range(12)])
        st = new_state("what to watch for?", many, "care_manager")
        assert len(router_mod.router_node(st)["sub_queries"]) == MAX_SUB_QUERIES

    def test_duplicate_sub_queries_are_removed(self, monkeypatch):
        self._mock_llm(monkeypatch, sub_queries=["same q", "same q", "other q"])
        st = new_state("what to watch for?", _ctx("a", "b"), "care_manager")
        assert router_mod.router_node(st)["sub_queries"] == ["same q", "other q"]

    def test_blank_sub_queries_are_dropped(self, monkeypatch):
        self._mock_llm(monkeypatch, sub_queries=["  ", "", "real query"])
        st = new_state("what to watch for?", _ctx("a", "b"), "care_manager")
        assert router_mod.router_node(st)["sub_queries"] == ["real query"]


# ---------------------------------------------------------------------------
# Retrieval fan-out
# ---------------------------------------------------------------------------

class TestRetrievalFanOut:

    def test_query_tools_run_per_sub_query_and_results_are_tagged(self, monkeypatch):
        seen: list[str] = []

        def fake_call(name, state, query_override=None, limit_override=None):
            if name == "search_drug_labels":
                seen.append(query_override)
                return [_ev(text=f"chunk for {query_override}")]
            return []

        monkeypatch.setattr(ret_mod, "_call_tool", fake_call)
        st = new_state("q", _ctx("metformin", "clopidogrel"), "care_manager")
        st["tools_to_call"] = ["search_drug_labels"]
        st["sub_queries"] = ["metformin adverse", "clopidogrel bleeding"]

        out = ret_mod.retrieval_node(st)
        assert sorted(seen) == ["clopidogrel bleeding", "metformin adverse"]
        tags = [e.metadata["sub_query"] for e in out["raw_evidence"]]
        assert sorted(tags) == ["clopidogrel bleeding", "metformin adverse"]

    def test_entity_tools_run_once_not_per_sub_query(self, monkeypatch):
        """fetch_drug_label ignores the query text — running it N times is pure waste."""
        calls: list[str] = []

        def fake_call(name, state, query_override=None, limit_override=None):
            calls.append(name)
            return []

        monkeypatch.setattr(ret_mod, "_call_tool", fake_call)
        st = new_state("q", _ctx("a", "b"), "care_manager")
        st["tools_to_call"] = ["search_drug_labels", "fetch_drug_label"]
        st["sub_queries"] = ["q1", "q2", "q3"]

        ret_mod.retrieval_node(st)
        assert calls.count("search_drug_labels") == 3
        assert calls.count("fetch_drug_label") == 1

    def test_no_sub_queries_uses_the_original_single_query_path(self, monkeypatch):
        def fake_call(name, state, query_override=None, limit_override=None):
            assert query_override is None      # normal path must not override
            return [_ev()]

        monkeypatch.setattr(ret_mod, "_call_tool", fake_call)
        st = new_state("q", None, "anonymous")
        st["tools_to_call"] = ["search_drug_labels"]
        out = ret_mod.retrieval_node(st)
        assert len(out["raw_evidence"]) == 1
        assert "sub_query" not in (out["raw_evidence"][0].metadata or {})

    def test_one_failing_sub_query_does_not_sink_the_rest(self, monkeypatch):
        def fake_call(name, state, query_override=None, limit_override=None):
            if query_override == "bad":
                raise RuntimeError("qdrant exploded")
            return [_ev(text=query_override)]

        monkeypatch.setattr(ret_mod, "_call_tool", fake_call)
        st = new_state("q", _ctx("a", "b"), "care_manager")
        st["tools_to_call"] = ["search_drug_labels"]
        st["sub_queries"] = ["good1", "bad", "good2"]
        out = ret_mod.retrieval_node(st)
        assert sorted(e.text for e in out["raw_evidence"]) == ["good1", "good2"]


# ---------------------------------------------------------------------------
# Evidence-filter grouped rerank
# ---------------------------------------------------------------------------

class TestGroupedRerank:

    def test_each_group_is_reranked_against_its_own_sub_query(self, monkeypatch):
        """THE fix: a per-drug chunk is graded on its own sub-query, not the diffuse ask."""
        used: list[str] = []

        def fake_rerank(query, evidence, top_k=None):
            used.append(query)
            for e in evidence:
                e.score = 0.9
            return list(evidence)

        monkeypatch.setattr(ef_mod.reranker, "rerank", fake_rerank)
        items = [_ev(text="metformin chunk", sub_query="metformin adverse"),
                 _ev(text="clopidogrel chunk", sub_query="clopidogrel bleeding")]
        ef_mod._rerank_by_sub_query(items, "the diffuse question")
        assert used == ["metformin adverse", "clopidogrel bleeding"]

    def test_untagged_items_use_the_original_query(self, monkeypatch):
        used: list[str] = []

        def fake_rerank(query, evidence, top_k=None):
            used.append(query)
            for e in evidence:
                e.score = 0.9
            return list(evidence)

        monkeypatch.setattr(ef_mod.reranker, "rerank", fake_rerank)
        ef_mod._rerank_by_sub_query([_ev(text="from an entity tool")], "original question")
        assert used == ["original question"]

    def test_merge_is_round_robin_so_every_drug_is_represented(self, monkeypatch):
        """A global sort would let one high-scoring drug take every slot."""
        def fake_rerank(query, evidence, top_k=None):
            for e in evidence:
                e.score = 0.99 if "A" in e.text else 0.10
            return sorted(evidence, key=lambda e: -(e.score or 0.0))

        monkeypatch.setattr(ef_mod.reranker, "rerank", fake_rerank)
        items = [_ev(text="A1", sub_query="qa"), _ev(text="A2", sub_query="qa"),
                 _ev(text="B1", sub_query="qb"), _ev(text="B2", sub_query="qb")]
        merged = ef_mod._rerank_by_sub_query(items, "q")
        # the first two slots must be one per drug, not both from A
        assert {merged[0].text, merged[1].text} == {"A1", "B1"}

    def test_per_group_keep_is_capped(self, monkeypatch):
        def fake_rerank(query, evidence, top_k=None):
            for i, e in enumerate(evidence):
                e.score = 1.0 - i * 0.01
            return list(evidence)

        monkeypatch.setattr(ef_mod.reranker, "rerank", fake_rerank)
        items = [_ev(text=f"c{i}", sub_query="qa") for i in range(6)]
        merged = ef_mod._rerank_by_sub_query(items, "q")
        assert len(merged) == ef_mod.PER_SUBQUERY_KEEP

    def test_low_scoring_chunks_are_still_dropped(self, monkeypatch):
        def fake_rerank(query, evidence, top_k=None):
            for e in evidence:
                e.score = 0.0            # below RERANK_MIN_SCORE
            return list(evidence)

        monkeypatch.setattr(ef_mod.reranker, "rerank", fake_rerank)
        items = [_ev(text="junk", sub_query="qa")]
        assert ef_mod._rerank_by_sub_query(items, "q") == []

    def test_filter_node_uses_the_fanout_path_when_tags_are_present(self, monkeypatch):
        monkeypatch.setattr(ef_mod.reranker, "is_available", lambda: True)

        def fake_rerank(query, evidence, top_k=None):
            for e in evidence:
                e.score = 0.9
            return list(evidence)

        monkeypatch.setattr(ef_mod.reranker, "rerank", fake_rerank)
        st = new_state("diffuse q", None, "care_manager")
        st["intent"] = "medication_info"
        st["raw_evidence"] = [_ev(text="m chunk", sub_query="metformin"),
                              _ev(text="c chunk", sub_query="clopidogrel")]
        out = ef_mod.evidence_filter_node(st)
        assert len(out["filtered_evidence"]) == 2
        assert any("fan-out" in t for t in out["trace"])

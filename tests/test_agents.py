"""
Tests for the agent pipeline (agents/ + graph/).

Strategy:
  - Deterministic nodes (evidence_filter, safety, router tool-mapping, state)
    are tested directly, no LLM.
  - LLM-dependent nodes (router, answer_generator, reviewer) are tested by
    monkeypatching the agents.llm helpers so no real API calls are made.
  - A small number of @pytest.mark.llm integration tests run the real pipeline
    (skipped unless ANTHROPIC_API_KEY is set and --run-llm is implied by marker).

Run with:  python -m pytest tests/test_agents.py -v
           python -m pytest tests/test_agents.py -m "not llm" -v   # skip live LLM
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
            override=True)

from agents.state import PipelineState, new_state, RouterDecision, ReviewResult, INTENTS
from models.schemas import Evidence


def _ev(source="fda_label", title="t", text="some text", score=None, **md):
    return Evidence(source=source, title=title, text=text, score=score,
                    metadata=md, citation=f"{source} cite")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

class TestState:
    def test_new_state_defaults(self):
        s = new_state("q")
        assert s["query"] == "q"
        assert s["user_role"] == "anonymous"
        assert s["iteration"] == 0
        assert s["max_iterations"] == 2
        assert s["patient_context"] is None

    def test_new_state_with_context(self):
        s = new_state("q", patient_context={"name": "X"}, user_role="care_manager")
        assert s["patient_context"]["name"] == "X"
        assert s["user_role"] == "care_manager"

    def test_intents_list(self):
        assert "medication_info" in INTENTS
        assert "drug_recall" in INTENTS
        assert "policy_eligibility" in INTENTS


# ---------------------------------------------------------------------------
# Router — tool mapping (deterministic) + node (mocked LLM)
# ---------------------------------------------------------------------------

class TestRouter:
    def test_intent_tool_map_complete(self):
        from agents.router import INTENT_TOOL_MAP
        for intent in INTENTS:
            assert intent in INTENT_TOOL_MAP
        # All retrieval intents have tools; patient_summary is intentionally empty
        # because it's handled by the dedicated patient_summary_node (serves the
        # pre-computed summary, bypassing tool retrieval).
        for intent in INTENTS:
            if intent == "patient_summary":
                assert INTENT_TOOL_MAP[intent] == []
            else:
                assert len(INTENT_TOOL_MAP[intent]) > 0

    def test_router_node_mocked(self, monkeypatch):
        import agents.router as router_mod
        monkeypatch.setattr(router_mod, "call_claude_structured", lambda **kw: {
            "intent": "medication_info",
            "shaped_query": "metformin side effects",
            "drug_name": "metformin",
            "state_name": None,
            "reasoning": "drug question",
        })
        state = new_state("side effects of metformin?")
        out = router_mod.router_node(state)
        assert out["intent"] == "medication_info"
        assert out["drug_name"] == "metformin"
        assert out["tools_to_call"] == ["search_drug_labels", "fetch_drug_label"]

    def test_router_invalid_intent_falls_back(self, monkeypatch):
        import agents.router as router_mod
        monkeypatch.setattr(router_mod, "call_claude_structured", lambda **kw: {
            "intent": "nonsense_intent",
            "shaped_query": "x",
            "reasoning": "r",
        })
        out = router_mod.router_node(new_state("q"))
        assert out["intent"] == "general"

    def test_router_clears_feedback(self, monkeypatch):
        import agents.router as router_mod
        monkeypatch.setattr(router_mod, "call_claude_structured", lambda **kw: {
            "intent": "general", "shaped_query": "x", "reasoning": "r",
        })
        state = new_state("q")
        state["reviewer_feedback"] = "fix this"
        state["route_back_to"] = "router"
        out = router_mod.router_node(state)
        assert out["route_back_to"] is None
        assert out["reviewer_feedback"] is None


# ---------------------------------------------------------------------------
# Evidence Filter (fully deterministic)
# ---------------------------------------------------------------------------

# NOTE: Evidence Filter tests moved to tests/test_reranker.py, which covers both
# the cross-encoder rerank path and the bi-encoder fallback path. (The filter was
# refactored in Phase 2 to use retrieve-then-rerank.)


# ---------------------------------------------------------------------------
# Answer Generator (mocked LLM)
# ---------------------------------------------------------------------------

class TestAnswerGenerator:
    def test_answer_node_mocked(self, monkeypatch):
        import agents.answer_generator as ag
        monkeypatch.setattr(ag, "call_claude_text",
                            lambda **kw: "Metformin treats diabetes [1].")
        state = new_state("what is metformin?")
        state["filtered_evidence"] = [_ev(title="Metformin", text="treats diabetes")]
        out = ag.answer_node(state) if hasattr(ag, "answer_node") else ag.answer_generator_node(state)
        assert "[1]" in out["answer"]
        assert len(out["citations"]) == 1

    def test_care_manager_uses_clinical_prompt(self):
        from agents.answer_generator import _system_for_role, _SYSTEM_CARE_MANAGER, _SYSTEM_PATIENT
        assert _system_for_role("care_manager") == _SYSTEM_CARE_MANAGER
        assert _system_for_role("patient") == _SYSTEM_PATIENT
        assert _system_for_role("anonymous") == _SYSTEM_PATIENT


# ---------------------------------------------------------------------------
# Reviewer (mocked LLM)
# ---------------------------------------------------------------------------

class TestReviewer:
    def test_reviewer_pass(self, monkeypatch):
        import agents.reviewer as rev
        monkeypatch.setattr(rev, "call_claude_structured", lambda **kw: {
            "relevant": True, "faithful": True, "passed": True,
            "route_back_to": None, "feedback": "",
        })
        state = new_state("q")
        state["answer"] = "a"
        out = rev.reviewer_node(state)
        assert out["review_passed"] is True
        assert out["route_back_to"] is None
        assert out["iteration"] == 1

    def test_reviewer_fail_routes_to_answer_gen(self, monkeypatch):
        import agents.reviewer as rev
        monkeypatch.setattr(rev, "call_claude_structured", lambda **kw: {
            "relevant": True, "faithful": False, "passed": False,
            "route_back_to": "answer_generator", "feedback": "claim 2 unsupported",
        })
        state = new_state("q")
        out = rev.reviewer_node(state)
        assert out["review_passed"] is False
        assert out["route_back_to"] == "answer_generator"
        assert out["reviewer_feedback"] == "claim 2 unsupported"

    def test_reviewer_fail_routes_to_router(self, monkeypatch):
        import agents.reviewer as rev
        monkeypatch.setattr(rev, "call_claude_structured", lambda **kw: {
            "relevant": False, "faithful": False, "passed": False,
            "route_back_to": "router", "feedback": "search for X instead",
        })
        out = rev.reviewer_node(new_state("q"))
        assert out["route_back_to"] == "router"

    def test_reviewer_enforces_consistency(self, monkeypatch):
        """passed must be False if not (relevant AND faithful), even if LLM says True."""
        import agents.reviewer as rev
        monkeypatch.setattr(rev, "call_claude_structured", lambda **kw: {
            "relevant": True, "faithful": False, "passed": True,  # inconsistent
            "route_back_to": "answer_generator", "feedback": "x",
        })
        out = rev.reviewer_node(new_state("q"))
        assert out["review_passed"] is False   # corrected

    def test_reviewer_invalid_route_defaults(self, monkeypatch):
        import agents.reviewer as rev
        monkeypatch.setattr(rev, "call_claude_structured", lambda **kw: {
            "relevant": True, "faithful": False, "passed": False,
            "route_back_to": "garbage", "feedback": "x",
        })
        out = rev.reviewer_node(new_state("q"))
        assert out["route_back_to"] == "answer_generator"  # safe default


# ---------------------------------------------------------------------------
# Safety (fully deterministic)
# ---------------------------------------------------------------------------

class TestSafety:
    def test_standard_disclaimer_added(self):
        from agents.safety_agent import safety_node, STANDARD_DISCLAIMER
        state = new_state("what is metformin?")
        state["answer"] = "answer"
        state["review_passed"] = True
        out = safety_node(state)
        assert STANDARD_DISCLAIMER in out["disclaimers"]
        assert STANDARD_DISCLAIMER in out["final_answer"]

    def test_unverified_note_when_review_failed(self):
        from agents.safety_agent import safety_node, UNVERIFIED_NOTE
        state = new_state("q")
        state["answer"] = "a"
        state["review_passed"] = False
        out = safety_node(state)
        assert UNVERIFIED_NOTE in out["disclaimers"]

    def test_no_unverified_note_when_passed(self):
        from agents.safety_agent import safety_node, UNVERIFIED_NOTE
        state = new_state("q")
        state["answer"] = "a"
        state["review_passed"] = True
        out = safety_node(state)
        assert UNVERIFIED_NOTE not in out["disclaimers"]

    def test_clinical_flag_detected(self):
        from agents.safety_agent import safety_node, CLINICAL_FLAG
        state = new_state("Should I stop taking my metformin?")
        state["answer"] = "a"
        state["review_passed"] = True
        out = safety_node(state)
        assert CLINICAL_FLAG in out["disclaimers"]

    def test_no_clinical_flag_for_info_query(self):
        from agents.safety_agent import safety_node, CLINICAL_FLAG
        state = new_state("What are the side effects of metformin?")
        state["answer"] = "a"
        state["review_passed"] = True
        out = safety_node(state)
        assert CLINICAL_FLAG not in out["disclaimers"]

    def test_citations_included(self):
        from agents.safety_agent import safety_node
        state = new_state("q")
        state["answer"] = "answer"
        state["citations"] = ["[1] FDA Label — Metformin"]
        state["review_passed"] = True
        out = safety_node(state)
        assert "[1] FDA Label — Metformin" in out["final_answer"]


# ---------------------------------------------------------------------------
# Graph routing logic (deterministic)
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def test_route_to_safety_on_pass(self):
        from graph.pipeline import _route_after_review
        state = new_state("q")
        state["review_passed"] = True
        assert _route_after_review(state) == "safety"

    def test_route_to_safety_at_max_iterations(self):
        from graph.pipeline import _route_after_review
        state = new_state("q", max_iterations=2)
        state["review_passed"] = False
        state["iteration"] = 2  # cap reached
        assert _route_after_review(state) == "safety"

    def test_route_back_to_router(self):
        from graph.pipeline import _route_after_review
        state = new_state("q", max_iterations=2)
        state["review_passed"] = False
        state["iteration"] = 1
        state["route_back_to"] = "router"
        assert _route_after_review(state) == "router"

    def test_route_back_to_answer_gen(self):
        from graph.pipeline import _route_after_review
        state = new_state("q", max_iterations=2)
        state["review_passed"] = False
        state["iteration"] = 1
        state["route_back_to"] = "answer_generator"
        assert _route_after_review(state) == "answer_generator"

    def test_pipeline_builds(self):
        from graph.pipeline import build_pipeline
        pipeline = build_pipeline()
        assert pipeline is not None


# ---------------------------------------------------------------------------
# Full pipeline integration (live LLM — marked, runs only when key present)
# ---------------------------------------------------------------------------

@pytest.mark.llm
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestPipelineIntegration:
    def test_medication_query_end_to_end(self):
        from graph.pipeline import answer_query
        final = answer_query("What are the warnings for metformin?")
        assert final["intent"] == "medication_info"
        assert final["final_answer"]
        assert len(final["raw_evidence"]) > 0
        assert "metformin" in final["final_answer"].lower()

    def test_recall_query_end_to_end(self):
        from graph.pipeline import answer_query
        final = answer_query("Is there a recall on semaglutide?")
        assert final["intent"] == "drug_recall"
        assert final["final_answer"]

    def test_policy_query_end_to_end(self):
        from graph.pipeline import answer_query
        final = answer_query("Medicaid income limit for pregnant women in Texas?")
        assert final["intent"] == "policy_eligibility"
        assert final["state_name"] == "Texas"

    def test_pipeline_terminates(self):
        """Even a hard query must terminate within the iteration budget."""
        from graph.pipeline import answer_query
        final = answer_query("What is the airspeed velocity of an unladen swallow?")
        assert final["final_answer"]
        assert final["iteration"] <= final["max_iterations"]

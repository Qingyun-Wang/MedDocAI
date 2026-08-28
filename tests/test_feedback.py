"""
Unit tests for the user-feedback loop.

Covers the SQLite layer (ingestion/sqlite_loader.py) and the eval-candidate
exporter (scripts/feedback_to_eval.py). Everything runs against a temp DB — no
API keys, no data/, no network.

Run with:  python -m pytest tests/test_feedback.py -v
"""

from __future__ import annotations

import pytest

from ingestion.sqlite_loader import MedDocDB
from scripts.feedback_to_eval import build_candidates


@pytest.fixture
def db(tmp_path):
    """A virgin DB — the feedback table does not exist until the first write."""
    return MedDocDB(str(tmp_path / "t.db"))


class TestFeedbackStorage:

    def test_reads_empty_before_the_table_exists(self, db):
        """A fresh deploy must not crash just because nobody has rated anything."""
        assert db.get_feedback() == []
        assert db.feedback_counts() == {"up": 0, "down": 0}

    def test_first_write_creates_its_own_table(self, db):
        """Self-ensuring DDL: nothing calls create_tables() at app startup."""
        db.save_feedback("q1", "up", question="q?", answer="a")
        assert db.feedback_counts() == {"up": 1, "down": 0}

    def test_anonymous_feedback_is_kept(self, db):
        """The whole point: most demo traffic has no patient selected.

        chat_history is only written when a patient is chosen, so feedback must
        store the question/answer itself rather than joining to it later.
        """
        db.save_feedback("q1", "down", question="what are metformin risks?",
                         answer="bad answer", comment="too vague")
        row = db.get_feedback(rating="down")[0]
        assert row["patient_id"] is None
        assert row["question"] == "what are metformin risks?"
        assert row["answer"] == "bad answer"

    def test_re_rating_replaces_rather_than_duplicates(self, db):
        db.save_feedback("q1", "down", question="q?")
        db.save_feedback("q1", "up", question="q?")
        assert db.feedback_counts() == {"up": 1, "down": 0}
        assert len(db.get_feedback()) == 1

    def test_comment_is_persisted(self, db):
        db.save_feedback("q1", "down", comment="cited the wrong drug")
        assert db.get_feedback()[0]["comment"] == "cited the wrong drug"

    def test_rejects_an_invalid_rating(self, db):
        with pytest.raises(ValueError):
            db.save_feedback("q1", "meh")

    def test_filters_by_rating(self, db):
        db.save_feedback("q1", "up")
        db.save_feedback("q2", "down")
        db.save_feedback("q3", "down")
        assert len(db.get_feedback(rating="down")) == 2
        assert len(db.get_feedback(rating="up")) == 1

    def test_filters_by_query_ids(self, db):
        """Used by the UI to re-hydrate ratings when history is reloaded."""
        for q in ("q1", "q2", "q3"):
            db.save_feedback(q, "up")
        got = {r["query_id"] for r in db.get_feedback(query_ids=["q1", "q3"])}
        assert got == {"q1", "q3"}

    def test_metadata_round_trips(self, db):
        db.save_feedback("q1", "down", intent="drug_recall", user_role="care_manager",
                         patient_id="pid-1", persona="care_manager",
                         review_passed=True, n_evidence=7, session_id="s1")
        r = db.get_feedback()[0]
        assert r["intent"] == "drug_recall"
        assert r["user_role"] == "care_manager"
        assert r["patient_id"] == "pid-1"
        assert r["review_passed"] == 1
        assert r["n_evidence"] == 7
        assert r["session_id"] == "s1"


class TestCandidateExport:

    def test_reference_is_left_empty(self, db):
        """The load-bearing discipline: a rejected answer must never become the key.

        Auto-filling `reference` from the failed answer would bake the failure into
        the thing that grades future failures.
        """
        rows = [{"query_id": "q1", "question": "q?", "answer": "the wrong answer",
                 "comment": "wrong", "intent": "medication_info", "user_role": "patient",
                 "patient_id": None, "review_passed": 0, "n_evidence": 3,
                 "created_at": "2026-01-01"}]
        cand = build_candidates(rows, db)[0]
        assert cand["reference"] == ""
        # ...but the human gets what they need to write one
        assert cand["_review"]["rejected_answer"] == "the wrong answer"
        assert cand["_review"]["why_flagged"] == "wrong"

    def test_intent_maps_to_eval_category(self, db):
        rows = [
            {"query_id": "a", "intent": "drug_recall", "question": "q", "answer": "a"},
            {"query_id": "b", "intent": "policy_eligibility", "question": "q", "answer": "a"},
        ]
        cats = [c["category"] for c in build_candidates(rows, db)]
        assert cats == ["recall", "policy"]

    def test_patientless_clinical_question_becomes_med_anon(self, db):
        rows = [{"query_id": "a", "intent": "medication_info", "question": "q",
                 "answer": "a", "patient_id": None}]
        assert build_candidates(rows, db)[0]["category"] == "med_anon"

    def test_unknown_intent_falls_back_to_general(self, db):
        """An unmapped intent should surface for review, not vanish."""
        rows = [{"query_id": "a", "intent": "something_new", "question": "q", "answer": "a"}]
        assert build_candidates(rows, db)[0]["category"] == "general"

    def test_ids_are_sequential_and_shaped_like_the_dataset(self, db):
        rows = [{"query_id": f"q{i}", "question": "q", "answer": "a"} for i in range(3)]
        assert [c["id"] for c in build_candidates(rows, db)] == ["fb_001", "fb_002", "fb_003"]

    def test_candidate_has_the_dataset_field_shape(self, db):
        """Candidates should drop into eval_dataset.json once a reference is written."""
        rows = [{"query_id": "q1", "question": "q?", "answer": "a",
                 "intent": "medication_info", "user_role": "care_manager"}]
        cand = build_candidates(rows, db)[0]
        for field in ("id", "category", "question", "user_role", "patient", "reference"):
            assert field in cand
        assert cand["user_role"] == "care_manager"

    def test_missing_note_is_labelled_not_blank(self, db):
        rows = [{"query_id": "q1", "question": "q?", "answer": "a", "comment": ""}]
        assert build_candidates(rows, db)[0]["_review"]["why_flagged"] == "(no note given)"

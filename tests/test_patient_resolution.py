"""Tests for eval patient resolution — tests/test_patient_resolution.py

The eval dataset pins 17 questions to patients by NAME PREFIX ("Cali421",
"Arlie395", ...). Those references were hand-verified against those exact records,
so resolving a prefix to the wrong patient does not fail — it produces a confidently
wrong score with no error.

That is not hypothetical. Expanding the demo cohort from 10 to ~100 patients pulled
in "Bogan983_Arlie395", which made the pinned prefix "Arlie395" match two people.
The old code assigned resolved[prefix] on every match, so the last one silently won.
"""

import pytest

import evaluation.run_eval as E


class _FakeDB:
    def __init__(self, names):
        self._rows = [{"patient_id": f"id{i}", "name": n} for i, n in enumerate(names)]

    def list_patients(self, limit=500):
        return list(self._rows)

    def get_patient(self, pid):
        return {"patient_id": pid}


@pytest.fixture
def db(monkeypatch):
    """Patch the MedDocDB that resolve_patients imports inside the function."""
    holder = {}

    def _install(names):
        import ingestion.sqlite_loader as S
        monkeypatch.setattr(S, "MedDocDB", lambda *a, **k: _FakeDB(names))
        holder["names"] = names
    return _install


def _q(prefix):
    return {"id": f"q_{prefix}", "category": "med_patient", "patient": prefix}


# ---------------------------------------------------------------------------
# The short-circuit that keeps CI from creating a stray database
# ---------------------------------------------------------------------------

def test_no_patient_questions_touches_no_database(monkeypatch):
    """CI has no data/, so opening MedDocDB there would create an empty file and
    then die on `no such table`. Must return {} without importing the DB at all."""
    import ingestion.sqlite_loader as S

    def explode(*a, **k):
        raise AssertionError("MedDocDB must not be constructed")

    monkeypatch.setattr(S, "MedDocDB", explode)
    assert E.resolve_patients([{"id": "x", "category": "med_anon"}]) == {}
    assert E.resolve_patients([{"id": "x", "patient": None}]) == {}


# ---------------------------------------------------------------------------
# The ambiguity guard
# ---------------------------------------------------------------------------

def test_unique_prefix_resolves(db):
    db(["Arlie395 Abbott509", "Cali421 Abernathy746"])
    out = E.resolve_patients([_q("Arlie395"), _q("Cali421")])
    assert set(out) == {"Arlie395", "Cali421"}


def test_duplicate_given_name_raises_instead_of_picking_one(db):
    """THE regression this file exists for. Two Arlie395s must be an error, not a
    coin flip — a silently wrong patient scores a hand-written reference."""
    db(["Arlie395 Abbott509", "Arlie395 Bogan983"])
    with pytest.raises(RuntimeError, match="Ambiguous"):
        E.resolve_patients([_q("Arlie395")])


def test_ambiguity_error_names_the_offenders(db):
    """The message has to be actionable — which patients collided."""
    db(["Arlie395 Abbott509", "Arlie395 Bogan983"])
    with pytest.raises(RuntimeError) as e:
        E.resolve_patients([_q("Arlie395")])
    assert "Abbott509" in str(e.value) and "Bogan983" in str(e.value)


def test_unrelated_duplicates_do_not_trip_the_guard(db):
    """Only prefixes the dataset actually pins matter; a crowded roster is fine."""
    db(["Arlie395 Abbott509", "Jude119 Smith1", "Jude119 Smith2"])
    out = E.resolve_patients([_q("Arlie395")])
    assert set(out) == {"Arlie395"}


def test_missing_patient_still_raises(db):
    db(["Cali421 Abernathy746"])
    with pytest.raises(RuntimeError, match="not found"):
        E.resolve_patients([_q("Arlie395")])


def test_missing_is_reported_before_ambiguity(db):
    """A missing patient is the more basic error; report it rather than masking it."""
    db(["Arlie395 Abbott509", "Arlie395 Bogan983"])
    with pytest.raises(RuntimeError, match="not found"):
        E.resolve_patients([_q("Arlie395"), _q("Nobody999")])


def test_prefix_matching_is_still_prefix_based(db):
    """Names are stored "Given Surname"; the dataset pins the given name only."""
    db(["Leopold359 Abbott509"])
    assert set(E.resolve_patients([_q("Leopold359")])) == {"Leopold359"}

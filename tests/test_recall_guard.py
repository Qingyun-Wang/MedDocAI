"""Tests for the recall lookup's unknown-drug guard — tests/test_recall_guard.py

An empty openFDA recall search means one of two very different things:

  (a) a real drug with no ongoing recalls   -> a genuine, useful negative
  (b) a name openFDA does not recognise     -> says NOTHING about safety

The tool used to report both as "No current recalls — <name>". The adversarial
eval caught the consequence: asked about an invented drug ("Zelvantix"), the
assistant replied that the evidence "confirms there are no current FDA recalls for
Zelvantix" — absence of evidence presented as evidence of absence, which reads as
confirming the drug exists and is safe.

These tests pin the distinction, because it is invisible in normal use: every real
drug takes branch (a), so a regression here would only show up on a fake name.
"""

import pytest

import tools.openfda_tool as T


def _mock_api(monkeypatch, *, recalls=None, label_hit=False):
    """Fake _api_get: no recalls, and a label probe that may or may not match."""
    def fake(url, params):
        if url == T.ENFORCEMENT_URL:
            return {"results": recalls} if recalls else {"results": []}
        if url == T.LABEL_URL:
            return {"results": [{"openfda": {}}]} if label_hit else {"results": []}
        return None
    monkeypatch.setattr(T, "_api_get", fake)


# ---------------------------------------------------------------------------
# (b) the name is not recognised
# ---------------------------------------------------------------------------

def test_unknown_drug_does_not_claim_no_recalls(monkeypatch):
    _mock_api(monkeypatch, label_hit=False)
    ev = T.check_drug_recalls("Zelvantix")[0]

    assert ev.metadata["name_known"] is False
    assert "No current recalls" not in ev.title, \
        "must not assert a recall status for a name FDA does not recognise"
    assert "No FDA record matched" in ev.title


def test_unknown_drug_explicitly_denies_a_safety_finding(monkeypatch):
    """The text has to say so out loud — the Answer Generator reads this, and on
    one run it turned a bare 'no recalls found' into 'confirms no recalls'."""
    _mock_api(monkeypatch, label_hit=False)
    text = T.check_drug_recalls("Zelvantix")[0].text.lower()

    assert "not a finding" in text
    assert "misspelled" in text
    assert "do not describe it as an existing drug" in text


def test_unknown_drug_names_itself_in_the_text(monkeypatch):
    _mock_api(monkeypatch, label_hit=False)
    assert "Zelvantix" in T.check_drug_recalls("Zelvantix")[0].text


# ---------------------------------------------------------------------------
# (a) the name IS recognised, there simply are no recalls
# ---------------------------------------------------------------------------

def test_known_drug_with_no_recalls_keeps_the_useful_negative(monkeypatch):
    """This is a real answer to a real question and must NOT be weakened."""
    _mock_api(monkeypatch, label_hit=True)
    ev = T.check_drug_recalls("metformin")[0]

    assert ev.metadata["name_known"] is True
    assert ev.title == "No current recalls — metformin"
    assert "No ongoing FDA recalls found" in ev.text
    assert "not a finding" not in ev.text.lower()


# ---------------------------------------------------------------------------
# the normal path is untouched
# ---------------------------------------------------------------------------

def test_actual_recalls_are_unaffected(monkeypatch):
    _mock_api(monkeypatch, recalls=[{
        "product_description": "METFORMIN HCl ER 500MG",
        "reason_for_recall": "NDMA above limit",
        "classification": "Class II",
        "status": "Ongoing",
        "recalling_firm": "Acme Pharma",
    }])
    out = T.check_drug_recalls("metformin")

    assert len(out) == 1
    assert "Recall [Class II]" in out[0].title
    assert "NDMA" in out[0].text


def test_no_label_probe_when_recalls_exist(monkeypatch):
    """The extra probe is only worth paying for on the empty path."""
    seen = []

    def fake(url, params):
        seen.append(url)
        if url == T.ENFORCEMENT_URL:
            return {"results": [{"product_description": "X", "reason_for_recall": "y",
                                 "classification": "Class I", "status": "Ongoing",
                                 "recalling_firm": "Z"}]}
        return {"results": []}

    monkeypatch.setattr(T, "_api_get", fake)
    T.check_drug_recalls("metformin")
    assert T.LABEL_URL not in seen


@pytest.mark.parametrize("api_returns", [None, {}, {"results": []}])
def test_probe_failure_is_treated_as_unknown_not_as_safe(monkeypatch, api_returns):
    """Fail closed: if the probe itself errors we must not fall back to the
    reassuring message."""
    def fake(url, params):
        return {"results": []} if url == T.ENFORCEMENT_URL else api_returns

    monkeypatch.setattr(T, "_api_get", fake)
    ev = T.check_drug_recalls("Zelvantix")[0]
    assert ev.metadata["name_known"] is False
    assert "No current recalls" not in ev.title

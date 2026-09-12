"""Tests for prompt caching of the static tools+system prefix — tests/test_caching.py

Prompt caching fails SILENTLY. A prefix below the model's minimum is not an
error: the API ignores the marker, returns zero for both cache counters, and the
bill quietly rises. So the things worth testing are not "does it save money" but:

  - the marker is actually attached when we ask for it,
  - our internal flag never leaks into the API call,
  - and a request that asked for caching and got none says so out loud.

That last one already earned its place: on its first live run it caught that the
Reviewer's prefix sits under Sonnet 4.5's 1,024-token minimum, which a
count_tokens estimate had wrongly said cleared it by +116.
"""

import logging

import pytest

import agents.llm as L


class _Usage:
    def __init__(self, i=100, o=10, read=0, write=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = read
        self.cache_creation_input_tokens = write


class _Block:
    type = "tool_use"
    input = {"ok": True}


class _Response:
    def __init__(self, usage):
        self.usage = usage
        self.content = [_Block()]


class _FakeClient:
    """Captures the kwargs the SDK would have received."""

    def __init__(self, usage=None):
        self.seen = {}
        self._usage = usage or _Usage()
        self.messages = self

    def create(self, **kwargs):
        self.seen = kwargs
        return _Response(self._usage)


@pytest.fixture
def client(monkeypatch):
    c = _FakeClient()
    monkeypatch.setattr(L, "_get_client", lambda: c)
    monkeypatch.setattr(L.observability, "record_llm_call", lambda **kw: None)
    return c


def _call(**kw):
    return L.call_claude_structured(
        system="S" * 50, user="u", tool_name="t", tool_description="d",
        input_schema={"type": "object", "properties": {}}, **kw,
    )


# ---------------------------------------------------------------------------
# The marker
# ---------------------------------------------------------------------------

def test_cache_marker_attached_by_default(client):
    _call()
    system = client.seen["system"]
    assert isinstance(system, list), "system must be a block list to carry cache_control"
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_cache_marker_omitted_when_opted_out(client):
    """The Reviewer opts out; its prefix is below the model minimum."""
    _call(cache_prefix=False)
    assert isinstance(client.seen["system"], str), "opted-out calls send a plain string"


def test_marker_goes_on_system_not_tools(client):
    """tools render BEFORE system, so one marker on system covers both."""
    _call()
    assert "cache_control" not in client.seen["tools"][0]
    assert "cache_control" in client.seen["system"][0]


def test_internal_flag_never_reaches_the_api(client):
    """_cache_expected is ours. Passing it to the SDK would be a TypeError."""
    for prefix in (True, False):
        _call(cache_prefix=prefix)
        assert "_cache_expected" not in client.seen


def test_system_text_is_preserved_either_way(client):
    _call()
    cached = client.seen["system"][0]["text"]
    _call(cache_prefix=False)
    assert cached == client.seen["system"] == "S" * 50


# ---------------------------------------------------------------------------
# The silent-failure alarm
# ---------------------------------------------------------------------------

def test_warns_when_caching_asked_for_but_did_nothing(client, caplog):
    """A too-short prefix returns zeros for both counters and no error."""
    client._usage = _Usage(read=0, write=0)
    with caplog.at_level(logging.WARNING, logger="agents.llm"):
        _call(cache_prefix=True)
    assert any("neither a cache read nor a write" in r.message for r in caplog.records)


@pytest.mark.parametrize("read,write", [(1375, 0), (0, 1375)])
def test_no_warning_when_cache_actually_worked(client, caplog, read, write):
    client._usage = _Usage(read=read, write=write)
    with caplog.at_level(logging.WARNING, logger="agents.llm"):
        _call(cache_prefix=True)
    assert not [r for r in caplog.records if "cache" in r.message]


def test_no_warning_when_caching_was_not_requested(client, caplog):
    """Opting out must be silent — otherwise the alarm cries wolf every query."""
    client._usage = _Usage(read=0, write=0)
    with caplog.at_level(logging.WARNING, logger="agents.llm"):
        _call(cache_prefix=False)
    assert not [r for r in caplog.records if "cache" in r.message]


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

def test_cache_tokens_reach_the_recorder(monkeypatch):
    seen = {}
    c = _FakeClient(_Usage(read=1375, write=7))
    monkeypatch.setattr(L, "_get_client", lambda: c)
    monkeypatch.setattr(L.observability, "record_llm_call",
                        lambda **kw: seen.update(kw))
    _call()
    assert seen["cache_read_tokens"] == 1375
    assert seen["cache_write_tokens"] == 7


def test_summary_aggregates_cache_tokens():
    """Surfaced per query so a silent stop is visible, not inferred."""
    from agents import observability as O
    O.start_query("q1")
    for _ in range(3):
        O.record_llm_call(model="claude-sonnet-4-5", caller="router",
                          input_tokens=440, output_tokens=10,
                          cache_read_tokens=1375, cache_write_tokens=0,
                          latency_ms=1.0)
    s = O.finish_query("q1")
    assert s["cache_read_tokens"] == 4125
    assert s["cache_write_tokens"] == 0


def test_cost_accounts_for_the_cache_discount():
    """A cache read is billed at 10% of input; treating it as full price would
    overstate the bill and hide the saving the change exists to produce."""
    from agents.observability import estimate_cost_usd
    full = estimate_cost_usd("claude-sonnet-4-5", 1375, 0, 0, 0)
    read = estimate_cost_usd("claude-sonnet-4-5", 0, 0, 1375, 0)
    assert read == pytest.approx(full * 0.1, rel=1e-6)


# ---------------------------------------------------------------------------
# Call sites
# ---------------------------------------------------------------------------

def test_reviewer_opts_out_of_caching(monkeypatch):
    """Load-bearing: the Reviewer's prefix is under the minimum. If someone
    'tidies' this flag away, every query starts logging a useless warning."""
    import agents.reviewer as V
    seen = {}
    monkeypatch.setattr(V, "call_claude_structured",
                        lambda **kw: seen.update(kw) or {
                            "relevant": True, "faithful": True, "passed": True})
    V.reviewer_node({"query": "q", "answer": "a", "filtered_evidence": [],
                     "iteration": 0, "trace": []})
    assert seen.get("cache_prefix") is False


def test_router_keeps_caching_on(monkeypatch):
    """The Router is the one that actually qualifies (1,375 tokens cached)."""
    import agents.router as R
    seen = {}
    monkeypatch.setattr(R, "call_claude_structured",
                        lambda **kw: seen.update(kw) or {
                            "intent": "general", "shaped_query": "q"})
    R.router_node({"query": "q", "trace": [], "attempted_queries": []})
    assert seen.get("cache_prefix", True) is True

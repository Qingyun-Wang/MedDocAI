"""
Per-query observability — agents/observability.py

Captures what every LLM-backed request actually costs: token usage, latency per
agent node, and an estimated dollar cost. Before this module the Anthropic
`response.usage` object was discarded, so the system had no idea what it spent.

Two instrumentation seams feed this module:
  1. agents/llm.py      -> record_llm_call()  (tokens + per-call latency)
  2. graph/pipeline.py  -> instrument_node()  (per-node latency)

WHY A ContextVar AND NOT PipelineState:
    PipelineState is a plain TypedDict with no LangGraph reducers, so a node
    returning `llm_calls: [...]` would OVERWRITE rather than append (the existing
    code hand-rolls read-concat-return for `trace`). The LLM call also happens in
    the middle of a node body, not at its return, so threading a record out would
    mean touching all six agent nodes and every test that monkeypatches them.
    A ContextVar accumulator keeps the agents and their tests untouched.

    CORRECTNESS CONSTRAINT: this is only sound because the graph is sequential
    and `.invoke()` runs in-thread. ContextVars isolate correctly across threads
    (a new thread starts at the default), which is what makes Streamlit's
    per-session ScriptRunner threads safe. If the graph ever gains parallel
    branches or async nodes, this must become a proper reducer channel.
"""

from __future__ import annotations

import functools
import logging
import time
from contextvars import ContextVar
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
# USD per MILLION tokens. Source: https://platform.claude.com/docs/en/about-claude/pricing
# Verified 2026-06-24 for claude-sonnet-4-5 (the model pinned in agents/llm.py):
#   base input $3, output $15, cache read (hits) $0.30, 5-minute cache write $3.75.
# Re-verify when changing models. An unpriced model yields cost=None (never a
# fake $0.00), so a stale table degrades visibly rather than silently.
MODEL_PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,   # 5-minute TTL
    },
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
}

# ---------------------------------------------------------------------------
# Per-query accumulators (see module docstring for the threading contract)
# ---------------------------------------------------------------------------

_calls: ContextVar[Optional[list]] = ContextVar("meddocai_calls", default=None)
_nodes: ContextVar[Optional[dict]] = ContextVar("meddocai_nodes", default=None)
_t0: ContextVar[Optional[float]] = ContextVar("meddocai_t0", default=None)
_qid: ContextVar[Optional[str]] = ContextVar("meddocai_qid", default=None)


def start_query(query_id: str) -> None:
    """Begin accumulating metrics for one pipeline run."""
    _calls.set([])
    _nodes.set({})
    _t0.set(time.perf_counter())
    _qid.set(query_id)


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Optional[float]:
    """Estimated USD cost for one call, or None if the model has no price entry.

    Returns None — not 0.0 — for an unknown model so the UI can honestly show
    "n/a" instead of implying the call was free.
    """
    price = MODEL_PRICING_USD_PER_MTOK.get(model)
    if not price:
        return None
    per_mtok = 1_000_000.0
    return (
        (input_tokens or 0) * price["input"] / per_mtok
        + (output_tokens or 0) * price["output"] / per_mtok
        + (cache_read_tokens or 0) * price.get("cache_read", 0.0) / per_mtok
        + (cache_write_tokens or 0) * price.get("cache_write", 0.0) / per_mtok
    )


def record_llm_call(
    *,
    model: str,
    caller: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    latency_ms: float = 0.0,
    ok: bool = True,
    error: Optional[str] = None,
) -> None:
    """Record one Claude call. No-op when no query is active (e.g. unit tests)."""
    calls = _calls.get()
    if calls is None:
        return
    calls.append({
        "caller": caller,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "latency_ms": round(latency_ms, 1),
        "ok": ok,
        "error": error,
        "cost_usd": estimate_cost_usd(
            model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
        ),
    })


def record_node(node: str, latency_ms: float) -> None:
    """Record time spent in one agent node (summed across corrective retries)."""
    nodes = _nodes.get()
    if nodes is None:
        return
    nodes[node] = round(nodes.get(node, 0.0) + latency_ms, 1)


def finish_query(query_id: str = "") -> dict:
    """Return the aggregated metrics for the run and clear the accumulators."""
    calls = _calls.get() or []
    nodes = _nodes.get() or {}
    t0 = _t0.get()
    total_ms = round((time.perf_counter() - t0) * 1000.0, 1) if t0 is not None else 0.0

    in_tok = sum(c["input_tokens"] for c in calls)
    out_tok = sum(c["output_tokens"] for c in calls)

    # None-safe: if ANY call used an unpriced model the total is None rather than
    # a partial sum that would silently understate the true cost.
    costs = [c["cost_usd"] for c in calls]
    cost_usd = round(sum(costs), 6) if calls and all(c is not None for c in costs) else None

    summary = {
        "query_id": query_id or (_qid.get() or ""),
        "total_latency_ms": total_ms,
        "llm_calls": len(calls),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "cost_usd": cost_usd,
        "by_node": dict(nodes),
        "calls": calls,
    }

    _calls.set(None)
    _nodes.set(None)
    _t0.set(None)
    _qid.set(None)
    return summary


# ---------------------------------------------------------------------------
# Node wrapper
# ---------------------------------------------------------------------------

def instrument_node(name: str, fn: Callable[[dict], Any]) -> Callable[[dict], Any]:
    """Wrap a LangGraph node to record its latency (and a LangSmith span).

    The wrapper MUST keep a single positional parameter: LangGraph inspects the
    callable's signature to decide whether to also inject a `config` argument.
    functools.wraps makes inspect.signature follow __wrapped__, and both the
    wrapper and every node take exactly one arg, so behaviour is unchanged.
    """
    try:
        from agents.tracing import traced
        inner = traced(f"node:{name}", run_type="chain")(fn)
    except Exception:   # tracing is optional — never let it break the pipeline
        inner = fn

    @functools.wraps(fn)
    def _wrapped(state):
        t0 = time.perf_counter()
        try:
            return inner(state)
        finally:
            record_node(name, (time.perf_counter() - t0) * 1000.0)

    return _wrapped


def format_summary(metrics: dict) -> str:
    """One-line human-readable summary (CLI + Streamlit caption)."""
    if not metrics:
        return ""
    secs = (metrics.get("total_latency_ms") or 0) / 1000.0
    calls = metrics.get("llm_calls", 0)
    toks = metrics.get("total_tokens", 0)
    parts = [f"{secs:.1f}s", f"{calls} LLM calls", f"{toks:,} tok"]
    cost = metrics.get("cost_usd")
    parts.append(f"${cost:.4f}" if cost is not None else "cost n/a")
    return " · ".join(parts)

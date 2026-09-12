"""
Shared Claude LLM helper — agents/llm.py

Two call styles used by the agents:
  - call_claude_structured(): forces JSON output via Anthropic tool-use.
    Used by Router and Reviewer (they need reliable structured decisions).
  - call_claude_text(): plain text generation.
    Used by Answer Generator.

A single shared AsyncAnthropic/Anthropic client is lazily created.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import anthropic
from dotenv import load_dotenv

from agents import observability, tracing

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            override=True)
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        # max_retries/timeout: the SDK retries 408/409/429/5xx with backoff.
        # maybe_wrap_anthropic is a no-op unless tracing is explicitly enabled.
        _client = tracing.maybe_wrap_anthropic(
            anthropic.Anthropic(api_key=key, max_retries=3, timeout=60.0)
        )
    return _client


def _timed_create(client, *, caller: str, **kwargs):
    """Call messages.create, recording latency + token usage either way.

    The Anthropic response carries `usage.input_tokens` / `usage.output_tokens`,
    which this codebase previously discarded — that is what made cost tracking
    impossible. Read defensively via getattr: Usage field names have shifted
    historically, and a metrics miss must never break an answer.
    """
    t0 = time.perf_counter()
    model = kwargs.get("model", "")
    cache_expected = kwargs.pop("_cache_expected", False)   # ours, not an API param
    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        observability.record_llm_call(
            model=model, caller=caller,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            ok=False, error=str(exc)[:200],
        )
        raise
    latency_ms = (time.perf_counter() - t0) * 1000.0
    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

    # A prefix shorter than the model's minimum is NOT an error — the API just
    # ignores the marker and returns zeros for both counters. That failure is
    # invisible: the bill quietly rises ~10% and nothing breaks. The Reviewer's
    # prefix sits only ~116 tokens above Sonnet 4.5's 1024 minimum, so a future
    # prompt trim could cross it. Make it audible where it happens.
    if cache_expected and not (cache_read or cache_write):
        logger.warning(
            "caching requested for %s but the response reported neither a cache "
            "read nor a write — the prefix is probably below the model minimum",
            caller or "?",
        )
    observability.record_llm_call(
        model=model, caller=caller,
        input_tokens=in_tok, output_tokens=out_tok,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        latency_ms=latency_ms,
    )
    logger.debug("llm %s model=%s in=%d out=%d %.0fms",
                 caller or "?", model, in_tok, out_tok, latency_ms)
    return response


# ---------------------------------------------------------------------------
# Structured output (tool-use forces a JSON schema)
# ---------------------------------------------------------------------------

def call_claude_structured(
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict,
    max_tokens: int = 1024,
    model: str = CLAUDE_MODEL,
    caller: str = "",
    cache_prefix: bool = True,
) -> dict:
    """Call Claude and force a structured JSON response matching input_schema.

    Uses Anthropic tool-use with tool_choice forced — guarantees the model
    returns arguments matching the schema (no free-text parsing needed).

    Returns the tool input dict (the structured result).
    """
    client = _get_client()
    # PROMPT CACHING (C2b). The cached prefix renders `tools` -> `system`, so the
    # marker on the system block covers BOTH. MEASURED IN PRODUCTION (not with
    # count_tokens, which overstated it by ~19%): the Router caches 1,375 tokens,
    # cutting its billed input from ~1,931 to ~440. The REVIEWER does not qualify —
    # its prefix lands just under Sonnet 4.5's 1,024-token minimum, so it opts out
    # via cache_prefix=False rather than marking a block the API silently ignores.
    # These prefixes are byte-identical
    # on every query, unlike the retrieved evidence (which is unique per question
    # and only repeats on the ~19% of queries whose retry reuses it — measured to
    # be worth +1.3%, i.e. nothing, once the write premium is paid by the 71% that
    # never retry).
    #
    # A read refreshes the 5-minute TTL, so traffic arriving under 5 minutes apart
    # keeps the entry warm indefinitely and pays the write only once. Break-even is
    # ~22% of calls landing warm. Caches are workspace-scoped, so concurrent users
    # of the deployed Space share one entry rather than each paying their own write.
    response = _timed_create(
        client,
        caller=caller,
        model=model,
        max_tokens=max_tokens,
        system=([{"type": "text", "text": system,
                  "cache_control": {"type": "ephemeral"}}]
                if cache_prefix else system),
        tools=[{
            "name": tool_name,
            "description": tool_description,
            "input_schema": input_schema,
        }],
        _cache_expected=cache_prefix,
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    # Should not happen with forced tool_choice, but fail gracefully
    raise RuntimeError("Claude did not return a tool_use block")


# ---------------------------------------------------------------------------
# Plain text generation
# ---------------------------------------------------------------------------

def call_claude_text(
    system: str,
    user: str,
    max_tokens: int = 1500,
    model: str = CLAUDE_MODEL,
    caller: str = "",
) -> str:
    """Call Claude for a plain-text response.

    Deliberately NOT prompt-cached. This is the Answer Generator, whose system
    prompts measure 85 and 78 tokens — far below Sonnet 4.5's 1024-token minimum
    cacheable prefix. Marking them would be silently ignored (no error, both cache
    counters zero) while still costing the write premium on anything that did
    qualify. Its bulk is the evidence, which lives in the user message and differs
    per question. See call_claude_structured for the paths that DO cache.
    """
    client = _get_client()
    response = _timed_create(
        client,
        caller=caller,
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [b.text for b in response.content if b.type == "text"]
    return "\n".join(parts).strip()

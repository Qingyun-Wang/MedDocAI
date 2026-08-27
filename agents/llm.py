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
    observability.record_llm_call(
        model=model, caller=caller,
        input_tokens=in_tok, output_tokens=out_tok,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
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
) -> dict:
    """Call Claude and force a structured JSON response matching input_schema.

    Uses Anthropic tool-use with tool_choice forced — guarantees the model
    returns arguments matching the schema (no free-text parsing needed).

    Returns the tool input dict (the structured result).
    """
    client = _get_client()
    response = _timed_create(
        client,
        caller=caller,
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[{
            "name": tool_name,
            "description": tool_description,
            "input_schema": input_schema,
        }],
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
    """Call Claude for a plain-text response."""
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

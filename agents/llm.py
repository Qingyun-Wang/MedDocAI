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
from typing import Optional

import anthropic
from dotenv import load_dotenv

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
        _client = anthropic.Anthropic(api_key=key)
    return _client


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
) -> dict:
    """Call Claude and force a structured JSON response matching input_schema.

    Uses Anthropic tool-use with tool_choice forced — guarantees the model
    returns arguments matching the schema (no free-text parsing needed).

    Returns the tool input dict (the structured result).
    """
    client = _get_client()
    response = client.messages.create(
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
) -> str:
    """Call Claude for a plain-text response."""
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [b.text for b in response.content if b.type == "text"]
    return "\n".join(parts).strip()

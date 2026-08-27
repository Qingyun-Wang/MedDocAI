"""
LangSmith tracing (optional) — agents/tracing.py

The agents call the raw Anthropic SDK, not langchain_anthropic, so LangSmith is
wired via `wrap_anthropic` on the shared client plus `@traced` spans on the
pipeline entry point and each graph node. That gives one trace per query with
nested node spans and real token counts — including a visible second
`node:answer_generator` span when the corrective-RAG loop retries.

ENABLEMENT REQUIRES BOTH A FLAG AND A KEY.
    Key-presence alone is deliberately NOT enough: a stray LANGCHAIN_API_KEY in
    someone's shell must never silently ship healthcare prompts to a SaaS. This
    is also what guarantees the no-op in unit tests, in CI, and on HF Spaces.

    The guard is evaluated at IMPORT time (decorators apply at import), so
    setting the env var after import has no effect. That is fine here: every
    entry point loads .env via load_dotenv(override=True) before importing the
    agents.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def tracing_enabled() -> bool:
    """True only when tracing is explicitly switched on AND a key is present."""
    flag = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower()
    if flag not in _TRUTHY:
        return False
    return bool(
        (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    )


def maybe_wrap_anthropic(client: Any) -> Any:
    """Return the client wrapped for LangSmith, or unchanged when disabled.

    `wrap_anthropic` mutates the client INSTANCE (it returns the same object with
    an instrumented `.messages.create`), so this cannot leak into the separate
    AsyncAnthropic client used by the offline batch pipeline.
    """
    if not tracing_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_anthropic
        wrapped = wrap_anthropic(client)
        logger.info("LangSmith tracing enabled (project=%s)",
                    os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "default")
        return wrapped
    except Exception as exc:   # noqa: BLE001 — tracing must never break serving
        logger.warning("LangSmith wrap failed (%s) — continuing untraced", exc)
        return client


def traced(name: str, run_type: str = "chain") -> Callable:
    """Decorator: langsmith.traceable when enabled, identity otherwise."""
    if not tracing_enabled():
        return lambda fn: fn
    try:
        from langsmith import traceable
        return traceable(name=name, run_type=run_type)
    except Exception as exc:   # noqa: BLE001
        logger.warning("LangSmith traceable unavailable (%s) — continuing untraced", exc)
        return lambda fn: fn

"""
RAGAS / LangChain 1.x compatibility shim — evaluation/ragas_compat.py

Problem: ragas 0.4.3 (latest) still imports Google Vertex AI classes from
`langchain_community`, but langchain-community 0.4.x (the sunset release that
is compatible with our langchain-core 1.x / langgraph 1.x stack) removed them:

    from langchain_community.chat_models.vertexai import ChatVertexAI   # gone
    from langchain_community.llms import VertexAI                       # gone

Downgrading langchain-community would force langchain-core < 1.0 and break the
agent pipeline (langgraph 1.2 / langchain-anthropic 1.4 require core 1.x).

Fix: stub the two missing Vertex AI symbols BEFORE importing ragas. We never
use Vertex AI — the stubs exist only so ragas's module-level imports succeed.
Everything we actually use (OpenAI judge LLM, the metric collections) is real.

Usage — import this module FIRST, before any ragas import:

    import evaluation.ragas_compat  # noqa: F401  (must precede ragas imports)
    from ragas.metrics.collections import Faithfulness, ...
"""

from __future__ import annotations

import sys
import types
import warnings


def _install_stubs() -> None:
    # Stub module: langchain_community.chat_models.vertexai
    modname = "langchain_community.chat_models.vertexai"
    if modname not in sys.modules:
        mod = types.ModuleType(modname)
        mod.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[modname] = mod

    # Patch missing attribute: langchain_community.llms.VertexAI
    try:
        import langchain_community.llms as llms_mod
        if not hasattr(llms_mod, "VertexAI"):
            llms_mod.VertexAI = type("VertexAI", (), {})
    except ImportError:
        pass

    # Quiet the deprecation chatter from the sunset langchain-community
    # and ragas's old-import warnings — not actionable for us.
    warnings.filterwarnings(
        "ignore", message=".*langchain-community.*", category=DeprecationWarning
    )


_install_stubs()

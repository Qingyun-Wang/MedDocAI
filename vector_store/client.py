"""
Qdrant client factory — vector_store/client.py

One place that decides HOW to connect to Qdrant, so the same code runs both:
  - Local dev:  embedded file-based Qdrant at data/qdrant_storage  (no server)
  - Docker:     a Qdrant SERVER over HTTP (set QDRANT_URL=http://qdrant:6333)

Selection is by environment variable:
  - QDRANT_URL set  -> server mode (QdrantClient(url=...))
  - otherwise       -> local embedded mode (QdrantClient(path=QDRANT_PATH))

Server mode also removes the single-locker limitation of local mode (only one
process can open the embedded store at a time).
"""

from __future__ import annotations

import os

from qdrant_client import QdrantClient

# Default local embedded store (overridable via env for non-standard layouts)
QDRANT_PATH = os.getenv("QDRANT_PATH", "data/qdrant_storage")


def get_qdrant_client() -> QdrantClient:
    """Return a QdrantClient connected per environment.

    QDRANT_URL set  -> server mode (e.g. http://qdrant:6333 in Docker, or
                       http://localhost:6333 for a locally-run server).
    unset           -> embedded local mode at QDRANT_PATH (dev default).
    """
    url = os.getenv("QDRANT_URL", "").strip()
    if url:
        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        return QdrantClient(url=url, api_key=api_key, timeout=60)
    return QdrantClient(path=QDRANT_PATH)


def is_server_mode() -> bool:
    """True if configured to use a Qdrant server (QDRANT_URL set)."""
    return bool(os.getenv("QDRANT_URL", "").strip())

"""
Qdrant local -> server migration — scripts/migrate_qdrant_to_server.py

Copies vectors + payloads from the LOCAL embedded Qdrant store
(data/qdrant_storage, created during Phase 1 ingestion) into a Qdrant SERVER,
WITHOUT re-embedding. This is the "snapshot migrate" path for the Docker setup:
the server starts with an empty volume, and this script seeds it from the
vectors you already paid OpenAI to compute.

It is idempotent and safe to re-run:
  - If a destination collection already holds >= the source's point count,
    it is left untouched (assumed already migrated).
  - Otherwise the collection is (re)created with the SAME vector config as the
    source, and all points are scrolled across and upserted in batches.

Selection of source / destination is explicit (NOT via vector_store.client,
because that factory returns only one client per process):
  - Source:      QDRANT_PATH        (default: data/qdrant_storage)
  - Destination: QDRANT_URL         (default: http://localhost:6333)
                 QDRANT_API_KEY     (optional)

Usage:
    # local store -> local/remote server
    QDRANT_URL=http://localhost:6333 python scripts/migrate_qdrant_to_server.py

    # inside Docker the entrypoint runs this with QDRANT_URL=http://qdrant:6333
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add project root to path so this runs from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Collections to migrate (must match the ingestion code)
COLLECTIONS = ["drug_labels", "medlineplus"]

SCROLL_BATCH = 256   # points per scroll page / upsert batch


def _source_client() -> QdrantClient:
    path = os.getenv("QDRANT_PATH", "data/qdrant_storage")
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Local Qdrant store not found at '{path}'. "
            "Run Phase 1 ingestion first, or set QDRANT_PATH."
        )
    return QdrantClient(path=path)


def _dest_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    return QdrantClient(url=url, api_key=api_key, timeout=120)


def _wait_for_server(client: QdrantClient, retries: int = 30, delay: float = 2.0) -> None:
    """Block until the destination server answers, or give up."""
    for attempt in range(1, retries + 1):
        try:
            client.get_collections()
            return
        except Exception as exc:  # noqa: BLE001 — broad on purpose during startup
            if attempt == retries:
                raise
            print(f"  [wait] server not ready ({exc}); retry {attempt}/{retries} in {delay}s")
            time.sleep(delay)


def _migrate_collection(src: QdrantClient, dst: QdrantClient, name: str, force: bool) -> None:
    # Does the source even have this collection?
    src_names = {c.name for c in src.get_collections().collections}
    if name not in src_names:
        print(f"  [skip] '{name}' absent from local store — nothing to migrate")
        return

    src_count = src.count(name, exact=True).count
    dst_names = {c.name for c in dst.get_collections().collections}

    # Idempotency: already populated?
    if name in dst_names and not force:
        dst_count = dst.count(name, exact=True).count
        if dst_count >= src_count:
            print(f"  [skip] '{name}' already on server ({dst_count} >= {src_count} pts)")
            return
        print(f"  [redo] '{name}' partial on server ({dst_count} < {src_count}); re-migrating")

    # (Re)create destination with the SAME vector config as the source.
    # delete + create rather than the deprecated recreate_collection (removed in 2.x).
    src_info = src.get_collection(name)
    vectors_config = src_info.config.params.vectors
    if name in dst_names:
        dst.delete_collection(collection_name=name)
    dst.create_collection(collection_name=name, vectors_config=vectors_config)
    print(f"  [make] '{name}' created on server (config copied from source)")

    # Scroll all points (with vectors + payload) and upsert in batches
    migrated = 0
    offset = None
    while True:
        points, offset = src.scroll(
            collection_name=name,
            limit=SCROLL_BATCH,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        if not points:
            break

        dst.upsert(
            collection_name=name,
            points=[
                PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )
        migrated += len(points)
        print(f"    ... {migrated}/{src_count} points", end="\r", flush=True)

        if offset is None:   # no more pages
            break

    print(f"\n  [done] '{name}': {migrated} points migrated")


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate local Qdrant vectors to a server.")
    ap.add_argument("--force", action="store_true",
                    help="Re-migrate even if the server collection already looks complete.")
    ap.add_argument("--collections", nargs="*", default=COLLECTIONS,
                    help=f"Collections to migrate (default: {COLLECTIONS}).")
    args = ap.parse_args()

    print("Qdrant migration: local embedded store -> server")
    print(f"  source (QDRANT_PATH): {os.getenv('QDRANT_PATH', 'data/qdrant_storage')}")
    print(f"  dest   (QDRANT_URL):  {os.getenv('QDRANT_URL', 'http://localhost:6333')}")

    src = _source_client()
    dst = _dest_client()
    _wait_for_server(dst)

    try:
        for name in args.collections:
            _migrate_collection(src, dst, name, force=args.force)
    finally:
        src.close()
        dst.close()

    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

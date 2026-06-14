#!/usr/bin/env bash
# MedDocAI container entrypoint.
#
# 1. (Optional) Seed the Qdrant SERVER from the bundled local embedded store.
#    Controlled by SEED_ON_BOOT:
#      - true  (default): local docker-compose — seeds the empty server volume
#        once, idempotent on later boots.
#      - false: hosted deploy (e.g. HF Spaces + Qdrant Cloud) where the managed
#        cluster is already seeded out-of-band, and no local store is shipped.
# 2. Launch the Streamlit app.
#
# The migration script itself waits for the Qdrant server to become reachable,
# so we don't need a separate wait loop here.
set -euo pipefail

SEED_ON_BOOT="${SEED_ON_BOOT:-true}"

if [ "$SEED_ON_BOOT" = "true" ]; then
    echo "[entrypoint] Seeding Qdrant server from local store (idempotent)..."
    if python scripts/migrate_qdrant_to_server.py; then
        echo "[entrypoint] Qdrant seed step complete."
    else
        # Don't hard-fail the whole app if the local store is absent (e.g. someone
        # ingested straight into the server). Log and continue — the app will still
        # work if the server already has the collections.
        echo "[entrypoint] WARNING: migration step failed or was skipped; continuing."
    fi
else
    echo "[entrypoint] SEED_ON_BOOT=false — skipping seed (assuming managed/seeded Qdrant)."
fi

PORT="${PORT:-8501}"
echo "[entrypoint] Starting Streamlit on 0.0.0.0:${PORT} ..."
exec streamlit run frontend/app.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

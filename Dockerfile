# MedDocAI — application image (Streamlit UI + LangGraph agent pipeline)
#
# Qdrant runs as a SEPARATE service (see docker-compose.yml); this image only
# builds the Python app and connects to it over HTTP via QDRANT_URL.
#
# Build:  docker compose build
# Run:    docker compose up

FROM python:3.11-slim

# --- System deps -----------------------------------------------------------
# build-essential: some wheels (qdrant-client/grpc, tokenizers) may compile.
# curl: used by the healthcheck / entrypoint wait loop.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Keep Python lean and unbuffered for clean container logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- Python deps (layer-cached separately from source) ---------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Application source -----------------------------------------------------
COPY . .

# Entrypoint seeds the Qdrant server (once) then launches Streamlit
RUN chmod +x /app/docker-entrypoint.sh \
    # Make the SQLite dir writable for any runtime UID (e.g. HF Spaces non-root):
    # the app writes chat_history back to data/meddocai.db (+ -wal/-journal files).
    && chmod -R a+rwX /app/data 2>/dev/null || true

EXPOSE 8501

# Streamlit healthcheck endpoint (PORT defaults to 8501; HF maps app_port)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]

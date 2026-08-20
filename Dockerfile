# ---------------------------------------------------------------------------
# Personal Knowledge Agent - backend (LangGraph dev server)
# Base: python:3.13-slim. Installs the project editable + langgraph CLI.
# Runtime config (API keys, QDRANT_URL, vault path) is supplied by
# docker-compose via environment variables - NO secrets are baked in.
# ---------------------------------------------------------------------------
FROM python:3.13-slim

# PYTHONUTF8=1: avoid GBK/UTF-8 errors on Chinese paths.
# PYTHONPATH=/app: makes `import src.agent...` resolve from the project root.
ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Build toolchain for any source-only wheels (removed from the layer cache).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only what is needed to install dependencies first (better layer cache).
COPY pyproject.toml ./
COPY src ./src
COPY langgraph.json ./

# Editable install of the project + the LangGraph dev server.
# [inmem] provides the in-memory checkpointer referenced by langgraph.json.
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install "langgraph-cli[inmem]" "langgraph-api"

# Entrypoint waits for Qdrant (if QDRANT_URL is set), then launches the server.
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Vault is mounted at runtime (see docker-compose.yml). Safe default.
ENV OBSIDIAN_VAULT_PATH=/app/vault

EXPOSE 3001

ENTRYPOINT ["/app/docker-entrypoint.sh"]

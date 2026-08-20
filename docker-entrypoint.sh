#!/bin/sh
# Entrypoint for the backend container.
# 1. Ensures an (empty) .env exists so langgraph.json's "env": ".env" is happy.
# 2. Waits for Qdrant when running in server mode (QDRANT_URL is set).
# 3. Launches `langgraph dev` bound to all interfaces on port 3001.
set -e

echo "==> Personal Knowledge Agent backend starting"

# langgraph.json declares "env": ".env". Make sure the file exists so the dev
# server does not error on a missing env file. It stays empty; real config
# comes from the process environment (supplied by docker-compose).
test -f /app/.env || touch /app/.env

# Wait for Qdrant when running in server mode (docker-compose sets QDRANT_URL).
if [ -n "$QDRANT_URL" ]; then
  HOST_PORT="${QDRANT_URL#*://}"
  HOST="${HOST_PORT%%:*}"
  PORT="${HOST_PORT##*:}"
  if [ "$PORT" = "$HOST_PORT" ]; then
    PORT=6333
  fi
  echo "==> Waiting for Qdrant at ${HOST}:${PORT} ..."
  i=0
  while [ "$i" -lt 60 ]; do
    if python - <<PY
import socket, sys
host, port = "${HOST}", ${PORT}
s = socket.socket()
s.settimeout(2)
try:
    s.connect((host, port))
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
    then
      echo "==> Qdrant is reachable."
      break
    fi
    i=$((i + 1))
    sleep 2
  done
  if [ "$i" -ge 60 ]; then
    echo "==> WARNING: Qdrant not reachable after 120s, continuing anyway."
  fi
fi

echo "==> Launching langgraph dev on 0.0.0.0:3001"
exec langgraph dev --host 0.0.0.0 --port 3001

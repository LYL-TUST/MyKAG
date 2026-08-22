"""API server bootstrap module for Personal Knowledge Agent.

This module is the ``http.app`` entry point wired in ``langgraph.json``, so it
is loaded as soon as the dev server boots.

NOTE: LLM warm-up is NOT wired here. The previous implementation used
``@app.on_event("startup")`` which conflicts with langgraph-api 0.12.3's
lifespan validation (``ValueError: Cannot merge lifespans with on_startup``)
and crashed the dev server. Warm-up now lives in ``src/agent/warmup.py`` and
is fired from ``router_graph.py`` module load (a daemon thread), which the
dev server necessarily imports, so the models are still pre-heated during
startup with no FastAPI lifespan involvement.
"""
from __future__ import annotations

from src.api.fastapi_app import app

__all__ = ["app"]

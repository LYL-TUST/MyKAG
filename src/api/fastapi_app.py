"""FastAPI application for Personal Knowledge Agent.

Provides health check, session title generation, cache management, and
vault note browsing endpoints.
"""

from __future__ import annotations

import logging
import os

import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env early so module-level constants below see OBSIDIAN_VAULT_PATH.
# knowledge_graph.py also loads it via config.py, but importing fastapi_app
# standalone (or in a different order) would otherwise see empty os.environ.
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personal Knowledge Agent API",
    description="API for the Obsidian-powered personal knowledge management agent",
    version="1.0.0",
)

# CORS configuration
# NOTE: keep in sync with frontend dev ports (3000 dev / 3001 dev:remote)
# and with CORS_ORIGINS in .env / .env.example.
_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:3001,http://127.0.0.1:3001,"
    "https://smith.langchain.com",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _ALLOWED_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH", ".")


def _get_vault_path() -> str:
    """Return the current vault path.

    Read live from the environment (not a module constant): POST /config
    hot-switches OBSIDIAN_VAULT_PATH, so endpoints that serve note content
    must always resolve it dynamically.
    """
    return os.environ.get("OBSIDIAN_VAULT_PATH", _VAULT_PATH)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "personal-knowledge-agent",
        "version": "1.0.0",
    }


@app.get("/notes")
async def list_notes() -> dict:
    """List all notes in the vault with metadata."""
    import asyncio
    from src.rag.ingestion import VaultIngestionPipeline
    from src.rag.wikilink import get_graph

    try:
        vault_path = _get_vault_path()
        logger.info(f"[notes] loading from vault: {vault_path}")
        if not os.path.isdir(vault_path):
            raise FileNotFoundError(f"Vault directory not found: {vault_path}")
        # VaultIngestionPipeline.ingest_all is sync (pathlib glob + read).
        # LangGraph dev rejects blocking calls in async context, so offload
        # the work to a worker thread.
        pipeline = VaultIngestionPipeline(vault_path)
        docs = await asyncio.to_thread(pipeline.ingest_all)
        logger.info(f"[notes] ingested {len(docs)} documents")
    except Exception as e:
        import traceback
        logger.error(f"[notes] list failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Vault load failed: {e}")

    graph = get_graph()

    notes = []
    for doc in docs:
        notes.append({
            "name": doc.note_name,
            "title": doc.title,
            "tags": doc.tags,
            "wikilink_count": len(doc.wikilinks),
            "file_path": doc.file_path,
        })

    return {
        "notes": notes,
        "total": len(notes),
        "graph_stats": {
            "nodes": graph.total_notes(),
            "links": graph.total_links(),
        },
    }


@app.get("/notes/{name}")
async def get_note(name: str) -> dict:
    """Get a single note's full content and metadata."""
    import asyncio
    from src.rag.ingestion import VaultIngestionPipeline
    from src.rag.wikilink import get_graph

    graph = get_graph()
    file_path = graph.get_note_path(name)
    if file_path is None:
        raise HTTPException(status_code=404, detail=f"Note not found: {name}")

    try:
        pipeline = VaultIngestionPipeline(_get_vault_path())
        doc = await asyncio.to_thread(pipeline.ingest_single, file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Note not found: {name}")

    return {
        "name": doc.note_name,
        "title": doc.title,
        "tags": doc.tags,
        "wikilinks": doc.wikilinks,
        "content": doc.body,
        "frontmatter": doc.frontmatter,
        "file_path": doc.file_path,
    }


@app.get("/notes/{name}/graph")
async def get_note_graph(name: str) -> dict:
    """Get a note's local [[wikilink]] graph (out-links and in-links)."""
    from src.rag.wikilink import get_graph

    graph = get_graph()
    if graph.get_note_path(name) is None:
        raise HTTPException(status_code=404, detail=f"Note not found: {name}")

    return graph.get_local_graph(name)


@app.get("/graph")
async def get_full_graph() -> dict:
    """Get the full vault [[wikilink]] graph for visualization."""
    from src.rag.wikilink import get_graph

    graph = get_graph()
    data = graph.export()
    return {
        **data,
        "total_nodes": len(data["nodes"]),
        "total_edges": len(data["edges"]),
    }


def _persist_vault_path(path: str) -> None:
    """Write OBSIDIAN_VAULT_PATH back to .env (preserving other lines)."""
    env_path = ".env"
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(env_path, "w", encoding="utf-8") as f:
            for line in lines:
                if line.startswith("OBSIDIAN_VAULT_PATH="):
                    f.write(f"OBSIDIAN_VAULT_PATH={path}\n")
                else:
                    f.write(line)
    except Exception as e:
        logger.warning(f"Failed to persist vault path to .env: {e}")


def _switch_vault(new_path: str) -> None:
    """Reset and re-index the vault at the new path (runs in a worker thread)."""
    from src.tools.vault_tools import _init_vault, _reset_vault

    _reset_vault()
    _init_vault(new_path, force_rebuild=True)


@app.get("/config")
async def get_config() -> dict:
    """Get current configuration (vault path and graph stats)."""
    from src.rag.wikilink import get_graph
    from src.tools.vault_tools import _vault_indexer

    graph = get_graph()
    return {
        "vault_path": os.environ.get("OBSIDIAN_VAULT_PATH", ""),
        "initialized": _vault_indexer is not None,
        "graph_stats": {
            "nodes": graph.total_notes(),
            "links": graph.total_links(),
        },
    }


@app.post("/config")
async def update_config(payload: dict) -> dict:
    """Update the vault path and re-index (hot switch, no restart needed)."""
    import asyncio

    new_path = (payload.get("vault_path") or "").strip()
    if not new_path:
        raise HTTPException(status_code=400, detail="vault_path is required")
    if not os.path.isdir(new_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {new_path}")

    os.environ["OBSIDIAN_VAULT_PATH"] = new_path
    _persist_vault_path(new_path)

    await asyncio.to_thread(_switch_vault, new_path)

    from src.rag.wikilink import get_graph

    graph = get_graph()
    return {
        "status": "ok",
        "vault_path": new_path,
        "graph_stats": {
            "nodes": graph.total_notes(),
            "links": graph.total_links(),
        },
    }


@app.post("/reindex")
async def reindex_vault() -> dict:
    """Force a full vault re-index (drops and rebuilds the vector index).

    Use this when the index has drifted from the vault (e.g. you added new
    notes that aren't being recalled, or switched the vault path). The
    underlying Qdrant collection is deleted and rebuilt from scratch; the
    BM25 index is rebuilt as well. Safe to call while the server is running.
    """
    import asyncio

    def _do_reindex() -> None:
        from src.tools.vault_tools import _init_vault, _reset_vault

        _reset_vault()
        vault_path = os.environ.get("OBSIDIAN_VAULT_PATH", _VAULT_PATH)
        _init_vault(vault_path, force_rebuild=True)

    await asyncio.to_thread(_do_reindex)

    from src.rag.wikilink import get_graph

    graph = get_graph()
    return {
        "status": "reindexed",
        "vault_path": os.environ.get("OBSIDIAN_VAULT_PATH", ""),
        "graph_stats": {
            "nodes": graph.total_notes(),
            "links": graph.total_links(),
        },
    }


class TitleRequest(BaseModel):
    """Payload sent by the frontend for conversation title generation.

    The frontend POSTs ``{userMessage, assistantResponse, maxLength}`` (see
    ``frontend/lib/utils/string/string-helpers.ts``), NOT a ``messages`` list.
    ``assistantResponse`` is accepted for future AI-based titles but is not
    currently used for the truncation fallback.
    """

    userMessage: str = ""
    assistantResponse: str | None = None
    maxLength: int = 60


def _truncate_title(text: str, max_len: int | None) -> str:
    """Return a trimmed, truncated title; empty input -> placeholder."""
    text = (text or "").strip()
    if not text:
        return "New Conversation"
    # Clamp to a sane range so a bad maxLength can't produce empty/huge titles.
    if max_len is None:
        max_len = 60
    max_len = max(1, min(int(max_len), 200))
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


@app.post("/generate-title")
async def generate_title(payload: TitleRequest | list[dict]) -> dict:
    """Generate a conversation title.

    Accepts the frontend's object payload ``{userMessage, assistantResponse,
    maxLength}`` (the primary contract). For backward compatibility, a raw
    ``messages`` list is still tolerated and uses the first user message.
    """
    # Legacy list format: pick the first user message.
    if isinstance(payload, list):
        first_user_msg = ""
        for msg in payload:
            if isinstance(msg, dict) and msg.get("role") == "user":
                first_user_msg = msg.get("content", "")
                break
        return {"title": _truncate_title(first_user_msg, 50)}

    # Primary object format. Prefer userMessage; fall back to assistantResponse.
    source = payload.userMessage or payload.assistantResponse or ""
    return {"title": _truncate_title(source, payload.maxLength)}


@app.get("/metrics/cache")
async def cache_metrics() -> dict:
    """Get cache statistics."""
    from src.tools.redis import RedisCache
    try:
        return RedisCache.get_stats()
    except Exception:
        return {"size": 0, "max_size": 0, "hits": 0, "misses": 0}


@app.post("/metrics/cache/clear")
async def clear_cache() -> dict:
    """Clear the in-memory cache."""
    from src.tools.redis import RedisCache
    try:
        RedisCache.clear()
        return {"status": "cleared"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

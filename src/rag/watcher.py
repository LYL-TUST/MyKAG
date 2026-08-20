"""File system watcher for incremental Obsidian vault index updates.

Monitors the vault directory for .md file changes (create/modify/delete),
and triggers incremental index updates without a full rebuild.

Uses watchdog for cross-platform file monitoring.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_WATCHER_RUNNING = False
_WATCHER_THREAD: Optional[threading.Thread] = None


class VaultFileWatcher:
    """Watches an Obsidian vault for .md file changes.

    When files change, triggers incremental re-indexing for the
    affected documents only.
    """

    def __init__(
        self,
        vault_path: str,
        on_create: callable = None,
        on_modify: callable = None,
        on_delete: callable = None,
        debounce_seconds: float = 2.0,
    ) -> None:
        self._vault_path = Path(vault_path)
        self._on_create = on_create
        self._on_modify = on_modify
        self._on_delete = on_delete
        self._debounce_seconds = debounce_seconds
        self._observer = None
        self._pending: dict = {}  # path -> event_type, debounced

    def start(self) -> None:
        """Start the file watcher in a background thread."""
        global _WATCHER_RUNNING, _WATCHER_THREAD

        if _WATCHER_RUNNING:
            logger.warning("Watcher already running")
            return

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning(
                "watchdog not installed. File watching disabled. "
                "Install with: pip install watchdog"
            )
            return

        class _VaultEventHandler(FileSystemEventHandler):
            def __init__(self, watcher):
                self._w = watcher

            def on_created(self, event):
                if event.src_path.endswith(".md"):
                    logger.info(f"[watcher] Created: {event.src_path}")
                    if self._w._on_create:
                        self._w._on_create(event.src_path)

            def on_modified(self, event):
                if event.src_path.endswith(".md"):
                    logger.info(f"[watcher] Modified: {event.src_path}")
                    if self._w._on_modify:
                        self._w._on_modify(event.src_path)

            def on_deleted(self, event):
                if event.src_path.endswith(".md"):
                    logger.info(f"[watcher] Deleted: {event.src_path}")
                    if self._w._on_delete:
                        self._w._on_delete(event.src_path)

        self._observer = Observer()
        handler = _VaultEventHandler(self)
        self._observer.schedule(
            handler, str(self._vault_path), recursive=True,
        )
        self._observer.start()

        _WATCHER_RUNNING = True
        _WATCHER_THREAD = threading.current_thread()

        logger.info(
            f"File watcher started for: {self._vault_path} "
            f"(debounce: {self._debounce_seconds}s)"
        )

    def stop(self) -> None:
        """Stop the file watcher."""
        global _WATCHER_RUNNING
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        _WATCHER_RUNNING = False
        logger.info("File watcher stopped")


def create_vault_watcher(vault_path: str) -> VaultFileWatcher:
    """Create a file watcher connected to the vault RAG pipeline.

    On file create/modify: re-ingest the affected document and update indexes.
    On file delete: remove the document from the wiki graph (vector cleanup
    requires rebuild, which is deferred).

    Args:
        vault_path: Absolute path to the Obsidian vault.

    Returns:
        Configured VaultFileWatcher.
    """
    from src.rag.ingestion import VaultIngestionPipeline
    from src.rag.wikilink import build_graph_from_vault

    pipeline = VaultIngestionPipeline(vault_path)

    def on_file_changed(file_path: str) -> None:
        """Handle file create or modify: re-ingest and re-index."""
        try:
            doc = pipeline.ingest_single(file_path)
            if doc is None:
                return

            chunks = pipeline.documents_to_chunks([doc])
            if not chunks:
                return

            logger.info(f"Re-indexed: {doc.note_name} ({len(chunks)} chunks)")

            # Update global indexer and retriever
            from src.tools.vault_tools import _vault_indexer, _vault_retriever
            if _vault_indexer is not None and _vault_indexer.index is not None:
                _vault_indexer.insert_documents(chunks)

            # Rebuild wiki-link graph (fast, reads all .md files)
            build_graph_from_vault(vault_path)

        except Exception:
            logger.warning(f"Failed to re-index: {file_path}", exc_info=True)

    def on_file_deleted(file_path: str) -> None:
        """Handle file delete: update wiki graph only."""
        try:
            logger.info(f"File deleted, rebuilding wiki graph: {file_path}")
            build_graph_from_vault(vault_path)
            # Vector index cleanup requires rebuild
            logger.info(
                "Note: vector index entry retained. Run full re-index to clean up."
            )
        except Exception:
            logger.warning(f"Failed to handle deletion: {file_path}", exc_info=True)

    watcher = VaultFileWatcher(
        vault_path=vault_path,
        on_create=on_file_changed,
        on_modify=on_file_changed,
        on_delete=on_file_deleted,
        debounce_seconds=2.0,
    )

    return watcher


def is_watcher_running() -> bool:
    """Check if the vault file watcher is running."""
    return _WATCHER_RUNNING

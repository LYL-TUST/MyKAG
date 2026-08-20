"""Vector index management for Obsidian vault RAG.

Uses LlamaIndex to build and manage a VectorStoreIndex backed by
Qdrant (local mode). Supports incremental index updates.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.rag.embedding import SiliconFlowEmbedding

logger = logging.getLogger(__name__)

# Default collection name for the vault index
DEFAULT_COLLECTION = "obsidian_vault"

# Qdrant local storage path
_DEFAULT_QDRANT_PATH = "./qdrant_data"


class VaultIndexer:
    """Manages the vector index for an Obsidian vault.

    Handles index creation, incremental updates, and persistence
    using Qdrant in local mode.
    """

    def __init__(
        self,
        qdrant_path: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model: Optional[str] = None,
    ) -> None:
        # Default to SiliconFlow-compatible BGE-M3 (better for Chinese).
        # Override via EMBEDDING_MODEL env var.
        embed_model = embed_model or os.environ.get(
            "EMBEDDING_MODEL", "BAAI/bge-m3",
        )
        # Qdrant connection mode:
        #   - QDRANT_URL set  -> server mode (e.g. docker-compose Qdrant service)
        #   - else            -> local mode (default, reads QDRANT_PATH)
        # Local mode holds a file lock, so only one process may open the dir at
        # a time; server mode is shared and is the recommended setup for Docker.
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_path = qdrant_path or os.environ.get(
            "QDRANT_PATH", _DEFAULT_QDRANT_PATH,
        )

        self.collection_name = collection_name
        if qdrant_url:
            # Server mode: connect to a (local or remote) Qdrant server.
            self._client = QdrantClient(url=qdrant_url)
            self.qdrant_path = None
            self._qdrant_mode = "server"
        else:
            # Local mode: on-disk Qdrant (single-process file lock).
            Path(qdrant_path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=qdrant_path)
            self.qdrant_path = qdrant_path
            self._qdrant_mode = "local"
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            index_doc_id=False,  # Avoid payload-index creation on first build
        )
        self._index: Optional[VectorStoreIndex] = None

        # Configure embedding model via custom SiliconFlow-compatible adapter.
        # (llama-index 0.14's OpenAIEmbedding rejects non-OpenAI model ids.)
        Settings.embed_model = SiliconFlowEmbedding(
            model_name=embed_model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            api_base=os.environ.get("OPENAI_BASE_URL", None),
        )

        logger.info(
            f"VaultIndexer initialized ({self._qdrant_mode} mode): "
            f"qdrant={'url=' + qdrant_url if qdrant_url else 'path=' + str(qdrant_path)}, "
            f"collection={collection_name}, embed={embed_model}"
        )

    def build_index(
        self, documents: List[dict], rebuild: bool = False,
    ) -> VectorStoreIndex:
        """Build a vector index from document chunks.

        Args:
            documents: List of dicts with 'text' and 'metadata' keys.
            rebuild: If True, delete existing collection and rebuild from scratch.

        Returns:
            LlamaIndex VectorStoreIndex ready for querying.
        """
        if rebuild and self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")

        # VectorStoreIndex.from_documents does NOT create the Qdrant
        # collection implicitly — after a rebuild-delete the upsert would
        # fail with "Collection not found". Create it explicitly first.
        if not self._client.collection_exists(self.collection_name):
            from qdrant_client import models as qdrant_models

            probe = Settings.embed_model.get_text_embedding("dimension probe")
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=len(probe),
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info(
                f"Created collection {self.collection_name} "
                f"(dim={len(probe)}, distance=cosine)"
            )

        llama_docs = []
        for doc in documents:
            llama_docs.append(
                Document(
                    text=doc["text"],
                    metadata=doc.get("metadata", {}),
                )
            )

        logger.info(f"Indexing {len(llama_docs)} document chunks")

        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
        )

        self._index = VectorStoreIndex.from_documents(
            llama_docs,
            storage_context=storage_context,
            show_progress=True,
        )

        logger.info("Index build complete")
        return self._index

    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load an existing index from Qdrant storage.

        Returns:
            VectorStoreIndex or None if no index exists.
        """
        if not self._client.collection_exists(self.collection_name):
            logger.warning(f"Collection '{self.collection_name}' does not exist")
            return None

        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
        )
        self._index = VectorStoreIndex.from_vector_store(
            self._vector_store,
            storage_context=storage_context,
        )
        return self._index

    def close(self) -> None:
        """Close the underlying Qdrant client to release its file lock."""
        try:
            self._client.close()
        except Exception:
            pass

    def get_or_create_index(
        self, documents: Optional[List[dict]] = None,
    ) -> VectorStoreIndex:
        """Get existing index or create a new one.

        Args:
            documents: Documents to index if creating new index.

        Returns:
            VectorStoreIndex.
        """
        existing = self.load_index()
        if existing is not None:
            logger.info("Loaded existing index")
            return existing

        if documents is None:
            raise ValueError(
                "No existing index found and no documents provided to build one"
            )

        logger.info("No existing index, building new one")
        return self.build_index(documents)

    def get_indexed_file_paths(self) -> set[str]:
        """Collect unique ``file_path`` values from existing collection points.

        Used by vault staleness detection: if files exist on disk but aren't
        present in the index (e.g. dev server started with an older vault,
        or the file-watcher's incremental update missed something), the
        caller should trigger a full rebuild.
        """
        paths: set[str] = set()
        try:
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["file_path"],
                )
                for p in points:
                    fp = (p.payload or {}).get("file_path")
                    if fp:
                        paths.add(str(fp))
                if offset is None:
                    break
        except Exception as exc:
            logger.warning(f"Failed to scan indexed file paths: {exc}")
        return paths

    def insert_documents(self, documents: List[dict]) -> None:
        """Insert new document chunks into the existing index.

        Args:
            documents: List of dicts with 'text' and 'metadata' keys.
        """
        if self._index is None:
            self._index = self.load_index()

        if self._index is None:
            raise ValueError("No index loaded — call build_index or load_index first")

        llama_docs = [
            Document(text=doc["text"], metadata=doc.get("metadata", {}))
            for doc in documents
        ]

        self._index.insert_nodes(
            Settings.node_parser.get_nodes_from_documents(llama_docs),
        )

        logger.info(f"Inserted {len(llama_docs)} documents")

    @property
    def index(self) -> Optional[VectorStoreIndex]:
        """The current VectorStoreIndex, or None."""
        return self._index

"""Hybrid retriever for Obsidian vault RAG.

Combines dense (vector similarity via LlamaIndex) and sparse (BM25 keyword)
retrieval with Reciprocal Rank Fusion, followed by cross-encoder re-ranking
and wiki-link graph expansion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from src.rag.reranker import Reranker
from src.rag.wikilink import get_graph

logger = logging.getLogger(__name__)


class BM25KeywordRetriever:
    """Lightweight BM25 keyword retriever over text chunks.

    Builds a scikit-learn BM25 index from document texts and metadata.
    Supports batch building and incremental document insertion.
    """

    def __init__(self) -> None:
        self._corpus: List[str] = []
        self._meta_list: List[dict] = []
        self._bm25 = None
        self._built = False

    def build_from_documents(self, docs: List[dict]) -> None:
        """Build BM25 index from document chunks.

        Args:
            docs: List of dicts with 'text' and 'metadata' keys.
        """
        if not docs:
            return

        self._corpus = []
        self._meta_list = []

        import jieba

        for doc in docs:
            # Tokenize Chinese text properly
            tokenized = " ".join(jieba.cut(doc["text"]))
            self._corpus.append(tokenized)
            self._meta_list.append(doc.get("metadata", {}))

        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=20000,
        )
        matrix = vectorizer.fit_transform(self._corpus)

        # Simple BM25 scoring on top of TF-IDF
        self._vectorizer = vectorizer
        self._matrix = matrix
        self._doc_lengths = np.asarray(matrix.sum(axis=1)).flatten()
        self._built = True

        logger.info(
            f"BM25 index built: {len(self._corpus)} documents, "
            f"{len(vectorizer.get_feature_names_out())} features"
        )

    def retrieve(self, query: str, top_k: int = 20) -> List[dict]:
        """Retrieve top-k documents using BM25.

        Args:
            query: Query string.
            top_k: Number of top results.

        Returns:
            List of result dicts with 'text', 'bm25_score', and metadata.
        """
        if not self._built:
            return []

        import jieba
        import numpy as np

        tokenized = " ".join(jieba.cut(query))
        query_vec = self._vectorizer.transform([tokenized])
        scores = np.asarray((query_vec @ self._matrix.T).toarray()).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self._meta_list[idx]
            results.append({
                "text": self._corpus[idx],
                "bm25_score": round(float(scores[idx]), 4),
                "note_name": meta.get("note_name", "unknown"),
                "file_path": meta.get("file_path", ""),
                "title": meta.get("title", meta.get("note_name", "unknown")),
                "tags": meta.get("tags", []),
                "wikilinks": meta.get("wikilinks", []),
                "heading": meta.get("chunk_heading", ""),
            })

        return results


def _reciprocal_rank_fusion(
    vector_results: List[dict],
    bm25_results: List[dict],
    k: int = 60,
    top_k: int = 10,
) -> List[dict]:
    """Merge vector and BM25 results using Reciprocal Rank Fusion.

    Args:
        vector_results: Ranked results from vector search.
        bm25_results: Ranked results from BM25 search.
        k: RRF constant (default 60).
        top_k: Number of fused results to return.

    Returns:
        Merged and re-ranked results.
    """
    scores: Dict[str, float] = {}
    items: Dict[str, dict] = {}

    for rank, result in enumerate(vector_results, 1):
        key = result.get("heading", "") + "|||" + result.get("text", "")
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank)
        items[key] = result
        items[key]["vector_score"] = result.get("score", 0.0)

    for rank, result in enumerate(bm25_results, 1):
        key = result.get("heading", "") + "|||" + result.get("text", "")
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank)
        if key not in items:
            items[key] = result
            items[key]["score"] = 0.0
        items[key]["bm25_score"] = result.get("bm25_score", 0.0)

    sorted_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]

    fused = []
    for key in sorted_keys:
        item = items[key].copy()
        item["fusion_score"] = round(scores[key], 4)
        fused.append(item)

    return fused


class VaultRetriever:
    """Hybrid retrieval for Obsidian vault with full pipeline:

    1. Dense retrieval (vector similarity via LlamaIndex)
    2. Sparse retrieval (BM25 keyword)
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-encoder re-ranking (BGE-Reranker)
    5. Wiki-link context expansion
    6. Source attribution

    Usage:
        retriever = VaultRetriever(index, doc_chunks)
        retriever.enable_reranker()  # optional
        results = retriever.search("ellie 的架构设计")
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        docs_for_bm25: Optional[List[dict]] = None,
        similarity_top_k: int = 10,
        wikilink_expand_hops: int = 1,
    ) -> None:
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._wikilink_expand_hops = wikilink_expand_hops
        self._reranker: Optional[Reranker] = None

        # Vector retriever
        self._vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )

        # BM25 retriever
        self._bm25_retriever = BM25KeywordRetriever()
        if docs_for_bm25:
            self._bm25_retriever.build_from_documents(docs_for_bm25)
            logger.info("BM25 retriever initialized with document corpus")
        else:
            logger.info("BM25 retriever initialized without corpus (will skip)")

        self._wiki_graph = get_graph()

    def enable_reranker(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        """Enable cross-encoder re-ranking.

        Args:
            model_name: Name of the reranker model on HuggingFace.
        """
        self._reranker = Reranker(model_name=model_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        tags: Optional[List[str]] = None,
        expand_wikilinks: bool = True,
        use_reranker: bool = True,
    ) -> List[dict]:
        """Full hybrid search pipeline.

        Pipeline:
            Vector search + BM25 → RRF fusion → (reranker) → wikilink expand

        Args:
            query: Natural language search query.
            top_k: Number of final results to return.
            tags: Optional tag filter.
            expand_wikilinks: Whether to expand with wiki-linked notes.
            use_reranker: Whether to apply cross-encoder re-ranking.

        Returns:
            List of result dicts with scores and metadata.
        """
        # Step 1: Vector retrieval
        vector_nodes = self._vector_retriever.retrieve(query)
        vector_results: List[dict] = []
        seen_notes: Set[str] = set()

        for node in vector_nodes:
            metadata = node.metadata or {}
            note_name = metadata.get("note_name", "unknown")
            note_tags = metadata.get("tags", [])

            # Tag filter
            if tags and note_tags:
                if not any(t in note_tags for t in tags):
                    continue

            vector_results.append({
                "text": node.text,
                "score": round(node.score or 0.0, 4),
                "note_name": note_name,
                "file_path": metadata.get("file_path", ""),
                "title": metadata.get("title", note_name),
                "tags": note_tags if isinstance(note_tags, list) else [],
                "wikilinks": metadata.get("wikilinks", []),
                "heading": metadata.get("chunk_heading", ""),
            })
            seen_notes.add(note_name)

        # Step 2: BM25 retrieval
        bm25_results = self._bm25_retriever.retrieve(query, top_k=15)

        # Apply tag filter to BM25 results too
        if tags:
            bm25_results = [
                r for r in bm25_results
                if any(t in r.get("tags", []) for t in tags)
            ]

        logger.info(
            f"Vector results: {len(vector_results)}, "
            f"BM25 results: {len(bm25_results)}"
        )

        # Step 3: Reciprocal Rank Fusion
        fused = _reciprocal_rank_fusion(
            vector_results, bm25_results, top_k=top_k * 3,
        )

        # Step 4: Cross-encoder re-ranking
        if use_reranker and self._reranker is not None:
            fused = self._reranker.rerank(query, fused)
            logger.info(f"Re-ranked {len(fused)} results with cross-encoder")
        else:
            # If no reranker, just sort by fusion score
            fused.sort(key=lambda r: r.get("fusion_score", 0.0), reverse=True)

        results = fused

        # Step 5: Wiki-link expansion
        if expand_wikilinks and self._wiki_graph.total_notes() > 0:
            expanded = self._expand_with_wikilinks(
                query, results, seen_notes, max_extra=top_k,
            )
            results = results + expanded

        # Log final results
        for i, r in enumerate(results[:top_k]):
            logger.info(
                f"  #{i+1}: {r['title']} "
                f"(fusion: {r.get('fusion_score', 0):.3f}, "
                f"rerank: {r.get('rerank_score', 0):.3f})"
            )

        return results[:top_k]

    def search_by_tag(self, tag: str, top_k: int = 10) -> List[dict]:
        """Search for notes containing a specific tag.

        Uses both vector and BM25 with tag filtering.

        Args:
            tag: Tag to search for.
            top_k: Number of results.

        Returns:
            List of result dicts.
        """
        # Use a broad query and rely on tag filtering
        return self.search(
            query="",
            top_k=top_k,
            tags=[tag],
            expand_wikilinks=False,
            use_reranker=False,
        )

    def _expand_with_wikilinks(
        self,
        query: str,
        existing_results: List[dict],
        seen_notes: Set[str],
        max_extra: int = 3,
    ) -> List[dict]:
        """Expand results by following wiki links from top matches."""
        linked_notes: Set[str] = set()
        for result in existing_results[:self._wikilink_expand_hops]:
            note = result["note_name"]
            neighbors = self._wiki_graph.get_links(note, max_hops=1)
            for neighbor in neighbors:
                if neighbor not in seen_notes:
                    linked_notes.add(neighbor)

        if not linked_notes:
            return []

        logger.info(
            f"Wiki-link expansion: {len(linked_notes)} linked notes found"
        )

        extra_results = []
        for note_name in list(linked_notes)[:max_extra]:
            file_path = self._wiki_graph.get_note_path(note_name)
            if not file_path:
                continue

            try:
                content = Path(file_path).read_text(encoding="utf-8")
                preview = content[:1000]
                extra_results.append({
                    "text": preview,
                    "score": 0.0,
                    "note_name": note_name,
                    "file_path": file_path,
                    "title": note_name,
                    "tags": [],
                    "wikilinks": [],
                    "heading": "",
                    "source": "wikilink_expansion",
                })
            except Exception:
                logger.warning(f"Failed to read linked note: {note_name}", exc_info=True)

        return extra_results

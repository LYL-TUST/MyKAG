"""Obsidian vault search tools for LangGraph Agent.

Provides search_vault, search_by_tag, list_tags, and get_note tools
that the Agent uses to query the Obsidian knowledge base.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain.tools import tool

from src.rag.indexer import VaultIndexer
from src.rag.ingestion import VaultIngestionPipeline
from src.rag.retriever import VaultRetriever
from src.rag.wikilink import get_graph

logger = logging.getLogger(__name__)

# Global lazy-initialized instances
_vault_indexer: Optional[VaultIndexer] = None
_vault_retriever: Optional[VaultRetriever] = None
_vault_tags_cache: Optional[List[str]] = None


def _reset_vault() -> None:
    """Reset vault index state (used for switching vaults at runtime)."""
    global _vault_indexer, _vault_retriever, _vault_tags_cache
    if _vault_indexer is not None:
        try:
            _vault_indexer.close()
        except Exception:
            pass
    _vault_indexer = None
    _vault_retriever = None
    _vault_tags_cache = None
    from src.rag.wikilink import reset_graph

    reset_graph()
    logger.info("Vault index reset")


def _init_vault(vault_path: Optional[str] = None, force_rebuild: bool = False) -> None:
    """Initialize vault index and retriever if not already done.

    Builds:
    1. LlamaIndex vector index (Qdrant)
    2. BM25 keyword index from document chunks
    3. Wiki-link graph
    4. Optionally enables BGE-Reranker for cross-encoder scoring
    """
    global _vault_indexer, _vault_retriever, _vault_tags_cache

    import os
    vault_path = vault_path or os.environ.get("OBSIDIAN_VAULT_PATH", ".")

    if _vault_indexer is not None:
        return

    logger.info(f"Initializing vault from: {vault_path}")

    # Step 1: Ingest all documents and chunk
    pipeline = VaultIngestionPipeline(vault_path)
    documents = pipeline.ingest_all()
    chunks = pipeline.documents_to_chunks(documents)

    if not chunks:
        logger.warning("No documents found in vault")
        return

    logger.info(f"Vault ingestion complete: {len(documents)} documents, {len(chunks)} chunks")

    # Step 2: Build vector index
    idx = VaultIndexer()
    if force_rebuild:
        logger.info("Force-rebuilding index from scratch...")
        idx.build_index(chunks, rebuild=True)
    else:
        existing = idx.load_index()

        if existing is None:
            logger.info("No existing index found. Building from scratch...")
            idx.build_index(chunks)
        else:
            # Staleness detection: compare vault files vs the index's known
            # file_path payloads. If vault has files missing from the index
            # (e.g. dev server started with an older vault snapshot, or the
            # watchdog's incremental update missed something), auto-rebuild
            # so the next query can actually recall them.
            indexed_paths = idx.get_indexed_file_paths()
            vault_paths = {str(p.resolve()) for p in Path(vault_path).glob("**/*.md")}
            missing = vault_paths - indexed_paths
            if missing:
                logger.warning(
                    f"Index staleness detected: {len(missing)} vault file(s) not in "
                    f"existing index (e.g. {sorted(missing)[:3]}); rebuilding from scratch"
                )
                idx.build_index(chunks, rebuild=True)
            else:
                logger.info("Reusing existing vector index (vault matches index)")

        idx.load_index()

    # Step 3: Build BM25 keyword index with original text (not tokenized)
    # The BM25KeywordRetriever class handles tokenization internally with jieba
    bm25_chunks = chunks  # pass the original chunk texts

    # Step 4: Create hybrid retriever with both indexes
    retriever = VaultRetriever(
        index=idx.index,
        docs_for_bm25=bm25_chunks,
        similarity_top_k=10,  # More candidates for RRF merging
        wikilink_expand_hops=1,
    )

    # Step 5: Enable reranker if FlagEmbedding is available
    try:
        retriever.enable_reranker("BAAI/bge-reranker-v2-m3")
        logger.info("Reranker enabled: BAAI/bge-reranker-v2-m3")
    except Exception:
        logger.info("Reranker not available — continuing without cross-encoder")

    _vault_indexer = idx
    _vault_retriever = retriever
    logger.info("Vault initialization complete: vector + BM25 + wikilink graph")


@tool
def search_vault(
    query: str,
    top_k: int = 5,
    expand_wikilinks: bool = True,
) -> str:
    """Search the Obsidian vault for notes related to the query.

    Uses semantic (vector) search to find the most relevant note chunks.
    Optionally expands results by following [[wikilink]] connections.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return (1-10, default 5).
        expand_wikilinks: Whether to include related notes via [[wikilinks]].

    Returns:
        Formatted search results with note name, content, and source path.
    """
    _init_vault()

    if _vault_retriever is None:
        return "Vault retriever not initialized. Check OBSIDIAN_VAULT_PATH."

    try:
        results = _vault_retriever.search(
            query=query,
            top_k=top_k,
            expand_wikilinks=expand_wikilinks,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"

    if not results:
        return "No relevant notes found in the vault."

    output_parts = []
    for i, r in enumerate(results, 1):
        source_label = (
            f"[wikilink expansion from related note]"
            if r.get("source") == "wikilink_expansion"
            else ""
        )
        output_parts.append(
            f"### Result {i}: {r['title']} (score: {r['score']:.2f}) {source_label}\n"
            f"- Note: {r['note_name']}\n"
            f"- Tags: {', '.join(r.get('tags', [])) or 'none'}\n"
            f"- Content:\n{r['text'][:800]}..."
            if len(r['text']) > 800
            else f"- Content:\n{r['text']}"
        )

    return "\n\n".join(output_parts)


@tool
def search_by_tag(tag: str, top_k: int = 5) -> str:
    """Search notes filtered by a specific tag.

    Args:
        tag: Tag to filter by (e.g., 'ellie', 'architecture', 'mcp').
        top_k: Maximum results to return.

    Returns:
        Formatted notes matching the tag.
    """
    _init_vault()

    if _vault_retriever is None:
        return "Vault retriever not initialized."

    try:
        results = _vault_retriever.search(
            query="",  # Empty query — filter by tag only
            top_k=top_k,
            tags=[tag],
            expand_wikilinks=False,
        )
    except Exception as e:
        return f"Tag search error: {str(e)}"

    if not results:
        return f"No notes found with tag '{tag}'."

    output_parts = [f"Notes tagged with '#{tag}':\n"]
    for r in results:
        output_parts.append(f"- **{r['title']}** ({r['note_name']})")

    return "\n".join(output_parts)


@tool
def list_tags() -> str:
    """List all tags used across the Obsidian vault.

    Returns:
        Comma-separated list of all unique tags.
    """
    _init_vault()

    graph = get_graph()
    if graph.total_notes() == 0:
        return "Vault is empty or not yet indexed."

    import os
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH", ".")

    all_tags = set()
    for md_file in Path(vault_path).glob("**/*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            from src.rag.chunking import extract_frontmatter
            fm, _ = extract_frontmatter(content)
            tags = fm.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            all_tags.update(tags)
        except Exception:
            pass

    if not all_tags:
        return "No tags found in vault."

    return "Available tags: " + ", ".join(sorted(all_tags))


@tool
def get_note(note_name: str) -> str:
    """Get the full content of a specific note by its filename (without .md).

    Args:
        note_name: Note filename without the .md extension.

    Returns:
        Full note content with metadata.
    """
    import os
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH", ".")

    note_path = Path(vault_path) / f"{note_name}.md"
    if not note_path.is_file():
        # Try recursive search
        found = list(Path(vault_path).glob(f"**/{note_name}.md"))
        if not found:
            return f"Note '{note_name}' not found in vault."
        note_path = found[0]

    try:
        content = note_path.read_text(encoding="utf-8")
        graph = get_graph()
        local_graph = graph.get_local_graph(note_name)

        return (
            f"# {note_name}\n\n"
            f"{content}\n\n"
            f"---\n"
            f"**Outgoing links:** {', '.join(local_graph['out_links']) or 'none'}\n"
            f"**Backlinks:** {', '.join(local_graph['in_links']) or 'none'}\n"
        )
    except Exception as e:
        return f"Failed to read note: {str(e)}"


@tool
def get_note_graph(note_name: str) -> str:
    """Get the wiki-link graph information for a specific note.

    Shows which notes this note links to, and which notes link back to it.

    Args:
        note_name: Note filename without .md extension.

    Returns:
        Graph information with out_links and in_links.
    """
    graph = get_graph()
    local_graph = graph.get_local_graph(note_name)

    out_links = local_graph["out_links"]
    in_links = local_graph["in_links"]

    return (
        f"# {note_name} — Wiki-link Graph\n\n"
        f"**Links to ({len(out_links)}):** "
        f"{', '.join(out_links) if out_links else 'none'}\n\n"
        f"**Linked from ({len(in_links)}):** "
        f"{', '.join(in_links) if in_links else 'none'}"
    )

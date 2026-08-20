"""Project source code search tool for LangGraph Agent.

Supports indexing and searching Python/TypeScript source code
across the user's project directories (ellie, Code Review Agent).

Uses file-system discovery + jieba-tokenized BM25 for fast
code-aware retrieval without needing AST analysis at this stage.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from langchain.tools import tool

logger = logging.getLogger(__name__)

# Pre-configured codebase registry
_COLLECTOR_REGISTRY: Dict[str, str] = {
    "ellie": "E:/agent-projects/ellie",
    "code-review": "E:/agent-projects/AI Code Review Agent",
}

# Lazy-built index: { project_name: { "files": [...], "bm25": [...] } }
_codebase_index: Dict[str, dict] = {}

# File extensions to index
_CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".json", ".yaml", ".yml", ".toml"}

# Min/max file sizes for indexing (skip empty and huge files)
_MIN_BYTES = 50
_MAX_BYTES = 200_000


def _discover_source_files(root: str) -> List[dict]:
    """Find all source files under a project root.

    Returns:
        List of dicts with 'path', 'relative_path', 'name', 'ext'.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        logger.warning(f"Project root not found: {root}")
        return []

    files = []
    for ext in _CODE_EXTENSIONS:
        for file_path in root_path.glob(f"**/*{ext}"):
            if file_path.is_file():
                size = file_path.stat().st_size
                if _MIN_BYTES <= size <= _MAX_BYTES:
                    # Skip virtual envs, node_modules, .git
                    rel = str(file_path.relative_to(root_path)).replace("\\", "/")
                    if any(skip in rel for skip in ("node_modules/", ".git/", "__pycache__/", ".egg-info/", "venv/")):
                        continue
                    files.append({
                        "path": str(file_path),
                        "relative_path": rel,
                        "name": file_path.name,
                        "ext": file_path.suffix,
                    })

    logger.info(f"Discovered {len(files)} source files in {root_path.name}")
    return files


def _build_codebase_index(project_name: str, root_path: str) -> None:
    """Build a BM25 index over a project's source files.

    Indexed per file (not per chunk) for code-level retrieval.
    """
    global _codebase_index

    files = _discover_source_files(root_path)
    if not files:
        logger.warning(f"No source files found in {project_name}")
        return

    import jieba
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Build corpus: one document per file
    corpus = []
    for f in files:
        try:
            content = Path(f["path"]).read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = ""
        # Tokenize
        tokenized = " ".join(jieba.cut(content[:5000]))  # first 5k chars per file
        corpus.append(tokenized)

    vectorizer = TfidfVectorizer(lowercase=True, max_features=15000)
    matrix = vectorizer.fit_transform(corpus)

    _codebase_index[project_name] = {
        "files": files,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "corpus": corpus,
    }

    logger.info(
        f"Codebase index built: {project_name} "
        f"({len(files)} files, {matrix.shape[1]} features)"
    )


def _ensure_codebase_index(project_name: str) -> bool:
    """Ensure the codebase index for a project is built."""
    global _codebase_index

    if project_name not in _COLLECTOR_REGISTRY:
        return False

    if project_name not in _codebase_index:
        root = _COLLECTOR_REGISTRY[project_name]
        _build_codebase_index(project_name, root)

    return project_name in _codebase_index


@tool
def search_codebase(query: str, project: str = "ellie", top_k: int = 5) -> str:
    """Search source code across the user's projects.

    Uses BM25 keyword search to find the most relevant source files.
    Returns file paths and code snippets.

    Args:
        query: What code/functionality you're looking for.
            E.g., "retry logic", "tool registration", "AST parser"
        project: Which project to search. Options: 'ellie' or 'code-review'.
        top_k: Number of matching files to return (1-10, default 5).

    Returns:
        Formatted file paths and relevant code snippets.
    """
    if project not in _COLLECTOR_REGISTRY:
        return (
            f"Unknown project '{project}'. "
            f"Available projects: {', '.join(_COLLECTOR_REGISTRY.keys())}"
        )

    if not _ensure_codebase_index(project):
        return f"Failed to build index for '{project}'."

    index = _codebase_index[project]

    import jieba
    import numpy as np

    tokenized = " ".join(jieba.cut(query))
    query_vec = index["vectorizer"].transform([tokenized])
    scores = np.asarray((query_vec @ index["matrix"].T).toarray()).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        score = scores[idx]
        if score <= 0:
            continue

        file_info = index["files"][idx]
        rel_path = file_info["relative_path"]

        # Read and extract relevant snippet
        try:
            full_content = Path(file_info["path"]).read_text(
                encoding="utf-8", errors="replace",
            )
            preview = full_content[:800]
            if len(full_content) > 800:
                preview += f"\n... ({len(full_content)} chars total, {full_content.count(chr(10))} lines)"
        except Exception:
            preview = "(file not readable)"

        results.append(
            f"### {rank}. {rel_path} (score: {score:.3f})\n"
            f"```{file_info['ext'].lstrip('.')}\n{preview}\n```"
        )

    if not results:
        return f"No matching files found in '{project}' for query: {query}"

    header = f"**{project}** — {len(results)} results for '{query}':\n\n"
    return header + "\n\n".join(results)


@tool
def list_codebase_projects() -> str:
    """List all codebase projects available for search.

    Returns:
        Names and paths of indexed projects.
    """
    lines = ["Available codebase projects:"]
    for name, path in _COLLECTOR_REGISTRY.items():
        exists = "✓" if Path(path).is_dir() else "✗"
        lines.append(f"  {exists} {name} → {path}")
    lines.append(f"\nUse `search_codebase(query, project='{list(_COLLECTOR_REGISTRY.keys())[0]}')` to search.")
    return "\n".join(lines)

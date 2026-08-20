"""Obsidian [[wikilink]] parser and in-memory knowledge graph.

Provides extraction, storage, and traversal of wiki-style bidirectional links
between notes. Builds an adjacency list representation of the vault's link graph.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.rag.chunking import extract_frontmatter

logger = logging.getLogger(__name__)

# Matches [[link]] and [[link|alias]] patterns
_WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


class WikiGraph:
    """In-memory adjacency list for Obsidian [[wikilink]] graph.

    Each note is a node. Edges are bidirectional — when A links to B,
    both A's out-link and B's in-link are recorded.
    """

    def __init__(self) -> None:
        self._out_links: Dict[str, Set[str]] = defaultdict(set)
        self._in_links: Dict[str, Set[str]] = defaultdict(set)
        self._note_paths: Dict[str, str] = {}  # note_name -> absolute_path
        self._note_tags: Dict[str, List[str]] = {}  # note_name -> tags
        self._note_mtimes: Dict[str, Optional[str]] = {}  # note_name -> ISO 8601 mtime

    def add_note(self, note_name: str, file_path: str, tags: Optional[List[str]] = None, mtime_iso: Optional[str] = None) -> None:
        """Register a note in the graph."""
        self._note_paths[note_name] = file_path
        self._note_tags[note_name] = tags or []
        self._note_mtimes[note_name] = mtime_iso

    def add_link(self, source: str, target: str) -> None:
        """Add a directional edge from source note to target note."""
        if target not in self._note_paths:
            logger.debug(f"Link to unknown note: {source} -> {target}")
        self._out_links[source].add(target)
        self._in_links[target].add(source)

    def get_links(self, note_name: str, max_hops: int = 1) -> Set[str]:
        """Get all notes reachable from a note within max_hops using BFS.

        Args:
            note_name: Starting note name (without .md suffix).
            max_hops: Maximum number of link hops to traverse.

        Returns:
            Set of note names reachable within max_hops.
        """
        visited: Set[str] = set()
        frontier: Set[str] = {note_name}

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                for neighbor in self._out_links.get(node, set()):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                    if neighbor not in visited and neighbor != note_name:
                        next_frontier.add(neighbor)
            visited.update(next_frontier)
            frontier = next_frontier

        return visited

    def get_backlinks(self, note_name: str) -> Set[str]:
        """Get all notes that link to this note."""
        return self._in_links.get(note_name, set())

    def get_local_graph(self, note_name: str) -> dict:
        """Get the local subgraph around a note (1-hop).

        Returns:
            dict with 'node', 'out_links', and 'in_links' keys.
        """
        return {
            "node": note_name,
            "out_links": sorted(self._out_links.get(note_name, set())),
            "in_links": sorted(self._in_links.get(note_name, set())),
        }

    def get_note_path(self, note_name: str) -> Optional[str]:
        """Get the absolute file path for a note, or None."""
        return self._note_paths.get(note_name)

    def total_notes(self) -> int:
        """Return the total number of notes in the graph."""
        return len(self._note_paths)

    def total_links(self) -> int:
        """Return the total number of directional edges."""
        return sum(len(links) for links in self._out_links.values())

    def export(self) -> dict:
        """Export the full graph as nodes + edges for visualization.

        Deduplicates bidirectional edges (A->B and B->A collapse to one
        undirected edge) and drops dangling links whose target note does
        not exist in the vault.

        Returns:
            dict with 'nodes' (list of {id, label, path}) and
            'edges' (list of {source, target}).
        """
        nodes = [
            {
                "id": name,
                "label": name,
                "path": path,
                "tags": self._note_tags.get(name, []),
                "created_at": self._note_mtimes.get(name),
            }
            for name, path in sorted(self._note_paths.items())
        ]
        known = set(self._note_paths.keys())

        edges: List[dict] = []
        seen: Set[tuple] = set()
        for source, targets in self._out_links.items():
            for target in targets:
                if target not in known:
                    continue  # dangling link to a non-existent note
                key = tuple(sorted((source, target)))
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"source": source, "target": target})

        return {"nodes": nodes, "edges": edges}


# Global graph instance
_wiki_graph = WikiGraph()


def extract_wikilinks(content: str) -> List[str]:
    """Extract all [[wikilink]] targets from markdown content.

    Args:
        content: Raw markdown text.

    Returns:
        List of link targets (without .md suffix, stripped).
    """
    targets = _WIKILINK_PATTERN.findall(content)
    return [t.strip() for t in targets if t.strip()]


def build_graph_from_vault(vault_path: str) -> WikiGraph:
    """Scan an Obsidian vault and build the complete link graph.

    Walks all .md files in the vault, extracts frontmatter and [[wikilinks]],
    and builds the adjacency list. Also updates the global graph instance
    returned by ``get_graph()``.

    Args:
        vault_path: Absolute path to the Obsidian vault root.

    Returns:
        Populated WikiGraph instance.
    """
    global _wiki_graph
    graph = WikiGraph()
    vault_dir = Path(vault_path)

    if not vault_dir.is_dir():
        logger.warning(f"Vault path not found: {vault_path}")
        _wiki_graph = graph
        return graph

    md_files = list(vault_dir.glob("**/*.md"))
    logger.info(f"Found {len(md_files)} markdown files in vault")

    # First pass: register all notes with their frontmatter tags
    for md_file in md_files:
        note_name = md_file.stem  # filename without .md
        tags: List[str] = []
        try:
            content = md_file.read_text(encoding="utf-8")
            frontmatter, _ = extract_frontmatter(content)
            raw_tags = frontmatter.get("tags", [])
            if isinstance(raw_tags, str):
                tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            elif isinstance(raw_tags, list):
                tags = [t for t in raw_tags if isinstance(t, str) and t.strip()]
        except Exception:
            logger.warning(f"Failed to read {md_file}", exc_info=True)
        try:
            mtime_iso = datetime.fromtimestamp(md_file.stat().st_mtime).isoformat()
        except Exception:
            mtime_iso = None
        graph.add_note(note_name, str(md_file), tags, mtime_iso)

    # Second pass: extract links
    for md_file in md_files:
        note_name = md_file.stem
        try:
            content = md_file.read_text(encoding="utf-8")
            links = extract_wikilinks(content)
            for target in links:
                graph.add_link(note_name, target)
        except Exception:
            logger.warning(f"Failed to read {md_file}", exc_info=True)

    logger.info(
        f"Graph built: {graph.total_notes()} notes, {graph.total_links()} links"
    )
    _wiki_graph = graph
    return graph


def get_graph() -> WikiGraph:
    """Get the global wiki graph instance."""
    return _wiki_graph


def reset_graph() -> None:
    """Reset the global graph (for testing)."""
    global _wiki_graph
    _wiki_graph = WikiGraph()

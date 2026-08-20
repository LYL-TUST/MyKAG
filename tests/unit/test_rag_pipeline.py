"""Unit tests for RAG pipeline modules.

Covers:
- wikilink.py: link extraction, graph building, BFS traversal
- chunking.py: frontmatter parsing, semantic chunking
- ingestion.py: vault document parsing, chunk conversion
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── WikiLink Tests ──


def test_extract_wikilinks_basic() -> None:
    """Extract simple [[wikilinks]] from markdown text."""
    from src.rag.wikilink import extract_wikilinks

    content = "See [[ellie 架构设计]] and [[MCP 协议设计]] for details."
    links = extract_wikilinks(content)
    assert links == ["ellie 架构设计", "MCP 协议设计"]


def test_extract_wikilinks_with_alias() -> None:
    """Extract wikilinks with pipe aliases."""
    from src.rag.wikilink import extract_wikilinks

    content = "See [[ellie 架构设计|ellie architecture]] for the design."
    links = extract_wikilinks(content)
    assert links == ["ellie 架构设计"]


def test_extract_wikilinks_empty() -> None:
    """Handle text with no wikilinks."""
    from src.rag.wikilink import extract_wikilinks

    content = "No links here, just plain text."
    links = extract_wikilinks(content)
    assert links == []


def test_extract_wikilinks_multiple() -> None:
    """Extract multiple wikilinks including duplicates."""
    from src.rag.wikilink import extract_wikilinks

    content = """
    Related: [[ellie 工具系统]], [[LangChain Tool 注册机制]], [[Code Review Agent MCP 协议]]
    Also see [[ellie 工具系统]] for more.
    """
    links = extract_wikilinks(content)
    assert "ellie 工具系统" in links
    assert "LangChain Tool 注册机制" in links
    assert len(links) == 4  # includes duplicate


def test_graph_build_and_traverse() -> None:
    """Build a wiki graph and verify BFS traversal."""
    from src.rag.wikilink import WikiGraph

    graph = WikiGraph()
    graph.add_note("A", "/vault/A.md")
    graph.add_note("B", "/vault/B.md")
    graph.add_note("C", "/vault/C.md")
    graph.add_note("D", "/vault/D.md")

    graph.add_link("A", "B")
    graph.add_link("A", "C")
    graph.add_link("B", "D")

    # 1-hop from A should return B, C
    links_1hop = graph.get_links("A", max_hops=1)
    assert "B" in links_1hop
    assert "C" in links_1hop
    assert "D" not in links_1hop  # 2 hops away
    assert "A" not in links_1hop  # not self

    # 2-hop from A should return B, C, D
    links_2hop = graph.get_links("A", max_hops=2)
    assert "D" in links_2hop


def test_graph_backlinks() -> None:
    """Verify backlink tracking."""
    from src.rag.wikilink import WikiGraph

    graph = WikiGraph()
    graph.add_note("A", "/vault/A.md")
    graph.add_note("B", "/vault/B.md")
    graph.add_link("A", "B")
    graph.add_link("C", "B")

    backlinks = graph.get_backlinks("B")
    assert backlinks == {"A", "C"}


# ── Chunking Tests ──


def test_extract_frontmatter_simple() -> None:
    """Parse basic YAML frontmatter."""
    from src.rag.chunking import extract_frontmatter

    content = """---
title: Test Note
tags: [python, rag]
created: 2026-01-01
---

# Heading
Body text here.
"""
    fm, body = extract_frontmatter(content)

    assert fm["title"] == "Test Note"
    assert fm["tags"] == ["python", "rag"]
    assert fm["created"] == "2026-01-01"
    assert "Body text here" in body


def test_extract_frontmatter_none() -> None:
    """Handle content without frontmatter."""
    from src.rag.chunking import extract_frontmatter

    content = "# Just a heading\nSome text."
    fm, body = extract_frontmatter(content)

    assert fm == {}
    assert body == content


def test_chunk_by_headings() -> None:
    """Verify chunks are split by headings."""
    from src.rag.chunking import chunk_markdown

    content = """## Section 1
This is the first section.

## Section 2
This is the second section with more content.
It spans multiple paragraphs.
"""
    chunks = chunk_markdown(content, chunk_size=200)

    assert len(chunks) >= 2
    # Each chunk should have a heading
    headings = [c["heading"] for c in chunks]
    assert "## Section 1" in headings
    assert "## Section 2" in headings


def test_chunk_small_content() -> None:
    """Small content should produce a single chunk."""
    from src.rag.chunking import chunk_markdown

    content = "Short single paragraph."
    chunks = chunk_markdown(content)
    assert len(chunks) == 1
    assert "Short single paragraph" in chunks[0]["text"]


# ── Ingestion Tests ──


def test_vault_document_properties() -> None:
    """Verify VaultDocument property accessors."""
    from src.rag.ingestion import VaultDocument

    doc = VaultDocument(
        file_path="/vault/test.md",
        note_name="test",
        frontmatter={
            "title": "Test Note",
            "tags": ["python", "test"],
        },
        body="Some content here.",
        wikilinks=["other-note"],
    )

    assert doc.title == "Test Note"
    assert doc.tags == ["python", "test"]
    assert doc.note_name == "test"
    assert "note_name" in doc.metadata_dict
    assert "title" in doc.metadata_dict


def test_vault_document_title_fallback() -> None:
    """Title should fall back to note name when no frontmatter title."""
    from src.rag.ingestion import VaultDocument

    doc = VaultDocument(
        file_path="/vault/untitled.md",
        note_name="untitled",
        frontmatter={},
        body="Content.",
        wikilinks=[],
    )

    assert doc.title == "untitled"
    assert doc.tags == []

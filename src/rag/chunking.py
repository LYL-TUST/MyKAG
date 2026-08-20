"""Semantic chunking for Obsidian markdown notes.

Provides context-aware chunking that respects markdown structure
(headings, code blocks, paragraphs) and preserves frontmatter metadata.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Pattern to extract YAML frontmatter
_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Min characters before we consider splitting a paragraph
_MIN_CHUNK_CHARS = 100
# Target chunk size (characters)
_TARGET_CHUNK_SIZE = 800
# Max chunk size before forced split
_MAX_CHUNK_SIZE = 1500


def extract_frontmatter(content: str) -> Tuple[dict, str]:
    """Extract YAML frontmatter from markdown content.

    Args:
        content: Raw markdown text.

    Returns:
        Tuple of (frontmatter_dict, body_text).
        frontmatter_dict is empty if no frontmatter found.
    """
    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    body = content[match.end():]

    metadata = {}
    for line in frontmatter_text.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Handle list values (e.g., tags: [a, b, c])
            if value.startswith("[") and value.endswith("]"):
                items = value[1:-1].split(",")
                value = [item.strip().strip('"').strip("'") for item in items if item.strip()]
            metadata[key] = value

    return metadata, body


def chunk_markdown(
    content: str,
    chunk_size: int = _TARGET_CHUNK_SIZE,
    max_chunk_size: int = _MAX_CHUNK_SIZE,
    overlap: int = 100,
) -> List[dict]:
    """Split markdown content into semantic chunks.

    Strategy:
    1. Split by headings first (### or ##) to get natural sections
    2. If a section is too large, split by paragraphs
    3. Preserve code blocks as single chunks
    4. Add overlap between consecutive chunks

    Args:
        content: Raw markdown text (body only, frontmatter already extracted).
        chunk_size: Target size per chunk (characters).
        max_chunk_size: Hard maximum before forced split.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk dicts with 'text', 'index', and 'heading' keys.
    """
    # Split by headings
    sections = _split_by_headings(content)

    chunks: List[dict] = []
    chunk_idx = 0

    for heading, section_text in sections:
        if not section_text.strip():
            continue

        if len(section_text) <= max_chunk_size:
            chunks.append({
                "text": section_text,
                "index": chunk_idx,
                "heading": heading,
            })
            chunk_idx += 1
        else:
            # Need to split large section into smaller chunks
            sub_chunks = _split_by_paragraphs(section_text, chunk_size, overlap)
            for sub in sub_chunks:
                chunks.append({
                    "text": sub,
                    "index": chunk_idx,
                    "heading": heading,
                })
                chunk_idx += 1

    return chunks


def _split_by_headings(content: str) -> List[Tuple[str, str]]:
    """Split content by ### or ## headings.

    Returns list of (heading_text, section_body) tuples.
    """
    # Match ## or ### headings with optional trailing text
    heading_pattern = re.compile(r"^(#{2,3}\s+.+)$", re.MULTILINE)

    positions = [(m.start(), m.group(1)) for m in heading_pattern.finditer(content)]

    sections = []
    if not positions:
        sections.append(("", content.strip()))
        return sections

    # Content before the first heading
    if positions[0][0] > 0:
        pre_content = content[:positions[0][0]].strip()
        if pre_content:
            sections.append(("", pre_content))

    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(content)
        section_body = content[start:end].strip()
        if section_body:
            sections.append((heading, section_body))

    return sections


def _split_by_paragraphs(
    text: str, target_size: int, overlap: int,
) -> List[str]:
    """Split large text into smaller overlapping chunks by paragraphs.

    Args:
        text: Text to split.
        target_size: Target characters per chunk.
        overlap: Character overlap between chunks.

    Returns:
        List of chunk texts.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    current_size = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_size = len(para)

        # Code blocks stay as their own chunk
        if para.startswith("```") and current_chunk:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para
            current_size = para_size
            continue

        # Flush and start new chunk if adding this paragraph exceeds target
        if current_size + para_size > target_size and current_size >= _MIN_CHUNK_CHARS:
            chunks.append(current_chunk.strip())

            # Keep overlap from end of previous chunk
            if overlap and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n\n" + para
                current_size = len(current_chunk)
            else:
                current_chunk = para
                current_size = para_size
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
            current_size += para_size

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

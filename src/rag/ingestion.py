"""Document ingestion pipeline for Obsidian vault.

Walks the vault, parses .md files (frontmatter + [[wikilinks]]),
chunks content, and feeds into the vector index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.rag.chunking import chunk_markdown, extract_frontmatter
from src.rag.wikilink import build_graph_from_vault, extract_wikilinks

logger = logging.getLogger(__name__)


class VaultDocument:
    """Represents a single Obsidian note with parsed metadata."""

    def __init__(
        self,
        file_path: str,
        note_name: str,
        frontmatter: dict,
        body: str,
        wikilinks: List[str],
    ) -> None:
        self.file_path = file_path
        self.note_name = note_name
        self.frontmatter = frontmatter
        self.body = body
        self.wikilinks = wikilinks

    @property
    def title(self) -> str:
        """Best-effort title: frontmatter title > filename."""
        return self.frontmatter.get("title", self.note_name)

    @property
    def tags(self) -> List[str]:
        """Tags from frontmatter or empty list."""
        tags = self.frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        return tags

    @property
    def metadata_dict(self) -> dict:
        """All metadata for the embedding index."""
        return {
            "note_name": self.note_name,
            "title": self.title,
            "file_path": self.file_path,
            "tags": self.tags,
            "wikilinks": self.wikilinks,
            **{k: v for k, v in self.frontmatter.items() if k not in ("tags", "title", "related")},
        }


class VaultIngestionPipeline:
    """Pipeline for ingesting an Obsidian vault into the RAG system.

    Walks all .md files, parses content, extracts metadata and links,
    and produces a list of VaultDocument objects ready for indexing.
    """

    def __init__(self, vault_path: str) -> None:
        if not Path(vault_path).is_dir():
            raise FileNotFoundError(f"Vault directory not found: {vault_path}")
        self.vault_path = Path(vault_path)

    def ingest_all(self) -> List[VaultDocument]:
        """Ingest all .md files in the vault.

        Returns:
            List of VaultDocument objects with parsed content and metadata.
        """
        documents: List[VaultDocument] = []
        md_files = sorted(self.vault_path.glob("**/*.md"))

        # Build the global wikilink graph
        build_graph_from_vault(str(self.vault_path))

        for md_file in md_files:
            try:
                doc = self._parse_file(md_file)
                if doc.body.strip():
                    documents.append(doc)
            except Exception:
                logger.warning(f"Failed to parse {md_file}", exc_info=True)

        logger.info(f"Ingested {len(documents)} documents from vault")
        return documents

    def ingest_single(self, file_path: str) -> VaultDocument | None:
        """Ingest a single .md file from the vault.

        Args:
            file_path: Path to the .md file (relative to vault or absolute).

        Returns:
            VaultDocument or None on failure.
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.vault_path / path

        if not path.suffix == ".md" or not path.is_file():
            logger.warning(f"Not a valid markdown file: {path}")
            return None

        try:
            return self._parse_file(path)
        except Exception:
            logger.warning(f"Failed to parse {path}", exc_info=True)
            return None

    def _parse_file(self, file_path: Path) -> VaultDocument:
        """Parse a single .md file into a VaultDocument."""
        content = file_path.read_text(encoding="utf-8")
        note_name = file_path.stem
        frontmatter, body = extract_frontmatter(content)
        wikilinks = extract_wikilinks(content)

        return VaultDocument(
            file_path=str(file_path),
            note_name=note_name,
            frontmatter=frontmatter,
            body=body,
            wikilinks=wikilinks,
        )

    def documents_to_chunks(self, documents: List[VaultDocument]) -> List[dict]:
        """Convert documents to embeddable chunks with metadata.

        Each chunk carries its source document's metadata for filtering.

        Returns:
            List of dicts with 'text', 'metadata' keys ready for LlamaIndex.
        """
        all_chunks = []
        for doc in documents:
            chunks = chunk_markdown(doc.body)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk["text"],
                    "metadata": {
                        **doc.metadata_dict,
                        "chunk_heading": chunk.get("heading", ""),
                        "chunk_index": chunk["index"],
                    },
                })
        return all_chunks

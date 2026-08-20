"""Unit tests for vault retriever and tools.

Tests:
- VaultRetriever search (no index required, test BM25 + RRF)
- Vault tools integration
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def test_reciprocal_rank_fusion_basic() -> None:
    """RRF should merge and re-rank vector + BM25 results."""
    from src.rag.retriever import _reciprocal_rank_fusion

    vector = [
        {"text": "A", "score": 0.9, "heading": "", "note_name": "doc_a"},
        {"text": "B", "score": 0.7, "heading": "", "note_name": "doc_b"},
        {"text": "C", "score": 0.5, "heading": "", "note_name": "doc_c"},
    ]

    bm25 = [
        {"text": "D", "bm25_score": 0.8, "heading": "", "note_name": "doc_d"},
        {"text": "B", "bm25_score": 0.6, "heading": "", "note_name": "doc_b"},
        {"text": "E", "bm25_score": 0.4, "heading": "", "note_name": "doc_e"},
    ]

    fused = _reciprocal_rank_fusion(vector, bm25, top_k=5)

    # Should merge and produce fusion scores
    assert len(fused) == 5  # 3 unique from vector + 2 new from BM25
    for r in fused:
        assert "fusion_score" in r


def test_reciprocal_rank_fusion_empty() -> None:
    """RRF with empty inputs should return empty."""
    from src.rag.retriever import _reciprocal_rank_fusion

    fused = _reciprocal_rank_fusion([], [], top_k=5)
    assert fused == []


def test_bm25_retriever_build_and_search() -> None:
    """BM25 index should build and retrieve documents."""
    from src.rag.retriever import BM25KeywordRetriever

    docs = [
        {
            "text": "ellie 是一个本地 Python Coding Agent，零依赖设计",
            "metadata": {"note_name": "ellie 概述", "title": "ellie 概述"},
        },
        {
            "text": "Code Review Agent 使用 MCP 协议做工具解耦",
            "metadata": {"note_name": "MCP 协议", "title": "MCP 协议"},
        },
        {
            "text": "LangGraph 提供了 Agent Loop 的状态机抽象",
            "metadata": {"note_name": "Agent Loop", "title": "Agent Loop 设计"},
        },
        {
            "text": "ellie 支持 5 个模型后端：DeepSeek, OpenAI, Anthropic, Google, Ollama",
            "metadata": {"note_name": "ellie 模型后端", "title": "模型后端"},
        },
    ]

    bm25 = BM25KeywordRetriever()
    bm25.build_from_documents(docs)

    results = bm25.retrieve("ellie 的模型后端", top_k=3)
    assert len(results) > 0

    # The model backend doc should be top result
    assert results[0]["note_name"] == "ellie 模型后端"


def test_bm25_retriever_empty() -> None:
    """Unbuilt BM25 should return empty."""
    from src.rag.retriever import BM25KeywordRetriever

    bm25 = BM25KeywordRetriever()
    results = bm25.retrieve("anything", top_k=5)
    assert results == []

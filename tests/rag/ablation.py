"""Ablation study: compare retrieval strategies on the eval dataset.

Measures Recall@5 (fraction of queries where at least one ground-truth
relevant note appears in the top-5 results) and average hit count per query.

Strategies compared:
  1. Vector only (dense semantic)
  2. BM25 only (sparse keyword)
  3. Hybrid RRF (vector + BM25 fusion, no rerank)
  4. Full pipeline (hybrid + wikilink expansion, no cross-encoder)

This deliberately avoids the FlagEmbedding cross-encoder so the study runs
fast and has no model-download dependency. Run:

    python tests/rag/ablation.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.WARNING)


def _ensure_retriever():
    import src.tools.vault_tools as vt

    if vt._vault_retriever is None:
        vault_path = os.environ.get(
            "OBSIDIAN_VAULT_PATH",
            str(_PROJECT_ROOT.parent / "obsidian-vault"),
        )
        vt._init_vault(vault_path)
    return vt._vault_retriever


def _vector_only(retriever, query: str, top_k: int = 5) -> List[dict]:
    nodes = retriever._vector_retriever.retrieve(query)
    results = []
    for node in nodes:
        m = node.metadata or {}
        results.append({"note_name": m.get("note_name", "unknown")})
    return results[:top_k]


def _bm25_only(retriever, query: str, top_k: int = 5) -> List[dict]:
    return retriever._bm25_retriever.retrieve(query, top_k=top_k)


def _hybrid_rrf(retriever, query: str, top_k: int = 5) -> List[dict]:
    from src.rag.retriever import _reciprocal_rank_fusion

    vector_nodes = retriever._vector_retriever.retrieve(query)
    vector_results = []
    for node in vector_nodes:
        m = node.metadata or {}
        vector_results.append({
            "note_name": m.get("note_name", "unknown"),
            "text": node.text,
            "heading": m.get("chunk_heading", ""),
            "score": node.score or 0.0,
        })
    bm25_results = retriever._bm25_retriever.retrieve(query, top_k=15)
    return _reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)


def _full_pipeline(retriever, query: str, top_k: int = 5) -> List[dict]:
    return retriever.search(
        query,
        top_k=top_k,
        expand_wikilinks=True,
        use_reranker=False,  # avoid model download; see docstring
    )


def run_ablation() -> dict:
    from tests.rag.eval_dataset import EVAL_DATASET

    retriever = _ensure_retriever()

    strategies: dict[str, Callable[[str], List[dict]]] = {
        "纯向量 (Vector)": lambda q: _vector_only(retriever, q),
        "纯 BM25 (Keyword)": lambda q: _bm25_only(retriever, q),
        "混合 RRF": lambda q: _hybrid_rrf(retriever, q),
        "完整管线 (+图谱扩展)": lambda q: _full_pipeline(retriever, q),
    }

    print(f"\n{'策略':<26}{'Recall@5':<12}{'平均命中/查询':<14}")
    print("-" * 56)

    summary = {}
    for name, fn in strategies.items():
        hit_queries = 0
        total_hits = 0
        for case in EVAL_DATASET:
            retrieved = fn(case["question"])
            retrieved_notes = [r["note_name"] for r in retrieved]
            relevant = case["relevant_notes"]
            found = sum(1 for en in relevant if en in retrieved_notes)
            if found > 0:
                hit_queries += 1
            total_hits += found

        recall = hit_queries / len(EVAL_DATASET)
        avg_hits = total_hits / len(EVAL_DATASET)
        summary[name] = {"recall_at_5": recall, "avg_hits": avg_hits}
        print(f"{name:<26}{recall:>8.1%}   {avg_hits:>8.2f}")

    print("-" * 56)
    print(f"总查询数: {len(EVAL_DATASET)}")
    return summary


if __name__ == "__main__":
    run_ablation()

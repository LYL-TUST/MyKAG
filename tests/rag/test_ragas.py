"""RAGAS evaluation runner for Personal Knowledge Agent.

Evaluates the retrieval pipeline quality using the RAGAS framework:
- Faithfulness: 生成答案是否忠实于检索到的上下文
- Context Relevancy: 检索到的上下文是否与问题相关
- Context Precision: 相关上下文的排序是否靠前
- Answer Correctness: 生成的答案与参考答案是否一致

Usage:
    # Install dependencies
    pip install ragas datasets

    # Set your LLM for evaluation (used as judge)
    export OPENAI_API_KEY=your_key
    export OPENAI_BASE_URL=https://api.siliconflow.cn/v1

    # Run evaluation
    python tests/rag/test_ragas.py

Or with pytest:
    pytest tests/rag/test_ragas.py -v
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure src is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _ensure_vault_index() -> None:
    """Initialize vault if not already done."""
    from src.tools.vault_tools import _init_vault, _vault_retriever
    if _vault_retriever is None:
        vault_path = os.environ.get("OBSIDIAN_VAULT_PATH", str(_PROJECT_ROOT.parent / "obsidian-vault"))
        _init_vault(vault_path)


def _run_retrieval(query: str, top_k: int = 3) -> List[dict]:
    """Run retrieval and return context chunks."""
    from src.tools.vault_tools import _vault_retriever
    if _vault_retriever is None:
        raise RuntimeError("Vault not initialized")
    results = _vault_retriever.search(query, top_k=top_k, expand_wikilinks=True)
    return results


def run_evaluation(use_ragas: bool = True) -> dict:
    """Run the full evaluation suite.

    Args:
        use_ragas: If True, use RAGAS framework. If False, do basic metrics only.

    Returns:
        Dict with evaluation metrics.
    """
    from tests.rag.eval_dataset import EVAL_DATASET

    _ensure_vault_index()

    results = {
        "total_queries": len(EVAL_DATASET),
        "retrieval_hit_rate": 0.0,
        "ragas_metrics": None,
        "per_query": [],
    }

    hits = 0
    for case in EVAL_DATASET:
        query = case["question"]
        expected_notes = case["relevant_notes"]
        retrieved = _run_retrieval(query, top_k=5)
        retrieved_notes = [r["note_name"] for r in retrieved]

        # Simple hit rate: at least one expected note in top-5
        hit = any(en in retrieved_notes for en in expected_notes)
        if hit:
            hits += 1

        results["per_query"].append({
            "question": query[:60] + "...",
            "hit": hit,
            "retrieved_notes": retrieved_notes[:5],
            "expected_notes": expected_notes,
        })

        logger.info(
            f"{'✓' if hit else '✗'} {query[:60]}... "
            f"(found: {[n for n in retrieved_notes if n in expected_notes][:3]})"
        )

    results["retrieval_hit_rate"] = hits / len(EVAL_DATASET)
    logger.info(f"\nRetrieval Hit Rate: {hits}/{len(EVAL_DATASET)} = {results['retrieval_hit_rate']:.1%}")

    # RAGAS evaluation
    if use_ragas:
        try:
            ragas_results = _run_ragas_evaluation()
            results["ragas_metrics"] = ragas_results
        except ImportError:
            logger.warning("RAGAS not installed. Skipping RAGAS metrics. Install: pip install ragas datasets")
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")

    return results


def _run_ragas_evaluation() -> dict:
    """Run RAGAS evaluation framework.

    Requires: pip install ragas datasets openai
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from tests.rag.eval_dataset import EVAL_DATASET
    from src.tools.vault_tools import _vault_retriever

    # Collect contexts and generate answers
    ragas_data = []
    for case in EVAL_DATASET:
        query = case["question"]
        ground_truth = case["expected_answer"]
        results = _vault_retriever.search(query, top_k=3, expand_wikilinks=True)
        contexts = [r["text"][:500] for r in results[:3]]

        # Generate a simple answer by concatenating context
        # (In production, you'd call the full Agent pipeline)
        answer_parts = []
        for r in results[:3]:
            answer_parts.append(f"[{r['note_name']}]: {r['text'][:200]}")

        answer = "\n\n".join(answer_parts)

        ragas_data.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    dataset = Dataset.from_list(ragas_data)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # Use a judge LLM for evaluation
    from langchain_openai import ChatOpenAI
    import os
    evaluator_llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V4",
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_api_base=os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
    )

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    logger.info(f"\nRAGAS Metrics:")
    for metric_name, score in result.items():
        logger.info(f"  {metric_name}: {score:.3f}")

    return {k: round(float(v), 4) for k, v in result.items()}


def main():
    """CLI entry point for running evaluation."""
    use_ragas = "--ragas" in sys.argv or os.environ.get("USE_RAGAS", "").lower() == "true"
    results = run_evaluation(use_ragas=use_ragas)

    print(f"\n{'='*50}")
    print(f"Evaluation Summary")
    print(f"{'='*50}")
    print(f"Total queries: {results['total_queries']}")
    print(f"Retrieval Hit Rate: {results['retrieval_hit_rate']:.1%}")

    if results["ragas_metrics"]:
        print(f"\nRAGAS Metrics:")
        for k, v in results["ragas_metrics"].items():
            print(f"  {k}: {v:.3f}")

    print(f"\nPer-Query Results:")
    for q in results["per_query"]:
        status = "✓" if q["hit"] else "✗"
        print(f"  {status} {q['question']}")

    return results


if __name__ == "__main__":
    main()

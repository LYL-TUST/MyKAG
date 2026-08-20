"""RAGAS evaluation runner for Personal Knowledge Agent.

Evaluates the retrieval + generation pipeline using the RAGAS framework:
- Faithfulness: 生成答案是否忠实于检索到的上下文
- Answer Relevancy: 答案是否切题
- Context Precision: 相关上下文的排序是否靠前
- Context Recall: 检索是否召回了全部相关上下文

Fix log (2026-08-20):
1. **answer 由真实 graph 生成**: 之前是"拼接检索片段"当答案,现在改用
   knowledge_agent 单 Agent 真实调用 (docs_agent.ainvoke)。
2. **judge 换 qwen3-8b**: 之前写死 `deepseek-ai/DeepSeek-V4`(硅基流动不存在的
   旧 id),现在走模型注册表 `JUDGE_MODEL_KEY`(默认 qwen3-8b,env 可覆盖),并
   关闭 thinking 提速。
3. **修旧模型 id**: 同上,registry 解析,不再出现不存在的模型。
4. **索引覆盖预检**: 评测前扫描 Qdrant 索引里的 note_name,与数据集的
   relevant_notes 对比,防止在残缺索引上跑出"幻觉高分"。

Usage:
    # Install dependencies
    pip install -e ".[eval]"

    # .env 需有 OPENAI_API_KEY / OPENAI_BASE_URL(硅基流动)与 Qdrant 索引
    python tests/rag/test_ragas.py --ragas
    python tests/rag/test_ragas.py            # 只跑检索命中率(无需 LLM 生成)

    RAGAS_METRICS=faithfulness,answer_relevancy python tests/rag/test_ragas.py --ragas
    QDRANT_PATH=./qdrant_data_v2 python tests/rag/test_ragas.py --ragas
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure src is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Graph answer generation timeout (single agent can be slow on first query;
# codebase indexing + reranker + deepseek-v4 rounds can exceed 4 min cold).
PER_QUERY_TIMEOUT = int(os.environ.get("RAGAS_ANSWER_TIMEOUT", "360"))


# ---------------------------------------------------------------------------
# Qdrant isolation: run against a private copy so we never collide with the
# running `langgraph dev` server (Qdrant local mode holds a file lock).
# ---------------------------------------------------------------------------


def _prepare_qdrant() -> None:
    """Point QDRANT_PATH at a private copy of the configured index."""
    import dotenv

    dotenv.load_dotenv()
    src = os.environ.get("QDRANT_PATH", "./qdrant_data_v2")
    if os.environ.get("QDRANT_EVAL_USE_EXISTING") == "1":
        return
    dst = str(_PROJECT_ROOT / "_ragas_qdrant")
    if Path(dst).is_dir() and (Path(dst) / "meta.json").is_file():
        os.environ["QDRANT_PATH"] = dst
        logger.info("Reusing eval index copy: %s", dst)
        return
    if Path(src).is_dir():
        if Path(dst).exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        os.environ["QDRANT_PATH"] = dst
        logger.info("Copied Qdrant index %s -> %s", src, dst)
    else:
        logger.warning("QDRANT_PATH=%s not found; will build from vault if needed", src)


def _ensure_vault_index() -> None:
    """Initialize vault if not already done.

    NOTE: must check `vault_tools._vault_retriever` as a module attribute —
    `from ... import _vault_retriever` binds the value at import time (None),
    so it would never see the initialized retriever.
    """
    from src.tools import vault_tools
    if vault_tools._vault_retriever is None:
        vault_path = os.environ.get(
            "OBSIDIAN_VAULT_PATH", str(_PROJECT_ROOT.parent / "obsidian-vault")
        )
        vault_tools._init_vault(vault_path)
        if vault_tools._vault_retriever is None:
            raise RuntimeError(
                f"Vault init failed (vault={vault_path}). "
                "Check OBSIDIAN_VAULT_PATH and QDRANT_PATH."
            )


def _check_dataset_coverage() -> dict:
    """Pre-flight: compare dataset relevant_notes against the Qdrant index.

    Prevents the 2026-08-20 incident where the benchmark ran against a partial
    index that did NOT contain the dataset's notes, producing hallucination-
    inflated scores.
    """
    from qdrant_client import QdrantClient

    from tests.rag.eval_dataset import EVAL_DATASET

    qdrant_path = os.environ.get("QDRANT_PATH", "./qdrant_data_v2")
    client = QdrantClient(path=qdrant_path)
    try:
        res = client.scroll(
            "obsidian_vault", limit=10_000, with_payload=["note_name"], with_vectors=False
        )
        indexed = {p.payload.get("note_name") for p in res[0] if p.payload.get("note_name")}
    finally:
        try:
            client.close()
        except Exception:
            pass

    expected = {n for case in EVAL_DATASET for n in case["relevant_notes"]}
    missing = sorted(expected - indexed)
    logger.info(
        "Dataset requires %d notes; index has %d of them -> %s",
        len(expected),
        len(expected & indexed),
        "OK" if not missing else f"MISSING: {missing}",
    )
    if missing:
        logger.warning(
            "WARNING: index %s lacks dataset notes: %s. Scores may be unreliable.",
            qdrant_path,
            missing,
        )
    return {"expected": expected, "indexed": indexed, "missing": missing}


def _run_retrieval(query: str, top_k: int = 5) -> List[dict]:
    """Run retrieval and return context chunks."""
    from src.tools import vault_tools
    if vault_tools._vault_retriever is None:
        raise RuntimeError("Vault not initialized")
    results = vault_tools._vault_retriever.search(
        query, top_k=top_k, expand_wikilinks=True
    )
    return results


# ---------------------------------------------------------------------------
# Answer generation via the REAL graph.
# Defaults to the multi_agent workflow (few rounds, ~3, so it completes
# reliably under slow SiliconFlow bursts). knowledge_agent (single) scores
# highest but often does 15-20 tool rounds and can exceed the timeout in
# batch runs. Switch with RAGAS_ANSWER_GRAPH=single|multi|supervisor.
# GuardrailsMiddleware registers async hooks, so the single agent must run
# via ainvoke on one asyncio loop (same constraint as benchmark_v2).
# ---------------------------------------------------------------------------


async def _generate_answer_single_async(question: str, idx: int) -> dict:
    from src.agent.knowledge_graph import docs_agent

    t0 = time.monotonic()
    try:
        state = await asyncio.wait_for(
            docs_agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": f"ragas-{idx}"}},
            ),
            timeout=PER_QUERY_TIMEOUT,
        )
        msgs = state.get("messages") or []
        answer = str(msgs[-1].content) if msgs else ""
        turns = sum(1 for m in msgs if getattr(m, "type", "") == "tool")
        return {
            "answer": answer,
            "latency": time.monotonic() - t0,
            "turns": turns,
            "error": None,
        }
    except asyncio.TimeoutError:
        return {"answer": "", "latency": time.monotonic() - t0,
                "turns": -1, "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"answer": "", "latency": time.monotonic() - t0,
                "turns": -1, "error": str(exc)[:200]}


async def _generate_answer_workflow_async(
    graph_name: str, question: str, idx: int
) -> dict:
    """multi_agent / supervisor are sync graphs -> run via asyncio.to_thread."""
    if graph_name == "multi":
        from src.agent.multi_agent_graph import multi_agent_graph

        def _invoke() -> dict:
            return multi_agent_graph.invoke(
                {"query": question},
                config={"configurable": {"thread_id": f"ragas-{idx}"}},
            )

        def _turns(final: dict) -> int:
            return int(final.get("attempts", 0) + 1)

        def _answer(final: dict) -> str:
            return str(final["messages"][-1].content) if final.get("messages") else ""
    else:  # supervisor
        from src.agent.supervisor_graph import supervisor_graph

        def _invoke() -> dict:
            return supervisor_graph.invoke(
                {"query": question},
                config={"configurable": {"thread_id": f"ragas-{idx}"}},
            )

        def _turns(final: dict) -> int:
            return int(final.get("iters", 0))

        def _answer(final: dict) -> str:
            return str(final.get("final", ""))

    t0 = time.monotonic()
    try:
        final = await asyncio.wait_for(
            asyncio.to_thread(_invoke), timeout=PER_QUERY_TIMEOUT
        )
        return {
            "answer": _answer(final),
            "latency": time.monotonic() - t0,
            "turns": _turns(final),
            "error": None,
        }
    except asyncio.TimeoutError:
        return {"answer": "", "latency": time.monotonic() - t0,
                "turns": -1, "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"answer": "", "latency": time.monotonic() - t0,
                "turns": -1, "error": str(exc)[:200]}


def _generate_answer_async(question: str, idx: int) -> "coroutine":
    graph = os.environ.get("RAGAS_ANSWER_GRAPH", "multi")
    if graph == "single":
        return _generate_answer_single_async(question, idx)
    return _generate_answer_workflow_async(graph, question, idx)


async def _generate_answers_async(questions: list[str]) -> list[dict]:
    """Gather answers inside one coroutine, capped at RAGAS_ANSWER_CONCURRENCY.

    asyncio.run needs a coroutine, not the raw Future from asyncio.gather.
    Running all N agents concurrently thrashes SiliconFlow (each answer does
    10-20 model calls), inflating per-query latency until queries hit the
    timeout: verified on 2026-08-20 - a query that finishes in 214s solo
    timed out at 360s under full 10-way concurrency. Pacing via a semaphore
    (default 2) fixes it; bump with RAGAS_ANSWER_CONCURRENCY.
    """
    sem = asyncio.Semaphore(int(os.environ.get("RAGAS_ANSWER_CONCURRENCY", "2")))

    async def _bounded(q: str, i: int) -> dict:
        async with sem:
            return await _generate_answer_async(q, i)

    return await asyncio.gather(*[_bounded(q, i) for i, q in enumerate(questions)])


# Cache generated answers so metric-only re-runs skip the slow graph phase.
_ANSWERS_CACHE = _PROJECT_ROOT / "tests" / "rag" / "_ragas_answers.json"


def _load_answer_cache() -> dict:
    if _ANSWERS_CACHE.is_file():
        try:
            import json

            return json.loads(_ANSWERS_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def _save_answer_cache(cache: dict) -> None:
    import json

    _ANSWERS_CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _generate_answers(questions: list[str]) -> list[dict]:
    """Run the real graph over all questions on a single asyncio loop.

    Caches answers in _ragas_answers.json; questions whose cached answer is
    non-empty are skipped. Set RAGAS_REFRESH_ANSWERS=1 to regenerate all.
    """
    cache = _load_answer_cache()
    refresh = os.environ.get("RAGAS_REFRESH_ANSWERS", "") == "1"
    todo = []
    results: list[dict] = []
    for i, q in enumerate(questions):
        cached = cache.get(q)
        if cached and cached.get("answer") and not refresh:
            results.append(cached)
        else:
            todo.append((i, q))
            results.append(None)

    if todo:
        logger.info("Generating answers with knowledge_agent graph (%d new/%d total)...",
                    len(todo), len(questions))
        new = asyncio.run(_generate_answers_async([q for _, q in todo]))
        for (i, q), rec in zip(todo, new):
            cache[q] = rec
            results[i] = rec
        _save_answer_cache(cache)
    return results


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------


def _build_judge_llm():
    """Build the judge LLM from the model registry (qwen3-8b by default)."""
    from langchain_openai import ChatOpenAI

    from src.agent.config import _get_model_by_key

    cfg = _get_model_by_key("JUDGE_MODEL_KEY", "qwen3-8b")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for RAGAS judge LLM")
    llm = ChatOpenAI(
        model=cfg.id,
        openai_api_key=api_key,
        openai_api_base=os.environ.get(
            "OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"
        ),
        temperature=0,
        max_tokens=1024,
        extra_body={"enable_thinking": False},  # judge 永远关 thinking 提速
    )
    logger.info("RAGAS judge model: %s (%s)", cfg.key, cfg.id)
    return llm, cfg


def _build_embeddings():
    """Embeddings for embedding-based RAGAS metrics (bge-m3 on SiliconFlow)."""
    from langchain_openai import OpenAIEmbeddings

    emb = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_api_base=os.environ.get(
            "OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"
        ),
    )
    logger.info("RAGAS embeddings: BAAI/bge-m3")
    return emb


# ---------------------------------------------------------------------------
# ragas compat shim
# ---------------------------------------------------------------------------


def _patch_ragas_imports() -> None:
    """Shim the langchain_community.vertexai imports ragas hard-requires.

    ragas 0.3/0.4 imports `langchain_community.chat_models.vertexai` and
    `langchain_community.llms.VertexAI` at module import time, but the
    langchain-community available in this environment doesn't ship that module
    (and a compatible version can't be installed without downgrading
    langchain-core). ragas only uses these classes for `isinstance` checks in
    MULTIPLE_COMPLETION_SUPPORTED, so dummy classes are safe.
    """
    import sys
    import types

    if "langchain_community.chat_models.vertexai" in sys.modules:
        return

    lc = types.ModuleType("langchain_community")
    chat = types.ModuleType("langchain_community.chat_models")
    vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    vertex.ChatVertexAI = type("ChatVertexAI", (), {})
    chat.vertexai = vertex
    lc.chat_models = chat
    llms = types.ModuleType("langchain_community.llms")
    llms.VertexAI = type("VertexAI", (), {})
    lc.llms = llms
    sys.modules["langchain_community"] = lc
    sys.modules["langchain_community.chat_models"] = chat
    sys.modules["langchain_community.chat_models.vertexai"] = vertex
    sys.modules["langchain_community.llms"] = llms


def _to_mean(value) -> float:
    """Convert a ragas score (float, list, or ndarray) to a scalar mean."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            arr = value.astype(float)
            return float(arr.mean()) if arr.size else 0.0
    except Exception:  # noqa: BLE001 - numpy optional
        pass
    if isinstance(value, (list, tuple)):
        return float(sum(value) / len(value)) if value else 0.0
    return float(value)


def _run_ragas_evaluation() -> dict:
    """Run RAGAS evaluation framework (answers generated by the real graph).

    Requires: pip install -e ".[eval]" (ragas + datasets)
    """
    _patch_ragas_imports()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from tests.rag.eval_dataset import EVAL_DATASET

    # Generate answers with the real graph first (single asyncio loop).
    logger.info("Generating answers with knowledge_agent graph (%d questions)...",
                len(EVAL_DATASET))
    results = _generate_answers([c["question"] for c in EVAL_DATASET])

    ragas_data = []
    for case, res in zip(EVAL_DATASET, results):
        query = case["question"]
        retrieved = _run_retrieval(query, top_k=3)
        contexts = [r["text"][:500] for r in retrieved[:3]]
        status = f" ({res['latency']:.0f}s, {res['turns']} tool rounds)" if not res["error"] else f" ({res['error']})"
        logger.info(
            "Q: %s... answer%s", query[:40], status
        )
        ragas_data.append({
            "question": query,
            "answer": res["answer"],
            "contexts": contexts,
            "ground_truth": case["expected_answer"],
        })

    dataset = Dataset.from_list(ragas_data)

    metric_names = [
        m for m in (
            os.environ.get("RAGAS_METRICS", "").split(",") if os.environ.get("RAGAS_METRICS")
            else ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        ) if m
    ]
    all_metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    metrics = [all_metrics[name] for name in metric_names]
    logger.info("RAGAS metrics: %s", metric_names)

    import ragas
    version = tuple(int(x) for x in ragas.__version__.split(".")[:2])

    if version >= (0, 3):
        # ragas 0.3+: metrics are instances; set llm/embeddings per metric.
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig

        llm_wrapper = LangchainLLMWrapper(_build_judge_llm()[0])
        emb_wrapper = LangchainEmbeddingsWrapper(_build_embeddings())
        for m in metrics:
            m.llm = llm_wrapper
            m.embeddings = emb_wrapper
        # Longer per-job timeout: SiliconFlow occasionally exceeds ragas's
        # default 60s job timeout and TimeoutError made faithfulness NaN on
        # 2026-08-20. Must be set on the WRAPPER itself (evaluate(run_config=)
        # does not propagate to a pre-configured metric llm).
        rc = RunConfig(timeout=240, max_retries=3)
        llm_wrapper = LangchainLLMWrapper(_build_judge_llm()[0])
        llm_wrapper.run_config = rc
        emb_wrapper = LangchainEmbeddingsWrapper(_build_embeddings())
        emb_wrapper.run_config = rc
        for m in metrics:
            m.llm = llm_wrapper
            m.embeddings = emb_wrapper
        result = evaluate(dataset, metrics=metrics)
        # RagasResult["metric"] returns float mean (0.3) or per-row list (0.4)
        scores = {
            name: round(_to_mean(result[name]), 4) for name in metric_names
        }
    else:
        # ragas 0.2.x: evaluate(..., llm=..., embeddings=...)
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=_build_judge_llm()[0],
            embeddings=_build_embeddings(),
        )
        scores = {k: round(float(v), 4) for k, v in result.items()}

    logger.info("RAGAS Metrics:")
    for name, score in scores.items():
        logger.info("  %s: %.3f", name, float(score))

    return {"metrics": {k: round(float(v), 4) for k, v in scores.items()},
            "per_query": [{
                "question": c["question"][:60],
                "answer_len": len(r["answer"]),
                "latency": round(r["latency"], 1),
                "turns": r["turns"],
                "error": r["error"],
            } for c, r in zip(EVAL_DATASET, results)]}


def run_evaluation(use_ragas: bool = True) -> dict:
    """Run the full evaluation suite.

    Args:
        use_ragas: If True, use RAGAS framework. If False, do basic metrics only.
    """
    from tests.rag.eval_dataset import EVAL_DATASET

    _prepare_qdrant()
    # Coverage scan must run BEFORE _ensure_vault_index: the indexer holds the
    # local Qdrant dir open, so a second QdrantClient would hit the file lock.
    _check_dataset_coverage()
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
            results["ragas_metrics"] = ragas_results["metrics"]
            results["answer_stats"] = ragas_results["per_query"]
        except ImportError as exc:
            logger.warning(
                "RAGAS not installed. Skipping RAGAS metrics. Install: pip install -e \".[eval]\" (%s)",
                exc,
            )
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")

    return results


def main():
    """CLI entry point for running evaluation."""
    use_ragas = "--ragas" in sys.argv or os.environ.get("USE_RAGAS", "").lower() == "true"
    results = run_evaluation(use_ragas=use_ragas)

    print(f"\n{'='*50}")
    print(f"Evaluation Summary")
    print(f"{'='*50}")
    print(f"Total queries: {results['total_queries']}")
    print(f"Retrieval Hit Rate: {results['retrieval_hit_rate']:.1%}")

    if results.get("ragas_metrics"):
        print(f"\nRAGAS Metrics:")
        for k, v in results["ragas_metrics"].items():
            print(f"  {k}: {v:.3f}")

    if results.get("answer_stats"):
        print(f"\nAnswer Generation Stats:")
        for q in results["answer_stats"]:
            err = f" ({q['error']})" if q["error"] else f" ({q['latency']}s, {q['turns']} rounds)"
            print(f"  {q['question']}{err}")

    print(f"\nPer-Query Results:")
    for q in results["per_query"]:
        status = "✓" if q["hit"] else "✗"
        print(f"  {status} {q['question']}")

    return results


if __name__ == "__main__":
    main()

"""Benchmark the three agent graphs on the eval dataset.

Compares:
  1. knowledge_agent  — single create_agent (original)
  2. multi_agent      — workflow-style multi-role (Planner-Executor-Summarizer-Critic)
  3. supervisor_agent — central-dispatcher (Supervisor + 3 workers)

Metrics per query:
  - answer quality (LLM-as-judge, 1-5)
  - latency (seconds)
  - rounds (tool-call turns / loop iterations)

Notes:
  - Uses a single async main loop (asyncio.run once) because
    knowledge_agent's GuardrailsMiddleware registers async hooks and
    running asyncio.run() inside worker threads can hang LangGraph's
    async stack. Everything (graphs + judge) is awaited on one loop.
  - Qdrant data dir is copied into a private dir to avoid colliding with
    a running `langgraph dev` server's file lock.
  - Requires a live API key (OPENAI_API_KEY / OPENAI_BASE_URL).

Run:
    python tests/rag/benchmark_graphs.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.WARNING)

PER_QUERY_TIMEOUT = 240  # 单 query 单 graph 超时(秒)


def _prepare_qdrant() -> str:
    """Isolate the benchmark from a running dev server's Qdrant lock.

    Qdrant local mode locks its data dir; if `langgraph dev` is running it
    already holds the lock on QDRANT_PATH. We copy the existing index into
    a private dir and point QDRANT_PATH at it (read-only usage here).
    Set QDRANT_BENCH_USE_EXISTING=1 to use the configured path directly.
    """
    src = os.environ.get("QDRANT_PATH", "./qdrant_data_v2")
    if os.environ.get("QDRANT_BENCH_USE_EXISTING") == "1":
        return src
    dst = str(_PROJECT_ROOT / "_bench_qdrant")
    dst_dir = Path(dst)
    if dst_dir.is_dir() and (dst_dir / "meta.json").is_file():
        os.environ["QDRANT_PATH"] = dst
        return dst
    if Path(src).is_dir():
        import shutil

        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src, dst)
        os.environ["QDRANT_PATH"] = dst
        print(f"[bench] copied Qdrant index {src} -> {dst}", flush=True)
    return dst


# ---------------------------------------------------------------------------
# Graph runners (async)
# ---------------------------------------------------------------------------


async def _run_knowledge_agent(query: str, tid: str) -> tuple[str, int]:
    """Single create_agent. Returns (answer, tool-call turns)."""
    from src.agent.knowledge_graph import docs_agent

    state = await docs_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": tid}},
    )
    msgs = state.get("messages") or []
    answer = str(msgs[-1].content) if msgs else ""
    turns = sum(1 for m in msgs if getattr(m, "type", "") == "tool")
    return answer, turns


async def _run_multi_agent(query: str, tid: str) -> tuple[str, int]:
    """Workflow-style multi-role graph. Returns (answer, retrieval rounds).

    Uses sync invoke via asyncio.to_thread: all multi-agent nodes are
    sync functions (no async middleware), and ainvoke timed out in testing.
    """
    from src.agent.multi_agent_graph import multi_agent_graph

    def _invoke() -> dict:
        return multi_agent_graph.invoke(
            {"query": query},
            config={"configurable": {"thread_id": tid}},
        )

    final = await asyncio.to_thread(_invoke)
    answer = str(final["messages"][-1].content) if final.get("messages") else ""
    rounds = final.get("attempts", 0) + 1  # attempts=重试轮,总轮=attempts+1
    return answer, rounds


async def _run_supervisor_agent(query: str, tid: str) -> tuple[str, int]:
    """Central-dispatcher graph. Returns (answer, supervisor rounds)."""
    from src.agent.supervisor_graph import supervisor_graph

    def _invoke() -> dict:
        return supervisor_graph.invoke(
            {"query": query},
            config={"configurable": {"thread_id": tid}},
        )

    final = await asyncio.to_thread(_invoke)
    answer = final.get("final", "")
    rounds = final.get("iters", 0)
    return answer, rounds


# ---------------------------------------------------------------------------
# LLM-as-judge (async)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "你是严格的答案质量评审。根据标准答案,对 AI 回答打分 1-5:\n"
    "5=准确且完整覆盖关键点;4=准确但略有遗漏;3=部分正确,有关键遗漏;"
    "2=大部分错误;1=完全错误或幻觉。\n"
    "只输出 JSON: {\"score\": 1..5, \"reason\": \"一句话理由\"}"
)


async def _judge(model, question: str, answer: str, expected: str) -> Optional[int]:
    """Score an answer against ground truth; None when unjudgeable."""
    if not answer:
        return None
    from langchain_core.messages import HumanMessage, SystemMessage

    user = (
        f"用户问题: {question}\n"
        f"标准答案: {expected}\n"
        f"AI 回答: {answer[:2000]}"
    )
    try:
        resp = await model.ainvoke(
            [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=user)]
        )
        text = str(resp.content or "").strip()
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        data = json.loads(text)
        score = int(data.get("score", 0))
        return score if 1 <= score <= 5 else None
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("Judge failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


async def _run_one(fn: Callable, query: str, tid: str) -> dict:
    """Run one graph on one query with timeout. Returns raw record."""
    t0 = time.monotonic()
    try:
        answer, rounds = await asyncio.wait_for(fn(query, tid), PER_QUERY_TIMEOUT)
        return {"answer": answer, "rounds": rounds,
                "latency": time.monotonic() - t0, "error": None}
    except asyncio.TimeoutError:
        return {"answer": "", "rounds": -1, "latency": time.monotonic() - t0,
                "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"answer": "", "rounds": -1, "latency": time.monotonic() - t0,
                "error": str(exc)[:200]}


async def _run_benchmark_async() -> dict:
    """Async core: iterate dataset, run graphs sequentially, judge each."""
    from src.agent.config import _get_model_by_key
    from langchain.chat_models import init_chat_model

    from tests.rag.eval_dataset import EVAL_DATASET

    judge_model = init_chat_model(
        model=_get_model_by_key("JUDGE_MODEL_KEY", "deepseek-v4").id,
        model_provider="openai",
        temperature=0,
    )

    graphs = {
        "单Agent knowledge_agent": _run_knowledge_agent,
        "多角色 multi_agent": _run_multi_agent,
        "总调度 supervisor_agent": _run_supervisor_agent,
    }

    per_query = []
    for idx, case in enumerate(EVAL_DATASET):
        q = case["question"]
        for name, fn in graphs.items():
            tid = f"bench-{idx}-{name}"
            rec = await _run_one(fn, q, tid)
            rec["graph"] = name
            rec["question"] = q
            rec["expected"] = case["expected_answer"]
            rec["score"] = (
                await _judge(judge_model, q, rec["answer"], case["expected_answer"])
                if not rec["error"] else None
            )
            per_query.append(rec)
            status = f"{rec['score']}/5" if rec["score"] else (
                f"ERR({rec['error']})" if rec["error"] else "n/a"
            )
            print(
                f"[{name}] Q{idx + 1}: score={status} "
                f"latency={rec['latency']:.1f}s rounds={rec['rounds']}",
                flush=True,
            )

    # ---- aggregate ----
    summary = {}
    for name in graphs:
        rows = [r for r in per_query if r["graph"] == name]
        ok = [r for r in rows if r["score"] is not None]
        summary[name] = {
            "avg_score": round(sum(r["score"] for r in ok) / len(ok), 2) if ok else None,
            "avg_latency": round(sum(r["latency"] for r in rows) / len(rows), 2),
            "avg_rounds": round(sum(r["rounds"] for r in rows) / len(rows), 2),
            "judged": len(ok),
            "errors": sum(1 for r in rows if r["error"]),
        }

    _print_summary(summary)

    out = {
        "summary": summary,
        "per_query": [
            {k: r[k] for k in ("graph", "question", "score", "latency", "rounds", "error")}
            for r in per_query
        ],
    }
    result_path = _PROJECT_ROOT / "tests" / "rag" / "benchmark_results.json"
    result_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRaw results saved to: {result_path}", flush=True)
    return summary


def run_benchmark() -> dict:
    """Entry point: prepare env, then run the async core on one loop."""
    _prepare_qdrant()
    return asyncio.run(_run_benchmark_async())


def _print_summary(summary: dict) -> None:
    print("\n================ 汇总 ================", flush=True)
    print(f"{'Graph':<32}{'质量分':<8}{'延迟(s)':<10}{'轮数':<8}{'错误'}", flush=True)
    print("-" * 68, flush=True)
    for name, s in summary.items():
        score = f"{s['avg_score']}/5" if s["avg_score"] else "n/a"
        print(
            f"{name:<32}{score:<8}{s['avg_latency']:<10}"
            f"{s['avg_rounds']:<8}{s['errors']}",
            flush=True,
        )


if __name__ == "__main__":
    run_benchmark()

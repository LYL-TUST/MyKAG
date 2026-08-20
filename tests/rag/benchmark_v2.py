"""Optimised benchmark: 3 graphs x eval dataset, with judge parallelism,
incremental persistence, and resume.

Improvements over benchmark_graphs.py (v1):
  - **judge parallelised per case**: after a case's 3 graph runs finish, all 3
    judges run concurrently via asyncio.gather (judge is a single cheap LLM
    call, safe to gather under the same loop).
  - **incremental persistence (即跑即存)**: each finished (graph, question)
    record is appended to a JSONL file and flushed immediately. Crashes no
    longer waste completed cases.
  - **resume (断点续跑)**: at start we load all previously recorded
    (graph, question) pairs from the JSONL and skip them. So if v1 got 20/30
    last time, v2 picks up where it left off and completes the remaining 10.
  - **judge uses a cheaper model**: defaults to qwen3-8b (SiliconFlow), which
    is faster and cheaper than deepseek-v4 for the simple 1-5 scoring task.
    Override with JUDGE_MODEL_KEY env var.
  - **single async loop preserved**: knowledge_agent's GuardrailsMiddleware
    registers async hooks, and asyncio.run() inside worker threads hangs
    LangGraph's async stack. So the graph calls themselves stay sequential
    on one loop. Concurrency is only at the judge layer (which uses no
    middleware).

Output:
  - tests/rag/_bench_results.jsonl  (append-only, one record per line)
  - tests/rag/benchmark_results.json (rebuilt at the end; compatible with
    report_benchmark.py)
  - docs/agent_benchmark.md (auto-generated when all cases finish)

Run:
    python tests/rag/benchmark_v2.py
    JUDGE_MODEL_KEY=qwen3-8b python tests/rag/benchmark_v2.py  # explicit
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

logging.basicConfig(level=logging.INFO)
# Keep the noisy HTTP/LLM client logs quiet; graph INFO (tool calls, vault
# init, retries) stays visible so a hang is diagnosable from the log.
for _noisy in ("httpx", "httpcore", "openai", "urllib3", "http.client"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Allow longer for the slowest cases (single agent can hit 400s+ on deep
# multi-round answers; v1 used 240s and v2's first run stalled on an un-
# guarded judge call). Both graph runs and judge get hard timeouts now.
PER_QUERY_TIMEOUT = int(os.environ.get("BENCH_QUERY_TIMEOUT", "600"))

# Result files ----------------------------------------------------------
_JSONL_PATH = _PROJECT_ROOT / "tests" / "rag" / "_bench_results.jsonl"
_JSON_PATH = _PROJECT_ROOT / "tests" / "rag" / "benchmark_results.json"


# ---------------------------------------------------------------------------
# Qdrant isolation (lifted from v1)
# ---------------------------------------------------------------------------


def _prepare_qdrant() -> str:
    """Copy index to a private dir to avoid colliding with `langgraph dev`.

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
# Graph runners (async) - identical logic to v1
# ---------------------------------------------------------------------------


async def _run_knowledge_agent(query: str, tid: str) -> tuple[str, int]:
    from src.agent.knowledge_graph import docs_agent

    # wait_for guards against a hung upstream call (SiliconFlow occasionally
    # stalls on a socket; without a timeout the whole benchmark stalls - seen
    # 2026-08-20, faulthandler dump showed the loop parked in asyncio.select).
    state = await asyncio.wait_for(
        docs_agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": tid}},
        ),
        timeout=PER_QUERY_TIMEOUT,
    )
    msgs = state.get("messages") or []
    answer = str(msgs[-1].content) if msgs else ""
    turns = sum(1 for m in msgs if getattr(m, "type", "") == "tool")
    return answer, turns


async def _run_multi_agent(query: str, tid: str) -> tuple[str, int]:
    from src.agent.multi_agent_graph import multi_agent_graph

    def _invoke() -> dict:
        return multi_agent_graph.invoke(
            {"query": query},
            config={"configurable": {"thread_id": tid}},
        )

    final = await asyncio.wait_for(
        asyncio.to_thread(_invoke), timeout=PER_QUERY_TIMEOUT
    )
    answer = str(final["messages"][-1].content) if final.get("messages") else ""
    rounds = final.get("attempts", 0) + 1
    return answer, rounds


async def _run_supervisor_agent(query: str, tid: str) -> tuple[str, int]:
    from src.agent.supervisor_graph import supervisor_graph

    def _invoke() -> dict:
        return supervisor_graph.invoke(
            {"query": query},
            config={"configurable": {"thread_id": tid}},
        )

    final = await asyncio.wait_for(
        asyncio.to_thread(_invoke), timeout=PER_QUERY_TIMEOUT
    )
    answer = final.get("final", "")
    rounds = final.get("iters", 0)
    return answer, rounds


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "你是严格的答案质量评审。根据标准答案,对 AI 回答打分 1-5:\n"
    "5=准确且完整覆盖关键点;4=准确但略有遗漏;3=部分正确,有关键遗漏;"
    "2=大部分错误;1=完全错误或幻觉。\n"
    "只输出 JSON: {\"score\": 1..5, \"reason\": \"一句话理由\"}"
)


async def _judge(model, question: str, answer: str, expected: str) -> Optional[int]:
    if not answer:
        return None
    from langchain_core.messages import HumanMessage, SystemMessage

    user = (
        f"用户问题: {question}\n"
        f"标准答案: {expected}\n"
        f"AI 回答: {answer[:2000]}"
    )
    try:
        resp = await asyncio.wait_for(
            model.ainvoke(
                [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=user)]
            ),
            timeout=int(os.environ.get("BENCH_JUDGE_TIMEOUT", "120")),
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
# Persistence helpers (即跑即存 + 断点续跑)
# ---------------------------------------------------------------------------


def _load_done_set() -> set[tuple[str, str]]:
    """Return the set of (graph, question) already recorded in JSONL.

    Only records with a non-None score count as done: a judge failure
    (SiliconFlow stall -> wait_for timeout -> score None) must NOT be
    treated as completed, or a later resume would skip re-judging it.
    """
    done: set[tuple[str, str]] = set()
    if _JSONL_PATH.is_file():
        for line in _JSONL_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("score") is not None:
                    done.add((rec["graph"], rec["question"]))
            except Exception:  # noqa: BLE001
                continue
    return done


def _append_jsonl(record: dict) -> None:
    """Append one record line + flush (durability across crashes)."""
    with _JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _rebuild_json_from_jsonl() -> int:
    """Re-read JSONL, aggregate summary + per_query, write benchmark_results.json."""
    if not _JSONL_PATH.is_file():
        return 0
    records: list[dict] = []
    for line in _JSONL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    summary: dict = {}
    for rec in records:
        name = rec["graph"]
        s = summary.setdefault(name, {
            "scores": [], "latencies": [], "rounds": [], "errors": 0
        })
        s["latencies"].append(rec.get("latency") or 0.0)
        s["rounds"].append(rec.get("rounds") if rec.get("rounds") is not None else 0)
        if rec.get("error"):
            s["errors"] += 1
        if rec.get("score") is not None:
            s["scores"].append(rec["score"])
    out_summary = {}
    for name, s in summary.items():
        out_summary[name] = {
            "avg_score": round(sum(s["scores"]) / len(s["scores"]), 2) if s["scores"] else None,
            "avg_latency": round(sum(s["latencies"]) / len(s["latencies"]), 2),
            "avg_rounds": round(sum(s["rounds"]) / len(s["rounds"]), 2),
            "judged": len(s["scores"]),
            "errors": s["errors"],
        }
    payload = {
        "summary": out_summary,
        "per_query": [
            {k: r[k] for k in ("graph", "question", "score", "latency", "rounds", "error")}
            for r in records
        ],
    }
    _JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(records)


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def _task_dump() -> str:
    """Stack traces of all pending asyncio tasks (diagnoses hangs: the
    faulthandler dump only shows the loop parked in select, not WHICH
    await never returned)."""
    import traceback

    lines: list[str] = []
    for t in asyncio.all_tasks():
        if t is not asyncio.current_task():
            st = t.get_stack()
            tb = "".join(traceback.format_list(st[-8:])) if st else "  (no stack)"
            lines.append(f"--- task {t.get_name()!r} done={t.done()} ---\n{tb}")
    return "\n".join(lines) or "  (no other tasks)"


async def _run_one(fn: Callable, query: str, tid: str) -> dict:
    t0 = time.monotonic()
    try:
        answer, rounds = await asyncio.wait_for(fn(query, tid), PER_QUERY_TIMEOUT)
        return {"answer": answer, "rounds": rounds,
                "latency": time.monotonic() - t0, "error": None}
    except asyncio.TimeoutError:
        logging.getLogger(__name__).warning(
            "TIMEOUT after %.0fs for %s; pending tasks:\n%s",
            time.monotonic() - t0, tid, _task_dump(),
        )
        return {"answer": "", "rounds": -1, "latency": time.monotonic() - t0,
                "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"answer": "", "rounds": -1, "latency": time.monotonic() - t0,
                "error": str(exc)[:200]}


async def _run_benchmark_async() -> dict:
    from src.agent.config import _get_model_by_key
    from langchain.chat_models import init_chat_model

    from tests.rag.eval_dataset import EVAL_DATASET

    # Judge: default to qwen3-8b (cheap/fast), override via env.
    judge_key = os.environ.get("JUDGE_MODEL_KEY", "qwen3-8b")
    judge_cfg = _get_model_by_key("JUDGE_MODEL_KEY", judge_key)
    judge_model = init_chat_model(
        model=judge_cfg.id,
        model_provider="openai",
        temperature=0,
        request_timeout=120,
        max_retries=0,
    )
    print(f"[bench] judge model: {judge_cfg.key} ({judge_cfg.id})", flush=True)

    graphs = {
        "单Agent knowledge_agent": _run_knowledge_agent,
        "多角色 multi_agent": _run_multi_agent,
        "总调度 supervisor_agent": _run_supervisor_agent,
    }

    done = _load_done_set()
    if done:
        print(f"[bench] resume: {len(done)} (graph, question) already done",
              flush=True)

    total_planned = len(EVAL_DATASET) * len(graphs)
    completed_before = len(done)

    for idx, case in enumerate(EVAL_DATASET):
        q = case["question"]
        exp = case["expected_answer"]
        case_records: list[dict] = []
        for name, fn in graphs.items():
            if (name, q) in done:
                continue
            tid = f"bench-v2-{idx}-{name}"
            rec = await _run_one(fn, q, tid)
            rec["graph"] = name
            rec["question"] = q
            rec["expected"] = exp
            rec["score"] = None  # filled below
            case_records.append(rec)

        # Judge pass: gather all pending judges for THIS case concurrently.
        if case_records:
            scores = await asyncio.gather(*[
                _judge(judge_model, r["question"], r["answer"], r["expected"])
                for r in case_records
            ])
            for r, s in zip(case_records, scores):
                r["score"] = s
                _append_jsonl(r)        # 即跑即存
                done.add((r["graph"], r["question"]))
                status = f"{r['score']}/5" if r["score"] else (
                    f"ERR({r['error']})" if r["error"] else "n/a"
                )
                print(
                    f"[{r['graph']}] Q{idx + 1}: score={status} "
                    f"latency={r['latency']:.1f}s rounds={r['rounds']}",
                    flush=True,
                )

    # Aggregate + dump .json (so report_benchmark.py can consume it).
    total = _rebuild_json_from_jsonl()
    _print_done_progress(completed_before, total_planned, total)

    summary: dict = {}
    if _JSON_PATH.is_file():
        data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
    _print_summary(summary)
    print(f"\nRaw results -> {_JSONL_PATH}\nAggregated -> {_JSON_PATH}",
          flush=True)

    # Auto-generate the human-readable report when all cases are done.
    if total >= total_planned:
        _try_generate_report()
    else:
        print(
            f"\n[bench] {total}/{total_planned} done; rerun to resume.",
            flush=True,
        )
    return summary


def _print_done_progress(before: int, total: int, now: int) -> None:
    print(
        f"\n[bench] progress: {now}/{total} "
        f"(newly added this run: {now - before})",
        flush=True,
    )


def _print_summary(summary: dict) -> None:
    print("\n================ 汇总 ================", flush=True)
    print(f"{'Graph':<32}{'质量分':<8}{'延迟(s)':<10}{'轮数':<8}{'错误'}",
          flush=True)
    print("-" * 68, flush=True)
    for name, s in summary.items():
        score = f"{s['avg_score']}/5" if s["avg_score"] else "n/a"
        print(
            f"{name:<32}{score:<8}{s['avg_latency']:<10}"
            f"{s['avg_rounds']:<8}{s['errors']}",
            flush=True,
        )


def _try_generate_report() -> None:
    try:
        from tests.rag.report_benchmark import main as report_main

        report_main()
        print(
            f"\n[bench] report generated -> docs/agent_benchmark.md",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[bench] report generation failed: {exc}", flush=True)


def run_benchmark() -> dict:
    # Auto-dump thread stacks if a step hangs (seen 2026-08-20: first run
    # appeared stuck with no output; WARNING-level logging hid the cause).
    import faulthandler

    faulthandler.dump_traceback_later(600, exit=True)
    _prepare_qdrant()
    _check_dataset_coverage()
    return asyncio.run(_run_benchmark_async())


def _check_dataset_coverage() -> None:
    """Pre-flight: warn loudly if the index lacks the eval dataset's notes.

    Guards against the 2026-08-20 incident where the benchmark ran against a
    partial index (only the new vault notes), so every answer about the old
    vault was hallucinated and the LLM-judge still scored it 3-5.
    """
    from qdrant_client import QdrantClient

    from tests.rag.eval_dataset import EVAL_DATASET

    qdrant_path = os.environ.get("QDRANT_PATH", "./qdrant_data_v2")
    client = QdrantClient(path=qdrant_path)
    try:
        res = client.scroll(
            "obsidian_vault", limit=20_000,
            with_payload=["note_name"], with_vectors=False,
        )
        indexed = {p.payload.get("note_name") for p in res[0] if p.payload.get("note_name")}
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    expected = {n for case in EVAL_DATASET for n in case["relevant_notes"]}
    missing = sorted(expected - indexed)
    if missing:
        print(
            f"\n[WARNING] Index {qdrant_path} is MISSING dataset notes: {missing}\n"
            f"          Benchmark answers for those questions will be hallucinated.\n"
            f"          Fix: point QDRANT_PATH at an index containing ALL of {sorted(expected)}.\n",
            flush=True,
        )
    else:
        print(
            f"[bench] index coverage OK: all {len(expected)} dataset notes present",
            flush=True,
        )


if __name__ == "__main__":
    run_benchmark()
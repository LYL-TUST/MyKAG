"""Unit tests for the supervisor-style multi-agent graph.

Covers routing, fallbacks, the worker loop cap, message accumulation,
and end-to-end flows with all LLM/tool calls mocked (no API keys needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.messages import AIMessage

import src.agent.supervisor_graph as sg


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def test_graph_compiles_and_has_all_nodes() -> None:
    """Supervisor + 3 workers + answer must all be registered."""
    graph = sg.supervisor_graph
    for node in ("supervisor", "search", "code", "graph", "answer"):
        assert node in graph.nodes, f"missing node: {node}"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_route_valid_next() -> None:
    """Valid next values route to the matching worker."""
    for nxt in ("search", "code", "graph", "answer"):
        assert sg._route({"next": nxt, "iters": 0}) == nxt


def test_route_invalid_next_falls_back_to_search() -> None:
    """Unknown next values fall back to search."""
    assert sg._route({"next": "nonsense", "iters": 0}) == "search"


def test_route_forced_answer_at_cap() -> None:
    """The loop cap forces answer regardless of the decision."""
    assert sg._route({"next": "search", "iters": sg.MAX_ITERS}) == "answer"
    assert sg._route({"next": "code", "iters": sg.MAX_ITERS + 1}) == "answer"


# ---------------------------------------------------------------------------
# Supervisor decision + fallbacks
# ---------------------------------------------------------------------------


def test_supervisor_decision_fallback_first_round(monkeypatch) -> None:
    """No history + LLM failure -> search the vault first."""
    monkeypatch.setattr(sg, "_llm_json", lambda *a, **k: None)
    state = {"query": "测试问题", "messages": []}
    decision = sg._supervisor_decision(state)
    assert decision["next"] == "search"
    assert decision["sub_query"] == "测试问题"


def test_supervisor_decision_fallback_after_work(monkeypatch) -> None:
    """History exists + LLM failure -> answer (avoid infinite loop)."""
    monkeypatch.setattr(sg, "_llm_json", lambda *a, **k: None)
    state = {
        "query": "测试问题",
        "messages": [AIMessage(content="[笔记检索] 已有结果")],
    }
    decision = sg._supervisor_decision(state)
    assert decision["next"] == "answer"


def test_supervisor_decision_parses_llm(monkeypatch) -> None:
    """LLM JSON decision is parsed and sanitized."""
    monkeypatch.setattr(
        sg, "_llm_json",
        lambda *a, **k: {"next": "code", "sub_query": "retry middleware"},
    )
    decision = sg._supervisor_decision({"query": "q", "messages": []})
    assert decision == {"next": "code", "sub_query": "retry middleware"}


def test_supervisor_decision_rejects_invalid_next(monkeypatch) -> None:
    """LLM returning an unknown next value degrades to answer."""
    monkeypatch.setattr(
        sg, "_llm_json", lambda *a, **k: {"next": "hack", "sub_query": "x"},
    )
    decision = sg._supervisor_decision({"query": "q", "messages": []})
    assert decision["next"] == "answer"


def test_supervisor_node_bumps_iters() -> None:
    """Each supervisor round increments iters."""
    out = sg.supervisor_node({"query": "q", "iters": 2})
    assert out["iters"] == 3


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def test_search_worker_appends_message(monkeypatch) -> None:
    """Search worker results are appended to the shared history."""
    monkeypatch.setattr(sg, "search_vault", _FakeTool("### Result 1: 笔记内容"))
    out = sg.search_worker({"query": "q", "sub_query": "ellie 架构"})
    assert len(out["messages"]) == 1
    assert "[笔记检索]" in out["messages"][0].content


def test_code_worker_appends_message(monkeypatch) -> None:
    """Code worker results are appended to the shared history."""
    monkeypatch.setattr(sg, "search_codebase", _FakeTool("retry_middleware.py"))
    out = sg.code_worker({"query": "q", "sub_query": "retry"})
    assert "[源码检索]" in out["messages"][0].content


def test_graph_worker_appends_message(monkeypatch) -> None:
    """Graph worker results are appended to the shared history."""
    monkeypatch.setattr(sg, "get_note_graph", _FakeTool("out: [[A]], in: [[B]]"))
    out = sg.graph_worker({"query": "q", "sub_query": "ellie 架构设计"})
    assert "[图谱]" in out["messages"][0].content


def test_worker_error_degrades_gracefully(monkeypatch) -> None:
    """Tool exceptions degrade to an error note instead of crashing."""
    class _Boom:
        def invoke(self, args):
            raise RuntimeError("boom")

    monkeypatch.setattr(sg, "search_vault", _Boom())
    out = sg.search_worker({"query": "q", "sub_query": "x"})
    assert "[search_vault error" in out["messages"][0].content


# ---------------------------------------------------------------------------
# End-to-end (all LLM + tools mocked)
# ---------------------------------------------------------------------------


def test_end_to_end_supervisor_loop_then_answer(monkeypatch) -> None:
    """search -> back to supervisor -> answer, with history accumulated."""
    decisions = iter([
        {"next": "search", "sub_query": "ellie 架构"},
        {"next": "answer", "sub_query": ""},
    ])

    monkeypatch.setattr(sg, "_llm_json", lambda *a, **k: next(decisions))
    monkeypatch.setattr(sg, "search_vault", _FakeTool("### Result: ellie 架构笔记"))
    monkeypatch.setattr(sg, "_summarize", lambda q, r: "最终综合回答")

    final = sg.supervisor_graph.invoke(
        {"query": "ellie 的架构是什么"},
        config={"configurable": {"thread_id": "sv-loop-1"}},
    )

    assert final["iters"] == 2  # 两轮调度
    assert final["final"] is not None
    last = final["messages"][-1].content
    assert "最终综合回答" in last
    # worker 结果确实进入了全局历史
    assert any("[笔记检索]" in str(m.content) for m in final["messages"])


def test_end_to_end_multi_worker_sequence(monkeypatch) -> None:
    """search -> graph -> answer: supervisor picks different workers."""
    decisions = iter([
        {"next": "search", "sub_query": "MCP 协议"},
        {"next": "graph", "sub_query": "MCP 协议设计"},
        {"next": "answer", "sub_query": ""},
    ])

    monkeypatch.setattr(sg, "_llm_json", lambda *a, **k: next(decisions))
    monkeypatch.setattr(sg, "search_vault", _FakeTool("### Result: MCP 笔记"))
    monkeypatch.setattr(sg, "get_note_graph", _FakeTool("out: [[MCP 2.0]], in: []"))
    monkeypatch.setattr(sg, "_summarize", lambda q, r: "多 worker 综合答案")

    final = sg.supervisor_graph.invoke(
        {"query": "MCP 协议相关"},
        config={"configurable": {"thread_id": "sv-multi-1"}},
    )

    assert final["iters"] == 3
    contents = [str(m.content) for m in final["messages"]]
    assert any("[笔记检索]" in c for c in contents)
    assert any("[图谱]" in c for c in contents)


def test_end_to_end_loop_capped(monkeypatch) -> None:
    """Supervisor always choosing search still terminates at MAX_ITERS."""
    monkeypatch.setattr(
        sg, "_llm_json",
        lambda *a, **k: {"next": "search", "sub_query": "一直搜"},
    )
    monkeypatch.setattr(sg, "search_vault", _FakeTool("### Result: 内容"))
    monkeypatch.setattr(sg, "_summarize", lambda q, r: "兜底回答")

    final = sg.supervisor_graph.invoke(
        {"query": "困难问题"},
        config={"configurable": {"thread_id": "sv-cap-1"}},
    )

    assert final["iters"] >= sg.MAX_ITERS
    assert "兜底回答" in final["messages"][-1].content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTool:
    """Minimal stand-in for a LangChain tool with .invoke()."""

    def __init__(self, output: str) -> None:
        self._output = output

    def invoke(self, args: dict) -> str:
        return self._output

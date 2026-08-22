"""Unit tests for the multi-agent graph (Planner-Executor-Summarizer-Critic).

Covers graph topology, planner/critic fallbacks, closed-loop routing,
and an end-to-end run with all LLM/tool calls mocked (no API keys needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import src.agent.multi_agent_graph as mag


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def test_graph_compiles_and_has_all_nodes() -> None:
    """All six nodes must be registered on the compiled graph."""
    graph = mag.multi_agent_graph
    for node in ("analyze", "retrieve", "synthesize", "critique", "rewrite", "answer"):
        assert node in graph.nodes, f"missing node: {node}"


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def test_planner_fallback_without_llm(monkeypatch) -> None:
    """When the LLM fails, fall back to the raw query as the only sub-query."""
    monkeypatch.setattr(mag, "_llm_json", lambda *a, **k: None)
    subs, codebase = mag._plan_sub_queries("随便问什么")
    assert subs == ["随便问什么"]
    assert codebase is None


def test_planner_parses_llm_json(monkeypatch) -> None:
    """LLM JSON output should be parsed into sub-queries + codebase."""
    payload = {
        "sub_queries": ["ellie 重试", "Code Review 重试"],
        "codebase": "ellie",
    }
    monkeypatch.setattr(mag, "_llm_json", lambda *a, **k: payload)
    subs, codebase = mag._plan_sub_queries("ellie 和 Code Review 的重试逻辑区别")
    assert subs == ["ellie 重试", "Code Review 重试"]
    assert codebase == "ellie"


def test_planner_codebase_hint_fallback(monkeypatch) -> None:
    """Codebase hint kicks in when the LLM omits it but query mentions code."""
    payload = {"sub_queries": ["retry"], "codebase": None}
    monkeypatch.setattr(mag, "_llm_json", lambda *a, **k: payload)
    subs, codebase = mag._plan_sub_queries("retry middleware 是怎么实现的")
    assert subs == ["retry"]
    assert codebase == "ellie"


def test_planner_rejects_invalid_codebase(monkeypatch) -> None:
    """Invalid codebase values are normalized to None (unless hint matches)."""
    payload = {"sub_queries": ["普通问题"], "codebase": "nonsense"}
    monkeypatch.setattr(mag, "_llm_json", lambda *a, **k: payload)
    _, codebase = mag._plan_sub_queries("普通问题")
    assert codebase is None


def test_planner_trims_to_max_sub_queries(monkeypatch) -> None:
    """More than MAX_SUB_QUERIES sub-queries are trimmed."""
    payload = {
        "sub_queries": ["a", "b", "c", "d", "e"],
        "codebase": None,
    }
    monkeypatch.setattr(mag, "_llm_json", lambda *a, **k: payload)
    subs, _ = mag._plan_sub_queries("问题")
    assert len(subs) <= mag.MAX_SUB_QUERIES


# ---------------------------------------------------------------------------
# Critic (rule channel + LLM channel)
# ---------------------------------------------------------------------------


def test_critic_rule_empty_results() -> None:
    """Empty or error-only results must fail the rule channel."""
    assert mag._results_have_content({}) is False
    assert mag._results_have_content({"q1": "No relevant notes found"}) is False
    assert mag._results_have_content(
        {"q1": "Vault retriever not initialized."}
    ) is False


def test_critic_rule_has_content() -> None:
    """Real retrieval output must pass the rule channel."""
    results = {
        "q1": "### Result 1: ellie 架构 (score: 0.90)\n"
              "- Note: ellie 架构设计\n- Content: ...",
    }
    assert mag._results_have_content(results) is True


def test_critic_judge_llm_channel(monkeypatch) -> None:
    """LLM judge verdict drives the final decision when there is content."""
    results = {
        "q1": "### Result 1: xxx (score: 0.9)\n"
              "有实质内容的检索结果文本,长度超过 40 字符阈值,"
              "用于验证 LLM judge 通道的判定逻辑。",
    }

    monkeypatch.setattr(
        mag, "_llm_json",
        lambda *a, **k: {"sufficient": False, "gap": "缺源码"},
    )
    assert mag._critic_judge("问题", results) is False

    monkeypatch.setattr(
        mag, "_llm_json", lambda *a, **k: {"sufficient": True},
    )
    assert mag._critic_judge("问题", results) is True


def test_critic_judge_llm_unavailable_defaults_sufficient(monkeypatch) -> None:
    """When the LLM judge is unavailable, do not block the flow."""
    results = {
        "q1": "### Result 1: xxx (score: 0.9)\n"
              "有实质内容的检索结果文本,长度超过 40 字符阈值,"
              "用于验证 LLM judge 通道的判定逻辑。",
    }
    monkeypatch.setattr(mag, "_llm_json", lambda *a, **k: None)
    assert mag._critic_judge("问题", results) is True


# ---------------------------------------------------------------------------
# Routing (closed loop)
# ---------------------------------------------------------------------------


def test_route_loop_and_cap() -> None:
    """Unsatisfied + under cap -> rewrite; otherwise -> answer.

    MAX_LOOP_ATTEMPTS=1 (2026-08-21 延迟优化):最多 rewrite 重试 1 轮,
    attempts>=1 即出答案,空检索问题不再做第二次全量重检索。
    """
    assert mag.MAX_LOOP_ATTEMPTS == 1
    assert mag._route_after_critique({"satisfied": False, "attempts": 0}) == "rewrite"
    assert mag._route_after_critique({"satisfied": False, "attempts": 1}) == "answer"
    assert mag._route_after_critique({"satisfied": False, "attempts": 2}) == "answer"
    assert mag._route_after_critique({"satisfied": True, "attempts": 0}) == "answer"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def test_retrieve_node_parallel(monkeypatch) -> None:
    """All sub-queries are executed and collected into retrieval_results."""
    monkeypatch.setattr(mag, "_run_one_subquery", lambda q, c: f"result-for-{q}")
    state = {"query": "q", "sub_queries": ["a", "b", "c"], "codebase": None}
    out = mag.retrieve_node(state)
    assert out["retrieval_results"] == {
        "a": "result-for-a",
        "b": "result-for-b",
        "c": "result-for-c",
    }


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


def test_extract_sources() -> None:
    """Note names, [[wikilinks]] and file paths are extracted for citation."""
    results = {
        "q1": "### Result 1: ellie 架构 (score: 0.90)\n"
              "- Note: ellie 架构设计\n- Content: 参考 [[MCP 协议设计]]",
        "q2": "E:/agent-projects/ellie/src/middleware/retry_middleware.py 内容",
    }
    sources = mag._extract_sources(results)
    assert "ellie 架构设计" in sources
    assert "MCP 协议设计" in sources
    assert any("retry_middleware.py" in s for s in sources)


# ---------------------------------------------------------------------------
# End-to-end (all LLM + tools mocked)
# ---------------------------------------------------------------------------


def test_end_to_end_loop_then_answer(monkeypatch) -> None:
    """First critique unsatisfied -> one rewrite round -> final answer."""
    judge_calls = {"n": 0}

    monkeypatch.setattr(mag, "_plan_sub_queries", lambda q: ([q], None))
    monkeypatch.setattr(mag, "_summarize", lambda q, r: "综合回答草稿")
    monkeypatch.setattr(
        mag, "_critic_judge",
        lambda q, r: (judge_calls.__setitem__("n", judge_calls["n"] + 1)
                      or judge_calls["n"] >= 2),
    )
    monkeypatch.setattr(mag, "_rewrite_queries", lambda q, r, s: ["改写后的查询"])
    monkeypatch.setattr(mag, "_run_one_subquery", lambda q, c: f"内容[{q}]")

    final = mag.multi_agent_graph.invoke(
        {"query": "测试问题"},
        config={"configurable": {"thread_id": "e2e-loop-1"}},
    )

    assert judge_calls["n"] == 2  # 闭环触发了一轮重检索
    assert final["attempts"] == 1
    assert "综合回答草稿" in final["messages"][-1].content


def test_end_to_end_satisfied_first_try(monkeypatch) -> None:
    """A sufficient first retrieval finishes without looping."""
    monkeypatch.setattr(mag, "_plan_sub_queries", lambda q: ([q], None))
    monkeypatch.setattr(mag, "_summarize", lambda q, r: "一次检索就够")
    monkeypatch.setattr(mag, "_critic_judge", lambda q, r: True)
    monkeypatch.setattr(mag, "_run_one_subquery", lambda q, c: "内容")

    final = mag.multi_agent_graph.invoke(
        {"query": "简单问题"},
        config={"configurable": {"thread_id": "e2e-ok-1"}},
    )

    # 未触发闭环时 attempts 可能不存在,按 0 处理
    assert final.get("attempts", 0) == 0
    assert "一次检索就够" in final["messages"][-1].content


def test_end_to_end_loop_capped_at_two(monkeypatch) -> None:
    """Persistent insufficiency still terminates after the attempt cap."""
    monkeypatch.setattr(mag, "_plan_sub_queries", lambda q: ([q], None))
    monkeypatch.setattr(mag, "_summarize", lambda q, r: "始终不充分")
    monkeypatch.setattr(mag, "_critic_judge", lambda q, r: False)
    monkeypatch.setattr(mag, "_rewrite_queries", lambda q, r, s: ["再试一次"])
    monkeypatch.setattr(mag, "_run_one_subquery", lambda q, c: "内容")

    final = mag.multi_agent_graph.invoke(
        {"query": "困难问题"},
        config={"configurable": {"thread_id": "e2e-cap-1"}},
    )

    assert final["attempts"] == mag.MAX_LOOP_ATTEMPTS
    assert "始终不充分" in final["messages"][-1].content

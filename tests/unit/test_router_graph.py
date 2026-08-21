"""Unit tests for the entry Router graph (layered hybrid orchestration).

Covers the two-channel classifier (rules + LLM fallback), routing
branches, and end-to-end runs with sub-graphs mocked (no API keys needed).
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import src.agent.router_graph as rg


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def test_graph_compiles_and_has_all_nodes() -> None:
    """Router + three run branches + answer must be registered."""
    graph = rg.router_graph
    for node in ("classify", "run_single", "run_workflow", "run_supervisor", "answer"):
        assert node in graph.nodes, f"missing node: {node}"


# ---------------------------------------------------------------------------
# Rule classifier
# ---------------------------------------------------------------------------


def test_rule_simple_factual() -> None:
    """'是什么' style factual queries map to single."""
    assert rg._classify_by_rules("什么是知识图谱") == "single"
    assert rg._classify_by_rules("TF-IDF 是怎么工作的") == "single"


def test_rule_workflow_statistical() -> None:
    """Counting/listing queries map to workflow."""
    assert rg._classify_by_rules("统计我所有带 #Python 标签的笔记数量") == "workflow"
    assert rg._classify_by_rules("列出所有笔记") == "workflow"
    assert rg._classify_by_rules("有多少篇笔记") == "workflow"


def test_rule_supervisor_complex() -> None:
    """Complex/ambiguous queries map to supervisor."""
    q = "帮我梳理一下我这半年在分布式系统方面学到了什么,并规划下个月学习重点"
    assert rg._classify_by_rules(q) == "supervisor"
    assert rg._classify_by_rules("对比一下两个项目的架构") == "supervisor"


def test_rule_priority_supervisor_over_workflow() -> None:
    """Supervisor hints take priority over workflow hints."""
    q = "统计一下并梳理我这半年的学习情况"  # 同时含统计+梳理
    assert rg._classify_by_rules(q) == "supervisor"


# ---------------------------------------------------------------------------
# Meta / fast path
# ---------------------------------------------------------------------------


def test_is_meta_greetings_and_short_input() -> None:
    """Greetings/self-intro/ultra-short inputs are detected as meta."""
    for q in ("你是谁", "你好", "hi", "在吗", "?", "介绍一下你自己"):
        assert rg._is_meta(q), f"should be meta: {q!r}"


def test_is_meta_false_for_substantive_questions() -> None:
    """Real questions must NOT be short-circuited to the fast path."""
    for q in ("介绍一下 LangGraph 的 RAG 流程", "A 和 B 有什么区别", "什么是知识图谱"):
        assert not rg._is_meta(q), f"should not be meta: {q!r}"


def test_decision_path_fast() -> None:
    """Meta question: classify -> fast (mapped to run_fast in the graph)."""
    state = rg.classify_node({"query": "你是谁"})
    assert state["mode"] == "fast"
    assert rg._route(state) == "fast"


# ---------------------------------------------------------------------------
# LLM channel + fallback
# ---------------------------------------------------------------------------


def test_classify_llm_when_no_rule_signal(monkeypatch) -> None:
    """No keyword hints -> LLM decides (e.g. compare queries)."""
    monkeypatch.setattr(
        rg, "_llm_json", lambda *a, **k: {"mode": "supervisor"},
    )
    assert rg._classify("A 和 B 在实现上有什么区别") == "supervisor"


def test_classify_llm_fallback_to_single(monkeypatch) -> None:
    """LLM unavailable -> default single, never blocks."""
    monkeypatch.setattr(rg, "_llm_json", lambda *a, **k: None)
    assert rg._classify("A 和 B 在实现上有什么区别") == "single"


def test_classify_llm_rejects_invalid_mode(monkeypatch) -> None:
    """Invalid LLM mode falls back to single."""
    monkeypatch.setattr(rg, "_llm_json", lambda *a, **k: {"mode": "hack"})
    assert rg._classify("A 和 B 在实现上有什么区别") == "single"


def test_classify_rule_wins_without_llm_call(monkeypatch) -> None:
    """Strong rule hints skip the LLM channel entirely (fast path)."""
    called = {"n": 0}

    def _fake_llm_json(*a, **k):
        called["n"] += 1
        return {"mode": "single"}

    monkeypatch.setattr(rg, "_llm_json", _fake_llm_json)
    assert rg._classify("什么是知识图谱") == "single"
    assert called["n"] == 0  # 规则命中,零 LLM 调用


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_route_valid_and_fallback() -> None:
    """Route by mode; unknown modes fall back to single."""
    assert rg._route({"mode": "single"}) == "single"
    assert rg._route({"mode": "workflow"}) == "workflow"
    assert rg._route({"mode": "supervisor"}) == "supervisor"
    assert rg._route({"mode": "nonsense"}) == "single"
    assert rg._route({}) == "single"


# ---------------------------------------------------------------------------
# End-to-end routing (classify -> route, without invoking sub-graphs)
# NOTE: the compiled graph fixes node references at compile time, so
# monkeypatching node functions has no effect on a compiled graph, and
# invoking real sub-graphs would touch Qdrant (dev-server lock) + LLMs.
# We therefore verify the full decision path: classify_node -> _route.
# ---------------------------------------------------------------------------


def test_decision_path_single() -> None:
    """Factual query: classify -> single (mapped to run_single in the graph)."""
    state = rg.classify_node({"query": "什么是知识图谱"})
    assert state["mode"] == "single"
    assert rg._route(state) == "single"


def test_decision_path_workflow() -> None:
    """Statistical query: classify -> workflow (mapped to run_workflow)."""
    state = rg.classify_node({"query": "统计我所有带 #Python 标签的笔记数量"})
    assert state["mode"] == "workflow"
    assert rg._route(state) == "workflow"


def test_decision_path_supervisor() -> None:
    """Complex query: classify -> supervisor (mapped to run_supervisor)."""
    state = rg.classify_node(
        {"query": "帮我梳理一下这半年在分布式系统方面学到了什么,并规划下个月学习重点"}
    )
    assert state["mode"] == "supervisor"
    assert rg._route(state) == "supervisor"


def test_graph_conditional_edges_cover_all_modes() -> None:
    """classify's conditional-edge mapping covers all three modes."""
    assert rg._route({"mode": "single"}) == "single"
    assert rg._route({"mode": "workflow"}) == "workflow"
    assert rg._route({"mode": "supervisor"}) == "supervisor"


# ---------------------------------------------------------------------------
# Timeout guards (防止无结果问题无限重试 -> 7 分钟卡死 / token 爆炸)
# ---------------------------------------------------------------------------


def test_timeout_constants_configured() -> None:
    """Timeout guards must be positive and sane."""
    assert rg.SINGLE_TIMEOUT > 0
    assert rg.MULTI_TIMEOUT > rg.SINGLE_TIMEOUT
    assert "超时" in rg.TIMEOUT_MESSAGE


def test_recursion_limit_capped_in_langgraph_json() -> None:
    """recursion_limit must be capped so agents can't loop forever."""
    import json
    from pathlib import Path

    cfg = json.loads(
        Path(_PROJECT_ROOT / "langgraph.json").read_text(encoding="utf-8")
    )
    assert cfg["config"]["recursion_limit"] <= 30, (
        "recursion_limit 过大会让 Agent 对库外问题无限重试"
    )


def test_run_single_degrades_on_timeout(monkeypatch) -> None:
    """Single-agent timeout must degrade to multi_agent, not hang forever."""
    import sys
    import types

    async def _fake_ainvoke(*a, **k):
        raise asyncio.TimeoutError()

    # 注入 fake knowledge_graph 模块,避免 import 触发 Qdrant 初始化
    fake_knowledge = types.ModuleType("src.agent.knowledge_graph")
    fake_knowledge.docs_agent = types.SimpleNamespace(ainvoke=_fake_ainvoke)
    monkeypatch.setitem(sys.modules, "src.agent.knowledge_graph", fake_knowledge)

    async def _fake_workflow(query: str) -> str:
        return "多 Agent 兜底回答"

    monkeypatch.setattr(rg, "_invoke_workflow", _fake_workflow)

    out = asyncio.run(rg.run_single_node({"query": "幂函数与 exp 函数区别"}))
    assert "已自动降级" in out["answer"]
    assert "多 Agent 兜底回答" in out["answer"]


def test_run_workflow_timeout_message(monkeypatch) -> None:
    """Workflow timeout returns a friendly error instead of hanging."""
    import time

    async def _never(query: str) -> str:
        await asyncio.sleep(3600)  # 永不返回

    monkeypatch.setattr(rg, "_invoke_workflow", _never)
    monkeypatch.setattr(rg, "MULTI_TIMEOUT", 0.1)  # 极短超时加速测试

    start = time.monotonic()
    out = asyncio.run(rg.run_workflow_node({"query": "统计笔记数量"}))
    elapsed = time.monotonic() - start
    assert elapsed < 5, "超时保护必须快速返回"
    assert "超时" in out["answer"]


# ---------------------------------------------------------------------------
# Semantic cache (2026-08-21 optimization)
# ---------------------------------------------------------------------------


def test_check_cache_node_hit(monkeypatch) -> None:
    """Cache hit short-circuits with the cached answer + mode."""
    monkeypatch.setattr(
        rg, "_sem_cache",
        types.SimpleNamespace(
            get=lambda q: {"answer": "缓存答案", "mode": "single", "similarity": 0.95}
        ),
    )
    state = rg.check_cache_node(
        {"messages": [{"role": "user", "content": "ellie 支持哪些 LLM 后端"}]}
    )
    assert state["cache_hit"] is True
    assert state["answer"] == "缓存答案"
    assert state["mode"] == "single"


def test_check_cache_node_miss(monkeypatch) -> None:
    """Cache miss falls through to classification."""
    monkeypatch.setattr(rg, "_sem_cache", types.SimpleNamespace(get=lambda q: None))
    state = rg.check_cache_node(
        {"messages": [{"role": "user", "content": "ellie 支持哪些 LLM 后端"}]}
    )
    assert state["cache_hit"] is False
    assert "answer" not in state


def test_answer_node_put_cache_on_miss(monkeypatch) -> None:
    """A normal (non-cached) answer is written back to the cache."""
    puts: list = []
    monkeypatch.setattr(
        rg, "_sem_cache", types.SimpleNamespace(put=lambda q, a, m: puts.append((q, a, m)))
    )
    state = rg.answer_node(
        {
            "messages": [{"role": "user", "content": "测试问题"}],
            "answer": "回答内容",
            "mode": "single",
        }
    )
    assert len(puts) == 1
    assert puts[0] == ("测试问题", "回答内容", "single")
    assert state["messages"][0].content.startswith("[编排模式: single]")


def test_answer_node_skip_cache_on_hit(monkeypatch) -> None:
    """A cache hit must NOT re-put itself (avoids refresh loops)."""
    puts: list = []
    monkeypatch.setattr(
        rg, "_sem_cache", types.SimpleNamespace(put=lambda q, a, m: puts.append((q, a, m)))
    )
    state = rg.answer_node(
        {
            "messages": [{"role": "user", "content": "测试问题"}],
            "answer": "回答内容",
            "mode": "single",
            "cache_hit": True,
        }
    )
    assert puts == []
    assert state["messages"][0].content.startswith("[缓存命中]")


def test_fast_path_expanded_meta() -> None:
    """2026-08-21 fast-path expansion: more greetings/meta hit fast."""
    for q in ("你能做什么", "你会什么", "好的", "嗯", "你怎么用"):
        assert rg._is_meta(q), f"expected meta: {q}"
    for q in ("ellie 支持哪些 LLM 后端", "什么是知识图谱", "统计笔记数量"):
        assert not rg._is_meta(q), f"expected NOT meta: {q}"

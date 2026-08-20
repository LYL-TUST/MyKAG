"""Supervisor-style multi-agent orchestration graph.

Central-dispatcher pattern: a single supervisor LLM reads the whole
conversation history and dynamically routes each round to one of three
workers (vault search / codebase search / wikilink graph) or to the
answer node. Worker results are appended to the shared message list so
the supervisor never repeats work. The loop is bounded by MAX_ITERS.

This is the "real multi-agent" mode compared with the fixed-pipeline
``multi_agent_graph`` (workflow-style): routing decisions happen at
runtime per round instead of being fixed by the graph structure.

Expose in langgraph.json:

    "supervisor_agent": "./src/agent/supervisor_graph.py:supervisor_graph"
"""

from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.config import GUARDRAILS_MODEL
from src.agent.multi_agent_graph import (
    _extract_sources,
    _init_model,
    _llm_json,
    _summarize,
)
from src.tools.codebase_tools import search_codebase
from src.tools.vault_tools import get_note_graph, search_vault

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ITERS = 6  # 总调度轮数上限(防死循环)
WORKERS = ("search", "code", "graph")
_VALID_NEXT = WORKERS + ("answer",)

_supervisor_model = _init_model("SUPERVISOR_MODEL_KEY", GUARDRAILS_MODEL)


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------


class SupervisorState(TypedDict, total=False):
    """State shared across supervisor + workers."""

    messages: Annotated[list, add_messages]  # 全局工作历史(worker 结果累积)
    query: str  # 原始用户问题
    next: str  # supervisor 本轮决策: search | code | graph | answer
    sub_query: str  # 本次派发给 worker 的子任务
    iters: int  # 已执行的调度轮数
    final: str  # 最终回答


# ---------------------------------------------------------------------------
# Node 1: Supervisor — 总调度
# ---------------------------------------------------------------------------

_SUPERVISOR_SYSTEM = (
    "你是个人知识库的总调度。基于用户问题和已完成的工作,决定下一步动作。\n"
    "只输出 JSON(不要其他文字),格式:\n"
    '{"next": "search"|"code"|"graph"|"answer", "sub_query": "子任务描述"}\n'
    "可选动作:\n"
    "- search: 对笔记库做语义检索(sub_query 填检索词)\n"
    "- code: 检索项目源码(sub_query 填要找的实现,如 retry middleware)\n"
    "- graph: 查看某篇笔记的 [[wikilink]] 关联(sub_query 填笔记名)\n"
    "- answer: 信息已足够回答用户问题\n"
    "规则:\n"
    "- 先分析用户问题,选择最可能命中的动作\n"
    "- 不要重复已完成的工作:如果某类检索已经做过,换其他动作\n"
    "- 已有足够信息时,next 必须为 answer\n"
)


def _format_history(state: SupervisorState) -> str:
    """Compile the worker history into a readable transcript."""
    msgs = state.get("messages") or []
    parts = []
    for m in msgs:
        content = str(m.content or "")
        parts.append(f"[{m.type}] {content[:500]}")
    return "\n".join(parts) or "(暂无已完成的工作)"


def _supervisor_decision(state: SupervisorState) -> dict:
    """Ask the supervisor model for the next action; degrade on failure."""
    history = _format_history(state)
    user = f"用户问题: {state.get('query', '')}\n\n已完成的工作:\n{history}"
    data = _llm_json(_supervisor_model, _SUPERVISOR_SYSTEM, user)
    if data is None:
        # 降级:没干过活就搜,干过就回答(保证不会死循环)
        nxt = "search" if not state.get("messages") else "answer"
        return {"next": nxt, "sub_query": state.get("query", "")}
    nxt = data.get("next", "answer")
    if nxt not in _VALID_NEXT:
        nxt = "answer"
    return {
        "next": nxt,
        "sub_query": data.get("sub_query") or state.get("query", ""),
    }


def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor node: decide next worker (or answer) and bump round count."""
    decision = _supervisor_decision(state)
    iters = state.get("iters", 0) + 1
    logger.info(
        "Supervisor round %d: next=%s sub_query=%r",
        iters, decision["next"], decision["sub_query"],
    )
    return {**decision, "iters": iters}


# ---------------------------------------------------------------------------
# Nodes 2-4: Workers — 执行子任务,结果追加到全局历史
# ---------------------------------------------------------------------------


def search_worker(state: SupervisorState) -> dict:
    """Vault semantic-search worker."""
    query = state.get("sub_query") or state.get("query", "")
    try:
        result = search_vault.invoke(
            {"query": query, "top_k": 5, "expand_wikilinks": True}
        )
    except Exception as exc:  # noqa: BLE001
        result = f"[search_vault error: {exc}]"
    return {"messages": [AIMessage(content=f"[笔记检索] {result}")]}


def code_worker(state: SupervisorState) -> dict:
    """Codebase search worker."""
    query = state.get("sub_query") or state.get("query", "")
    try:
        result = search_codebase.invoke(
            {"query": query, "project": "ellie", "top_k": 5}
        )
    except Exception as exc:  # noqa: BLE001
        result = f"[search_codebase error: {exc}]"
    return {"messages": [AIMessage(content=f"[源码检索] {result}")]}


def graph_worker(state: SupervisorState) -> dict:
    """Wikilink graph worker (sub_query should be a note name)."""
    note = state.get("sub_query") or ""
    try:
        result = get_note_graph.invoke({"note_name": note})
    except Exception as exc:  # noqa: BLE001
        result = f"[get_note_graph error: {exc}]"
    return {"messages": [AIMessage(content=f"[图谱] {result}")]}


# ---------------------------------------------------------------------------
# Node 5: Answer — 融合全部 worker 结果输出最终回答
# ---------------------------------------------------------------------------


def answer_node(state: SupervisorState) -> dict:
    """Answer node: fuse the whole worker history into the final reply."""
    history = "\n\n".join(str(m.content or "") for m in (state.get("messages") or []))
    draft = _summarize(state.get("query", ""), {"综合检索结果": history})
    sources = _extract_sources({"history": history})
    footer = (
        f"\n\n---\n**信息来源**: {', '.join(sources)}"
        if sources else "\n\n---\n**信息来源**: 无(检索未命中)"
    )
    final = draft + footer
    return {"final": final, "messages": [AIMessage(content=final)]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route(state: SupervisorState) -> str:
    """Route to the decided worker; force answer when the cap is reached."""
    if state.get("iters", 0) >= MAX_ITERS:
        return "answer"
    nxt = state.get("next", "search")
    return nxt if nxt in _VALID_NEXT else "search"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_builder = StateGraph(SupervisorState)
_builder.add_node("supervisor", supervisor_node)
_builder.add_node("search", search_worker)
_builder.add_node("code", code_worker)
_builder.add_node("graph", graph_worker)
_builder.add_node("answer", answer_node)

_builder.add_edge(START, "supervisor")
_builder.add_conditional_edges(
    "supervisor",
    _route,
    {"search": "search", "code": "code", "graph": "graph", "answer": "answer"},
)
# 所有 worker 干完都回到 supervisor 再决策 —— 中央调度的精髓
for _w in WORKERS:
    _builder.add_edge(_w, "supervisor")
_builder.add_edge("answer", END)

# NOTE: no custom checkpointer — LangGraph dev/platform manages persistence
# itself and rejects user-supplied checkpointers (graph load would fail).
supervisor_graph = _builder.compile()

logger.info(
    "Supervisor graph compiled: supervisor -> (search|code|graph -> "
    "supervisor)*<=%d -> answer", MAX_ITERS,
)

__all__ = [
    "SupervisorState",
    "supervisor_graph",
    "MAX_ITERS",
    # nodes
    "supervisor_node",
    "search_worker",
    "code_worker",
    "graph_worker",
    "answer_node",
    # routing / helpers (unit-testable)
    "_route",
    "_supervisor_decision",
    "_format_history",
]

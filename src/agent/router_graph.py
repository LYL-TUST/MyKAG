"""Entry Router graph — layered hybrid orchestration.

Routes each user query to the best-suited orchestration mode:

- **single**     : simple factual Q&A          -> knowledge_agent (fastest)
- **workflow**   : fixed/statistical tasks      -> multi_agent (deterministic pipeline)
- **supervisor** : complex/ambiguous tasks      -> supervisor_agent (dynamic dispatch)

Routing is a two-channel classifier:
  1. Rule channel (zero LLM cost, instant): keyword hints for each mode.
  2. LLM channel: only consulted when rules give no strong signal
     (e.g. "A 和 B 有什么区别" has no hint keywords); falls back to
     "single" when the LLM is unavailable so the graph never blocks.

The sub-graphs are invoked from async nodes: knowledge_agent must be
awaited (its GuardrailsMiddleware registers async hooks), while the two
multi-agent graphs are sync and run via asyncio.to_thread.

Expose in langgraph.json:

    "router_agent": "./src/agent/router_graph.py:router_graph"
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.config import GUARDRAILS_MODEL
from src.agent.multi_agent_graph import _init_model, _llm_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES = ("single", "workflow", "supervisor")
DEFAULT_MODE = "single"

# Meta / trivial 问题(问候、自我介绍、极短输入)——直接走轻模型 fast path,
# 跳过 router LLM 分类 + guardrails + RAG + 主模型思考链,首字延迟从 20s+ 降到 ~4s。
_META_PATTERNS = (
    "你是谁", "你叫什么", "你是什么", "介绍下你自己", "介绍一下你",
    "你好", "您好", "hi", "hello", "在吗", "谢谢", "再见", "早上好", "晚上好",
)
_META_MAX_LEN = 6  # 极短输入(≤6 字符)也按 meta 处理
_FAST_SYSTEM = "你是个人知识助手。用中文简短友好地回答用户的问候或简单问题,不超过两句话。"

# 子 graph 运行超时(秒)——防止无结果问题让 Agent 无限重试
SINGLE_TIMEOUT = 180  # knowledge_agent(可被 asyncio.wait_for 真正取消)
MULTI_TIMEOUT = 240  # multi_agent / supervisor_agent(sync,超时后线程残留但 router 快速返回)
TIMEOUT_MESSAGE = "⚠️ 处理超时(超过{timeout}s),未能完成检索。请稍后重试或换个问法。"

_router_model = _init_model("ROUTER_MODEL_KEY", GUARDRAILS_MODEL)

# 规则通道关键词(简单/统计/复杂 三档)
_SINGLE_HINTS = ("是什么", "什么是", "怎么", "如何", "为什么", "解释", "介绍", "含义")
_WORKFLOW_HINTS = (
    "统计", "数量", "多少个", "几个", "有多少", "列表", "列出", "列举", "清单",
    "聚合", "计数", "汇总", "标签为", "带 #", "都包含哪些",
)
_SUPERVISOR_HINTS = (
    "梳理", "规划", "对比", "总结", "综述", "复盘", "学习重点", "半年",
    "年度", "趋势", "分析", "规划一下", "体系", "方法论",
)

_CLASSIFY_SYSTEM = (
    "你是任务复杂度分类器。判断用户问题最适合哪种编排模式:\n"
    '- "single": 简单事实问答,一步检索即可回答\n'
    '- "workflow": 固定、可枚举的分析/统计任务\n'
    '- "supervisor": 复杂、模糊、需要拆解为多个子任务的任务\n'
    "只输出 JSON: {\"mode\": \"single\"|\"workflow\"|\"supervisor\"}\n"
)


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------


class RouterState(TypedDict, total=False):
    """State for the entry router graph."""

    query: str  # 原始用户问题
    mode: str  # single | workflow | supervisor
    answer: str  # 子 graph 的输出
    messages: Annotated[list, add_messages]  # 最终回答(AIMessage)


# ---------------------------------------------------------------------------
# Classifier (rule channel + LLM channel)
# ---------------------------------------------------------------------------


def _classify_by_rules(query: str) -> Optional[str]:
    """Rule channel: map keyword hints to a mode; None when no signal.

    Priority: supervisor > workflow > single. Returns None if the query
    carries no keyword signal at all (defer to the LLM channel).
    """
    if any(h in query for h in _SUPERVISOR_HINTS):
        return "supervisor"
    if any(h in query for h in _WORKFLOW_HINTS):
        return "workflow"
    if any(h in query for h in _SINGLE_HINTS):
        return "single"
    return None


def _is_meta(query: str) -> bool:
    """True for greetings/self-intro/ultra-short inputs that need no retrieval."""
    q = query.strip()
    if not q:
        return False
    lowered = q.lower()
    if any(p in lowered for p in _META_PATTERNS):
        return True
    # 极短输入(如 "?", "hi", "你是谁" 未命中 pattern 时的兜底)
    return len(q) <= _META_MAX_LEN


def _extract_query(state: RouterState) -> str:
    """Extract the user query from state.

    The frontend only sends ``messages`` (no top-level ``query`` field), so
    ``state.get("query", "")`` is always empty. Fall back to the last user
    message's text content. This is the single source of truth for the query
    in every node below.
    """
    q = (state.get("query") or "").strip()
    if q:
        return q
    for m in reversed(state.get("messages") or []):
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(content, str):
            s = content.strip()
            if s:
                return s
        elif isinstance(content, list):
            # multimodal: [{"type":"text","text":...}, ...]
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = (block.get("text") or "").strip()
                    if t:
                        return t
    return ""


def _classify_by_llm(query: str) -> Optional[str]:
    """LLM channel: judge mode for queries without strong keyword hints."""
    data = _llm_json(_router_model, _CLASSIFY_SYSTEM, f"用户问题: {query}")
    mode = data.get("mode") if data else None
    return mode if mode in MODES else None


def _classify(query: str) -> str:
    """Two-channel classify: rules first, LLM only when rules give no signal."""
    rule = _classify_by_rules(query)
    if rule is not None:
        # 规则明确命中(含 simple),直接采纳 —— 零 LLM 成本,简单任务最快
        return rule
    # 无关键词信号(如"A 和 B 有什么区别"):让 LLM 复查一次
    llm_mode = _classify_by_llm(query)
    if llm_mode:
        logger.info("Router LLM classified mode=%s", llm_mode)
        return llm_mode
    return DEFAULT_MODE


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def classify_node(state: RouterState) -> dict:
    """Classify the query and decide the orchestration mode."""
    query = _extract_query(state)
    # Meta/trivial 问题直接走 fast path,省掉 LLM 分类这一跳
    if _is_meta(query):
        logger.info("Router fast path (meta question): %r", query)
        return {"mode": "fast"}
    mode = _classify(query)
    logger.info("Router classified mode=%s", mode)
    return {"mode": mode}


async def run_fast_node(state: RouterState) -> dict:
    """Fast path: single lightweight model call, no retrieval / guardrails."""
    query = _extract_query(state)
    try:
        resp = await asyncio.to_thread(
            _router_model.invoke,
            [SystemMessage(content=_FAST_SYSTEM), HumanMessage(content=query)],
        )
        answer = str(resp.content or "")
    except Exception as exc:  # noqa: BLE001 - 降级到空回答,避免卡死
        logger.warning("fast node failed: %s", exc)
        answer = ""
    return {"answer": answer or "未生成回答"}


async def _invoke_workflow(query: str) -> str:
    """Run multi_agent (with Critic rewrite loop) off the event loop."""
    from src.agent.multi_agent_graph import multi_agent_graph

    tid = f"router-workflow-{uuid.uuid4()}"

    def _run() -> dict:
        return multi_agent_graph.invoke(
            {"query": query},
            config={"configurable": {"thread_id": tid}},
        )

    final = await asyncio.to_thread(_run)
    return str(final["messages"][-1].content) if final.get("messages") else ""


async def run_single_node(state: RouterState) -> dict:
    """Route to knowledge_agent (must be awaited: async guardrails hook).

    Timeout-guarded: a question with no hits in the vault used to make the
    agent retry forever, ballooning latency and token usage. On timeout we
    degrade to multi_agent, whose Critic loop terminates honestly.
    """
    from src.agent.knowledge_graph import docs_agent

    query = _extract_query(state)

    # 每次查询用独立 thread:固定 thread_id 在 dev server 重启后失效,
    # 会残留历史/触发"摘要失败"注入,导致答非所问
    tid = f"router-single-{uuid.uuid4()}"

    try:
        result = await asyncio.wait_for(
            docs_agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"configurable": {"thread_id": tid}},
            ),
            timeout=SINGLE_TIMEOUT,
        )
        msgs = result.get("messages") or []
        answer = str(msgs[-1].content) if msgs else ""
    except asyncio.TimeoutError:
        logger.warning(
            "knowledge_agent timed out after %ss; degrading to multi_agent",
            SINGLE_TIMEOUT,
        )
        degraded = await _invoke_workflow(query)
        answer = f"⚠️ 单 Agent 处理超时({SINGLE_TIMEOUT}s),已自动降级多 Agent 重试。\n\n{degraded}"

    return {"answer": answer or "未生成回答"}


async def run_workflow_node(state: RouterState) -> dict:
    """Route to multi_agent (sync graph, run off the event loop)."""
    query = _extract_query(state)

    try:
        answer = await asyncio.wait_for(
            _invoke_workflow(query), timeout=MULTI_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("multi_agent timed out after %ss", MULTI_TIMEOUT)
        answer = TIMEOUT_MESSAGE.format(timeout=MULTI_TIMEOUT)

    return {"answer": answer or "未生成回答"}


async def run_supervisor_node(state: RouterState) -> dict:
    """Route to supervisor_agent (sync graph, run off the event loop)."""
    from src.agent.supervisor_graph import supervisor_graph

    query = _extract_query(state)

    # 每次查询用独立 thread(同上,避免历史残留)
    tid = f"router-supervisor-{uuid.uuid4()}"

    def _invoke() -> dict:
        return supervisor_graph.invoke(
            {"query": query},
            config={"configurable": {"thread_id": tid}},
        )

    try:
        final = await asyncio.wait_for(
            asyncio.to_thread(_invoke), timeout=MULTI_TIMEOUT,
        )
        answer = final.get("final") or ""
    except asyncio.TimeoutError:
        logger.warning("supervisor_agent timed out after %ss", MULTI_TIMEOUT)
        answer = TIMEOUT_MESSAGE.format(timeout=MULTI_TIMEOUT)

    return {"answer": answer or "未生成回答"}


def answer_node(state: RouterState) -> dict:
    """Emit the final answer tagged with the orchestration mode used."""
    mode = state.get("mode", DEFAULT_MODE)
    answer = state.get("answer", "") or "未生成回答"
    body = f"[编排模式: {mode}]\n\n{answer}"
    return {"messages": [AIMessage(content=body)]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route(state: RouterState) -> str:
    mode = state.get("mode", DEFAULT_MODE)
    if mode == "fast":
        return "fast"
    return mode if mode in MODES else DEFAULT_MODE


# ---------------------------------------------------------------------------
# Graph assembly (no custom checkpointer — platform manages persistence)
# ---------------------------------------------------------------------------

_builder = StateGraph(RouterState)
_builder.add_node("classify", classify_node)
_builder.add_node("run_fast", run_fast_node)
_builder.add_node("run_single", run_single_node)
_builder.add_node("run_workflow", run_workflow_node)
_builder.add_node("run_supervisor", run_supervisor_node)
_builder.add_node("answer", answer_node)

_builder.add_edge(START, "classify")
_builder.add_conditional_edges(
    "classify",
    _route,
    {
        "fast": "run_fast",
        "single": "run_single",
        "workflow": "run_workflow",
        "supervisor": "run_supervisor",
    },
)
_builder.add_edge("run_fast", "answer")
_builder.add_edge("run_single", "answer")
_builder.add_edge("run_workflow", "answer")
_builder.add_edge("run_supervisor", "answer")
_builder.add_edge("answer", END)

router_graph = _builder.compile()

logger.info(
    "Router graph compiled: classify -> fast|single|workflow|supervisor -> answer"
)

__all__ = [
    "RouterState",
    "router_graph",
    "MODES",
    "DEFAULT_MODE",
    "SINGLE_TIMEOUT",
    "MULTI_TIMEOUT",
    # classifier (unit-testable)
    "_classify",
    "_classify_by_rules",
    "_classify_by_llm",
    "_is_meta",
    # nodes
    "classify_node",
    "run_fast_node",
    "run_single_node",
    "run_workflow_node",
    "run_supervisor_node",
    "answer_node",
    # helpers
    "_invoke_workflow",
    # routing
    "_route",
]

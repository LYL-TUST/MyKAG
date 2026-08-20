"""Multi-agent knowledge graph with a tool-call closed loop.

Upgrade from the single ``create_agent`` pipeline to an explicit
Planner -> Executor -> Summarizer -> Critic state graph:

- **Planner**  : decomposes the user query into sub-queries and picks a
  retrieval strategy (notes / codebase / both).
- **Executor** : runs the sub-queries in parallel through the existing
  vault + codebase tools.
- **Summarizer**: fuses multi-source results into a draft answer with
  source citations.
- **Critic**   : checks whether retrieval was sufficient (rule channel
  + LLM judge channel). If not, a query-rewrite node rewrites the
  sub-queries and the loop retries — bounded to MAX_LOOP_ATTEMPTS.

Every LLM call degrades gracefully: without API keys (or on model
failure) the graph still runs via fallbacks, so it is testable without
credentials and never deadlocks.

Expose in langgraph.json:

    "multi_agent": "./src/agent/multi_agent_graph.py:multi_agent_graph"
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, List, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.config import (
    DEFAULT_MODEL,
    GUARDRAILS_MODEL,
    MODELS,
    ModelConfig,
    _thinking_kwargs,
)
from src.tools.codebase_tools import search_codebase
from src.tools.vault_tools import search_vault

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LOOP_ATTEMPTS = 2  # 闭环最多重检索轮数(防死循环)
MAX_SUB_QUERIES = 3  # Planner 拆分子查询上限
DEFAULT_TOOL_TIMEOUT = 30  # 单个工具调用超时(秒)

# 子查询触发源码检索的关键词(Planner 未显式指定 codebase 时的启发式兜底)
_CODE_HINTS = (
    "源码", "代码", "实现", "函数", "接口", "模块",
    "retry", "middleware", "tool", "server", "ast",
)


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Shared state across the multi-agent graph nodes."""

    messages: Annotated[list, add_messages]  # 最终回答(AIMessage)
    query: str  # 原始用户问题
    sub_queries: List[str]  # Planner / Rewrite 输出的检索子查询
    codebase: Optional[str]  # None | "ellie" | "code-review"
    retrieval_results: dict  # {子查询: 工具输出文本}
    draft: str  # Summarizer 草稿
    satisfied: bool  # Critic 充分性检查结论
    attempts: int  # 已执行的闭环重试轮数


# ---------------------------------------------------------------------------
# Models (分层用模型:规划/评审/改写用小模型,综合用主模型)
# ---------------------------------------------------------------------------


def _init_model(key_env: str, default: ModelConfig):
    """Build a chat model from the registry, overridable by env var."""
    key = os.getenv(key_env, default.key)
    cfg = MODELS.get(key, default)
    return init_chat_model(
        model=cfg.id,
        model_provider="openai",
        temperature=0,
        **_thinking_kwargs(),
    )


_planner_model = _init_model("PLANNER_MODEL_KEY", GUARDRAILS_MODEL)
_critic_model = _init_model("CRITIC_MODEL_KEY", GUARDRAILS_MODEL)
_rewrite_model = _init_model("REWRITE_MODEL_KEY", GUARDRAILS_MODEL)
_summarizer_model = _init_model("SUMMARIZER_MODEL_KEY", DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# LLM helpers (all degrade to fallbacks on failure)
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _llm_json(model, system: str, user: str) -> Optional[dict]:
    """Ask the model for a JSON object; return parsed dict or None."""
    try:
        resp = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        text = _strip_code_fences(str(resp.content or ""))
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001 - LLM/provider 任意失败都降级
        logger.warning("LLM JSON call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Node 1: Planner — 查询分析
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = (
    "你是个人知识库的查询规划器。把用户问题拆解成适合检索的子查询。"
    "只输出 JSON(不要任何其他文字),格式:\n"
    '{"sub_queries": ["子查询1", "子查询2"], "codebase": null | "ellie" | "code-review"}\n'
    "规则:\n"
    "- sub_queries: 1-3 条互不重复的检索词/短句,中文保留原文,技术术语可补充英文\n"
    "- codebase: 问题涉及具体代码实现、源码对比、某模块怎么写时填项目名,否则 null\n"
    "- 例子: 用户问 'ellie 的重试逻辑和 Code Review 有什么区别' -> "
    '{"sub_queries": ["ellie 重试 踩坑", "Code Review 重试 反馈", "retry middleware"], "codebase": "ellie"}'
)


def _plan_sub_queries(query: str) -> tuple[List[str], Optional[str]]:
    """Return (sub_queries, codebase). Falls back to the raw query."""
    data = _llm_json(_planner_model, _PLANNER_SYSTEM, f"用户问题: {query}")
    if not data:
        return [query], None

    subs = [s for s in (data.get("sub_queries") or [])
            if isinstance(s, str) and s.strip()]
    if not subs:
        subs = [query]
    else:
        subs = subs[:MAX_SUB_QUERIES]

    codebase = data.get("codebase")
    if codebase not in ("ellie", "code-review"):
        codebase = "ellie" if any(h in query for h in _CODE_HINTS) else None
    return subs, codebase


def analyze_node(state: AgentState) -> dict:
    """Planner node: decompose query into sub-queries."""
    query = state.get("query") or ""
    sub_queries, codebase = _plan_sub_queries(query)
    logger.info(
        "Planner: %d sub-query(ies), codebase=%s", len(sub_queries), codebase
    )
    return {"sub_queries": sub_queries, "codebase": codebase}


# ---------------------------------------------------------------------------
# Node 2: Executor — 并行检索(复用现有工具)
# ---------------------------------------------------------------------------


def _run_one_subquery(sub_query: str, codebase: Optional[str]) -> str:
    """Run vault (+ optional codebase) tools for a single sub-query."""
    parts: List[str] = []
    try:
        parts.append(search_vault.invoke(
            {"query": sub_query, "top_k": 5, "expand_wikilinks": True}
        ))
    except Exception as exc:  # noqa: BLE001
        parts.append(f"[search_vault error: {exc}]")
    if codebase:
        try:
            parts.append(search_codebase.invoke(
                {"query": sub_query, "project": codebase, "top_k": 5}
            ))
        except Exception as exc:  # noqa: BLE001
            parts.append(f"[search_codebase error: {exc}]")
    return "\n\n".join(parts)


def retrieve_node(state: AgentState) -> dict:
    """Executor node: run sub-queries in parallel and collect results."""
    subs = state.get("sub_queries") or [state.get("query") or ""]
    codebase = state.get("codebase")
    results: dict = {}

    with ThreadPoolExecutor(max_workers=min(len(subs), 4)) as pool:
        futures = {pool.submit(_run_one_subquery, q, codebase): q for q in subs}
        for fut in futures:
            q = futures[fut]
            try:
                results[q] = fut.result(timeout=DEFAULT_TOOL_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                results[q] = f"[tool timeout/error: {exc}]"

    logger.info("Executor: %d sub-query result(s) collected", len(results))
    return {"retrieval_results": results}


# ---------------------------------------------------------------------------
# Node 3: Summarizer — 综合回答
# ---------------------------------------------------------------------------

_SUMMARIZER_SYSTEM = (
    "你是个人知识库的综合回答者。融合多个检索结果,用中文给出简洁、准确、"
    "有技术深度的回答。必须标注信息来源(笔记名/文件路径)。\n"
    "如果检索结果不足以回答(例如用户问通用技术概念、笔记库未命中),"
    "不要直接拒绝:用你自身的通用知识补全回答,但必须明确说明"
    "\"以下基于通用知识,笔记库中未检索到相关笔记\","
    "且不要编造笔记名或文件路径。"
)


def _fallback_draft(query: str, results: dict) -> str:
    """No-LLM fallback: concatenate retrieval output so the loop still runs."""
    lines = [f"问题: {query}", ""]
    for sub_q, text in results.items():
        lines.append(f"### 检索 [{sub_q}]")
        lines.append((text or "")[:2000])
    return "\n".join(lines)


def _summarize(query: str, results: dict) -> str:
    """Fuse retrieval results into a draft answer (with source citations)."""
    block = "\n\n".join(f"[查询: {q}]\n{r}" for q, r in results.items())
    user = f"用户问题: {query}\n\n检索结果:\n{block[:12000]}"
    try:
        resp = _summarizer_model.invoke(
            [SystemMessage(content=_SUMMARIZER_SYSTEM),
             HumanMessage(content=user)]
        )
        text = str(resp.content or "").strip()
        return text or _fallback_draft(query, results)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Summarizer failed: %s", exc)
        return _fallback_draft(query, results)


def synthesize_node(state: AgentState) -> dict:
    """Summarizer node: produce a draft answer from all results."""
    draft = _summarize(state.get("query", ""), state.get("retrieval_results", {}))
    return {"draft": draft}


# ---------------------------------------------------------------------------
# Node 4: Critic — 充分性检查(工具闭环的开关)
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = (
    "你是检索充分性评审。根据检索结果判断:能否充分回答用户问题? "
    "只输出 JSON:\n"
    '{"sufficient": true|false, "gap": "缺少什么,一句话,可为空"}'
)


def _results_have_content(results: dict) -> bool:
    """Rule channel: does any result carry real content?"""
    if not results:
        return False
    bad_markers = (
        "not initialized", "not found", "No relevant notes",
        "No matching", "error", "没有找到", "未找到", "Vault is empty",
    )
    for text in results.values():
        t = (text or "").strip()
        if len(t) < 40:
            continue
        if any(m in t for m in bad_markers):
            continue
        return True
    return False


def _critic_judge(query: str, results: dict) -> bool:
    """Return True when retrieval is sufficient (LLM judge only if needed)."""
    # 规则通道:没有实质内容 -> 直接判不充分(不浪费一次 LLM 调用)
    if not _results_have_content(results):
        logger.info("Critic rule channel: no usable content")
        return False

    block = "\n\n".join(f"[{q}]\n{r[:1500]}" for q, r in results.items())
    data = _llm_json(
        _critic_model, _CRITIC_SYSTEM,
        f"用户问题: {query}\n\n检索结果:\n{block[:8000]}",
    )
    if data is None:
        # LLM 不可用时不阻塞流程:默认充分
        return True
    return bool(data.get("sufficient", True))


def critique_node(state: AgentState) -> dict:
    """Critic node: decide whether to loop back or finish."""
    satisfied = _critic_judge(
        state.get("query", ""), state.get("retrieval_results", {})
    )
    logger.info(
        "Critic: sufficient=%s (attempts=%d)",
        satisfied, state.get("attempts", 0),
    )
    return {"satisfied": satisfied}


# ---------------------------------------------------------------------------
# Node 5: Rewrite — 闭环重检索
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = (
    "你是查询改写器。基于已有检索结果和缺口,把子查询改写得更容易命中。"
    "只输出 JSON:\n"
    '{"sub_queries": ["改写后的子查询1", "..."]}\n'
    "规则:保留原有意图,补充同义词/英文术语/更具体的限定词,1-3 条。"
)


def _rewrite_queries(query: str, results: dict, old_subs: List[str]) -> List[str]:
    """Rewrite sub-queries for another retrieval round."""
    block = "\n\n".join(f"[{q}]\n{r[:800]}" for q, r in results.items())
    user = (
        f"原问题: {query}\n"
        f"原子查询: {old_subs}\n"
        f"现有检索结果:\n{block[:6000]}"
    )
    data = _llm_json(_rewrite_model, _REWRITE_SYSTEM, user)
    if data:
        subs = [s for s in (data.get("sub_queries") or [])
                if isinstance(s, str) and s.strip()]
        if subs:
            return subs[:MAX_SUB_QUERIES]
    # 降级:保持原查询(避免退化),靠 attempts 上限兜底
    return old_subs or [query]


def rewrite_node(state: AgentState) -> dict:
    """Rewrite node: bump attempts and rewrite sub-queries."""
    new_subs = _rewrite_queries(
        state.get("query", ""),
        state.get("retrieval_results", {}),
        state.get("sub_queries", []),
    )
    return {
        "sub_queries": new_subs,
        "attempts": state.get("attempts", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node 6: Answer — 最终回答
# ---------------------------------------------------------------------------


def _extract_sources(results: dict) -> List[str]:
    """Best-effort source extraction from tool outputs (notes/paths)."""
    sources: List[str] = []
    for text in results.values():
        sources += re.findall(r"\[\[([^\]]+)\]\]", text or "")
        sources += [m.strip() for m in re.findall(r"Note: ([^\n]+)", text or "")]
        sources += re.findall(r"[A-Za-z]:/[^\s\)\]'\"`]+\.(?:py|ts|tsx|js)", text or "")
    seen, out = set(), []
    for s in sources:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:10]


def answer_node(state: AgentState) -> dict:
    """Answer node: emit final draft with source attribution."""
    draft = state.get("draft") or "未能生成回答(检索未命中)。"
    sources = _extract_sources(state.get("retrieval_results", {}))
    footer = (
        f"\n\n---\n**信息来源**: {', '.join(sources)}"
        if sources else "\n\n---\n**信息来源**: 无(检索未命中)"
    )
    return {"messages": [AIMessage(content=draft + footer)]}


# ---------------------------------------------------------------------------
# Routing (closed loop)
# ---------------------------------------------------------------------------


def _route_after_critique(state: AgentState) -> str:
    """Loop back to rewrite when unsatisfied and under the attempt cap."""
    if not state.get("satisfied") and state.get("attempts", 0) < MAX_LOOP_ATTEMPTS:
        return "rewrite"
    return "answer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
# NOTE: no custom checkpointer here. LangGraph dev / platform manages
# persistence itself and rejects user-supplied checkpointers, which would
# fail graph loading at `langgraph dev` startup. Stateless compile is fine:
# the platform injects its own persistence at runtime.

_builder = StateGraph(AgentState)
_builder.add_node("analyze", analyze_node)
_builder.add_node("retrieve", retrieve_node)
_builder.add_node("synthesize", synthesize_node)
_builder.add_node("critique", critique_node)
_builder.add_node("rewrite", rewrite_node)
_builder.add_node("answer", answer_node)

_builder.add_edge(START, "analyze")
_builder.add_edge("analyze", "retrieve")
_builder.add_edge("retrieve", "synthesize")
_builder.add_edge("synthesize", "critique")
_builder.add_conditional_edges(
    "critique",
    _route_after_critique,
    {"rewrite": "rewrite", "answer": "answer"},
)
_builder.add_edge("rewrite", "retrieve")
_builder.add_edge("answer", END)

multi_agent_graph = _builder.compile()

logger.info(
    "Multi-agent graph compiled: analyze -> retrieve -> synthesize -> "
    "critique -> (rewrite -> retrieve)*<=%d -> answer",
    MAX_LOOP_ATTEMPTS,
)

__all__ = [
    "AgentState",
    "multi_agent_graph",
    "MAX_LOOP_ATTEMPTS",
    # nodes
    "analyze_node",
    "retrieve_node",
    "synthesize_node",
    "critique_node",
    "rewrite_node",
    "answer_node",
    # routing
    "_route_after_critique",
    # helpers (unit-testable)
    "_plan_sub_queries",
    "_critic_judge",
    "_rewrite_queries",
    "_results_have_content",
]

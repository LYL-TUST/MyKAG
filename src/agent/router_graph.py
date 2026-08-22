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
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.config import GUARDRAILS_MODEL
from src.agent.multi_agent_graph import _init_model, _llm_json
from src.agent.semantic_cache import SemanticAnswerCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES = ("single", "workflow", "supervisor")
DEFAULT_MODE = "single"

# Meta / trivial 问题(问候、自我介绍、极短输入)——直接走轻模型 fast path,
# 跳过 router LLM 分类 + guardrails + RAG + 主模型思考链,首字延迟从 20s+ 降到 ~4s。
# 只收录"寒暄/关于助手自身"的确定性元问题;事实查询(如"ellie 支持哪些模型")
# 不能走 fast(不检索会答错),留给规则/LLM 通道路由到检索模式。
_META_PATTERNS = (
    "你是谁", "你叫什么", "你是什么", "介绍下你自己", "介绍一下你",
    "你好", "您好", "hi", "hello", "在吗", "谢谢", "再见", "早上好", "晚上好",
    # fast path 扩面:助手元问题/寒暄(2026-08-21 优化)
    "你能做什么", "你会什么", "你有什么功能", "你能帮我做什么", "你怎么用",
    "你是用什么做的", "谁做的你", "谁开发你", "你厉害吗", "你真棒", "你好棒",
    "好的", "好", "嗯", "行", "哦", "明白", "知道了", "不错", "哈哈", "可以的",
)
_META_MAX_LEN = 3  # 极短输入(≤3 字符,如 "?")按 meta 处理;再长留给规则/LLM 分类,
# 避免 "统计笔记数量" 这类 4-6 字实质问题被误判为寒暄(2026-08-21 收紧)

# ---------------------------------------------------------------------------
# Fast path 扩面 2:通用技术常识问答(2026-08-21)
# ---------------------------------------------------------------------------
# "JS 有哪些数据类型" / "html5 有哪些特性" / "script 标签的 defer" 这类短技术
# 常识问题,答案不依赖个人笔记库(vault 未命中时反而浪费 2-4 轮工具循环检索)。
# 命中后走 fast_tech 模式:轻模型单次调用直接回答(~9-13s),跳过 RAG + guardrails
# + 多轮工具循环(原来 25-65s)。
#
# 判定规则(三条件同时满足):
#   1. 命中 _TECH_FACT_KEYWORDS 中任一具体概念词(必须是"具体技术词",不包含
#      "有哪些/是什么/区别"这类通用触发器——它们会误伤 "向量检索 vs TF-IDF"
#      这类 vault 笔记查询);
#   2. 查询 ≤ _TECH_FACT_MAX_LEN 字符(长问题留给完整编排);
#   3. 不命中 _VAULT_OWNED_WORDS(涉及个人笔记主题的查询必须走 RAG)。
_TECH_FACT_KEYWORDS = (
    # 截图用例(必覆盖)
    "数据类型", "html5", "html", "css", "defer", "async", "script",
    # 前端 / JS
    "javascript", "闭包", "原型链", "原型", "promise", "异步", "回调",
    "事件循环", "宏任务", "微任务", "作用域", "提升", "箭头函数", "深拷贝",
    "浅拷贝", "防抖", "节流", "设计模式", "单例", "观察者模式", "发布订阅",
    "继承", "多态", "封装", "抽象", "泛型", "装饰器", "生成器", "迭代器",
    "递归", "尾调用", "柯里化", "纯函数", "内存泄漏", "垃圾回收", "this",
    "bind", "apply", "call", "vue", "react", "node", "npm", "webpack",
    "vite", "组件", "props", "state", "flex", "grid", "媒体查询", "响应式",
    "事件冒泡", "事件委托", "选择器", "伪类", "伪元素", "box-sizing",
    "z-index", "localstorage", "sessionstorage", "dom", "bom",
    # 网络 / 协议
    "http", "https", "tcp", "udp", "dns", "cdn", "websocket", "rest",
    "graphql", "cookie", "session", "token", "jwt", "cors", "xss", "csrf",
    "json", "xml", "状态码",
    # 数据 / 算法
    "链表", "哈希", "栈", "队列", "二分查找", "动态规划", "快速排序",
    "归并排序", "冒泡排序", "时间复杂度", "空间复杂度", "大 o", "big o",
    # 后端 / 基础设施
    "sql", "mysql", "redis", "索引", "事务", "线程", "进程", "并发",
    "goroutine", "docker", "k8s", "kubernetes", "linux", "git", "正则",
    "表达式", "命令",
)
_TECH_FACT_MAX_LEN = 24
# 涉及个人笔记主题的查询词——命中则强制走 RAG,绝不吃 fast 通用回答
_VAULT_OWNED_WORDS = (
    "ellie", "mcp", "vault", "笔记", "知识库", "我的", "踩坑", "记录",
    "code review", "codereview", "rag", "langchain", "langgraph", "qdrant",
    "llamaindex", "tf-idf", "tfidf", "向量检索", "反馈飞轮", "调用图",
    "harness", "岗位", "简历", "面试", "agent loop", "设计对比",
)

_FAST_SYSTEM = "你是个人知识助手。用中文简短友好地回答用户的问候或简单问题,不超过两句话。"
_FAST_TECH_SYSTEM = (
    "你是资深软件工程师,用中文直接、准确地回答用户的编程/技术常识问题。\n"
    "要求:\n"
    "- 简明扼要,用要点列出关键信息,不要寒暄\n"
    "- 涉及代码时给出 1-3 行核心示例\n"
    "- 控制在 150-300 字,不引用来源,不编造不存在的事实"
)

# 子 graph 运行超时(秒)——防止无结果问题让 Agent 无限重试
SINGLE_TIMEOUT = 180  # knowledge_agent(可被 asyncio.wait_for 真正取消)
MULTI_TIMEOUT = 240  # multi_agent / supervisor_agent(sync,超时后线程残留但 router 快速返回)
TIMEOUT_MESSAGE = "⚠️ 处理超时(超过{timeout}s),未能完成检索。请稍后重试或换个问法。"

_router_model = _init_model("ROUTER_MODEL_KEY", GUARDRAILS_MODEL)

# 语义缓存:重复/相似问题直接返回历史答案(一次 embedding ~0.5s,免去 30-95s
# 完整编排)。阈值/容量/TTL 可用 env 覆盖,命中时走 "answer" 短路。
_sem_cache = SemanticAnswerCache(
    capacity=int(os.environ.get("SEM_CACHE_CAPACITY", "256")),
    threshold=float(os.environ.get("SEM_CACHE_THRESHOLD", "0.92")),
    ttl_seconds=int(os.environ.get("SEM_CACHE_TTL", "86400")),
)

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
    mode: str  # single | workflow | supervisor | fast | fast_tech
    answer: str  # 子 graph 的输出
    cache_hit: bool  # 语义缓存命中(直接短路到 answer)
    prefetch: Optional[str]  # RAG 并行预取结果(供 run_single 注入,省一次工具往返)
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


def _is_tech_fact(query: str) -> bool:
    """True for short general tech-fact questions answered without retrieval.

    Catches "JS 有哪些数据类型" / "html5 有哪些特性" / "script 标签的 defer" —
    answers don't depend on the personal vault, so a full tool-loop RAG run
    only wastes 25-65s. Conservative by design: the trigger list is concrete
    tech terms only (no generic "是什么/有哪些" patterns, which would wrongly
    route vault-topic questions like "向量检索 vs TF-IDF 区别" to the fast path).
    """
    q = query.strip()
    if not q or len(q) > _TECH_FACT_MAX_LEN:
        return False
    lowered = q.lower()
    if any(w in lowered for w in _VAULT_OWNED_WORDS):
        return False
    return any(k in lowered for k in _TECH_FACT_KEYWORDS)


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
# RAG 并行预取(2026-08-22,Tier 1.1)
# ---------------------------------------------------------------------------
# 无规则信号的查询要过 LLM 分类(2-9s)。这段时间 RAG 检索(embedding 0.5s +
# 检索 0.5s)可以免费并行完成:分类返回时预取结果已就绪,注入 run_single 的
# knowledge_agent 后省掉它的首次 search_vault 工具往返(再省 2-5s)。
# 预取失败静默返回 None,不影响任何路径。
_prefetch_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="router-prefetch",
)


def _prefetch_retrieval(query: str) -> Optional[str]:
    """Best-effort RAG prefetch: formatted top-5 note chunks or None."""
    try:
        from src.tools.vault_tools import _init_vault, _vault_retriever

        _init_vault()
        if _vault_retriever is None:
            return None
        results = _vault_retriever.search(
            query=query,
            top_k=5,
            expand_wikilinks=False,  # 预取只求快,不做 wikilink 扩展
            use_reranker=False,  # 跳过 cross-encoder(1-3s),预取足够
        )
        if not results:
            return None
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title") or r.get("note_name") or "unknown"
            text = (r.get("text") or "")[:600]
            parts.append(f"### {i}. {title}\n{text}")
        return "\n\n".join(parts)
    except Exception as exc:  # noqa: BLE001 - 预取是 best-effort
        logger.debug("prefetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def classify_node(state: RouterState) -> dict:
    """Classify the query and decide the orchestration mode.

    When the rules give no signal (LLM classification needed, 2-9s), a RAG
    prefetch runs in parallel on a worker thread; its result is stored in
    ``prefetch`` and consumed by run_single_node.
    """
    query = _extract_query(state)
    # Meta/trivial 问题直接走 fast path,省掉 LLM 分类这一跳
    if _is_meta(query):
        logger.info("Router fast path (meta question): %r", query)
        return {"mode": "fast"}
    # 通用技术常识问题(短、不依赖笔记库)走 fast_tech:轻模型单次调用
    if _is_tech_fact(query):
        logger.info("Router fast path (tech fact): %r", query)
        return {"mode": "fast_tech"}
    rule = _classify_by_rules(query)
    if rule is not None:
        logger.info("Router classified by rules mode=%s", rule)
        return {"mode": rule}
    # 无规则信号 -> LLM 分类,期间并行 RAG 预取(LLM 慢时检索免费完成)
    prefetch_future = _prefetch_executor.submit(_prefetch_retrieval, query)
    llm_mode = _classify_by_llm(query)
    mode = llm_mode if llm_mode in MODES else DEFAULT_MODE
    try:
        prefetch = prefetch_future.result(timeout=8)
    except Exception:  # noqa: BLE001 - 预取结果缺失不影响分类
        prefetch = None
    logger.info(
        "Router LLM classified mode=%s (prefetch=%s)",
        mode, "ok" if prefetch else "none",
    )
    return {"mode": mode, "prefetch": prefetch}


def check_cache_node(state: RouterState) -> dict:
    """Semantic-cache lookup: a hit short-circuits the whole orchestration.

    Runs BEFORE classification so repeated questions skip every LLM call
    (one embedding request ~0.5s instead of 30-95s of graph work).
    """
    query = _extract_query(state)
    if not query:
        return {"cache_hit": False}
    hit = _sem_cache.get(query)
    if hit:
        return {
            "answer": hit["answer"],
            "mode": hit["mode"],
            "cache_hit": True,
        }
    return {"cache_hit": False}


async def run_fast_node(state: RouterState) -> dict:
    """Fast path: single lightweight model call, no retrieval / guardrails.

    Two flavours selected by mode:
    - ``fast``      : greetings / self-intro (short friendly reply)
    - ``fast_tech`` : short general tech-fact questions (concise technical reply)
    """
    query = _extract_query(state)
    system = (
        _FAST_TECH_SYSTEM if state.get("mode") == "fast_tech" else _FAST_SYSTEM
    )
    try:
        resp = await asyncio.to_thread(
            _router_model.invoke,
            [SystemMessage(content=system), HumanMessage(content=query)],
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

    # 并行预取结果注入:classify 期间已检索到笔记,前置给 agent 作参考,
    # 省掉它的首次 search_vault 工具往返(2-5s)。预取为空时行为不变。
    messages: list = [{"role": "user", "content": query}]
    prefetch = state.get("prefetch")
    if prefetch:
        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "[预检索结果(并行预取,优先参考)]\n"
                    f"{prefetch[:8000]}\n\n"
                    "请优先基于以上笔记内容回答用户问题;"
                    "若内容不足以回答,可再调用工具检索补充。"
                ),
            },
        )

    try:
        result = await asyncio.wait_for(
            docs_agent.ainvoke(
                {"messages": messages},
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
    """Emit the final answer tagged with the orchestration mode used.

    Non-cache-hits are written back into the semantic cache so the next
    (near-)identical question is answered instantly.
    """
    mode = state.get("mode", DEFAULT_MODE)
    answer = state.get("answer", "") or "未生成回答"
    cache_hit = state.get("cache_hit", False)
    if not cache_hit:
        q = _extract_query(state)
        if q and answer and answer != "未生成回答":
            _sem_cache.put(q, answer, mode)
    tag = "[缓存命中]" if cache_hit else f"[编排模式: {mode}]"
    body = f"{tag}\n\n{answer}"
    return {"messages": [AIMessage(content=body)]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route(state: RouterState) -> str:
    mode = state.get("mode", DEFAULT_MODE)
    if mode in ("fast", "fast_tech"):
        return mode
    return mode if mode in MODES else DEFAULT_MODE


# ---------------------------------------------------------------------------
# Graph assembly (no custom checkpointer — platform manages persistence)
# ---------------------------------------------------------------------------

_builder = StateGraph(RouterState)
_builder.add_node("check_cache", check_cache_node)
_builder.add_node("classify", classify_node)
_builder.add_node("run_fast", run_fast_node)
_builder.add_node("run_single", run_single_node)
_builder.add_node("run_workflow", run_workflow_node)
_builder.add_node("run_supervisor", run_supervisor_node)
_builder.add_node("answer", answer_node)

_builder.add_edge(START, "check_cache")
_builder.add_conditional_edges(
    "check_cache",
    lambda s: "answer" if s.get("cache_hit") else "classify",
    {"answer": "answer", "classify": "classify"},
)
_builder.add_conditional_edges(
    "classify",
    _route,
    {
        "fast": "run_fast",
        "fast_tech": "run_fast",
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
    "Router graph compiled: check_cache -> fast|single|workflow|supervisor -> answer"
)

# ---------------------------------------------------------------------------
# SiliconFlow 模型实例预热(2026-08-22)
# ---------------------------------------------------------------------------
# SiliconFlow 对模型首次请求要调度 GPU 实例(冷启动 70s+,实测 74.5s->9.4s)。
# dev server 启动编译要 1-5 分钟,后台预热把冷启动藏进启动时间,用户首问不撞冷实例。
# 失败静默降级(无 key/网络错误都不影响启动)。
try:
    from src.agent.warmup import start_warmup_background
    start_warmup_background()
except Exception:  # noqa: BLE001 - 预热是 best-effort
    logger.debug("model warm-up not started", exc_info=True)

__all__ = [
    "RouterState",
    "router_graph",
    "MODES",
    "DEFAULT_MODE",
    "SINGLE_TIMEOUT",
    "MULTI_TIMEOUT",
    "SemanticAnswerCache",
    "_sem_cache",
    # classifier (unit-testable)
    "_classify",
    "_classify_by_rules",
    "_classify_by_llm",
    "_is_meta",
    "_is_tech_fact",
    # nodes
    "check_cache_node",
    "classify_node",
    "run_fast_node",
    "run_single_node",
    "run_workflow_node",
    "run_supervisor_node",
    "answer_node",
    # helpers
    "_invoke_workflow",
    "_prefetch_retrieval",
    # routing
    "_route",
]

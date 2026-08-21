# Personal Knowledge Agent — LangGraph agent entry point.
# Reuses middleware from Health Knowledge LLM Agent, replaces tools and prompts.
import logging
import os

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from src.agent.config import (
    AGENT_MODEL,
    GUARDRAILS_MODEL,
    GUARDRAILS_MODEL_ID,
    SUMMARY_MODEL,
    SUMMARY_MODEL_ID,
    agent_model,
    configurable_model,
    model_fallback_middleware,
    model_retry_middleware,
    tool_retry_middleware,
)
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.guardrails_middleware import (
    guardrails_prompt_commit,
    guardrails_prompt_source,
)
from src.prompts.knowledge_agent import knowledge_agent_prompt as _local_prompt
from src.prompts.context_summary_prompt import context_summary_prompt
from src.tools.vault_tools import (
    get_note,
    get_note_graph,
    list_tags,
    search_by_tag,
    search_vault,
)
from src.tools.codebase_tools import (
    list_codebase_projects,
    search_codebase,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.info("Knowledge agent module loaded")

# Prompt loading strategy (same pattern as health agent)
_USE_LOCAL_PROMPTS = os.getenv("USE_LOCAL_PROMPTS", "").lower() in {
    "1",
    "true",
    "yes",
}

if _USE_LOCAL_PROMPTS:
    docs_agent_prompt = _local_prompt
    prompt_commit = None
    prompt_source = "local:src/prompts/knowledge_agent.py"
    logger.info("Using local knowledge agent prompt")
else:
    # Fall back to local prompt (no LangSmith hub setup for this project yet)
    docs_agent_prompt = _local_prompt
    prompt_commit = None
    prompt_source = "local:src/prompts/knowledge_agent.py"
    logger.info("Using local knowledge agent prompt")

# Guardrails middleware
guardrails_middleware = GuardrailsMiddleware(
    model=GUARDRAILS_MODEL_ID,
    block_off_topic=False,
)
logger.info(f"Guardrails middleware using {GUARDRAILS_MODEL.name}")

# Context summarization middleware
# NOTE: SummarizationMiddleware calls init_chat_model(model) WITHOUT a
# provider, so the id must carry a "provider:" prefix ("Qwen/Qwen3-8B" alone
# cannot be provider-inferred). Build it from SUMMARY_MODEL.provider so a
# SUMMARY_MODEL_KEY override to another provider still works.
context_summary_middleware = SummarizationMiddleware(
    model=f"{SUMMARY_MODEL.provider}:{SUMMARY_MODEL_ID}",
    trigger=("tokens", 130_000),
    keep=("tokens", 30_000),
    summary_prompt=context_summary_prompt,
    trim_tokens_to_summarize=None,
)
logger.info(
    "Context summarization enabled at 130k tokens, preserving latest 30k tokens"
)

# Knowledge agent tools
knowledge_agent_tools = [
    search_vault,
    search_by_tag,
    list_tags,
    get_note,
    get_note_graph,
    search_codebase,
    list_codebase_projects,
]

# Middleware stack
knowledge_agent_middleware = [
    guardrails_middleware,
    context_summary_middleware,
    tool_retry_middleware,
    model_retry_middleware,
]

if model_fallback_middleware is not None:
    knowledge_agent_middleware.append(model_fallback_middleware)

# Create the agent
# 工具循环走快模型(agent_model,默认 qwen3-8b ~2s/次;env AGENT_MODEL_KEY 可切回
# deepseek-v4)。工具循环模型只负责"决定调哪个工具/读哪篇笔记",快模型足够,
# 慢模型留给最终长回答(路由层)。若快模型初始化失败(无 key 等)回退 configurable_model。
docs_agent = create_agent(
    model=agent_model or configurable_model,
    tools=knowledge_agent_tools,
    system_prompt=docs_agent_prompt,
    middleware=knowledge_agent_middleware,
)

# 轮数上限:create_agent 没有 max_iterations 参数,用 recursion_limit 限制
# 图执行节点总数(每轮 ≈ model 节点 + tools 节点 2 个节点)。默认 4 轮
# (recursion_limit=10),env AGENT_MAX_ROUNDS 可调;配合 prompt 的"命中即答"
# 规则,把 benchmark 实测的 11 轮收敛到 3-4 轮,94.6s -> ~30s。
AGENT_MAX_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "4"))
_AGENT_RECURSION_LIMIT = 2 + max(AGENT_MAX_ROUNDS, 1) * 2  # START + 每轮 2 节点 + END

# Attach runtime info for LangGraph Studio
_prompt_metadata: dict[str, str] = {
    "prompt_source": prompt_source,
    "guardrails_prompt_source": guardrails_prompt_source,
}
if prompt_commit:
    _prompt_metadata["prompt_commit"] = prompt_commit
if guardrails_prompt_commit:
    _prompt_metadata["guardrails_prompt_commit"] = guardrails_prompt_commit
if _revision_id := os.environ.get("LANGCHAIN_REVISION_ID"):
    _prompt_metadata["LANGSMITH_AGENT_VERSION"] = _revision_id

docs_agent = docs_agent.with_config(
    recursion_limit=_AGENT_RECURSION_LIMIT,
    metadata=_prompt_metadata,
)
docs_agent.tools = knowledge_agent_tools
docs_agent.middleware = knowledge_agent_middleware

# ---------------------------------------------------------------------------
# Vault initialization and file watcher
# ---------------------------------------------------------------------------
# When the agent module loads, eagerly initialize the vault RAG pipeline
# (vector index + BM25 + wiki-link graph) so the first query is fast.
#
# The file watcher monitors vault changes and triggers incremental re-indexing
# without a full rebuild.

_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH", ".")

try:
    from src.tools.vault_tools import _init_vault
    _init_vault(_VAULT_PATH)
    logger.info(f"Vault RAG pipeline initialized: {_VAULT_PATH}")

    # Start file watcher for incremental updates
    try:
        from src.rag.watcher import create_vault_watcher, is_watcher_running
        if not is_watcher_running():
            _watcher = create_vault_watcher(_VAULT_PATH)
            _watcher.start()
            logger.info("Vault file watcher started")
        else:
            logger.info("Vault file watcher already running")
    except ImportError:
        logger.info("watchdog not installed — file watching disabled")
    except Exception:
        logger.warning("Failed to start file watcher", exc_info=True)

except Exception:
    logger.warning(
        f"Vault initialization failed for path: {_VAULT_PATH}. "
        f"Agent will start but search tools will return errors. "
        f"Check OBSIDIAN_VAULT_PATH in .env",
        exc_info=True,
    )

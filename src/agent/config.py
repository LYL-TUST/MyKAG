# Shared configuration for all agents (models, middleware, API keys)
import logging
import os
from dataclasses import dataclass
from typing import Optional

import dotenv
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model

# 必须在任何 src.* import 之前加载 .env:
# guardrails_middleware 在 import 时读取 USE_LOCAL_PROMPTS,若 load_dotenv 滞后
# 会误判为未启用并尝试从 LangSmith Hub 拉 prompt(网络超时 7s+,拖慢 auth 加载)。
dotenv.load_dotenv()

from src.middleware.retry_middleware import ModelRetryMiddleware
from src.middleware.tool_retry_middleware import ToolRetryMiddleware

logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry
# =============================================================================


@dataclass
class ModelConfig:
    key: str  # Internal config key, e.g. "deepseek-v4"
    id: str  # LangChain provider model id, e.g. "deepseek-ai/DeepSeek-V4-Flash"
    name: str  # Display name, e.g., "DeepSeek V4"
    provider: str  # LangChain provider name, e.g. "openai", "deepseek", "google", "baseten"
    api_key_env: str  # Environment variable for API key
    description: Optional[str] = None


# All backend-supported models. This intentionally mirrors the frontend's
# selectable model IDs plus the guardrails/fallback models.
#
# Naming convention:
# - provider: LangChain provider name used by init_chat_model()
# - id: provider-specific model ID
# - vendor/transport details belong in description, not provider
MODELS: dict[str, ModelConfig] = {
    # Anthropic
    "claude-haiku-4.5": ModelConfig(
        key="claude-haiku-4.5",
        id="anthropic:claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        description="Anthropic model for fast general-purpose responses",
    ),
    # OpenAI-compatible models
    "gpt-5.4-nano": ModelConfig(
        key="gpt-5.4-nano",
        id="openai:gpt-5.4-nano",
        name="GPT-5.4 Nano",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="OpenAI-compatible model for simple high-volume tasks",
    ),
    "qwen3-8b": ModelConfig(
        key="qwen3-8b",
        id="Qwen/Qwen3-8B",
        name="Qwen3 8B",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="Qwen model accessed through an OpenAI-compatible endpoint",
    ),
    "gpt-5.4-mini": ModelConfig(
        key="gpt-5.4-mini",
        id="openai:gpt-5.4-mini",
        name="GPT-5.4 Mini",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="OpenAI-compatible model for coding, computer use, and subagents",
    ),
    "deepseek-v4": ModelConfig(
        key="deepseek-v4",
        id="deepseek-ai/DeepSeek-V4-Flash",
        name="DeepSeek V4",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="DeepSeek model accessed through an OpenAI-compatible endpoint",
    ),
    # Google
    "gemini-2.5-flash": ModelConfig(
        key="gemini-2.5-flash",
        id="google_genai:gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Fast and capable Google model",
    ),
    "gemini-3.1-flash-lite": ModelConfig(
        key="gemini-3.1-flash-lite",
        id="google_genai:gemini-3.1-flash-lite-preview",
        name="Gemini 3.1 Flash Lite",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Fastest, most cost-effective Gemini",
    ),
    # Baseten
    "glm-5": ModelConfig(
        key="glm-5",
        id="baseten:zai-org/GLM-5",
        name="GLM 5",
        provider="baseten",
        api_key_env="BASETEN_API_KEY",
        description="Z.ai GLM 5 served via Baseten",
    ),
}

# Default models for different use cases.
# Prefer explicit env overrides first, then fall back to safe defaults.
def _get_model_by_key(env_var: str, fallback_key: str) -> ModelConfig:
    model_key = os.getenv(env_var, fallback_key).strip()
    return MODELS.get(model_key, MODELS[fallback_key])

DEFAULT_MODEL = _get_model_by_key("DEFAULT_MODEL_KEY", "deepseek-v4")
GUARDRAILS_MODEL = _get_model_by_key("GUARDRAILS_MODEL_KEY", "qwen3-8b")
# Context-summarization model (SummarizationMiddleware). gpt-5.4-nano is NOT
# served on the configured OpenAI-compatible endpoint (SiliconFlow), so default
# to qwen3-8b; override with SUMMARY_MODEL_KEY.
SUMMARY_MODEL = _get_model_by_key("SUMMARY_MODEL_KEY", "qwen3-8b")

DEFAULT_MODEL_PROVIDER = "openai"
GUARDRAILS_MODEL_PROVIDER = "openai"

# Backward-compatible aliases for code paths that still expect model IDs.
DEFAULT_MODEL_ID = DEFAULT_MODEL.id
GUARDRAILS_MODEL_ID = GUARDRAILS_MODEL.id
SUMMARY_MODEL_ID = SUMMARY_MODEL.id

# Models public API callers are allowed to select. This mirrors the frontend
# deployment allowlist; backend-only guardrails/fallback models stay excluded.
PUBLIC_MODEL_KEYS = [
    "gpt-5.4-mini",
    "deepseek-v4",
    "qwen3-8b",
    "gemini-3.1-flash-lite",
    "glm-5",
]
PUBLIC_MODEL_IDS = {MODELS[key].id for key in PUBLIC_MODEL_KEYS}
PUBLIC_MODEL_PROVIDERS = {
    DEFAULT_MODEL_PROVIDER,
    GUARDRAILS_MODEL_PROVIDER,
}

# Fallback chain (in order of preference)
# Only include providers that have a configured API key to avoid import-time failures.
FALLBACK_MODELS = [
    model
    for model in [
        MODELS["gemini-2.5-flash"],
        MODELS["claude-haiku-4.5"],
    ]
    if os.getenv(model.api_key_env)
]

# =============================================================================
# API Key Setup
# =============================================================================

# `OPENAI_API_KEY` is used for OpenAI-compatible endpoints.
# `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, and `BASETEN_API_KEY` are used by
# their respective LangChain providers.
API_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "BASETEN_API_KEY",
]

for key in API_KEYS:
    if value := os.getenv(key):
        os.environ[key] = value.strip()
        logger.info(f"{key} configured")


# =============================================================================
# Model Initialization
# =============================================================================

# Retry configuration
# Keep retries conservative in local dev to avoid long queue buildup when a provider is slow.
MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "1"))

# ---------------------------------------------------------------------------
# Thinking / reasoning toggle
# ---------------------------------------------------------------------------
# SiliconFlow serves reasoning models (DeepSeek-V4, Qwen3) with the chain-of-
# thought turned ON by default. A trivial "你是谁" then takes ~18s because the
# model emits hundreds of hidden reasoning tokens before the first visible one.
# The `enable_thinking=false` body flag disables CoT and cuts first-token
# latency roughly in half (18s -> 9s for DeepSeek-V4, 12s -> 4.6s for Qwen3).
# Set DISABLE_THINKING=0 to restore reasoning (higher quality, slower).
_DISABLE_THINKING = os.getenv("DISABLE_THINKING", "1").lower() not in {
    "0", "false", "no", "off",
}
# Injected as extra_body (merged into the provider request body). None = no-op.
THINKING_EXTRA_BODY = {"enable_thinking": False} if _DISABLE_THINKING else None

# Per-request timeout for LLM HTTP calls (seconds). SiliconFlow occasionally
# stalls a socket indefinitely; the openai SDK default (600s) then wedges the
# whole event loop in asyncio.select with no cancellation point. A 120s cap
# makes hung calls fail fast so middleware/benchmark timeouts can act.
MODEL_REQUEST_TIMEOUT = int(os.getenv("MODEL_REQUEST_TIMEOUT", "120"))


def _thinking_kwargs() -> dict:
    """Return the **kwargs fragment to pass thinking toggle into init_chat_model."""
    if THINKING_EXTRA_BODY is None:
        return {}
    return {"extra_body": THINKING_EXTRA_BODY}


# Primary configurable model (can be switched at runtime).
# The runtime uses the LangChain provider name, not the vendor/transport label.
#
# Wrapped in try/except so importing this module NEVER fails on missing
# credentials. In environments without OPENAI_API_KEY (e.g. CI unit tests)
# the proxy is left as None; any actual .invoke() will then raise a clear
# error at the call site instead of crashing all module imports.
try:
    configurable_model = init_chat_model(
        model=DEFAULT_MODEL.id,
        model_provider=DEFAULT_MODEL_PROVIDER,
        configurable_fields=("model",),
        request_timeout=MODEL_REQUEST_TIMEOUT,
        **_thinking_kwargs(),
    )
except Exception as exc:  # noqa: BLE001 - intentional: credentials may be absent
    configurable_model = None
    logger.warning(
        "configurable_model init skipped (%s: %s); "
        "set OPENAI_API_KEY for runtime use",
        exc.__class__.__name__,
        exc,
    )
logger.info(
    "Default model: %s (%s, provider=%s, runtime_provider=%s)",
    DEFAULT_MODEL.name,
    DEFAULT_MODEL.id,
    DEFAULT_MODEL.provider,
    DEFAULT_MODEL_PROVIDER,
)

# =============================================================================
# Middleware
# =============================================================================

model_retry_middleware = ModelRetryMiddleware(max_retries=MAX_RETRIES)
tool_retry_middleware = ToolRetryMiddleware(max_attempts=3)

if FALLBACK_MODELS:
    model_fallback_middleware = ModelFallbackMiddleware(*[m.id for m in FALLBACK_MODELS])
    logger.info("Fallback chain: %s", " -> ".join(f"{m.name} [{m.provider}]" for m in FALLBACK_MODELS))
else:
    model_fallback_middleware = None
    logger.info("Fallback chain disabled (no API keys available)")

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models
    "MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_ID",
    "GUARDRAILS_MODEL",
    "GUARDRAILS_MODEL_ID",
    "SUMMARY_MODEL",
    "SUMMARY_MODEL_ID",
    "FALLBACK_MODELS",
    "PUBLIC_MODEL_IDS",
    "PUBLIC_MODEL_KEYS",
    "ModelConfig",
    # Configurable models
    "configurable_model",
    "MODEL_REQUEST_TIMEOUT",
    "THINKING_EXTRA_BODY",
    "_thinking_kwargs",
    # Middleware
    "model_retry_middleware",
    "tool_retry_middleware",
    "model_fallback_middleware",
    # Config
    "MAX_RETRIES",
    "logger",
]

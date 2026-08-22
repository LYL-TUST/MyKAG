"""SiliconFlow model-instance warm-up.

SiliconFlow schedules a GPU inference instance lazily on the FIRST request for
a model; that cold call can take 70s+ (measured: same prompt 74.5s -> 9.4s ->
9.5s). The dev server already spends 1-5 minutes compiling graphs + loading the
vault at startup, so firing one tiny keep-warm request per model in the
background hides the cold-start cost behind the boot time.

Warm-up is best-effort: any failure (no API key, network, provider error) is
logged and swallowed — the app must never block or fail because of it.
"""

from __future__ import annotations

import logging
import os
import threading

from langchain.chat_models import init_chat_model

from src.agent.config import (
    MODEL_MAX_RETRIES,
    MODEL_REQUEST_TIMEOUT,
    MODELS,
    _thinking_kwargs,
)

logger = logging.getLogger(__name__)

# 需要预热的模型:router/guardrails 用 qwen3-8b(最常调用),主模型 deepseek-v4
WARMUP_MODEL_KEYS = ("qwen3-8b", "deepseek-v4")

# 极短 prompt:只触发实例调度,不产生有意义输出
_WARMUP_PROMPT = "回复:ok"


def warmup_models() -> None:
    """Send one minimal request per configured model. Best-effort, sync."""
    for key in WARMUP_MODEL_KEYS:
        cfg = MODELS.get(key)
        if cfg is None:
            continue
        if not os.getenv(cfg.api_key_env):
            logger.info("warmup skipped %s (no %s)", key, cfg.api_key_env)
            continue
        try:
            model = init_chat_model(
                model=cfg.id,
                model_provider="openai",
                temperature=0,
                request_timeout=MODEL_REQUEST_TIMEOUT,
                max_retries=MODEL_MAX_RETRIES,
                **_thinking_kwargs(),
            )
            resp = model.invoke(_WARMUP_PROMPT)
            logger.info(
                "warmup OK %s: %r", key,
                str(resp.content or "")[:30],
            )
        except Exception as exc:  # noqa: BLE001 - 预热失败绝不影响启动
            logger.warning("warmup failed %s: %s", key, exc)


def start_warmup_background() -> None:
    """Fire warm-up on a daemon thread so it never blocks startup."""
    try:
        t = threading.Thread(
            target=warmup_models, name="model-warmup", daemon=True,
        )
        t.start()
        logger.info("model warm-up started in background thread")
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to start warm-up thread: %s", exc)


__all__ = ["warmup_models", "start_warmup_background", "WARMUP_MODEL_KEYS"]

"""Custom embedding adapter for SiliconFlow-compatible endpoints.

llama-index 0.14.x enforces an enum of OpenAI-only model names in
`OpenAIEmbedding`, which rejects third-party OpenAI-compatible models
(e.g. SiliconFlow's BAAI/bge-m3). This adapter subclasses BaseEmbedding
directly and calls the OpenAI-compatible `/embeddings` endpoint via the
`openai` SDK, so any model id works.

This is also a nice architectural talking point: it demonstrates
understanding of llama-index's embedding abstraction layer.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class SiliconFlowEmbedding(BaseEmbedding):
    """Embedding adapter for any OpenAI-compatible embedding endpoint.

    Uses the `openai` SDK under the hood, hitting the configured
    `base_url` (SiliconFlow by default). Accepts arbitrary model ids
    such as `BAAI/bge-m3`.
    """

    model_name: str = "BAAI/bge-m3"
    api_key: str = ""
    api_base: Optional[str] = None
    embed_batch_size: int = 32
    _client: Any = None

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        api_key: str = "",
        api_base: Optional[str] = None,
        embed_batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        import os
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = api_base or os.environ.get("OPENAI_BASE_URL", None)

        from openai import OpenAI
        client_kwargs: dict = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        self._client = OpenAI(**client_kwargs)

        logger.info(
            f"SiliconFlowEmbedding initialized: model={self.model_name}, "
            f"base_url={self.api_base}"
        )

    @classmethod
    def class_name(cls) -> str:
        return "siliconflow_embedding"

    def _get_text_embedding(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(
            model=self.model_name, input=text,
        )
        return resp.data[0].embedding

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Chunk to respect batch size
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            resp = self._client.embeddings.create(
                model=self.model_name, input=batch,
            )
            all_embeddings.extend(d.embedding for d in resp.data)
        return all_embeddings

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

"""Semantic answer cache for the entry router.

Repeated or near-duplicate questions are the most common traffic pattern in a
personal knowledge agent (interview prep, revisiting the same topic). Without a
cache every repeat pays the full orchestration cost (30-95s). This cache:

- embeds each query with the same bge-m3 model used by the RAG pipeline;
- on lookup, returns the stored answer when cosine similarity >= threshold;
- is a plain in-memory list (capacity-capped, TTL-expiring) — no extra
  dependencies, restart-safe by design (cache is best-effort, not durable).

A hit costs one embedding call (~0.5s) instead of a full graph run.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SemanticAnswerCache:
    """In-memory semantic cache: query embedding -> stored answer.

    Two lookup layers:
    1. **Exact LRU layer** (0ms): normalized string match in an OrderedDict.
       Repeated questions in the same session hit instantly with zero network
       calls — no embedding request, no cosine scan.
    2. **Semantic layer** (~0.5s): bge-m3 embedding + cosine similarity scan,
       catches near-duplicate phrasings. Only reached on exact-LRU miss.
    """

    def __init__(
        self,
        capacity: int = 256,
        threshold: float = 0.92,
        ttl_seconds: int = 86400,
        model: str = "BAAI/bge-m3",
    ) -> None:
        self.capacity = capacity
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.model = model
        self._entries: list[dict[str, Any]] = []
        self._exact: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._client: Any = None
        self._embed_failures = 0

    # -- exact LRU layer ---------------------------------------------------

    @staticmethod
    def _normalize(query: str) -> str:
        """Normalize a query for exact matching (case/whitespace-insensitive)."""
        return " ".join(query.strip().lower().split())

    def _exact_get(self, query: str) -> Optional[dict[str, Any]]:
        """Exact-LRU lookup: O(1) OrderedDict hit or None.

        On hit, re-inserts the key at the tail to keep LRU order and checks TTL.
        """
        norm = self._normalize(query)
        if not norm:
            return None
        entry = self._exact.pop(norm, None)
        if entry is None:
            return None
        if time.time() - entry["ts"] > self.ttl:
            logger.debug("exact cache entry expired, dropping: %r", query)
            return None
        self._exact[norm] = entry  # move to tail (most-recently-used)
        return {
            "answer": entry["answer"],
            "mode": entry["mode"],
            "similarity": 1.0,
        }

    def _exact_put(self, query: str, answer: str, mode: str) -> None:
        norm = self._normalize(query)
        if not norm:
            return
        self._exact[norm] = {"answer": answer, "mode": mode, "ts": time.time()}
        self._exact.move_to_end(norm)
        while len(self._exact) > self.capacity:
            self._exact.popitem(last=False)  # evict LRU

    # -- embedding ---------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                base_url=os.environ.get("OPENAI_BASE_URL")
                or "https://api.siliconflow.cn/v1",
                timeout=30,
                max_retries=0,
            )
        return self._client

    def _embed(self, text: str) -> Optional[list[float]]:
        try:
            resp = self._get_client().embeddings.create(
                model=self.model, input=[text[:2000]]
            )
            return list(resp.data[0].embedding)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            self._embed_failures += 1
            if self._embed_failures <= 3:
                logger.warning("semantic cache embedding failed: %s", exc)
            return None

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    # -- API ---------------------------------------------------------------

    def get(self, query: str) -> Optional[dict[str, Any]]:
        """Return cached {answer, mode, similarity} or None (miss / embed fail).

        Exact-LRU layer first (0ms, no network), then the semantic layer
        (one bge-m3 embedding call ~0.5s + cosine scan).
        """
        hit = self._exact_get(query)
        if hit is not None:
            logger.info("semantic cache exact HIT mode=%s", hit["mode"])
            return hit

        emb = self._embed(query)
        if emb is None:
            return None
        now = time.time()
        best: Optional[dict[str, Any]] = None
        best_sim = 0.0
        for entry in self._entries:
            if now - entry["ts"] > self.ttl:
                continue
            sim = self._cos(emb, entry["emb"])
            if sim > best_sim:
                best_sim, best = sim, entry
        if best is not None and best_sim >= self.threshold:
            logger.info(
                "semantic cache HIT (sim=%.3f) mode=%s", best_sim, best["mode"]
            )
            return {
                "answer": best["answer"],
                "mode": best["mode"],
                "similarity": round(best_sim, 3),
            }
        return None

    def put(self, query: str, answer: str, mode: str) -> None:
        """Store (or overwrite) the answer for a query. Best-effort."""
        emb = self._embed(query)
        if emb is None:
            return
        now = time.time()
        self._entries = [
            e for e in self._entries if e["query"] != query
        ]
        self._entries.append(
            {"query": query, "emb": emb, "answer": answer, "mode": mode, "ts": now}
        )
        # Expire old entries, then cap capacity (keep newest).
        self._entries = [
            e for e in self._entries if now - e["ts"] <= self.ttl
        ]
        if len(self._entries) > self.capacity:
            self._entries = self._entries[-self.capacity:]

        # Keep the exact-LRU layer in sync (no embedding cost here).
        self._exact_put(query, answer, mode)

    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries = []
        self._exact.clear()

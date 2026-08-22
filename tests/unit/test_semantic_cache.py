"""Unit tests for the semantic answer cache (no credentials needed)."""

import math
import time

from src.agent.semantic_cache import SemanticAnswerCache


def _vec(vals):
    """Deterministic fake embedding."""
    n = math.sqrt(sum(v * v for v in vals))
    return [v / n for v in vals]


def _mk_cache(monkeypatch, **kwargs):
    cache = SemanticAnswerCache(**kwargs)

    def fake_embed(text):
        # Deterministic character-level pseudo-embedding: different text ->
        # different vector; identical text -> identical vector (sim=1.0).
        return _vec([(ord(c) % 97) / 97.0 for c in text])

    monkeypatch.setattr(cache, "_embed", fake_embed)
    return cache


def test_put_get_exact_hit(monkeypatch):
    cache = _mk_cache(monkeypatch)
    cache.put("ellie 支持哪些 LLM 后端", "答案是...", "single")
    hit = cache.get("ellie 支持哪些 LLM 后端")
    assert hit is not None
    assert hit["answer"] == "答案是..."
    assert hit["mode"] == "single"
    assert hit["similarity"] >= 0.99


def test_miss_on_different_query(monkeypatch):
    cache = _mk_cache(monkeypatch)
    cache.put("ellie 支持哪些 LLM 后端", "答案是...", "single")
    hit = cache.get("今天天气怎么样")
    assert hit is None


def test_threshold_respected(monkeypatch):
    # threshold=1.0 → only near-identical queries hit
    cache = _mk_cache(monkeypatch, threshold=1.0)
    cache.put("ellie 支持哪些 LLM 后端", "答案是...", "single")
    hit = cache.get("ellie 支持哪些 LLM 后端?")
    assert hit is None  # slight difference drops below 1.0


def test_capacity_evicts_oldest(monkeypatch):
    cache = _mk_cache(monkeypatch, capacity=2)
    cache.put("ellie 支持哪些 LLM 后端", "a1", "single")
    cache.put("今天天气怎么样", "a2", "single")
    cache.put("AST 调用图压缩率是多少", "a3", "single")
    assert cache.size() == 2
    assert cache.get("ellie 支持哪些 LLM 后端") is None  # evicted (oldest)
    assert cache.get("AST 调用图压缩率是多少") is not None


def test_ttl_expiry(monkeypatch):
    cache = _mk_cache(monkeypatch, ttl_seconds=1)
    cache.put("q1", "a1", "single")
    assert cache.get("q1") is not None
    # Simulate time passing beyond TTL in BOTH layers (exact LRU + semantic).
    for e in cache._entries:
        e["ts"] -= 2
    for e in cache._exact.values():
        e["ts"] -= 2
    assert cache.get("q1") is None


def test_overwrite_same_query(monkeypatch):
    cache = _mk_cache(monkeypatch)
    cache.put("q1", "a1", "single")
    cache.put("q1", "a2", "workflow")
    hit = cache.get("q1")
    assert hit["answer"] == "a2"
    assert hit["mode"] == "workflow"
    assert cache.size() == 1


def test_clear(monkeypatch):
    cache = _mk_cache(monkeypatch)
    cache.put("q1", "a1", "single")
    cache.clear()
    assert cache.size() == 0
    assert cache.get("q1") is None

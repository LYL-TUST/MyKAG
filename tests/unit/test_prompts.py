"""Prompt-level tests: the knowledge assistant must do general-knowledge
fallback instead of "honestly refusing" vault-miss questions.

Regression guard for the fix that lets the assistant answer general tech
questions (e.g. "HTML5 新特性") from its own knowledge when the vault has no
matching note, while still forbidding fabricated note sources.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import src.agent.multi_agent_graph as mag
from src.prompts.guardrails_prompts import guardrails_system_prompt
from src.prompts.knowledge_agent import knowledge_agent_prompt


# ---------------------------------------------------------------------------
# Single-agent (knowledge_agent) prompt
# ---------------------------------------------------------------------------


def test_knowledge_agent_prompt_allows_general_knowledge_fallback() -> None:
    """The prompt must instruct fallback to general knowledge, not refusal."""
    assert "通用知识" in knowledge_agent_prompt
    assert "不要直接拒绝" in knowledge_agent_prompt
    # It must still forbid fabricating note sources.
    assert "不要编造笔记名" in knowledge_agent_prompt
    assert "不要假装有笔记依据" in knowledge_agent_prompt


def test_knowledge_agent_prompt_dropped_absolute_refusal_rule() -> None:
    """The old hard rule ('只能根据检索到的笔记内容回答') must be gone."""
    assert "你只能根据检索到的笔记内容回答" not in knowledge_agent_prompt


def test_knowledge_agent_prompt_mentions_vault_miss_example() -> None:
    """The vault-miss guidance should cite a concrete empty-result signal.

    The prompt must tell the model what a 'no notes' response looks like
    (``No relevant notes found``) so it can branch to general knowledge.
    """
    assert "No relevant notes found" in knowledge_agent_prompt


# ---------------------------------------------------------------------------
# Multi-agent summarizer prompt (shared by workflow + supervisor modes)
# ---------------------------------------------------------------------------


def test_summarizer_prompt_allows_general_knowledge_fallback() -> None:
    assert "通用知识" in mag._SUMMARIZER_SYSTEM
    assert "不要直接拒绝" in mag._SUMMARIZER_SYSTEM
    assert "不要编造" in mag._SUMMARIZER_SYSTEM


# ---------------------------------------------------------------------------
# Guardrails prompt
# ---------------------------------------------------------------------------


def test_guardrails_prompt_allows_general_tech_questions() -> None:
    """General tech questions must be explicitly allowed through guardrails."""
    assert "通用技术概念问题" in guardrails_system_prompt

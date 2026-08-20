"""Unit tests for the FastAPI app (title generation contract fix).

Covers the ``/generate-title`` endpoint: the frontend POSTs an object
``{userMessage, assistantResponse, maxLength}``, not a ``messages`` list,
so the endpoint must accept that shape (and keep tolerating the legacy list).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.fastapi_app import (
    TitleRequest,
    _truncate_title,
    app,
    generate_title,
)


# ---------------------------------------------------------------------------
# _truncate_title helper
# ---------------------------------------------------------------------------


def test_truncate_title_basic() -> None:
    assert _truncate_title("什么是知识图谱", 60) == "什么是知识图谱"


def test_truncate_title_truncates_and_appends_ellipsis() -> None:
    assert _truncate_title("A" * 100, 60) == "A" * 60 + "..."


def test_truncate_title_empty_returns_placeholder() -> None:
    assert _truncate_title("", 60) == "New Conversation"
    assert _truncate_title("   ", 60) == "New Conversation"


def test_truncate_title_clamps_bad_max_length() -> None:
    # max_length below 1 is clamped up to 1 (never produces an empty title).
    assert _truncate_title("hello world", 0) == "h..."
    # max_length above the 200 cap is clamped down.
    assert len(_truncate_title("x" * 300, 999)) <= 203


# ---------------------------------------------------------------------------
# generate_title — object payload (the frontend contract)
# ---------------------------------------------------------------------------


def test_generate_title_object_payload() -> None:
    out = asyncio.run(
        generate_title(TitleRequest(userMessage="什么是知识图谱?", maxLength=60))
    )
    assert out["title"] == "什么是知识图谱?"


def test_generate_title_object_respects_max_length() -> None:
    out = asyncio.run(
        generate_title(TitleRequest(userMessage="x" * 100, maxLength=30))
    )
    assert out["title"] == "x" * 30 + "..."


def test_generate_title_empty_user_message() -> None:
    out = asyncio.run(generate_title(TitleRequest(userMessage="", maxLength=60)))
    assert out["title"] == "New Conversation"


def test_generate_title_falls_back_to_assistant_response() -> None:
    out = asyncio.run(
        generate_title(
            TitleRequest(
                userMessage="", assistantResponse="这是助手回答", maxLength=60
            )
        )
    )
    assert out["title"] == "这是助手回答"


# ---------------------------------------------------------------------------
# generate_title — legacy list payload (backward compatibility)
# ---------------------------------------------------------------------------


def test_generate_title_legacy_list_payload() -> None:
    msgs = [
        {"role": "user", "content": "第一句"},
        {"role": "assistant", "content": "回答"},
    ]
    out = asyncio.run(generate_title(msgs))
    assert out["title"] == "第一句"


def test_generate_title_legacy_list_empty() -> None:
    assert asyncio.run(generate_title([]))["title"] == "New Conversation"


# ---------------------------------------------------------------------------
# End-to-end via TestClient: the 422 regression must be gone
# ---------------------------------------------------------------------------


def test_endpoint_accepts_object_payload() -> None:
    """POSTing the frontend's object shape must return 200, not 422."""
    client = TestClient(app)
    resp = client.post(
        "/generate-title",
        json={
            "userMessage": "什么是知识图谱?",
            "assistantResponse": "回答内容",
            "maxLength": 60,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] == "什么是知识图谱?"


def test_endpoint_accepts_legacy_list_payload() -> None:
    client = TestClient(app)
    resp = client.post(
        "/generate-title",
        json=[{"role": "user", "content": "第一句"}],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] == "第一句"

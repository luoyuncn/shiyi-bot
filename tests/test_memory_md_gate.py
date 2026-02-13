"""Tests for markdown gatekeeping in memory fact application."""

from __future__ import annotations

from pydantic import BaseModel
import pytest

from core.session_manager import SessionManager
from memory.md_gate import MarkdownGatekeeper


class MemoryConfig(BaseModel):
    sqlite_path: str = ":memory:"
    cache_size: int = 10
    auto_flush_interval: int = 60
    memory_root: str


def test_md_gatekeeper_user_blacklist_rejects_daily_event():
    gate = MarkdownGatekeeper()
    decision = gate.decide(
        scope="user",
        fact_key="today_activity",
        fact_value="今天吃了牛肉面",
        confidence=0.95,
    )
    assert decision.allow_md_write is False
    assert decision.reason.startswith("blacklisted:")
    assert decision.target_doc == "User.md"


def test_md_gatekeeper_user_whitelist_accepts_profession():
    gate = MarkdownGatekeeper()
    decision = gate.decide(
        scope="user",
        fact_key="profession",
        fact_value="Rust 开发工程师",
        confidence=0.95,
    )
    assert decision.allow_md_write is True
    assert decision.target_doc == "User.md"


@pytest.mark.asyncio
async def test_apply_memory_fact_keeps_short_term_event_out_of_user_md(tmp_path):
    manager = SessionManager(
        MemoryConfig(
            memory_root=str(tmp_path / "memory"),
            sqlite_path=str(tmp_path / "sessions.db"),
        )
    )
    await manager.initialize()

    applied = await manager._apply_memory_fact(
        fact={
            "scope": "user",
            "fact_type": "habit",
            "fact_key": "today_activity",
            "fact_value": "今天吃了牛肉面",
        },
        confidence=0.92,
        source_message_id="msg-today-activity",
    )

    assert applied is True
    facts = await manager.list_memory_facts(scope="user")
    assert any(f.fact_key == "today_activity" for f in facts)

    user_text = manager.documents.user_path.read_text(encoding="utf-8")
    assert "today_activity" not in user_text
    assert "今天吃了牛肉面" not in user_text

    events = await manager.list_memory_events(event_type="md_write_rejected")
    assert any(e.payload.get("fact_key") == "today_activity" for e in events)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_apply_memory_fact_writes_profession_to_user_md(tmp_path):
    manager = SessionManager(
        MemoryConfig(
            memory_root=str(tmp_path / "memory"),
            sqlite_path=str(tmp_path / "sessions.db"),
        )
    )
    await manager.initialize()

    applied = await manager._apply_memory_fact(
        fact={
            "scope": "user",
            "fact_type": "identity",
            "fact_key": "profession",
            "fact_value": "Rust 开发工程师",
        },
        confidence=0.95,
        source_message_id="msg-profession",
    )

    assert applied is True
    user_text = manager.documents.user_path.read_text(encoding="utf-8")
    assert "profession:" in user_text
    assert "Rust 开发工程师" in user_text

    events = await manager.list_memory_events(event_type="md_write_applied")
    assert any(e.payload.get("fact_key") == "profession" for e in events)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_apply_memory_fact_rejects_low_signal_project_update_for_md(tmp_path):
    manager = SessionManager(
        MemoryConfig(
            memory_root=str(tmp_path / "memory"),
            sqlite_path=str(tmp_path / "sessions.db"),
        )
    )
    await manager.initialize()

    applied = await manager._apply_memory_fact(
        fact={
            "scope": "project",
            "fact_type": "project",
            "fact_key": "status_update",
            "fact_value": "项目开始了",
        },
        confidence=0.90,
        source_message_id="msg-project-short",
    )

    assert applied is True
    facts = await manager.list_memory_facts(scope="project")
    assert any(f.fact_key == "status_update" for f in facts)

    project_text = manager.documents.project_path.read_text(encoding="utf-8")
    assert "项目开始了" not in project_text

    events = await manager.list_memory_events(event_type="md_write_rejected")
    assert any(e.payload.get("fact_key") == "status_update" for e in events)

    await manager.cleanup()

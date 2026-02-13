"""Regression tests for memory recall quality and personal milestone capture."""

import pytest
from pydantic import BaseModel

from core.session_manager import SessionManager


class MemoryConfig(BaseModel):
    sqlite_path: str = ":memory:"
    cache_size: int = 10
    auto_flush_interval: int = 60
    memory_root: str


@pytest.mark.asyncio
async def test_prepare_messages_does_not_recall_current_user_message(tmp_path):
    """Current user query should never be echoed back as retrieved history."""
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()
    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一。",
        user_identity="我是腿哥。",
        display_name="腿哥",
    )

    session = await manager.create_session({"channel": "test"})
    await manager.save_message(
        session.session_id,
        "user",
        "明天情人节了，我和我老婆是情人节领证的，2018年2月14号，明天是我们的结婚纪念日了。",
    )

    context = await manager.get_session(session.session_id)
    prepared = await manager.prepare_messages_for_agent(context.messages)

    assert not any(
        item["role"] == "system" and "[相关历史记忆]" in item["content"]
        for item in prepared
    )

    await manager.cleanup()


@pytest.mark.asyncio
async def test_recall_prompt_filters_low_relevance_hits(tmp_path, monkeypatch):
    """Low-overlap semantic noise should be filtered out of recall prompt."""
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()

    async def fake_search(*args, **kwargs):
        return [
            {
                "source_type": "message",
                "source_id": "m1",
                "content": "根据最新市场分析，2026年上海房价趋势：整体止跌回稳。",
                "semantic_score": 0.25,
                "keyword_score": 0.0,
                "freshness_score": 0.9,
                "final_score": 0.26,
                "retrieval_type": "semantic",
            },
            {
                "source_type": "message",
                "source_id": "m2",
                "content": "你家布偶猫叫富贵，还有一只金渐层叫酥酥。",
                "semantic_score": 0.24,
                "keyword_score": 0.0,
                "freshness_score": 0.9,
                "final_score": 0.24,
                "retrieval_type": "semantic",
            },
            {
                "source_type": "message",
                "source_id": "m3",
                "content": "你们2018年2月14日领证，明天是结婚纪念日。",
                "semantic_score": 0.36,
                "keyword_score": 0.12,
                "freshness_score": 0.8,
                "final_score": 0.38,
                "retrieval_type": "semantic",
            },
        ]

    monkeypatch.setattr(manager, "search_memory_hybrid", fake_search)

    prompt = await manager._build_recall_prompt(
        [{"role": "user", "content": "明天是我和老婆的结婚纪念日，礼物买什么好？"}]
    )

    assert prompt is not None
    assert "结婚纪念日" in prompt
    assert "房价趋势" not in prompt
    assert "布偶猫" not in prompt

    await manager.cleanup()


@pytest.mark.asyncio
async def test_apply_memory_fact_accepts_milestone_anniversary(tmp_path):
    """Anniversary milestone should be storable into user memory and User.md."""
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
            "fact_type": "milestone",
            "fact_key": "wedding_anniversary_date",
            "fact_value": "2018年2月14号",
        },
        confidence=0.95,
        source_message_id="msg-anniversary",
    )

    assert applied is True
    facts = await manager.list_memory_facts(scope="user")
    assert any(f.fact_key == "wedding_anniversary_date" for f in facts)

    user_text = manager.documents.user_path.read_text(encoding="utf-8")
    assert "wedding_anniversary_date" in user_text

    await manager.cleanup()


@pytest.mark.asyncio
async def test_recall_prompt_logs_candidates_when_all_filtered(tmp_path, monkeypatch):
    """When all candidates are filtered out, logs should still expose top candidates and reasons."""
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()

    async def fake_search(*args, **kwargs):
        return [
            {
                "source_type": "message",
                "source_id": "m1",
                "content": "根据最新市场分析，2026年上海房价趋势：整体止跌回稳。",
                "semantic_score": 0.34,
                "keyword_score": 0.0,
                "overlap_score": 0.01,
                "freshness_score": 0.9,
                "final_score": 0.35,
                "retrieval_type": "semantic",
                "message_role": "assistant",
            },
            {
                "source_type": "message",
                "source_id": "m2",
                "content": "你家布偶猫叫富贵，还有一只金渐层叫酥酥。",
                "semantic_score": 0.33,
                "keyword_score": 0.0,
                "overlap_score": 0.01,
                "freshness_score": 0.9,
                "final_score": 0.34,
                "retrieval_type": "semantic",
                "message_role": "assistant",
            },
        ]

    log_records: list[str] = []

    class _FakeLogger:
        @staticmethod
        def debug(msg):
            return None

        @staticmethod
        def warning(msg):
            return None

        def info(self, msg):
            log_records.append(str(msg))

    monkeypatch.setattr(manager, "search_memory_hybrid", fake_search)
    monkeypatch.setattr("core.session_manager.logger", _FakeLogger())

    prompt = await manager._build_recall_prompt(
        [{"role": "user", "content": "上海浦东2月分成交数据"}]
    )

    assert prompt is None
    assert any("候选#1" in item and "filtered=" in item for item in log_records)

    await manager.cleanup()

"""End-to-end tests for memory summarize/retrieval/metabolism pipeline."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from core.session_manager import SessionManager
from memory.documents import MemoryDocumentStore


class MemoryConfig(BaseModel):
    sqlite_path: str = ":memory:"
    memory_root: str
    cache_size: int = 10
    auto_flush_interval: int = 60


@pytest.mark.asyncio
async def test_summarize_and_store_routes_confidence_levels(tmp_path):
    """High confidence should auto-write; medium confidence should go pending."""
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()
    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一。",
        user_identity="我是腿哥。",
        display_name="腿哥",
    )

    await manager.summarize_and_store(
        "我偏好Go，我最近在用Rust，今天状态不错。",
        source_message_id="msg-1",
    )

    facts = await manager.list_memory_facts(scope="user")
    assert any(f.fact_key == "preferred_tech" and f.fact_value == "Go" for f in facts)

    pending = await manager.list_memory_pending(status="pending")
    assert any(
        p.candidate_fact.get("fact_key") == "preferred_tech"
        and p.candidate_fact.get("fact_value") == "Rust"
        for p in pending
    )

    await manager.cleanup()


@pytest.mark.asyncio
async def test_hybrid_search_can_recall_semantic_hits(tmp_path):
    """Hybrid retrieval should return semantic results even when keyword overlap is weak."""
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
        "这次故障是 Bad Gateway，核心原因是网关超时。",
    )
    await manager.run_embedding_pipeline(max_jobs=20, ignore_schedule=True)

    results = await manager.search_memory_hybrid("网关错误怎么定位", limit=5)

    assert len(results) >= 1
    assert any("Bad Gateway" in item["content"] for item in results)
    assert any(item["retrieval_type"] == "semantic" for item in results)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_embedding_retry_dead_letter_flow(tmp_path):
    """Embedding failures should retry then move into dead-letter queue."""
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()

    job_id = await manager.enqueue_embedding_job(
        source_type="message",
        source_id="msg-fail",
        content="[[force_embedding_error]]",
    )

    for _ in range(4):
        await manager.run_embedding_pipeline(max_jobs=1, ignore_schedule=True)

    dead_jobs = await manager.list_embedding_jobs(status="dead_letter")
    assert any(job.id == job_id for job in dead_jobs)

    events = await manager.list_memory_events(event_type="embedding_dead_letter")
    assert any(event.payload.get("job_id") == job_id for event in events)

    await manager.cleanup()


def test_markdown_metabolism_for_project_and_insights(tmp_path: Path):
    """Project uses rolling summary; insights keep a 10-item hot pool."""
    store = MemoryDocumentStore(str(tmp_path / "memory"))
    store.ensure_initialized()

    for i in range(120):
        store.append_project_update(f"任务进展 {i}")

    project_text = store.project_path.read_text(encoding="utf-8")
    assert "历史归档摘要" in project_text
    assert project_text.count("- 任务进展") < 120

    for i in range(12):
        store.add_insight(f"经验条目 {i}")

    insight_lines = [
        line
        for line in store.insights_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("- ")
    ]
    assert len(insight_lines) == 10
    assert any("经验条目 11" in line for line in insight_lines)
    assert all("经验条目 0" not in line for line in insight_lines)

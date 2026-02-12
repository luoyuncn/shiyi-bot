"""Tests for memory-related API endpoints."""

import pytest
import httpx
from pydantic import BaseModel

from channels.text_api_channel import TextAPIChannel
from core.session_manager import SessionManager


class MemoryConfig(BaseModel):
    sqlite_path: str = ":memory:"
    memory_root: str
    cache_size: int = 10
    auto_flush_interval: int = 60


class APIConfig(BaseModel):
    channels: dict = {"api": {"cors_origins": ["*"]}}


class DummyAgentCore:
    async def process_message_stream(self, messages, enable_tools=True):
        yield {"type": "done"}


@pytest.mark.asyncio
async def test_get_memory_user_state(tmp_path):
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()
    channel = TextAPIChannel(APIConfig(), manager, DummyAgentCore())
    transport = httpx.ASGITransport(app=channel.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/memory/user")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "global"
    assert payload["identity_confirmed"] is False

    await manager.cleanup()


@pytest.mark.asyncio
async def test_memory_onboarding_endpoint_updates_state_and_docs(tmp_path):
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()
    channel = TextAPIChannel(APIConfig(), manager, DummyAgentCore())
    transport = httpx.ASGITransport(app=channel.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/memory/onboarding",
            json={
                "shiyi_identity": "你是十一，资深开发助手。",
                "user_identity": "我是腿哥，偏好 Python。",
                "display_name": "腿哥",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["identity_confirmed"] is True
        assert payload["display_name"] == "腿哥"

        user_state_response = await client.get("/api/memory/user")
        assert user_state_response.status_code == 200
        assert user_state_response.json()["identity_confirmed"] is True

    assert channel.session_manager.documents.shiyi_path.exists()
    assert channel.session_manager.documents.user_path.exists()

    await manager.cleanup()


@pytest.mark.asyncio
async def test_memory_pending_list_and_status_update(tmp_path):
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()
    pending_id = await manager.save_memory_pending(
        candidate_fact={"scope": "user", "fact_key": "language"},
        confidence=0.7,
        source_message_id="msg-1",
    )
    channel = TextAPIChannel(APIConfig(), manager, DummyAgentCore())
    transport = httpx.ASGITransport(app=channel.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        pending_response = await client.get("/api/memory/pending")
        assert pending_response.status_code == 200
        pending_items = pending_response.json()
        assert any(item["id"] == pending_id for item in pending_items)

        update_response = await client.post(
            f"/api/memory/pending/{pending_id}",
            json={"status": "confirmed"},
        )
        assert update_response.status_code == 200
        assert update_response.json()["status"] == "ok"

        confirmed_response = await client.get("/api/memory/pending?status=confirmed")
        assert confirmed_response.status_code == 200
        confirmed_items = confirmed_response.json()
        assert any(item["id"] == pending_id for item in confirmed_items)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_memory_facts_and_events_endpoints(tmp_path):
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()
    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一。",
        user_identity="我是腿哥。",
        display_name="腿哥",
    )
    session = await manager.create_session({"channel": "test"})
    await manager.save_message(session.session_id, "user", "我更喜欢Python")

    channel = TextAPIChannel(APIConfig(), manager, DummyAgentCore())
    transport = httpx.ASGITransport(app=channel.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        facts_response = await client.get("/api/memory/facts?scope=user")
        assert facts_response.status_code == 200
        facts = facts_response.json()
        assert any(f["fact_key"] == "preferred_tech" and f["fact_value"] == "Python" for f in facts)

        events_response = await client.get("/api/memory/events")
        assert events_response.status_code == 200
        events = events_response.json()
        assert any(e["event_type"] == "identity_onboarding_completed" for e in events)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_memory_search_endpoint_returns_keyword_hits(tmp_path):
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()
    await manager.complete_identity_onboarding(
        shiyi_identity="你是十一。",
        user_identity="我是腿哥。",
        display_name="腿哥",
    )
    session = await manager.create_session({"channel": "test"})
    await manager.save_message(session.session_id, "user", "我们讨论过 Redis timeout 的排查方案")

    channel = TextAPIChannel(APIConfig(), manager, DummyAgentCore())
    transport = httpx.ASGITransport(app=channel.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/memory/search?q=Redis timeout&limit=5")
        assert response.status_code == 200
        items = response.json()
        assert len(items) >= 1
        assert any("Redis" in item["content"] for item in items)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_memory_metrics_endpoint(tmp_path):
    manager = SessionManager(MemoryConfig(memory_root=str(tmp_path / "memory")))
    await manager.initialize()
    channel = TextAPIChannel(APIConfig(), manager, DummyAgentCore())
    transport = httpx.ASGITransport(app=channel.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/memory/metrics")
        assert response.status_code == 200
        payload = response.json()
        assert "pending_by_status" in payload
        assert "active_facts" in payload
        assert "dead_letter_jobs" in payload
        assert "retrieval_fail_count" in payload

    await manager.cleanup()

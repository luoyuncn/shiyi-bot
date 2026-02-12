"""Tests for memory-system storage primitives (M0)."""

import pytest

from memory.storage import MemoryStorage


@pytest.mark.asyncio
async def test_initialize_creates_global_user_and_session_defaults():
    """Storage should bootstrap a global user and bind new sessions to it."""
    storage = MemoryStorage(db_path=":memory:")
    await storage.initialize()

    user_state = await storage.get_global_user_state()
    assert user_state["user_id"] == "global"
    assert user_state["identity_confirmed"] is False
    assert user_state["onboarding_prompted"] is False

    session_id = await storage.create_session({"channel": "test"})
    session = await storage.get_session(session_id)
    assert session is not None
    assert session.user_id == "global"

    await storage.cleanup()


@pytest.mark.asyncio
async def test_set_global_user_identity_state():
    """Global user identity state should be updateable."""
    storage = MemoryStorage(db_path=":memory:")
    await storage.initialize()

    await storage.set_global_user_identity_state(
        identity_confirmed=True,
        display_name="腿哥",
    )
    user_state = await storage.get_global_user_state()

    assert user_state["identity_confirmed"] is True
    assert user_state["display_name"] == "腿哥"

    await storage.cleanup()


@pytest.mark.asyncio
async def test_mark_onboarding_prompted():
    """Onboarding prompt marker should persist in DB."""
    storage = MemoryStorage(db_path=":memory:")
    await storage.initialize()

    user_state = await storage.get_global_user_state()
    assert user_state["onboarding_prompted"] is False

    await storage.mark_onboarding_prompted()
    user_state = await storage.get_global_user_state()
    assert user_state["onboarding_prompted"] is True

    await storage.cleanup()


@pytest.mark.asyncio
async def test_memory_pending_lifecycle():
    """Pending memory records should support create, list and status transitions."""
    storage = MemoryStorage(db_path=":memory:")
    await storage.initialize()

    pending_id = await storage.save_memory_pending(
        candidate_fact={"scope": "user", "fact_type": "preference", "fact_key": "language"},
        confidence=0.72,
        source_message_id="msg-1",
    )
    pending_items = await storage.list_memory_pending(status="pending")
    assert len(pending_items) == 1
    assert pending_items[0].id == pending_id

    await storage.update_memory_pending_status(pending_id, status="confirmed")
    pending_items = await storage.list_memory_pending(status="pending")
    confirmed_items = await storage.list_memory_pending(status="confirmed")

    assert len(pending_items) == 0
    assert len(confirmed_items) == 1
    assert confirmed_items[0].id == pending_id

    await storage.cleanup()

"""Tests for session manager"""
import pytest
from core.session_manager import SessionManager
from pydantic import BaseModel


class MemoryConfig(BaseModel):
    """Memory config for testing"""
    sqlite_path: str = ":memory:"
    cache_size: int = 10
    auto_flush_interval: int = 60


@pytest.mark.asyncio
async def test_create_and_get_session():
    """Test session creation and retrieval"""
    config = MemoryConfig()
    manager = SessionManager(config)
    await manager.initialize()

    # Create session
    context = await manager.create_session({"channel": "test"})
    assert context.session_id is not None

    # Get session from cache
    retrieved = await manager.get_session(context.session_id)
    assert retrieved is not None
    assert retrieved.session_id == context.session_id

    await manager.cleanup()


@pytest.mark.asyncio
async def test_save_and_load_messages():
    """Test message persistence"""
    config = MemoryConfig()
    manager = SessionManager(config)
    await manager.initialize()

    context = await manager.create_session({})
    session_id = context.session_id

    # Save messages
    await manager.save_message(session_id, "user", "你好")
    await manager.save_message(session_id, "assistant", "你好！")

    # Clear cache to force database load
    manager.cache.clear()

    # Load from database
    loaded = await manager.get_session(session_id)
    assert len(loaded.messages) == 2

    await manager.cleanup()

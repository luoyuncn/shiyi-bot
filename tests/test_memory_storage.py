"""Tests for memory storage layer"""
import pytest
from memory.storage import MemoryStorage


@pytest.mark.asyncio
async def test_create_session():
    """Test creating a new session"""
    storage = MemoryStorage(db_path=":memory:")
    await storage.initialize()

    session_id = await storage.create_session({"channel": "test"})

    assert session_id is not None
    assert len(session_id) == 36  # UUID format

    # Verify session exists
    session = await storage.get_session(session_id)
    assert session is not None
    assert session.session_metadata["channel"] == "test"

    await storage.cleanup()


@pytest.mark.asyncio
async def test_save_and_get_messages():
    """Test saving and retrieving messages"""
    storage = MemoryStorage(db_path=":memory:")
    await storage.initialize()

    session_id = await storage.create_session({})

    # Save messages
    await storage.save_message(session_id, "user", "你好")
    await storage.save_message(session_id, "assistant", "你好！有什么可以帮你的？")

    # Retrieve messages
    messages = await storage.get_messages(session_id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "你好"
    assert messages[1].role == "assistant"

    await storage.cleanup()


@pytest.mark.asyncio
async def test_keyword_search_messages_with_fts():
    """FTS-based keyword search should return matching messages."""
    storage = MemoryStorage(db_path=":memory:")
    await storage.initialize()

    session_id = await storage.create_session({})
    await storage.save_message(session_id, "user", "我们之前排查过 Redis timeout 问题")
    await storage.save_message(session_id, "assistant", "先看连接池和网络延迟")

    results = await storage.search_messages_by_keyword("Redis timeout", limit=5)
    assert len(results) >= 1
    assert "Redis" in results[0]["content"]
    assert results[0]["session_id"] == session_id

    await storage.cleanup()

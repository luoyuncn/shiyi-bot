"""Tests for LRU cache"""
import pytest
from datetime import datetime
from memory.cache import LRUCache, ConversationContext


def test_cache_put_and_get():
    """Test basic cache operations"""
    cache = LRUCache(max_size=3)

    ctx1 = ConversationContext(session_id="session1")
    ctx2 = ConversationContext(session_id="session2")

    cache.put("session1", ctx1)
    cache.put("session2", ctx2)

    assert cache.get("session1") == ctx1
    assert cache.get("session2") == ctx2
    assert cache.get("session3") is None


def test_cache_lru_eviction():
    """Test LRU eviction when cache is full"""
    cache = LRUCache(max_size=2)

    ctx1 = ConversationContext(session_id="session1")
    ctx2 = ConversationContext(session_id="session2")
    ctx3 = ConversationContext(session_id="session3")

    cache.put("session1", ctx1)
    cache.put("session2", ctx2)
    cache.put("session3", ctx3)  # Should evict session1

    assert cache.get("session1") is None
    assert cache.get("session2") == ctx2
    assert cache.get("session3") == ctx3


def test_conversation_context():
    """Test ConversationContext operations"""
    ctx = ConversationContext(session_id="test", metadata={"channel": "cli"})

    ctx.add_message("user", "你好")
    ctx.add_message("assistant", "你好！")

    assert len(ctx.messages) == 2
    assert ctx.messages[0]["role"] == "user"
    assert ctx.messages[1]["content"] == "你好！"

    # Test to_dict
    data = ctx.to_dict()
    assert data["session_id"] == "test"
    assert len(data["messages"]) == 2

"""LRU cache for hot session data"""
from collections import OrderedDict
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationContext:
    """Conversation context"""
    session_id: str
    messages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, metadata: dict = None):
        """Add message"""
        self.messages.append({
            "role": role,
            "content": content,
            "metadata": metadata or {}
        })
        self.last_active = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dict"""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat()
        }


class LRUCache:
    """LRU cache for active sessions"""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, ConversationContext] = OrderedDict()
        self._max_size = max_size

    def get(self, session_id: str) -> Optional[ConversationContext]:
        """Get session (LRU: move to end)"""
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
            return self._cache[session_id]
        return None

    def put(self, session_id: str, context: ConversationContext):
        """Put into cache"""
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
        else:
            if len(self._cache) >= self._max_size:
                # Evict oldest
                # 淘汰最旧会话
                oldest_id, _ = self._cache.popitem(last=False)
                from loguru import logger
                logger.debug(f"缓存满，淘汰会话: {oldest_id}")

        self._cache[session_id] = context

    def remove(self, session_id: str):
        """Remove session"""
        self._cache.pop(session_id, None)

    def clear(self):
        """Clear cache"""
        self._cache.clear()

    def size(self) -> int:
        """Current cache size"""
        return len(self._cache)

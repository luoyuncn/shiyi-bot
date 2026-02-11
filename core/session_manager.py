"""Session manager - integrates storage and cache"""
import asyncio
from typing import Optional
from loguru import logger

from memory.storage import MemoryStorage, SessionRecord
from memory.cache import LRUCache, ConversationContext


class SessionManager:
    """Session manager - singleton"""

    def __init__(self, memory_config):
        self.config = memory_config

        # Storage layer
        self.storage = MemoryStorage(memory_config.sqlite_path)

        # Cache layer
        self.cache = LRUCache(max_size=memory_config.cache_size)

        # Auto flush task
        self._flush_task = None
        self._running = False

    async def initialize(self):
        """Initialize"""
        await self.storage.initialize()

        # Start auto flush loop
        self._running = True
        self._flush_task = asyncio.create_task(self._auto_flush_loop())

        logger.info("会话管理器初始化完成")

    async def create_session(self, metadata: dict = None) -> ConversationContext:
        """Create new session"""
        # Create in database
        session_id = await self.storage.create_session(metadata or {})

        # Create in-memory context
        context = ConversationContext(
            session_id=session_id,
            metadata=metadata or {}
        )

        # Put into cache
        self.cache.put(session_id, context)

        logger.info(f"创建会话: {session_id}")
        return context

    async def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get session (from cache first, then database)"""
        # Try cache first
        context = self.cache.get(session_id)
        if context:
            return context

        # Load from database
        record = await self.storage.get_session(session_id)
        if not record:
            return None

        # Load messages
        messages = await self.storage.get_messages(session_id)

        # Rebuild context
        context = ConversationContext(
            session_id=session_id,
            messages=[
                {
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.message_metadata
                }
                for msg in messages
            ],
            metadata=record.session_metadata,
            created_at=record.created_at,
            last_active=record.last_active
        )

        # Put into cache
        self.cache.put(session_id, context)

        return context

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict = None
    ):
        """Save message"""
        # Update cache
        context = self.cache.get(session_id)
        if context:
            context.add_message(role, content, metadata)

        # Async write to database (using await for async I/O, fast enough)
        await self.storage.save_message(session_id, role, content, metadata)

    async def list_sessions(self, limit: int = 50) -> list[SessionRecord]:
        """List all sessions"""
        return await self.storage.list_sessions(limit=limit)

    async def delete_session(self, session_id: str):
        """Delete session"""
        self.cache.remove(session_id)
        await self.storage.delete_session(session_id)

    async def _auto_flush_loop(self):
        """Auto flush loop"""
        interval = self.config.auto_flush_interval

        while self._running:
            await asyncio.sleep(interval)
            logger.debug(f"缓存状态: {self.cache.size()} 个活跃会话")

    async def cleanup(self):
        """Cleanup resources"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            await asyncio.gather(self._flush_task, return_exceptions=True)

        await self.storage.cleanup()
        self.cache.clear()

"""SQLite storage layer for session persistence and memory pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    delete,
    func,
    select,
    text,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from memory.embeddings import LocalEmbeddingEngine

Base = declarative_base()
GLOBAL_USER_ID = "global"


class SessionRecord(Base):
    """Session record table."""

    __tablename__ = "sessions"

    session_id = Column(String(36), primary_key=True)
    user_id = Column(String(64), nullable=False, default=GLOBAL_USER_ID, index=True)
    created_at = Column(DateTime, nullable=False)
    last_active = Column(DateTime, nullable=False)
    session_metadata = Column(JSON, default=dict)
    message_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)


class MessageRecord(Base):
    """Message record table."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    message_metadata = Column(JSON, default=dict)


class UserRecord(Base):
    """Global user state."""

    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True)
    identity_confirmed = Column(Boolean, nullable=False, default=False)
    onboarding_prompted = Column(Boolean, nullable=False, default=False)
    display_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class MemoryPendingRecord(Base):
    """Low/medium confidence memories waiting user confirmation."""

    __tablename__ = "memory_pending"

    id = Column(String(36), primary_key=True)
    candidate_fact = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    source_message_id = Column(String(36), nullable=True)
    cooldown_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class MemoryEventRecord(Base):
    """Memory pipeline events for observability and audit trail."""

    __tablename__ = "memory_events"

    id = Column(String(36), primary_key=True)
    event_type = Column(String(50), nullable=False, index=True)
    operation_id = Column(String(64), nullable=True, index=True)
    payload = Column(JSON, default=dict)
    created_at = Column(DateTime, nullable=False)


class MemoryFactRecord(Base):
    """Normalized memory facts with confidence and status."""

    __tablename__ = "memory_facts"

    id = Column(String(36), primary_key=True)
    scope = Column(String(30), nullable=False, index=True)
    fact_type = Column(String(30), nullable=False, index=True)
    fact_key = Column(String(100), nullable=False, index=True)
    fact_value = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="active", index=True)
    source_message_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class MemoryEmbeddingRecord(Base):
    """Vectorized memory records used for semantic retrieval."""

    __tablename__ = "memory_embeddings"

    id = Column(String(36), primary_key=True)
    source_type = Column(String(30), nullable=False, index=True)
    source_id = Column(String(64), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class EmbeddingJobRecord(Base):
    """Async embedding queue with retry/dead-letter states."""

    __tablename__ = "embedding_jobs"

    id = Column(String(36), primary_key=True)
    source_type = Column(String(30), nullable=False, index=True)
    source_id = Column(String(64), nullable=False, index=True)
    content = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="pending", index=True)
    retry_count = Column(Integer, nullable=False, default=0)
    next_retry_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class MemoryStorage:
    """SQLite storage manager."""

    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = db_path
        self.engine = None
        self.session_factory = None

    async def initialize(self):
        """Initialize database and run lightweight migrations."""
        from pathlib import Path

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.db_path}", echo=False)

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await self._migrate_existing_schema(conn)
            await self._ensure_message_fts(conn)

        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        await self._ensure_global_user()
        logger.info(f"数据库初始化完成: {self.db_path}")

    async def _migrate_existing_schema(self, conn):
        """Run lightweight migrations for existing SQLite deployments."""
        columns_result = await conn.execute(text("PRAGMA table_info(sessions)"))
        session_columns = [row[1] for row in columns_result]
        if "user_id" not in session_columns:
            await conn.execute(
                text(
                    "ALTER TABLE sessions "
                    "ADD COLUMN user_id VARCHAR(64) DEFAULT 'global'"
                )
            )
            await conn.execute(
                text(
                    "UPDATE sessions SET user_id = 'global' "
                    "WHERE user_id IS NULL OR user_id = ''"
                )
            )

        user_columns_result = await conn.execute(text("PRAGMA table_info(users)"))
        user_columns = [row[1] for row in user_columns_result]
        if user_columns and "onboarding_prompted" not in user_columns:
            await conn.execute(
                text(
                    "ALTER TABLE users "
                    "ADD COLUMN onboarding_prompted BOOLEAN DEFAULT 0"
                )
            )
            await conn.execute(
                text(
                    "UPDATE users SET onboarding_prompted = 0 "
                    "WHERE onboarding_prompted IS NULL"
                )
            )

    async def _ensure_message_fts(self, conn):
        """Create and backfill FTS5 index for message keyword retrieval."""
        await conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS message_fts "
                "USING fts5(message_id UNINDEXED, session_id UNINDEXED, content)"
            )
        )

        fts_count = await conn.execute(text("SELECT COUNT(*) FROM message_fts"))
        if (fts_count.scalar() or 0) > 0:
            return

        await conn.execute(
            text(
                "INSERT INTO message_fts(message_id, session_id, content) "
                "SELECT id, session_id, content FROM messages"
            )
        )

    async def _ensure_global_user(self):
        """Guarantee a singleton global user record exists."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(UserRecord).where(UserRecord.user_id == GLOBAL_USER_ID)
            )
            if result.scalar_one_or_none():
                return

            now = datetime.now()
            session.add(
                UserRecord(
                    user_id=GLOBAL_USER_ID,
                    identity_confirmed=False,
                    onboarding_prompted=False,
                    display_name=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()

    async def create_session(self, metadata: dict) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        async with self.session_factory() as session:
            session.add(
                SessionRecord(
                    session_id=session_id,
                    user_id=GLOBAL_USER_ID,
                    created_at=now,
                    last_active=now,
                    session_metadata=metadata,
                    message_count=0,
                    total_tokens=0,
                )
            )
            await session.commit()
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get session by ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SessionRecord).where(SessionRecord.session_id == session_id)
            )
            return result.scalar_one_or_none()

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        """Persist one message and update session counters."""
        message_id = str(uuid.uuid4())
        now = datetime.now()

        async with self.session_factory() as session:
            session.add(
                MessageRecord(
                    id=message_id,
                    session_id=session_id,
                    role=role,
                    content=content,
                    timestamp=now,
                    message_metadata=metadata or {},
                )
            )
            await session.execute(
                text(
                    "INSERT INTO message_fts(message_id, session_id, content) "
                    "VALUES (:message_id, :session_id, :content)"
                ),
                {
                    "message_id": message_id,
                    "session_id": session_id,
                    "content": content,
                },
            )
            await session.execute(
                update(SessionRecord)
                .where(SessionRecord.session_id == session_id)
                .values(
                    last_active=now,
                    message_count=SessionRecord.message_count + 1,
                )
            )
            await session.commit()

        return message_id

    async def get_messages(self, session_id: str, limit: int = 100) -> list[MessageRecord]:
        """Get ordered message list for a session."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(MessageRecord)
                .where(MessageRecord.session_id == session_id)
                .order_by(MessageRecord.timestamp.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def search_messages_by_keyword(self, query: str, limit: int = 5) -> list[dict]:
        """Search messages by keyword using SQLite FTS5 (with LIKE fallback)."""
        query = (query or "").strip()
        if not query:
            return []

        async with self.session_factory() as session:
            try:
                result = await session.execute(
                    text(
                        "SELECT m.id, m.session_id, m.content, m.timestamp, "
                        "bm25(message_fts) AS bm25_score "
                        "FROM message_fts "
                        "JOIN messages m ON m.id = message_fts.message_id "
                        "WHERE message_fts MATCH :query "
                        "ORDER BY bm25_score "
                        "LIMIT :limit"
                    ),
                    {"query": query, "limit": limit},
                )
                rows = result.fetchall()
                return [
                    {
                        "message_id": row[0],
                        "source_id": row[0],
                        "source_type": "message",
                        "session_id": row[1],
                        "content": row[2],
                        "timestamp": row[3].isoformat() if row[3] else None,
                        "keyword_score": self._normalize_keyword_score(row[4]),
                        "retrieval_type": "keyword",
                    }
                    for row in rows
                ]
            except Exception:
                like_query = f"%{query}%"
                result = await session.execute(
                    select(MessageRecord)
                    .where(MessageRecord.content.like(like_query))
                    .order_by(MessageRecord.timestamp.desc())
                    .limit(limit)
                )
                rows = list(result.scalars().all())
                return [
                    {
                        "message_id": row.id,
                        "source_id": row.id,
                        "source_type": "message",
                        "session_id": row.session_id,
                        "content": row.content,
                        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                        "keyword_score": 0.4,
                        "retrieval_type": "keyword",
                    }
                    for row in rows
                ]

    @staticmethod
    def _normalize_keyword_score(bm25_score: float | None) -> float:
        if bm25_score is None:
            return 0.0
        return 1.0 / (1.0 + abs(float(bm25_score)))

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> list[SessionRecord]:
        """List sessions ordered by activity time."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SessionRecord)
                .order_by(SessionRecord.last_active.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def delete_session(self, session_id: str):
        """Delete session and messages."""
        async with self.session_factory() as session:
            await session.execute(delete(MessageRecord).where(MessageRecord.session_id == session_id))
            await session.execute(delete(SessionRecord).where(SessionRecord.session_id == session_id))
            await session.commit()

    async def get_global_user_state(self) -> dict:
        """Get global user onboarding state."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(UserRecord).where(UserRecord.user_id == GLOBAL_USER_ID)
            )
            record = result.scalar_one_or_none()
            if not record:
                raise RuntimeError("global user missing after initialization")
            return {
                "user_id": record.user_id,
                "identity_confirmed": record.identity_confirmed,
                "onboarding_prompted": bool(record.onboarding_prompted),
                "display_name": record.display_name,
            }

    async def set_global_user_identity_state(
        self,
        identity_confirmed: bool,
        display_name: Optional[str] = None,
        onboarding_prompted: Optional[bool] = None,
    ):
        """Update global user identity confirmation metadata."""
        values = {
            "identity_confirmed": identity_confirmed,
            "updated_at": datetime.now(),
        }
        if display_name is not None:
            values["display_name"] = display_name
        if onboarding_prompted is not None:
            values["onboarding_prompted"] = onboarding_prompted

        async with self.session_factory() as session:
            await session.execute(
                update(UserRecord)
                .where(UserRecord.user_id == GLOBAL_USER_ID)
                .values(**values)
            )
            await session.commit()

    async def mark_onboarding_prompted(self):
        """Mark first-run onboarding prompt as already shown."""
        async with self.session_factory() as session:
            await session.execute(
                update(UserRecord)
                .where(UserRecord.user_id == GLOBAL_USER_ID)
                .values(
                    onboarding_prompted=True,
                    updated_at=datetime.now(),
                )
            )
            await session.commit()

    async def save_memory_pending(
        self,
        candidate_fact: dict,
        confidence: float,
        source_message_id: Optional[str] = None,
    ) -> str:
        """Create a pending memory candidate."""
        now = datetime.now()
        pending_id = str(uuid.uuid4())
        async with self.session_factory() as session:
            session.add(
                MemoryPendingRecord(
                    id=pending_id,
                    candidate_fact=candidate_fact,
                    confidence=confidence,
                    status="pending",
                    source_message_id=source_message_id,
                    cooldown_until=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
        return pending_id

    async def list_memory_pending(
        self,
        status: Optional[str] = "pending",
        limit: int = 20,
    ) -> list[MemoryPendingRecord]:
        """List pending-memory records by status."""
        async with self.session_factory() as session:
            query = select(MemoryPendingRecord).order_by(MemoryPendingRecord.created_at.desc())
            if status:
                query = query.where(MemoryPendingRecord.status == status)
            result = await session.execute(query.limit(limit))
            return list(result.scalars().all())

    async def update_memory_pending_status(
        self,
        pending_id: str,
        status: str,
        cooldown_until: Optional[datetime] = None,
    ):
        """Transition a pending memory candidate status."""
        values = {
            "status": status,
            "updated_at": datetime.now(),
        }
        if cooldown_until is not None:
            values["cooldown_until"] = cooldown_until

        async with self.session_factory() as session:
            await session.execute(
                update(MemoryPendingRecord)
                .where(MemoryPendingRecord.id == pending_id)
                .values(**values)
            )
            await session.commit()

    async def get_memory_pending(self, pending_id: str) -> Optional[MemoryPendingRecord]:
        """Get one pending-memory record by ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(MemoryPendingRecord).where(MemoryPendingRecord.id == pending_id)
            )
            return result.scalar_one_or_none()

    async def upsert_memory_fact(
        self,
        scope: str,
        fact_type: str,
        fact_key: str,
        fact_value: str,
        confidence: float,
        source_message_id: Optional[str] = None,
        status: str = "active",
    ) -> str:
        """Insert or update an active memory fact by (scope, type, key)."""
        now = datetime.now()
        async with self.session_factory() as session:
            result = await session.execute(
                select(MemoryFactRecord).where(
                    MemoryFactRecord.scope == scope,
                    MemoryFactRecord.fact_type == fact_type,
                    MemoryFactRecord.fact_key == fact_key,
                    MemoryFactRecord.status == "active",
                )
            )
            record = result.scalar_one_or_none()
            if record:
                record.fact_value = fact_value
                record.confidence = confidence
                record.source_message_id = source_message_id
                record.updated_at = now
                await session.commit()
                return record.id

            fact_id = str(uuid.uuid4())
            session.add(
                MemoryFactRecord(
                    id=fact_id,
                    scope=scope,
                    fact_type=fact_type,
                    fact_key=fact_key,
                    fact_value=fact_value,
                    confidence=confidence,
                    status=status,
                    source_message_id=source_message_id,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
            return fact_id

    async def list_memory_facts(
        self,
        scope: Optional[str] = None,
        fact_type: Optional[str] = None,
        status: str = "active",
        limit: int = 100,
    ) -> list[MemoryFactRecord]:
        """List memory facts with optional filters."""
        async with self.session_factory() as session:
            query = select(MemoryFactRecord)
            if status:
                query = query.where(MemoryFactRecord.status == status)
            if scope:
                query = query.where(MemoryFactRecord.scope == scope)
            if fact_type:
                query = query.where(MemoryFactRecord.fact_type == fact_type)
            result = await session.execute(
                query.order_by(MemoryFactRecord.updated_at.desc()).limit(limit)
            )
            return list(result.scalars().all())

    async def save_memory_event(
        self,
        event_type: str,
        payload: dict,
        operation_id: Optional[str] = None,
    ) -> str:
        """Append a memory event for observability with idempotent operation IDs."""
        async with self.session_factory() as session:
            if operation_id:
                result = await session.execute(
                    select(MemoryEventRecord).where(
                        MemoryEventRecord.event_type == event_type,
                        MemoryEventRecord.operation_id == operation_id,
                    )
                )
                existing = result.scalar_one_or_none()
                if existing:
                    return existing.id

            event_id = str(uuid.uuid4())
            session.add(
                MemoryEventRecord(
                    id=event_id,
                    event_type=event_type,
                    operation_id=operation_id,
                    payload=payload,
                    created_at=datetime.now(),
                )
            )
            await session.commit()
            return event_id

    async def list_memory_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[MemoryEventRecord]:
        """List memory events for observability."""
        async with self.session_factory() as session:
            query = select(MemoryEventRecord).order_by(MemoryEventRecord.created_at.desc())
            if event_type:
                query = query.where(MemoryEventRecord.event_type == event_type)
            result = await session.execute(query.limit(limit))
            return list(result.scalars().all())

    async def save_memory_embedding(
        self,
        source_type: str,
        source_id: str,
        content: str,
        embedding: list[float],
    ) -> str:
        """Upsert embedding for a source object."""
        now = datetime.now()
        async with self.session_factory() as session:
            result = await session.execute(
                select(MemoryEmbeddingRecord).where(
                    MemoryEmbeddingRecord.source_type == source_type,
                    MemoryEmbeddingRecord.source_id == source_id,
                )
            )
            record = result.scalar_one_or_none()
            if record:
                record.content = content
                record.embedding = embedding
                record.updated_at = now
                await session.commit()
                return record.id

            embedding_id = str(uuid.uuid4())
            session.add(
                MemoryEmbeddingRecord(
                    id=embedding_id,
                    source_type=source_type,
                    source_id=source_id,
                    content=content,
                    embedding=embedding,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
            return embedding_id

    async def search_memory_embeddings(self, query_embedding: list[float], limit: int = 20) -> list[dict]:
        """Semantic search over stored embeddings using cosine similarity."""
        async with self.session_factory() as session:
            result = await session.execute(select(MemoryEmbeddingRecord))
            items = list(result.scalars().all())

        scored = []
        for item in items:
            score = LocalEmbeddingEngine.cosine_similarity(query_embedding, item.embedding or [])
            if score <= 0:
                continue
            scored.append(
                {
                    "embedding_id": item.id,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "content": item.content,
                    "timestamp": item.updated_at.isoformat() if item.updated_at else None,
                    "semantic_score": score,
                    "retrieval_type": "semantic",
                }
            )

        scored.sort(key=lambda x: x["semantic_score"], reverse=True)
        return scored[:limit]

    async def enqueue_embedding_job(
        self,
        source_type: str,
        source_id: str,
        content: str,
        status: str = "pending",
        retry_count: int = 0,
        next_retry_at: datetime | None = None,
        last_error: str | None = None,
    ) -> str:
        """Create one embedding job."""
        now = datetime.now()
        job_id = str(uuid.uuid4())
        async with self.session_factory() as session:
            session.add(
                EmbeddingJobRecord(
                    id=job_id,
                    source_type=source_type,
                    source_id=source_id,
                    content=content,
                    status=status,
                    retry_count=retry_count,
                    next_retry_at=next_retry_at,
                    last_error=last_error,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
        return job_id

    async def list_embedding_jobs(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[EmbeddingJobRecord]:
        """List embedding jobs by status."""
        async with self.session_factory() as session:
            query = select(EmbeddingJobRecord).order_by(EmbeddingJobRecord.updated_at.asc())
            if status:
                query = query.where(EmbeddingJobRecord.status == status)
            result = await session.execute(query.limit(limit))
            return list(result.scalars().all())

    async def list_due_embedding_jobs(
        self,
        limit: int = 20,
        now: datetime | None = None,
        ignore_schedule: bool = False,
    ) -> list[EmbeddingJobRecord]:
        """List due jobs in pending state."""
        now = now or datetime.now()
        async with self.session_factory() as session:
            query = select(EmbeddingJobRecord).where(EmbeddingJobRecord.status == "pending")
            if not ignore_schedule:
                query = query.where(
                    (EmbeddingJobRecord.next_retry_at.is_(None))
                    | (EmbeddingJobRecord.next_retry_at <= now)
                )
            result = await session.execute(query.order_by(EmbeddingJobRecord.updated_at.asc()).limit(limit))
            return list(result.scalars().all())

    async def update_embedding_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        retry_count: int | None = None,
        next_retry_at: datetime | None = None,
        last_error: str | None = None,
    ):
        """Update embedding job fields."""
        values = {"updated_at": datetime.now()}
        if status is not None:
            values["status"] = status
        if retry_count is not None:
            values["retry_count"] = retry_count
        if next_retry_at is not None or status in {"completed", "dead_letter"}:
            values["next_retry_at"] = next_retry_at
        if last_error is not None:
            values["last_error"] = last_error

        async with self.session_factory() as session:
            await session.execute(
                update(EmbeddingJobRecord)
                .where(EmbeddingJobRecord.id == job_id)
                .values(**values)
            )
            await session.commit()

    async def get_memory_metrics(self) -> dict:
        """Collect lightweight memory pipeline metrics."""
        async with self.session_factory() as session:
            pending_rows = await session.execute(
                select(MemoryPendingRecord.status, func.count())
                .group_by(MemoryPendingRecord.status)
            )
            pending_status = {row[0]: row[1] for row in pending_rows.fetchall()}

            active_facts_count = await session.execute(
                select(func.count()).select_from(MemoryFactRecord).where(MemoryFactRecord.status == "active")
            )
            dead_letter_count = await session.execute(
                select(func.count()).select_from(EmbeddingJobRecord).where(
                    EmbeddingJobRecord.status == "dead_letter"
                )
            )
            retrieval_fail_count = await session.execute(
                select(func.count()).select_from(MemoryEventRecord).where(
                    MemoryEventRecord.event_type == "retrieval_fail"
                )
            )

            return {
                "pending_by_status": pending_status,
                "active_facts": int(active_facts_count.scalar() or 0),
                "dead_letter_jobs": int(dead_letter_count.scalar() or 0),
                "retrieval_fail_count": int(retrieval_fail_count.scalar() or 0),
            }

    async def cleanup(self):
        """Cleanup resources."""
        if self.engine:
            await self.engine.dispose()

"""Kuzu writer - async event/entity/fact persistence."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from loguru import logger

from memory.embeddings import EmbeddingConfig, LocalEmbeddingEngine
from memory.kuzu_store import EMBEDDING_DIM, KuzuStore, execute_sync


class KuzuWriter:
    """
    异步写入器：Event 节点、Entity 节点、Fact 节点、关系边。
    所有 Kuzu 操作是同步的，通过 asyncio.to_thread() 非阻塞化。
    """

    def __init__(self, store: KuzuStore):
        self._store = store
        self._embedder = LocalEmbeddingEngine(EmbeddingConfig(dimension=EMBEDDING_DIM))

    # ── Session ─────────────────────────────────────────────────────────────

    async def ensure_session(self, session_id: str, title: str = "") -> None:
        """确保 Session 节点存在（幂等）。"""
        await asyncio.to_thread(self._upsert_session_sync, session_id, title)

    def _upsert_session_sync(self, session_id: str, title: str) -> None:
        conn = self._store.conn
        conn.execute(
            "MERGE (s:Session {id: $id}) "
            "ON CREATE SET s.title = $title, s.created_at = $ts "
            "ON MATCH SET s.title = $title",
            {"id": session_id, "title": title or session_id, "ts": int(time.time())},
        )

    # ── Event ────────────────────────────────────────────────────────────────

    async def write_event(
        self,
        session_id: str,
        role: str,
        content: str,
        event_id: str | None = None,
        summary: str = "",
        timestamp: int | None = None,
    ) -> str:
        """写入 Event 节点，自动计算 embedding，返回 event_id。"""
        eid = event_id or uuid.uuid4().hex
        ts = timestamp or int(time.time() * 1000)
        embedding = self._embedder.embed(content)
        await asyncio.to_thread(
            self._write_event_sync, eid, session_id, ts, role, content, summary, embedding
        )
        return eid

    def _write_event_sync(
        self,
        eid: str,
        session_id: str,
        ts: int,
        role: str,
        content: str,
        summary: str,
        embedding: list[float],
    ) -> None:
        conn = self._store.conn
        # 写 Event 节点
        conn.execute(
            "CREATE (:Event {id: $id, session_id: $sid, timestamp: $ts, role: $role, "
            "content: $content, summary: $summary, embedding: $emb})",
            {
                "id": eid,
                "sid": session_id,
                "ts": ts,
                "role": role,
                "content": content[:2000],  # 截断超长内容
                "summary": summary[:500],
                "emb": embedding,
            },
        )
        # 连接到 Session
        conn.execute(
            "MATCH (e:Event {id: $eid}), (s:Session {id: $sid}) "
            "CREATE (e)-[:IN_SESSION]->(s)",
            {"eid": eid, "sid": session_id},
        )

    async def link_event_chain(self, prev_event_id: str, next_event_id: str) -> None:
        """创建时间链关系 Event → NEXT → Event。"""
        await asyncio.to_thread(self._link_chain_sync, prev_event_id, next_event_id)

    def _link_chain_sync(self, prev_id: str, next_id: str) -> None:
        conn = self._store.conn
        conn.execute(
            "MATCH (a:Event {id: $prev}), (b:Event {id: $next}) CREATE (a)-[:NEXT]->(b)",
            {"prev": prev_id, "next": next_id},
        )

    # ── Entity ───────────────────────────────────────────────────────────────

    async def upsert_entity(self, name: str, entity_type: str) -> None:
        """插入或更新 Entity 节点（MERGE 幂等）。"""
        await asyncio.to_thread(self._upsert_entity_sync, name, entity_type)

    def _upsert_entity_sync(self, name: str, entity_type: str) -> None:
        conn = self._store.conn
        conn.execute(
            "MERGE (e:Entity {name: $name}) ON CREATE SET e.type = $type ON MATCH SET e.type = $type",
            {"name": name, "type": entity_type},
        )

    async def link_event_to_entity(self, event_id: str, entity_name: str) -> None:
        """创建 Event → MENTIONS → Entity 关系。"""
        await asyncio.to_thread(self._link_event_entity_sync, event_id, entity_name)

    def _link_event_entity_sync(self, event_id: str, entity_name: str) -> None:
        conn = self._store.conn
        conn.execute(
            "MATCH (e:Event {id: $eid}), (en:Entity {name: $ename}) CREATE (e)-[:MENTIONS]->(en)",
            {"eid": event_id, "ename": entity_name},
        )

    async def upsert_entity_relation(
        self,
        from_entity: str,
        to_entity: str,
        rel_type: str,
        weight: float = 1.0,
    ) -> None:
        """确保两个实体之间存在指定类型的关系（先检查再创建）。"""
        await asyncio.to_thread(
            self._upsert_relation_sync, from_entity, to_entity, rel_type, weight
        )

    def _upsert_relation_sync(
        self, from_entity: str, to_entity: str, rel_type: str, weight: float
    ) -> None:
        conn = self._store.conn
        # 检查是否已存在同类型关系
        rows = execute_sync(
            conn,
            "MATCH (a:Entity {name: $from})-[r:RELATED_TO {rel_type: $rt}]->(b:Entity {name: $to}) "
            "RETURN count(r) AS cnt",
            {"from": from_entity, "to": to_entity, "rt": rel_type},
        )
        if rows and rows[0][0] > 0:
            return  # 已存在，跳过
        conn.execute(
            "MATCH (a:Entity {name: $from}), (b:Entity {name: $to}) "
            "CREATE (a)-[:RELATED_TO {rel_type: $rt, weight: $w}]->(b)",
            {"from": from_entity, "to": to_entity, "rt": rel_type, "w": weight},
        )

    # ── Fact ─────────────────────────────────────────────────────────────────

    async def upsert_fact(
        self,
        scope: str,
        key: str,
        value: str,
        confidence: float,
        linked_entity: str | None = None,
    ) -> str:
        """插入或更新 Fact 节点，可选关联到 Entity。返回 fact_id。"""
        fact_id = f"{scope}:{key}"
        await asyncio.to_thread(
            self._upsert_fact_sync, fact_id, scope, key, value, confidence, linked_entity
        )
        return fact_id

    def _upsert_fact_sync(
        self,
        fact_id: str,
        scope: str,
        key: str,
        value: str,
        confidence: float,
        linked_entity: str | None,
    ) -> None:
        conn = self._store.conn
        ts = int(time.time())
        conn.execute(
            "MERGE (f:Fact {id: $id}) "
            "ON CREATE SET f.scope=$scope, f.key=$key, f.value=$value, "
            "              f.confidence=$conf, f.updated_at=$ts "
            "ON MATCH SET  f.value=$value, f.confidence=$conf, f.updated_at=$ts",
            {
                "id": fact_id,
                "scope": scope,
                "key": key,
                "value": value,
                "conf": confidence,
                "ts": ts,
            },
        )
        # 关联到实体
        if linked_entity:
            # 确保关系不重复
            rows = execute_sync(
                conn,
                "MATCH (f:Fact {id: $fid})-[:ABOUT]->(e:Entity {name: $ename}) RETURN count(*) AS cnt",
                {"fid": fact_id, "ename": linked_entity},
            )
            if not rows or rows[0][0] == 0:
                conn.execute(
                    "MATCH (f:Fact {id: $fid}), (e:Entity {name: $ename}) CREATE (f)-[:ABOUT]->(e)",
                    {"fid": fact_id, "ename": linked_entity},
                )

    # ── 批量迁移（从 SQLite memory_facts 迁移）────────────────────────────────

    async def migrate_from_sqlite_facts(self, facts: list[dict[str, Any]]) -> int:
        """将 SQLite memory_facts 数据批量导入 Kuzu，返回成功条数。"""
        count = 0
        for f in facts:
            try:
                await self.upsert_fact(
                    scope=f.get("scope", "user"),
                    key=f.get("fact_key", ""),
                    value=f.get("fact_value", ""),
                    confidence=float(f.get("confidence", 0.7)),
                )
                count += 1
            except Exception as e:
                logger.warning(f"迁移 fact {f.get('fact_key')} 失败: {e}")
        logger.info(f"从 SQLite 迁移了 {count}/{len(facts)} 条记忆事实到 Kuzu")
        return count

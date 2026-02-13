"""Kuzu retriever - semantic / graph / hybrid memory search."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from memory.embeddings import EmbeddingConfig, LocalEmbeddingEngine
from memory.kuzu_store import EMBEDDING_DIM, KuzuStore, execute_sync


@dataclass
class MemoryHit:
    """一条记忆检索结果。"""

    event_id: str = ""
    content: str = ""
    role: str = ""
    score: float = 0.0
    source: str = "semantic"  # semantic | graph | fact
    entities: list[str] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)

    def to_text(self, max_chars: int = 200) -> str:
        text = self.content[:max_chars]
        if self.facts:
            facts_str = ", ".join(f"{f['key']}={f['value']}" for f in self.facts[:3])
            return f"{text} [{facts_str}]"
        return text


class KuzuRetriever:
    """
    三种记忆检索模式：
      - semantic：向量相似度搜索 Event 节点
      - graph：从实体出发图遍历，返回关联 Fact 和关系
      - hybrid（默认）：semantic 初筛 → 取涉及实体 → 图遍历扩展 → 融合排序
    """

    def __init__(self, store: KuzuStore):
        self._store = store
        self._embedder = LocalEmbeddingEngine(EmbeddingConfig(dimension=EMBEDDING_DIM))

    # ── Public async API ─────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> list[MemoryHit]:
        """统一入口，mode: semantic | graph | hybrid。"""
        mode = mode.strip().lower()
        if mode == "semantic":
            return await asyncio.to_thread(self._semantic_search_sync, query, top_k)
        if mode == "graph":
            entities = _extract_entity_hints(query)
            return await asyncio.to_thread(self._graph_search_sync, entities, top_k)
        return await asyncio.to_thread(self._hybrid_search_sync, query, top_k)

    async def get_top_facts(self, top_k: int = 5, min_confidence: float = 0.75) -> list[dict]:
        """获取高置信度 Fact 节点列表（用于 kuzu_prefetch 注入）。"""
        return await asyncio.to_thread(self._get_top_facts_sync, top_k, min_confidence)

    async def keyword_prefetch(self, query: str, top_k: int = 2) -> list[dict]:
        """关键词匹配高置信 Fact，零 LLM，用于 context_builder 预注入。"""
        return await asyncio.to_thread(self._keyword_prefetch_sync, query, top_k)

    # ── Semantic search ──────────────────────────────────────────────────────

    def _semantic_search_sync(self, query: str, top_k: int) -> list[MemoryHit]:
        conn = self._store.conn
        query_emb = self._embedder.embed(query)
        emb_str = _format_embedding(query_emb)
        dim = EMBEDDING_DIM

        rows = execute_sync(
            conn,
            f"""
            MATCH (e:Event)
            WHERE e.embedding IS NOT NULL
            RETURN e.id, e.content, e.role,
                   array_cosine_similarity(e.embedding, CAST({emb_str} AS FLOAT[{dim}])) AS score
            ORDER BY score DESC
            LIMIT {top_k * 2}
            """,
        )

        hits = []
        for row in rows:
            score = float(row[3]) if row[3] is not None else 0.0
            if score < 0.08:
                continue
            hits.append(MemoryHit(
                event_id=row[0],
                content=row[1] or "",
                role=row[2] or "",
                score=score,
                source="semantic",
            ))
        return hits[:top_k]

    # ── Graph search ─────────────────────────────────────────────────────────

    def _graph_search_sync(self, entity_names: list[str], top_k: int) -> list[MemoryHit]:
        if not entity_names:
            return []

        conn = self._store.conn
        hits = []

        for entity_name in entity_names[:3]:  # 最多遍历 3 个实体
            # 1. 从实体出发，找关联 Fact
            fact_rows = execute_sync(
                conn,
                """
                MATCH (f:Fact)-[:ABOUT]->(e:Entity {name: $name})
                RETURN f.key, f.value, f.confidence, f.scope
                ORDER BY f.confidence DESC
                LIMIT 5
                """,
                {"name": entity_name},
            )
            facts = [
                {"key": r[0], "value": r[1], "confidence": float(r[2]), "scope": r[3]}
                for r in fact_rows
            ]

            # 2. 从实体出发，找 2 跳内的相关实体
            related_rows = execute_sync(
                conn,
                """
                MATCH (a:Entity {name: $name})-[:RELATED_TO]->(b:Entity)
                OPTIONAL MATCH (f:Fact)-[:ABOUT]->(b)
                RETURN b.name, b.type, f.key, f.value
                LIMIT 8
                """,
                {"name": entity_name},
            )

            # 3. 找 mention 这个实体的最近事件
            event_rows = execute_sync(
                conn,
                """
                MATCH (e:Event)-[:MENTIONS]->(en:Entity {name: $name})
                RETURN e.id, e.content, e.role, e.timestamp
                ORDER BY e.timestamp DESC
                LIMIT 3
                """,
                {"name": entity_name},
            )

            if facts or event_rows:
                if event_rows:
                    for er in event_rows:
                        hits.append(MemoryHit(
                            event_id=er[0],
                            content=er[1] or "",
                            role=er[2] or "",
                            score=0.8,
                            source="graph",
                            entities=[entity_name],
                            facts=facts,
                        ))
                else:
                    # 只有 Fact，无对应 Event，构造虚拟 hit
                    facts_text = "; ".join(f"{f['key']}: {f['value']}" for f in facts)
                    hits.append(MemoryHit(
                        event_id=f"fact:{entity_name}",
                        content=f"关于 {entity_name}: {facts_text}",
                        role="memory",
                        score=0.75,
                        source="graph",
                        entities=[entity_name],
                        facts=facts,
                    ))

        return hits[:top_k]

    # ── Hybrid search ─────────────────────────────────────────────────────────

    def _hybrid_search_sync(self, query: str, top_k: int) -> list[MemoryHit]:
        # Step 1: 语义检索
        semantic_hits = self._semantic_search_sync(query, top_k)

        # Step 2: 从语义命中中提取实体
        entity_names: list[str] = []
        for hit in semantic_hits:
            conn = self._store.conn
            rows = execute_sync(
                conn,
                "MATCH (e:Event {id: $eid})-[:MENTIONS]->(en:Entity) RETURN en.name",
                {"eid": hit.event_id},
            )
            for r in rows:
                if r[0] not in entity_names:
                    entity_names.append(r[0])

        # Step 3: 提取 query 中的实体线索
        hint_entities = _extract_entity_hints(query)
        for e in hint_entities:
            if e not in entity_names:
                entity_names.append(e)

        # Step 4: 图遍历扩展
        graph_hits = self._graph_search_sync(entity_names, top_k) if entity_names else []

        # Step 5: Event 内容关键词搜索（兜底，弥补向量弱语义的缺陷）
        keyword_hits = self._event_keyword_search_sync(query, top_k)

        # Step 6: 融合去重，按 score 排序
        seen_ids = set()
        merged: list[MemoryHit] = []
        for h in semantic_hits + graph_hits + keyword_hits:
            if h.event_id not in seen_ids:
                seen_ids.add(h.event_id)
                merged.append(h)

        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:top_k]

    def _event_keyword_search_sync(self, query: str, top_k: int) -> list[MemoryHit]:
        """在 Event.content 上做关键词 CONTAINS 搜索，弥补向量弱语义。"""
        conn = self._store.conn
        # 提取 2 字以上中文词 + 英文词
        keywords = [w for w in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", query)][:4]
        if not keywords:
            return []

        hits: list[MemoryHit] = []
        seen_ids: set[str] = set()
        for kw in keywords:
            rows = execute_sync(
                conn,
                """
                MATCH (e:Event)
                WHERE e.content CONTAINS $kw AND e.role IN ['user', 'assistant']
                RETURN e.id, e.content, e.role
                ORDER BY e.timestamp DESC
                LIMIT 5
                """,
                {"kw": kw},
            )
            for row in rows:
                eid = row[0]
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    hits.append(MemoryHit(
                        event_id=eid,
                        content=row[1] or "",
                        role=row[2] or "",
                        score=0.55,
                        source="keyword",
                    ))
        return hits[:top_k]

    # ── Fact prefetch ─────────────────────────────────────────────────────────

    def _get_top_facts_sync(self, top_k: int, min_confidence: float) -> list[dict]:
        conn = self._store.conn
        rows = execute_sync(
            conn,
            """
            MATCH (f:Fact)
            WHERE f.confidence >= $min_conf
            RETURN f.scope, f.key, f.value, f.confidence
            ORDER BY f.confidence DESC, f.updated_at DESC
            LIMIT $k
            """,
            {"min_conf": min_confidence, "k": top_k},
        )
        return [
            {"scope": r[0], "key": r[1], "value": r[2], "confidence": float(r[3])}
            for r in rows
        ]

    def _keyword_prefetch_sync(self, query: str, top_k: int) -> list[dict]:
        """简单关键词匹配 Fact.key 或 Fact.value，不调 LLM。"""
        conn = self._store.conn
        # 提取 2 字以上的词作为关键词
        keywords = [w for w in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", query)][:5]
        if not keywords:
            return self._get_top_facts_sync(top_k, 0.75)

        results: list[dict] = []
        seen_keys: set[str] = set()
        for kw in keywords:
            rows = execute_sync(
                conn,
                """
                MATCH (f:Fact)
                WHERE f.key CONTAINS $kw OR f.value CONTAINS $kw
                RETURN f.scope, f.key, f.value, f.confidence
                ORDER BY f.confidence DESC
                LIMIT 3
                """,
                {"kw": kw},
            )
            for r in rows:
                fact_key = f"{r[0]}:{r[1]}"
                if fact_key not in seen_keys:
                    seen_keys.add(fact_key)
                    results.append({
                        "scope": r[0], "key": r[1], "value": r[2], "confidence": float(r[3])
                    })

        return results[:top_k]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _format_embedding(emb: list[float]) -> str:
    """格式化 embedding 为 Kuzu CAST 表达式所需的字面量字符串。"""
    return "[" + ", ".join(f"{v:.6f}" for v in emb) + "]"


def _extract_entity_hints(query: str) -> list[str]:
    """
    从 query 中提取可能的实体名（简单启发式：2字中文二元组 + 英文大写词）。
    使用固定 2 字切片避免贪婪匹配吞掉短实体名。
    不调用 LLM，纯规则。
    """
    # 中文：固定 2 字切片（不贪婪，确保能匹配人名/技术词等短名词）
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", query)
    bigrams = [chinese_chars[i] + chinese_chars[i + 1] for i in range(len(chinese_chars) - 1)]
    # 英文：大写开头或全大写
    english = re.findall(r"[A-Z][a-zA-Z]{1,}|[A-Z]{2,}|[a-z]{3,}", query)
    # 去重，取前 5 个
    seen: set[str] = set()
    result: list[str] = []
    for e in bigrams + english:
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result[:5]

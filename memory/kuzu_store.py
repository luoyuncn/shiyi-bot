"""Kuzu graph database - schema initialization and connection management."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import kuzu
from loguru import logger

# embedding 维度，与 LocalEmbeddingEngine 保持一致
EMBEDDING_DIM = 128


class KuzuStore:
    """
    Kuzu 图数据库初始化与连接管理。
    Kuzu 是同步 API，所有阻塞操作通过 asyncio.to_thread() 包装。
    """

    def __init__(self, db_path: str = "data/kuzu"):
        self.db_path = db_path
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    def _init_sync(self) -> None:
        """同步初始化：建 DB、建表。"""
        # 只确保父目录存在，不预建 db_path 本身（Kuzu 自行创建 DB 目录）
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(self.db_path)
        self._conn = kuzu.Connection(self._db)
        self._create_schema()
        logger.info(f"KuzuStore 初始化完成: {self.db_path}")

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_sync)

    @property
    def conn(self) -> kuzu.Connection:
        if self._conn is None:
            raise RuntimeError("KuzuStore 尚未初始化，请先调用 initialize()")
        return self._conn

    def _create_schema(self) -> None:
        """建表（幂等，已存在则跳过）。"""
        c = self._conn
        assert c is not None

        # ── 节点表 ──────────────────────────────────────────────────────────
        _exec_if_not_exists(c, "Session", f"""
            CREATE NODE TABLE Session(
                id       STRING,
                title    STRING,
                created_at INT64,
                PRIMARY KEY(id)
            )
        """)

        _exec_if_not_exists(c, "Event", f"""
            CREATE NODE TABLE Event(
                id         STRING,
                session_id STRING,
                timestamp  INT64,
                role       STRING,
                content    STRING,
                summary    STRING,
                embedding  FLOAT[{EMBEDDING_DIM}],
                PRIMARY KEY(id)
            )
        """)

        _exec_if_not_exists(c, "Entity", """
            CREATE NODE TABLE Entity(
                name STRING,
                type STRING,
                PRIMARY KEY(name)
            )
        """)

        _exec_if_not_exists(c, "Fact", """
            CREATE NODE TABLE Fact(
                id         STRING,
                scope      STRING,
                key        STRING,
                value      STRING,
                confidence FLOAT,
                updated_at INT64,
                PRIMARY KEY(id)
            )
        """)

        # ── 关系表 ──────────────────────────────────────────────────────────
        _exec_rel_if_not_exists(c, "IN_SESSION", """
            CREATE REL TABLE IN_SESSION(FROM Event TO Session)
        """)

        _exec_rel_if_not_exists(c, "NEXT", """
            CREATE REL TABLE NEXT(FROM Event TO Event)
        """)

        _exec_rel_if_not_exists(c, "MENTIONS", """
            CREATE REL TABLE MENTIONS(FROM Event TO Entity)
        """)

        _exec_rel_if_not_exists(c, "RELATED_TO", """
            CREATE REL TABLE RELATED_TO(FROM Entity TO Entity, rel_type STRING, weight FLOAT)
        """)

        _exec_rel_if_not_exists(c, "ABOUT", """
            CREATE REL TABLE ABOUT(FROM Fact TO Entity)
        """)

    async def cleanup(self) -> None:
        """关闭连接（Kuzu 连接无显式 close，GC 自动处理）。"""
        self._conn = None
        self._db = None


def _exec_if_not_exists(conn: kuzu.Connection, table_name: str, ddl: str) -> None:
    """尝试建节点表，已存在则静默跳过。"""
    try:
        conn.execute(ddl.strip())
    except Exception as e:
        if "already exists" in str(e).lower() or table_name.lower() in str(e).lower():
            pass  # 表已存在，正常
        else:
            logger.warning(f"建表 {table_name} 时出现非预期错误: {e}")


def _exec_rel_if_not_exists(conn: kuzu.Connection, rel_name: str, ddl: str) -> None:
    """尝试建关系表，已存在则静默跳过。"""
    try:
        conn.execute(ddl.strip())
    except Exception as e:
        if "already exists" in str(e).lower() or rel_name.lower() in str(e).lower():
            pass
        else:
            logger.warning(f"建关系表 {rel_name} 时出现非预期错误: {e}")


def execute_sync(conn: kuzu.Connection, query: str, params: dict[str, Any] | None = None) -> list[list]:
    """执行查询并返回所有行（同步）。"""
    result = conn.execute(query, params or {})
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows

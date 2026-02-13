"""Kuzu 全局单例管理器 - store / writer / retriever 生命周期管理。"""

from __future__ import annotations

from loguru import logger

from memory.kuzu_store import KuzuStore
from memory.kuzu_writer import KuzuWriter
from memory.kuzu_retriever import KuzuRetriever

_store: KuzuStore | None = None
_writer: KuzuWriter | None = None
_retriever: KuzuRetriever | None = None


async def initialize(db_path: str = "data/kuzu") -> None:
    """初始化 Kuzu 全局单例，应在应用启动时调用一次。"""
    global _store, _writer, _retriever

    _store = KuzuStore(db_path=db_path)
    await _store.initialize()

    _writer = KuzuWriter(_store)
    _retriever = KuzuRetriever(_store)

    logger.info(f"Kuzu 记忆层初始化完成: {db_path}")


async def cleanup() -> None:
    """清理 Kuzu 资源，应在应用关闭时调用。"""
    global _store, _writer, _retriever
    if _store:
        await _store.cleanup()
    _store = None
    _writer = None
    _retriever = None


def get_store() -> KuzuStore | None:
    return _store


def get_writer() -> KuzuWriter | None:
    return _writer


def get_retriever() -> KuzuRetriever | None:
    return _retriever


def is_initialized() -> bool:
    return _store is not None

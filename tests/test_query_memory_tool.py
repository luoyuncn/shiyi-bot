from __future__ import annotations

import pytest

from tools.builtin.query_memory import Tool


class _FakeHit:
    def __init__(self, text: str, score: float):
        self._text = text
        self.score = score

    def to_text(self, max_chars: int = 200) -> str:
        return self._text[:max_chars]


class _FakeRetriever:
    async def search(self, query: str, mode: str, top_k: int):
        return [_FakeHit("user:identity:display_name:腿哥", 0.83)]


@pytest.mark.asyncio
async def test_query_memory_output_uses_plain_header_without_bracket_markup(monkeypatch):
    import memory.kuzu_manager as kuzu_manager

    monkeypatch.setattr(kuzu_manager, "get_retriever", lambda: _FakeRetriever())

    tool = Tool()
    result = await tool.execute(query="腿哥今天吃了什么 饮食记录", mode="hybrid", top_k=5)

    first_line = result.splitlines()[0]
    assert first_line.startswith("记忆查询结果")
    assert "[" not in first_line
    assert "]" not in first_line

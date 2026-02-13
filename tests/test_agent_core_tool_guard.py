from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agent_orchestration import IntentRoute, IntentType
from core.agent_core import AgentCore
from tools.registry import ToolRegistry


class _MockConfig:
    llm = {
        "api_base": "https://example.com/v1",
        "api_key": "test-key",
        "model": "test-model",
        "system_prompt": "system model={model} time={datetime}",
        "temperature": 0.0,
        "max_tokens": 200,
    }
    memory = SimpleNamespace(memory_root="data/memory")
    agent = SimpleNamespace(
        max_tool_iterations=3,
        context_budget={"total_tokens": 6000, "system_reserved_tokens": 800},
        complexity_detector={"enabled": False},
    )


@pytest.mark.asyncio
async def test_memory_intent_only_exposes_query_memory_tool(monkeypatch):
    agent = AgentCore(_MockConfig())

    async def fake_route_async(messages):
        return IntentRoute(
            intent=IntentType.MEMORY,
            user_query="你还记得我今天吃了什么吗",
            reason="test",
            confidence=0.9,
        )

    captured_tools = []

    async def fake_stream(messages, tools=None):
        captured_tools.append(tools)
        yield {"type": "text_delta", "content": "我来查一下记忆。"}

    monkeypatch.setattr(agent.router, "route_async", fake_route_async)
    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [
            {"function": {"name": "query_memory"}},
            {"function": {"name": "write_file"}},
            {"function": {"name": "edit_file"}},
        ]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)

    _ = [
        e
        async for e in agent.process_message_stream(
            [{"role": "user", "content": "你还记得我今天吃了什么吗"}],
            enable_tools=True,
        )
    ]

    tools = captured_tools[-1] or []
    names = sorted(t["function"]["name"] for t in tools)
    assert names == ["query_memory"]


@pytest.mark.asyncio
async def test_memory_intent_blocks_write_file_even_if_model_calls_it(monkeypatch):
    agent = AgentCore(_MockConfig())

    async def fake_route_async(messages):
        return IntentRoute(
            intent=IntentType.MEMORY,
            user_query="你还记得我今天吃了什么吗",
            reason="test",
            confidence=0.9,
        )

    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "name": "write_file",
                        "arguments": '{"path":"data/memory/shared/User.md","content":"bad","mode":"overwrite"}',
                    }
                ],
            }
        else:
            yield {"type": "text_delta", "content": "不会写入文件。"}

    executed = []

    async def fake_execute(tool_name, parameters):
        executed.append(tool_name)
        return "ok"

    monkeypatch.setattr(agent.router, "route_async", fake_route_async)
    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [
            {"function": {"name": "query_memory"}},
            {"function": {"name": "write_file"}},
        ]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)
    monkeypatch.setattr(agent, "_execute_tool", fake_execute)

    tool_results = []
    async for event in agent.process_message_stream(
        [{"role": "user", "content": "你还记得我今天吃了什么吗"}],
        enable_tools=True,
    ):
        if event["type"] == "tool_result":
            tool_results.append(event["result"])

    assert executed == []
    assert any("策略限制" in r for r in tool_results)

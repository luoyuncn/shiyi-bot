import asyncio
from types import SimpleNamespace

import pytest

from core.agent_core import AgentCore
from tools.registry import ToolRegistry


class MockConfig:
    llm = {
        "api_base": "https://example.com/v1",
        "api_key": "test-key",
        "model": "test-model",
        "system_prompt": "system model={model} time={datetime}",
        "temperature": 0.0,
        "max_tokens": 200,
    }
    agent = SimpleNamespace(max_tool_iterations=4)


@pytest.mark.asyncio
async def test_agent_core_memory_intent_disables_tools(monkeypatch):
    agent = AgentCore(MockConfig())

    captured_tools = []

    async def fake_stream(messages, tools=None):
        captured_tools.append(tools)
        yield {"type": "text_delta", "content": "记得"}

    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [
            {"function": {"name": "read_file"}},
            {"function": {"name": "search_web"}},
            {"function": {"name": "super_search"}},
        ]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)

    events = [
        event
        async for event in agent.process_message_stream(
            [{"role": "user", "content": "你还记得我今天中午吃的啥吗"}],
            enable_tools=True,
        )
    ]

    assert any(e["type"] == "text" for e in events)
    assert captured_tools[-1] is None


@pytest.mark.asyncio
async def test_agent_core_realtime_intent_uses_search_allowlist(monkeypatch):
    agent = AgentCore(MockConfig())

    captured_tools = []

    async def fake_stream(messages, tools=None):
        captured_tools.append(tools)
        yield {"type": "text_delta", "content": "已查询"}

    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [
            {"function": {"name": "read_file"}},
            {"function": {"name": "search_web"}},
            {"function": {"name": "super_search"}},
            {"function": {"name": "edit_file"}},
        ]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)

    _ = [
        event
        async for event in agent.process_message_stream(
            [{"role": "user", "content": "帮我查一下上海浦东2月成交数据"}],
            enable_tools=True,
        )
    ]

    tools = captured_tools[-1]
    names = sorted(item["function"]["name"] for item in tools)
    assert names == ["search_web", "super_search"]


@pytest.mark.asyncio
async def test_agent_core_housing_price_query_uses_search_allowlist(monkeypatch):
    agent = AgentCore(MockConfig())

    captured_tools = []

    async def fake_stream(messages, tools=None):
        captured_tools.append(tools)
        yield {"type": "text_delta", "content": "已查询"}

    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [
            {"function": {"name": "read_file"}},
            {"function": {"name": "search_web"}},
            {"function": {"name": "super_search"}},
        ]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)

    _ = [
        event
        async for event in agent.process_message_stream(
            [{"role": "user", "content": "2026年上海房价"}],
            enable_tools=True,
        )
    ]

    tools = captured_tools[-1]
    names = sorted(item["function"]["name"] for item in tools)
    assert names == ["search_web", "super_search"]


@pytest.mark.asyncio
async def test_agent_core_uses_llm_intent_classification_not_keyword_only(monkeypatch):
    agent = AgentCore(MockConfig())

    captured_tools = []

    async def fake_route_llm(
        messages,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        return {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "id": "tc-intent-1",
                    "name": "classify_intent",
                    "arguments": '{"intent":"realtime_info","reason":"需要实时外部数据","confidence":0.91}',
                }
            ],
        }

    async def fake_stream(messages, tools=None):
        captured_tools.append(tools)
        yield {"type": "text_delta", "content": "已查询"}

    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [
            {"function": {"name": "read_file"}},
            {"function": {"name": "search_web"}},
            {"function": {"name": "super_search"}},
        ]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools", fake_route_llm)
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)

    _ = [
        event
        async for event in agent.process_message_stream(
            [{"role": "user", "content": "给我看下明年走势"}],
            enable_tools=True,
        )
    ]

    tools = captured_tools[-1]
    names = sorted(item["function"]["name"] for item in tools)
    assert names == ["search_web", "super_search"]


@pytest.mark.asyncio
async def test_agent_core_appends_evidence_block_when_tools_used(monkeypatch):
    agent = AgentCore(MockConfig())

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
                        "name": "search_web",
                        "arguments": '{"query":"上海浦东2月成交数据"}',
                    }
                ],
            }
        else:
            yield {"type": "text_delta", "content": "上海浦东2月成交量正在回升。"}

    async def fake_execute(tool_name, parameters):
        assert tool_name == "search_web"
        return "来源A：浦东2月成交量环比提升。"

    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [{"function": {"name": "search_web"}}]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)
    monkeypatch.setattr(agent, "_execute_tool", fake_execute)

    text_chunks = []
    async for event in agent.process_message_stream(
        [{"role": "user", "content": "帮我查一下上海浦东2月成交数据"}],
        enable_tools=True,
    ):
        if event["type"] == "text":
            text_chunks.append(event["content"])

    full_text = "".join(text_chunks)
    assert "[Evidence]" in full_text
    assert "search_web" in full_text

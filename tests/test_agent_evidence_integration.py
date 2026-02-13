from types import SimpleNamespace

import pytest

from core.agent_core import AgentCore
from tools.registry import ToolRegistry


class BudgetedConfig:
    llm = {
        "api_base": "https://example.com/v1",
        "api_key": "test-key",
        "model": "test-model",
        "system_prompt": "system model={model} time={datetime}",
        "temperature": 0.0,
        "max_tokens": 200,
    }
    agent = {
        "max_tool_iterations": 5,
        "orchestration": {
            "enabled": True,
            "max_plan_steps": 3,
            "force_evidence_section": True,
            "tool_budget_by_intent": {
                "chat": 0,
                "memory": 0,
                "realtime_info": 1,
                "workspace_action": 3,
            },
        },
    }


@pytest.mark.asyncio
async def test_realtime_tool_budget_from_config_limits_llm_iterations(monkeypatch):
    agent = AgentCore(BudgetedConfig())

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
            yield {"type": "text_delta", "content": "不应执行到这里"}

    async def fake_execute(tool_name, parameters):
        return "来源A"

    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [{"function": {"name": "search_web"}}]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)
    monkeypatch.setattr(agent, "_execute_tool", fake_execute)

    _ = [
        event
        async for event in agent.process_message_stream(
            [{"role": "user", "content": "帮我查一下上海浦东2月成交数据"}],
            enable_tools=True,
        )
    ]

    assert call_count == 1


@pytest.mark.asyncio
async def test_memory_query_keeps_tools_off(monkeypatch):
    agent = AgentCore(BudgetedConfig())

    captured_tools = []

    async def fake_stream(messages, tools=None):
        captured_tools.append(tools)
        yield {"type": "text_delta", "content": "你中午吃的是牛肉面"}

    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [
            {"function": {"name": "search_web"}},
            {"function": {"name": "super_search"}},
        ]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)

    _ = [
        event
        async for event in agent.process_message_stream(
            [{"role": "user", "content": "你还记得我今天中午吃的啥吗"}],
            enable_tools=True,
        )
    ]

    assert captured_tools[-1] is None


@pytest.mark.asyncio
async def test_tool_answer_contains_evidence_summary(monkeypatch):
    agent = AgentCore(BudgetedConfig())

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
            yield {"type": "text_delta", "content": "核心结论：成交量回升。"}

    async def fake_execute(tool_name, parameters):
        return "来源A：浦东2月成交环比提升。"

    monkeypatch.setattr(
        ToolRegistry,
        "get_tool_definitions",
        classmethod(lambda cls: [{"function": {"name": "search_web"}}]),
    )
    monkeypatch.setattr(agent.llm_engine, "chat_with_tools_stream", fake_stream)
    monkeypatch.setattr(agent, "_execute_tool", fake_execute)

    text_parts = []
    async for event in agent.process_message_stream(
        [{"role": "user", "content": "帮我查一下上海浦东2月成交数据"}],
        enable_tools=True,
    ):
        if event["type"] == "text":
            text_parts.append(event["content"])

    final_text = "".join(text_parts)
    assert "[Evidence]" in final_text
    assert "search_web" in final_text

"""Tests for agent core"""
import pytest
from core.agent_core import AgentCore
from pydantic import BaseModel


class MockConfig(BaseModel):
    """Mock config for testing"""
    llm: dict = {
        "api_base": "https://example.com/v1",
        "api_key": "test-key",
        "model": "test-model",
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_tokens": 100
    }


@pytest.mark.asyncio
async def test_agent_core_initialization():
    """Test agent core initialization"""
    config = MockConfig()
    agent = AgentCore(config)

    # Just test initialization, don't actually call API
    assert agent.llm_engine is not None
    assert agent.config == config

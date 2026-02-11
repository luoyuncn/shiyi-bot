"""Tests for tool system"""
import pytest
from tools.base import BaseTool, ToolDefinition, ToolParameter
from tools.registry import ToolRegistry


class DummyTool(BaseTool):
    """Dummy tool for testing"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dummy_tool",
            description="A dummy tool",
            parameters={
                "text": ToolParameter(
                    type="string",
                    description="Text input",
                    required=True
                )
            }
        )

    async def execute(self, text: str) -> str:
        return f"Processed: {text}"


@pytest.mark.asyncio
async def test_tool_execution():
    """Test tool execution"""
    tool = DummyTool()

    result = await tool.run(text="hello")
    assert result == "Processed: hello"


def test_tool_definition_to_openai_format():
    """Test OpenAI format conversion"""
    tool = DummyTool()
    openai_format = tool.definition.to_openai_format()

    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "dummy_tool"
    assert "text" in openai_format["function"]["parameters"]["properties"]
    assert "text" in openai_format["function"]["parameters"]["required"]


def test_tool_registry():
    """Test tool registry"""
    ToolRegistry._tools.clear()

    tool = DummyTool()
    ToolRegistry.register(tool)

    retrieved = ToolRegistry.get_tool("dummy_tool")
    assert retrieved is not None
    assert retrieved.definition.name == "dummy_tool"

    tools = ToolRegistry.list_tools()
    assert "dummy_tool" in tools

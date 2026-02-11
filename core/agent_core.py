"""Agent core - LLM reasoning and tool calling"""
import asyncio
from typing import AsyncIterator, Optional
from loguru import logger

from engines.llm.openai_compatible_engine import OpenAICompatibleEngine
from tools.registry import ToolRegistry


class AgentCore:
    """Main agent core - LLM reasoning + tool calling"""

    def __init__(self, config):
        self.config = config

        # LLM engine
        self.llm_engine = OpenAICompatibleEngine(
            api_base=config.llm["api_base"],
            api_key=config.llm["api_key"],
            model=config.llm["model"],
            system_prompt=config.llm["system_prompt"],
            temperature=config.llm.get("temperature", 0.7),
            max_tokens=config.llm.get("max_tokens", 2000)
        )

    async def initialize(self):
        """Initialize agent core"""
        await self.llm_engine.initialize()
        logger.info("AgentCore 初始化完成")

    async def process_message_stream(
        self,
        messages: list[dict],
        enable_tools: bool = True
    ) -> AsyncIterator[dict]:
        """
        Process message with streaming response

        Args:
            messages: Conversation messages
            enable_tools: Whether to enable tool calling

        Yields:
            Event dict: {"type": "text", "content": "..."}
                       {"type": "tool_call", "tool": "...", "args": {...}}
                       {"type": "tool_result", "tool": "...", "result": "..."}
                       {"type": "done"}
                       {"type": "error", "error": "..."}
        """
        try:
            # Get tool definitions if enabled
            tools = None
            if enable_tools:
                tool_defs = ToolRegistry.get_tool_definitions()
                if tool_defs:
                    tools = tool_defs

            # Stream LLM response
            full_response = ""
            tool_calls = []

            async for token in self.llm_engine.chat_stream(
                messages[-1]["content"] if messages else ""
            ):
                full_response += token
                yield {"type": "text", "content": token}

            # For now, just return text response
            # TODO: Implement actual tool calling when LLM engine supports it
            yield {"type": "done"}

        except Exception as e:
            logger.error(f"AgentCore处理失败: {e}")
            yield {"type": "error", "error": str(e)}

    async def cleanup(self):
        """Cleanup resources"""
        await self.llm_engine.cleanup()

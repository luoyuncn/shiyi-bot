"""Agent core - LLM reasoning and tool calling"""
import asyncio
import json
from datetime import datetime
from typing import AsyncIterator, Optional, Any
from loguru import logger

from engines.llm.openai_compatible_engine import OpenAICompatibleEngine
from tools.registry import ToolRegistry


class AgentCore:
    """Main agent core - LLM reasoning + tool calling"""

    def __init__(self, config):
        self.config = config

        # LLM engine
        self.llm_engine = OpenAICompatibleEngine(
            api_base=config.llm.api_base,
            api_key=config.llm.api_key,
            model=config.llm.model,
            system_prompt=config.llm.system_prompt,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        self.max_tool_iterations = 5  # 防止无限循环

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
        Process message with streaming response and tool calling

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
                    logger.debug(f"已加载 {len(tools)} 个工具定义")

            # Build system prompt with dynamic model name and current time
            now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            system_content = self.config.llm.system_prompt.format(
                model=self.config.llm.model,
                datetime=now
            )
            full_messages = [
                {"role": "system", "content": system_content},
                *messages
            ]

            # Tool calling loop
            iteration = 0
            while iteration < self.max_tool_iterations:
                iteration += 1

                # Call LLM with tool support
                response = await self.llm_engine.chat_with_tools(
                    full_messages,
                    tools=tools if enable_tools else None
                )

                # Handle response type
                if response["type"] == "text":
                    # Text response - yield and finish
                    content = response["content"]
                    yield {"type": "text", "content": content}

                    # Add to messages for context
                    full_messages.append({"role": "assistant", "content": content})
                    break

                elif response["type"] == "tool_calls":
                    # Tool calls - execute them
                    tool_calls = response["tool_calls"]

                    # Add assistant message with tool calls
                    full_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"]
                                }
                            }
                            for tc in tool_calls
                        ]
                    })

                    # Execute each tool
                    for tool_call in tool_calls:
                        tool_name = tool_call["name"]
                        tool_args_str = tool_call["arguments"]
                        tool_id = tool_call["id"]

                        try:
                            # Parse arguments
                            tool_args = json.loads(tool_args_str)

                            # Yield tool call event
                            yield {
                                "type": "tool_call",
                                "tool": tool_name,
                                "args": tool_args
                            }

                            # Execute tool
                            result = await self._execute_tool(tool_name, tool_args)

                            # Yield tool result event
                            yield {
                                "type": "tool_result",
                                "tool": tool_name,
                                "result": str(result)
                            }

                            # Add tool result to messages
                            full_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": str(result)
                            })

                        except Exception as e:
                            error_msg = f"工具执行失败: {e}"
                            logger.error(f"执行工具 {tool_name} 失败: {e}")

                            yield {
                                "type": "tool_result",
                                "tool": tool_name,
                                "result": error_msg
                            }

                            full_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": error_msg
                            })

                    # Continue loop to get LLM's response with tool results

            else:
                # Exhausted max iterations without a text response — force a final answer
                logger.warning(f"工具调用达到最大次数 ({self.max_tool_iterations})，强制获取最终回复")
                full_messages.append({
                    "role": "user",
                    "content": "请根据以上工具查询结果，直接给出最终回答，不要再调用任何工具。"
                })
                final = await self.llm_engine.chat_with_tools(full_messages, tools=None)
                if final["type"] == "text":
                    yield {"type": "text", "content": final["content"]}

            yield {"type": "done"}

        except Exception as e:
            logger.error(f"AgentCore处理失败: {e}")
            yield {"type": "error", "error": str(e)}

    async def _execute_tool(self, tool_name: str, parameters: dict) -> Any:
        """
        Execute a tool by name

        Args:
            tool_name: Tool name
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        tool = ToolRegistry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"工具不存在: {tool_name}")

        logger.debug(f"执行工具: {tool_name} with {parameters}")
        result = await tool.execute(**parameters)
        return result

    async def cleanup(self):
        """Cleanup resources"""
        await self.llm_engine.cleanup()

    async def process_with_sub_agent(
        self,
        agent_name: str,
        task: str,
        context: dict
    ) -> AsyncIterator[dict]:
        """Delegate task to a sub-agent"""
        from agents.registry import AgentRegistry

        agent = AgentRegistry.get_agent(agent_name)
        if not agent:
            yield {"type": "error", "error": f"子Agent不存在: {agent_name}"}
            return

        yield {"type": "sub_agent_start", "agent": agent_name, "task": task}

        async for event in agent.execute(task, context):
            yield event

        yield {"type": "sub_agent_done", "agent": agent_name}

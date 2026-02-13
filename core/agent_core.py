"""Agent core - LLM reasoning and tool calling"""
import json
from datetime import datetime
from typing import AsyncIterator, Any
from types import SimpleNamespace
from loguru import logger

from engines.llm.openai_compatible_engine import OpenAICompatibleEngine
from tools.registry import ToolRegistry


class AgentCore:
    """Main agent core - LLM reasoning + tool calling"""

    def __init__(self, config):
        self.config = config
        self.llm_config = self._normalize_llm_config(config.llm)

        # LLM engine
        # LLM 引擎
        self.llm_engine = OpenAICompatibleEngine(
            api_base=self.llm_config.api_base,
            api_key=self.llm_config.api_key,
            model=self.llm_config.model,
            system_prompt=self.llm_config.system_prompt,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens
        )
        self.max_tool_iterations = 5  # 防止无限循环

    @staticmethod
    def _normalize_llm_config(llm_config: Any) -> Any:
        """Allow both object-style and dict-style llm configs."""
        if isinstance(llm_config, dict):
            defaults = {
                "temperature": 0.7,
                "max_tokens": 500,
                "system_prompt": "",
            }
            defaults.update(llm_config)
            return SimpleNamespace(**defaults)
        return llm_config

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
            # 如果启用工具，则加载工具定义
            tools = None
            if enable_tools:
                tool_defs = ToolRegistry.get_tool_definitions()
                if tool_defs:
                    tools = tool_defs
                    logger.debug(f"已加载 {len(tools)} 个工具定义")

            # Build system prompt with dynamic model name and current time
            # 使用动态模型名和当前时间构建系统提示词
            now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            system_content = self.llm_config.system_prompt.format(
                model=self.llm_config.model,
                datetime=now
            )
            full_messages = [
                {"role": "system", "content": system_content},
                *messages
            ]

            # Tool calling loop
            # 工具调用循环
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            iteration = 0
            while iteration < self.max_tool_iterations:
                iteration += 1

                # Call LLM with tool support (streaming)
                # 调用支持工具的 LLM（流式）
                stream = self.llm_engine.chat_with_tools_stream(
                    full_messages,
                    tools=tools if enable_tools else None
                )

                # Collect full response for history
                # 收集完整回复用于历史记录
                current_content = ""
                tool_calls = []

                async for chunk in stream:
                    # Accumulate usage
                    # 累加 token 使用量
                    if chunk["type"] == "usage":
                        for k in total_usage:
                            total_usage[k] += chunk["usage"].get(k, 0)
                        # Forward usage event immediately
                        # 立即透传 usage 事件
                        yield chunk

                    elif chunk["type"] == "text_delta":
                        content_delta = chunk["content"]
                        current_content += content_delta
                        yield {"type": "text", "content": content_delta}  # Use "text" type for TUI compat, but it's a delta now
                        # 为兼容 TUI 使用 "text" 事件类型，但内容是增量片段

                    elif chunk["type"] == "tool_calls":
                        tool_calls = chunk["tool_calls"]

                # Decide next step based on what we got
                # 根据本轮返回结果决定下一步
                if tool_calls:
                    # We have tool calls
                    # 收到工具调用
                    # 1. Add assistant message (if any text preceded tool calls)
                    # 1. 先补入助手文本消息（若工具调用前已有文本）
                    if current_content:
                        full_messages.append({"role": "assistant", "content": current_content})

                    # 2. Add assistant message with tool calls
                    # 2. 追加包含 tool_calls 的助手消息
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

                    # 3. Execute tools
                    # 3. 执行工具
                    for tc in tool_calls:
                        tool_name = tc["name"]
                        tool_args_str = tc["arguments"]
                        tool_id = tc["id"]

                        try:
                            tool_args = json.loads(tool_args_str)
                            yield {
                                "type": "tool_call",
                                "tool": tool_name,
                                "args": tool_args
                            }

                            result = await self._execute_tool(tool_name, tool_args)

                            yield {
                                "type": "tool_result",
                                "tool": tool_name,
                                "result": str(result)
                            }

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

                    # Loop continues to send tool results back to LLM...
                    # 循环继续，把工具结果回传给 LLM

                elif current_content:
                    # Just text, no tools. We are done.
                    # 仅有文本、没有工具调用，流程结束
                    full_messages.append({"role": "assistant", "content": current_content})
                    # Break the loop, we are done
                    # 跳出循环，当前轮处理完成
                    break

                else:
                    # Empty response? Should not happen usually
                    # 空响应（通常不应出现）
                    break

            else:
                # Exhausted max iterations
                # 达到最大迭代次数
                logger.warning(f"工具调用达到最大次数 ({self.max_tool_iterations})")
                yield {"type": "text", "content": "\n\n(已达到最大工具调用次数，停止执行)"}

            # Yield accumulated usage if not already yielded (though we yield incrementally now)
            # 如果还有未透传的累计 usage，则在结束前补发（虽然现在通常是增量透传）
            # Just to be safe if total_usage is updated but not yielded
            # 只是兜底，防止 total_usage 更新了但未产出事件
            # (We rely on stream yielding usage events)
            # 实际上主要依赖流式输出中的 usage 事件

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

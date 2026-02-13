"""Sub-agent base class"""
import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any
from loguru import logger


class BaseAgent(ABC):
    """Sub-agent abstract base class"""

    def __init__(self, config):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name (unique identifier)"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description (used by main agent to decide when to delegate)"""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Specialized system prompt for this agent"""
        pass

    @property
    def available_tools(self) -> list[str]:
        """Tools this agent can use (empty = all tools)"""
        return []

    @property
    def temperature(self) -> float:
        """LLM temperature for this agent"""
        return 0.7

    @property
    def max_iterations(self) -> int:
        """Max LLM + tool calling iterations"""
        return 5

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: dict[str, Any]
    ) -> AsyncGenerator[dict, None]:
        """
        Execute task, yield event stream

        Yields:
            {"type": "text", "content": "..."}
            {"type": "tool_call", "tool": "...", "args": {...}}
            {"type": "tool_result", "tool": "...", "result": "..."}
            {"type": "error", "error": "..."}
            {"type": "done"}
        """
        pass

    async def _run_llm_loop(self, task: str) -> AsyncGenerator[dict, None]:
        """
        Shared LLM + tool calling loop for sub-agents.
        Subclasses call this from execute() to avoid code duplication.
        """
        from engines.llm.openai_compatible_engine import OpenAICompatibleEngine
        from tools.registry import ToolRegistry

        llm = OpenAICompatibleEngine(
            api_base=self.config.llm.api_base,
            api_key=self.config.llm.api_key,
            model=self.config.llm.model,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.config.llm.max_tokens
        )
        await llm.initialize()

        try:
            all_tools = ToolRegistry.get_tool_definitions()
            tools = [
                t for t in all_tools
                if t["function"]["name"] in self.available_tools
            ] if self.available_tools else all_tools

            # Warn about configured tools that don't exist
            # 对配置了但未注册的工具给出告警
            if self.available_tools:
                registered_names = {t["function"]["name"] for t in all_tools}
                for tool_name in self.available_tools:
                    if tool_name not in registered_names:
                        logger.warning(f"子Agent {self.name} 配置了不存在的工具: {tool_name}")

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task}
            ]

            for _ in range(self.max_iterations):
                response = await llm.chat_with_tools(messages, tools=tools)

                if response["type"] == "text":
                    content = response["content"]
                    yield {"type": "text", "content": content}
                    messages.append({"role": "assistant", "content": content})
                    break

                elif response["type"] == "tool_calls":
                    tool_calls = response["tool_calls"]
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]}
                            }
                            for tc in tool_calls
                        ]
                    })

                    for tc in tool_calls:
                        tool_name = tc["name"]
                        tool_args = json.loads(tc["arguments"])

                        yield {"type": "tool_call", "tool": tool_name, "args": tool_args}

                        try:
                            tool = ToolRegistry.get_tool(tool_name)
                            if tool:
                                result = await tool.execute(**tool_args)
                            else:
                                result = f"工具不存在: {tool_name}"
                                logger.error(f"工具不存在: {tool_name}")
                        except Exception as e:
                            result = f"工具执行失败: {e}"
                            logger.error(f"工具 {tool_name} 执行失败: {e}")

                        yield {"type": "tool_result", "tool": tool_name, "result": str(result)}
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tool_name,
                            "content": str(result)
                        })

        except Exception as e:
            logger.error(f"子Agent {self.name} LLM循环失败: {e}")
            yield {"type": "error", "error": str(e)}
        finally:
            await llm.cleanup()

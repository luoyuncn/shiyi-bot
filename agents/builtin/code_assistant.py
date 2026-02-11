"""Code assistant sub-agent"""
from typing import AsyncGenerator, Any

from agents.base_agent import BaseAgent


class CodeAssistantAgent(BaseAgent):
    """Code assistant - writing, debugging, testing code"""

    @property
    def name(self) -> str:
        return "code_assistant"

    @property
    def description(self) -> str:
        return "代码助手，擅长编写、调试、测试各种编程语言的代码"

    @property
    def system_prompt(self) -> str:
        return """你是专业的代码助手。
工作流程：理解需求 → 分析方案 → 编写代码 → 必要时测试验证 → 解释说明
要求：
- 代码简洁、注重可读性
- 提供简短的解释
- 遇到错误主动调试修复"""

    @property
    def available_tools(self) -> list[str]:
        return ["execute_shell", "file_operations", "search_web"]

    @property
    def temperature(self) -> float:
        return 0.3

    @property
    def max_iterations(self) -> int:
        return 5

    async def execute(
        self,
        task: str,
        context: dict[str, Any]
    ) -> AsyncGenerator[dict, None]:
        """Execute coding task"""
        async for event in self._run_llm_loop(task):
            yield event
        yield {"type": "done"}

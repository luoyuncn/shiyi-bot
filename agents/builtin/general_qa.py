"""General Q&A sub-agent"""
from typing import AsyncGenerator, Any

from agents.base_agent import BaseAgent


class GeneralQAAgent(BaseAgent):
    """General Q&A agent - knowledge, analysis, writing"""

    @property
    def name(self) -> str:
        return "general_qa"

    @property
    def description(self) -> str:
        return "通用问答助手，擅长知识查询、分析推理、文本写作"

    @property
    def system_prompt(self) -> str:
        return """你是通用问答助手。
要求：
- 回答准确、客观
- 结构清晰
- 必要时使用搜索工具获取最新信息"""

    @property
    def available_tools(self) -> list[str]:
        return ["search_web"]

    @property
    def temperature(self) -> float:
        return 0.7

    @property
    def max_iterations(self) -> int:
        return 3

    async def execute(
        self,
        task: str,
        context: dict[str, Any]
    ) -> AsyncGenerator[dict, None]:
        """Execute Q&A task"""
        async for event in self._run_llm_loop(task):
            yield event
        yield {"type": "done"}

"""query_memory tool - LLM 主动查询记忆图谱。"""

from __future__ import annotations

from loguru import logger

from tools.base import BaseTool, ToolDefinition, ToolParameter


class Tool(BaseTool):
    """LLM 主动查询 Kuzu 记忆图谱的工具。"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="query_memory",
            description=(
                "查询长期记忆。当需要回忆过去的对话、用户信息、项目进展或实体关系时调用此工具。"
                "支持三种模式：hybrid（默认，语义+图谱混合）、semantic（模糊回忆）、graph（实体推理）。"
            ),
            parameters={
                "query": ToolParameter(
                    type="string",
                    description="查询内容，用自然语言描述你想找的信息",
                    required=True,
                ),
                "mode": ToolParameter(
                    type="string",
                    description="检索模式：hybrid（默认）| semantic | graph",
                    required=False,
                    enum=["hybrid", "semantic", "graph"],
                ),
                "top_k": ToolParameter(
                    type="integer",
                    description="返回结果数量，默认 5，最大 10",
                    required=False,
                ),
            },
        )

    async def execute(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        **kwargs,
    ) -> str:
        """执行记忆查询，返回格式化文本结果。"""
        from memory.kuzu_manager import get_retriever

        retriever = get_retriever()
        if retriever is None:
            return "记忆系统尚未初始化，无法查询。"

        top_k = min(max(1, int(top_k)), 10)
        mode = mode if mode in ("hybrid", "semantic", "graph") else "hybrid"

        try:
            hits = await retriever.search(query=query, mode=mode, top_k=top_k)
        except Exception as e:
            logger.error(f"query_memory 查询失败: {e}")
            return f"记忆查询失败: {e}"

        if not hits:
            return "未找到相关记忆。"

        lines = [f"[记忆查询结果 | mode={mode} | query={query[:50]}]"]
        for i, hit in enumerate(hits, 1):
            lines.append(f"{i}. {hit.to_text(max_chars=200)} (score={hit.score:.2f})")

        return "\n".join(lines)

"""Web search tool (placeholder implementation)"""
from tools.base import BaseTool, ToolDefinition, ToolParameter


class Tool(BaseTool):
    """Web search tool"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_web",
            description="搜索互联网获取最新信息",
            parameters={
                "query": ToolParameter(
                    type="string",
                    description="搜索关键词",
                    required=True
                ),
                "max_results": ToolParameter(
                    type="number",
                    description="最大结果数量",
                    required=False
                )
            }
        )

    async def execute(self, query: str, max_results: int = 5) -> str:
        """Execute search (placeholder)"""
        # Placeholder: return mock result
        return f"搜索结果（模拟）：关于'{query}'的信息暂未实现真实搜索API，这是一个占位符返回。"

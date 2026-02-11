"""Web search tool using DuckDuckGo"""
import asyncio

from loguru import logger

from tools.base import BaseTool, ToolDefinition, ToolParameter


class Tool(BaseTool):
    """Web search tool (DuckDuckGo, no API key required)"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_web",
            description="搜索互联网获取最新信息，适合查询新闻、事实、最新数据等",
            parameters={
                "query": ToolParameter(
                    type="string",
                    description="搜索关键词",
                    required=True
                ),
                "max_results": ToolParameter(
                    type="number",
                    description="最大结果数量，默认5，最大50",
                    required=False
                )
            }
        )

    async def validate_params(self, params: dict):
        """Validate search parameters"""
        query = params.get("query", "")
        if not query or not str(query).strip():
            raise ValueError("搜索关键词不能为空")
        if len(str(query)) > 500:
            raise ValueError("搜索关键词不能超过500个字符")

        max_results = params.get("max_results", 5)
        try:
            max_results_int = int(max_results)
        except (TypeError, ValueError):
            raise ValueError(f"max_results 必须是整数，收到: {max_results}")
        if not (1 <= max_results_int <= 50):
            raise ValueError(f"max_results 必须在 1-50 之间，收到: {max_results_int}")

    async def execute(self, query: str, max_results: int = 5) -> str:
        """Execute web search via DuckDuckGo"""
        try:
            from ddgs import DDGS

            max_results_int = int(max_results)
            logger.debug(f"搜索: {query}, 最大结果: {max_results_int}")

            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: list(DDGS().text(query, max_results=max_results_int))
                ),
                timeout=10.0
            )

            if not results:
                return f"未找到关于'{query}'的搜索结果"

            logger.debug(f"搜索完成，获得 {len(results)} 条结果")

            formatted = [f"搜索结果：{query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "无标题")
                body = r.get("body", "无摘要")
                href = r.get("href", "")
                formatted.append(f"{i}. **{title}**\n   {body}\n   {href}\n")

            return "\n".join(formatted)

        except ImportError:
            return "搜索功能未安装，请运行: uv add ddgs"
        except asyncio.TimeoutError:
            logger.warning(f"搜索超时: {query}")
            return "搜索超时，请重试或更换关键词"
        except ValueError as e:
            logger.error(f"搜索参数错误: {e}")
            return f"参数错误: {e}"
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return f"搜索失败: {e}"

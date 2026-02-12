"""Web search tool — Tavily preferred, DuckDuckGo fallback"""
import asyncio
import os

from loguru import logger

from tools.base import BaseTool, ToolDefinition, ToolParameter


class Tool(BaseTool):
    """Web search tool (Tavily API, DuckDuckGo fallback)"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_web",
            description="搜索互联网获取最新信息，适合查询新闻、事实、最新数据等。支持基础搜索和深度搜索两种模式。",
            parameters={
                "query": ToolParameter(
                    type="string",
                    description="搜索关键词",
                    required=True,
                ),
                "max_results": ToolParameter(
                    type="number",
                    description="最大结果数量，默认5，最大10",
                    required=False,
                ),
                "search_depth": ToolParameter(
                    type="string",
                    description="搜索深度: basic(快速) 或 advanced(深度浏览网页，仅Tavily支持)",
                    required=False,
                    enum=["basic", "advanced"],
                ),
            },
        )

    async def validate_params(self, params: dict):
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
        if not (1 <= max_results_int <= 10):
            raise ValueError(f"max_results 必须在 1-10 之间，收到: {max_results_int}")

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> str:
        """Search: try Tavily first, fall back to DuckDuckGo."""
        max_results_int = min(int(max_results), 10)

        # ── Try Tavily first ─────────────────────────────────
        tavily_result = await self._tavily_search(query, max_results_int, search_depth)
        if tavily_result is not None:
            return tavily_result

        # ── Fallback: DuckDuckGo ─────────────────────────────
        logger.info("Tavily 不可用，降级到 DuckDuckGo")
        return await self._ddg_search(query, max_results_int)

    async def _tavily_search(
        self, query: str, max_results: int, search_depth: str
    ) -> str | None:
        """Tavily search. Returns None if unavailable (triggers fallback)."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.debug("TAVILY_API_KEY 未配置，将使用 DuckDuckGo 兜底")
            return None

        try:
            from tavily import TavilyClient
        except ImportError:
            logger.debug("tavily-python 未安装，将使用 DuckDuckGo 兜底")
            return None

        depth = search_depth if search_depth in ("basic", "advanced") else "basic"
        logger.debug(f"Tavily搜索: {query}, 深度: {depth}, 最大结果: {max_results}")

        try:
            loop = asyncio.get_event_loop()
            client = TavilyClient(api_key=api_key)
            response = await loop.run_in_executor(
                None,
                lambda: client.search(
                    query=query,
                    search_depth=depth,
                    include_answer=True,
                    max_results=max_results,
                ),
            )

            answer = response.get("answer", "")
            results = response.get("results", [])

            if not answer and not results:
                return f"未找到关于'{query}'的搜索结果"

            logger.debug(f"Tavily搜索完成，获得 {len(results)} 条结果")

            formatted = [f"搜索结果：{query}\n"]
            if answer:
                formatted.append(f"**摘要**: {answer}\n")
            for i, r in enumerate(results, 1):
                title = r.get("title", "无标题")
                content = r.get("content", "无摘要")
                url = r.get("url", "")
                formatted.append(f"{i}. **{title}**\n   {content}\n   {url}\n")

            return "\n".join(formatted)

        except Exception as e:
            logger.warning(f"Tavily搜索失败，降级到 DuckDuckGo: {e}")
            return None

    async def _ddg_search(self, query: str, max_results: int) -> str:
        """DuckDuckGo fallback search (no API key required)."""
        try:
            from ddgs import DDGS
        except ImportError:
            return "搜索不可用: Tavily 未配置且 ddgs 未安装。请设置 TAVILY_API_KEY 或运行 uv add ddgs"

        logger.debug(f"DuckDuckGo搜索: {query}, 最大结果: {max_results}")

        try:
            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: list(DDGS().text(query, max_results=max_results)),
                ),
                timeout=10.0,
            )

            if not results:
                return f"未找到关于'{query}'的搜索结果"

            logger.debug(f"DuckDuckGo搜索完成，获得 {len(results)} 条结果")

            formatted = [f"搜索结果：{query}（via DuckDuckGo）\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "无标题")
                body = r.get("body", "无摘要")
                href = r.get("href", "")
                formatted.append(f"{i}. **{title}**\n   {body}\n   {href}\n")

            return "\n".join(formatted)

        except asyncio.TimeoutError:
            logger.warning(f"DuckDuckGo搜索超时: {query}")
            return "搜索超时，请重试或更换关键词"
        except Exception as e:
            logger.error(f"DuckDuckGo搜索失败: {e}")
            return f"搜索失败: {e}"

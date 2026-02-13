"""Tool registry"""
from typing import Dict, List
from loguru import logger

from tools.base import BaseTool


class ToolRegistry:
    """Tool registry"""
    _tools: Dict[str, BaseTool] = {}
    _initialized = False

    @classmethod
    async def initialize(cls, tools_config: dict):
        """Initialize tool system"""
        if cls._initialized:
            return

        # Load built-in tools
        # 加载内置工具
        builtin_tools = tools_config.builtin
        for tool_name in builtin_tools:
            await cls._load_builtin_tool(tool_name)

        # Load MCP tools if enabled
        # 若启用 MCP，则加载 MCP 工具
        mcp_config = tools_config.mcp
        if isinstance(mcp_config, dict) and mcp_config.get("enabled", False):
            servers = mcp_config.get("servers", [])
            if servers:
                from tools.mcp_client import MCPClient
                count = await MCPClient.initialize(servers, cls)
                logger.info(f"MCP 工具加载完成，共 {count} 个工具")
            else:
                logger.debug("MCP已启用但未配置服务器")

        cls._initialized = True
        logger.info(f"工具注册器初始化完成，已注册 {len(cls._tools)} 个工具")

    @classmethod
    async def _load_builtin_tool(cls, tool_name: str):
        """Load built-in tool"""
        try:
            module = __import__(
                f"tools.builtin.{tool_name}",
                fromlist=["Tool"]
            )
            tool_class = getattr(module, "Tool")
            tool_instance = tool_class()

            cls.register(tool_instance)
            logger.debug(f"已加载内置工具: {tool_name}")

        except Exception as e:
            logger.error(f"加载工具失败 {tool_name}: {e}")

    @classmethod
    def register(cls, tool: BaseTool):
        """Register tool"""
        cls._tools[tool.definition.name] = tool

    @classmethod
    def get_tool(cls, name: str) -> BaseTool | None:
        """Get tool"""
        return cls._tools.get(name)

    @classmethod
    def list_tools(cls) -> List[str]:
        """List all tool names"""
        return list(cls._tools.keys())

    @classmethod
    def get_tool_definitions(cls) -> List[dict]:
        """Get all tool definitions (OpenAI format)"""
        return [
            tool.definition.to_openai_format()
            for tool in cls._tools.values()
        ]

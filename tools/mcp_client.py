"""MCP (Model Context Protocol) client - connects to external tool servers"""
import httpx
from loguru import logger

from tools.base import BaseTool, ToolDefinition, ToolParameter


class MCPTool(BaseTool):
    """Wrapper for a tool provided by an MCP server"""

    def __init__(self, server_url: str, tool_schema: dict):
        self._server_url = server_url
        self._definition = self._parse_schema(tool_schema)

    def _parse_schema(self, schema: dict) -> ToolDefinition:
        """Parse MCP tool schema into ToolDefinition"""
        params = {}
        input_schema = schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_list = input_schema.get("required", [])

        for param_name, param_schema in properties.items():
            params[param_name] = ToolParameter(
                type=param_schema.get("type", "string"),
                description=param_schema.get("description", ""),
                required=param_name in required_list,
                enum=param_schema.get("enum")
            )

        return ToolDefinition(
            name=schema["name"],
            description=schema.get("description", ""),
            parameters=params
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> str:
        """Call the MCP server to execute the tool"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._server_url}/tools/call",
                    json={
                        "name": self.definition.name,
                        "arguments": kwargs
                    }
                )
                response.raise_for_status()
                result = response.json()

                # MCP standard response: {"content": [{"type": "text", "text": "..."}]}
                # MCP 标准响应格式示例：{"content": [{"type": "text", "text": "..."}]}
                content = result.get("content", [])
                if content:
                    return "\n".join(
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text"
                    )
                return str(result)

        except httpx.TimeoutException:
            raise TimeoutError(f"MCP server 超时: {self._server_url}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"MCP server 返回错误 {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"MCP 工具调用失败: {e}")


class MCPClient:
    """MCP protocol client - discovers and registers tools from MCP servers"""

    @classmethod
    async def initialize(cls, servers: list[dict], tool_registry) -> int:
        """
        Connect to MCP servers and register their tools.

        Args:
            servers: List of server configs, e.g. [{"url": "http://...", "name": "..."}]
            tool_registry: ToolRegistry class to register tools into

        Returns:
            Total number of tools registered
        """
        total = 0
        for server_config in servers:
            url = server_config.get("url", "").rstrip("/")
            name = server_config.get("name", url)
            if not url:
                logger.warning("MCP服务器配置缺少 url 字段，跳过")
                continue

            try:
                count = await cls._load_server_tools(url, name, tool_registry)
                total += count
                logger.info(f"MCP服务器 '{name}' 已加载 {count} 个工具")
            except Exception as e:
                logger.warning(f"MCP服务器 '{name}' ({url}) 加载失败: {e}")

        return total

    @classmethod
    async def _load_server_tools(cls, url: str, name: str, tool_registry) -> int:
        """Load and register tools from a single MCP server"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{url}/tools/list")
            response.raise_for_status()
            data = response.json()

        tools = data.get("tools", [])
        if not tools:
            logger.debug(f"MCP服务器 '{name}' 没有返回任何工具")
            return 0

        for tool_schema in tools:
            mcp_tool = MCPTool(server_url=url, tool_schema=tool_schema)
            tool_registry.register(mcp_tool)
            logger.debug(f"注册MCP工具: {tool_schema['name']} (来自 {name})")

        return len(tools)

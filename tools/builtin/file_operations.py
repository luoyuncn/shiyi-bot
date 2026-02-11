"""File operations tool"""
from pathlib import Path
from tools.base import BaseTool, ToolDefinition, ToolParameter


class Tool(BaseTool):
    """File operations tool"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_operations",
            description="读取或写入文件",
            parameters={
                "operation": ToolParameter(
                    type="string",
                    description="操作类型",
                    required=True,
                    enum=["read", "write", "append", "list"]
                ),
                "path": ToolParameter(
                    type="string",
                    description="文件路径",
                    required=True
                ),
                "content": ToolParameter(
                    type="string",
                    description="写入的内容（write/append时需要）",
                    required=False
                )
            }
        )

    async def execute(
        self,
        operation: str,
        path: str,
        content: str = ""
    ) -> str:
        """Execute file operation"""
        file_path = Path(path)

        try:
            if operation == "read":
                if not file_path.exists():
                    return f"文件不存在: {path}"
                return file_path.read_text(encoding="utf-8")

            elif operation == "write":
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                return f"已写入文件: {path}"

            elif operation == "append":
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)
                return f"已追加到文件: {path}"

            elif operation == "list":
                if not file_path.is_dir():
                    return f"不是目录: {path}"
                files = [f.name for f in file_path.iterdir()]
                return "\n".join(files)

            else:
                return f"未知操作: {operation}"

        except Exception as e:
            return f"文件操作失败: {str(e)}"

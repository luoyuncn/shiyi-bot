"""Write file tool — create or overwrite a file."""
from pathlib import Path

from tools.base import BaseTool, ToolDefinition, ToolParameter

_WORKDIR = Path.cwd()


def _safe_path(p: str) -> Path:
    path = (_WORKDIR / p).resolve()
    if not path.is_relative_to(_WORKDIR):
        raise ValueError(f"路径越界（不允许访问工作目录以外）: {p}")
    return path


class Tool(BaseTool):
    """Write or append content to a file, creating parent dirs as needed."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description=(
                "将内容写入文件（覆盖或追加）。路径相对于项目根目录。"
                "如需精确替换某段文字，请用 edit_file。"
            ),
            parameters={
                "path": ToolParameter(
                    type="string",
                    description="目标文件相对路径",
                    required=True,
                ),
                "content": ToolParameter(
                    type="string",
                    description="要写入的内容",
                    required=True,
                ),
                "mode": ToolParameter(
                    type="string",
                    description="写入模式：'overwrite'（默认，覆盖）或 'append'（追加）",
                    required=False,
                    enum=["overwrite", "append"],
                ),
            },
        )

    async def execute(self, path: str, content: str, mode: str = "overwrite") -> str:
        try:
            fp = _safe_path(path)
        except ValueError as e:
            return f"错误: {e}"

        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with fp.open("a", encoding="utf-8") as f:
                    f.write(content)
                return f"已追加 {len(content)} 字符到: {path}"
            else:
                fp.write_text(content, encoding="utf-8")
                return f"已写入 {len(content)} 字符到: {path}"

        except Exception as e:
            return f"写入失败: {e}"

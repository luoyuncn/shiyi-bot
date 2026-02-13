"""Read file tool — read file or list directory contents."""
from pathlib import Path

from tools.base import BaseTool, ToolDefinition, ToolParameter

_WORKDIR = Path.cwd()
_MAX_CHARS = 50_000


def _safe_path(p: str) -> Path:
    path = (_WORKDIR / p).resolve()
    if not path.is_relative_to(_WORKDIR):
        raise ValueError(f"路径越界（不允许访问工作目录以外）: {p}")
    return path


class Tool(BaseTool):
    """Read a file or list a directory, paths relative to project root."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description=(
                "读取文件内容，或列出目录下的文件。路径相对于项目根目录。"
                "例如：path='data/memory/system/ShiYi.md'  或  path='data/memory/'"
            ),
            parameters={
                "path": ToolParameter(
                    type="string",
                    description="相对路径（文件或目录）",
                    required=True,
                ),
                "limit": ToolParameter(
                    type="number",
                    description="最多读取的行数（仅限文件，默认全部）",
                    required=False,
                ),
            },
        )

    async def execute(self, path: str, limit: int = None) -> str:
        try:
            fp = _safe_path(path)
        except ValueError as e:
            return f"错误: {e}"

        try:
            if fp.is_dir():
                entries = sorted(fp.iterdir(), key=lambda x: (x.is_file(), x.name))
                lines = [f"{'[d]' if e.is_dir() else '[f]'} {e.name}" for e in entries]
                return "\n".join(lines) or "（空目录）"

            if not fp.exists():
                return f"文件不存在: {path}"

            text = fp.read_text(encoding="utf-8", errors="replace")
            if limit:
                split = text.splitlines()
                if limit < len(split):
                    omit = len(split) - limit
                    text = "\n".join(split[:limit]) + f"\n... （省略后续 {omit} 行）"

            if len(text) > _MAX_CHARS:
                text = text[:_MAX_CHARS] + f"\n... [已截断，共 {len(text)} 字符]"

            return text

        except Exception as e:
            return f"读取失败: {e}"

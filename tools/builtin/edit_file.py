"""Edit file tool — surgical find-and-replace inside a file."""
from pathlib import Path

from tools.base import BaseTool, ToolDefinition, ToolParameter

_WORKDIR = Path.cwd()


def _safe_path(p: str) -> Path:
    path = (_WORKDIR / p).resolve()
    if not path.is_relative_to(_WORKDIR):
        raise ValueError(f"路径越界（不允许访问工作目录以外）: {p}")
    return path


class Tool(BaseTool):
    """Replace exact text in a file (first occurrence only)."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="edit_file",
            description=(
                "精确替换文件中的一段文字（只替换第一次出现）。"
                "old_text 必须与文件内容完全匹配。"
                "适合修改函数、配置项等局部改动；整体重写请用 write_file。"
            ),
            parameters={
                "path": ToolParameter(
                    type="string",
                    description="目标文件相对路径",
                    required=True,
                ),
                "old_text": ToolParameter(
                    type="string",
                    description="要替换的原始文字（必须精确匹配）",
                    required=True,
                ),
                "new_text": ToolParameter(
                    type="string",
                    description="替换后的新文字",
                    required=True,
                ),
            },
        )

    async def execute(self, path: str, old_text: str, new_text: str) -> str:
        try:
            fp = _safe_path(path)
        except ValueError as e:
            return f"错误: {e}"

        if not fp.exists():
            return f"文件不存在: {path}"

        try:
            content = fp.read_text(encoding="utf-8", errors="replace")

            if old_text not in content:
                # Give a helpful context snippet if the text is not found
                preview = content[:300].replace("\n", "↵")
                return (
                    f"未找到要替换的文字。\n"
                    f"文件前300字符预览: {preview}"
                )

            new_content = content.replace(old_text, new_text, 1)
            fp.write_text(new_content, encoding="utf-8")
            return f"已替换 {path}"

        except Exception as e:
            return f"编辑失败: {e}"

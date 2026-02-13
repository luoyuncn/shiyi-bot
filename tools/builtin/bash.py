"""Bash tool — run any shell command in the project working directory."""
import asyncio
from pathlib import Path

from loguru import logger
from tools.base import BaseTool, ToolDefinition, ToolParameter

_WORKDIR = Path.cwd()

_DANGEROUS = [
    "rm -rf /",
    "rm -rf ~",
    "dd if=",
    "mkfs",
    "> /dev/sda",
    ":(){ :|:& };:",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
]

_MAX_OUTPUT = 50_000  # chars


class Tool(BaseTool):
    """Run a shell command with the project root as cwd."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description=(
                "在项目根目录下执行 Shell 命令。"
                "适用于：运行脚本、git 操作、pip/uv 安装、grep/find 搜索、列目录等。"
                "文件读写请优先使用 read_file / write_file / edit_file 工具。"
            ),
            parameters={
                "command": ToolParameter(
                    type="string",
                    description="要执行的 Shell 命令",
                    required=True,
                ),
                "timeout": ToolParameter(
                    type="number",
                    description="超时时间（秒，默认 60）",
                    required=False,
                ),
            },
        )

    async def validate_params(self, params: dict):
        cmd = params.get("command", "")
        for danger in _DANGEROUS:
            if danger in cmd:
                raise ValueError(f"禁止执行危险命令: {danger}")

    async def execute(self, command: str, timeout: int = 60) -> str:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_WORKDIR),
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            out = stdout.decode("utf-8", errors="replace") if stdout else ""
            err = stderr.decode("utf-8", errors="replace") if stderr else ""
            combined = (out + err).strip()

            if process.returncode != 0:
                result = f"[exit {process.returncode}]\n{combined}" if combined else f"[exit {process.returncode}]"
            else:
                result = combined or "（命令成功，无输出）"

            if len(result) > _MAX_OUTPUT:
                result = result[:_MAX_OUTPUT] + f"\n... [输出已截断，共 {len(result)} 字符]"

            logger.debug(f"bash | cwd={_WORKDIR} | cmd={command[:80]!r}")
            return result

        except asyncio.TimeoutError:
            return f"命令执行超时（{timeout} 秒）"
        except Exception as e:
            return f"执行失败: {e}"

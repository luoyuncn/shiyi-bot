"""Shell command execution tool (use with caution)"""
import asyncio
from tools.base import BaseTool, ToolDefinition, ToolParameter


class Tool(BaseTool):
    """Shell command execution tool"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="execute_shell",
            description="执行Shell命令（仅限安全命令）",
            parameters={
                "command": ToolParameter(
                    type="string",
                    description="要执行的Shell命令",
                    required=True
                ),
                "timeout": ToolParameter(
                    type="number",
                    description="超时时间（秒）",
                    required=False
                )
            }
        )

    async def validate_params(self, params: dict):
        """Safety check: block dangerous commands"""
        command = params.get("command", "")

        # Blacklist
        # 危险命令黑名单
        dangerous_commands = [
            "rm -rf", "dd if=", "mkfs", "format",
            "> /dev/sda", ":(){ :|:& };:"
        ]

        for danger in dangerous_commands:
            if danger in command:
                raise ValueError(f"禁止执行危险命令: {danger}")

    async def execute(self, command: str, timeout: int = 30) -> str:
        """Execute shell command"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            output = stdout.decode() if stdout else ""
            error = stderr.decode() if stderr else ""

            if process.returncode != 0:
                return f"命令执行失败（退出码 {process.returncode}）:\n{error}"

            return output or "命令执行成功（无输出）"

        except asyncio.TimeoutError:
            return f"命令执行超时（{timeout}秒）"

        except Exception as e:
            return f"执行失败: {str(e)}"

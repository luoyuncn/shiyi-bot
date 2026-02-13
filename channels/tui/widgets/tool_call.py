"""Tool call collapsible block widget"""
import json
from rich.markup import escape
from textual.widgets import Collapsible, Static

from channels.tui.icons import icons


class ToolCallBlock(Collapsible):
    """Minimal collapsible block showing a tool call."""

    def __init__(self, tool_name: str, args: dict, **kwargs):
        self.tool_name = tool_name

        # Minimal title
        args_str = json.dumps(args, ensure_ascii=False)
        if len(args_str) > 40:
            args_str = args_str[:38] + "..."

        title = f"  {icons.tool} {tool_name} {escape(args_str)}"

        self._result_static = Static("...", classes="tool-result")
        super().__init__(self._result_static, title=title, collapsed=True, **kwargs)

    def set_result(self, result: str, elapsed: float = 0):
        # Truncate long results for minimal view
        display_result = result
        if len(display_result) > 300:
            display_result = display_result[:300] + "..."

        self._result_static.update(f"{escape(display_result)} ({elapsed:.2f}s)")

        # Update title to indicate completion
        self.title = f"  {icons.tool} {self.tool_name} {icons.success}"

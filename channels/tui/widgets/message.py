"""Message widgets — user bubble, assistant markdown, thinking indicator, notices, welcome"""
from rich.markup import escape
from textual.widget import Widget
from textual.widgets import Static, Markdown
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich import box

from channels.tui.icons import icons


class UserMessage(Widget):
    """用户消息 — 右对齐，带标签和右竖线，与 AI 消息对称。"""

    def __init__(self, content: str, **kwargs):
        super().__init__(**kwargs)
        self._content = content

    def compose(self):
        yield Static(f"You {icons.user}", classes="user-label")
        yield Static(escape(self._content), classes="user-bubble")


class AssistantMessage(Widget):
    """AI 消息 — 左对齐，带橙色标签和左竖线，Markdown 渲染。"""

    def __init__(self, content: str = "", **kwargs):
        super().__init__(**kwargs)
        self._content = content

    def compose(self):
        yield Static(f"{icons.assistant} {icons.app_name}", classes="ai-label")
        yield Markdown(self._content, classes="ai-content")

    async def update_content(self, content: str):
        self._content = content
        try:
            md = self.query_one(Markdown)
            await md.update(content)
        except Exception:
            pass


class ThinkingIndicator(Widget):
    """Animated braille spinner — lightweight thinking indicator.

    Cycles through braille dot patterns at 100ms intervals.
    Auto-cleans timer on unmount.
    """

    _FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._frame = 0
        self._timer = None

    def on_mount(self):
        self._timer = self.set_interval(1 / 10, self._tick)

    def _tick(self):
        self._frame = (self._frame + 1) % len(self._FRAMES)
        self.refresh()

    def render(self):
        spinner = self._FRAMES[self._frame]
        return Text.assemble(
            ("  ", ""),
            (spinner, "#FFA500"),
            (" Thinking...", "italic #7a7a7a"),
        )

    def on_unmount(self):
        if self._timer:
            self._timer.stop()


# ── Cyberpunk pixel-art icon for ShiYi (一一) ───
def _make_pixel_icon():
    """Build a pixel-art icon representing "一一" (Yi-Yi).

    Two parallel vertical bars symbolizing "一一" - simple, clean, cyberpunk.
    Left bar in orange, right bar in cyan - representing warmth + tech.
    """
    # Block characters for pixel effect
    B = "\u2588"  # █ full block

    # 5x5 pixel "一一" icon - two vertical bars
    # Each bar is 2px wide with 1px gap, with 2-space indent for alignment
    indent = "  "
    lines = [
        f"{indent}{B}{B} {B}{B}",   #   ██ ██
        f"{indent}{B}{B} {B}{B}",   #   ██ ██
        f"{indent}{B}{B} {B}{B}",   #   ██ ██
        f"{indent}{B}{B} {B}{B}",   #   ██ ██
        f"{indent}{B}{B} {B}{B}",   #   ██ ██
    ]

    t = Text()
    for i, line in enumerate(lines):
        # Indent (unstyled)
        t.append(indent, style="")
        # Left bar: orange (warmth/family)
        t.append(B + B, style="bold #FFA500")
        # Gap
        t.append(" ", style="")
        # Right bar: cyan (tech/future)
        t.append(B + B, style="bold #00FFFF")

        if i < len(lines) - 1:
            t.append("\n")
    return t


class WelcomeView(Widget):
    """Welcome screen — cyberpunk minimal design.

    Clean interface with pixel-art icon and essential info.
    Shown on startup / /new / /clear; removed on first message.
    """

    def __init__(self, model_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name

    def render(self):
        # ── Header: pixel icon + brand (vertically stacked, aligned) ──
        icon = _make_pixel_icon()

        brand = Text()
        brand.append("\n")
        brand.append("  ShiYi", style="bold #FFA500")
        brand.append(" v2.0", style="#00FFFF")
        if self._model_name:
            brand.append("\n  ", style="")
            brand.append(self._model_name, style="#7a7a7a")
        brand.append("\n  Personal AI Assistant", style="italic #5a5550")

        header = Table.grid(padding=(0, 2))
        header.add_column()
        header.add_column()
        header.add_row(icon, brand)

        # ── Simple greeting ──
        greeting = Text()
        greeting.append("\n  How can I help you today?\n", style="#e0e0e0")

        # ── Single row shortcuts ──
        shortcuts = Text()
        shortcuts.append("\n  ")
        shortcuts.append(icons.arrow_right, style="#00FFFF")
        shortcuts.append(" /help", style="#8a8580")
        shortcuts.append("  ")
        shortcuts.append(icons.arrow_right, style="#00FFFF")
        shortcuts.append(" /new", style="#8a8580")
        shortcuts.append("  ")
        shortcuts.append(icons.arrow_right, style="#00FFFF")
        shortcuts.append(" /list", style="#8a8580")
        shortcuts.append("     ")
        shortcuts.append("Ctrl+C", style="#5a5550")
        shortcuts.append(" interrupt  ", style="#4a4540")
        shortcuts.append("Ctrl+D", style="#5a5550")
        shortcuts.append(" quit", style="#4a4540")
        shortcuts.append("\n")

        # ── Panel with brighter border ──
        return Panel(
            Group(header, greeting, shortcuts),
            border_style="#5a5045",  # Brighter border
            box=box.ROUNDED,
            padding=(1, 2),
        )


class SystemNotice(Static):
    """系统通知 — 灰色斜体。"""

    def __init__(self, content: str, **kwargs):
        super().__init__(f"  {content}", **kwargs)


class ErrorNotice(Static):
    """错误通知 — 红色左边线。"""

    def __init__(self, content: str, **kwargs):
        super().__init__(f"  {content}", **kwargs)

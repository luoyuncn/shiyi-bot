"""Message widgets — user bubble, assistant markdown, thinking indicator, notices, welcome"""
from textual.widget import Widget
from textual.widgets import Static, Markdown
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich import box


class UserMessage(Widget):
    """用户消息 — 右对齐气泡。"""

    def __init__(self, content: str, **kwargs):
        super().__init__(**kwargs)
        self._content = content

    def compose(self):
        yield Static(self._content, classes="user-bubble")


class AssistantMessage(Widget):
    """AI 消息 — 左对齐，Markdown 渲染，透明背景。"""

    def __init__(self, content: str = "", **kwargs):
        super().__init__(**kwargs)
        self._content = content

    def compose(self):
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

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

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
            (spinner, "#7aa2f7"),
            (" Thinking...", "italic #565f89"),
        )

    def on_unmount(self):
        if self._timer:
            self._timer.stop()


# ── Dragon logo (half-block + box-drawing pixel art) ─────
_DRAGON = (
    "  ▄▀▄     ▄▀▄ \n"
    " ─█ ● ▄▄▄ ● █─\n"
    "   ▀▄╰─◇─╯▄▀  \n"
    "     ▀▀▀▀▀▀    "
)


class WelcomeView(Widget):
    """Welcome screen — dragon logo, brand info, tips, shortcuts.

    Renders a single bordered panel matching input-area width.
    Shown on startup / /new / /clear; removed on first message.
    """

    def __init__(self, model_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name

    def render(self):
        # ── Header: dragon logo + brand info side by side ──
        dragon = Text(_DRAGON, style="bold #7aa2f7")

        info = Text()
        info.append("ShiYi", style="bold #7aa2f7")
        info.append("  v2.0\n", style="#565f89")
        if self._model_name:
            info.append(self._model_name, style="#9ece6a")
            info.append("\n")
        info.append("Your Personal AI Assistant", style="italic #565f89")

        header = Table.grid(padding=(0, 2))
        header.add_column()
        header.add_column()
        header.add_row(dragon, info)

        # ── Separator ──
        sep = Text("─" * 42, style="#2e3c56")

        # ── Greeting ──
        greeting = Text("\nHi! Ask me anything to get started.\n", style="#c0caf5")

        # ── Try suggestions (2×2 with bullets) ──
        try_grid = Table.grid(padding=(0, 3))
        try_grid.add_column()
        try_grid.add_column()
        try_grid.add_row(
            Text.assemble(("  ▸ ", "#ff9e64"), ('"What\'s the weather"', "italic #565f89")),
            Text.assemble(("  ▸ ", "#ff9e64"), ('"Write Python code"', "italic #565f89")),
        )
        try_grid.add_row(
            Text.assemble(("  ▸ ", "#ff9e64"), ('"Search latest news"', "italic #565f89")),
            Text.assemble(("  ▸ ", "#ff9e64"), ('"Read a file for me"', "italic #565f89")),
        )

        # ── Commands + Shortcuts ──
        footer = Text()
        footer.append("\n  Commands  ", style="#565f89")
        footer.append("/help ", style="bold #ff9e64")
        footer.append("/new ", style="bold #ff9e64")
        footer.append("/list ", style="bold #ff9e64")
        footer.append("/clear\n", style="bold #ff9e64")
        footer.append("  Shortcuts ", style="#565f89")
        footer.append("Ctrl+C", style="#9ece6a")
        footer.append(" interrupt  ", style="#565f89")
        footer.append("Ctrl+L", style="#9ece6a")
        footer.append(" clear  ", style="#565f89")
        footer.append("Ctrl+D", style="#9ece6a")
        footer.append(" quit", style="#565f89")

        # ── Assemble into a single bordered panel ──
        return Panel(
            Group(header, sep, greeting, try_grid, footer),
            border_style="#2e3c56",
            box=box.SQUARE,
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

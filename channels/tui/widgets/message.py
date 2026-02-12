"""Message widgets — user bubble, assistant markdown, thinking indicator, notices, welcome"""
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
        yield Static(self._content, classes="user-bubble")


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


# ── SHIYI text logo (cyberpunk double-line mechanical) ───
def _make_logo():
    """Build the SHIYI text logo — double-line box-drawing, heavy industrial.

    Uses ╔╗╚╝║═╠╣╦╩ for thick strokes and a schematic/blueprint aesthetic.
    Each letter is 5 chars wide with 2-char gaps. All lines = 33 chars.
    """
    # Double-line box-drawing alphabet
    H, V = "\u2550", "\u2551"          # ═ ║
    TL, TR = "\u2554", "\u2557"        # ╔ ╗
    BL, BR = "\u255a", "\u255d"        # ╚ ╝
    LT, RT = "\u2560", "\u2563"        # ╠ ╣
    TT, BT = "\u2566", "\u2569"        # ╦ ╩

    # Per-letter shapes (5 chars wide × 3 rows)
    s1, s2, s3 = f"{TL}{H*3}{TR}", f"{BL}{H*3}{TR}", f"{BL}{H*3}{BR}"
    h1, h2, h3 = f"{V}   {V}", f"{LT}{H*3}{RT}", f"{V}   {V}"
    i1, i2, i3 = f"{H*2}{TT}{H*2}", f"  {V}  ", f"{H*2}{BT}{H*2}"
    y1, y2, y3 = f"{V}   {V}", f"{BL}{H}{TT}{H}{BR}", f"  {V}  "

    g = "  "  # gap between letters
    t = Text()
    t.append(f"{s1}{g}{h1}{g}{i1}{g}{y1}{g}{i1}\n", style="bold #FFA500")
    t.append(f"{s2}{g}{h2}{g}{i2}{g}{y2}{g}{i2}\n", style="bold #FFA500")
    t.append(f"{s3}{g}{h3}{g}{i3}{g}{y3}{g}{i3}", style="bold #FFD700")
    return t


class WelcomeView(Widget):
    """Welcome screen — SHIYI logo, brand info, tips, shortcuts.

    Renders a single bordered panel matching input-area width.
    Shown on startup / /new / /clear; removed on first message.
    """

    def __init__(self, model_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name

    def render(self):
        # ── Header: SHIYI logo (left) + brand info (right) ──
        logo = _make_logo()

        info = Text()
        info.append(f"{icons.assistant} ", style="")
        info.append(icons.app_name, style="bold #FFA500")
        info.append("  v2.0\n", style="#7a7a7a")
        if self._model_name:
            info.append(self._model_name, style="#9ece6a")
            info.append("\n")
        info.append("Your Personal AI Assistant\n", style="italic #7a7a7a")
        if icons.is_nerd_font:
            info.append("Nerd Font ", style="#9ece6a")
            info.append(icons.success, style="#9ece6a")
        else:
            info.append("Nerd Font off", style="#4a4540")

        header = Table.grid(padding=(0, 4))
        header.add_column()
        header.add_column()
        header.add_row(logo, info)

        # ── Separator ──
        sep = Text(icons.separator * 42, style="#3a3530")

        # ── Greeting ──
        greeting = Text("\nHi! Ask me anything to get started.\n", style="#e0e0e0")

        # ── Try suggestions (2x2 with bullets) ──
        try_grid = Table.grid(padding=(0, 3))
        try_grid.add_column()
        try_grid.add_column()
        try_grid.add_row(
            Text.assemble((f"  {icons.arrow_right} ", "#FFA500"), ('"What\'s the weather"', "italic #a09080")),
            Text.assemble((f"  {icons.arrow_right} ", "#FFA500"), ('"Write Python code"', "italic #a09080")),
        )
        try_grid.add_row(
            Text.assemble((f"  {icons.arrow_right} ", "#FFA500"), ('"Search latest news"', "italic #a09080")),
            Text.assemble((f"  {icons.arrow_right} ", "#FFA500"), ('"Read a file for me"', "italic #a09080")),
        )

        # ── Commands + Shortcuts ──
        footer = Text()
        footer.append("\n  Commands  ", style="#7a7a7a")
        footer.append("/help ", style="bold #FFA500")
        footer.append("/new ", style="bold #FFA500")
        footer.append("/list ", style="bold #FFA500")
        footer.append("/clear\n", style="bold #FFA500")
        footer.append("  Shortcuts ", style="#7a7a7a")
        footer.append("Ctrl+C", style="#9ece6a")
        footer.append(" interrupt  ", style="#7a7a7a")
        footer.append("Ctrl+L", style="#9ece6a")
        footer.append(" clear  ", style="#7a7a7a")
        footer.append("Ctrl+D", style="#9ece6a")
        footer.append(" quit", style="#7a7a7a")

        if not icons.is_nerd_font:
            footer.append("\n\n  ", style="")
            footer.append("Tip", style="italic #4a4540")
            footer.append(" Install a Nerd Font for better icons ", style="#4a4540")
            footer.append("fonts/README.md", style="underline #4a4540")

        # ── Assemble into a single bordered panel ──
        return Panel(
            Group(header, sep, greeting, try_grid, footer),
            border_style="#3a3530",
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

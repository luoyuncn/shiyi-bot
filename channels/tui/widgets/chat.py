"""Chat view — scrollable message container"""
from textual.containers import VerticalScroll

from .message import (
    UserMessage,
    AssistantMessage,
    ThinkingIndicator,
    WelcomeView,
    SystemNotice,
    ErrorNotice,
)
from .tool_call import ToolCallBlock


class ChatView(VerticalScroll):
    """Main chat area — scrollable container for messages."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_tool_block: ToolCallBlock | None = None
        self._last_assistant_msg: AssistantMessage | None = None
        self._thinking: ThinkingIndicator | None = None
        self._welcome: WelcomeView | None = None

    # ── Welcome ─────────────────────────────────────────

    async def add_welcome(self, model_name: str = ""):
        self._welcome = WelcomeView(model_name=model_name)
        await self.mount(self._welcome)

    def remove_welcome(self):
        if self._welcome:
            self._welcome.remove()
            self._welcome = None

    # ── Messages ────────────────────────────────────────

    async def add_user_message(self, content: str):
        self.remove_welcome()
        msg = UserMessage(content)
        await self.mount(msg)
        self.call_after_refresh(self.scroll_end, animate=False)

    async def add_assistant_message(self, content: str = "") -> AssistantMessage:
        msg = AssistantMessage(content if content else "...")
        self._last_assistant_msg = msg
        await self.mount(msg)
        self.call_after_refresh(self.scroll_end, animate=False)
        return msg

    async def update_assistant_message(self, content: str):
        if self._last_assistant_msg:
            await self._last_assistant_msg.update_content(content)
            # 流式输出时自动滚动到底部
            self.call_after_refresh(self.scroll_end, animate=False)

    # ── Thinking ────────────────────────────────────────

    async def add_thinking(self) -> ThinkingIndicator:
        indicator = ThinkingIndicator()
        self._thinking = indicator
        await self.mount(indicator)
        self.call_after_refresh(self.scroll_end, animate=False)
        return indicator

    def remove_thinking(self):
        if self._thinking:
            self._thinking.remove()
            self._thinking = None

    # ── Tool Calls ──────────────────────────────────────

    async def add_tool_call(self, tool_name: str, args: dict) -> ToolCallBlock:
        block = ToolCallBlock(tool_name, args)
        self._last_tool_block = block
        await self.mount(block)
        self.call_after_refresh(self.scroll_end, animate=False)
        return block

    def update_tool_result(self, result: str, elapsed: float = 0):
        if self._last_tool_block:
            self._last_tool_block.set_result(result, elapsed)

    # ── Notices ─────────────────────────────────────────

    async def add_system_notice(self, content: str):
        await self.mount(SystemNotice(content))
        self.scroll_end(animate=False)

    async def add_error(self, content: str):
        await self.mount(ErrorNotice(content))
        self.scroll_end(animate=False)

    # ── Utilities ───────────────────────────────────────

    async def clear_messages(self):
        await self.remove_children()
        self._last_tool_block = None
        self._last_assistant_msg = None
        self._thinking = None
        self._welcome = None

    def remove_last_message(self):
        """Remove the last message widget (used for temporary status)."""
        if self.children:
            self.children[-1].remove()
            if not self.children:
                self._last_tool_block = None
                self._last_assistant_msg = None
                self._thinking = None

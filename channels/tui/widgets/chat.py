"""Chat view — scrollable message container"""
from textual.containers import VerticalScroll

from .message import UserMessage, AssistantMessage, SystemNotice, ErrorNotice
from .tool_call import ToolCallBlock


class ChatView(VerticalScroll):
    """Main chat area — scrollable container for messages."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_tool_block: ToolCallBlock | None = None
        self._last_assistant_msg: AssistantMessage | None = None

    async def add_user_message(self, content: str):
        msg = UserMessage(content)
        await self.mount(msg)
        # Ensure scroll happens after layout update
        self.call_after_refresh(self.scroll_end, animate=False)

    async def add_assistant_message(self, content: str = "") -> AssistantMessage:
        """Add assistant message. If content is empty, shows a 'Thinking...' state."""
        msg = AssistantMessage(content if content else "✨ 思考中...")
        self._last_assistant_msg = msg
        await self.mount(msg)
        self.call_after_refresh(self.scroll_end, animate=False)
        return msg

    def update_assistant_message(self, content: str):
        if self._last_assistant_msg:
            self._last_assistant_msg.update_content(content)
            # We need to scroll end if content grows
            # But calling it every char might be too much.
            # Textual's VerticalScroll should handle auto-scroll if stickiness is enabled?
            # Let's force it for now but maybe throttle it if needed.
            # self.scroll_end(animate=False)
            pass

    async def add_tool_call(self, tool_name: str, args: dict) -> ToolCallBlock:
        block = ToolCallBlock(tool_name, args)
        self._last_tool_block = block
        await self.mount(block)
        self.call_after_refresh(self.scroll_end, animate=False)
        return block

    def update_tool_result(self, result: str, elapsed: float = 0):
        if self._last_tool_block:
            self._last_tool_block.set_result(result, elapsed)

    async def add_system_notice(self, content: str):
        await self.mount(SystemNotice(content))
        self.scroll_end(animate=False)

    async def add_error(self, content: str):
        await self.mount(ErrorNotice(content))
        self.scroll_end(animate=False)

    async def clear_messages(self):
        await self.remove_children()
        self._last_tool_block = None
        self._last_assistant_msg = None

    def remove_last_message(self):
        """Remove the last message widget (used for temporary status like 'Thinking...')."""
        if self.children:
            self.children[-1].remove()
            # Update references if needed
            if not self.children:
                self._last_tool_block = None
                self._last_assistant_msg = None

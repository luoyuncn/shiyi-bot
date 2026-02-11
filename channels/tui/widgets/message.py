"""Message widgets — user, assistant, system, error"""
from textual.widgets import Static, Markdown
from textual.containers import Vertical, Container


class UserMessage(Vertical):
    """A user message in the chat."""

    def __init__(self, content: str, **kwargs):
        super().__init__(**kwargs)
        self.content = content

    def compose(self):
        # Bubble container for the message content
        with Container(classes="bubble"):
            yield Static("You", classes="message-label")
            yield Markdown(self.content)


class AssistantMessage(Vertical):
    """An assistant message with Markdown rendering."""

    def __init__(self, content: str = "", **kwargs):
        super().__init__(**kwargs)
        self._content = content

    def compose(self):
        with Container(classes="bubble"):
            yield Static("✨ ShiYi", classes="message-label")
            yield Markdown(self._content)

    def update_content(self, content: str):
        self._content = content
        try:
            md = self.query_one(Markdown)
            md.update(content)
        except Exception:
            pass


class SystemNotice(Static):
    """System notification message."""

    def __init__(self, content: str, **kwargs):
        super().__init__(f"  {content}", **kwargs)


class ErrorNotice(Static):
    """Error message."""

    def __init__(self, content: str, **kwargs):
        super().__init__(f"  ❌ {content}", **kwargs)

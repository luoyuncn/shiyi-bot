"""ShiYi TUI Application — main Textual app"""
import time

from loguru import logger
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input
from .widgets import (
    HeaderBar,
    ChatView,
    StatusBar,
    LogPanel,
)

# Loguru sink ID, used to remove the sink on cleanup
_log_sink_id: int | None = None


class ShiYiApp(App):
    """ShiYi TUI — terminal user interface for the AI assistant."""

    CSS_PATH = "styles/theme.tcss"

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "中断/退出", priority=True),
        Binding("ctrl+d", "quit", "退出"),
        Binding("ctrl+l", "clear_chat", "清屏"),
    ]

    def __init__(
        self,
        config,
        session_manager,
        agent_core,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.session_manager = session_manager
        self.agent_core = agent_core
        self.debug_mode = debug
        self.current_session = None
        self._processing = False
        self._interrupt_requested = False
        self._input_history: list[str] = []
        self._history_index: int = -1

    # ── Layout ──────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        yield ChatView(id="chat-view")
        # yield StatusBar(id="status-bar") # Remove status bar for cleaner look, info is in header
        if self.debug_mode:
            yield LogPanel(id="log-panel")

        # Input container for better styling
        yield Input(
            placeholder="输入您的问题... (Enter 发送)",
            id="message-input",
        )

    # ── Lifecycle ───────────────────────────────────────────

    async def on_mount(self) -> None:
        # Create default session
        self.current_session = await self.session_manager.create_session(
            {"channel": "tui"}
        )

        # Update header
        header = self.query_one("#header", HeaderBar)
        header.model_name = self.config.llm.model
        header.session_id = self.current_session.session_id

        # Setup loguru sink for debug panel
        if self.debug_mode:
            self._setup_log_sink()

        # Welcome message
        chat = self.query_one("#chat-view", ChatView)
        await chat.add_system_notice(
            "欢迎使用 ShiYi ✦  会话已创建  |  输入 /help 查看命令"
        )

        # Focus input
        self.query_one("#message-input", Input).focus()

    def _setup_log_sink(self) -> None:
        global _log_sink_id
        log_panel = self.query_one("#log-panel", LogPanel)

        def _sink(message):
            # message is the pre-formatted string from loguru
            try:
                self.call_from_thread(log_panel.write_log, str(message))
            except Exception:
                # App might be shutting down
                pass

        _log_sink_id = logger.add(
            _sink,
            level="DEBUG",
            format="{time:HH:mm:ss} | {level:8} | {name}:{function} - {message}",
            colorize=False,
        )

    # ── Input Handling ──────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        event.input.value = ""

        # Save to history
        self._input_history.append(text)
        self._history_index = -1

        if text.startswith("/"):
            await self._handle_command(text)
        else:
            if self._processing:
                chat = self.query_one("#chat-view", ChatView)
                await chat.add_system_notice("请等待当前回复完成...")
                return
            self._send_message(text)

    # ── Message Processing (worker) ─────────────────────────

    def _send_message(self, text: str) -> None:
        """Start async worker to process the message."""
        self.run_worker(
            self._process_message(text),
            name="process_message",
            exclusive=True,
        )

    async def _process_message(self, text: str) -> None:
        self._processing = True
        self._interrupt_requested = False
        chat = self.query_one("#chat-view", ChatView)

        try:
            # Save user message
            await self.session_manager.save_message(
                self.current_session.session_id, "user", text
            )
            await chat.add_user_message(text)

            # Add thinking indicator
            await chat.add_system_notice("Thinking... ⏳")
            thinking_removed = False

            # Get conversation context
            context = await self.session_manager.get_session(
                self.current_session.session_id
            )
            messages = context.messages

            # Track timing
            t0 = time.monotonic()
            first_event = True

            # Stream response
            full_response_text = ""
            current_bubble_text = ""
            # The placeholder is the "current" assistant message initially
            current_assistant_msg = chat._last_assistant_msg
            thinking_cleared = False

            async for event in self.agent_core.process_message_stream(messages):
                if self._interrupt_requested:
                    if not thinking_removed:
                        chat.remove_last_message()
                        thinking_removed = True
                    await chat.add_system_notice("已中断回复")
                    break

                # Remove thinking indicator on first event
                if not thinking_removed:
                    chat.remove_last_message()
                    thinking_removed = True

                etype = event["type"]

                if etype == "text":
                    delta = event["content"]
                    full_response_text += delta
                    current_bubble_text += delta

                    if current_assistant_msg is None:
                        current_assistant_msg = await chat.add_assistant_message(current_bubble_text)
                        if first_event:
                            # Latency tracking removed for now
                            # latency = int((time.monotonic() - t0) * 1000)
                            first_event = False
                    else:
                        chat.update_assistant_message(current_bubble_text)

                elif etype == "tool_call":
                    # Start new text bubble after tool call
                    current_assistant_msg = None
                    current_bubble_text = ""

                    await chat.add_tool_call(
                        event["tool"], event.get("args", {})
                    )

                elif etype == "tool_result":
                    elapsed = time.monotonic() - t0
                    chat.update_tool_result(event.get("result", ""), elapsed)

                elif etype == "error":
                    await chat.add_error(event["error"])

                elif etype == "usage":
                    # Usage info is now handled in header or logs if needed
                    pass

                elif etype == "done":
                    pass

            # Save assistant response (full accumulated text)
            if full_response_text:
                await self.session_manager.save_message(
                    self.current_session.session_id, "assistant", full_response_text
                )

            # status.message_count += 1  # Removed status bar

        except Exception as e:
            if not thinking_removed:
                chat.remove_last_message()
            await chat.add_error(f"处理失败: {e}")
            logger.error(f"TUI消息处理失败: {e}")

        finally:
            self._processing = False

    # ── Slash Commands ──────────────────────────────────────

    async def _handle_command(self, cmd: str) -> None:
        chat = self.query_one("#chat-view", ChatView)
        header = self.query_one("#header", HeaderBar)

        if cmd == "/new":
            self.current_session = await self.session_manager.create_session(
                {"channel": "tui"}
            )
            header.session_id = self.current_session.session_id
            # status.reset_usage()
            await chat.clear_messages()
            await chat.add_system_notice(
                f"新会话已创建: {self.current_session.session_id[:8]}"
            )

        elif cmd == "/list":
            sessions = await self.session_manager.list_sessions()
            if not sessions:
                await chat.add_system_notice("暂无会话记录")
                return
            lines = ["会话列表:"]
            for s in sessions:
                marker = " ◀" if s.session_id == self.current_session.session_id else ""
                lines.append(f"  {s.session_id[:8]}  (最后活跃: {s.last_active}){marker}")
            await chat.add_system_notice("\n".join(lines))

        elif cmd.startswith("/switch "):
            session_id = cmd.split(maxsplit=1)[1].strip()
            # Try to find session matching prefix
            sessions = await self.session_manager.list_sessions()
            match = None
            for s in sessions:
                if s.session_id.startswith(session_id):
                    match = s.session_id
                    break
            if not match:
                await chat.add_error(f"会话不存在: {session_id}")
                return

            self.current_session = await self.session_manager.get_session(match)
            if self.current_session:
                header.session_id = self.current_session.session_id
                # status.reset_usage()
                await chat.clear_messages()
                # Replay existing messages
                for msg in self.current_session.messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        await chat.add_user_message(content)
                    elif role == "assistant" and content:
                        await chat.add_assistant_message(content)
                await chat.add_system_notice(f"已切换到会话: {match[:8]}")
            else:
                await chat.add_error(f"会话不存在: {session_id}")

        elif cmd == "/clear":
            await chat.clear_messages()

        elif cmd == "/help":
            await chat.add_system_notice(
                "可用命令:\n"
                "  /new          创建新会话\n"
                "  /list         列出所有会话\n"
                "  /switch <id>  切换到指定会话\n"
                "  /clear        清屏\n"
                "  /help         显示帮助\n"
                "\n"
                "快捷键:\n"
                "  Ctrl+C        中断回复 / 退出\n"
                "  Ctrl+D        退出\n"
                "  Ctrl+L        清屏"
            )

        else:
            await chat.add_error(f"未知命令: {cmd}")

    # ── Key Bindings ────────────────────────────────────────

    def action_interrupt(self) -> None:
        if self._processing:
            self._interrupt_requested = True
        else:
            self.exit()

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat-view", ChatView)
        self.run_worker(chat.clear_messages(), exclusive=False)

    # ── Cleanup ─────────────────────────────────────────────

    async def on_unmount(self) -> None:
        global _log_sink_id
        if _log_sink_id is not None:
            try:
                logger.remove(_log_sink_id)
            except ValueError:
                pass
            _log_sink_id = None

"""ShiYi TUI Application — main Textual app (Warm Amber theme)"""
import time

from loguru import logger
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Container
from textual.widgets import Input, Static
from .widgets import (
    ChatView,
    LogPanel,
)
from channels.tui.icons import icons, init_icons

# Loguru sink ID, used to remove the sink on cleanup
_log_sink_id: int | None = None

# Braille spinner frames for prompt animation
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class ShiYiApp(App):
    """ShiYi TUI — Warm Amber themed terminal interface."""

    CSS_PATH = "styles/theme.tcss"

    BINDINGS = [
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

        # Initialise icon set from config (nerd_font toggle)
        use_nerd = getattr(getattr(config, "tui", None), "nerd_font", False)
        init_icons(use_nerd_font=use_nerd)
        self.current_session = None
        self._processing = False
        self._interrupt_requested = False
        self._input_history: list[str] = []
        self._history_index: int = -1
        self._prompt_timer = None
        self._prompt_frame: int = 0

    # ── Layout ──────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield ChatView(id="chat-view")
        if self.debug_mode:
            yield LogPanel(id="log-panel")
        # Centering wrapper — full-width dock, centers the 80% input-area
        with Container(id="input-dock"):
            with Horizontal(id="input-area"):
                yield Static(icons.prompt, classes="prompt-symbol")
                yield Input(
                    placeholder="Type a message or command...",
                    id="message-input",
                )

    # ── Lifecycle ───────────────────────────────────────────

    async def on_mount(self) -> None:
        # Create default session
        self.current_session = await self.session_manager.create_session(
            {"channel": "tui"}
        )

        # Setup loguru sink for debug panel
        if self.debug_mode:
            self._setup_log_sink()

        # Welcome screen
        chat = self.query_one("#chat-view", ChatView)
        await chat.add_welcome(model_name=self.config.llm.model)

        # Focus input
        self.query_one("#message-input", Input).focus()

    def _setup_log_sink(self) -> None:
        global _log_sink_id
        log_panel = self.query_one("#log-panel", LogPanel)

        def _sink(message):
            try:
                self.call_from_thread(log_panel.write_log, str(message))
            except Exception:
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

    def on_key(self, event: events.Key) -> None:
        """Key handling — Ctrl+C interrupt, Escape refocus, Up/Down history."""
        inp = self.query_one("#message-input", Input)

        # Ctrl+C: only intercept when AI is processing; otherwise let terminal handle copy
        if event.key == "ctrl+c":
            if self._processing:
                self._interrupt_requested = True
                event.stop()
                event.prevent_default()
            else:
                event.prevent_default()  # prevent app quit, but don't stop()
            return

        # Escape or any typing key: refocus input if not already focused
        if not inp.has_focus:
            if event.key == "escape" or event.is_printable:
                inp.focus()
                return

        # Up/Down: navigate input history (only when input has focus)
        if not inp.has_focus:
            return

        if event.key == "up" and self._input_history:
            if self._history_index == -1:
                self._history_index = len(self._input_history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            else:
                return
            inp.value = self._input_history[self._history_index]
            inp.cursor_position = len(inp.value)
            event.stop()

        elif event.key == "down" and self._history_index >= 0:
            if self._history_index < len(self._input_history) - 1:
                self._history_index += 1
                inp.value = self._input_history[self._history_index]
            else:
                self._history_index = -1
                inp.value = ""
            inp.cursor_position = len(inp.value)
            event.stop()

    # ── Processing State & Animation ────────────────────────

    def _set_processing(self, active: bool) -> None:
        """Toggle processing visual state — animated spinner on prompt."""
        self._processing = active
        input_area = self.query_one("#input-area")
        prompt = self.query_one(".prompt-symbol", Static)

        if active:
            input_area.add_class("processing")
            self._prompt_frame = 0
            self._prompt_timer = self.set_interval(1 / 10, self._tick_prompt)
        else:
            input_area.remove_class("processing")
            if self._prompt_timer:
                self._prompt_timer.stop()
                self._prompt_timer = None
            prompt.update(icons.prompt)  # restore prompt icon

    def _tick_prompt(self) -> None:
        """Animate prompt symbol with braille spinner."""
        try:
            prompt = self.query_one(".prompt-symbol", Static)
            prompt.update(_SPINNER[self._prompt_frame])
            self._prompt_frame = (self._prompt_frame + 1) % len(_SPINNER)
        except Exception:
            pass

    # ── Message Processing (worker) ─────────────────────────

    def _send_message(self, text: str) -> None:
        """Start async worker to process the message."""
        self.run_worker(
            self._process_message(text),
            name="process_message",
            exclusive=True,
        )

    async def _process_message(self, text: str) -> None:
        self._set_processing(True)
        self._interrupt_requested = False
        chat = self.query_one("#chat-view", ChatView)

        try:
            # Save user message
            await self.session_manager.save_message(
                self.current_session.session_id, "user", text
            )
            await chat.add_user_message(text)

            # Add thinking indicator
            await chat.add_thinking()
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
            current_assistant_msg = None

            async for event in self.agent_core.process_message_stream(messages):
                if self._interrupt_requested:
                    if not thinking_removed:
                        chat.remove_thinking()
                        thinking_removed = True
                    await chat.add_system_notice("已中断回复")
                    break

                # Remove thinking indicator on first event
                if not thinking_removed:
                    chat.remove_thinking()
                    thinking_removed = True

                etype = event["type"]

                if etype == "text":
                    delta = event["content"]
                    full_response_text += delta
                    current_bubble_text += delta

                    if current_assistant_msg is None:
                        current_assistant_msg = await chat.add_assistant_message(
                            current_bubble_text
                        )
                        if first_event:
                            first_event = False
                    else:
                        await chat.update_assistant_message(current_bubble_text)

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
                    pass

                elif etype == "done":
                    pass

            # Save assistant response (full accumulated text)
            if full_response_text:
                await self.session_manager.save_message(
                    self.current_session.session_id, "assistant", full_response_text
                )

        except Exception as e:
            if not thinking_removed:
                chat.remove_thinking()
            await chat.add_error(f"处理失败: {e}")
            logger.error(f"TUI消息处理失败: {e}")

        finally:
            self._set_processing(False)
            self.query_one("#message-input", Input).focus()

    # ── Slash Commands ──────────────────────────────────────

    async def _handle_command(self, cmd: str) -> None:
        chat = self.query_one("#chat-view", ChatView)

        if cmd == "/new":
            self.current_session = await self.session_manager.create_session(
                {"channel": "tui"}
            )
            await chat.clear_messages()
            await chat.add_welcome(model_name=self.config.llm.model)

        elif cmd == "/list":
            sessions = await self.session_manager.list_sessions()
            if not sessions:
                await chat.add_system_notice("暂无会话记录")
                return
            lines = ["会话列表:"]
            for s in sessions:
                marker = " <" if s.session_id == self.current_session.session_id else ""
                lines.append(
                    f"  {s.session_id[:8]}  (最后活跃: {s.last_active}){marker}"
                )
            await chat.add_system_notice("\n".join(lines))

        elif cmd.startswith("/switch "):
            session_id = cmd.split(maxsplit=1)[1].strip()
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
            await chat.add_welcome(model_name=self.config.llm.model)

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
                "  Ctrl+C        中断回复\n"
                "  Ctrl+D        退出\n"
                "  Ctrl+L        清屏"
            )

        else:
            await chat.add_error(f"未知命令: {cmd}")

        # Always refocus input after command
        self.query_one("#message-input", Input).focus()

    # ── Actions ────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat-view", ChatView)
        self.run_worker(chat.clear_messages(), exclusive=False)

    # ── Cleanup ─────────────────────────────────────────────

    async def on_unmount(self) -> None:
        global _log_sink_id
        if self._prompt_timer:
            self._prompt_timer.stop()
        if _log_sink_id is not None:
            try:
                logger.remove(_log_sink_id)
            except ValueError:
                pass
            _log_sink_id = None

"""CLI text channel"""
import asyncio
from loguru import logger
from channels.base import BaseChannel


class TextCLIChannel(BaseChannel):
    """Command line interface channel"""

    def __init__(self, config, session_manager, agent_core):
        self.config = config
        self.session_manager = session_manager
        self.agent_core = agent_core
        self.running = False
        self.current_session = None

    async def start(self):
        """Start CLI loop"""
        self._print_welcome()

        # Create default session
        # 创建默认会话
        self.current_session = await self.session_manager.create_session({
            "channel": "cli"
        })

        logger.info(f"会话ID: {self.current_session.session_id}")

        self.running = True

        # Main loop
        # 主循环
        while self.running:
            try:
                # Read user input
                # 读取用户输入
                user_input = await asyncio.to_thread(
                    input,
                    "\n👤 你: "
                )

                if not user_input.strip():
                    continue

                # Handle commands
                # 处理命令输入
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Save user message
                # 保存用户消息
                await self.session_manager.save_message(
                    self.current_session.session_id,
                    "user",
                    user_input
                )

                # Process with AgentCore
                # 交给 AgentCore 处理
                print("🤖 助手: ", end="", flush=True)

                # Get conversation context
                # 获取会话上下文
                context = await self.session_manager.get_session(self.current_session.session_id)
                messages = await self.session_manager.prepare_messages_for_agent(
                    context.messages
                )

                # Stream response
                # 流式输出回复
                response_text = ""
                tool_call_count = 0
                _TOOL_ICONS = {
                    "search_web": "🔍", "super_search": "🔍",
                    "bash": "⚡", "read_file": "📄",
                    "write_file": "✏️", "edit_file": "✏️",
                    "query_memory": "🧠",
                }
                async for event in self.agent_core.process_message_stream(messages):
                    if event["type"] == "text":
                        print(event["content"], end="", flush=True)
                        response_text += event["content"]
                    elif event["type"] == "tool_call":
                        tool_call_count += 1
                        tool = event["tool"]
                        icon = _TOOL_ICONS.get(tool, "🔧")
                        args = event.get("args", {})
                        arg_summary = ""
                        if isinstance(args, dict) and args:
                            first_val = next(iter(args.values()), "")
                            arg_summary = f': "{str(first_val)[:60]}"' if first_val else ""
                        print(f"\n  {icon} [{tool_call_count}] {tool}{arg_summary}", flush=True)
                    elif event["type"] == "tool_result":
                        result = str(event.get("result", ""))
                        first_line = result.split("\n")[0][:80] if result else ""
                        summary = f" → {first_line}" if first_line else ""
                        print(f"  ✓ [{tool_call_count}]{summary}", flush=True)
                    elif event["type"] == "error":
                        print(f"\n❌ 错误: {event['error']}", flush=True)

                print()  # 换行

                # Save assistant message
                # 保存助手消息
                await self.session_manager.save_message(
                    self.current_session.session_id,
                    "assistant",
                    response_text
                )

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"CLI错误: {e}")

    async def stop(self):
        """Stop CLI"""
        self.running = False

    async def _handle_command(self, cmd: str):
        """Handle CLI commands"""
        if cmd == "/new":
            self.current_session = await self.session_manager.create_session({
                "channel": "cli"
            })
            print(f"✅ 新会话: {self.current_session.session_id}")

        elif cmd == "/list":
            sessions = await self.session_manager.list_sessions()
            for s in sessions:
                print(f"- {s.session_id} (最后活跃: {s.last_active})")

        elif cmd.startswith("/switch "):
            session_id = cmd.split()[1]
            self.current_session = await self.session_manager.get_session(session_id)
            if self.current_session:
                print(f"✅ 切换到会话: {session_id}")
            else:
                print(f"❌ 会话不存在: {session_id}")

        elif cmd == "/help":
            self._print_help()

    def _print_welcome(self):
        print("=" * 60)
        print("🏠 Shiyi - CLI模式")
        print("命令: /new /list /switch <id> /help")
        print("=" * 60)

    def _print_help(self):
        print("""
可用命令:
  /new          - 创建新会话
  /list         - 列出所有会话
  /switch <id>  - 切换到指定会话
  /help         - 显示帮助
  Ctrl+C        - 退出
        """)

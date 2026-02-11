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
        self.current_session = await self.session_manager.create_session({
            "channel": "cli"
        })

        logger.info(f"会话ID: {self.current_session.session_id}")

        self.running = True

        # Main loop
        while self.running:
            try:
                # Read user input
                user_input = await asyncio.to_thread(
                    input,
                    "\n👤 你: "
                )

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Save user message
                await self.session_manager.save_message(
                    self.current_session.session_id,
                    "user",
                    user_input
                )

                # Process with AgentCore
                print("🤖 助手: ", end="", flush=True)

                # Get conversation context
                context = await self.session_manager.get_session(self.current_session.session_id)
                messages = context.messages + [{"role": "user", "content": user_input}]

                # Stream response
                response_text = ""
                async for event in self.agent_core.process_message_stream(messages):
                    if event["type"] == "text":
                        print(event["content"], end="", flush=True)
                        response_text += event["content"]
                    elif event["type"] == "tool_call":
                        print(f"\n[调用工具: {event['tool']}]", flush=True)
                    elif event["type"] == "tool_result":
                        print("[工具返回]", flush=True)
                    elif event["type"] == "error":
                        print(f"\n❌ 错误: {event['error']}", flush=True)

                print()  # 换行

                # Save assistant message
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
        print("🏠 ShiYiBot - CLI模式")
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

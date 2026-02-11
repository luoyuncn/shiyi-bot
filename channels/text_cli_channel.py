"""CLI text channel"""
import asyncio
from loguru import logger
from channels.base import BaseChannel


class TextCLIChannel(BaseChannel):
    """Command line interface channel"""

    def __init__(self, config, session_manager):
        self.config = config
        self.session_manager = session_manager
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

                # Echo for now (will integrate AgentCore later)
                print(f"🤖 助手: [收到消息] {user_input}")

                # Save to session
                await self.session_manager.save_message(
                    self.current_session.session_id,
                    "user",
                    user_input
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

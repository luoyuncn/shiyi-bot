"""Orchestrator - manages all channels and core components"""
import asyncio
from loguru import logger
from config.settings import Settings

from core.session_manager import SessionManager
from core.agent_core import AgentCore
from channels.text_cli_channel import TextCLIChannel
from tools.registry import ToolRegistry


class Orchestrator:
    """Main orchestrator"""

    def __init__(self, config: Settings):
        self.config = config
        self.running = False

        # Initialize core components
        self.session_manager = SessionManager(config.memory)
        self.agent_core = AgentCore(config)

        # Initialize channels
        self.channels = []

        # For now, only CLI channel
        self.channels.append(
            TextCLIChannel(config, self.session_manager, self.agent_core)
        )

    async def start(self):
        """Start all components"""
        logger.info("=" * 60)
        logger.info(f"🚀 {self.config.system.name} 正在启动...")
        logger.info("=" * 60)

        try:
            # Initialize core
            await self._initialize_core()

            # Start all channels
            self.running = True
            channel_tasks = [
                asyncio.create_task(channel.start(), name=f"channel_{i}")
                for i, channel in enumerate(self.channels)
            ]

            logger.info("✅ 所有通道已启动")
            logger.info("=" * 60)

            # Wait for all channels
            await asyncio.gather(*channel_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"启动失败: {e}")
            raise

    async def stop(self):
        """Stop all components"""
        logger.info("正在停止所有通道...")
        self.running = False

        for channel in self.channels:
            try:
                await channel.stop()
            except Exception as e:
                logger.error(f"停止通道失败: {e}")

        await self.agent_core.cleanup()
        await self.session_manager.cleanup()

    async def _initialize_core(self):
        """Initialize core components"""
        logger.info("正在初始化核心组件...")

        # Initialize tool registry
        await ToolRegistry.initialize(self.config.tools)
        logger.info(f"已注册 {len(ToolRegistry.list_tools())} 个工具")

        # Initialize agent core
        await self.agent_core.initialize()

        # Initialize session manager
        await self.session_manager.initialize()

        logger.info("核心组件初始化完成")

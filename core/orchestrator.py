"""Orchestrator - manages all channels and core components"""
import asyncio
from loguru import logger
from config.settings import Settings

from core.session_manager import SessionManager
from core.agent_core import AgentCore
from channels.text_cli_channel import TextCLIChannel
from channels.text_api_channel import TextAPIChannel
from channels.voice_channel import VoiceChannel
from tools.registry import ToolRegistry


class Orchestrator:
    """Main orchestrator"""

    def __init__(self, config: Settings, tui_mode: bool = False, debug_mode: bool = False):
        self.config = config
        self.running = False
        self.tui_mode = tui_mode
        self.debug_mode = debug_mode

        # Initialize core components
        # 初始化核心组件
        self.session_manager = SessionManager(config.memory, llm_config=config.llm)
        self.agent_core = AgentCore(config)

        # Initialize channels based on config
        # 根据配置初始化通道
        self.channels = []

        # Voice channel
        # 语音通道
        if config.channels.get("voice", {}).get("enabled", False):
            self.channels.append(VoiceChannel(config))

        # CLI channel (skip if TUI mode — TUI is launched separately by main.py)
        # CLI 通道（TUI 模式下跳过，TUI 由 main.py 单独启动）
        if not tui_mode and config.channels.get("cli", {}).get("enabled", True):
            self.channels.append(
                TextCLIChannel(config, self.session_manager, self.agent_core)
            )

        # API channel
        # API 通道
        if config.channels.get("api", {}).get("enabled", False):
            self.channels.append(
                TextAPIChannel(config, self.session_manager, self.agent_core)
            )

    async def initialize_core(self):
        """Initialize core components (public, for TUI mode pre-init)"""
        await self._initialize_core()

    async def start(self):
        """Start all components"""
        logger.info("=" * 60)
        logger.info(f"🚀 {self.config.system.name} 正在启动...")
        logger.info("=" * 60)

        try:
            # Initialize core
            # 初始化核心
            await self._initialize_core()

            # Start all channels
            # 启动全部通道
            self.running = True
            channel_tasks = [
                asyncio.create_task(channel.start(), name=f"channel_{i}")
                for i, channel in enumerate(self.channels)
            ]

            logger.info("✅ 所有通道已启动")
            logger.info("=" * 60)

            # Wait for all channels
            # 等待所有通道结束
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
        # 初始化工具注册器
        await ToolRegistry.initialize(self.config.tools)
        logger.info(f"已注册 {len(ToolRegistry.list_tools())} 个工具")

        # Initialize Agent registry
        # 初始化 Agent 注册器
        agent_config = getattr(self.config, 'agent', {}) or {}
        if isinstance(agent_config, dict) and agent_config.get("enable_sub_agents", False):
            from agents.registry import AgentRegistry
            await AgentRegistry.initialize(self.config)

        # Initialize agent core
        # 初始化 Agent Core
        await self.agent_core.initialize()

        # Initialize session manager
        # 初始化会话管理器
        await self.session_manager.initialize()

        logger.info("核心组件初始化完成")

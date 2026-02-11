"""Voice channel - wraps existing AssistantCore"""
from loguru import logger
from channels.base import BaseChannel


class VoiceChannel(BaseChannel):
    """Voice channel adapter - wraps existing voice assistant"""

    def __init__(self, config):
        self.config = config
        self.assistant_core = None

    async def start(self):
        """Start voice channel"""
        try:
            # Import here to avoid circular dependency
            from core.assistant import AssistantCore

            logger.info("🎤 语音通道启动中...")

            # Create and start voice assistant
            self.assistant_core = AssistantCore(self.config)
            await self.assistant_core.start()

        except ImportError:
            logger.warning("语音通道未启用（AssistantCore 未找到）")
        except Exception as e:
            logger.error(f"语音通道启动失败: {e}")
            raise

    async def stop(self):
        """Stop voice channel"""
        if self.assistant_core:
            await self.assistant_core.stop()
        logger.info("语音通道已停止")

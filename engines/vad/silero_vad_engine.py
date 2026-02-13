"""Silero VAD语音活动检测引擎"""
import torch
import numpy as np
from io import BytesIO
from engines.base import BaseEngine
from audio.recorder import AudioRecorder
from loguru import logger
import asyncio
from typing import Optional


class SileroVADEngine(BaseEngine):
    """Silero VAD引擎 - 检测语音活动和静音"""

    def __init__(
        self,
        recorder: AudioRecorder,
        silence_duration_ms: int = 500,
        max_recording_seconds: int = 10
    ):
        """
        初始化VAD引擎

        Args:
            recorder: 音频录制器实例
            silence_duration_ms: 静音持续时间阈值(毫秒)
            max_recording_seconds: 最大录音时长(秒)
        """
        self.recorder = recorder
        self.silence_duration_ms = silence_duration_ms
        self.max_recording_seconds = max_recording_seconds
        self.sample_rate = recorder.sample_rate

        self.model = None
        self.get_speech_timestamps = None

    async def initialize(self):
        """加载Silero VAD模型"""
        try:
            # 加载模型
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]

            logger.info("Silero VAD模型已加载")

        except Exception as e:
            logger.error(f"加载VAD模型失败: {e}")
            raise

    # Silero VAD 要求固定输入长度：16kHz→512, 8kHz→256
    VAD_FRAME_SIZE = 512

    def _vad_prob(self, chunk: np.ndarray) -> float:
        """将任意长度 chunk 切成 VAD_FRAME_SIZE 小块求最大概率"""
        max_prob = 0.0
        audio_float = chunk.astype(np.float32) / 32768.0
        for start in range(0, len(audio_float), self.VAD_FRAME_SIZE):
            frame = audio_float[start:start + self.VAD_FRAME_SIZE]
            if len(frame) < self.VAD_FRAME_SIZE:
                frame = np.pad(frame, (0, self.VAD_FRAME_SIZE - len(frame)))
            tensor = torch.from_numpy(frame)
            with torch.no_grad():
                prob = self.model(tensor, self.sample_rate).item()
            if prob > max_prob:
                max_prob = prob
        return max_prob

    async def record_until_silence(self) -> bytes:
        """
        录音直到检测到静音

        Returns:
            录音的音频数据 (bytes格式)
        """
        buffer = BytesIO()
        silence_chunks = 0
        total_chunks = 0

        # 计算静音阈值（多少个chunk算静音）
        silence_threshold = int(
            self.silence_duration_ms / 1000 * self.sample_rate / self.recorder.chunk_size
        )

        # 最大录音chunk数
        max_chunks = int(
            self.max_recording_seconds * self.sample_rate / self.recorder.chunk_size
        )

        logger.info("🎤 开始录音...")

        # 最多等待 3 秒让用户开口；开口前不计静音
        max_wait_chunks = int(3.0 * self.sample_rate / self.recorder.chunk_size)
        speech_started = False

        try:
            for i in range(max_chunks + max_wait_chunks):
                # 在线程中读取，避免阻塞事件循环，响应取消信号
                chunk = await asyncio.to_thread(self.recorder.read_chunk)
                total_chunks += 1

                # VAD检测（自动分帧）
                speech_prob = self._vad_prob(chunk)

                if not speech_started:
                    if speech_prob >= 0.5:
                        speech_started = True
                        buffer.write(chunk.tobytes())
                        logger.debug("🗣️ 检测到说话声，开始录音缓冲")
                    # 未开口前丢弃音频，避免唤醒词残留
                    elif i >= max_wait_chunks:
                        logger.info("⏱️ 等待说话超时，取消本次录音")
                        return b""
                else:
                    buffer.write(chunk.tobytes())
                    # 判断是否为静音
                    if speech_prob < 0.5:
                        silence_chunks += 1
                        if silence_chunks >= silence_threshold:
                            logger.info(f"🔇 检测到{self.silence_duration_ms}ms静音，停止录音")
                            break
                    else:
                        silence_chunks = 0

            audio_bytes = buffer.getvalue()
            duration = len(audio_bytes) / (self.sample_rate * 2)  # int16 = 2 bytes
            # int16 每个采样点占 2 字节
            logger.info(f"✅ 录音完成: {len(audio_bytes)} bytes ({duration:.1f}秒)")

            return audio_bytes

        except Exception as e:
            logger.error(f"录音过程出错: {e}")
            raise

    async def listen_with_timeout(self, timeout: float = 3.0) -> bool:
        """
        连续对话窗口：在指定时间内检测是否有人声

        Args:
            timeout: 超时时间(秒)

        Returns:
            是否检测到人声
        """
        start_time = asyncio.get_event_loop().time()

        logger.debug(f"👂 进入连续对话窗口 ({timeout}秒)...")

        try:
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                chunk = await asyncio.to_thread(self.recorder.read_chunk)
                speech_prob = self._vad_prob(chunk)

                if speech_prob >= 0.5:
                    logger.info("🔊 连续对话窗口检测到人声")
                    return True

            logger.info("⏱️ 连续对话窗口超时，未检测到人声")
            return False

        except Exception as e:
            logger.error(f"连续对话窗口检测出错: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        self.model = None
        self.get_speech_timestamps = None
        logger.info("VAD引擎已清理")

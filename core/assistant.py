"""助理核心控制器 - 整合所有模块的主控制逻辑"""
import asyncio
import sys
import numpy as np
from loguru import logger
from core.state_machine import AssistantState
from core.sentence_splitter import SentenceSplitter
from audio.recorder import AudioRecorder
from audio.player import AudioPlayer
from engines.vad import SileroVADEngine
from engines.stt import TencentSTTEngine
from engines.llm import OpenAICompatibleEngine
from engines.tts import EdgeTTSEngine
from config.settings import Settings


class AssistantCore:
    """助理核心控制器 - 状态机驱动的异步架构"""

    def __init__(self, config: Settings):
        """
        初始化助理核心

        Args:
            config: 配置对象
        """
        self.config = config
        self.state = AssistantState.IDLE

        # 音频设备
        self.recorder = AudioRecorder(
            sample_rate=config.system.audio_sample_rate,
            chunk_size=config.audio.chunk_size,
            channels=config.audio.input_channels,
            device_index=config.audio.input_device_index
        )
        self.player = AudioPlayer(
            sample_rate=24000,  # Edge-TTS输出24kHz
            device_index=config.audio.output_device_index,
            pulse_sink=config.audio.pulse_sink,
        )

        # 唤醒词引擎（可选）
        self._skip_wake_word = getattr(config.wake_word, "skip_wake_word", False)
        self.wake_engine = None
        if not self._skip_wake_word:
            try:
                from engines.wake_word import OpenWakeWordEngine
                self.wake_engine = OpenWakeWordEngine(
                    model_path=config.wake_word.model_path,
                    threshold=config.wake_word.threshold
                )
            except ImportError:
                logger.warning("openwakeword 未安装，自动启用 skip_wake_word 模式")
                self._skip_wake_word = True
        self.vad_engine = SileroVADEngine(
            recorder=self.recorder,
            silence_duration_ms=config.vad.silence_duration_ms,
            max_recording_seconds=config.vad.max_recording_seconds
        )
        self.stt_engine = TencentSTTEngine(
            app_id=config.stt.app_id,
            secret_id=config.stt.secret_id,
            secret_key=config.stt.secret_key,
            region=config.stt.region,
            sample_rate=config.system.audio_sample_rate
        )
        self.llm_engine = OpenAICompatibleEngine(
            api_base=config.llm.api_base,
            api_key=config.llm.api_key,
            model=config.llm.model,
            system_prompt=config.llm.system_prompt,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        self.tts_engine = EdgeTTSEngine(
            voice=config.tts.voice,
            rate=config.tts.rate,
            pitch=config.tts.pitch
        )

        # 句子队列：LLM生成的句子传递给TTS
        self.sentence_queue = asyncio.Queue()

        # 控制标志
        self.running = False
        self.continuous_window_seconds = config.vad.continuous_window_seconds

        # 音量显示
        self._level_frame = 0

    async def initialize(self):
        """初始化所有组件"""
        logger.info("=" * 60)
        logger.info(f"🏠 {self.config.system.name} 正在初始化...")
        logger.info("=" * 60)

        try:
            # 启动音频设备
            self.recorder.start()
            self.player.start()

            # 初始化所有引擎
            logger.info("正在加载AI引擎...")
            if self.wake_engine:
                await self.wake_engine.initialize()
            else:
                logger.info("唤醒词已跳过，直接进入监听模式")
            await self.vad_engine.initialize()
            await self.stt_engine.initialize()
            await self.llm_engine.initialize()
            await self.tts_engine.initialize()

            logger.info("=" * 60)
            logger.info("✅ 初始化完成！")
            logger.info("💡 说出唤醒词开始对话...")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    async def start(self):
        """启动助理主循环"""
        await self.initialize()

        self.running = True
        self._tasks = [
            asyncio.create_task(self._main_loop(), name="assistant_main_loop"),
            asyncio.create_task(self._tts_playback_loop(), name="assistant_tts_loop"),
        ]

        try:
            await asyncio.gather(*self._tasks)

        except asyncio.CancelledError:
            logger.info("助理任务被取消")
            raise
        finally:
            # 确保任务退出
            self.running = False
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self):
        """停止助理主循环"""
        if not self.running:
            return
        self.running = False
        if hasattr(self, "_tasks"):
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _main_loop(self):
        """主控制循环 - 处理唤醒、录音、识别、推理"""
        while self.running:
            try:
                if self.state == AssistantState.IDLE:
                    # 待机态：等待唤醒词（或直接进入监听）
                    if self._skip_wake_word:
                        self.state = AssistantState.LISTENING
                    else:
                        await self._wait_for_wake_word()

                elif self.state == AssistantState.LISTENING:
                    # 监听态：VAD录音
                    await self._listen_and_process()

                elif self.state == AssistantState.CONTINUOUS:
                    # 连续对话窗口
                    await self._continuous_dialogue_window()

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"主循环出错: {e}", exc_info=True)
                self.state = AssistantState.IDLE
                await asyncio.sleep(1)

    async def _wait_for_wake_word(self):
        """等待唤醒词"""
        chunk = await asyncio.to_thread(self.recorder.read_chunk)

        # 每 16 帧（约 1 秒）打印一次音量表
        self._level_frame += 1
        if self._level_frame >= 16:
            self._level_frame = 0
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            # 映射到 0-20 格的 bar，满格约 3000 RMS
            bars = min(20, int(rms / 150))
            bar = "█" * bars + "░" * (20 - bars)
            sys.stdout.write(f"\r🎤 [{bar}] {rms:6.0f}  ")
            sys.stdout.flush()

        if await self.wake_engine.detect(chunk):
            sys.stdout.write("\n")
            logger.info(f"🌟 唤醒成功！切换到监听模式")
            self.state = AssistantState.LISTENING

    async def _listen_and_process(self):
        """监听用户说话并处理"""
        # VAD录音
        audio_data = await self.vad_engine.record_until_silence()

        if not audio_data:
            # 等待说话超时，直接回到待机
            self.state = AssistantState.IDLE
            return

        # 切换到处理态
        self.state = AssistantState.PROCESSING

        # STT识别
        text = await self.stt_engine.transcribe(audio_data)

        if not text or not text.strip():
            logger.warning("未识别到有效文本")
            self.state = AssistantState.IDLE
            return

        logger.info(f"👤 用户: {text}")

        # LLM流式生成 + 句子切分
        await self._stream_llm_to_tts(text)

        # 等待TTS播放完成
        await self.sentence_queue.join()

        # 进入连续对话窗口
        self.state = AssistantState.CONTINUOUS

    async def _stream_llm_to_tts(self, user_message: str):
        """LLM流式生成并切分句子送入TTS队列"""
        splitter = SentenceSplitter()

        try:
            async for token in self.llm_engine.chat_stream(user_message):
                # 尝试切分句子
                sentence = splitter.add_token(token)
                if sentence:
                    # 将完整句子放入队列
                    await self.sentence_queue.put(sentence)

            # 刷新剩余内容
            remaining = splitter.flush()
            if remaining:
                await self.sentence_queue.put(remaining)

        except Exception as e:
            logger.error(f"LLM处理失败: {e}")
            await self.sentence_queue.put("抱歉，我遇到了一些问题。")

        finally:
            # 发送结束信号
            await self.sentence_queue.put(None)

    async def _tts_playback_loop(self):
        """TTS播放循环 - 从队列获取句子并播放"""
        while self.running:
            try:
                # 从队列获取句子
                sentence = await self.sentence_queue.get()

                if sentence is None:
                    # 结束信号
                    self.sentence_queue.task_done()
                    continue

                # 切换到播放态
                self.state = AssistantState.SPEAKING

                # TTS合成
                audio_chunks = []
                async for chunk in self.tts_engine.synthesize_stream(sentence):
                    audio_chunks.append(chunk)

                # 播放
                audio_data = b''.join(audio_chunks)
                await self.player.play_audio(audio_data)

                self.sentence_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS播放失败: {e}", exc_info=True)
                self.sentence_queue.task_done()

    async def _continuous_dialogue_window(self):
        """连续对话窗口 - 等待用户继续说话"""
        has_speech = await self.vad_engine.listen_with_timeout(
            self.continuous_window_seconds
        )

        if has_speech:
            # 继续对话
            logger.info("🔄 继续对话...")
            self.state = AssistantState.LISTENING
        else:
            # 回到待机
            logger.info("💤 回到待机状态")
            self.state = AssistantState.IDLE

    async def cleanup(self):
        """清理所有资源"""
        logger.info("正在清理资源...")

        self.running = False

        # 清理引擎
        if self.wake_engine:
            await self.wake_engine.cleanup()
        await self.vad_engine.cleanup()
        await self.stt_engine.cleanup()
        await self.llm_engine.cleanup()
        await self.tts_engine.cleanup()

        # 清理音频设备
        self.recorder.cleanup()
        self.player.cleanup()

        logger.info("资源清理完成")

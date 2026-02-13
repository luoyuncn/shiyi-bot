"""音频播放器"""
import asyncio
import subprocess
from loguru import logger
from typing import Optional


def _pulse_available() -> bool:
    try:
        return subprocess.run(
            ["pactl", "info"], capture_output=True, timeout=2
        ).returncode == 0
    except Exception:
        return False


class AudioPlayer:
    """音频播放器 - 优先走 PulseAudio（蓝牙），不可用时回退 PyAudio"""

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        device_index: Optional[int] = None,
        pulse_sink: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.pulse_sink = pulse_sink

        self._use_pulse = _pulse_available()
        self._pa_stream = None
        self._pa_audio = None

        if self._use_pulse:
            logger.info(
                f"播放器使用 PulseAudio 模式"
                + (f"，输出设备: {pulse_sink}" if pulse_sink else "（默认设备）")
            )
        else:
            logger.info("PulseAudio 不可用，使用 PyAudio 模式")

    def start(self):
        """启动播放器（PyAudio 模式下打开流）"""
        if self._use_pulse:
            # PulseAudio 模式使用子进程播放，不需要预开流
            return  # PulseAudio 模式下无需预先打开流

        import pyaudio
        self._pa_audio = pyaudio.PyAudio()
        self._pa_stream = self._pa_audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.device_index,
        )
        logger.info(f"PyAudio 播放流已启动: {self.sample_rate}Hz")

    async def play_audio(self, audio_data: bytes):
        """
        播放音频数据。

        PulseAudio 模式：audio_data 为 edge-tts 原始 MP3 数据，
                         通过 ffmpeg 解码后经 paplay 输出到 PulseAudio 设备（含蓝牙）。
        PyAudio 模式：同样先用 ffmpeg 解码 MP3 → PCM，再写入 PyAudio 流。
        """
        if not audio_data:
            return

        if self._use_pulse:
            await self._play_pulse(audio_data)
        else:
            await self._play_pyaudio(audio_data)

    async def _play_pulse(self, mp3_data: bytes):
        """ffmpeg 解码 MP3 → PCM → paplay (PulseAudio)"""
        # Step 1: ffmpeg MP3 → raw s16le PCM
        # 第一步：用 ffmpeg 把 MP3 解码为原始 s16le PCM
        ff_cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-i", "pipe:0",
            "-f", "s16le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "pipe:1",
        ]
        try:
            ff = await asyncio.create_subprocess_exec(
                *ff_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            pcm_data, _ = await asyncio.wait_for(ff.communicate(input=mp3_data), timeout=30)
        except Exception as e:
            logger.error(f"ffmpeg 解码失败: {e}")
            return

        if not pcm_data:
            logger.warning("ffmpeg 解码后无 PCM 数据")
            return

        # Step 2: paplay
        # 第二步：通过 paplay 输出到 PulseAudio
        pa_cmd = [
            "paplay",
            "--raw",
            f"--format=s16le",
            f"--rate={self.sample_rate}",
            f"--channels={self.channels}",
        ]
        if self.pulse_sink:
            pa_cmd.append(f"--device={self.pulse_sink}")

        try:
            pa = await asyncio.create_subprocess_exec(
                *pa_cmd,
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr_data = await asyncio.wait_for(pa.communicate(input=pcm_data), timeout=60)
            if pa.returncode != 0:
                err_msg = stderr_data.decode(errors="replace").strip() if stderr_data else "unknown error"
                logger.error(f"paplay 播放失败 (returncode={pa.returncode}): {err_msg}")
            else:
                logger.debug(f"播放完成: {len(pcm_data)} bytes PCM")
        except Exception as e:
            logger.error(f"paplay 播放失败: {e}")

    async def _play_pyaudio(self, mp3_data: bytes):
        """ffmpeg 解码 MP3 → PCM → PyAudio 流"""
        ff_cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-i", "pipe:0",
            "-f", "s16le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "pipe:1",
        ]
        try:
            ff = await asyncio.create_subprocess_exec(
                *ff_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            pcm_data, _ = await asyncio.wait_for(ff.communicate(input=mp3_data), timeout=30)
        except Exception as e:
            logger.error(f"ffmpeg 解码失败: {e}")
            return

        if not self._pa_stream or not self._pa_stream.is_active():
            logger.error("PyAudio 播放流未就绪")
            return

        try:
            chunk_size = 4096
            for i in range(0, len(pcm_data), chunk_size):
                self._pa_stream.write(pcm_data[i:i + chunk_size])
                if i % (chunk_size * 10) == 0:
                    await asyncio.sleep(0)
            logger.debug(f"PyAudio 播放完成: {len(pcm_data)} bytes")
        except Exception as e:
            logger.error(f"PyAudio 播放失败: {e}")

    def stop(self):
        if self._pa_stream:
            try:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
                logger.info("PyAudio 播放流已停止")
            except Exception as e:
                logger.error(f"停止播放流失败: {e}")

    def cleanup(self):
        self.stop()
        if self._pa_audio:
            self._pa_audio.terminate()
            logger.debug("PyAudio 已清理")

    def list_devices(self):
        if not self._use_pulse:
            import pyaudio
            if self._pa_audio is None:
                self._pa_audio = pyaudio.PyAudio()
            logger.info("可用的播放设备:")
            for i in range(self._pa_audio.get_device_count()):
                info = self._pa_audio.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:
                    logger.info(f"  [{i}] {info['name']} (输出通道: {info['maxOutputChannels']})")
        else:
            result = subprocess.run(["pactl", "list", "sinks", "short"], capture_output=True, text=True)
            logger.info("PulseAudio 输出设备:\n" + result.stdout)

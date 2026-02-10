"""音频录制器"""
import numpy as np
from loguru import logger
from typing import Optional
from pathlib import Path
import subprocess
import os


class AudioRecorder:
    """音频录制器 - 管理麦克风输入流"""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device_index: Optional[int] = None
    ):
        """
        初始化录音器

        Args:
            sample_rate: 采样率 (Hz)
            chunk_size: 每次读取的帧数
            channels: 声道数 (1=单声道, 2=立体声)
            device_index: 音频设备索引，None表示默认设备
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index

        self.audio = None   # 延迟初始化，避免 PipeWire 环境下 PyAudio 枚举超时
        self.stream = None
        self.backend = "pyaudio"
        self._proc = None

    @staticmethod
    def _pipewire_pulse_running() -> bool:
        """PipeWire-Pulse 独占硬件设备时，PyAudio/ALSA 直接访问会超时"""
        try:
            return subprocess.run(
                ["pgrep", "-x", "pipewire-pulse"],
                capture_output=True,
            ).returncode == 0
        except Exception:
            return False

    @staticmethod
    def _pw_record_available() -> bool:
        try:
            return subprocess.run(
                ["which", "pw-record"], capture_output=True
            ).returncode == 0
        except Exception:
            return False

    def _detect_alsa_capture_card(self) -> Optional[str]:
        cards_path = Path("/proc/asound/cards")
        pcm_path = Path("/proc/asound/pcm")
        if not cards_path.exists() or not pcm_path.exists():
            return None

        card_names = {}
        for line in cards_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            left = parts[0].strip()  # e.g. "3 [seeed2micvoicec]"
            card_idx = left.split()[0]  # numeric index
            if "[" in left and "]" in left:
                name = left.split("[", 1)[1].split("]", 1)[0].strip()
            else:
                name = card_idx
            card_names[card_idx] = name

        capture_cards = []
        for line in pcm_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "capture" not in line:
                continue
            try:
                raw_idx = line.split("-", 1)[0].strip()
                card_idx = str(int(raw_idx)) if raw_idx.isdigit() else raw_idx
                if card_idx in card_names:
                    capture_cards.append(card_names[card_idx])
            except Exception:
                continue

        if not capture_cards:
            return None

        prefer_keys = ["seeed", "respeaker", "mic", "voice"]
        for name in capture_cards:
            low = name.lower()
            if any(k in low for k in prefer_keys):
                return name
        return capture_cards[0]

    def _start_pw_record(self):
        """通过 pw-record 子进程录音（PipeWire 原生，不会超时）"""
        cmd = [
            "pw-record",
            "--rate", str(self.sample_rate),
            "--channels", str(self.channels),
            "--format", "s16",
            "-",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=os.environ.copy(),
        )
        self.backend = "arecord"
        logger.info(f"录音流已启动(pw-record): {self.sample_rate}Hz")

    def _start_arecord(self):
        """通过 arecord 子进程录音（纯 ALSA，无 PipeWire 时使用）"""
        card_name = self._detect_alsa_capture_card()
        if not card_name:
            raise RuntimeError("未找到可用的 ALSA 录音设备")
        dev = f"plughw:{card_name},0"
        cmd = [
            "arecord",
            "-D", dev,
            "-f", "S16_LE",
            "-c", str(self.channels),
            "-r", str(self.sample_rate),
            "-t", "raw",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=os.environ.copy(),
        )
        self.backend = "arecord"
        logger.info(f"录音流已启动(arecord): {self.sample_rate}Hz, 设备: {dev}")

    def start(self):
        """启动录音流"""
        import pyaudio

        # PipeWire-Pulse 运行时独占硬件，PyAudio 枚举会触发 30s 超时
        # 直接走 pw-record，无需尝试 PyAudio
        if self._pipewire_pulse_running():
            if self._pw_record_available():
                logger.info("检测到 PipeWire-Pulse，使用 pw-record 录音")
                self._start_pw_record()
                return
            # 无 pw-record 时继续尝试 PyAudio（可能超时）
            logger.warning("PipeWire-Pulse 运行中但未找到 pw-record，尝试 PyAudio...")

        # 延迟初始化 PyAudio
        if self.audio is None:
            self.audio = pyaudio.PyAudio()

        def iter_input_devices():
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    yield i, info

        def iter_alsa_input_devices():
            for ha_idx in range(self.audio.get_host_api_count()):
                ha = self.audio.get_host_api_info_by_index(ha_idx)
                if ha.get("type") != pyaudio.paALSA:
                    continue
                for j in range(ha.get("deviceCount", 0)):
                    info = self.audio.get_device_info_by_host_api_device_index(ha_idx, j)
                    if info.get("maxInputChannels", 0) > 0:
                        yield info["index"], info

        try:
            input_devices = list(iter_input_devices())
            if input_devices:
                names = ", ".join([f"{i}:{info.get('name')}" for i, info in input_devices])
                logger.info(f"检测到输入设备: {names}")

            if not input_devices:
                raise RuntimeError(
                    "未检测到可用的录音设备。请连接麦克风或启动音频服务后重试。"
                )

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=None
            )

            device_name = "默认设备"
            if self.device_index is not None:
                device_info = self.audio.get_device_info_by_index(self.device_index)
                device_name = device_info.get('name', '未知设备')

            logger.info(f"录音流已启动: {self.sample_rate}Hz, 设备: {device_name}")

        except Exception as e:
            if self.device_index is None:
                logger.warning(f"默认输入设备启动失败: {e}，尝试ALSA输入设备...")
                alsa_devices = list(iter_alsa_input_devices())
                if not alsa_devices:
                    logger.error(f"启动录音流失败: {e}")
                    raise

                preferred = None
                for idx, info in alsa_devices:
                    name = info.get("name", "").lower()
                    if any(k in name for k in ["seeed", "mic", "respeaker", "voice"]):
                        preferred = (idx, info)
                        break
                if preferred is None:
                    preferred = alsa_devices[0]

                self.device_index = preferred[0]
                logger.info(f"尝试ALSA设备索引: {self.device_index} ({preferred[1].get('name')})")
                try:
                    self.stream = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=self.device_index,
                        frames_per_buffer=self.chunk_size,
                        stream_callback=None
                    )
                    device_name = preferred[1].get("name", "未知设备")
                    logger.info(f"录音流已启动(回退ALSA): {self.sample_rate}Hz, 设备: {device_name}")
                except Exception as e2:
                    logger.warning(f"ALSA设备打开失败: {e2}，尝试子进程回退...")
                    if self._pw_record_available():
                        self._start_pw_record()
                    else:
                        self._start_arecord()
            else:
                logger.error(f"启动录音流失败: {e}")
                raise

    def read_chunk(self) -> np.ndarray:
        """
        读取一个音频块

        Returns:
            音频数据 (numpy数组, int16格式)
        """
        if not self.stream or not self.stream.is_active():
            if self.backend != "arecord":
                raise RuntimeError("录音流未启动或已停止")

        try:
            if self.backend == "arecord" and self._proc and self._proc.stdout:
                data = self._proc.stdout.read(self.chunk_size * 2 * self.channels)
                if not data:
                    raise RuntimeError("arecord录音流已停止")
            else:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            # 多声道下混为单声道
            if self.channels > 1:
                audio = audio.reshape(-1, self.channels).mean(axis=1).astype(np.int16)
            return audio
        except Exception as e:
            logger.error(f"读取音频数据失败: {e}")
            raise

    def stop(self):
        """停止录音流"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                logger.info("录音流已停止")
            except Exception as e:
                logger.error(f"停止录音流失败: {e}")
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def cleanup(self):
        """清理资源"""
        self.stop()
        if self.audio:
            self.audio.terminate()
            logger.debug("PyAudio已清理")

    def list_devices(self):
        """列出所有可用的录音设备"""
        import pyaudio
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
        logger.info("可用的录音设备:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                logger.info(f"  [{i}] {info['name']} (输入通道: {info['maxInputChannels']})")

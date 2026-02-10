"""程序主入口"""
import asyncio
import os
import signal
from pathlib import Path

# 优先使用本项目的ALSA配置（避免默认走PulseAudio导致超时）
def _detect_alsa_capture_card() -> str | None:
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
        # format: "03-00: ..." — strip leading zeros to match card_names keys
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


def _pipewire_pulse_running() -> bool:
    """检查 PipeWire-Pulse 是否正在运行（占用了硬件设备）"""
    import subprocess as _sp
    try:
        result = _sp.run(
            ["pgrep", "-x", "pipewire-pulse"],
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _ensure_alsa_config():
    if "ALSA_CONFIG_PATH" in os.environ:
        return
    # PipeWire-Pulse 运行时独占硬件，不能用 hw: 直接访问，跳过覆盖
    if _pipewire_pulse_running():
        return
    card_name = _detect_alsa_capture_card()
    if not card_name:
        return
    conf_path = Path("config/auto_asound.conf")
    conf_text = (
        "pcm.!default {\n"
        "    type plug\n"
        f"    slave.pcm \"hw:{card_name},0\"\n"
        "}\n\n"
        "ctl.!default {\n"
        "    type hw\n"
        f"    card \"{card_name}\"\n"
        "}\n"
    )
    conf_path.write_text(conf_text, encoding="utf-8")
    os.environ["ALSA_CONFIG_PATH"] = str(conf_path.resolve())


_ensure_alsa_config()
from loguru import logger
from config.settings import load_config
from utils.logger import setup_logger
from core.assistant import AssistantCore


async def main():
    """程序主函数"""
    # 检查.env文件
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ 错误: 未找到.env文件")
        print("请复制.env.example为.env并填入你的API密钥")
        print("命令: cp .env.example .env")
        return

    try:
        # 加载配置
        config = load_config()

        # 设置日志
        setup_logger(config.system.log_level)

        # 初始化助理
        assistant = AssistantCore(config)

        # 处理退出信号
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _request_stop():
            if not stop_event.is_set():
                stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except NotImplementedError:
                pass

        # 启动助理
        run_task = asyncio.create_task(assistant.start())

        # 等待退出信号
        await stop_event.wait()
        await assistant.stop()
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)

    except KeyboardInterrupt:
        logger.info("\n👋 接收到退出信号 (Ctrl+C)")

    except Exception as e:
        logger.exception(f"💥 程序异常退出: {e}")

    finally:
        if 'assistant' in locals():
            await assistant.cleanup()

        logger.info("=" * 60)
        logger.info("🏠 小跟班已关闭，再见！")
        logger.info("=" * 60)


if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())

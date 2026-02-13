"""程序主入口"""
import argparse
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
        # 例如："3 [seeed2micvoicec]"
        card_idx = left.split()[0]  # numeric index
        # 声卡数字索引
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
        # 格式如 "03-00: ..."，去掉前导零以匹配 card_names 的键
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


def _pulseaudio_running() -> bool:
    """检查 PulseAudio 是否正在运行"""
    import subprocess as _sp
    try:
        return _sp.run(["pgrep", "-x", "pulseaudio"], capture_output=True).returncode == 0
    except Exception:
        return False


def _ensure_alsa_config():
    if "ALSA_CONFIG_PATH" in os.environ:
        return
    # PulseAudio/PipeWire-Pulse 运行时独占硬件，不能用 hw: 直接访问，跳过覆盖
    if _pipewire_pulse_running() or _pulseaudio_running():
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
from core.orchestrator import Orchestrator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShiYi — 智能助手")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用 TUI debug 模式（底部日志面板）",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="禁用 TUI，使用原始 CLI 模式",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="启动前重置所有记忆数据（从模板恢复，如无模板则恢复默认值）",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace):
    """程序主函数"""
    # 检查.env文件
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ 错误: 未找到.env文件")
        print("请复制.env.example为.env并填入你的API密钥")
        print("命令: cp .env.example .env")
        return

    tui_mode = not args.no_tui
    debug_mode = args.debug

    try:
        # 加载配置
        config = load_config()

        # 设置日志
        if tui_mode:
            # TUI 模式：抑制 stdout 日志，仅保留文件日志
            # debug 面板会通过自定义 sink 接收日志
            # debug panel 会通过自定义 sink 接收日志
            setup_logger(config.system.log_level, suppress_stdout=True)
        else:
            setup_logger(config.system.log_level)

        # 初始化 Orchestrator（传入 TUI 参数）
        orchestrator = Orchestrator(config, tui_mode=tui_mode, debug_mode=debug_mode)

        if tui_mode and config.channels.get("cli", {}).get("enabled", True):
            # TUI 模式：Textual App 接管事件循环
            # 先初始化核心组件
            await orchestrator.initialize_core()

            # --reset: wipe memory before launching TUI
            if args.reset:
                result = await orchestrator.session_manager.reset_all_memory()
                tpl = "（从模板恢复）" if result["used_template"] else "（恢复默认值）"
                logger.info(f"[--reset] 记忆已重置 {tpl}: {result['restored_files']}")

            # 启动非 CLI 通道（如语音通道）作为后台任务
            bg_task = None
            if orchestrator.channels:
                async def _run_channels():
                    await asyncio.gather(
                        *[ch.start() for ch in orchestrator.channels],
                        return_exceptions=True,
                    )
                bg_task = asyncio.create_task(_run_channels())

            from channels.tui.app import ShiYiApp

            app = ShiYiApp(
                config=config,
                session_manager=orchestrator.session_manager,
                agent_core=orchestrator.agent_core,
                debug=debug_mode,
            )

            try:
                await app.run_async()
            finally:
                if bg_task:
                    bg_task.cancel()
                    await asyncio.gather(bg_task, return_exceptions=True)
                await orchestrator.stop()
        else:
            # 原始模式：Orchestrator 管理所有 channel
            loop = asyncio.get_running_loop()
            stop_event = asyncio.Event()

            def _request_stop():
                if not stop_event.is_set():
                    stop_event.set()

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, _request_stop)
                except (NotImplementedError, OSError):
                    try:
                        signal.signal(sig, lambda *_: _request_stop())
                    except (OSError, ValueError):
                        pass

            run_task = asyncio.create_task(orchestrator.start())
            run_task.add_done_callback(lambda _: _request_stop())

            await stop_event.wait()
            await orchestrator.stop()
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)

    except KeyboardInterrupt:
        logger.info("\n👋 接收到退出信号 (Ctrl+C)")

    except Exception as e:
        logger.exception(f"💥 程序异常退出: {e}")

    finally:
        logger.info("🏠 Shiyi已关闭，再见！")


def run():
    """Entry point for `shiyi` CLI command."""
    args = _parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    run()

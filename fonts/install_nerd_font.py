#!/usr/bin/env python3
"""Nerd Font 下载助手 — 下载推荐的 JetBrainsMono Nerd Font 到本地。

Usage:
    python fonts/install_nerd_font.py [font_name]

Examples:
    python fonts/install_nerd_font.py                  # 默认: JetBrainsMono
    python fonts/install_nerd_font.py FiraCode          # 下载 FiraCode Nerd Font
    python fonts/install_nerd_font.py CascadiaCode      # 下载 CascadiaCode Nerd Font
"""
import os
import sys
import platform
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Nerd Fonts release base URL (latest)
_BASE_URL = "https://github.com/ryanoasis/nerd-fonts/releases/latest/download"

# Supported font names
FONTS = {
    "JetBrainsMono": "JetBrainsMono.zip",
    "FiraCode": "FiraCode.zip",
    "CascadiaCode": "CascadiaCode.zip",
    "Hack": "Hack.zip",
    "Meslo": "Meslo.zip",
}


def _download(font_name: str, dest_dir: Path) -> Path:
    """Download a Nerd Font zip to dest_dir. Returns the zip path."""
    filename = FONTS.get(font_name)
    if not filename:
        print(f"未知字体: {font_name}")
        print(f"可选: {', '.join(FONTS.keys())}")
        sys.exit(1)

    url = f"{_BASE_URL}/{filename}"
    zip_path = dest_dir / filename
    dest_dir.mkdir(parents=True, exist_ok=True)

    if zip_path.exists():
        print(f"已存在: {zip_path}")
        return zip_path

    print(f"正在下载 {font_name} Nerd Font ...")
    print(f"  URL: {url}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urlretrieve(url, zip_path, reporthook=_progress)
    print()  # newline after progress bar
    return zip_path


def _extract(zip_path: Path, font_dir: Path) -> None:
    """Extract .ttf / .otf files from the zip."""
    font_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        extracted = 0
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith((".ttf", ".otf")) and not name.startswith("__"):
                zf.extract(name, font_dir)
                extracted += 1
        print(f"已解压 {extracted} 个字体文件到: {font_dir}")


def _install_hint(font_dir: Path) -> None:
    """Print platform-specific install instructions."""
    system = platform.system()

    print("\n" + "=" * 50)
    print("下一步：安装字体到系统")
    print("=" * 50)

    if system == "Windows":
        print(f"\n  1. 打开文件夹: {font_dir}")
        print("  2. 全选 .ttf 文件 → 右键 → 为所有用户安装")
        print("  3. 在终端设置中选择 Nerd Font 字体")
        # Try to open the folder
        try:
            os.startfile(str(font_dir))
            print("\n  (已自动打开字体文件夹)")
        except Exception:
            pass

    elif system == "Darwin":
        target = Path.home() / "Library" / "Fonts"
        print(f"\n  cp {font_dir}/*.ttf {target}/")
        print("  然后在终端偏好设置中选择 Nerd Font 字体")

    elif system == "Linux":
        target = Path.home() / ".local" / "share" / "fonts"
        print(f"\n  cp {font_dir}/*.ttf {target}/")
        print("  fc-cache -fv")
        print("  然后在终端设置中选择 Nerd Font 字体")

    print(f"\n  最后编辑 config/config.yaml:")
    print("  tui:")
    print("    nerd_font: true")
    print()


def main():
    font_name = sys.argv[1] if len(sys.argv) > 1 else "JetBrainsMono"

    # Download to fonts/ directory (same as this script)
    script_dir = Path(__file__).parent
    zip_path = _download(font_name, script_dir)

    # Extract
    font_dir = script_dir / font_name
    _extract(zip_path, font_dir)

    # Hint
    _install_hint(font_dir)


if __name__ == "__main__":
    main()

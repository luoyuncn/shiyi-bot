"""Nerd Font icon rendering diagnostic — run this in your terminal to check which icon ranges work.

Usage:
    python fonts/test_icons.py

Run this in the SAME terminal (Windows Terminal) with the SAME font you use for ShiYi TUI.
If an icon renders as a diamond ◆, empty box □, or question mark ?, that range doesn't work.
"""
import sys
import io

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

print("=" * 60)
print("  Nerd Font Icon Rendering Diagnostic")
print("=" * 60)

groups = {
    "Powerline": [
        ("\ue0a0", "branch"),
        ("\ue0b0", "right_hard_divider"),
        ("\ue0b2", "left_hard_divider"),
    ],
    "Powerline Extra": [
        ("\ue0b4", "right_half_circle"),
        ("\ue0b6", "left_half_circle"),
        ("\ue0c0", "flame_thick"),
        ("\ue0c6", "pixelated_right"),
    ],
    "Font Awesome (FA4 core)": [
        ("\uf002", "search"),
        ("\uf005", "star"),
        ("\uf007", "user"),
        ("\uf017", "clock"),
        ("\uf058", "check_circle"),
        ("\uf057", "times_circle"),
        ("\uf061", "arrow_right"),
        ("\uf071", "warning"),
        ("\uf0ad", "wrench"),
        ("\uf0c1", "link"),
        ("\uf110", "spinner"),
        ("\uf111", "circle"),
        ("\uf120", "terminal"),
        ("\uf15b", "file"),
    ],
    "Font Awesome (FA5 extended)": [
        ("\uf544", "robot"),
        ("\uf53f", "code_branch"),
    ],
    "Devicons": [
        ("\ue718", "python"),
        ("\ue781", "react"),
    ],
    "Seti-UI": [
        ("\ue5fa", "config"),
        ("\ue5fb", "folder"),
    ],
    "Codicons": [
        ("\uea7b", "account"),
        ("\ueb44", "gear"),
        ("\ueba4", "sparkle"),
        ("\uea86", "check"),
        ("\uea87", "close"),
        ("\ueb2d", "tools"),
    ],
    "Octicons": [
        ("\uf408", "search_oct"),
        ("\uf41b", "gear_oct"),
    ],
    "Weather Icons": [
        ("\ue300", "cloud"),
    ],
}

for group_name, icons in groups.items():
    print(f"\n  {group_name}:")
    for char, name in icons:
        codepoint = f"U+{ord(char):04X}"
        print(f"    {name:25s} [{char}]  {codepoint}")

print("\n" + "=" * 60)
print("  If icons show as ◆ □ ? or empty, that range is broken.")
print("  Working icons will show distinct shapes/symbols.")
print("=" * 60)

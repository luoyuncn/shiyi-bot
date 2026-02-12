"""Dual-mode icon manager — Nerd Font glyphs vs ASCII/emoji fallback.

When ``nerd_font=True`` the TUI uses Nerd Font patched glyphs (requires the
user to have a Nerd Font installed in their terminal emulator).  Otherwise
plain Unicode emoji / ASCII symbols are used so everything works out-of-box.

Usage::

    from channels.tui.icons import icons, init_icons
    init_icons(use_nerd_font=True)   # call once at app startup
    print(icons.assistant)           # 🍊  or  nf-md-robot
"""


class _Icons:
    """Singleton icon set — switch between Nerd Font and fallback."""

    # ── Fallback (emoji / ASCII) ──────────────────────────
    _FALLBACK = {
        # Brand / identity
        "assistant": "\U0001f34a",      # 🍊
        "user": "\u276f",               # ❯
        "app_name": "ShiYi",

        # Status
        "thinking": "\u2026",           # …
        "success": "\u2714",            # ✔
        "error": "\u2718",              # ✘
        "warning": "\u26a0",            # ⚠

        # Navigation / actions
        "prompt": "\u276f",             # ❯
        "arrow_right": "\u25b8",        # ▸
        "bullet": "\u2022",             # •
        "separator": "\u2500",          # ─

        # Tool / system
        "tool": "\u2692",               # ⚒
        "search": "\u2315",             # ⌕
        "file": "\u2630",               # ☰
        "shell": "$",
        "clock": "\u231a",              # ⌚

        # Misc
        "sparkle": "\u2728",            # ✨ (actually emoji)
        "link": "\u2192",               # →
        "node_start": "\u25c6",         # ◆
        "node_end": "\u25c6",           # ◆
    }

    # ── Nerd Font glyphs (BMP range — Font Awesome / Powerline) ──
    # Uses U+E000-U+F2E0 codepoints for maximum terminal compatibility.
    # Avoid Supplementary PUA-A (U+F0000+) — many terminals can't render those.
    _NERD = {
        # Brand / identity
        "assistant": "\uf0d0",          #  nf-fa-magic (wand/sparkle)
        "user": "\uf007",              #  nf-fa-user
        "app_name": "ShiYi",

        # Status
        "thinking": "\uf110",          #  nf-fa-spinner
        "success": "\uf058",           #  nf-fa-check_circle
        "error": "\uf057",             #  nf-fa-times_circle
        "warning": "\uf071",           #  nf-fa-exclamation_triangle

        # Navigation / actions
        "prompt": "\ue0b0",            #  nf-pl-right_hard_divider
        "arrow_right": "\uf061",       #  nf-fa-arrow_right
        "bullet": "\uf111",            #  nf-fa-circle
        "separator": "\u2500",          # ─ (same)

        # Tool / system
        "tool": "\uf0ad",              #  nf-fa-wrench
        "search": "\uf002",            #  nf-fa-search
        "file": "\uf15b",              #  nf-fa-file
        "shell": "\uf120",             #  nf-fa-terminal
        "clock": "\uf017",             #  nf-fa-clock_o

        # Misc
        "sparkle": "\uf005",           #  nf-fa-star
        "link": "\uf0c1",              #  nf-fa-link
        "node_start": "\ue0b6",        #  nf-ple-left_half_circle_thick
        "node_end": "\ue0b4",          #  nf-ple-right_half_circle_thick
    }

    def __init__(self):
        self._use_nerd = False
        self._map = self._FALLBACK

    def set_nerd_font(self, enabled: bool) -> None:
        """Switch icon set at runtime."""
        self._use_nerd = enabled
        self._map = self._NERD if enabled else self._FALLBACK

    @property
    def is_nerd_font(self) -> bool:
        return self._use_nerd

    def __getattr__(self, name: str) -> str:
        # Allow attribute-style access: icons.assistant
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._map[name]
        except KeyError:
            raise AttributeError(f"No icon named '{name}'") from None

    def get(self, name: str, default: str = "?") -> str:
        """Dict-style access with fallback."""
        return self._map.get(name, default)


# Module-level singleton
icons = _Icons()


def init_icons(use_nerd_font: bool = False) -> _Icons:
    """Initialise the global icon set. Call once at app startup."""
    icons.set_nerd_font(use_nerd_font)
    return icons

"""Markdown document storage for long-term memory summaries."""

from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path


class MemoryDocumentStore:
    """Manage memory markdown documents under a configurable root directory."""

    def __init__(self, memory_root: str = "data/memory"):
        self.root = Path(memory_root)
        self.system_dir = self.root / "system"
        self.shared_dir = self.root / "shared"

        self.shiyi_path = self.system_dir / "ShiYi.md"
        self.identity_state_path = self.system_dir / "IdentityState.md"
        self.user_path = self.shared_dir / "User.md"
        self.project_path = self.shared_dir / "Project.md"
        self.insights_path = self.shared_dir / "Insights.md"

    def ensure_initialized(self):
        """Ensure all baseline folders and markdown files exist."""
        self.system_dir.mkdir(parents=True, exist_ok=True)
        self.shared_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_file(self.shiyi_path, self._default_shiyi())
        self._ensure_file(self.identity_state_path, self._default_identity_state())
        self._ensure_file(self.user_path, self._default_user())
        self._ensure_file(self.project_path, self._default_project())
        self._ensure_file(self.insights_path, self._default_insights())

    def read_identity_state(self) -> dict:
        """Read explicit identity confirmation state from dedicated markdown file."""
        self.ensure_initialized()
        text = self.identity_state_path.read_text(encoding="utf-8")
        matched = re.search(r"identity_confirmed:\s*(true|false)", text, flags=re.IGNORECASE)
        display_name_match = re.search(r"display_name:\s*(.*)", text)
        confirmed = None
        if matched:
            confirmed = matched.group(1).lower() == "true"
        display_name = display_name_match.group(1).strip() if display_name_match else ""
        return {
            "identity_confirmed": confirmed,
            "display_name": display_name or None,
        }

    def write_identity_state(self, identity_confirmed: bool, display_name: str | None = None):
        """Persist explicit identity confirmation marker."""
        self.ensure_initialized()
        display = (display_name or "").strip()
        content = (
            "# IdentityState\n\n"
            f"identity_confirmed: {'true' if identity_confirmed else 'false'}\n"
            f"display_name: {display}\n"
            f"updated_at: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        self._atomic_write(self.identity_state_path, content)

    def write_initial_identity(self, shiyi_identity: str, user_identity: str):
        """Persist first-run identity definitions for assistant and user."""
        self.ensure_initialized()
        self._atomic_write(
            self.shiyi_path,
            "# ShiYi\n\n## 核心身份\n\n"
            f"{shiyi_identity.strip()}\n",
        )
        self._atomic_write(
            self.user_path,
            "# User\n\n## 用户画像\n\n"
            f"{user_identity.strip()}\n",
        )

    def build_system_memory_card(self, max_chars: int = 1400) -> str:
        """Build a compact system prompt memory card from markdown docs."""
        self.ensure_initialized()
        shiyi_text = self._trim_text(self.shiyi_path.read_text(encoding="utf-8"), max_chars // 3)
        user_text = self._trim_text(self.user_path.read_text(encoding="utf-8"), max_chars // 3)
        project_text = self._trim_text(self.project_path.read_text(encoding="utf-8"), max_chars // 6)
        insights_text = self._trim_text(self.insights_path.read_text(encoding="utf-8"), max_chars // 6)
        return (
            "以下是长期记忆卡片，请作为高优先级上下文参考。\n\n"
            "[ShiYi]\n"
            f"{shiyi_text}\n\n"
            "[User]\n"
            f"{user_text}\n\n"
            "[Project]\n"
            f"{project_text}\n\n"
            "[Insights]\n"
            f"{insights_text}"
        )

    def upsert_user_fact(self, fact_key: str, fact_value: str):
        """Upsert a user profile fact as a single bullet line."""
        self.ensure_initialized()
        key = fact_key.strip()
        value = str(fact_value).strip()
        target_line = f"- {key}: {value}"

        lines = self.user_path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(f"- {key}:"):
                lines[i] = target_line
                self._atomic_write(self.user_path, "\n".join(lines).rstrip() + "\n")
                return

        if lines and lines[-1].strip():
            lines.append("")
        lines.append(target_line)
        self._atomic_write(self.user_path, "\n".join(lines).rstrip() + "\n")

    def append_project_update(self, update: str, max_lines: int = 100):
        """Append project update with rolling summary metabolism."""
        self.ensure_initialized()
        text = self.project_path.read_text(encoding="utf-8")
        bullet_lines = [line for line in text.splitlines() if line.startswith("- ")]
        bullet_lines.append(f"- {update.strip()}")

        if len(bullet_lines) > max_lines:
            archived_count = len(bullet_lines) - 60
            keep_lines = bullet_lines[-60:]
            summary = (
                f"- 历史归档摘要: 已归档 {archived_count} 条进展，"
                f"归档时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            bullet_lines = [summary, *keep_lines]

        content = "# Project\n\n## 当前阶段\n\n" + "\n".join(bullet_lines) + "\n"
        self._atomic_write(self.project_path, content)

    def add_insight(self, insight: str, hot_limit: int = 10):
        """Insert one insight and keep only top N hot entries."""
        self.ensure_initialized()
        text = self.insights_path.read_text(encoding="utf-8")
        current = [line[2:].strip() for line in text.splitlines() if line.startswith("- ")]
        candidate = insight.strip()
        if not candidate:
            return

        ordered = [candidate, *[item for item in current if item != candidate]]
        hot_items = ordered[:hot_limit]
        lines = "\n".join(f"- {item}" for item in hot_items)
        content = "# Insights\n\n## 热点经验\n\n" + lines + "\n"
        self._atomic_write(self.insights_path, content)

    @staticmethod
    def _atomic_write(path: Path, content: str):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)

    @staticmethod
    def _ensure_file(path: Path, content: str):
        if path.exists():
            return
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _default_shiyi() -> str:
        return (
            "# ShiYi\n\n"
            "## 核心身份\n\n"
            "- 待初始化\n"
        )

    @staticmethod
    def _default_identity_state() -> str:
        return (
            "# IdentityState\n\n"
            "identity_confirmed: \n"
            "display_name: \n"
            "updated_at: \n"
        )

    @staticmethod
    def _default_user() -> str:
        return (
            "# User\n\n"
            "## 用户画像\n\n"
            "- 待初始化\n"
        )

    @staticmethod
    def _default_project() -> str:
        return (
            "# Project\n\n"
            "## 当前阶段\n\n"
            "- 暂无项目摘要\n"
        )

    @staticmethod
    def _default_insights() -> str:
        return (
            "# Insights\n\n"
            "## 热点经验\n\n"
            "- 暂无经验条目\n"
        )

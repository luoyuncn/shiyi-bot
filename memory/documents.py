"""Markdown document storage for long-term memory summaries."""

from __future__ import annotations

import shutil
from datetime import datetime
import re
from pathlib import Path

import yaml


class MemoryDocumentStore:
    """Manage memory markdown documents under a configurable root directory."""

    _USER_HARD_KEY_MAP = {
        "display_name": "display_name",
        "nickname": "display_name",
        "role": "role",
        "location": "location",
        "habit": "work_style",
        "work_style": "work_style",
        "profession": "profession",
        "preferred_style": "preferred_style",
    }
    _SHIYI_HARD_KEY_MAP = {
        "name": "name",
        "version": "version",
        "persona": "persona",
        "identity": "persona",
        "tone": "tone",
        "boundary": "boundaries",
        "boundaries": "boundaries",
        "constraint": "constraints",
        "constraints": "constraints",
    }

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
        now = datetime.now().isoformat(timespec="seconds")

        shiyi_meta, _ = self._read_document(self.shiyi_path)
        shiyi_meta.setdefault("name", "ShiYi")
        shiyi_meta.setdefault("version", "2.0")
        shiyi_meta["persona"] = shiyi_identity.strip()
        shiyi_meta["updated_at"] = now
        shiyi_body = (
            "# ShiYi\n\n"
            "## 核心指令\n\n"
            f"- {shiyi_identity.strip()}\n"
        )
        self._write_document(self.shiyi_path, shiyi_meta, shiyi_body)

        user_meta, _ = self._read_document(self.user_path)
        user_meta["updated_at"] = now
        display_name = self._extract_display_name(user_identity)
        if display_name:
            user_meta["display_name"] = display_name
        user_body = (
            "# User\n\n"
            "## 用户画像\n\n"
            f"- {user_identity.strip()}\n"
        )
        self._write_document(self.user_path, user_meta, user_body)

    def build_system_memory_card(self, max_chars: int = 1400) -> str:
        """Build a compact system prompt memory card from markdown docs."""
        self.ensure_initialized()
        shiyi_text = self._trim_text(self.shiyi_path.read_text(encoding="utf-8"), max_chars // 3)
        user_text = self._trim_text(self.user_path.read_text(encoding="utf-8"), max_chars // 3)
        project_text = self._trim_text(self.project_path.read_text(encoding="utf-8"), max_chars // 6)
        insights_text = self._trim_text(self.insights_path.read_text(encoding="utf-8"), max_chars // 6)
        # Expose actual paths so the LLM uses correct locations if it ever needs to call file tools.
        # 暴露实际路径，避免 LLM 用工具时乱猜路径。
        path_hint = (
            f"[记忆文件路径（相对工作目录）]\n"
            f"  ShiYi   : {self.shiyi_path}\n"
            f"  User    : {self.user_path}\n"
            f"  Project : {self.project_path}\n"
            f"  Insights: {self.insights_path}\n"
            "以上内容已完整注入，无需用工具重复读取。\n"
        )
        return (
            "以下是长期记忆卡片，请作为高优先级上下文参考。\n\n"
            f"{path_hint}\n"
            "[ShiYi]\n"
            f"{shiyi_text}\n\n"
            "[User]\n"
            f"{user_text}\n\n"
            "[Project]\n"
            f"{project_text}\n\n"
            "[Insights]\n"
            f"{insights_text}"
        )

    # ── Template management ──────────────────────────────────
    # Template dir: data/memory/template/
    # Template files: ShiYiTemplate.md / UserTemplate.md / ProjectTemplate.md / InsightsTemplate.md

    def get_template_dir(self) -> Path:
        return self.root / "template"

    # Mapping: template filename → target MD path
    def _template_mappings(self) -> list[tuple[Path, Path, str]]:
        """Returns list of (template_filename, target_md_path, fallback_default)."""
        tpl_dir = self.get_template_dir()
        return [
            (tpl_dir / "ShiYiTemplate.md",   self.shiyi_path,    self._default_shiyi()),
            (tpl_dir / "UserTemplate.md",    self.user_path,     self._default_user()),
            (tpl_dir / "ProjectTemplate.md", self.project_path,  self._default_project()),
            (tpl_dir / "InsightsTemplate.md",self.insights_path, self._default_insights()),
        ]

    def save_as_template(self) -> list[str]:
        """Copy current 4 MD files back to template/ directory. Returns saved template file names."""
        self.ensure_initialized()
        tpl_dir = self.get_template_dir()
        tpl_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for tpl_path, src_path, _ in self._template_mappings():
            if src_path.exists():
                shutil.copy2(src_path, tpl_path)
                saved.append(tpl_path.name)
        return saved

    def restore_from_template(self) -> list[str]:
        """Copy template files → actual MD files. Falls back to built-in defaults if template missing.
        Also resets IdentityState.md to default. Returns list of restored MD file names."""
        self.ensure_initialized()
        restored = []
        for tpl_path, dst_path, default in self._template_mappings():
            if tpl_path.exists():
                shutil.copy2(tpl_path, dst_path)
            else:
                dst_path.write_text(default, encoding="utf-8")
            restored.append(dst_path.name)
        # Always reset identity state
        self._atomic_write(self.identity_state_path, self._default_identity_state())
        return restored

    def has_template(self) -> bool:
        """Return True if at least one template file exists."""
        return any(tpl.exists() for tpl, _, _ in self._template_mappings())

    # ── Rebuild from DB ──────────────────────────────────────

    def rebuild_user_md_from_facts(self, facts: list[dict]):
        """Rebuild User.md from active DB memory_facts on startup to prevent amnesia after restart."""
        self.ensure_initialized()
        meta, _ = self._read_document(self.user_path)
        body = "# User\n\n## 用户画像\n\n"
        for fact in facts:
            fact_key = fact.get("fact_key", "")
            fact_value = fact.get("fact_value", "")
            if not fact_key or not fact_value:
                continue
            applied = self._apply_user_hard_field(meta, fact_key, fact_value)
            if not applied:
                body += f"- {fact_key}: {fact_value}\n"
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_document(self.user_path, meta, body)

    def rebuild_shiyi_md_from_facts(self, facts: list[dict]):
        """Rebuild ShiYi.md from active DB memory_facts (scope=system) on startup."""
        self.ensure_initialized()
        meta, body = self._read_document(self.shiyi_path)
        for fact in facts:
            fact_key = fact.get("fact_key", "")
            fact_value = fact.get("fact_value", "")
            if not fact_key or not fact_value:
                continue
            applied = self._apply_shiyi_hard_field(meta, fact_key, fact_value)
            if not applied:
                body = self._upsert_key_value_bullet(body, fact_key, fact_value)
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_document(self.shiyi_path, meta, body)

    def rebuild_insights_md_from_facts(self, insights: list[str]):
        """Rebuild Insights.md from active DB memory_facts (scope=insight) on startup."""
        self.ensure_initialized()
        if not insights:
            return
        meta, _ = self._read_document(self.insights_path)
        hot_items = insights[:10]  # keep top 10
        lines = "\n".join(f"- {item}" for item in hot_items)
        content = "# Insights\n\n## 热点经验\n\n" + lines + "\n"
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_document(self.insights_path, meta, content)

    def rebuild_project_md_from_facts(self, updates: list[str]):
        """Rebuild Project.md from active DB memory_facts (scope=project) on startup."""
        self.ensure_initialized()
        if not updates:
            return
        meta, _ = self._read_document(self.project_path)
        bullet_lines = [f"- {u}" for u in updates]
        content = "# Project\n\n## 当前阶段\n\n" + "\n".join(bullet_lines) + "\n"
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_document(self.project_path, meta, content)

    def upsert_user_fact(self, fact_key: str, fact_value: str):
        """Apply one structured patch to User.md with hard/soft field separation."""
        self.ensure_initialized()
        key = fact_key.strip()
        value = str(fact_value).strip()
        if not key or not value:
            return

        meta, body = self._read_document(self.user_path)
        applied_hard = self._apply_user_hard_field(meta, key, value)
        if not applied_hard:
            body = self._upsert_key_value_bullet(body, key, value)

        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_document(self.user_path, meta, body)

    def upsert_shiyi_fact(self, fact_key: str, fact_value: str):
        """Apply one structured patch to ShiYi.md with hard/soft field separation."""
        self.ensure_initialized()
        key = fact_key.strip()
        value = str(fact_value).strip()
        if not key or not value:
            return

        meta, body = self._read_document(self.shiyi_path)
        applied_hard = self._apply_shiyi_hard_field(meta, key, value)
        if not applied_hard:
            body = self._upsert_key_value_bullet(body, key, value)

        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_document(self.shiyi_path, meta, body)

    def append_project_update(self, update: str, max_lines: int = 100):
        """Append project update with rolling summary metabolism."""
        self.ensure_initialized()
        meta, body = self._read_document(self.project_path)
        bullet_lines = [line for line in body.splitlines() if line.startswith("- ")]
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
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_document(self.project_path, meta, content)

    def add_insight(self, insight: str, hot_limit: int = 10):
        """Insert one insight and keep only top N hot entries."""
        self.ensure_initialized()
        meta, body = self._read_document(self.insights_path)
        current = [line[2:].strip() for line in body.splitlines() if line.startswith("- ")]
        candidate = insight.strip()
        if not candidate:
            return

        ordered = [candidate, *[item for item in current if item != candidate]]
        hot_items = ordered[:hot_limit]
        lines = "\n".join(f"- {item}" for item in hot_items)
        content = "# Insights\n\n## 热点经验\n\n" + lines + "\n"
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_document(self.insights_path, meta, content)

    def _apply_user_hard_field(self, meta: dict, key: str, value: str) -> bool:
        mapped = self._USER_HARD_KEY_MAP.get(key)
        if mapped:
            meta[mapped] = value
            return True

        if key in {"preferred_tech", "tech_stack"}:
            current = meta.get("tech_stack", [])
            if isinstance(current, str):
                current_items = [item.strip() for item in re.split(r"[，,;/|]+", current) if item.strip()]
            elif isinstance(current, list):
                current_items = [str(item).strip() for item in current if str(item).strip()]
            else:
                current_items = []

            incoming = [item.strip() for item in re.split(r"[，,;/|]+", value) if item.strip()]
            if not incoming:
                return True

            merged = [*current_items]
            for item in incoming:
                if item not in merged:
                    merged.append(item)
            meta["tech_stack"] = merged
            return True

        return False

    def _apply_shiyi_hard_field(self, meta: dict, key: str, value: str) -> bool:
        mapped = self._SHIYI_HARD_KEY_MAP.get(key)
        if not mapped:
            return False

        if mapped in ("boundaries", "constraints"):
            items = [item.strip() for item in re.split(r"[，,;/|]+", value) if item.strip()]
            meta["constraints"] = items
            return True

        meta[mapped] = value
        return True

    @staticmethod
    def _upsert_key_value_bullet(body: str, key: str, value: str) -> str:
        lines = body.splitlines()
        if not lines:
            lines = ["# User", "", "## 用户画像", ""]

        lines = [line for line in lines if line.strip() != "- 待初始化"]
        target_line = f"- {key}: {value}"
        for index, line in enumerate(lines):
            if line.strip().startswith(f"- {key}:"):
                lines[index] = target_line
                return "\n".join(lines).rstrip() + "\n"

        if lines and lines[-1].strip():
            lines.append("")
        lines.append(target_line)
        return "\n".join(lines).rstrip() + "\n"

    def _read_document(self, path: Path) -> tuple[dict, str]:
        text = path.read_text(encoding="utf-8")
        meta, body = self._split_frontmatter(text)
        return meta, body

    def _write_document(self, path: Path, meta: dict, body: str):
        serialized_meta = yaml.safe_dump(meta or {}, allow_unicode=True, sort_keys=False).strip()
        normalized_body = body.rstrip() + "\n"
        content = f"---\n{serialized_meta}\n---\n{normalized_body}"
        self._atomic_write(path, content)

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[dict, str]:
        normalized = text.replace("\r\n", "\n")
        if not normalized.startswith("---\n"):
            return {}, normalized

        end_index = normalized.find("\n---\n", 4)
        if end_index < 0:
            return {}, normalized

        raw_meta = normalized[4:end_index]
        body = normalized[end_index + len("\n---\n") :]
        try:
            parsed = yaml.safe_load(raw_meta) or {}
            if not isinstance(parsed, dict):
                parsed = {}
        except yaml.YAMLError:
            parsed = {}
        return parsed, body

    @staticmethod
    def _extract_display_name(identity_text: str) -> str | None:
        match = re.search(r"我(?:叫|是)\s*([^\s，。！？,!.?]{1,20})", identity_text or "")
        if not match:
            return None
        return match.group(1).strip()

    @staticmethod
    def _atomic_write(path: Path, content: str):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        try:
            tmp_path.replace(path)
        except PermissionError:
            # Windows can reject replace when the target is temporarily locked
            # Windows 下若目标文件被临时占用，replace 可能失败。
            # by another reader; fall back to direct write to preserve progress.
            # 此时回退为直接写入，避免进度丢失。
            path.write_text(content, encoding="utf-8")
            if tmp_path.exists():
                tmp_path.unlink()

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
            "---\n"
            "name: ShiYi\n"
            "version: '2.0'\n"
            "persona: ''\n"
            "tone: ''\n"
            "constraints: []\n"
            "updated_at: ''\n"
            "---\n"
            "# ShiYi\n\n"
            "## 人设配置\n\n"
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
            "---\n"
            "display_name: ''\n"
            "profession: ''\n"
            "tech_stack: []\n"
            "location: ''\n"
            "preferred_style: ''\n"
            "updated_at: ''\n"
            "---\n"
            "# User\n\n"
            "## 用户画像\n\n"
            "- 待初始化\n"
        )

    @staticmethod
    def _default_project() -> str:
        return (
            "---\n"
            "updated_at: ''\n"
            "---\n"
            "# Project\n\n"
            "## 当前阶段\n\n"
            "- 暂无项目摘要\n"
        )

    @staticmethod
    def _default_insights() -> str:
        return (
            "---\n"
            "updated_at: ''\n"
            "---\n"
            "# Insights\n\n"
            "## 热点经验\n\n"
            "- 暂无经验条目\n"
        )

"""Markdown write gatekeeper for structured memory facts.

Controls what can be persisted into long-term markdown files (User/Project/Insights/ShiYi)
while still allowing facts to be stored in SQLite/Kuzu for retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class MarkdownGateDecision:
    """Decision for whether a fact can be written into markdown docs."""

    allow_md_write: bool
    reason: str
    target_doc: str


class MarkdownGatekeeper:
    """Rule-based gate for MD writes to avoid document pollution."""

    USER_BLACKLIST = {
        "today_activity",
        "daily_activity",
        "today_meal",
        "meal_record",
        "today_food",
        "today_diet",
        "temporary_plan",
        "temporary_schedule",
        "today_mood",
    }

    USER_THRESHOLD = {
        "display_name": 0.90,
        "profession": 0.90,
        "tech_stack": 0.80,
        "preferred_tech": 0.80,
        "preferred_style": 0.80,
        "location": 0.85,
        "spouse": 0.80,
        "wedding_registration_date": 0.90,
        "wedding_anniversary_date": 0.90,
        "role": 0.85,
        "work_style": 0.80,
        "habit": 0.80,
    }

    SYSTEM_THRESHOLD = {
        "name": 0.90,
        "persona": 0.85,
        "tone": 0.80,
        "constraint": 0.85,
        "constraints": 0.85,
    }

    _LOW_SIGNAL_PATTERN = re.compile(
        r"^(项目开始了|项目启动了|开始了|已开始|推进中|继续推进|OK|ok|好的|收到|明白)$"
    )

    _TARGET_DOC = {
        "user": "User.md",
        "system": "ShiYi.md",
        "project": "Project.md",
        "insight": "Insights.md",
    }

    def decide(
        self,
        *,
        scope: str,
        fact_key: str,
        fact_value: str,
        confidence: float,
        confirmed_by_user: bool = False,
    ) -> MarkdownGateDecision:
        """Return markdown write decision with reason for auditing."""
        scope = (scope or "").strip().lower()
        fact_key = (fact_key or "").strip()
        value = (fact_value or "").strip()
        target_doc = self._TARGET_DOC.get(scope, "unknown")
        effective_conf = 1.0 if confirmed_by_user else float(confidence)

        if scope == "user":
            return self._decide_user(
                fact_key=fact_key,
                confidence=effective_conf,
                target_doc=target_doc,
            )
        if scope == "system":
            return self._decide_system(
                fact_key=fact_key,
                confidence=effective_conf,
                target_doc=target_doc,
            )
        if scope == "project":
            return self._decide_project_or_insight(
                scope=scope,
                value=value,
                confidence=effective_conf,
                target_doc=target_doc,
            )
        if scope == "insight":
            return self._decide_project_or_insight(
                scope=scope,
                value=value,
                confidence=effective_conf,
                target_doc=target_doc,
            )

        return MarkdownGateDecision(False, f"unsupported_scope:{scope}", target_doc)

    def _decide_user(self, *, fact_key: str, confidence: float, target_doc: str) -> MarkdownGateDecision:
        if fact_key in self.USER_BLACKLIST or fact_key.startswith("today_"):
            return MarkdownGateDecision(False, f"blacklisted:{fact_key}", target_doc)

        threshold = self.USER_THRESHOLD.get(fact_key)
        if threshold is None:
            return MarkdownGateDecision(False, f"not_whitelisted:{fact_key}", target_doc)

        if confidence < threshold:
            return MarkdownGateDecision(
                False,
                f"low_confidence:{confidence:.2f}<{threshold:.2f}",
                target_doc,
            )
        return MarkdownGateDecision(True, "allowed:user_whitelist", target_doc)

    def _decide_system(self, *, fact_key: str, confidence: float, target_doc: str) -> MarkdownGateDecision:
        threshold = self.SYSTEM_THRESHOLD.get(fact_key)
        if threshold is None:
            return MarkdownGateDecision(False, f"not_whitelisted:{fact_key}", target_doc)
        if confidence < threshold:
            return MarkdownGateDecision(
                False,
                f"low_confidence:{confidence:.2f}<{threshold:.2f}",
                target_doc,
            )
        return MarkdownGateDecision(True, "allowed:system_whitelist", target_doc)

    def _decide_project_or_insight(
        self,
        *,
        scope: str,
        value: str,
        confidence: float,
        target_doc: str,
    ) -> MarkdownGateDecision:
        min_chars = 10 if scope == "project" else 8
        if len(value) < min_chars:
            return MarkdownGateDecision(False, f"too_short:{len(value)}<{min_chars}", target_doc)
        if self._LOW_SIGNAL_PATTERN.match(value):
            return MarkdownGateDecision(False, "low_signal_phrase", target_doc)
        if confidence < 0.80:
            return MarkdownGateDecision(
                False,
                f"low_confidence:{confidence:.2f}<0.80",
                target_doc,
            )
        return MarkdownGateDecision(True, f"allowed:{scope}", target_doc)

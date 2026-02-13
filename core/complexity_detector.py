"""Complexity detector - heuristic rules to trigger planning mode (no LLM call)."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger


class ComplexityDetector:
    """
    基于启发式规则检测用户消息是否属于复杂任务。
    所有规则从 config.yaml 的 complexity_detector 节点读取，不硬编码。
    触发时在 system prompt 追加任务规划指引，不额外调用 LLM。
    """

    _PLANNING_HINT = (
        "[任务规划模式]\n"
        "这是一个多步任务。请先在回复开头列出执行计划：\n"
        "  计划：\n"
        "  1. <步骤一>\n"
        "  2. <步骤二>\n"
        "  ...\n"
        "然后立即开始执行，无需等待用户确认。"
    )

    def __init__(self, config: Any) -> None:
        self._cfg = self._load_config(config)

    @staticmethod
    def _load_config(config: Any) -> dict:
        """读取 complexity_detector 配置，返回标准化 dict。"""
        agent_cfg = getattr(config, "agent", None)
        if isinstance(agent_cfg, dict):
            raw = agent_cfg.get("complexity_detector", {})
        elif agent_cfg is not None:
            raw = getattr(agent_cfg, "complexity_detector", {})
        else:
            raw = {}

        if not isinstance(raw, dict):
            raw = {}

        return {
            "enabled": bool(raw.get("enabled", True)),
            "step_keywords": list(raw.get("step_keywords", [
                "步骤", "先.*再.*然后", "分析并", "重构", "迁移", "帮我做", "帮我搞",
            ])),
            "multi_tool_domains": raw.get("multi_tool_domains", {
                "search": ["搜索", "查一下", "找找"],
                "file":   ["文件", "代码", "读取"],
                "shell":  ["执行", "运行", "命令"],
            }),
            "multi_tool_threshold": int(raw.get("multi_tool_threshold", 2)),
            "message_length_threshold": int(raw.get("message_length_threshold", 80)),
            "continuation_markers": list(raw.get("continuation_markers", [
                "接下来", "第一步", "下一步",
            ])),
        }

    def _get_last_user_message(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return str(content).strip() if content else ""
        return ""

    def _get_last_assistant_message(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                return str(content).strip() if content else ""
        return ""

    def _check_step_keywords(self, text: str) -> bool:
        for pattern in self._cfg["step_keywords"]:
            try:
                if re.search(pattern, text):
                    return True
            except re.error:
                if pattern in text:
                    return True
        return False

    def _check_multi_tool_domains(self, text: str) -> bool:
        domains = self._cfg["multi_tool_domains"]
        threshold = self._cfg["multi_tool_threshold"]
        hit_count = 0
        for _domain, keywords in domains.items():
            for kw in keywords:
                if kw in text:
                    hit_count += 1
                    break
            if hit_count >= threshold:
                return True
        return False

    def _check_length(self, text: str) -> bool:
        return len(text) > self._cfg["message_length_threshold"]

    def _check_continuation(self, messages: list[dict]) -> bool:
        last_assistant = self._get_last_assistant_message(messages)
        if not last_assistant:
            return False
        for marker in self._cfg["continuation_markers"]:
            if marker in last_assistant:
                return True
        return False

    def is_complex(self, messages: list[dict]) -> bool:
        """判断是否属于复杂多步任务。"""
        if not self._cfg["enabled"]:
            return False

        user_msg = self._get_last_user_message(messages)
        if not user_msg:
            return False

        if self._check_step_keywords(user_msg):
            logger.debug("[ComplexityDetector] 触发：步骤关键词")
            return True

        if self._check_multi_tool_domains(user_msg):
            logger.debug("[ComplexityDetector] 触发：多工具领域关键词")
            return True

        if self._check_length(user_msg):
            logger.debug("[ComplexityDetector] 触发：消息长度超阈值")
            return True

        if self._check_continuation(messages):
            logger.debug("[ComplexityDetector] 触发：上轮助手回复包含未完成标记")
            return True

        return False

    def get_planning_hint(self, messages: list[dict]) -> str | None:
        """若为复杂任务，返回任务规划提示词；否则返回 None。"""
        if self.is_complex(messages):
            return self._PLANNING_HINT
        return None

"""Context builder - token-budgeted sliding window for LLM messages."""

from __future__ import annotations

from typing import Any

from loguru import logger


def _estimate_tokens(text: str) -> int:
    """混合中英文粗估 token 数（1 中文字 ≈ 1 token，4 英文字符 ≈ 1 token）。"""
    if not text:
        return 0
    return max(1, len(text) // 2)


def _message_tokens(msg: dict) -> int:
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return _estimate_tokens(str(content)) + 4  # role overhead


class TokenBudget:
    """简单 token 预算管理器。"""

    def __init__(self, total: int = 6000):
        self.total = total
        self._used = 0

    def can_fit(self, tokens: int) -> bool:
        return self._used + tokens <= self.total

    def consume(self, tokens: int) -> None:
        self._used += tokens

    @property
    def remaining(self) -> int:
        return max(0, self.total - self._used)


def build_history_window(
    messages: list[dict],
    budget_tokens: int,
) -> list[dict]:
    """
    从最新消息往前取，直到 token 预算耗尽。
    始终保留最新一条 user 消息（即使超预算）。
    返回按时间顺序排列的消息列表。
    """
    if not messages:
        return []

    budget = TokenBudget(total=budget_tokens)
    selected: list[dict] = []

    # 从尾部往前遍历
    for msg in reversed(messages):
        t = _message_tokens(msg)
        if budget.can_fit(t):
            budget.consume(t)
            selected.append(msg)
        else:
            # 超预算时，如果 selected 为空（连最新一条都没取到），强制保留
            if not selected:
                selected.append(msg)
            break

    selected.reverse()
    dropped = len(messages) - len(selected)
    if dropped > 0:
        logger.debug(f"[ContextBuilder] 滑动窗口裁剪了 {dropped} 条历史消息（token 预算 {budget_tokens}）")
    return selected


def get_context_budget(config: Any) -> dict[str, int]:
    """从 config 读取 context 预算配置，返回各槽位的 token 数。"""
    agent_cfg = getattr(config, "agent", None)
    if isinstance(agent_cfg, dict):
        ctx = agent_cfg.get("context_budget", {})
    elif agent_cfg is not None:
        ctx = getattr(agent_cfg, "context_budget", {})
    else:
        ctx = {}

    if isinstance(ctx, dict):
        total = int(ctx.get("total_tokens", 6000))
        system_reserved = int(ctx.get("system_reserved_tokens", 800))
    else:
        total = 6000
        system_reserved = 800

    history_budget = total - system_reserved
    return {
        "total": total,
        "system_reserved": system_reserved,
        "history_budget": max(500, history_budget),
    }

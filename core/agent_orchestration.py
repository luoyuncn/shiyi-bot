"""Agent orchestration domain layer: router, policy, planner, evidence."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger


class IntentType(str, Enum):
    """High-level intent classification for scheduling."""

    CHAT = "chat"
    MEMORY = "memory"
    REALTIME_INFO = "realtime_info"
    WORKSPACE_ACTION = "workspace_action"


@dataclass(slots=True)
class IntentRoute:
    """Intent routing decision."""

    intent: IntentType
    user_query: str
    reason: str
    confidence: float


@dataclass(slots=True)
class ExecutionPolicy:
    """Tool/runtime constraints selected by intent."""

    allow_tools: bool
    allowed_tools: list[str] = field(default_factory=list)
    max_iterations: int = 1
    requires_evidence: bool = False


@dataclass(slots=True)
class ExecutionPlan:
    """Minimal executable plan for the selected intent."""

    plan_id: str
    intent: IntentType
    steps: list[str]
    requires_tools: bool
    verification: str


@dataclass(slots=True)
class EvidenceItem:
    """One evidence item produced by tool execution."""

    tool_name: str
    query: str
    snippet: str
    timestamp: str


class OrchestrationRouter:
    """Intent router: structured LLM classification first, deterministic fallback second."""

    _MEMORY_PATTERNS = re.compile(
        r"(还记得|记得|之前|上次|我刚|今天中午|今天早上|我说过|回忆|想不想得起来)",
        re.IGNORECASE,
    )
    _WORKSPACE_PATTERNS = re.compile(
        r"(代码|报错|bug|修复|重构|文件|目录|脚本|命令|终端|pytest|git|class|function|api)",
        re.IGNORECASE,
    )
    _REALTIME_PATTERNS = re.compile(
        r"(查一下|搜索|最新|新闻|天气|汇率|股价|价格|成交|数据|行情|实时|房价|楼市|二手房|新房|租金|房地产)",
        re.IGNORECASE,
    )

    _INTENT_TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": "classify_intent",
            "description": "Classify user intent to one of fixed coarse-grained types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["chat", "memory", "realtime_info", "workspace_action"],
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["intent", "reason", "confidence"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(
        self,
        llm_engine: Any | None = None,
        use_llm_classifier: bool = True,
    ) -> None:
        self.llm_engine = llm_engine
        self.use_llm_classifier = use_llm_classifier

    @staticmethod
    def _extract_user_query(messages: list[dict]) -> str:
        for item in reversed(messages):
            if item.get("role") == "user":
                return str(item.get("content", "")).strip()
        return ""

    @staticmethod
    def _coerce_intent(value: Any) -> IntentType | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        aliases = {
            "realtime": "realtime_info",
            "real_time": "realtime_info",
            "实时信息": "realtime_info",
            "记忆": "memory",
            "工作区": "workspace_action",
            "workspace": "workspace_action",
            "chatting": "chat",
        }
        normalized = aliases.get(text, text)
        try:
            return IntentType(normalized)
        except ValueError:
            return None

    async def _route_by_llm(self, user_query: str) -> IntentRoute | None:
        if not user_query or self.llm_engine is None:
            return None

        classify_messages = [
            {
                "role": "system",
                "content": (
                    "You are an intent router. Always call classify_intent function. "
                    "Use exactly one of: chat, memory, realtime_info, workspace_action.\n"
                    "- memory: user asks about remembered personal/context history\n"
                    "- realtime_info: user needs fresh external facts/news/prices/market data\n"
                    "- workspace_action: user asks coding/file/terminal/tool operations\n"
                    "- chat: all other normal conversation"
                ),
            },
            {"role": "user", "content": user_query},
        ]

        result = await self.llm_engine.chat_with_tools(
            messages=classify_messages,
            tools=[self._INTENT_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "classify_intent"}},
            temperature=0.0,
            max_tokens=80,
        )

        payload: dict[str, Any] | None = None

        if result.get("type") == "tool_calls":
            calls = result.get("tool_calls") or []
            call = next((c for c in calls if c.get("name") == "classify_intent"), None)
            if call is None and calls:
                call = calls[0]

            if call:
                arguments = call.get("arguments", "{}")
                if isinstance(arguments, str):
                    payload = json.loads(arguments)
                elif isinstance(arguments, dict):
                    payload = arguments
        elif result.get("type") == "text":
            raw = str(result.get("content", "")).strip()
            if raw:
                payload = json.loads(raw)

        if not isinstance(payload, dict):
            return None

        intent = self._coerce_intent(payload.get("intent"))
        if intent is None:
            return None

        reason = str(payload.get("reason") or "llm_classifier").strip()
        confidence_raw = payload.get("confidence", 0.85)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.85
        confidence = max(0.0, min(1.0, confidence))

        return IntentRoute(
            intent=intent,
            user_query=user_query,
            reason=f"llm_structured:{reason[:80]}",
            confidence=confidence,
        )

    async def route_async(self, messages: list[dict]) -> IntentRoute:
        user_query = self._extract_user_query(messages)

        if self.use_llm_classifier and self.llm_engine is not None:
            try:
                route = await self._route_by_llm(user_query)
                if route is not None:
                    return route
                logger.warning("[Orchestrator] LLM意图分类无效，回退规则路由")
            except Exception as exc:
                logger.warning(f"[Orchestrator] LLM意图分类失败，回退规则路由: {exc}")

        return self._route_by_rules(user_query)

    def route(self, messages: list[dict]) -> IntentRoute:
        user_query = self._extract_user_query(messages)
        return self._route_by_rules(user_query)

    def _route_by_rules(self, user_query: str) -> IntentRoute:
        if self._MEMORY_PATTERNS.search(user_query):
            return IntentRoute(
                intent=IntentType.MEMORY,
                user_query=user_query,
                reason="memory_pattern",
                confidence=0.9,
            )

        if self._WORKSPACE_PATTERNS.search(user_query):
            return IntentRoute(
                intent=IntentType.WORKSPACE_ACTION,
                user_query=user_query,
                reason="workspace_pattern",
                confidence=0.86,
            )

        if self._REALTIME_PATTERNS.search(user_query):
            return IntentRoute(
                intent=IntentType.REALTIME_INFO,
                user_query=user_query,
                reason="realtime_pattern",
                confidence=0.88,
            )

        return IntentRoute(
            intent=IntentType.CHAT,
            user_query=user_query,
            reason="default_chat",
            confidence=0.72,
        )


class PolicyScheduler:
    """Map intent to execution policy and tool allowlist."""

    def build(self, intent: IntentType) -> ExecutionPolicy:
        if intent == IntentType.CHAT:
            return ExecutionPolicy(
                allow_tools=False,
                allowed_tools=[],
                max_iterations=1,
                requires_evidence=False,
            )

        if intent == IntentType.MEMORY:
            return ExecutionPolicy(
                allow_tools=True,
                allowed_tools=["query_memory"],
                max_iterations=2,
                requires_evidence=False,
            )

        if intent == IntentType.REALTIME_INFO:
            return ExecutionPolicy(
                allow_tools=True,
                allowed_tools=["search_web", "super_search"],
                max_iterations=2,
                requires_evidence=True,
            )

        return ExecutionPolicy(
            allow_tools=True,
            allowed_tools=[
                "bash",
                "read_file",
                "write_file",
                "edit_file",
                "search_web",
                "super_search",
                "query_memory",
            ],
            max_iterations=3,
            requires_evidence=True,
        )


class LightweightPlanner:
    """Generate small deterministic plan used by executor prompt context."""

    def build(
        self,
        route: IntentRoute,
        policy: ExecutionPolicy,
        messages: list[dict],
    ) -> ExecutionPlan:
        if route.intent == IntentType.REALTIME_INFO:
            steps = [
                "明确要查询的实时信息范围",
                "按策略选择检索工具并获取候选信息",
                "提取关键数字与时间并回答",
            ]
            verification = "结论需要包含可追溯依据"
        elif route.intent == IntentType.WORKSPACE_ACTION:
            steps = [
                "定位相关代码/文件",
                "执行修改并验证",
                "总结变更与风险",
            ]
            verification = "修改必须可验证且无明显回归"
        elif route.intent == IntentType.MEMORY:
            steps = [
                "优先利用会话和记忆上下文回答",
                "不满足时明确说明不确定性",
            ]
            verification = "答案应与历史上下文一致"
        else:
            steps = [
                "直接回答用户问题",
                "保持简洁清晰",
            ]
            verification = "回答准确且不过度调用工具"

        return ExecutionPlan(
            plan_id=uuid.uuid4().hex[:12],
            intent=route.intent,
            steps=steps,
            requires_tools=policy.allow_tools,
            verification=verification,
        )


class EvidenceCollector:
    """Collect and render compact evidence records."""

    def __init__(self, max_items: int = 5):
        self.max_items = max_items
        self._items: list[EvidenceItem] = []

    def reset(self) -> None:
        self._items.clear()

    def add_tool_evidence(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: str,
    ) -> None:
        query = ""
        if isinstance(tool_args, dict):
            query = str(tool_args.get("query") or tool_args.get("path") or "").strip()
            if not query:
                query = json.dumps(tool_args, ensure_ascii=False)

        snippet = str(tool_result or "").replace("\n", " ").strip()[:120]

        self._items.append(
            EvidenceItem(
                tool_name=tool_name,
                query=query,
                snippet=snippet,
                timestamp=datetime.now().isoformat(timespec="seconds"),
            )
        )

        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

    @property
    def items(self) -> list[EvidenceItem]:
        return list(self._items)

    def render_summary(self) -> str:
        if not self._items:
            return ""

        lines = ["[Evidence]"]
        for item in self._items:
            lines.append(
                f"- {item.tool_name} | query={item.query} | snippet={item.snippet}"
            )
        return "\n".join(lines)

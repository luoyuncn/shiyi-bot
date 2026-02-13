"""Agent core - LLM reasoning and tool calling."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any, AsyncIterator

from loguru import logger

from core.agent_orchestration import EvidenceCollector, OrchestrationRouter, PolicyScheduler
from core.complexity_detector import ComplexityDetector
from core.context_builder import build_history_window, get_context_budget
from engines.llm.openai_compatible_engine import OpenAICompatibleEngine
from tools.registry import ToolRegistry


class AgentCore:
    """主 Agent 核心：精简 ReAct 循环，LLM 完全自主决策工具调用。"""

    def __init__(self, config):
        self.config = config
        self.llm_config = self._normalize_llm_config(config.llm)

        self.llm_engine = OpenAICompatibleEngine(
            api_base=self.llm_config.api_base,
            api_key=self.llm_config.api_key,
            model=self.llm_config.model,
            system_prompt=self.llm_config.system_prompt,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )

        agent_cfg = getattr(config, "agent", None)
        if isinstance(agent_cfg, dict):
            self.max_tool_iterations = int(agent_cfg.get("max_tool_iterations", 5))
        elif agent_cfg is not None:
            self.max_tool_iterations = int(getattr(agent_cfg, "max_tool_iterations", 5))
        else:
            self.max_tool_iterations = 5

        self.complexity_detector = ComplexityDetector(config)
        self._context_budget = get_context_budget(config)
        self.use_llm_intent_classifier = self._read_use_llm_intent_classifier(config)
        self.router = OrchestrationRouter(
            llm_engine=self.llm_engine,
            use_llm_classifier=self.use_llm_intent_classifier,
        )
        self.policy_scheduler = PolicyScheduler()
        self.evidence_collector = EvidenceCollector(max_items=5)

    @staticmethod
    def _normalize_llm_config(llm_config: Any) -> Any:
        """兼容 dict 和对象两种格式的 llm config。"""
        if isinstance(llm_config, dict):
            defaults = {
                "temperature": 0.7,
                "max_tokens": 500,
                "system_prompt": "",
            }
            defaults.update(llm_config)
            return SimpleNamespace(**defaults)
        return llm_config

    @staticmethod
    def _read_use_llm_intent_classifier(config: Any) -> bool:
        agent_cfg = getattr(config, "agent", None)
        if isinstance(agent_cfg, dict):
            orchestration = agent_cfg.get("orchestration", {})
            if isinstance(orchestration, dict):
                return bool(orchestration.get("use_llm_intent_classifier", True))
            return True

        orchestration = getattr(agent_cfg, "orchestration", None)
        if isinstance(orchestration, dict):
            return bool(orchestration.get("use_llm_intent_classifier", True))
        if orchestration is not None:
            return bool(getattr(orchestration, "use_llm_intent_classifier", True))
        return True

    @staticmethod
    def _filter_tool_definitions(tool_defs: list[dict], allowed_tools: list[str]) -> list[dict]:
        allowset = {str(name).strip() for name in (allowed_tools or []) if str(name).strip()}
        if not allowset:
            return []
        return [
            tool
            for tool in tool_defs
            if str(tool.get("function", {}).get("name", "")).strip() in allowset
        ]

    async def initialize(self):
        """初始化 Agent Core。"""
        await self.llm_engine.initialize()
        logger.info("AgentCore 初始化完成")

    def _load_shiyi_persona(self) -> str:
        """从 ShiYi.md 动态加载人设。"""
        try:
            from pathlib import Path
            import yaml

            memory_root = getattr(self.config.memory, "memory_root", "data/memory")
            shiyi_path = Path(memory_root) / "system" / "ShiYi.md"

            if not shiyi_path.exists():
                return "你叫十一（ShiYi），一个私人智能助理。"

            text = shiyi_path.read_text(encoding="utf-8")
            if text.startswith("---"):
                try:
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        meta = yaml.safe_load(parts[1])
                        name = meta.get("name", "十一")
                        persona = meta.get("persona", "私人智能助理")
                        tone = meta.get("tone", "简洁高效")
                        return (
                            f"- 名字：{name}\n"
                            f"- 人设：{persona}\n"
                            f"- 语气：{tone}"
                        )
                except Exception as e:
                    logger.warning(f"解析 ShiYi.md 失败: {e}")

            return "你叫十一（ShiYi），一个私人智能助理。"
        except Exception as e:
            logger.warning(f"加载身份失败: {e}")
            return "你叫十一（ShiYi），一个私人智能助理。"

    def _build_system_prompt(self) -> str:
        """构建完整 system prompt（人设 + 时间 + 工具使用原则）。"""
        now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        shiyi_persona = self._load_shiyi_persona()
        prompt_template = self.llm_config.system_prompt

        format_args = {
            "model": self.llm_config.model,
            "datetime": now,
            "shiyi_persona": shiyi_persona,
        }
        try:
            return prompt_template.format(**format_args)
        except KeyError:
            try:
                return prompt_template.format(model=self.llm_config.model, datetime=now)
            except KeyError:
                return prompt_template

    async def process_message_stream(
        self,
        messages: list[dict],
        enable_tools: bool = True,
    ) -> AsyncIterator[dict]:
        """处理一轮对话，流式输出事件。LLM 自主决策何时调用工具。"""
        try:
            self.evidence_collector.reset()
            # ── 1. 意图路由 + 工具白名单 ────────────────────────────────
            route = await self.router.route_async(messages)
            policy = self.policy_scheduler.build(route.intent)

            tool_defs = ToolRegistry.get_tool_definitions() if enable_tools else []
            tools = None
            if enable_tools and tool_defs and policy.allow_tools:
                filtered = self._filter_tool_definitions(tool_defs, policy.allowed_tools)
                tools = filtered if filtered else None

            max_iterations = min(self.max_tool_iterations, max(policy.max_iterations, 1))
            active_allowset = {
                str(item.get("function", {}).get("name", "")).strip()
                for item in (tools or [])
                if str(item.get("function", {}).get("name", "")).strip()
            }

            if tools:
                logger.debug(f"已加载 {len(tools)} 个工具定义")
            logger.info(
                f"[ToolPolicy] intent={route.intent.value}, reason={route.reason}, "
                f"allow_tools={policy.allow_tools}, allowed={sorted(active_allowset)}"
            )

            # ── 2. 滑动窗口裁剪会话历史 ──────────────────────────────────
            history_budget = self._context_budget["history_budget"]
            # 过滤掉 system 消息（由 agent_core 自行组装），只做历史窗口
            history_messages = [m for m in messages if m.get("role") != "system"]
            windowed = build_history_window(history_messages, budget_tokens=history_budget)

            # ── 3. 构建 system prompt（含可选的复杂任务规划提示）──────────
            system_content = self._build_system_prompt()
            planning_hint = self.complexity_detector.get_planning_hint(windowed)
            if planning_hint:
                system_content = system_content + "\n\n" + planning_hint
                logger.debug("[AgentCore] 复杂任务模式已激活，注入规划提示")

            # 保留 session_manager 注入的 system 消息（记忆卡片、RAG 召回等）
            injected_system = [m for m in messages if m.get("role") == "system"]

            full_messages: list[dict] = [
                {"role": "system", "content": system_content},
                *injected_system,
                *windowed,
            ]

            # ── DEBUG: 每轮 token 分布 ─────────────────────────────────────
            from core.context_builder import _estimate_tokens, _message_tokens
            sys_tokens = _estimate_tokens(system_content)
            injected_tokens = sum(_message_tokens(m) for m in injected_system)
            history_tokens = sum(_message_tokens(m) for m in windowed)
            total_est = sys_tokens + injected_tokens + history_tokens
            logger.debug(
                f"[Context] sys={sys_tokens}tok injected={injected_tokens}tok"
                f"({len(injected_system)}条) history={history_tokens}tok"
                f"({len(windowed)}条) total≈{total_est}tok"
            )
            for i, m in enumerate(injected_system, 1):
                preview = str(m.get("content", ""))[:80].replace("\n", " ")
                logger.debug(f"[Context] 注入[{i}]: {preview}")

            # ── 4. ReAct 工具循环 ─────────────────────────────────────────
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            tool_call_cache: dict[str, Any] = {}
            iteration = 0
            full_text_fragments: list[str] = []

            while iteration < max_iterations:
                iteration += 1
                stream = self.llm_engine.chat_with_tools_stream(
                    full_messages,
                    tools=(tools if (enable_tools and tools) else None),
                )

                current_content = ""
                tool_calls: list[dict] = []

                async for chunk in stream:
                    if chunk["type"] == "usage":
                        for key in total_usage:
                            total_usage[key] += chunk["usage"].get(key, 0)
                        yield chunk
                    elif chunk["type"] == "text_delta":
                        delta = chunk["content"]
                        current_content += delta
                        full_text_fragments.append(delta)
                        yield {"type": "text", "content": delta}
                    elif chunk["type"] == "tool_calls":
                        tool_calls = chunk["tool_calls"]

                if tool_calls:
                    if current_content:
                        full_messages.append({"role": "assistant", "content": current_content})

                    full_messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments"],
                                    },
                                }
                                for tc in tool_calls
                            ],
                        }
                    )

                    for tc in tool_calls:
                        tool_name = tc["name"]
                        tool_args_str = tc["arguments"]
                        tool_id = tc["id"]

                        try:
                            tool_args = json.loads(tool_args_str)

                            if tool_name not in active_allowset:
                                blocked_msg = f"策略限制：当前请求不允许调用工具 `{tool_name}`。"
                                logger.warning(
                                    f"[ToolPolicy] 已拦截工具调用: tool={tool_name}, intent={route.intent.value}"
                                )
                                yield {"type": "tool_result", "tool": tool_name, "result": blocked_msg}
                                full_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_id,
                                        "name": tool_name,
                                        "content": blocked_msg,
                                    }
                                )
                                continue

                            cache_key = f"{tool_name}:{tool_args_str}"

                            if cache_key in tool_call_cache:
                                logger.debug(f"工具调用去重: {tool_name}")
                                result = tool_call_cache[cache_key]
                            else:
                                yield {"type": "tool_call", "tool": tool_name, "args": tool_args}
                                result = await self._execute_tool(tool_name, tool_args)
                                tool_call_cache[cache_key] = result

                            self.evidence_collector.add_tool_evidence(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                tool_result=str(result),
                            )
                            yield {"type": "tool_result", "tool": tool_name, "result": str(result)}
                            full_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": str(result),
                                }
                            )
                        except Exception as exc:
                            error_msg = f"工具执行失败: {exc}"
                            logger.error(f"执行工具 {tool_name} 失败: {exc}")
                            yield {"type": "tool_result", "tool": tool_name, "result": error_msg}
                            full_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": error_msg,
                                }
                            )
                    continue

                # 没有工具调用，LLM 直接回复，循环结束
                if current_content:
                    full_messages.append({"role": "assistant", "content": current_content})
                break

            else:
                logger.warning(f"工具调用达到最大次数 ({max_iterations})")
                tail = "\n\n(已达到最大工具调用次数，停止执行)"
                full_text_fragments.append(tail)
                yield {"type": "text", "content": tail}

            final_text = "".join(full_text_fragments)
            evidence_summary = self.evidence_collector.render_summary()
            if (
                policy.requires_evidence
                and evidence_summary
                and "[Evidence]" not in final_text
            ):
                yield {"type": "text", "content": f"\n\n{evidence_summary}"}

            yield {"type": "done"}

        except Exception as exc:
            logger.error(f"AgentCore 处理失败: {exc}")
            yield {"type": "error", "error": str(exc)}

    async def _execute_tool(self, tool_name: str, parameters: dict) -> Any:
        """按名称执行工具。"""
        tool = ToolRegistry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"工具不存在: {tool_name}")
        logger.debug(f"执行工具: {tool_name} with {parameters}")
        return await tool.execute(**parameters)

    async def cleanup(self):
        """释放资源。"""
        await self.llm_engine.cleanup()

    async def process_with_sub_agent(
        self,
        agent_name: str,
        task: str,
        context: dict,
    ) -> AsyncIterator[dict]:
        """委派任务给子 Agent。"""
        from agents.registry import AgentRegistry

        agent = AgentRegistry.get_agent(agent_name)
        if not agent:
            yield {"type": "error", "error": f"子Agent不存在: {agent_name}"}
            return

        yield {"type": "sub_agent_start", "agent": agent_name, "task": task}
        async for event in agent.execute(task, context):
            yield event
        yield {"type": "sub_agent_done", "agent": agent_name}

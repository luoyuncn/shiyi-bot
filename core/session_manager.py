"""Session manager - integrates storage, markdown memory and retrieval pipeline."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from memory.cache import ConversationContext, LRUCache
from memory.documents import MemoryDocumentStore
from memory.embeddings import EmbeddingConfig, LocalEmbeddingEngine
from memory.storage import MemoryStorage, SessionRecord

# ── LLM memory extraction prompt ──────────────────────────
# ── LLM 记忆提取提示词 ──────────────────────────
_MEMORY_EXTRACTION_PROMPT = """\
你是一个记忆提取器。从下面的对话内容中提取值得长期记住的用户信息。

只提取 **明确表达** 的事实，不要推测。输出 JSON 数组，每个元素格式：
{"scope":"user|project|insight", "fact_type":"identity|preference|habit|background|skill", "fact_key":"简短键名", "fact_value":"值", "confidence":0.0~1.0}

提取规则：
- identity: 姓名、称呼、昵称、年龄、职业、身份
- preference: 技术偏好、工具选择、风格喜好
- habit: 使用习惯、工作流程
- background: 教育、公司、项目经历
- skill: 擅长的技术/语言/框架
- project: 当前正在做的项目进展
- insight: 总结的经验教训

confidence 评分标准：
- 0.95: 直接自我介绍 ("我叫XX", "叫我XX")
- 0.90: 明确偏好 ("我喜欢Python", "我用VSCode")
- 0.80: 间接表达 ("最近在搞微服务")
- 0.70: 轻度提及 ("试了一下Rust")
- < 0.60: 不确定的，不要提取

如果没有值得提取的内容，返回空数组 []。
只输出 JSON，不要输出任何其他文字。"""

_ALLOWED_MEMORY_SCOPES = {"user", "project", "insight", "system"}
_ALLOWED_FACT_TYPES = {
    "identity",
    "preference",
    "habit",
    "background",
    "skill",
    "project",
    "insight",
    "persona",
    "constraint",
}
_FACT_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")
_MAX_FACT_VALUE_CHARS = 500


class SessionManager:
    """Session manager - singleton."""

    def __init__(self, memory_config, llm_config=None):
        self.config = memory_config
        self._llm_config = llm_config
        self._llm_client = None  # lazy init
        # 懒加载初始化

        self.storage = MemoryStorage(memory_config.sqlite_path)
        memory_root = getattr(memory_config, "memory_root", "data/memory")
        self.documents = MemoryDocumentStore(memory_root)

        dimension = int(getattr(memory_config, "embedding_dimension", 128))
        self.embedding_engine = LocalEmbeddingEngine(EmbeddingConfig(dimension=dimension))
        self.embedding_retry_limit = int(getattr(memory_config, "embedding_retry_limit", 3))
        self.embedding_retry_base_seconds = int(
            getattr(memory_config, "embedding_retry_base_seconds", 10)
        )
        self.embedding_poll_interval = int(getattr(memory_config, "embedding_poll_interval", 5))

        self.cache = LRUCache(max_size=memory_config.cache_size)

        self._flush_task = None
        self._embedding_task = None
        self._extraction_tasks: set[asyncio.Task] = set()
        self._running = False

    async def initialize(self):
        """Initialize."""
        await self.storage.initialize()
        await asyncio.to_thread(self.documents.ensure_initialized)

        # DB is the single source of truth for identity confirmation.
        # DB 是身份确认状态的唯一事实来源。
        # IdentityState.md is a mirror record only; it must never drive DB state.
        # IdentityState.md 仅做镜像记录，不能反向驱动 DB 状态。
        user_state = await self.storage.get_global_user_state()
        await asyncio.to_thread(
            self.documents.write_identity_state,
            user_state["identity_confirmed"],
            user_state.get("display_name"),
        )

        self._running = True
        self._flush_task = asyncio.create_task(self._auto_flush_loop())
        self._embedding_task = asyncio.create_task(self._embedding_worker_loop())

        logger.info("会话管理器初始化完成")

    async def create_session(self, metadata: dict = None) -> ConversationContext:
        """Create new session."""
        session_id = await self.storage.create_session(metadata or {})
        context = ConversationContext(session_id=session_id, metadata=metadata or {})
        self.cache.put(session_id, context)
        logger.info(f"创建会话: {session_id}")
        return context

    async def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get session (from cache first, then database)."""
        context = self.cache.get(session_id)
        if context:
            return context

        record = await self.storage.get_session(session_id)
        if not record:
            return None

        messages = await self.storage.get_messages(session_id)
        context = ConversationContext(
            session_id=session_id,
            messages=[
                {
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.message_metadata,
                }
                for msg in messages
            ],
            metadata=record.session_metadata,
            created_at=record.created_at,
            last_active=record.last_active,
        )
        self.cache.put(session_id, context)
        return context

    def _get_llm_client(self):
        """Lazy-init a lightweight OpenAI client for memory extraction."""
        if self._llm_client is not None:
            return self._llm_client
        if self._llm_config is None:
            return None
        try:
            from openai import AsyncOpenAI
            cfg = self._llm_config
            api_base = getattr(cfg, "api_base", None) or cfg.get("api_base", "")
            api_key = getattr(cfg, "api_key", None) or cfg.get("api_key", "")
            if not api_base or not api_key:
                return None
            self._llm_client = AsyncOpenAI(api_key=api_key, base_url=api_base)
            return self._llm_client
        except Exception as e:
            logger.warning(f"记忆提取 LLM 客户端初始化失败: {e}")
            return None

    def _get_llm_model(self) -> str:
        """Get model name from llm config."""
        if self._llm_config is None:
            return ""
        return getattr(self._llm_config, "model", None) or self._llm_config.get("model", "")

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict = None,
    ):
        """Save message and trigger summarize/embedding pipelines."""
        context = self.cache.get(session_id)
        if context:
            context.add_message(role, content, metadata)

        message_id = await self.storage.save_message(session_id, role, content, metadata)
        await self.enqueue_embedding_job(
            source_type="message",
            source_id=message_id,
            content=content,
        )

        if role in ("user", "assistant"):
            if role == "user":
                await self._try_complete_identity_onboarding_from_message(content)

            # Regex fast-path (sync, instant)
            # 正则快速路径（同步、即时）
            await self.summarize_and_store(content, source_message_id=message_id)
            # LLM deep extraction (async, fire-and-forget)
            # LLM 深度提取（异步、fire-and-forget）
            user_state = await self.storage.get_global_user_state()
            if self._memory_updates_enabled(user_state):
                self._fire_llm_extraction(content, source_message_id=message_id)

    async def _try_complete_identity_onboarding_from_message(self, content: str) -> bool:
        """Try to complete onboarding from one structured user confirmation message."""
        user_state = await self.storage.get_global_user_state()
        if user_state["identity_confirmed"]:
            return False

        fields: dict[str, str] = {}
        for key, value in re.findall(
            r"(十一人设|用户身份|称呼|确认)\s*[:：]\s*([^\n。；;]+)",
            content,
            flags=re.IGNORECASE,
        ):
            fields[key.strip().lower()] = value.strip()

        shiyi_identity = fields.get("十一人设")
        user_identity = fields.get("用户身份")
        confirm_value = fields.get("确认", "").lower()
        confirmed = confirm_value in {"是", "确认", "yes", "ok", "true"}

        # Natural-language fallback for onboarding confirmation.
        # onboarding 确认的自然语言兜底解析。
        if not shiyi_identity:
            shiyi_match = re.search(
                r"(?:十一|你)(?:的)?(?:人设|定位|身份)?(?:是|为)\s*([^\n。；;]{4,120})",
                content,
            )
            if shiyi_match:
                shiyi_identity = shiyi_match.group(1).strip()
        if not user_identity:
            user_match = re.search(
                r"我(?:是|叫)\s*([^\n。；;]{2,120})",
                content,
            )
            if user_match:
                user_identity = f"我是{user_match.group(1).strip()}"
        if not confirmed and re.search(r"(确认|就按这个|没问题|可以|是的|对)", content):
            confirmed = True

        if not (shiyi_identity and user_identity and confirmed):
            return False

        display_name = fields.get("称呼")
        if not display_name:
            name_match = re.search(r"我(?:叫|是)\s*([^\s，。！？,!.?]{1,20})", user_identity)
            if name_match:
                display_name = name_match.group(1)

        await self.complete_identity_onboarding(
            shiyi_identity=shiyi_identity,
            user_identity=user_identity,
            display_name=display_name,
        )
        await self.storage.save_memory_event(
            event_type="identity_onboarding_completed_inline",
            payload={
                "display_name": display_name,
                "source": "user_message",
            },
        )
        return True

    # ── LLM-based memory extraction (async, fire-and-forget) ──────
    # ── 基于 LLM 的记忆提取（异步 fire-and-forget） ──────

    def _fire_llm_extraction(self, content: str, source_message_id: str | None = None):
        """Launch LLM memory extraction as a background task."""
        client = self._get_llm_client()
        if client is None:
            return
        task = asyncio.create_task(
            self._extract_memory_via_llm(content, source_message_id)
        )
        self._extraction_tasks.add(task)
        task.add_done_callback(self._extraction_tasks.discard)

    async def _extract_memory_via_llm(
        self, content: str, source_message_id: str | None = None
    ):
        """Call LLM to extract structured memory facts from content."""
        client = self._get_llm_client()
        if client is None:
            return
        user_state = await self.storage.get_global_user_state()
        if not self._memory_updates_enabled(user_state):
            return

        # Skip very short content
        # 跳过过短文本
        if len(content.strip()) < 6:
            return

        try:
            response = await client.chat.completions.create(
                model=self._get_llm_model(),
                messages=[
                    {"role": "system", "content": _MEMORY_EXTRACTION_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            raw = (response.choices[0].message.content or "").strip()
            # Strip markdown code fences if present
            # 如果有 Markdown 代码围栏则去掉
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            candidates = json.loads(raw)
            if not isinstance(candidates, list):
                return

            applied = 0
            pending = 0
            for item in candidates:
                try:
                    confidence = float(item.get("confidence", 0))
                except (TypeError, ValueError):
                    continue

                fact, reason = self._normalize_memory_patch(item)
                if not fact:
                    await self._record_rejected_patch(
                        fact=item,
                        reason=reason or "normalize_failed",
                        source_message_id=source_message_id,
                    )
                    continue

                if confidence >= 0.85:
                    if await self._apply_memory_fact(
                        fact=fact,
                        confidence=confidence,
                        source_message_id=source_message_id,
                    ):
                        applied += 1
                elif confidence >= 0.60:
                    await self.save_memory_pending(
                        candidate_fact=fact,
                        confidence=confidence,
                        source_message_id=source_message_id,
                    )
                    pending += 1

            if applied or pending:
                logger.info(f"LLM 记忆提取: {applied} 条直接写入, {pending} 条待确认")
                await self.storage.save_memory_event(
                    event_type="llm_extraction_completed",
                    payload={
                        "source_message_id": source_message_id,
                        "applied": applied,
                        "pending": pending,
                    },
                )

        except json.JSONDecodeError:
            logger.debug(f"LLM 记忆提取返回非 JSON: {raw[:100]}")
        except Exception as e:
            logger.warning(f"LLM 记忆提取失败: {e}")

    async def list_sessions(self, limit: int = 50) -> list[SessionRecord]:
        """List all sessions."""
        return await self.storage.list_sessions(limit=limit)

    async def delete_session(self, session_id: str):
        """Delete session."""
        self.cache.remove(session_id)
        await self.storage.delete_session(session_id)

    async def get_global_user_state(self) -> dict:
        """Get singleton global user state."""
        return await self.storage.get_global_user_state()

    async def complete_identity_onboarding(
        self,
        shiyi_identity: str,
        user_identity: str,
        display_name: str | None = None,
    ):
        """Finalize first-run onboarding and persist markdown identity docs."""
        await asyncio.to_thread(self.documents.write_initial_identity, shiyi_identity, user_identity)
        await self.storage.set_global_user_identity_state(
            identity_confirmed=True,
            display_name=display_name,
            onboarding_prompted=True,
        )
        await asyncio.to_thread(
            self.documents.write_identity_state,
            True,
            display_name,
        )
        await self.storage.save_memory_event(
            event_type="identity_onboarding_completed",
            payload={
                "shiyi_identity_length": len(shiyi_identity),
                "user_identity_length": len(user_identity),
                "display_name": display_name,
            },
        )

    async def summarize_and_store(self, content: str, source_message_id: str | None = None):
        """Incremental summarize pipeline with confidence-based routing."""
        user_state = await self.storage.get_global_user_state()
        if not self._memory_updates_enabled(user_state):
            return

        high_count = 0
        pending_count = 0
        for candidate in self._extract_memory_candidates(content):
            confidence = candidate["confidence"]
            fact, reason = self._normalize_memory_patch(candidate["fact"])
            if not fact:
                await self._record_rejected_patch(
                    fact=candidate["fact"],
                    reason=reason or "normalize_failed",
                    source_message_id=source_message_id,
                )
                continue
            if confidence >= 0.85:
                if await self._apply_memory_fact(
                    fact=fact,
                    confidence=confidence,
                    source_message_id=source_message_id,
                ):
                    high_count += 1
            elif confidence >= 0.60:
                await self.save_memory_pending(
                    candidate_fact=fact,
                    confidence=confidence,
                    source_message_id=source_message_id,
                )
                pending_count += 1
                await self.storage.save_memory_event(
                    event_type="memory_pending_created",
                    payload={
                        "fact": fact,
                        "confidence": confidence,
                        "source_message_id": source_message_id,
                    },
                )

        project_update = self._extract_project_update(content)
        if project_update:
            await asyncio.to_thread(self.documents.append_project_update, project_update)

        insight = self._extract_insight(content)
        if insight:
            await asyncio.to_thread(self.documents.add_insight, insight)

        await self.storage.save_memory_event(
            event_type="summarize_and_store_completed",
            payload={
                "source_message_id": source_message_id,
                "high_confidence_count": high_count,
                "pending_count": pending_count,
            },
            operation_id=f"summarize:{source_message_id}" if source_message_id else None,
        )

    def _extract_memory_candidates(self, content: str) -> list[dict]:
        """Extract candidate facts with confidence scores.

        Handles both user messages ("我叫XX") and assistant confirmations
        ("好的，已记住你叫XX", "记住了，你是XX").
        """
        candidates: list[dict] = []

        # ── User self-introduction ──
        # ── 用户自我介绍 ──
        name_match = re.search(r"我叫\s*([^\s，。！？,!.?]{1,20})", content)
        if name_match:
            candidates.append(
                {
                    "confidence": 0.95,
                    "fact": {
                        "scope": "user",
                        "fact_type": "identity",
                        "fact_key": "display_name",
                        "fact_value": name_match.group(1),
                    },
                }
            )

        # ── User identity variants ("你可以叫我X", "叫我X就行") ──
        # ── 用户身份表达变体（“你可以叫我X”“叫我X就行”） ──
        alias_match = re.search(
            r"(?:你可以)?叫我\s*([^\s，。！？,!.?]{1,20})",
            content,
        )
        if alias_match and not name_match:
            candidates.append(
                {
                    "confidence": 0.90,
                    "fact": {
                        "scope": "user",
                        "fact_type": "identity",
                        "fact_key": "display_name",
                        "fact_value": alias_match.group(1),
                    },
                }
            )

        # ── Assistant confirmation of user name ──
        # ── 助手确认用户称呼 ──
        confirm_name = re.search(
            r"(?:记住了|好的|了解|收到)[，,]?\s*(?:你(?:叫|是)|称呼.*?为?)\s*([^\s，。！？,!.?]{1,20})",
            content,
        )
        if confirm_name:
            candidates.append(
                {
                    "confidence": 0.88,
                    "fact": {
                        "scope": "user",
                        "fact_type": "identity",
                        "fact_key": "display_name",
                        "fact_value": confirm_name.group(1),
                    },
                }
            )

        # ── User profession/role ("我是程序员", "我做后端的") ──
        # ── 用户职业/角色（“我是程序员”“我做后端的”） ──
        role_match = re.search(
            r"我(?:是|做)\s*([^\s，。！？,!.?]{2,20}?)(?:的|$)",
            content,
        )
        if role_match:
            val = role_match.group(1).strip()
            if len(val) >= 2 and val not in ("什么", "不是", "这个", "那个"):
                candidates.append(
                    {
                        "confidence": 0.75,
                        "fact": {
                            "scope": "user",
                            "fact_type": "identity",
                            "fact_key": "profession",
                            "fact_value": val,
                        },
                    }
                )

        # ── Tech preference (explicit) ──
        # ── 技术偏好（显式表达） ──
        pref_match = re.search(
            r"我(?:更)?(?:喜欢|偏好|偏爱)\s*([A-Za-z][A-Za-z0-9+._-]{1,30})",
            content,
        )
        if pref_match:
            candidates.append(
                {
                    "confidence": 0.9,
                    "fact": {
                        "scope": "user",
                        "fact_type": "preference",
                        "fact_key": "preferred_tech",
                        "fact_value": pref_match.group(1),
                    },
                }
            )

        # ── Tech preference (medium confidence) ──
        # ── 技术偏好（中等置信度） ──
        medium_pref_match = re.search(
            r"我最近在用\s*([A-Za-z][A-Za-z0-9+._-]{1,30})",
            content,
        )
        if medium_pref_match:
            candidates.append(
                {
                    "confidence": 0.72,
                    "fact": {
                        "scope": "user",
                        "fact_type": "preference",
                        "fact_key": "preferred_tech",
                        "fact_value": medium_pref_match.group(1),
                    },
                }
            )

        # ── General preference ("我习惯X", "我常用X") ──
        # ── 通用偏好（“我习惯X”“我常用X”） ──
        habit_match = re.search(
            r"我(?:习惯|常用|一直用)\s*([^\s，。！？,!.?]{2,30})",
            content,
        )
        if habit_match:
            candidates.append(
                {
                    "confidence": 0.70,
                    "fact": {
                        "scope": "user",
                        "fact_type": "preference",
                        "fact_key": "habit",
                        "fact_value": habit_match.group(1),
                    },
                }
            )

        return candidates

    @staticmethod
    def _extract_project_update(content: str) -> str | None:
        match = re.search(r"项目进展[:：]\s*(.+)", content)
        if not match:
            return None
        return match.group(1).strip()

    @staticmethod
    def _extract_insight(content: str) -> str | None:
        match = re.search(r"(?:经验|复盘)[:：]\s*(.+)", content)
        if not match:
            return None
        return match.group(1).strip()

    async def save_memory_pending(
        self,
        candidate_fact: dict,
        confidence: float,
        source_message_id: str | None = None,
    ) -> str:
        """Create a pending memory candidate."""
        return await self.storage.save_memory_pending(
            candidate_fact=candidate_fact,
            confidence=confidence,
            source_message_id=source_message_id,
        )

    async def list_memory_pending(
        self,
        status: str | None = "pending",
        limit: int = 20,
    ):
        """List pending memory candidates."""
        return await self.storage.list_memory_pending(status=status, limit=limit)

    async def update_memory_pending_status(
        self,
        pending_id: str,
        status: str,
        cooldown_until: datetime | None = None,
    ):
        """Update status for a pending memory candidate."""
        if status == "confirmed":
            pending = await self.storage.get_memory_pending(pending_id)
            if pending and pending.candidate_fact:
                await self._apply_memory_fact(
                    pending.candidate_fact,
                    confidence=pending.confidence,
                    source_message_id=pending.source_message_id,
                )

        if status == "snoozed" and cooldown_until is None:
            cooldown_until = datetime.now() + timedelta(hours=24)

        await self.storage.update_memory_pending_status(
            pending_id=pending_id,
            status=status,
            cooldown_until=cooldown_until,
        )
        await self.storage.save_memory_event(
            event_type="memory_pending_status_updated",
            payload={
                "pending_id": pending_id,
                "status": status,
                "cooldown_until": cooldown_until.isoformat() if cooldown_until else None,
            },
        )

    async def list_memory_facts(
        self,
        scope: str | None = None,
        fact_type: str | None = None,
        status: str = "active",
        limit: int = 100,
    ):
        """List stored memory facts."""
        return await self.storage.list_memory_facts(
            scope=scope,
            fact_type=fact_type,
            status=status,
            limit=limit,
        )

    async def list_memory_events(
        self,
        event_type: str | None = None,
        limit: int = 100,
    ):
        """List memory pipeline events."""
        return await self.storage.list_memory_events(event_type=event_type, limit=limit)

    async def get_memory_metrics(self) -> dict:
        """Expose memory observability counters."""
        return await self.storage.get_memory_metrics()

    async def enqueue_embedding_job(
        self,
        source_type: str,
        source_id: str,
        content: str,
    ) -> str:
        """Enqueue one embedding task."""
        return await self.storage.enqueue_embedding_job(
            source_type=source_type,
            source_id=source_id,
            content=content,
        )

    async def list_embedding_jobs(self, status: str | None = None, limit: int = 100):
        """List embedding jobs."""
        return await self.storage.list_embedding_jobs(status=status, limit=limit)

    async def run_embedding_pipeline(
        self,
        max_jobs: int = 20,
        ignore_schedule: bool = False,
    ) -> dict:
        """Process due embedding jobs with retry/backoff/dead-letter handling."""
        jobs = await self.storage.list_due_embedding_jobs(
            limit=max_jobs,
            now=datetime.now(),
            ignore_schedule=ignore_schedule,
        )
        if not jobs:
            return {"processed": 0, "failed": 0}

        processed = 0
        failed = 0
        for job in jobs:
            try:
                vector = self.embedding_engine.embed(job.content)
                await self.storage.save_memory_embedding(
                    source_type=job.source_type,
                    source_id=job.source_id,
                    content=job.content,
                    embedding=vector,
                )
                await self.storage.update_embedding_job(
                    job.id,
                    status="completed",
                    last_error="",
                    next_retry_at=None,
                )
                await self.storage.save_memory_event(
                    event_type="embedding_job_completed",
                    payload={"job_id": job.id, "source_type": job.source_type, "source_id": job.source_id},
                )
                processed += 1
            except Exception as exc:
                failed += 1
                retry_count = (job.retry_count or 0) + 1
                if retry_count >= self.embedding_retry_limit:
                    await self.storage.update_embedding_job(
                        job.id,
                        status="dead_letter",
                        retry_count=retry_count,
                        last_error=str(exc),
                        next_retry_at=None,
                    )
                    await self.storage.save_memory_event(
                        event_type="embedding_dead_letter",
                        payload={"job_id": job.id, "error": str(exc), "retry_count": retry_count},
                    )
                    continue

                next_retry_at = datetime.now() + timedelta(
                    seconds=self.embedding_retry_base_seconds * (2 ** (retry_count - 1))
                )
                await self.storage.update_embedding_job(
                    job.id,
                    status="pending",
                    retry_count=retry_count,
                    next_retry_at=next_retry_at,
                    last_error=str(exc),
                )
                await self.storage.save_memory_event(
                    event_type="embedding_retry_scheduled",
                    payload={
                        "job_id": job.id,
                        "error": str(exc),
                        "retry_count": retry_count,
                        "next_retry_at": next_retry_at.isoformat(),
                    },
                )

        return {"processed": processed, "failed": failed}

    async def search_memory_by_keyword(self, query: str, limit: int = 5) -> list[dict]:
        """Keyword-only memory search."""
        return await self.storage.search_messages_by_keyword(query=query, limit=limit)

    async def search_memory_hybrid(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid search: semantic + keyword + freshness."""
        semantic_hits: list[dict] = []
        keyword_hits: list[dict] = []

        try:
            query_embedding = self.embedding_engine.embed(query)
            semantic_hits = await self.storage.search_memory_embeddings(query_embedding, limit=20)
        except Exception as exc:
            await self.storage.save_memory_event(
                event_type="retrieval_fail",
                payload={"query": query, "reason": "semantic_error", "error": str(exc)},
            )

        try:
            keyword_hits = await self.storage.search_messages_by_keyword(query=query, limit=20)
        except Exception as exc:
            await self.storage.save_memory_event(
                event_type="retrieval_fail",
                payload={"query": query, "reason": "keyword_error", "error": str(exc)},
            )

        merged: dict[tuple[str, str], dict] = {}
        for hit in semantic_hits:
            key = (hit["source_type"], hit["source_id"])
            merged[key] = {
                **hit,
                "semantic_score": hit.get("semantic_score", 0.0),
                "keyword_score": 0.0,
            }
        for hit in keyword_hits:
            key = (hit.get("source_type", "message"), hit.get("source_id", hit.get("message_id")))
            if key not in merged:
                merged[key] = {
                    **hit,
                    "semantic_score": 0.0,
                    "keyword_score": hit.get("keyword_score", 0.0),
                }
            else:
                merged[key]["keyword_score"] = max(
                    merged[key].get("keyword_score", 0.0),
                    hit.get("keyword_score", 0.0),
                )
                if not merged[key].get("content"):
                    merged[key]["content"] = hit.get("content")
                if not merged[key].get("timestamp"):
                    merged[key]["timestamp"] = hit.get("timestamp")

        scored = []
        for item in merged.values():
            semantic_score = float(item.get("semantic_score", 0.0))
            keyword_score = float(item.get("keyword_score", 0.0))
            freshness = self._freshness_score(item.get("timestamp"))
            final_score = 0.55 * semantic_score + 0.30 * keyword_score + 0.15 * freshness
            retrieval_type = "semantic" if semantic_score >= keyword_score else "keyword"
            scored.append(
                {
                    **item,
                    "freshness_score": freshness,
                    "final_score": final_score,
                    "retrieval_type": retrieval_type,
                }
            )

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top_hits = scored[:limit]
        await self.storage.save_memory_event(
            event_type="retrieval_success" if top_hits else "retrieval_fail",
            payload={
                "query": query,
                "result_count": len(top_hits),
                "reason": None if top_hits else "no_hits",
            },
        )
        return top_hits

    @staticmethod
    def _freshness_score(timestamp: str | None) -> float:
        if not timestamp:
            return 0.5
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            return 0.5
        delta_days = max((datetime.now() - dt).total_seconds() / 86400.0, 0.0)
        return max(0.0, 1.0 / (1.0 + delta_days / 14.0))

    @staticmethod
    def _memory_updates_enabled(user_state: dict) -> bool:
        """Enable memory write/extraction after first onboarding touchpoint."""
        return bool(
            user_state.get("identity_confirmed")
            or user_state.get("onboarding_prompted")
        )

    async def prepare_messages_for_agent(self, messages: list[dict]) -> list[dict]:
        """Prepend onboarding guidance or memory card before model inference."""
        user_state = await self.storage.get_global_user_state()
        memory_card = await asyncio.to_thread(self.documents.build_system_memory_card)
        system_messages = [{"role": "system", "content": memory_card}]

        if not user_state["identity_confirmed"]:
            if not user_state.get("onboarding_prompted"):
                await self.storage.mark_onboarding_prompted()
                onboarding_prompt = (
                    "系统状态：用户尚未完成身份初始化。\n"
                    "请先引导用户确认两件事：\n"
                    "1) 十一的人设定位与行为边界；\n"
                    "2) 用户身份（称呼、背景、偏好）。\n"
                    "请用户尽量按以下格式一次回复：\n"
                    "十一人设：...\n"
                    "用户身份：...\n"
                    "称呼：...\n"
                    "确认：是\n"
                    "在用户明确确认前，不要假设长期身份信息。"
                )
                return [{"role": "system", "content": onboarding_prompt}, *system_messages, *messages]
            return [*system_messages, *messages]

        recall_prompt = await self._build_recall_prompt(messages)
        if recall_prompt:
            system_messages.append({"role": "system", "content": recall_prompt})

        pending_prompt = await self._build_pending_confirmation_prompt()
        if pending_prompt:
            system_messages.append({"role": "system", "content": pending_prompt})

        return [*system_messages, *messages]

    async def _build_recall_prompt(self, messages: list[dict]) -> str | None:
        """Build a recall snippet when user asks historical questions."""
        last_user_message = ""
        for item in reversed(messages):
            if item.get("role") == "user":
                last_user_message = item.get("content", "")
                break
        if not last_user_message:
            return None

        if not re.search(r"(还记得|之前|上次|历史|以前|曾经)", last_user_message):
            return None

        tokens = re.findall(r"[A-Za-z][A-Za-z0-9+._-]{1,30}", last_user_message)
        if tokens:
            query = " ".join(tokens[:4])
        else:
            query = last_user_message

        hits = await self.search_memory_hybrid(query, limit=5)
        if not hits:
            await self.storage.save_memory_event(
                event_type="retrieval_fail",
                payload={"query": query, "reason": "no_hits"},
            )
            return None

        lines = ["历史记忆检索（混合）:"]
        for hit in hits[:3]:
            lines.append(f"- [{hit['retrieval_type']}] {hit['content'][:120]}")
        return "\n".join(lines)

    async def _build_pending_confirmation_prompt(self) -> str | None:
        pending_items = await self.list_memory_pending(status="pending", limit=3)
        if not pending_items:
            return None

        lines = [
            "你有待确认记忆，请在回复末尾用简洁格式提示用户可执行动作：确认/忽略/稍后。",
            "候选记忆：",
        ]
        for item in pending_items:
            fact = item.candidate_fact or {}
            lines.append(
                f"- [{item.id[:8]}] {fact.get('fact_key', 'unknown')} = {fact.get('fact_value', '')}"
            )
        return "\n".join(lines)

    def _normalize_memory_patch(self, fact: dict) -> tuple[dict | None, str | None]:
        """Normalize and validate one structured memory patch."""
        if not isinstance(fact, dict):
            return None, "fact_not_object"

        scope = str(fact.get("scope", "user")).strip().lower()
        if scope not in _ALLOWED_MEMORY_SCOPES:
            return None, "invalid_scope"

        fact_type = str(fact.get("fact_type", "preference")).strip().lower()
        if fact_type not in _ALLOWED_FACT_TYPES:
            return None, "invalid_fact_type"

        raw_key = fact.get("fact_key")
        fact_key = str(raw_key).strip().lower() if raw_key is not None else ""
        if not fact_key:
            return None, "empty_fact_key"
        if not _FACT_KEY_PATTERN.match(fact_key):
            return None, "invalid_fact_key"

        raw_value = fact.get("fact_value")
        if raw_value is None:
            return None, "empty_fact_value"
        if isinstance(raw_value, list):
            fact_value = ", ".join(
                [str(item).strip() for item in raw_value if str(item).strip()]
            )
        elif isinstance(raw_value, dict):
            return None, "invalid_fact_value_type"
        else:
            fact_value = str(raw_value).strip()

        if not fact_value:
            return None, "empty_fact_value"
        if len(fact_value) > _MAX_FACT_VALUE_CHARS:
            return None, "fact_value_too_long"

        return {
            "scope": scope,
            "fact_type": fact_type,
            "fact_key": fact_key,
            "fact_value": fact_value,
        }, None

    async def _record_rejected_patch(
        self,
        fact: dict,
        reason: str,
        source_message_id: str | None = None,
    ):
        await self.storage.save_memory_event(
            event_type="memory_patch_rejected",
            payload={
                "reason": reason,
                "fact": fact,
                "fact_key": fact.get("fact_key") if isinstance(fact, dict) else None,
                "source_message_id": source_message_id,
            },
        )

    async def _apply_memory_fact(
        self,
        fact: dict,
        confidence: float,
        source_message_id: str | None = None,
    ) -> bool:
        """Persist memory fact to structured DB and Markdown docs."""
        normalized_fact, reason = self._normalize_memory_patch(fact)
        if not normalized_fact:
            await self._record_rejected_patch(
                fact=fact,
                reason=reason or "normalize_failed",
                source_message_id=source_message_id,
            )
            return False

        scope = normalized_fact["scope"]
        fact_type = normalized_fact["fact_type"]
        fact_key = normalized_fact["fact_key"]
        fact_value = normalized_fact["fact_value"]

        fact_id = await self.storage.upsert_memory_fact(
            scope=scope,
            fact_type=fact_type,
            fact_key=fact_key,
            fact_value=fact_value,
            confidence=confidence,
            source_message_id=source_message_id,
        )

        await self.storage.save_memory_event(
            event_type="memory_fact_applied",
            payload={
                "fact_id": fact_id,
                "scope": scope,
                "fact_type": fact_type,
                "fact_key": fact_key,
                "fact_value": fact_value,
                "confidence": confidence,
                "source_message_id": source_message_id,
            },
        )

        await self.enqueue_embedding_job(
            source_type="fact",
            source_id=fact_id,
            content=f"{scope}:{fact_type}:{fact_key}:{fact_value}",
        )

        if scope == "user":
            if fact_key == "display_name":
                user_state = await self.storage.get_global_user_state()
                await self.storage.set_global_user_identity_state(
                    identity_confirmed=user_state["identity_confirmed"],
                    display_name=fact_value,
                )
                if user_state["identity_confirmed"]:
                    await asyncio.to_thread(
                        self.documents.write_identity_state,
                        True,
                        fact_value,
                    )
            await asyncio.to_thread(self.documents.upsert_user_fact, fact_key, fact_value)
        elif scope == "system":
            await asyncio.to_thread(self.documents.upsert_shiyi_fact, fact_key, fact_value)
        elif scope == "project":
            await asyncio.to_thread(self.documents.append_project_update, fact_value)
        elif scope == "insight":
            await asyncio.to_thread(self.documents.add_insight, fact_value)

        return True

    async def _auto_flush_loop(self):
        """Auto flush loop."""
        interval = self.config.auto_flush_interval
        while self._running:
            await asyncio.sleep(interval)
            logger.debug(f"缓存状态: {self.cache.size()} 个活跃会话")

    async def _embedding_worker_loop(self):
        """Background worker for embedding jobs."""
        while self._running:
            try:
                await self.run_embedding_pipeline(max_jobs=20, ignore_schedule=False)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Embedding worker loop failed: {exc}")
            await asyncio.sleep(self.embedding_poll_interval)

    async def cleanup(self):
        """Cleanup resources."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
        if self._embedding_task:
            self._embedding_task.cancel()
        # Wait for background extraction tasks to finish
        # 等待后台提取任务结束
        if self._extraction_tasks:
            await asyncio.gather(*self._extraction_tasks, return_exceptions=True)
        await asyncio.gather(
            *(task for task in [self._flush_task, self._embedding_task] if task),
            return_exceptions=True,
        )
        if self._llm_client:
            await self._llm_client.close()
        await self.storage.cleanup()
        self.cache.clear()

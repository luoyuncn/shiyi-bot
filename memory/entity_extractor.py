"""异步 LLM 实体抽取 - 从对话中提取实体和关系，写入 Kuzu 图谱。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

_EXTRACTION_PROMPT = """\
你是一个实体关系抽取器。从以下对话片段中抽取值得记入知识图谱的实体和关系。

## 输出格式
JSON 对象，包含：
{
  "entities": [{"name": "实体名", "type": "person|technology|project|concept|organization"}],
  "relations": [{"from": "实体A", "to": "实体B", "type": "uses|knows|works_on|belongs_to|related_to"}],
  "facts": [{"scope": "user|project|insight", "key": "简短英文键", "value": "值", "confidence": 0.0~1.0, "entity": "关联实体名（可选）"}]
}

## 提取标准
- 只提取明确提到的实体，不要推断
- confidence < 0.6 的事实不提取
- 实体名要精简（2-10字），不要提取泛化词（"问题"、"代码"、"信息"）
- 无内容可提取时返回 {"entities":[],"relations":[],"facts":[]}
- 只输出 JSON，不要解释

## 对话片段
{conversation}
"""


class EntityExtractor:
    """
    异步 LLM 实体抽取器。
    在每次助手回复后 fire-and-forget，不阻塞主流程。
    """

    def __init__(self, llm_engine: Any):
        self._llm = llm_engine
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """启动后台 worker。"""
        self._worker_task = asyncio.create_task(self._worker(), name="entity_extractor")
        logger.debug("EntityExtractor worker 已启动")

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def enqueue(self, messages: list[dict]) -> None:
        """将最近几条对话入队（非阻塞，满则丢弃）。"""
        if self._queue.full():
            logger.debug("EntityExtractor 队列已满，跳过本次入队")
            return
        try:
            self._queue.put_nowait(messages)
        except asyncio.QueueFull:
            pass

    async def _worker(self) -> None:
        """后台持续消费队列，逐条抽取实体写 Kuzu。"""
        while True:
            try:
                messages = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                await self._extract_and_write(messages)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"EntityExtractor worker 出错: {e}")

    async def _extract_and_write(self, messages: list[dict]) -> None:
        """调用 LLM 抽取实体，写入 Kuzu。"""
        from memory.kuzu_manager import get_writer

        writer = get_writer()
        if writer is None:
            return

        # 只取最近 3 条（user + assistant）
        recent = [m for m in messages if m.get("role") in ("user", "assistant")][-3:]
        if not recent:
            return

        conversation_text = "\n".join(
            f"[{m['role']}]: {str(m.get('content', ''))[:300]}"
            for m in recent
        )

        try:
            result = await self._llm.chat(
                messages=[
                    {"role": "user", "content": _EXTRACTION_PROMPT.format(conversation=conversation_text)}
                ],
                temperature=0.0,
                max_tokens=400,
            )
        except Exception as e:
            logger.debug(f"实体抽取 LLM 调用失败: {e}")
            return

        raw = (result or "").strip()
        if not raw:
            return

        # 解析 JSON
        try:
            # 提取 JSON 块（兼容 LLM 可能包裹在 ```json``` 里的情况）
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
        except (json.JSONDecodeError, IndexError) as e:
            logger.debug(f"实体抽取 JSON 解析失败: {e} | raw={raw[:100]}")
            return

        if not isinstance(data, dict):
            return

        entities = data.get("entities", [])
        relations = data.get("relations", [])
        facts = data.get("facts", [])

        # 写实体
        for ent in entities:
            name = str(ent.get("name", "")).strip()
            etype = str(ent.get("type", "concept")).strip()
            if name and len(name) <= 30:
                try:
                    await writer.upsert_entity(name, etype)
                except Exception as e:
                    logger.debug(f"写实体失败 {name}: {e}")

        # 写关系
        for rel in relations:
            frm = str(rel.get("from", "")).strip()
            to = str(rel.get("to", "")).strip()
            rtype = str(rel.get("type", "related_to")).strip()
            if frm and to and frm != to:
                try:
                    await writer.upsert_entity_relation(frm, to, rtype)
                except Exception as e:
                    logger.debug(f"写关系失败 {frm}->{to}: {e}")

        # 写事实
        for fact in facts:
            scope = str(fact.get("scope", "user"))
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            confidence = float(fact.get("confidence", 0.7))
            entity = str(fact.get("entity", "")).strip() or None
            if key and value and confidence >= 0.6:
                try:
                    await writer.upsert_fact(scope, key, value, confidence, entity)
                except Exception as e:
                    logger.debug(f"写事实失败 {key}: {e}")

        entity_count = len(entities)
        fact_count = len([f for f in facts if float(f.get("confidence", 0)) >= 0.6])
        if entity_count or fact_count:
            logger.debug(f"实体抽取完成: {entity_count} 个实体, {fact_count} 条事实")

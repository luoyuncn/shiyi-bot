"""Super Search — 5-phase intelligent search tool using Tavily + LLM orchestration.

Pipeline:
  Phase 1: Query expansion   → 3-5 parallel search variants
  Phase 2: Broad search      → Tavily basic, fast coverage
  Phase 3: Gap analysis      → LLM judges if results suffice
  Phase 4: Deep search       → Tavily advanced, targeted deep-dive (conditional)
  Phase 5: Synthesis         → LLM produces structured final answer
"""
import asyncio
import json
import os
import time
from datetime import datetime

from loguru import logger
from openai import AsyncOpenAI

from tools.base import BaseTool, ToolDefinition, ToolParameter


class Tool(BaseTool):
    """高质量多阶段智能搜索工具 (Tavily + LLM 编排)"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="super_search",
            description=(
                "高质量互联网深度搜索工具。采用先广后深的智能搜索策略："
                "①自动将问题扩展为多角度并行搜索；"
                "②智能判断是否需要深度补充搜索；"
                "③综合所有信息生成结构化权威答案。"
                "适用于实时资讯、技术研究、事实核查、深度分析等复杂问题。"
                "注意：此工具耗时较长，简单搜索请用 search_web。"
            ),
            parameters={
                "query": ToolParameter(
                    type="string",
                    description="自然语言问题或搜索主题（中英文均可）",
                    required=True,
                ),
            },
        )

    async def validate_params(self, params: dict):
        query = params.get("query", "")
        if not query or not str(query).strip():
            raise ValueError("搜索问题不能为空")
        if len(str(query)) > 1000:
            raise ValueError("搜索问题不能超过1000个字符")

    async def execute(self, query: str) -> str:
        """Run the 5-phase super search pipeline."""
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return "Tavily API Key 未配置，请在 .env 中设置 TAVILY_API_KEY"

        llm_key = os.getenv("LLM_API_KEY")
        llm_base = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        if not llm_key:
            return "LLM API Key 未配置，请在 .env 中设置 LLM_API_KEY"

        try:
            from tavily import TavilyClient
        except ImportError:
            return "Tavily 未安装，请运行: uv add tavily-python"

        tavily_client = TavilyClient(api_key=tavily_key)
        llm_client = AsyncOpenAI(api_key=llm_key, base_url=llm_base)

        total_start = time.time()

        try:
            # ── Phase 1: Query expansion ─────────────────────────
            logger.debug(f"SuperSearch Phase 1: 查询扩展 — {query}")
            queries = await self._expand_queries(llm_client, llm_model, query)
            logger.debug(f"SuperSearch: 扩展为 {len(queries)} 个查询")

            # ── Phase 2: Broad search (parallel) ─────────────────
            logger.debug(f"SuperSearch Phase 2: 广度搜索 ({len(queries)} 个)")
            broad_results = await self._broad_search(tavily_client, queries)
            logger.debug(f"SuperSearch: 广搜完成，{len(broad_results)} 条结果")

            # ── Phase 3: Gap analysis ────────────────────────────
            logger.debug("SuperSearch Phase 3: 缺口分析")
            gap_analysis = await self._analyze_gaps(
                llm_client, llm_model, query, broad_results
            )
            need_deep = gap_analysis.get("need_deep_search", False)
            gaps = gap_analysis.get("gaps", [])
            logger.debug(
                f"SuperSearch: {'需要深搜' if need_deep else '广搜已充分'}"
            )

            all_results = list(broad_results)

            # ── Phase 4: Deep search (conditional) ───────────────
            if need_deep and gaps:
                logger.debug(
                    f"SuperSearch Phase 4: 深度搜索 ({len(gaps)} 个缺口)"
                )
                deep_results = await self._deep_search(tavily_client, gaps)
                all_results.extend(deep_results)
                logger.debug(f"SuperSearch: 深搜补充 {len(deep_results)} 条")
            else:
                logger.debug("SuperSearch Phase 4: 跳过深搜")

            # ── Phase 5: Synthesis ───────────────────────────────
            logger.debug("SuperSearch Phase 5: 综合生成答案")
            final_answer = await self._synthesize(
                llm_client, llm_model, query, all_results
            )

            total_duration = time.time() - total_start
            broad_count = len(
                [r for r in all_results if r["phase"] == "broad"]
            )
            deep_count = len(
                [r for r in all_results if r["phase"] == "deep"]
            )
            logger.info(
                f"SuperSearch 完成: 广搜{broad_count} 深搜{deep_count} "
                f"耗时{total_duration:.1f}s"
            )

            return final_answer

        except Exception as e:
            logger.error(f"SuperSearch 失败: {e}")
            return f"超级搜索失败: {e}"

    # ── Phase 1: Query Expansion ─────────────────────────────────

    async def _expand_queries(
        self, llm: AsyncOpenAI, model: str, question: str
    ) -> list[str]:
        current_date = datetime.now().strftime("%Y年%m月%d日")
        prompt = (
            f"你是一个专业的搜索策略师。当前日期: {current_date}\n\n"
            f"用户问题: {question}\n\n"
            "请将这个问题分解为 3-5 个【互相独立、覆盖不同角度】的搜索查询。\n\n"
            "要求:\n"
            "1. 每个查询独立，不依赖其他查询的结果\n"
            "2. 覆盖不同维度: 背景/现状/对比/趋势/具体数据等\n"
            "3. 技术类问题建议中英文混合查询\n"
            "4. 查询词要具体精准\n\n"
            '只返回 JSON 字符串数组:\n["query1", "query2", "query3"]'
        )

        try:
            resp = await llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            text = resp.choices[0].message.content or ""
            cleaned = _clean_json(text)
            queries = json.loads(cleaned)
            if isinstance(queries, list) and all(
                isinstance(q, str) for q in queries
            ):
                return queries[:5]
        except Exception as e:
            logger.warning(f"查询扩展失败，使用原始问题: {e}")
        return [question]

    # ── Phase 2: Broad Search ────────────────────────────────────

    async def _broad_search(self, tavily, queries: list[str]) -> list[dict]:
        loop = asyncio.get_event_loop()

        async def _search_one(query: str) -> dict:
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda q=query: tavily.search(
                        query=q,
                        search_depth="basic",
                        include_answer=True,
                        max_results=5,
                    ),
                )
                answer = response.get("answer", "")
                results = response.get("results", [])
                content = answer if answer else "\n".join(
                    r.get("content", "") for r in results[:3]
                )
                return {"query": query, "result": content, "phase": "broad"}
            except Exception as e:
                logger.warning(f"广搜失败 [{query[:30]}]: {e}")
                return {
                    "query": query,
                    "result": f"搜索失败: {e}",
                    "phase": "broad",
                }

        results = await asyncio.gather(*[_search_one(q) for q in queries])
        return list(results)

    # ── Phase 3: Gap Analysis ────────────────────────────────────

    async def _analyze_gaps(
        self,
        llm: AsyncOpenAI,
        model: str,
        question: str,
        broad_results: list[dict],
    ) -> dict:
        results_text = "\n\n".join(
            f"查询: {r['query']}\n结果摘要: {r['result'][:300]}"
            for r in broad_results
        )

        prompt = (
            "你是一个信息质量分析师。\n\n"
            f"用户原始问题: {question}\n\n"
            f"已有广搜结果:\n{results_text}\n\n"
            "请判断这些结果是否已足够回答用户问题。\n\n"
            "返回 JSON（只返回 JSON）:\n"
            "{\n"
            '  "need_deep_search": true或false,\n'
            '  "reason": "判断原因（一句话）",\n'
            '  "gaps": [\n'
            '    {"query": "深挖搜索词", "focus": "需要补充的具体信息点"}\n'
            "  ]\n"
            "}\n\n"
            "规则:\n"
            "- need_deep_search 为 false 时 gaps 返回空数组\n"
            "- 最多 3 个最重要的信息缺口\n"
            "- 广搜已全面准确时无需深搜"
        )

        try:
            resp = await llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            text = resp.choices[0].message.content or ""
            cleaned = _clean_json(text)
            result = json.loads(cleaned)
            if isinstance(result, dict) and "need_deep_search" in result:
                return result
        except Exception as e:
            logger.warning(f"缺口分析失败: {e}")
        return {
            "need_deep_search": False,
            "reason": "分析失败，跳过深搜",
            "gaps": [],
        }

    # ── Phase 4: Deep Search ─────────────────────────────────────

    async def _deep_search(self, tavily, gaps: list[dict]) -> list[dict]:
        loop = asyncio.get_event_loop()

        async def _deep_one(gap: dict) -> dict:
            query = gap.get("query", "")
            if not query:
                return {"query": "", "result": "", "phase": "deep"}
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda q=query: tavily.search(
                            query=q,
                            search_depth="advanced",
                            include_answer=True,
                            max_results=3,
                        ),
                    ),
                    timeout=30,
                )
                answer = response.get("answer", "")
                results = response.get("results", [])
                details = []
                if answer:
                    details.append(f"摘要: {answer}")
                for r in results[:3]:
                    title = r.get("title", "")
                    content = r.get("content", "")[:500]
                    details.append(f"[{title}]: {content}")
                return {
                    "query": query,
                    "result": "\n\n".join(details) or "未找到信息",
                    "phase": "deep",
                }
            except asyncio.TimeoutError:
                logger.warning(f"深搜超时: {query[:30]}")
                return {"query": query, "result": "", "phase": "deep"}
            except Exception as e:
                logger.warning(f"深搜失败 [{query[:30]}]: {e}")
                return {"query": query, "result": "", "phase": "deep"}

        results = await asyncio.gather(*[_deep_one(g) for g in gaps[:3]])
        return [r for r in results if r["result"]]

    # ── Phase 5: Synthesis ───────────────────────────────────────

    async def _synthesize(
        self,
        llm: AsyncOpenAI,
        model: str,
        question: str,
        all_results: list[dict],
    ) -> str:
        results_text = "\n\n".join(
            f"### [{'广搜' if r['phase'] == 'broad' else '深搜'}] "
            f"{r['query']}\n{r['result']}"
            for r in all_results
        )

        prompt = (
            "你是专业的信息综合专家，基于搜索结果提供高质量综合回答。\n\n"
            f"用户问题: {question}\n\n"
            f"=== 搜索结果 ===\n{results_text}\n================\n\n"
            "输出要求:\n"
            "1. **结论先行**: 开头直接给出核心答案（2-3句）\n"
            "2. **分层展开**: 用清晰标题和层次组织详细内容\n"
            "3. **数据支撑**: 引用具体数据、时间、来源\n"
            "4. **客观中立**: 如有不同观点客观呈现\n"
            "5. **使用中文回答**\n\n"
            "请给出完整、权威的回答:"
        )

        try:
            resp = await llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content or "综合生成失败"
        except Exception as e:
            logger.error(f"结果综合失败: {e}")
            fallback = [
                f"## {r['query']}\n{r['result']}" for r in all_results
            ]
            return (
                "（综合生成失败，以下为原始搜索结果）\n\n"
                + "\n\n---\n\n".join(fallback)
            )


def _clean_json(text: str) -> str:
    """Extract JSON from LLM response that may contain markdown fences."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return text

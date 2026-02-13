import pytest

from core.agent_orchestration import (
    EvidenceCollector,
    IntentType,
    OrchestrationRouter,
    PolicyScheduler,
    LightweightPlanner,
)


def test_router_detects_memory_query_intent():
    router = OrchestrationRouter()
    route = router.route([
        {"role": "user", "content": "你还记得我今天中午吃的啥吗"}
    ])
    assert route.intent == IntentType.MEMORY


def test_router_detects_realtime_info_intent():
    router = OrchestrationRouter()
    route = router.route([
        {"role": "user", "content": "帮我查一下上海浦东2月成交数据"}
    ])
    assert route.intent == IntentType.REALTIME_INFO


def test_router_detects_realtime_info_intent_for_housing_price_query():
    router = OrchestrationRouter()
    route = router.route([
        {"role": "user", "content": "2026年上海房价"}
    ])
    assert route.intent == IntentType.REALTIME_INFO


@pytest.mark.asyncio
async def test_router_uses_llm_structured_classifier_when_available():
    class StubLLM:
        async def chat_with_tools(
            self,
            messages,
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            response_format=None,
        ):
            return {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "name": "classify_intent",
                        "arguments": '{"intent":"realtime_info","reason":"需要查询最新市场数据","confidence":0.93}',
                    }
                ],
            }

    router = OrchestrationRouter(llm_engine=StubLLM(), use_llm_classifier=True)
    route = await router.route_async([
        {"role": "user", "content": "给我看下明年上海房价走势"}
    ])
    assert route.intent == IntentType.REALTIME_INFO
    assert route.reason.startswith("llm_structured:")
    assert route.confidence == pytest.approx(0.93, rel=1e-6)


@pytest.mark.asyncio
async def test_router_falls_back_when_llm_returns_invalid_intent():
    class StubLLM:
        async def chat_with_tools(
            self,
            messages,
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            response_format=None,
        ):
            return {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "name": "classify_intent",
                        "arguments": '{"intent":"not_a_valid_intent","reason":"bad","confidence":0.9}',
                    }
                ],
            }

    router = OrchestrationRouter(llm_engine=StubLLM(), use_llm_classifier=True)
    route = await router.route_async([
        {"role": "user", "content": "你还记得我今天中午吃的啥吗"}
    ])
    assert route.intent == IntentType.MEMORY


def test_policy_scheduler_returns_realtime_search_allowlist():
    scheduler = PolicyScheduler()
    policy = scheduler.build(IntentType.REALTIME_INFO)
    assert policy.allow_tools is True
    assert "search_web" in policy.allowed_tools
    assert "super_search" in policy.allowed_tools


def test_policy_scheduler_disables_tools_for_memory_intent():
    scheduler = PolicyScheduler()
    policy = scheduler.build(IntentType.MEMORY)
    assert policy.allow_tools is False
    assert policy.allowed_tools == []


def test_planner_builds_steps_and_tool_requirement():
    router = OrchestrationRouter()
    scheduler = PolicyScheduler()
    planner = LightweightPlanner()

    route = router.route([
        {"role": "user", "content": "帮我查今天美元兑人民币汇率"}
    ])
    policy = scheduler.build(route.intent)
    plan = planner.build(route=route, policy=policy, messages=[
        {"role": "user", "content": "帮我查今天美元兑人民币汇率"}
    ])

    assert plan.plan_id
    assert len(plan.steps) >= 1
    assert plan.requires_tools is True


def test_evidence_collector_renders_summary():
    collector = EvidenceCollector(max_items=3)
    collector.add_tool_evidence(
        tool_name="search_web",
        tool_args={"query": "上海浦东2月成交数据"},
        tool_result="来源A: 2月成交量xx",
    )

    summary = collector.render_summary()
    assert "[Evidence]" in summary
    assert "search_web" in summary
    assert "上海浦东2月成交数据" in summary

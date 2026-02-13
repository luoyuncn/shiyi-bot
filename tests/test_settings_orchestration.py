from config.settings import AgentConfig


def test_agent_config_has_orchestration_defaults():
    agent = AgentConfig()
    assert agent.orchestration.enabled is True
    assert agent.orchestration.use_llm_intent_classifier is True
    assert agent.orchestration.max_plan_steps == 3
    assert agent.orchestration.force_evidence_section is True
    assert agent.orchestration.tool_budget_by_intent["chat"] == 0


def test_agent_config_parses_custom_orchestration_values():
    agent = AgentConfig(
        orchestration={
            "enabled": True,
            "use_llm_intent_classifier": False,
            "max_plan_steps": 4,
            "force_evidence_section": False,
            "tool_budget_by_intent": {
                "chat": 0,
                "memory": 0,
                "realtime_info": 2,
                "workspace_action": 3,
            },
        }
    )

    assert agent.orchestration.use_llm_intent_classifier is False
    assert agent.orchestration.max_plan_steps == 4
    assert agent.orchestration.force_evidence_section is False
    assert agent.orchestration.tool_budget_by_intent["workspace_action"] == 3

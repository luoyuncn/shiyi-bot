"""Agent registry"""
from loguru import logger
from agents.base_agent import BaseAgent


class AgentRegistry:
    """Agent registry - manages all sub-agents"""
    _agents: dict[str, BaseAgent] = {}

    @classmethod
    async def initialize(cls, config):
        """Load built-in agents"""
        from agents.builtin.code_assistant import CodeAssistantAgent
        from agents.builtin.general_qa import GeneralQAAgent

        cls.register(CodeAssistantAgent(config))
        cls.register(GeneralQAAgent(config))
        logger.info(f"AgentRegistry 初始化完成，已加载 {len(cls._agents)} 个子Agent")

    @classmethod
    def register(cls, agent: BaseAgent):
        """Register an agent"""
        cls._agents[agent.name] = agent
        logger.debug(f"注册子Agent: {agent.name}")

    @classmethod
    def get_agent(cls, name: str) -> BaseAgent | None:
        """Get agent by name"""
        return cls._agents.get(name)

    @classmethod
    def list_agents(cls) -> list[dict]:
        """List all agents with descriptions"""
        return [
            {"name": name, "description": agent.description}
            for name, agent in cls._agents.items()
        ]

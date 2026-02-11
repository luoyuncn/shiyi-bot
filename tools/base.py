"""Tool base classes"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """Tool parameter definition"""
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    """Tool definition - converts to OpenAI function calling format"""
    name: str
    description: str
    parameters: Dict[str, ToolParameter]

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function definition format"""
        properties = {}
        required = []

        for param_name, param in self.parameters.items():
            properties[param_name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param_name]["enum"] = param.enum

            if param.required:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class BaseTool(ABC):
    """Tool base class"""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Tool definition"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute tool (subclass implements)"""
        pass

    async def validate_params(self, params: dict):
        """Validate parameters (optional override)"""
        pass

    async def run(self, **kwargs) -> Any:
        """Template method: validate → execute → log"""
        await self.validate_params(kwargs)
        result = await self.execute(**kwargs)
        await self._log_execution(kwargs, result)
        return result

    async def _log_execution(self, params: dict, result: Any):
        """Log execution"""
        from loguru import logger
        logger.debug(f"工具执行: {self.definition.name} | 参数: {params}")

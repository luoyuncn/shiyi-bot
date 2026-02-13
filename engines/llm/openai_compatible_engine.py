"""OpenAI协议兼容的大语言模型引擎"""
from openai import AsyncOpenAI
from engines.base import BaseEngine
from loguru import logger
from typing import AsyncGenerator, List, Dict


class OpenAICompatibleEngine(BaseEngine):
    """OpenAI协议兼容的LLM引擎（支持DeepSeek等）"""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        初始化LLM引擎

        Args:
            api_base: API基础URL
            api_key: API密钥
            model: 模型名称
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = None
        self.conversation_history: List[Dict[str, str]] = []

    async def initialize(self):
        """初始化OpenAI客户端"""
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            logger.info(f"LLM引擎已初始化: {self.model}")
            logger.debug(f"API Base: {self.api_base}")

        except Exception as e:
            logger.error(f"初始化LLM引擎失败: {e}")
            raise

    async def chat_stream(self, message: str) -> AsyncGenerator[str, None]:
        """
        流式对话

        Args:
            message: 用户消息

        Yields:
            生成的token
        """
        if not self.client:
            raise RuntimeError("LLM引擎未初始化")

        # 添加用户消息到历史
        self.conversation_history.append({"role": "user", "content": message})

        # 构造消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]

        try:
            # 流式请求
            full_response = ""

            logger.debug(f"发起LLM请求: {message[:50]}...")

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield token

            # 添加助手回复到历史
            self.conversation_history.append({"role": "assistant", "content": full_response})

            logger.debug(f"🤖 LLM回复: {full_response}")

        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            # 生成错误提示
            error_msg = "抱歉，我遇到了一些问题。"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            yield error_msg

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None
    ) -> Dict:
        """
        支持工具调用的对话（非流式）

        Args:
            messages: 消息列表
            tools: 工具定义列表（OpenAI格式）

        Returns:
            响应字典，包含 content 或 tool_calls
        """
        if not self.client:
            raise RuntimeError("LLM引擎未初始化")

        try:
            logger.debug(f"发起LLM请求（支持工具）: {len(messages)} 条消息")

            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            # 如果提供了工具定义，添加到请求中
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**request_params)

            choice = response.choices[0]
            message = choice.message

            # Extract usage data if available
            # 如果有 usage 数据则提取出来
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }

            # 检查是否有工具调用
            if hasattr(message, 'tool_calls') and message.tool_calls:
                logger.debug(f"🔧 LLM请求调用工具: {[tc.function.name for tc in message.tool_calls]}")
                result = {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                        for tc in message.tool_calls
                    ],
                }
            else:
                # 普通文本响应
                content = message.content or ""
                logger.debug(f"🤖 LLM回复: {content}")
                result = {
                    "type": "text",
                    "content": content,
                }

            if usage:
                result["usage"] = usage
            return result

        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            raise

    async def chat_with_tools_stream(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        支持工具调用的流式对话

        Args:
            messages: 消息列表
            tools: 工具定义列表（OpenAI格式）

        Yields:
            响应字典:
            {"type": "text_delta", "content": "..."}
            {"type": "tool_calls", "tool_calls": [...]}
            {"type": "usage", "usage": {...}}
        """
        if not self.client:
            raise RuntimeError("LLM引擎未初始化")

        try:
            logger.debug(f"发起LLM流式请求（支持工具）: {len(messages)} 条消息")

            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                # Add stream_options for usage if supported (OpenAI standard)
                # 若服务端支持，可开启 stream_options 返回 usage（OpenAI 标准）
                # request_params["stream_options"] = {"include_usage": True}
                # 示例：开启增量流中的 usage 统计

            stream = await self.client.chat.completions.create(**request_params)

            tool_calls_buffer = {}  # index -> dict
            # 工具调用增量缓冲：索引 -> 调用对象

            async for chunk in stream:
                # Handle usage if present in the chunk (some providers send it in the last chunk)
                # 处理 chunk 中携带的 usage（部分服务商会在最后一个 chunk 返回）
                if hasattr(chunk, "usage") and chunk.usage:
                    yield {
                        "type": "usage",
                        "usage": {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 1. Text Content
                # 1. 文本增量
                if delta.content:
                    yield {"type": "text_delta", "content": delta.content}

                # 2. Tool Calls
                # 2. 工具调用增量
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": "",
                                "function": {"name": "", "arguments": ""}
                            }

                        entry = tool_calls_buffer[idx]
                        if tc.id:
                            entry["id"] += tc.id
                        if tc.function:
                            if tc.function.name:
                                entry["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                entry["function"]["arguments"] += tc.function.arguments

            # End of stream logic
            # 流结束后的收尾逻辑
            if tool_calls_buffer:
                # Convert buffer to list
                # 把缓冲字典转换为有序列表
                tool_calls = []
                for idx in sorted(tool_calls_buffer.keys()):
                    entry = tool_calls_buffer[idx]
                    tool_calls.append({
                        "id": entry["id"],
                        "name": entry["function"]["name"],
                        "arguments": entry["function"]["arguments"]
                    })

                logger.debug(f"🔧 LLM流式调用工具: {[t['name'] for t in tool_calls]}")
                yield {
                    "type": "tool_calls",
                    "tool_calls": tool_calls
                }

            # Note: For text responses, we've already yielded all deltas.
            # 注意：文本响应的所有增量已经在上面逐步产出。
            # The AgentCore will responsible for accumulating them for history.
            # 后续由 AgentCore 负责把这些增量拼接并写入历史。

        except Exception as e:
            logger.error(f"LLM流式生成失败: {e}")
            raise

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        logger.debug("对话历史已清空")

    def get_history_length(self) -> int:
        """获取对话历史长度"""
        return len(self.conversation_history)

    async def cleanup(self):
        """清理资源"""
        if self.client:
            await self.client.close()
        self.conversation_history = []
        logger.info("LLM引擎已清理")

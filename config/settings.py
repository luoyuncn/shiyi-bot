"""配置加载器"""
from pydantic import BaseModel, Field
import yaml
import os
import re
from pathlib import Path
from typing import Optional


class SystemConfig(BaseModel):
    """系统配置"""
    name: str = "Shiyi"
    log_level: str = "INFO"
    audio_sample_rate: int = 16000


class WakeWordConfig(BaseModel):
    """唤醒词配置"""
    engine: str = "openwakeword"
    model_path: Optional[str] = None
    threshold: float = 0.5
    skip_wake_word: bool = False  # 跳过唤醒词，直接进入监听（调试用）


class VADConfig(BaseModel):
    """VAD配置"""
    engine: str = "silero"
    silence_duration_ms: int = 500
    max_recording_seconds: int = 10
    continuous_window_seconds: int = 3


class STTConfig(BaseModel):
    """语音识别配置"""
    engine: str = "tencent"
    app_id: str
    secret_id: str
    secret_key: str
    region: str = "ap-guangzhou"


class LLMConfig(BaseModel):
    """大语言模型配置"""
    engine: str = "openai_compatible"
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    stream: bool = True
    system_prompt: str


class TTSConfig(BaseModel):
    """语音合成配置"""
    engine: str = "edge"
    voice: str
    rate: str = "+0%"
    pitch: str = "+0Hz"


class AudioConfig(BaseModel):
    """音频设备配置"""
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    pulse_sink: Optional[str] = None  # PulseAudio 输出设备名，None 表示默认
    chunk_size: int = 1024
    input_channels: int = 1


class ToolsConfig(BaseModel):
    """Tools configuration"""
    builtin: list[str] = []
    mcp: dict = {"enabled": False, "servers": []}


class MemoryConfig(BaseModel):
    """Memory configuration"""
    sqlite_path: str = "data/sessions.db"
    memory_root: str = "data/memory"
    cache_size: int = 100
    auto_flush_interval: int = 60
    embedding_dimension: int = 128
    embedding_retry_limit: int = 3
    embedding_retry_base_seconds: int = 10
    embedding_poll_interval: int = 5
    kuzu_path: str = "data/kuzu"  # Kuzu 图数据库路径


class TUIConfig(BaseModel):
    """TUI 终端界面配置"""
    nerd_font: bool = False  # 启用 Nerd Font 图标（需用户自行安装字体）


class ContextBudgetConfig(BaseModel):
    """Context token 预算配置"""
    total_tokens: int = 6000
    system_reserved_tokens: int = 800


class ComplexityDetectorConfig(BaseModel):
    """复杂任务启发式检测配置（零额外 LLM 调用）"""
    enabled: bool = True
    message_length_threshold: int = 80
    multi_tool_threshold: int = 2
    step_keywords: list[str] = Field(default_factory=lambda: [
        "步骤", "先.*再.*然后", "分析并", "重构", "迁移", "帮我做", "帮我搞",
    ])
    multi_tool_domains: dict[str, list[str]] = Field(default_factory=lambda: {
        "search": ["搜索", "查一下", "找找"],
        "file":   ["文件", "代码", "读取"],
        "shell":  ["执行", "运行", "命令"],
    })
    continuation_markers: list[str] = Field(default_factory=lambda: [
        "接下来", "第一步", "下一步",
    ])


class AgentConfig(BaseModel):
    """Agent 配置"""
    enable_sub_agents: bool = True
    max_tool_iterations: int = 5
    context_budget: ContextBudgetConfig = Field(default_factory=ContextBudgetConfig)
    complexity_detector: ComplexityDetectorConfig = Field(default_factory=ComplexityDetectorConfig)


class Settings(BaseModel):
    """完整配置"""
    system: SystemConfig
    channels: dict = {}
    tui: TUIConfig = TUIConfig()
    wake_word: WakeWordConfig
    vad: VADConfig
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig
    audio: AudioConfig
    agent: AgentConfig = AgentConfig()
    tools: ToolsConfig
    memory: MemoryConfig


def _load_dotenv(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        # Strip optional quotes
        # 去掉可选的引号包裹
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ[key] = value


def _apply_llm_env_compatibility() -> None:
    """Normalize LLM env vars and keep backward compatibility."""
    legacy_to_new = {
        "DEEPSEEK_API_KEY": "LLM_API_KEY",
        "DEEPSEEK_API_BASE": "LLM_BASE_URL",
        "DEEPSEEK_MODEL": "LLM_MODEL",
    }

    for legacy_var, new_var in legacy_to_new.items():
        if new_var not in os.environ and legacy_var in os.environ:
            os.environ[new_var] = os.environ[legacy_var]

    # Keep previous behavior as defaults when provider/model is not explicitly set
    # 当 provider/model 未显式设置时，保持旧行为作为默认值
    os.environ.setdefault("LLM_BASE_URL", "https://api.deepseek.com/v1")
    os.environ.setdefault("LLM_MODEL", "deepseek-chat")


def load_config(config_path: str = "config/config.yaml") -> Settings:
    """
    加载配置文件并替换环境变量

    Args:
        config_path: 配置文件路径

    Returns:
        Settings对象
    """
    # 先加载.env到环境变量（若存在）
    _load_dotenv(Path(".env"))
    _apply_llm_env_compatibility()

    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_text = f.read()

    # 替换环境变量 ${VAR_NAME}
    def replace_env(match):
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"环境变量 {var_name} 未设置，请检查.env文件")
        return value

    config_text = re.sub(r'\$\{(\w+)\}', replace_env, config_text)

    # 解析YAML
    config_dict = yaml.safe_load(config_text)

    # 创建Settings对象
    return Settings(**config_dict)

# ShiYiBot - 私人智能助理 V2.0

基于树莓派4B的智能助手，支持**语音**、**CLI文字**、**HTTP API** 三通道并行运行，具备工具调用、子Agent协作和会话持久化能力。

## 功能特性

### 通道（三选一或并行）

| 通道 | 说明 | 启用方式 |
|------|------|---------|
| 语音通道 | 唤醒词 → VAD录音 → STT → LLM → TTS | `channels.voice.enabled: true` |
| CLI通道 | 终端文字交互，支持多会话管理 | `channels.cli.enabled: true` |
| API通道 | FastAPI HTTP服务，JSONL流式响应 | `channels.api.enabled: true` |

### 工具调用

LLM可调用以下内置工具：

| 工具 | 功能 |
|------|------|
| `search_web` | DuckDuckGo搜索，无需API密钥 |
| `file_operations` | 读写文件（read/write/append/list）|
| `execute_shell` | 执行Shell命令（含安全黑名单）|
| MCP工具 | 可接入外部MCP协议工具服务器 |

### 子Agent系统

主Agent可将任务委派给专业子Agent：

- **code_assistant** — 代码编写、调试、测试（temperature=0.3）
- **general_qa** — 知识查询、分析推理（temperature=0.7）

### 会话管理

- SQLite持久化 + LRU内存缓存（双层架构）
- 单用户多会话隔离
- 自动token窗口管理

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) 包管理工具

### 2. 安装

```bash
git clone <your-repo-url> shiyi-bot
cd shiyi-bot

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

uv pip install -e .
```

### 3. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 API 密钥
```

需要配置的密钥：

```env
DEEPSEEK_API_KEY=your_key_here

# 仅语音通道需要：
TENCENT_APP_ID=your_app_id
TENCENT_SECRET_ID=your_secret_id
TENCENT_SECRET_KEY=your_secret_key
```

### 4. 运行

```bash
# 默认启动 CLI 文字通道（config.yaml 中 cli.enabled: true）
python main.py

# 同时启动 CLI + API 通道
# 修改 config.yaml: channels.api.enabled: true
python main.py
```

---

## CLI 通道使用

```
ShiYiBot > 你好

ShiYiBot > 帮我搜索一下 Python asyncio 最佳实践
[工具调用] search_web: {"query": "Python asyncio 最佳实践"}
[工具结果] search_web: 搜索结果：...
这里是搜索结果的总结...

ShiYiBot > /new       # 创建新会话
ShiYiBot > /list      # 列出所有会话
ShiYiBot > /switch    # 切换会话
ShiYiBot > /help      # 帮助
Ctrl+C                # 退出
```

---

## API 通道

启用后默认监听 `http://0.0.0.0:8000`。

### 端点

```
POST /api/chat              非流式对话
POST /api/chat/stream       流式对话（JSONL格式）
GET  /api/sessions          列出会话
POST /api/sessions          创建会话
DELETE /api/sessions/{id}   删除会话
GET  /health                健康检查
```

### 流式响应格式（JSONL）

```json
{"type": "text", "content": "你好"}
{"type": "tool_call", "tool": "search_web", "args": {"query": "..."}}
{"type": "tool_result", "tool": "search_web", "result": "..."}
{"type": "sub_agent_start", "agent": "code_assistant", "task": "..."}
{"type": "sub_agent_done", "agent": "code_assistant"}
{"type": "done"}
```

### 示例请求

```bash
# 流式对话
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我写一个 Python 冒泡排序", "session_id": "test-1"}'

# 创建会话
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"source": "curl"}}'
```

---

## 配置说明

主配置文件：`config/config.yaml`

```yaml
# 通道开关
channels:
  voice:
    enabled: false   # 需要硬件（麦克风）
  cli:
    enabled: true
  api:
    enabled: false
    host: "0.0.0.0"
    port: 8000

# Agent配置
agent:
  enable_sub_agents: true   # 启用子Agent委派
  max_context_tokens: 4000

# 工具配置
tools:
  builtin:
    - file_operations
    - execute_shell
    - search_web
  mcp:
    enabled: false
    servers: []
    # 接入外部MCP服务器示例：
    # - url: "http://localhost:3000"
    #   name: "my_tools"

# 记忆系统
memory:
  sqlite_path: "data/sessions.db"
  cache_size: 100
```

---

## 项目结构

```
shiyi-bot/
├── main.py                    # 主入口
├── config/
│   ├── config.yaml            # 主配置
│   └── settings.py            # 配置加载（Pydantic）
├── channels/                  # 通道层
│   ├── base.py                # 通道抽象基类
│   ├── text_cli_channel.py    # CLI通道
│   ├── text_api_channel.py    # FastAPI通道
│   └── voice_channel.py       # 语音通道（包装器）
├── core/
│   ├── orchestrator.py        # 总调度器
│   ├── agent_core.py          # Agent核心（LLM推理+工具调用）
│   ├── session_manager.py     # 会话管理器
│   ├── assistant.py           # 语音版控制器（保留）
│   ├── state_machine.py       # 状态机
│   └── sentence_splitter.py   # 句子切分器
├── agents/                    # 子Agent系统
│   ├── base_agent.py          # 抽象基类（含共享LLM循环）
│   ├── registry.py            # Agent注册器
│   └── builtin/
│       ├── code_assistant.py  # 代码助手
│       └── general_qa.py      # 通用问答
├── tools/                     # 工具系统
│   ├── base.py                # 工具基类
│   ├── registry.py            # 工具注册器
│   ├── mcp_client.py          # MCP协议客户端
│   └── builtin/
│       ├── search_web.py      # DuckDuckGo搜索
│       ├── file_operations.py # 文件操作
│       └── execute_shell.py   # Shell命令执行
├── memory/
│   ├── storage.py             # SQLite异步存储
│   └── cache.py               # LRU内存缓存
├── engines/                   # AI引擎（语音版）
│   ├── llm/                   # LLM引擎（文字版复用）
│   ├── stt/                   # 腾讯云语音识别
│   ├── tts/                   # Edge-TTS语音合成
│   ├── vad/                   # Silero VAD
│   └── wake_word/             # openWakeWord
└── audio/                     # 音频处理
```

---

## 技术栈

| 模块 | 技术 |
|------|------|
| LLM | DeepSeek（OpenAI兼容接口）|
| Web框架 | FastAPI + Streamable HTTP |
| 流式协议 | JSONL (application/x-ndjson) |
| 数据库 | SQLite + SQLAlchemy async |
| 缓存 | LRU内存缓存 |
| 搜索 | DuckDuckGo (ddgs，免费无需API) |
| MCP工具 | httpx异步HTTP客户端 |
| 唤醒词 | openWakeWord（本地）|
| VAD | Silero VAD (torch) |
| STT | 腾讯云一句话识别 |
| TTS | Microsoft Edge-TTS |
| 包管理 | uv |

---

## 故障排除

### PyAudio安装失败

**Windows:**
```bash
# 下载预编译wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
uv pip install PyAudio-0.2.14-cpXX-cpXX-win_amd64.whl
```

**Linux:**
```bash
sudo apt install portaudio19-dev
uv pip install pyaudio
```

### 找不到音频设备

```python
from audio.recorder import AudioRecorder
AudioRecorder().list_devices()
```

在 `config/config.yaml` 中指定设备索引：

```yaml
audio:
  input_device_index: 1
  output_device_index: 2
```

### API调用失败

1. 检查 `.env` 文件中的密钥是否正确
2. 检查网络连接
3. 开启 DEBUG 日志：`config.yaml` 中 `system.log_level: "DEBUG"`

---

## 致谢

- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Edge-TTS](https://github.com/rany2/edge-tts)
- [ddgs](https://github.com/deedy5/ddgs)
- [uv](https://github.com/astral-sh/uv)

---

MIT License

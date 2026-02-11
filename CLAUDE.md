# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Shiyi** (V2.0) is a Chinese-language personal assistant deployable on any host, supporting three parallel channels: voice, CLI (terminal), and API (FastAPI HTTP). It uses a multi-channel orchestrator, an LLM agent with tool calling, a sub-agent delegation system, and persistent session memory.

## Commands

```bash
# Setup (uses uv package manager)
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env  # Fill in: DEEPSEEK_API_KEY, TENCENT_* (voice only)

# Run (default: CLI channel)
python main.py        # or: shiyi  (after uv pip install -e .)

# Debug (set log_level: "DEBUG" in config/config.yaml first)
python main.py

# List audio devices (voice channel)
python -c "from audio.recorder import AudioRecorder; AudioRecorder().list_devices()"

# Tests
pytest
pytest tests/test_agent_core.py          # single file
pytest tests/test_specific.py::test_fn   # single test

# Lint / format
ruff check .
ruff format .
```

## Architecture

### Orchestrator Pattern (`core/orchestrator.py`)

`Orchestrator` is the top-level coordinator. On startup it:
1. Initialises `ToolRegistry` (loads built-in tools + optional MCP tools)
2. Initialises `AgentRegistry` (discovers sub-agents)
3. Starts `SessionManager` (dual-layer memory)
4. Launches all enabled channels as parallel async tasks via `asyncio.gather`

All channels share a single `AgentCore` and `SessionManager` instance.

### Message Processing Pipeline

Every channel feeds the same flow:

```
User input (text / voice)
  → SessionManager.save_message()   # persist to cache + DB
  → SessionManager.get_session()    # full conversation history
  → AgentCore.process_message_stream(messages)
      ├─ LLM call with OpenAI tool definitions
      ├─ If tool_calls: ToolRegistry.execute() → append results → loop (max 5)
      └─ Yield typed events: "text" | "tool_call" | "tool_result" | "error" | "done"
  → Channel renders events (print / JSONL stream / TTS)
  → SessionManager.save_message()   # persist assistant response
```

### Channels (`channels/`)

| Channel | Class | Notes |
|---------|-------|-------|
| CLI | `TextCLIChannel` | `asyncio.to_thread(input)`, session commands `/new /list /switch /help` |
| API | `TextAPIChannel` | FastAPI on `:8000`, streaming JSONL; routes: `POST /api/chat/stream`, CRUD `/api/sessions` |
| Voice | `VoiceChannel` | Wraps legacy `AssistantCore`; adds wake-word → VAD → STT → TTS pipeline |

Enable channels in `config/config.yaml` under the `channels:` key.

### Agent & Tool System

**Tool extensibility:** drop a class in `tools/builtin/` with a `Tool` class and `definition` property — `ToolRegistry` auto-discovers it.

Built-in tools: `search_web` (DuckDuckGo), `file_operations` (read/write/list), `execute_shell` (blacklisted commands).

**Sub-agent extensibility:** drop a `BaseAgent` subclass in `agents/builtin/` — `AgentRegistry` auto-discovers it.

Built-in agents: `code_assistant` (temp=0.3, tools: shell + file ops), `general_qa` (temp=0.7, tools: search).

Both registries use dynamic import; no manual registration needed.

### Memory (`memory/`)

Dual-layer architecture managed by `core/session_manager.py`:
- **LRU cache** (`memory/cache.py`): hot sessions in an OrderedDict (default 100 entries)
- **SQLite** (`memory/storage.py`): async SQLAlchemy + aiosqlite, tables `SessionRecord` / `MessageRecord`

Reads hit cache first; cache misses load from DB. Auto-flush loop persists cache stats periodically.

### Voice Engine System (`engines/`)

All engines extend `BaseEngine` (abstract: `initialize()`, `cleanup()`):

| Engine | Implementation | Tech |
|--------|---------------|------|
| Wake word | `wake_word/openwakeword_engine.py` | openWakeWord (local) |
| VAD | `vad/silero_vad_engine.py` | Silero VAD (torch) |
| STT | `stt/tencent_stt_engine.py` | Tencent Cloud API |
| LLM | `llm/openai_compatible_engine.py` | DeepSeek (OpenAI-compatible) |
| TTS | `tts/edge_tts_engine.py` | Microsoft Edge-TTS |

Voice streaming pipeline: LLM token stream → `SentenceSplitter` (splits on 。！？；… or after 100+ chars) → sentence queue → TTS synthesis → chunked audio playback via producer-consumer `asyncio.Queue`.

Audio recording fallback chain: PyAudio default → ALSA device → `arecord` subprocess.

### State Machine (`core/state_machine.py`)

Used by the voice channel only: `IDLE → LISTENING → PROCESSING → SPEAKING → CONTINUOUS`

- **CONTINUOUS**: 3-second follow-up window (no re-activation needed)

## Configuration

- `config/config.yaml`: All settings; supports `${VAR_NAME}` env var interpolation
- `.env`: `DEEPSEEK_API_KEY`, `TENCENT_APP_ID`, `TENCENT_SECRET_ID`, `TENCENT_SECRET_KEY`
- `config/settings.py`: Pydantic V2 models — each section (`SystemConfig`, `LLMConfig`, etc.) maps 1-to-1 to a YAML key

Key config sections: `system`, `channels`, `llm`, `agent`, `tools`, `memory`, `wake_word`, `vad`, `stt`, `tts`, `audio`

## Dependencies

Managed via `uv` / `pyproject.toml`. Notable constraints: `numpy<2`, `torch==2.1.2`, `torchaudio==2.1.2` (Silero VAD compatibility). Linter: `ruff` (line-length=100, target=py310).

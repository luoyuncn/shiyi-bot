# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**小跟班 (TUI-Assistant)** is a Chinese-language voice assistant for Raspberry Pi 4B. It uses local wake word detection, VAD-based recording, Tencent Cloud STT, DeepSeek LLM, and Microsoft Edge TTS.

## Commands

```bash
# Setup (uses uv package manager)
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env  # Then fill in API keys

# Run
python main.py

# Debug (set log_level: "DEBUG" in config/config.yaml first)
python main.py

# List audio devices
python -c "from audio.recorder import AudioRecorder; AudioRecorder().list_devices()"

# Tests
pytest
pytest tests/test_specific.py::test_function  # single test
pytest-asyncio for async tests
```

## Architecture

### State Machine

Five states in `core/state_machine.py`: `IDLE → LISTENING → PROCESSING → SPEAKING → CONTINUOUS`

- **IDLE**: Waiting for wake word
- **LISTENING**: VAD-based recording until silence (500ms default)
- **PROCESSING**: STT transcription + LLM streaming
- **SPEAKING**: TTS playback
- **CONTINUOUS**: 3-second window for follow-up (no re-activation needed)

### Core Orchestrator (`core/assistant.py`)

`AssistantCore` manages all async tasks:
- `_main_loop()`: State-driven dispatch
- `_tts_playback_loop()`: Independent TTS consumer task
- Producer-consumer pattern via `asyncio.Queue`: LLM stream → sentence queue → TTS

### Engine System (`engines/`)

All engines extend `BaseEngine` (abstract: `initialize()`, `cleanup()`):

| Engine | Implementation | Tech |
|--------|---------------|------|
| Wake word | `openwakeword_engine.py` | openWakeWord (local) |
| VAD | `silero_vad_engine.py` | Silero VAD (torch) |
| STT | `tencent_stt_engine.py` | Tencent Cloud API |
| LLM | `openai_compatible_engine.py` | DeepSeek (OpenAI-compatible) |
| TTS | `edge_tts_engine.py` | Microsoft Edge-TTS |

### Streaming Pipeline

LLM token stream → `SentenceSplitter` (splits on Chinese punctuation 。！？；…, or after 100+ chars) → sentence queue → TTS synthesis → chunked audio playback

### Audio Fallback Chain (`audio/recorder.py`)

Tries in order: PyAudio default → ALSA device → `arecord` subprocess (Raspberry Pi compatibility)

## Configuration

- `config/config.yaml`: All settings; supports `${VAR_NAME}` env var interpolation
- `.env`: Credentials (`TENCENT_APP_ID`, `TENCENT_SECRET_ID`, `TENCENT_SECRET_KEY`, `DEEPSEEK_API_KEY`)
- `config/asound.conf`: Auto-generated ALSA config for Raspberry Pi

Key config sections: `system`, `wake_word`, `vad`, `stt`, `llm`, `tts`, `audio`

## Dependencies

Managed via `uv` / `pyproject.toml`. Notable constraints: `numpy<2`, `torch==2.1.2`, `torchaudio==2.1.2` (Silero VAD compatibility).

# reachy-voice-server (isolated env)

The Kokoro voice engines need `transformers`/`kokoro`, which require
**`huggingface-hub>=1.5`** — but the main app pins **`huggingface-hub==1.3.0`**
(for robot / HF-Space compatibility). Running both in one venv means `uv sync`
keeps reverting hf-hub and the Kokoro server stops importing.

This folder is a **separate uv project** so the voice server gets its own
environment. It runs the repo's self-contained `scripts/voice_server.py`
(no `reachy_local_assistant` import on the Kokoro path).

## Setup
```bash
cd voice-server
uv sync                      # creates ./.venv with kokoro + hf-hub>=1.5
```

## Run (Swedish neural Kokoro, MPS, auto-language)
```bash
cd voice-server
PYTORCH_ENABLE_MPS_FALLBACK=1 \
SWEDISH_KOKORO_PATH=/abs/path/to/ai-smarthome/swedish-kokoro \
  uv run python ../scripts/voice_server.py --engine kokoro-svml --host 0.0.0.0 --port 8880
```
Other engines: `--engine kokoro-sv` (ONNX Swedish) or `--engine kokoro` (base Kokoro).
(For the light **Piper** engine, just run `scripts/voice_server.py --engine piper`
from the *app's* venv — Piper has no hf-hub conflict.)

## Point the app at it
```dotenv
TTS_BACKEND=remote
TTS_URL=http://<host>:8880/v1/audio/speech
```

# kokoro-voice-server

A standalone, OpenAI-compatible voice server: **Kokoro TTS** on
`/v1/audio/speech` and **faster-whisper STT** on `/v1/audio/transcriptions`.
Client apps stay thin — they hold no speech models, just a URL.

Includes the fine-tuned **Swedish Kokoro** with 10 named voices
(weights auto-download from [Joakim/kokoro-sv-voices](https://huggingface.co/Joakim/kokoro-sv-voices)),
plus base Kokoro's other languages, with per-utterance language routing.

Used by:
- [reachy_mini_conversation_app](https://github.com/joakimeriksson/reachy_mini_conversation_app) — talks to it via `TTS_URL`/`STT_URL`
- **CandyTron 4000** ([mcp-agents](https://github.com/joakimeriksson/mcp-agents)) — the candy-robot demo's face/speech clients

Extracted from `reachy_mini_conversation_app` (history preserved) once a second
project started using it. It was already its own uv project there: the
Kokoro/torch/transformers stack fights typical app pins (`huggingface-hub`,
`pydantic`), so it wants an isolated environment either way.

## Setup

```bash
uv sync        # creates ./.venv with kokoro + torch + faster-whisper + kokoro-sv
```

## Run

```bash
# Swedish neural Kokoro (10 named voices) + English etc., with transcription
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  uv run python voice_server.py \
      --engine kokoro-svml --voice Stina --host 0.0.0.0 --port 8880 --whisper base
```

| Engine | What it is | Needs |
|--------|------------|-------|
| `kokoro` (default) | Base Kokoro (en/es/fr/it/pt/hi/zh/ja — **no Swedish**) | nothing extra |
| `kokoro-svml` | Fine-tuned Swedish + all base languages, neural NST g2p, named voice packs (Stina, Björn, Nils, …) | nothing extra — the g2p is vendored in the `kokoro-sv` dependency; weights auto-download from `--voices-repo` |
| `kokoro-sv` | Older Swedish-only ONNX path, espeak `sv` g2p, single voice | `--svml-path`/`SWEDISH_KOKORO_PATH` pointing at a [swedish-kokoro](https://github.com/joakimeriksson/swedish-kokoro) checkout |

`--svml-path` (or `SWEDISH_KOKORO_PATH`) also works with `kokoro-svml` to use a
local swedish-kokoro checkout's g2p instead of the vendored one — handy when
developing the g2p itself.

`--whisper <tiny|base|small|medium>` enables transcription (default `off`).
Clients can use it to store **text** rather than raw audio in conversation
history — without it conversations still work, but history stays heavy.

### Voices, and which languages are allowed (`kokoro-svml`)

Ten fine-tuned **Swedish** voice packs: `Stina` (default), `Alice`, `Ebba`,
`Elsa`, `Greta`, `Anton`, `Björn`, `Lars`, `Nils`, `Oskar`.

`--langs` (default `sv,en,fr,es,it`, or `$KOKORO_SV_LANGS`) is the allow-list of
languages the server may speak. A detected language outside it is **clamped to
the first entry** — that guard is deliberate: it stops one mis-heard snippet
from sending a robot off into, say, Hindi.

| Code | Language | Voice used |
|------|----------|-----------|
| `sv` | Swedish | the fine-tuned packs (Stina, Björn, …) |
| `en` | English | `af_heart` |
| `fr` | French | `ff_siwis` |
| `es` | Spanish | `ef_dora` |
| `it` | Italian | `if_sara` |

Also available in base Kokoro but **off by default**: `pt` (`pf_dora`),
`hi` (`hf_alpha`), `en-gb` (`bf_emma`), plus `ja` and `zh` — those two need
extra g2p backends (`uv add "misaki[ja]" "misaki[zh]"`).

### How a language is chosen for each utterance

`/v1/audio/speech` takes two optional language fields, in priority order:

1. **`language`** — authoritative. Speak exactly this (send it when STT
   already identified the language). Ignored if it isn't in `--langs`.
2. Otherwise **detect from the text**, via lingua restricted to `--langs`.
3. **`language_hint`** — advisory, the conversation's language so far. Used only
   when step 2's confidence is below `HINT_MIN_CONFIDENCE` (0.5), and as the
   fallback when the result gets clamped.

Step 3 exists because a conversation is mostly *short* replies, and one or two
words carry almost no signal: "Absolut." detects as French at 0.29 confidence and
"Sure!" as French at 0.39, so a Swedish robot audibly says one word in French.
Adding more languages to `--langs` makes this more likely, not less. Feed the
hint from your STT's per-turn detection of the **user's own speech**, weighted
by length and recency — far better evidence than the reply fragment itself.

> **The clamp changes pronunciation, not just accent.** Clamping French to `sv`
> runs French text through the *Swedish* g2p, so the words come out genuinely
> mangled. Add a language to `--langs` and it is phonemized properly instead.
>
> **An allowed non-Swedish language brings its own voice.** Only `sv` uses the
> fine-tuned packs; `fr` is spoken by base Kokoro's `ff_siwis`, `it` by
> `if_sara`, and so on. Correct foreign pronunciation and a Swedish timbre are
> mutually exclusive.

To keep a robot strictly Swedish + English, start with `--langs sv,en`.

> `kokoro-svml` defaults to the neural g2p and only checks that the *adapter*
> imports, not the model — the startup log must say `swedish g2p backend=neural`.

## Speaker verification (`/v1/audio/speaker`)

Start with `--speaker ecapa` and the server also returns a **speaker embedding**
for an utterance: an ECAPA-TDNN 192-d unit vector, the audio twin of a face
embedding. Cosine similarity between two vectors tells whether the same person
spoke. A client can use it to tell a bystander's sentence from the focused
visitor's, to keep a conversation attached to a person whose face turned away,
or to recognize a returning voice.

```bash
curl -s localhost:8880/v1/audio/speaker -F file=@utterance.wav
# {"embedding": [0.03, ...], "dim": 192, "seconds": 2.4, "model": "speechbrain/spkrec-ecapa-voxceleb"}
```

Privacy, by construction: the server keeps **no** identities and **no** audio.
It turns a clip into a vector and forgets it; any database of voices belongs to
the client, on its own disk. Runs on CPU, ~20 ms per 3 s clip. The model
(~80 MB) is fetched once from the HuggingFace hub into
`~/.cache/speechbrain/` (override with `SPEAKER_MODEL_DIR`); set
`HF_HUB_OFFLINE=1` afterwards for a guaranteed no-network run. `/health`
reports `"speaker": true` when enabled. Clips shorter than 0.25 s return an
empty embedding; clips under ~2 s give noticeably less reliable matches.

## Verify

```bash
curl -s localhost:8880/health
# {"status":"ok","engine":"KokoroSVMLEngine","default_voice":"Stina","tts":true,"stt":true}

curl -s localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hej! Nu pratar jag svenska.", "voice": "Stina", "language": "sv"}' \
  -o hej.wav
```

`"stt":false` in `/health` means it was started without `--whisper`.

## Point a client at it

```dotenv
TTS_URL=http://<host>:8880/v1/audio/speech
TTS_VOICE=Stina
STT_URL=http://<host>:8880/v1/audio/transcriptions
```

## Run it as a service (Linux)

```ini
# /etc/systemd/system/kokoro-voice-server.service
[Unit]
Description=Kokoro voice server (TTS + Whisper STT)
After=network-online.target

[Service]
Type=simple
User=reachy
WorkingDirectory=/opt/kokoro-voice-server
ExecStart=/usr/bin/uv run python voice_server.py \
    --engine kokoro-svml --voice Stina --host 0.0.0.0 --port 8880 --whisper base
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now kokoro-voice-server
journalctl -u kokoro-voice-server -f
```

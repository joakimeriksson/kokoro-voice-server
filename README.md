# reachy-voice-server (isolated env)

The voice half of the local stack: **Kokoro TTS** on `/v1/audio/speech` and
**faster-whisper STT** on `/v1/audio/transcriptions`, both OpenAI-compatible.
The app is a thin client — it holds no speech models.

## Why a separate project

The Kokoro engines pull in `kokoro`, `torch` and `transformers`, whose
`huggingface-hub` and `pydantic` requirements fight the ones `reachy-mini` needs.
Sharing one venv means every `uv sync` breaks whichever side synced last. So this
folder is its **own uv project**, running the repo's self-contained
`scripts/voice_server.py` (which imports nothing from `reachy_local_assistant`).

## Setup

```bash
cd voice-server
uv sync                      # creates ./.venv with kokoro + torch + faster-whisper
```

## Run

```bash
# Swedish neural Kokoro (10 named voices) + English, with transcription
PYTORCH_ENABLE_MPS_FALLBACK=1 \
SWEDISH_KOKORO_PATH=/abs/path/to/swedish-kokoro \
  uv run python ../scripts/voice_server.py \
      --engine kokoro-svml --voice Stina --host 0.0.0.0 --port 8880 --whisper base
```

| Engine | What it is | Needs |
|--------|------------|-------|
| `kokoro` (default) | Base Kokoro (en/es/fr/it/pt/hi/zh/ja — **no Swedish**) | nothing extra |
| `kokoro-svml` | Fine-tuned Swedish + all base languages, neural NST g2p, named voice packs (Stina, Björn, Nils, …) | `SWEDISH_KOKORO_PATH` for the g2p module; weights auto-download from `--voices-repo` |
| `kokoro-sv` | Older Swedish-only ONNX path, espeak `sv` g2p, single voice | `SWEDISH_KOKORO_PATH` |

`--whisper <tiny|base|small|medium>` enables transcription (default `off`). The
app uses it to store **text** rather than raw audio in the conversation history —
without it conversations still work, but history stays heavy and the transcript
panel shows only a mic glyph for the user's turns.

### Voices, and which languages are allowed (`kokoro-svml`)

Ten fine-tuned **Swedish** voice packs: `Stina` (default), `Alice`, `Ebba`,
`Elsa`, `Greta`, `Anton`, `Björn`, `Lars`, `Nils`, `Oskar`.

`--langs` (default `sv,en,fr,es,it`, or `$KOKORO_SV_LANGS`) is the allow-list of
languages the robot may speak. A detected language outside it is **clamped to
the first entry** — that guard is deliberate: it stops one mis-heard snippet
from sending the robot off into, say, Hindi.

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

1. **`language`** — authoritative. Speak exactly this (the app sends it when STT
   already identified the language). Ignored if it isn't in `--langs`.
2. Otherwise **detect from the text**, via lingua restricted to `--langs`.
3. **`language_hint`** — advisory, the conversation's language so far. Used only
   when step 2's confidence is below `HINT_MIN_CONFIDENCE` (0.5), and as the
   fallback when the result gets clamped.

Step 3 exists because a conversation is mostly *short* replies, and one or two
words carry almost no signal: "Absolut." detects as French at 0.29 confidence and
"Sure!" as French at 0.39, so a Swedish robot audibly says one word in French.
Adding more languages to `--langs` makes this more likely, not less. The app
feeds the hint from Whisper's per-turn detection of the **user's own speech**,
which is far better evidence than Reachy's reply fragment.

The app weights those turns rather than counting them — by **length** (a full
sentence pins the language, a one-word "Ja!" barely moves it) and by **recency**
(what you are speaking now beats what you spoke a minute ago). So one long
sentence in a new language flips the hint immediately, while a stray "Oui." in
the middle of a Swedish conversation does not.

Because the hint only breaks ties, a genuine language switch is still followed
the moment the speaker produces one confident sentence — "Bien sûr." (1.00)
is spoken in French even while the hint says Swedish.

> **The clamp changes pronunciation, not just accent.** Clamping French to `sv`
> runs French text through the *Swedish* g2p, so the words come out genuinely
> mangled (Whisper hears them as Norwegian). Add a language to `--langs` and it
> is phonemized properly instead.
>
> **An allowed non-Swedish language brings its own voice.** Only `sv` uses the
> fine-tuned packs; `fr` is spoken by base Kokoro's `ff_siwis`, `it` by
> `if_sara`, and so on. So French sounds correctly French — but not like Stina.
> Correct foreign pronunciation and a Swedish timbre are mutually exclusive.

To keep the robot strictly Swedish + English, start it with `--langs sv,en`.

> `kokoro-svml` defaults to the neural g2p and only checks that the *adapter*
> imports, not the model — on a machine without the g2p model it logs `neural`
> and then fails at synth time. On espeak-only machines use `kokoro-sv`.
> The startup log must say `swedish g2p backend=neural`.

## Verify

```bash
curl -s localhost:8880/health
# {"status":"ok","engine":"KokoroSVMLEngine","default_voice":"Stina","tts":true,"stt":true}
```

`"stt":false` means it was started without `--whisper`. The app probes this
endpoint at startup and reports both flags in its log and on the settings page.

## Point the app at it

```dotenv
TTS_URL=http://<host>:8880/v1/audio/speech
TTS_VOICE=Stina
# STT_URL defaults to the TTS host's /v1/audio/transcriptions
```

Or set them on the app's settings page ("Ollama & voice server"), which persists
them to the instance `.env` and applies live.

## Run it as a service (Linux)

```ini
# /etc/systemd/system/reachy-voice-server.service
[Unit]
Description=Reachy voice server (Kokoro TTS + Whisper STT)
After=network-online.target

[Service]
Type=simple
User=reachy
WorkingDirectory=/opt/reachy_mini_conversation_app/voice-server
Environment=SWEDISH_KOKORO_PATH=/opt/swedish-kokoro
ExecStart=/usr/bin/uv run python ../scripts/voice_server.py \
    --engine kokoro-svml --voice Stina --host 0.0.0.0 --port 8880 --whisper base
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now reachy-voice-server
journalctl -u reachy-voice-server -f
```

On macOS there is no systemd — use `./scripts/start_stack.sh` from the repo root.

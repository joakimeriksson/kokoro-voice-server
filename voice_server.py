"""OpenAI-compatible voice server (/v1/audio/speech) — Piper or Kokoro.

Lets reachy_local_assistant (``TTS_BACKEND=remote``) use a network voice
generator so the client stays thin. Pick the engine at launch:

    pip install '.[voiceserver]'                 # fastapi + uvicorn (Piper engine)
    # for the kokoro engine also: pip install kokoro

    # Piper (reuses this repo's Piper voices)
    python scripts/voice_server.py --engine piper  --voice en_US-lessac-medium

    # Kokoro
    python scripts/voice_server.py --engine kokoro --voice af_heart

Then point the app at it (in .env):

    TTS_BACKEND=remote
    TTS_URL=http://<host>:8880/v1/audio/speech
    TTS_VOICE=<a voice name for the chosen engine>

Server tool only — lives outside ``src/`` so it never ships in the robot wheel.
"""

import io
import sys
import logging
import argparse
from typing import Any, Tuple
from pathlib import Path


# Make the in-repo package importable (used by the Piper engine).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger("voice_server")


class PiperEngine:
    """Synthesise with this repo's Piper backend (offline, ONNX)."""

    def __init__(self, default_voice: str, data_dir: str) -> None:
        from reachy_local_assistant.audio.piper_tts import PiperTTS

        self._tts = PiperTTS(default_voice, data_dir)
        self._default = default_voice

    def synth(self, text: str, voice: str | None) -> Tuple[int, NDArray[np.int16]]:
        chunks = list(self._tts.synthesize(text, voice=voice or self._default))
        if not chunks:
            return 22050, np.zeros(0, dtype=np.int16)
        sample_rate = chunks[0][0]
        return sample_rate, np.concatenate([c[1] for c in chunks])


class KokoroEngine:
    """Synthesise with Kokoro (hexgrad/Kokoro-82M), 24 kHz float audio."""

    def __init__(self, default_voice: str, lang_code: str) -> None:
        try:
            from kokoro import KPipeline
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Kokoro engine needs: pip install kokoro") from exc
        self._pipeline = KPipeline(lang_code=lang_code)
        self._default = default_voice

    def synth(self, text: str, voice: str | None) -> Tuple[int, NDArray[np.int16]]:
        parts = []
        for _gs, _ps, audio in self._pipeline(text, voice=voice or self._default):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            parts.append(arr.astype(np.float32).reshape(-1))
        if not parts:
            return 24000, np.zeros(0, dtype=np.int16)
        full = np.concatenate(parts)
        pcm = np.clip(full * 32767.0, -32768, 32767).astype(np.int16)
        return 24000, pcm


def _to_audio_bytes(sample_rate: int, pcm: NDArray[np.int16], fmt: str) -> Tuple[bytes, str]:
    import soundfile as sf

    sf_format = {"wav": "WAV", "flac": "FLAC", "ogg": "OGG"}.get(fmt, "WAV")
    media = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}.get(fmt, "audio/wav")
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format=sf_format, subtype="PCM_16" if sf_format == "WAV" else None)
    return buf.getvalue(), media


def build_app(engine: Any) -> Any:
    from fastapi import FastAPI, Response
    from pydantic import BaseModel

    class SpeechRequest(BaseModel):
        """OpenAI /v1/audio/speech request body (extra fields ignored)."""

        input: str
        voice: str | None = None
        model: str = ""
        response_format: str = "wav"

    app = FastAPI(title="reachy_local_assistant voice server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/audio/speech")
    def speech(body: SpeechRequest) -> Response:
        text = (body.input or "").strip()
        if not text:
            return Response(content=b"", media_type="audio/wav")
        sample_rate, pcm = engine.synth(text, body.voice)
        audio, media = _to_audio_bytes(sample_rate, pcm, body.response_format)
        return Response(content=audio, media_type=media)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="OpenAI-compatible Piper/Kokoro voice server.")
    p.add_argument("--engine", choices=["piper", "kokoro"], default="piper")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8880)
    p.add_argument("--voice", default=None, help="default voice (engine-specific)")
    p.add_argument("--voice-dir", default="piper_voices", help="Piper voices dir")
    p.add_argument("--lang", default="a", help="Kokoro lang_code (a=US English, b=UK, ...)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if args.engine == "piper":
        engine: Any = PiperEngine(args.voice or "en_US-lessac-medium", args.voice_dir)
    else:
        engine = KokoroEngine(args.voice or "af_heart", args.lang)
    logger.info("Voice server: engine=%s on %s:%d", args.engine, args.host, args.port)

    import uvicorn

    uvicorn.run(build_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

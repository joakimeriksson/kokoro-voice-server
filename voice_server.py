"""OpenAI-compatible voice server (/v1/audio/speech) — Piper or Kokoro.

Lets reachy_local_assistant (``TTS_BACKEND=remote``) use a network voice
generator so the client stays thin. Pick the engine at launch:

    pip install '.[voiceserver]'                 # fastapi + uvicorn (Piper engine)
    # for the kokoro engine also: pip install kokoro

    # Piper (reuses this repo's Piper voices)
    python scripts/voice_server.py --engine piper  --voice en_US-lessac-medium

    # Kokoro (English + its other supported languages, NOT Swedish)
    python scripts/voice_server.py --engine kokoro --voice af_heart

    # Swedish Kokoro — fine-tuned voice from the sibling swedish-kokoro project
    # (torch-free ONNX model + espeak 'sv' g2p). Needs that project on disk plus:
    #   pip install onnxruntime misaki phonemizer-fork espeakng_loader torch
    SWEDISH_KOKORO_PATH=../ai-smarthome/swedish-kokoro \
      python scripts/voice_server.py --engine kokoro-sv

Then point the app at it (in .env):

    TTS_BACKEND=remote
    TTS_URL=http://<host>:8880/v1/audio/speech
    TTS_VOICE=<a voice name for the chosen engine>

Server tool only — lives outside ``src/`` so it never ships in the robot wheel.
"""

import io
import os
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


class SwedishKokoroEngine:
    """Fine-tuned Swedish Kokoro via ONNX Runtime (torch-free model path).

    Loads the model/voicepack/config + espeak 'sv' g2p from a sibling
    ``swedish-kokoro`` project (see github.com/.../ai-smarthome). Point at it with
    ``--svml-path`` or ``$SWEDISH_KOKORO_PATH``. Needs (in this env): onnxruntime,
    misaki, phonemizer-fork, espeakng_loader, and torch (only to load the .pt
    voicepack). The model itself runs in onnxruntime.
    """

    def __init__(self, svml_path: str) -> None:
        import json

        import onnxruntime as ort

        path = Path(svml_path).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"--svml-path not found: {path}")
        sys.path.insert(0, str(path))
        try:
            from sv_weights import resolve  # local deploy/ or HF auto-download
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"swedish-kokoro not importable from {path}: {exc}")

        self._vocab = json.load(open(resolve("config.json")))["vocab"]
        self._voice = self._load_voicepack(resolve("sv_female.pt"))
        self._session = ort.InferenceSession(resolve("kokoro_sv.onnx"), providers=["CPUExecutionProvider"])
        from misaki import espeak

        self._g2p = espeak.EspeakG2P(language="sv")

    @staticmethod
    def _load_voicepack(path: str) -> NDArray[np.float32]:
        import torch

        return torch.load(path, map_location="cpu", weights_only=True).numpy()

    @staticmethod
    def _trim_eos_tail(audio: NDArray[np.float32], pred_dur: Any, sr: int = 24000, fade_ms: int = 12) -> NDArray[np.float32]:
        pd = np.asarray(pred_dur).astype(np.int64)
        total = int(pd.sum())
        if total <= 0:
            return audio
        spf = len(audio) / total
        keep = max(1, min(int(round((total - int(pd[-1])) * spf)), len(audio)))
        audio = audio[:keep].copy()
        fade = int(fade_ms / 1000 * sr)
        if 0 < fade < len(audio):
            audio[-fade:] *= np.linspace(1.0, 0.0, fade).astype(audio.dtype)
        return audio

    def synth(self, text: str, voice: str | None) -> Tuple[int, NDArray[np.int16]]:
        phonemes, _ = self._g2p(text)
        ipa = phonemes.replace("ʏ", "y")
        ids = [j for j in (self._vocab.get(p) for p in ipa) if j is not None]
        if not ids:
            return 24000, np.zeros(0, dtype=np.int16)
        input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
        ref_s = self._voice[len(ids) - 1]
        audio, pred_dur = self._session.run(None, {"input_ids": input_ids, "ref_s": ref_s})
        audio = self._trim_eos_tail(np.asarray(audio).reshape(-1).astype(np.float32), pred_dur)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        return 24000, pcm


class KokoroSVMLEngine:
    """Torch KokoroSVML: Swedish + the other Kokoro languages in one model.

    Wraps the swedish-kokoro project's ``KokoroSVML``. Its ``SwedishG2P``
    auto-uses the **neural NST g2p** when that model is importable
    (``SV_NEURAL_G2P`` / ``SV_G2P_DIR``) and falls back to espeak otherwise — the
    active backend is logged at startup. Needs (in this env): kokoro, torch, and
    the swedish-kokoro deps (misaki, phonemizer-fork, espeakng_loader).
    """

    def __init__(self, svml_path: str, lang: str = "sv", device: str | None = None) -> None:
        path = Path(svml_path).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"--svml-path not found: {path}")
        sys.path.insert(0, str(path))
        try:
            from kokoro_svml import KokoroSVML
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"KokoroSVML not importable from {path}: {exc}")
        import torch

        if device is None:  # prefer GPU: cuda (3090) > mps (Apple Silicon) > cpu
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # a few TTS ops lack MPS kernels
        self._tts = KokoroSVML(device=device)
        self._lang = lang
        backend = getattr(getattr(self._tts, "g2p_sv", None), "backend", "?")
        logger.info("KokoroSVML ready: lang=%s, device=%s, swedish g2p backend=%s", lang, device, backend)

    def synth(self, text: str, voice: str | None) -> Tuple[int, NDArray[np.int16]]:
        audio = np.asarray(self._tts.generate(text, lang=self._lang, voice=voice or None), dtype=np.float32).reshape(-1)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
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

    from reachy_local_assistant.audio.text_chunk import split_sentences

    @app.post("/v1/audio/speech")
    def speech(body: SpeechRequest) -> Response:
        text = (body.input or "").strip()
        if not text:
            return Response(content=b"", media_type="audio/wav")
        # Chunk long text by sentence: Kokoro caps at ~510 tokens/utterance, and it
        # keeps any single engine call small. Concatenate the PCM back into one WAV.
        sample_rate = 24000
        parts = []
        for chunk in split_sentences(text) or [text]:
            sample_rate, pcm = engine.synth(chunk, body.voice)
            if len(pcm):
                parts.append(pcm)
        pcm = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
        audio, media = _to_audio_bytes(sample_rate, pcm, body.response_format)
        return Response(content=audio, media_type=media)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="OpenAI-compatible Piper/Kokoro voice server.")
    p.add_argument("--engine", choices=["piper", "kokoro", "kokoro-sv", "kokoro-svml"], default="piper")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8880)
    p.add_argument("--voice", default=None, help="default voice (engine-specific)")
    p.add_argument("--voice-dir", default="piper_voices", help="Piper voices dir")
    p.add_argument("--lang", default="a", help="Kokoro lang_code (a=US English, b=UK, ...)")
    p.add_argument(
        "--svml-path",
        default=os.environ.get("SWEDISH_KOKORO_PATH", "../ai-smarthome/swedish-kokoro"),
        help="path to the swedish-kokoro project (for --engine kokoro-sv)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if args.engine == "piper":
        engine: Any = PiperEngine(args.voice or "en_US-lessac-medium", args.voice_dir)
    elif args.engine == "kokoro-sv":
        engine = SwedishKokoroEngine(args.svml_path)
    elif args.engine == "kokoro-svml":
        engine = KokoroSVMLEngine(args.svml_path, lang="sv" if args.lang == "a" else args.lang)
    else:
        engine = KokoroEngine(args.voice or "af_heart", args.lang)
    logger.info("Voice server: engine=%s on %s:%d", args.engine, args.host, args.port)

    import uvicorn

    uvicorn.run(build_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

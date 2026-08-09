"""OpenAI-compatible voice server: Kokoro TTS + Whisper STT.

Serves ``/v1/audio/speech`` (TTS) and ``/v1/audio/transcriptions`` (Whisper), so
reachy_local_assistant stays thin — the models live here. Run it from its own
environment (``cd voice-server && uv sync``); see voice-server/README.md.

Pick the engine at launch:

    # Kokoro (English + its other supported languages, NOT Swedish)
    python scripts/voice_server.py --engine kokoro --voice af_heart

    # Multilingual: the finished fine-tuned Swedish + base Kokoro languages, one
    # model. Named Swedish voice packs (Stina, Björn, Nils, …) and the neural
    # Swedish g2p; auto-routes on the caller's / STT-detected language. Pulls the
    # weights + voices from HF (--voices-repo, default Joakim/kokoro-sv-voices);
    # needs the swedish-kokoro repo on disk for its g2p module, plus:
    #   pip install kokoro torch scipy misaki phonemizer-fork espeakng_loader
    SWEDISH_KOKORO_PATH=../ai-smarthome/swedish-kokoro \
      python scripts/voice_server.py --engine kokoro-svml --voice Stina

    # Swedish-only ONNX path (older single voice, espeak 'sv' g2p):
    SWEDISH_KOKORO_PATH=../ai-smarthome/swedish-kokoro \
      python scripts/voice_server.py --engine kokoro-sv

Add ``--whisper base`` (or tiny/small/medium) to enable transcription, which the
app uses to store text rather than raw audio in the conversation history.

Then point the app at it (in .env):

    TTS_URL=http://<host>:8880/v1/audio/speech
    TTS_VOICE=<a voice name for the chosen engine, e.g. Stina for kokoro-svml>
    STT_URL=http://<host>:8880/v1/audio/transcriptions   # defaults to the TTS host

Server tool only — lives outside ``src/`` so it never ships in the robot wheel.
"""

import io
import os
import sys
import logging
import argparse
from typing import Any, Tuple
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger("voice_server")

# Cheap Swedish-vs-English guess so one KokoroSVML server can answer in whichever
# language the user spoke (the model is multilingual). Defaults to English.
import re as _re  # noqa: E402


_SV_CHARS = set("åäöÅÄÖ")
_SV_WORDS = {
    "och", "är", "jag", "att", "det", "en", "ett", "vi", "du", "han", "hon",
    "inte", "med", "för", "på", "som", "den", "har", "kan", "ska", "hej", "tack",
}


# Kokoro's languages, by lingua Language name -> Kokoro lang code.
_LINGUA_TO_KOKORO = {
    "SWEDISH": "sv", "ENGLISH": "en", "SPANISH": "es", "FRENCH": "fr",
    "ITALIAN": "it", "PORTUGUESE": "pt", "HINDI": "hi", "CHINESE": "zh", "JAPANESE": "ja",
}
_lingua: Any = None  # lazily-built lingua detector (False if the dep is unavailable)


# Below this detector confidence, one utterance is not enough to decide: prefer the
# caller's conversation-history hint. Measured on short replies — genuine misses
# ("Absolut." -> fr at 0.29, "Sure!" -> fr at 0.39) all sit under 0.40 with a ~0.03
# margin, while correct calls on a real sentence are >= 0.53.
HINT_MIN_CONFIDENCE = 0.5


def _detect_lang(text: str) -> Tuple[str, float]:
    """Detect *text*'s language among Kokoro's set; return ``(lang, confidence)``.

    Uses lingua, restricted to Kokoro's languages for accuracy (~0.03 ms/call),
    falling back to a cheap sv/en heuristic when it isn't installed. The confidence
    matters as much as the guess: a short reply like "Ja!" or "Sure!" carries almost
    no signal, so the caller uses it to decide whether to trust the conversation's
    language instead (see ``HINT_MIN_CONFIDENCE``).
    """
    global _lingua
    if _lingua is None:
        try:
            from lingua import Language, LanguageDetectorBuilder

            langs = [getattr(Language, name) for name in _LINGUA_TO_KOKORO]
            _lingua = LanguageDetectorBuilder.from_languages(*langs).build()
        except Exception:
            _lingua = False
    if _lingua:
        ranked = _lingua.compute_language_confidence_values(text)
        if ranked:
            best = ranked[0]
            return _LINGUA_TO_KOKORO.get(best.language.name, "en"), float(best.value)
    # Fallback: sv-vs-en heuristic. Swedish glyphs are near-conclusive; a word-list
    # hit is weak enough that the hint should win.
    if any(c in _SV_CHARS for c in text):
        return "sv", 0.9
    words = set(_re.findall(r"[a-zåäö]+", text.lower()))
    return ("sv", 0.4) if words & _SV_WORDS else ("en", 0.4)


# Languages KokoroSVML can speak; an unrecognised caller label falls back to detect.
_SVML_LANGS = {"sv", "en", "en-us", "en-gb", "es", "fr", "hi", "it", "pt", "zh", "ja"}


def _norm_lang(s: str | None) -> str:
    """Normalise an STT/caller language label to a KokoroSVML code (best effort)."""
    s = (s or "").strip().lower()
    return {"swedish": "sv", "svenska": "sv", "se": "sv", "english": "en", "eng": "en"}.get(s, s)


_SENT_BOUNDARY = _re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, max_chars: int = 300) -> list:
    """Split *text* into sentence chunks, hard-wrapping any over *max_chars*.

    Inlined (not imported from the app) so this server runs in its own environment
    without the reachy_local_assistant package. Keeps each chunk within Kokoro's
    ~510-token per-utterance cap and enables low-latency per-sentence streaming.
    """
    text = (text or "").strip()
    if not text:
        return []
    out = []
    for piece in _SENT_BOUNDARY.split(text):
        piece = piece.strip()
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            head = piece[:cut].strip()
            if head:
                out.append(head)
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out


class KokoroEngine:
    """Synthesise with Kokoro (hexgrad/Kokoro-82M), 24 kHz float audio."""

    def __init__(self, default_voice: str, lang_code: str) -> None:
        try:
            from kokoro import KPipeline
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Kokoro engine needs: pip install kokoro") from exc
        self._pipeline = KPipeline(lang_code=lang_code)
        self._default = default_voice
        self.default_voice = default_voice  # reported by /health

    def _resolve_voice(self, voice: str | None) -> str:
        """Return a usable base-Kokoro voice, falling back to the default.

        Base Kokoro voices are all ``<lang><gender>_<name>`` (af_heart, bm_george).
        Anything else — most often a Swedish pack name like "Stina" left over from
        a kokoro-svml config — is not downloadable here and would make every
        request 500, i.e. a robot that hears you and then says nothing. Warn and
        speak in the default voice instead; wrong voice beats no voice.
        """
        if voice and _re.match(r"^[a-z][fm]_", voice):
            return voice
        if voice:
            logger.warning(
                "voice %r is not a base-Kokoro voice (expected e.g. af_heart); using %r. "
                "Swedish voice packs need --engine kokoro-svml.",
                voice, self._default,
            )
        return self._default

    def synth(
        self, text: str, voice: str | None, language: str | None = None, language_hint: str | None = None
    ) -> Tuple[int, NDArray[np.int16]]:
        parts = []
        for _gs, _ps, audio in self._pipeline(text, voice=self._resolve_voice(voice)):
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

    def synth(
        self, text: str, voice: str | None, language: str | None = None, language_hint: str | None = None
    ) -> Tuple[int, NDArray[np.int16]]:
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
    """Multilingual Kokoro: the finished fine-tuned Swedish + base Kokoro langs.

    Self-contained port of the swedish-kokoro ``examples/speak.py`` recipe — it
    does NOT wrap the sibling repo's ``KokoroSVML``; only its ``g2p_sv`` /
    ``nst_g2p`` modules are imported (the neural Swedish grapheme->phoneme). The
    fine-tuned weights and the **named voice packs** (Stina, Björn, Nils, …) are
    pulled from the HF voices repo (``--voices-repo`` / ``$KOKORO_SV_VOICES``).

    Swedish path: neural g2p -> KModel.forward_with_tokens -> four upsampler-tone
    notch filters (2400/4800/7200/9600 Hz). Because Kokoro's KModel is
    language-blind (IPA -> audio), the SAME weights serve the other languages via
    a base ``KPipeline`` with a per-language default voice.

    Needs (in this env): kokoro, torch, scipy, huggingface_hub, and the
    swedish-kokoro repo on ``--svml-path`` (for its ``g2p_sv`` module).
    """

    # lang -> (KPipeline lang_code, default base voice). Swedish handled separately.
    _LANGS = {
        "en": ("a", "af_heart"), "en-us": ("a", "af_heart"), "en-gb": ("b", "bf_emma"),
        "es": ("e", "ef_dora"), "fr": ("f", "ff_siwis"), "hi": ("h", "hf_alpha"),
        "it": ("i", "if_sara"), "pt": ("p", "pf_dora"), "zh": ("z", "zf_xiaobei"),
        "ja": ("j", "jf_alpha"),
    }
    _NOTCH_HZ = (2400, 4800, 7200, 9600)  # remove the fine-tune upsampler tones (lossless)

    def __init__(
        self,
        svml_path: str,
        voices_repo: str,
        default_sv_voice: str = "Stina",
        lang: str = "sv",
        allowed_langs: str = "sv,en",
        device: str | None = None,
    ) -> None:
        path = Path(svml_path).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"--svml-path not found: {path}")
        sys.path.insert(0, str(path))  # for `import g2p_sv`
        os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")  # prefer the neural Swedish g2p
        import torch
        from kokoro import KModel
        from huggingface_hub import hf_hub_download

        try:
            from g2p_sv import SwedishG2P
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"g2p_sv not importable from {path}: {exc}")

        if device is None:  # prefer GPU: cuda (3090) > mps (Apple Silicon) > cpu
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # a few TTS ops lack MPS kernels

        self._torch = torch
        self._device = device
        self._voices_repo = voices_repo
        self._default_sv_voice = default_sv_voice
        self.default_voice = default_sv_voice  # reported by /health
        self._lang = lang
        # Allow-list: the robot only ever speaks these (default sv,en). Any stray
        # STT-detected / auto-detected language outside it is clamped to _primary,
        # so a mis-heard snippet can never make it wander into Hindi/Chinese/etc.
        self._allowed = {_norm_lang(x) for x in allowed_langs.split(",") if x.strip()} or {"sv"}
        self._primary = next((_norm_lang(x) for x in allowed_langs.split(",") if x.strip()), "sv")
        cfg = hf_hub_download(voices_repo, "config.json")
        wts = hf_hub_download(voices_repo, "kokoro_sv.pth")
        self._model = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg, model=wts).to(device).eval()
        self._g2p = SwedishG2P(backend="neural")
        self._sv_voices: dict[str, Any] = {}  # name -> voicepack tensor (lazy, cached)
        self._pipes: dict[str, Any] = {}  # KPipeline lang_code -> pipeline (lazy, cached)
        backend = getattr(self._g2p, "backend", "?")
        logger.info(
            "KokoroSVML ready: device=%s, swedish g2p=%s, voices=%s, default sv voice=%s, langs=%s (primary %s)",
            device, backend, voices_repo, default_sv_voice, sorted(self._allowed), self._primary,
        )

    def _sv_voicepack(self, name: str) -> Any:
        """Lazily fetch+cache the ``voices/<name>.pt`` pack from the voices repo."""
        if name not in self._sv_voices:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(self._voices_repo, f"voices/{name}.pt")
            self._sv_voices[name] = self._torch.load(p, map_location=self._device, weights_only=True)
        return self._sv_voices[name]

    def _pipe(self, code: str) -> Any:
        """Lazily build a base-Kokoro ``KPipeline`` for a non-Swedish lang code."""
        if code not in self._pipes:
            from kokoro import KPipeline

            self._pipes[code] = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M", model=self._model)
        return self._pipes[code]

    def _notch(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        from scipy.signal import filtfilt, iirnotch

        for f0 in self._NOTCH_HZ:
            b, a = iirnotch(f0, Q=35, fs=24000)
            audio = filtfilt(b, a, audio)
        return audio.astype(np.float32)

    @staticmethod
    def _trim_silence(
        audio: NDArray[np.float32], sr: int = 24000, pad_ms: int = 40, hop_ms: int = 10
    ) -> NDArray[np.float32]:
        """Strip leading/trailing near-silence, keeping a short pad + edge fades.

        The model pads each utterance with long BOS/EOS pauses (a bare "Hej!" is
        ~1.1 s of leading + ~0.9 s of trailing silence around 0.4 s of speech).
        The server synthesises one utterance per sentence and concatenates them,
        so those pauses would pile up into big gaps. Trim to the voiced region.
        """
        if audio.size == 0:
            return audio
        peak = float(np.abs(audio).max())
        if peak <= 0.0:
            return audio
        hop = max(1, int(hop_ms / 1000 * sr))
        frames = audio[: len(audio) // hop * hop].reshape(-1, hop)
        env = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
        thr = max(0.02 * peak, 1e-4)  # 2% of peak, with an absolute floor
        voiced = np.where(env > thr)[0]
        if voiced.size == 0:
            return audio
        pad = int(pad_ms / 1000 * sr)
        start = max(0, voiced[0] * hop - pad)
        end = min(len(audio), (voiced[-1] + 1) * hop + pad)
        out = audio[start:end].copy()
        fade = int(0.008 * sr)  # 8 ms in/out fade to avoid clicks at the cut
        if 0 < fade < len(out):
            out[:fade] *= np.linspace(0.0, 1.0, fade).astype(out.dtype)
            out[-fade:] *= np.linspace(1.0, 0.0, fade).astype(out.dtype)
        return out

    def _synth_sv(self, text: str, voice: str | None) -> NDArray[np.float32]:
        # A base-Kokoro-style name (af_heart) isn't a Swedish pack -> use the default.
        name = voice if (voice and not _re.match(r"^[a-z][fm]_", voice)) else self._default_sv_voice
        try:
            vp = self._sv_voicepack(name)
        except Exception:
            logger.warning("swedish voice %r not found in %s; using %s", name, self._voices_repo, self._default_sv_voice)
            vp = self._sv_voicepack(self._default_sv_voice)
        ipa = self._g2p(text).replace("ʏ", "y")
        ids = [i for i in (self._model.vocab.get(p) for p in ipa) if i is not None]
        if not ids:
            return np.zeros(0, dtype=np.float32)
        torch = self._torch
        with torch.no_grad():
            audio_t, _pred_dur = self._model.forward_with_tokens(
                torch.LongTensor([[0, *ids, 0]]).to(self._device),
                vp[len(ids) - 1].to(self._device),
                speed=1.0,
            )
        audio = self._notch(audio_t.squeeze().cpu().numpy().astype(np.float32))
        return self._trim_silence(audio)

    def _synth_other(self, text: str, lang: str, voice: str | None) -> NDArray[np.float32]:
        code, default_voice = self._LANGS.get(lang, self._LANGS["en"])
        # Only forward a real base-Kokoro voice (af_heart); ignore Swedish/Piper/OpenAI
        # names so each language uses its own default voice.
        kvoice = voice if (voice and _re.match(r"^[a-z][fm]_", voice)) else default_voice
        chunks = [
            a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
            for _gs, _ps, a in self._pipe(code)(text, voice=kvoice)
        ]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)

    def synth(
        self, text: str, voice: str | None, language: str | None = None, language_hint: str | None = None
    ) -> Tuple[int, NDArray[np.int16]]:
        hint = _norm_lang(language_hint)
        hint = hint if hint in self._allowed else ""
        # An explicit, allowed language from the caller (e.g. the STT-detected
        # language) wins; otherwise auto-detect (--lang auto) or use the fixed lang.
        if language and _norm_lang(language) in self._allowed:
            lang = _norm_lang(language)
        elif self._lang == "auto":
            lang, confidence = _detect_lang(text)
            # One short utterance ("Ja!", "Absolut.", "Sure!") carries almost no
            # signal, and with several languages allowed it lands on the wrong one
            # often enough to be audible. When the detector is unsure, the
            # conversation so far is the better evidence than this fragment.
            if hint and confidence < HINT_MIN_CONFIDENCE and hint != lang:
                logger.info(
                    "lang %s (%.2f) below %.2f for %r — using conversation hint %s",
                    lang, confidence, HINT_MIN_CONFIDENCE, text[:40], hint,
                )
                lang = hint
        else:
            lang = self._lang
        # Clamp to the allow-list: a mis-heard snippet must never make the robot
        # speak a language it isn't supposed to (this is what caused the Hindi).
        # The conversation's own language beats the static primary here too.
        if lang not in self._allowed:
            lang = hint or self._primary
        if lang in ("sv", "swedish", "se"):
            audio = self._synth_sv(text, voice)
        else:
            audio = self._synth_other(text, lang, voice)
        pcm = np.clip(audio.reshape(-1) * 32767.0, -32768, 32767).astype(np.int16)
        return 24000, pcm


def _to_audio_bytes(sample_rate: int, pcm: NDArray[np.int16], fmt: str) -> Tuple[bytes, str]:
    import soundfile as sf

    sf_format = {"wav": "WAV", "flac": "FLAC", "ogg": "OGG"}.get(fmt, "WAV")
    media = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}.get(fmt, "audio/wav")
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format=sf_format, subtype="PCM_16" if sf_format == "WAV" else None)
    return buf.getvalue(), media


class WhisperSTT:
    """faster-whisper transcription (CPU int8) for /v1/audio/transcriptions.

    Lets the app keep a TEXT conversation history without a second gemma call: in
    direct-audio mode the app fires this AFTER the LLM has replied (off the reply's
    critical path), and stores the returned transcript instead of the raw audio.
    Small multilingual Whisper (base) auto-detects the language; runs on CPU so it
    never contends with Ollama on the GPU.
    """

    def __init__(self, model_size: str = "base", initial_prompt: str | None = "Reachy Mini, a small friendly robot.") -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._initial_prompt = initial_prompt
        logger.info("Whisper STT ready: %s (cpu/int8)", model_size)

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> Tuple[str, str]:
        """Return (text, detected_language). faster-whisper decodes+resamples internally.

        ``vad_filter`` strips non-speech before decoding. Without it, Whisper
        *hallucinates* on noise-only audio — a chair scrape comes back as a
        plausible sentence — which breaks the app's noise gate (it drops turns
        whose transcript is empty, so noise must actually produce empty text).
        """
        segments, info = self._model.transcribe(
            io.BytesIO(audio_bytes),
            beam_size=1,  # greedy = fastest
            language=language or None,
            initial_prompt=self._initial_prompt,
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language


def build_app(engine: Any, stt: Any = None) -> Any:
    from fastapi import File, Form, FastAPI, Response, UploadFile
    from pydantic import BaseModel
    from fastapi.responses import JSONResponse

    class SpeechRequest(BaseModel):
        """OpenAI /v1/audio/speech request body (extra fields ignored)."""

        input: str
        voice: str | None = None
        # Authoritative: speak exactly this language (e.g. the STT-detected one).
        language: str | None = None
        # Advisory: the conversation's language so far. Used only to break ties
        # when detecting this utterance alone is unreliable — never overrides
        # `language`, nor a confident detection (so switching language mid-chat
        # still works as soon as the user says a full sentence in it).
        language_hint: str | None = None
        model: str = ""
        response_format: str = "wav"

    app = FastAPI(title="reachy_local_assistant voice server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Report what this server can actually do, not just that it is up.

        The app probes this at startup. "Reachable but transcription disabled"
        is otherwise a silent failure: the conversation works while the text
        history quietly keeps raw audio instead of transcripts.
        """
        return {
            "status": "ok",
            "engine": type(engine).__name__,
            "default_voice": getattr(engine, "default_voice", "") or "",
            "tts": True,
            "stt": stt is not None,
        }

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
            sample_rate, pcm = engine.synth(chunk, body.voice, body.language, body.language_hint)
            if len(pcm):
                parts.append(pcm)
        pcm = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
        audio, media = _to_audio_bytes(sample_rate, pcm, body.response_format)
        return Response(content=audio, media_type=media)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(""),
        language: str | None = Form(None),
    ) -> Any:
        """OpenAI-compatible Whisper transcription: multipart file -> {text, language}."""
        if stt is None:
            return JSONResponse({"error": "transcription not enabled (start with --whisper)"}, status_code=501)
        data = await file.read()
        if not data:
            return {"text": "", "language": language or ""}
        text, lang = stt.transcribe(data, language=language)
        return {"text": text, "language": lang}

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="OpenAI-compatible Kokoro voice server (TTS + Whisper STT).")
    p.add_argument("--engine", choices=["kokoro", "kokoro-sv", "kokoro-svml"], default="kokoro")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8880)
    p.add_argument("--voice", default=None, help="default voice (engine-specific)")
    p.add_argument("--lang", default="a", help="Kokoro lang_code (a=US English, b=UK, ...)")
    p.add_argument(
        "--svml-path",
        default=os.environ.get("SWEDISH_KOKORO_PATH", "../ai-smarthome/swedish-kokoro"),
        help="path to the swedish-kokoro project (for --engine kokoro-sv / kokoro-svml g2p)",
    )
    p.add_argument(
        "--voices-repo",
        default=os.environ.get("KOKORO_SV_VOICES", "Joakim/kokoro-sv-voices"),
        help="HF repo with the finished Swedish model + named voice packs (--engine kokoro-svml)",
    )
    p.add_argument(
        "--langs",
        default=os.environ.get("KOKORO_SV_LANGS", "sv,en,fr,es,it"),
        help="kokoro-svml allow-list of spoken languages (first = primary/fallback); "
        "any detected language outside it is clamped so the robot never wanders off. "
        "An allowed non-Swedish language is spoken by base Kokoro with ITS OWN voice "
        "(fr -> ff_siwis), not the Swedish pack — only 'sv' uses the fine-tuned voices",
    )
    p.add_argument(
        "--whisper",
        default=os.environ.get("WHISPER_MODEL", "off"),
        help="faster-whisper size for /v1/audio/transcriptions (tiny/base/small/medium), or 'off'",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if args.engine == "kokoro-sv":
        engine: Any = SwedishKokoroEngine(args.svml_path)
    elif args.engine == "kokoro-svml":
        engine = KokoroSVMLEngine(
            args.svml_path,
            voices_repo=args.voices_repo,
            default_sv_voice=args.voice or "Stina",
            lang="auto" if args.lang == "a" else args.lang,
            allowed_langs=args.langs,
        )
    else:
        engine = KokoroEngine(args.voice or "af_heart", args.lang)

    stt = None
    if args.whisper and args.whisper.lower() != "off":
        stt = WhisperSTT(args.whisper)
    logger.info(
        "Voice server: engine=%s whisper=%s on %s:%d", args.engine, args.whisper, args.host, args.port
    )

    import uvicorn

    uvicorn.run(build_app(engine, stt), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

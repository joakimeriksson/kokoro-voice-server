"""OpenAI-compatible voice server: Kokoro TTS + Whisper STT.

Serves ``/v1/audio/speech`` (TTS) and ``/v1/audio/transcriptions`` (Whisper) so
client apps stay thin — the models live here. Run it from this repo's own
environment (``uv sync``); see README.md.

Pick the engine at launch:

    # Kokoro (English + its other supported languages, NOT Swedish)
    uv run python voice_server.py --engine kokoro --voice af_heart

    # Multilingual: the fine-tuned Swedish + base Kokoro languages, one model.
    # Named Swedish voice packs (Stina, Björn, Nils, …) and the neural Swedish
    # g2p (from the kokoro-sv package); auto-routes on the caller's /
    # STT-detected language. Pulls the weights + voices from HF
    # (--voices-repo, default Joakim/kokoro-sv-voices).
    uv run python voice_server.py --engine kokoro-svml --voice Stina

    # Swedish-only ONNX path (older single voice, espeak 'sv' g2p); this one
    # still needs a swedish-kokoro checkout on disk:
    SWEDISH_KOKORO_PATH=../swedish-kokoro \
      uv run python voice_server.py --engine kokoro-sv

Add ``--whisper base`` (or tiny/small/medium) to enable transcription, which
clients can use to store text rather than raw audio in conversation history.

Then point the client app at it:

    TTS_URL=http://<host>:8880/v1/audio/speech
    TTS_VOICE=<a voice name for the chosen engine, e.g. Stina for kokoro-svml>
    STT_URL=http://<host>:8880/v1/audio/transcriptions   # defaults to the TTS host
"""

import io
import json
import os
import sys
import logging
import argparse
import time
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


# Inline tags, ElevenLabs-style: "[Viktor] Hej! [paus 0.5] [viskar] Kom hit."
# A tag changes the state for the text after it. A styled line goes to VoxCPM2
# when the voice has a --vox-voice reference (see VoxEngine); otherwise Kokoro
# uses the pack "<Voice>_<style>" if one exists, else the plain pack. English
# aliases map to the Swedish names; any other tag is a free description for
# VoxCPM2 ("[sarcastic]") and is never read aloud.
_STYLE_TAGS = {
    "viskar": "viskar", "viska": "viskar", "whisper": "viskar", "whispers": "viskar",
    "glad": "glad", "excited": "glad", "happy": "glad",
    "skrattar": "skrattar", "skratt": "skrattar", "laughs": "skrattar", "laughing": "skrattar",
    "ledsen": "ledsen", "sad": "ledsen",
    "arg": "arg", "angry": "arg",
    "tänker": "tänker", "thinking": "tänker",
}
_SPEED_TAGS = {"långsamt": 0.85, "slow": 0.85, "snabbt": 1.2, "fast": 1.2}
_RESET_TAGS = {"normal", "neutral", "normalt"}
_TAG = _re.compile(r"\[([^\]\[]{1,80})\](?!\()|<break\s+time=\"([\d.]+)(m?s)\"\s*/?>", _re.I)
_PAUSE = _re.compile(r"(?:paus|pause)(?:\s+([\d.]+))?$", _re.I)
_TEMPO = _re.compile(r"tempo\s+([\d.]+)$", _re.I)
_VOICE = _re.compile(r"^[A-ZÅÄÖ][a-zåäö]+$")


def parse_tags(text: str) -> list:
    """Split tagged text into ("say", text, voice, style, speed) and ("pause", s).

    voice None = the request's voice; speed multiplies the request's speed.
    An unknown tag becomes a free-text style (lowercased) for VoxCPM2; no tag
    is ever read aloud. A pronunciation link "[ord](/IPA/)" is not a tag; the
    Swedish synthesizer handles it.
    """
    out, voice, style, speed, pos = [], None, None, 1.0, 0

    def say(chunk: str) -> None:
        if not chunk.strip():
            return
        if out and out[-1][0] == "say" and out[-1][2:] == (voice, style, speed):
            out[-1] = ("say", f"{out[-1][1]} {chunk.strip()}", voice, style, speed)
        else:
            out.append(("say", chunk.strip(), voice, style, speed))

    for m in _TAG.finditer(text):
        say(text[pos:m.start()])
        pos = m.end()
        if m.group(2):  # <break time="1.5s"/>
            out.append(("pause", float(m.group(2)) / (1000.0 if m.group(3).lower() == "ms" else 1.0)))
            continue
        tag = m.group(1).strip()
        low = tag.lower()
        if (p := _PAUSE.match(low)):
            out.append(("pause", min(5.0, float(p.group(1) or 0.5))))
        elif low in _STYLE_TAGS:
            style = _STYLE_TAGS[low]
        elif low in _SPEED_TAGS:
            speed = _SPEED_TAGS[low]
        elif (t := _TEMPO.match(low)):
            speed = max(0.5, min(2.0, float(t.group(1))))
        elif low in _RESET_TAGS:
            style, speed = None, 1.0
        elif _VOICE.match(tag):
            voice, style = tag, None
        else:
            style = low
    say(text[pos:])
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
        self.kmodel = self._pipeline.model  # for warm_shapes()
        self.device = str(next(self.kmodel.parameters()).device) if self.kmodel is not None else "cpu"
        if self.device.startswith("mps"):
            install_hybrid_forward(self.kmodel, self.device)

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
        self, text: str, voice: str | None, language: str | None = None,
        language_hint: str | None = None, speed: float = 1.0
    ) -> Tuple[int, NDArray[np.int16]]:
        parts = []
        for _gs, _ps, audio in self._pipeline(text, voice=self._resolve_voice(voice), speed=speed):
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
        self, text: str, voice: str | None, language: str | None = None,
        language_hint: str | None = None, speed: float = 1.0
    ) -> Tuple[int, NDArray[np.int16]]:
        phonemes, _ = self._g2p(text)
        ipa = phonemes.replace("ʏ", "y")
        ids = [j for j in (self._vocab.get(p) for p in ipa) if j is not None]
        if not ids:
            return 24000, np.zeros(0, dtype=np.int16)
        input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
        ref_s = self._voice[len(ids) - 1]
        audio, pred_dur = self._session.run(None, {"input_ids": input_ids, "ref_s": ref_s, "speed": np.array([speed], dtype=np.float32)} if "speed" in {i.name for i in self._session.get_inputs()} else {"input_ids": input_ids, "ref_s": ref_s})
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

    Needs (in this env): kokoro, torch, scipy, huggingface_hub. The Swedish
    ``g2p_sv`` module comes from the ``kokoro-sv`` package's vendored copy by
    default; pass ``--svml-path`` to use a swedish-kokoro checkout instead.
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
        svml_path: str | None,
        voices_repo: str,
        default_sv_voice: str = "Stina",
        lang: str = "sv",
        allowed_langs: str = "sv,en",
        device: str | None = None,
        other_langs_weights: str = "base",
        extra_voices: str | None = None,
    ) -> None:
        self._extra_voices = os.path.expanduser(extra_voices) if extra_voices else ""
        if other_langs_weights not in ("base", "swedish"):
            raise SystemExit(f"other_langs_weights must be 'base' or 'swedish', not {other_langs_weights!r}")
        if svml_path:  # override: g2p_sv from a swedish-kokoro checkout
            path = Path(svml_path).expanduser().resolve()
            if not path.is_dir():
                raise SystemExit(f"--svml-path not found: {path}")
            sys.path.insert(0, str(path))  # for `import g2p_sv`
        else:  # default: the kokoro-sv package's vendored g2p chain
            try:
                import kokoro_sv
            except ImportError as exc:  # pragma: no cover
                raise SystemExit(
                    f"kokoro-sv package not installed and no --svml-path given: {exc}"
                )
            vend = Path(kokoro_sv.__file__).resolve().parent / "_vendor"
            sys.path.insert(0, str(vend))
            os.environ.setdefault("SV_G2P_DIR", str(vend / "g2p"))
        os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")  # prefer the neural Swedish g2p
        import torch
        from kokoro import KModel
        from huggingface_hub import hf_hub_download

        try:
            from g2p_sv import SwedishG2P
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"g2p_sv not importable: {exc}")

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
        self.kmodel = self._model  # for warm_shapes()
        self.device = device
        if device.startswith("mps"):
            install_hybrid_forward(self._model, device)
        # Non-Swedish languages: the ORIGINAL Kokoro-82M weights by default.
        # The Swedish fine-tune shares one network, and training it on Swedish
        # only measurably degraded the others (vs base, same voices, Whisper
        # small: fr CER 0.007 -> 0.121; heard-as-language es 0.98 -> 0.80,
        # it 0.95 -> 0.66; en unchanged). A second 82M model (~330 MB) makes
        # en/fr/es/it exactly base Kokoro, whatever later Swedish training does.
        self.other_langs_weights = other_langs_weights
        if other_langs_weights == "base":
            self._other_model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
            if device.startswith("mps"):
                install_hybrid_forward(self._other_model, device)
        else:
            self._other_model = self._model
        self.kmodels = [self._model] + ([self._other_model] if self._other_model is not self._model else [])
        # Voices with their own weights. A pack alone only moves timbre (the
        # prosody lives in the shared predictor), so a voice fine-tuned on its
        # own data ships as its own model and the other voices stay on
        # kokoro_sv.pth. Published ones are named by "model" in the repo's
        # voices_manifest.json; <extra-voices>/<Name>.pth next to <Name>.pt
        # adds or overrides one locally. Loaded up front (~330 MB each) so the
        # MPS shape warm-up covers them too.
        model_files: dict[str, str] = {}
        try:
            manifest = json.loads(Path(hf_hub_download(voices_repo, "voices_manifest.json")).read_text("utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("no voices_manifest.json in %s (%s); no published per-voice models", voices_repo, e)
            manifest = {}
        for name, info in manifest.items():
            if isinstance(info, dict) and info.get("model"):
                model_files[name] = hf_hub_download(voices_repo, info["model"])
        if self._extra_voices and os.path.isdir(self._extra_voices):
            for f in sorted(Path(self._extra_voices).glob("*.pth")):
                model_files[f.stem] = str(f)
        self._voice_models: dict[str, Any] = {}
        for name, f in sorted(model_files.items()):
            m = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg, model=f)
            # KModel loads with strict=False, so a StyleTTS2 export ("module."
            # prefixes, parametrized weight_norm) leaves the decoder random and
            # the voice is pure noise. Reload in KModel's key format and refuse
            # weights it does not recognise (missing keys are fine: kokoro_sv.pth
            # itself has no AdaIN norm affines).
            state = self._torch.load(f, map_location="cpu", weights_only=True)
            for key, sd in state.items():
                res = getattr(m, key).load_state_dict({
                    (k[7:] if k.startswith("module.") else k)
                    .replace("parametrizations.weight.original0", "weight_g")
                    .replace("parametrizations.weight.original1", "weight_v"): v
                    for k, v in sd.items()}, strict=False)
                if res.unexpected_keys:
                    raise RuntimeError(f"{f}: {key} has unknown weights, e.g. {res.unexpected_keys[:3]}")
            m = m.to(device).eval()
            if device.startswith("mps"):
                install_hybrid_forward(m, device)
            self._voice_models[name] = m
            self.kmodels.append(m)
        self.voice_models = sorted(self._voice_models)  # reported by /health
        self._g2p = SwedishG2P(backend="neural")
        self._sv_voices: dict[str, Any] = {}  # name -> voicepack tensor (lazy, cached)
        self._pipes: dict[str, Any] = {}  # KPipeline lang_code -> pipeline (lazy, cached)
        backend = getattr(self._g2p, "backend", "?")
        logger.info(
            "KokoroSVML ready: device=%s, swedish g2p=%s, voices=%s, default sv voice=%s, langs=%s (primary %s), "
            "other languages on %s weights, voices with own weights: %s",
            device, backend, voices_repo, default_sv_voice, sorted(self._allowed), self._primary,
            other_langs_weights, ", ".join(self.voice_models) or "none",
        )

    def _sv_voicepack(self, name: str) -> Any:
        """Lazily fetch+cache a Swedish pack: ``<extra-voices>/<name>.pt`` if
        present (local packs, e.g. freshly fitted voices not yet published),
        else ``voices/<name>.pt`` from the voices repo."""
        if name not in self._sv_voices:
            from huggingface_hub import hf_hub_download

            local = os.path.join(self._extra_voices, f"{name}.pt") if self._extra_voices else ""
            p = local if local and os.path.isfile(local) else hf_hub_download(self._voices_repo, f"voices/{name}.pt")
            self._sv_voices[name] = self._torch.load(p, map_location=self._device, weights_only=True)
        return self._sv_voices[name]

    def _pipe(self, code: str) -> Any:
        """Lazily build a base-Kokoro ``KPipeline`` for a non-Swedish lang code."""
        if code not in self._pipes:
            from kokoro import KPipeline

            self._pipes[code] = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M", model=self._other_model)
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

    def _synth_sv(self, text: str, voice: str | None, speed: float = 1.0) -> NDArray[np.float32]:
        # A base-Kokoro-style name (af_heart) isn't a Swedish pack -> use the default.
        name = voice if (voice and not _re.match(r"^[a-z][fm]_", voice)) else self._default_sv_voice
        vp = None
        # "Klara_viskar" (a style tag) falls back to "Klara" if that voice has no such pack
        for cand in dict.fromkeys((name, name.split("_")[0], self._default_sv_voice)):
            try:
                vp = self._sv_voicepack(cand)
                break
            except Exception:
                logger.warning("swedish voice %r not found in %s", cand, self._voices_repo)
        if vp is None:
            raise RuntimeError(f"no Swedish voice pack for {name!r}")
        base = cand.split("_")[0]
        model = self._voice_models.get(cand) or self._voice_models.get(base, self._model)
        # Digits are spelled out inside kokoro-sv's SwedishG2P (>= 0.1.2).
        # "[ord](/IPA/)" pins one word's pronunciation, as in English Kokoro.
        pieces = _re.split(r"\[([^\]]+)\]\(/([^)/]+)/\)", text)
        ipa = " ".join(
            p.strip() if i % 3 == 2 else self._g2p(p).strip()
            for i, p in enumerate(pieces) if i % 3 != 1 and p.strip()
        ).replace("ʏ", "y")
        ids = [i for i in (model.vocab.get(p) for p in ipa) if i is not None]
        if not ids:
            return np.zeros(0, dtype=np.float32)
        torch = self._torch
        with torch.no_grad():
            audio_t, _pred_dur = model.forward_with_tokens(
                torch.LongTensor([[0, *ids, 0]]).to(self._device),
                vp[len(ids) - 1].to(self._device),
                speed=speed,
            )
        audio = self._notch(audio_t.squeeze().cpu().numpy().astype(np.float32))
        return self._trim_silence(audio)

    def _synth_other(self, text: str, lang: str, voice: str | None, speed: float = 1.0) -> NDArray[np.float32]:
        code, default_voice = self._LANGS.get(lang, self._LANGS["en"])
        # Only forward a real base-Kokoro voice (af_heart); ignore Swedish/Piper/OpenAI
        # names so each language uses its own default voice.
        kvoice = voice if (voice and _re.match(r"^[a-z][fm]_", voice)) else default_voice
        chunks = [
            a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
            for _gs, _ps, a in self._pipe(code)(text, voice=kvoice, speed=speed)
        ]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)

    def resolve_lang(self, text: str, language: str | None = None, language_hint: str | None = None) -> str:
        """The language *text* will be spoken in (explicit > detected > hint > primary)."""
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
        return lang

    def synth(
        self, text: str, voice: str | None, language: str | None = None,
        language_hint: str | None = None, speed: float = 1.0
    ) -> Tuple[int, NDArray[np.int16]]:
        lang = self.resolve_lang(text, language, language_hint)
        if lang in ("sv", "swedish", "se"):
            audio = self._synth_sv(text, voice, speed)
        else:
            audio = self._synth_other(text, lang, voice, speed)
        pcm = np.clip(audio.reshape(-1) * 32767.0, -32768, 32767).astype(np.int16)
        return 24000, pcm


class VoxEngine:
    """Expressive speech with VoxCPM2 (openbmb, 2.4B) through MLX, Apple Silicon only.

    Kokoro stays the everyday voice (~0.1 s a sentence); lines carrying a style
    tag ([skrattar], [arg], [viskar], ... or any free description) go here,
    where laughter, whispering and anger are real rather than a nudged style
    vector (the Kokoro style fine-tunes were inaudible, Oct 2026). A voice is a
    VoxCPM2 reference clip; Klara and Viktor are cloned from the takes their
    Kokoro voices were distilled from, so both engines sound like one person.

    8-bit weights at 5 diffusion steps: 12/12 short sentences correct by
    Whisper vs 10/12 for 4-bit (it garbled "Ja, Plopp har vi!"), ~0.5 s a short
    sentence on an M5 Max, ~6 GB. More steps were not more reliable.

    MLX arrays belong to the thread that made them, so load() and every
    synth() must run on the same thread: the server's TTS thread.
    """

    SAMPLE_RATE = 24000  # resampled from VoxCPM2's 48 kHz to match Kokoro
    # Style tag -> VoxCPM2 instruction. Any other tag is passed on verbatim, so
    # "[sarcastic]" or "[talking like a pirate]" work too.
    INSTRUCT = {
        "viskar": "whispering softly, very quiet",
        "glad": "happy and excited, bright and lively",
        "skrattar": "laughing while speaking, amused",
        "ledsen": "sad and subdued, slow and quiet",
        "arg": "angry and stern, raised voice",
        "tänker": "thinking out loud, hesitant and soft",
    }
    # Said "while thinking", so the listener hears the robot react the moment
    # they stop talking and the reply can start ~1 s later.
    FILLERS = {
        "sv": ["Hmm…", "Mm, vänta lite…", "Ja, okej…", "Okej…", "Öh, vänta…", "Mm…"],
        "en": ["Hmm…", "Mm, let me see…", "Okay…", "Uh, one sec…", "Right…", "Mm…"],
        "de": ["Hmm…", "Mal sehen…", "Okay…", "Moment…", "Mm…"],
        "fr": ["Hmm…", "Euh, voyons…", "D'accord…", "Alors…", "Mm…"],
        "es": ["Mmm…", "A ver…", "Vale…", "Bueno…", "Mm…"],
        "it": ["Mmm…", "Vediamo…", "Allora…", "Okay…", "Mm…"],
    }

    def __init__(self, voices: dict, repo: str = "mlx-community/VoxCPM2-8bit", steps: int = 5,
                 cache_dir: str | None = None, filler_langs: str = "sv,en,de,fr,es,it") -> None:
        self.voices = {name: str(Path(p).expanduser()) for name, p in voices.items()}
        for name, path in self.voices.items():
            if not Path(path).is_file():
                raise SystemExit(f"--vox-voice {name}: no such file {path}")
        self.repo, self.steps = repo, steps
        self.cache_dir = Path(cache_dir or "~/.cache/kokoro-voice-server/vox").expanduser()
        self.filler_langs = [x.strip() for x in filler_langs.split(",") if x.strip() in self.FILLERS]
        self.model: Any = None
        self.fillers: dict = {}   # (voice, lang) -> [int16 pcm]
        self._recent: dict = {}   # (voice, lang) -> index played last
        self._cache: dict = {}    # (voice, instruct, text) -> int16 pcm, short lines only
        # (pcm, lang) -> transcript, set by the server when Whisper is on; used
        # to throw away fillers that say something else ("Okej…" came out as
        # "Ja, ja, vad är det?").
        self.transcribe: Any = None

    def load(self) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load

        # MLX keeps freed buffers for reuse, and every new sentence length
        # frees buffers of new sizes. The default cap is ~the whole machine:
        # the server grew 3 GB a request to 134 GB and swapped (Oct 2026).
        # 2 GB holds it at ~6 GB with no loss of speed (RTF 0.26 either way).
        mx.set_cache_limit(2 * 1024**3)
        t0 = time.perf_counter()
        self.model = load(self.repo)
        ref = next(iter(self.voices.values()))
        next(self.model.generate(text="Hej.", ref_audio=ref, inference_timesteps=self.steps))  # warm-up
        logger.info("VoxCPM2 %s loaded in %.1f s, voices %s", self.repo, time.perf_counter() - t0,
                    ", ".join(self.voices))

    def has_voice(self, voice: str | None) -> bool:
        return bool(voice) and voice.split("_")[0] in self.voices

    def _generate(self, text: str, ref: str, instruct: str | None) -> NDArray[np.float32]:
        from scipy.signal import resample_poly

        r = next(self.model.generate(text=text, instruct=instruct, ref_audio=ref,
                                     inference_timesteps=self.steps))
        audio = np.asarray(r.audio, dtype=np.float32).reshape(-1)
        return resample_poly(audio, self.SAMPLE_RATE, r.sample_rate).astype(np.float32)

    def synth(self, text: str, voice: str, style: str | None = None) -> Tuple[int, NDArray[np.int16]]:
        """*voice* is a --vox-voice name (a "_style" suffix is ignored); *style*
        a style tag or a free English description of how to say it."""
        if self.model is None:
            self.load()
        voice = voice.split("_")[0]
        instruct = self.INSTRUCT.get(style, style) if style else None
        key = (voice, instruct, text)
        if key in self._cache:
            return self.SAMPLE_RATE, self._cache[key]
        pcm = self._take(text, voice, instruct)
        if len(text) <= 60:
            if len(self._cache) >= 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = pcm
        return self.SAMPLE_RATE, pcm

    def _take(self, text: str, voice: str, instruct: str | None) -> NDArray[np.int16]:
        audio = self._generate(text, self.voices[voice], instruct)
        # Now and then VoxCPM2 runs on: babble after the words, or seconds of
        # "mm" for "Mm." Speech is ~14 chars/s; far beyond that, try once more
        # and keep the shorter take.
        limit = 1.5 + 0.2 * len(text)
        if len(audio) / self.SAMPLE_RATE > limit:
            again = self._generate(text, self.voices[voice], instruct)
            logger.info("vox: %.1f s for %r (limit %.1f), retried: %.1f s", len(audio) / self.SAMPLE_RATE,
                        text[:40], limit, len(again) / self.SAMPLE_RATE)
            audio = min(audio, again, key=len)
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    # --- thinking fillers ---------------------------------------------------

    def filler_jobs(self) -> list:
        """One callable per missing filler (run each on the TTS thread). Takes
        already on disk are loaded at once; delete a bad one there to redo it."""
        import soundfile as sf

        jobs = []
        for voice in self.voices:
            for lang in self.filler_langs:
                bank = self.fillers.setdefault((voice, lang), [])
                for i, text in enumerate(self.FILLERS[lang]):
                    path = self.cache_dir / "fillers" / voice / lang / f"{i:02d}.wav"
                    if path.is_file():
                        bank.append(sf.read(str(path), dtype="int16")[0])
                    else:
                        jobs.append(lambda v=voice, l=lang, t=text, p=path: self._make_filler(v, l, t, p))
        return jobs

    @staticmethod
    def _words(text: str) -> str:
        return " ".join(_re.sub(r"[^\w\s]", " ", text.lower()).split())

    def _heard(self, pcm: NDArray[np.int16], lang: str, text: str) -> bool:
        """Whisper hears every word of *text* (letter-level similarity is too
        lax for one-word fillers: "Ave" is 75% "A ver"). Hums ("Hmm…", "Mm…")
        always pass: Whisper writes those any old way."""
        want = [w for w in self._words(text).split() if not _re.fullmatch(r"[hmueö]+", w)]
        if self.transcribe is None or not want:
            return True
        heard = self._words(self.transcribe(pcm, lang)).split()
        return all(w in heard for w in want)

    def _make_filler(self, voice: str, lang: str, text: str, path: Path) -> None:
        import soundfile as sf

        for _ in range(3):
            pcm = self._take(text, voice, self.INSTRUCT["tänker"])
            # a filler that drags is worse than none
            if len(pcm) <= self.SAMPLE_RATE * 2.5 and self._heard(pcm, lang, text):
                break
        else:
            logger.info("vox: no good take of filler %r (%s/%s)", text, voice, lang)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), pcm, self.SAMPLE_RATE, subtype="PCM_16")
        self.fillers.setdefault((voice, lang), []).append(pcm)

    def filler(self, voice: str, lang: str) -> NDArray[np.int16] | None:
        """A random filler for *voice* in *lang*, never the same twice in a row."""
        import random

        bank = self.fillers.get((voice.split("_")[0], lang)) or []
        if not bank:
            return None
        choices = [i for i in range(len(bank)) if i != self._recent.get((voice, lang))] or [0]
        i = random.choice(choices)
        self._recent[(voice, lang)] = i
        return bank[i]


# --- Apple MPS: split the model, warm the decoder ---------------------------
#
# On MPS every NEW tensor shape costs a graph compile: measured on an M5 Max,
# a never-seen sentence length took 0.65-0.85 s to synthesize vs 0.07 s once
# that shape was cached (0.16 s if macOS's on-disk shader cache had it from an
# earlier process). A conversation is made of novel sentences, so nearly every
# reply paid it — that was the "sporadic 0.4-0.8 s TTS spike" in the app's
# latency budget. Two shapes vary per utterance: the token length L (BERT, text
# encoder, duration LSTM) and the frame count F (F0/N LSTMs, decoder).
#
# The LSTM-heavy front is cheap on CPU (~30 ms total, no shape cost) and the
# LSTM compiles on MPS grow with length; the decoder (iSTFTNet convs) is the one
# part that is slow on CPU (~0.35 s) and fast on MPS (~45 ms). So on MPS the
# model is split: front on CPU, decoder on MPS. Only F then needs warming —
# one small decoder forward per frame count, ~0.3 s each the first time ever,
# ~0.05 s from the OS cache — done in the background at startup. Result:
# novel sentences synthesize in ~0.1-0.2 s flat instead of 0.4-0.8 s.
#
# Memory: each warmed frame count also kept ~30 MB of MPS buffers alive, so
# warming 8-480 frames for two voice models held 65 GB (found Oct 2026, hidden
# in swap). torch.mps.empty_cache() after every step and every request drops
# the buffers but keeps what made warming worth it: the shader compiles, which
# macOS caches on disk. A novel length then costs ~0.19 s instead of ~1 s cold
# (0.07 s with buffers kept), and the server stays at ~5 GB.
# Kokoro: 24 kHz, 600 samples per frame → 40 frames per second of speech.
WARM_FRAMES_PER_SECOND = 40


def release_mps_buffers() -> None:
    """Give torch's cached MPS buffers back to the system (see above)."""
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def install_hybrid_forward(model: Any, gpu: str) -> None:
    """Move Kokoro's text front to CPU and keep only the decoder on *gpu*.

    Replaces ``model.forward_with_tokens`` on the instance with the same
    computation split across devices, so both ``KModel.forward`` (base
    Kokoro's pipeline) and direct ``forward_with_tokens`` callers (the Swedish
    engine) pick it up unchanged.
    """
    import torch

    for sub in (model.bert, model.bert_encoder, model.predictor, model.text_encoder):
        sub.to("cpu")
    model.decoder.to(gpu)

    @torch.no_grad()
    def forward_with_tokens(input_ids: Any, ref_s: Any, speed: float = 1) -> Tuple[Any, Any]:
        input_ids = input_ids.cpu()
        ref_s = ref_s.cpu()
        L = input_ids.shape[-1]
        input_lengths = torch.full((input_ids.shape[0],), L, dtype=torch.long)
        text_mask = torch.arange(L).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(L), pred_dur)
        pred_aln_trg = torch.zeros((L, indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = model.decoder(asr.to(gpu), F0_pred.to(gpu), N_pred.to(gpu), ref_s[:, :128].to(gpu)).squeeze()
        return audio, pred_dur

    model.forward_with_tokens = forward_with_tokens
    logger.info("Kokoro split: text front on cpu, decoder on %s (MPS shape-compile workaround)", gpu)


def warm_shape_steps(model: Any, device: str, max_seconds: float = 12.0):
    """Yield zero-arg callables, one per decoder frame count to warm on *device*.

    They are yielded rather than run so the caller can interleave them with
    real requests on the inference thread: a request never waits behind more
    than one step (< 0.5 s).
    """
    import torch

    spk = torch.zeros(1, 128, device=device)

    def frame_step(F: int):
        def run() -> None:
            with torch.no_grad():
                model.decoder(
                    torch.zeros(1, 512, F, device=device),
                    torch.zeros(1, 2 * F, device=device),
                    torch.zeros(1, 2 * F, device=device),
                    spk,
                )
            release_mps_buffers()
        return run

    # Short replies (few frames) first so the common case warms earliest.
    for F in range(8, int(max_seconds * WARM_FRAMES_PER_SECOND) + 1):
        yield frame_step(F)


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


class SpeakerEncoder:
    """Speaker-verification embeddings (ECAPA-TDNN) for /v1/audio/speaker.

    Returns one 192-d unit vector per utterance; cosine similarity between two
    vectors says whether the same person spoke. This is the audio twin of the
    face embedder: the SERVER holds no identities and no audio -- it turns a
    clip into a vector and forgets it. Whatever database of voices exists lives
    in the client, on its own disk.

    Runs on CPU (about 20 ms for a 3 s clip), so it never contends with the
    TTS decoder on the GPU. The model (~80 MB) is fetched once from the
    HuggingFace hub into the local cache; set HF_HUB_OFFLINE=1 afterwards to
    guarantee no network access at all.
    """

    MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    SAMPLE_RATE = 16000

    def __init__(self) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        cache = Path(os.environ.get("SPEAKER_MODEL_DIR",
                                    Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"))
        self._torch = torch
        self._enc = EncoderClassifier.from_hparams(
            source=self.MODEL, savedir=str(cache), run_opts={"device": "cpu"})
        self._enc.encode_batch(torch.zeros(1, self.SAMPLE_RATE))  # warm the graph
        logger.info("Speaker encoder ready: %s (cpu, 192-d)", self.MODEL)

    def embed(self, audio_bytes: bytes) -> Tuple[list, float]:
        """Return (unit-norm embedding as a list, clip length in seconds)."""
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if sr != self.SAMPLE_RATE:
            n = int(len(mono) * self.SAMPLE_RATE / sr)
            mono = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono).astype(np.float32)
        seconds = len(mono) / self.SAMPLE_RATE
        if len(mono) < self.SAMPLE_RATE // 4:   # < 0.25 s: nothing to fingerprint
            return [], seconds
        with self._torch.no_grad():
            emb = self._enc.encode_batch(self._torch.from_numpy(mono)[None, :])
        vec = emb.squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist(), seconds


def build_app(engine: Any, stt: Any = None, warm_shapes: bool = True, warm_seconds: float = 12.0,
              speaker: Any = None, vox: Any = None) -> Any:
    import time
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import File, Form, FastAPI, Response, UploadFile
    from pydantic import BaseModel
    from fastapi.responses import JSONResponse

    # ONE dedicated thread per engine, for the life of the process. Starlette's
    # default threadpool (up to 40 workers) is unusable for torch inference:
    # macOS/glibc malloc keeps per-thread free-lists, so every worker that has
    # ever run inference retains a few hundred MB of cached allocations
    # (measured: 8 days of service = 17 GB RSS). A fixed thread per engine caps
    # that at two caches, and keeps each (non-thread-safe) engine serialised —
    # while letting TTS and Whisper overlap, which the conversation app relies
    # on: it transcribes the user's turn WHILE the reply is being spoken/prepared.
    tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
    stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
    spk_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speaker")

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
        # Speech rate multiplier: 1.0 = normal, <1 slower, >1 faster (clamped 0.5-2.0).
        speed: float = 1.0
        model: str = ""
        response_format: str = "wav"

    app = FastAPI(title="kokoro voice server")

    warm_state = {"done": 0, "total": 0, "finished": not warm_shapes}
    mps = str(getattr(engine, "device", "cpu")).startswith("mps")

    def _warm_in_background() -> None:
        """Run warm_shape_steps on the TTS thread, one step at a time.

        Steps are submitted individually so real requests queue between them:
        a request during warm-up waits for at most one step (< 1 s), and the
        server is usable from the first second — just not yet at full speed.
        """
        models = getattr(engine, "kmodels", None) or [getattr(engine, "kmodel", None)]
        models = [m for m in models if m is not None]
        device = str(getattr(engine, "device", "cpu"))
        if not models or not device.startswith("mps"):
            warm_state["finished"] = True
            return  # only MPS pays per-shape compiles; CPU/CUDA need no warming
        # Each model has its own decoder, so each pays its own shape compiles.
        steps = [st for m in models for st in warm_shape_steps(m, device, max_seconds=warm_seconds)]
        warm_state["total"] = len(steps)
        t0 = time.perf_counter()
        logger.info("Warming %d MPS shapes in the background (TTS answers meanwhile)...", len(steps))
        for i, step in enumerate(steps, 1):
            try:
                tts_pool.submit(step).result()
            except Exception as exc:  # a warm step must never take the server down
                logger.warning("shape warm-up step %d failed: %s", i, exc)
                break
            warm_state["done"] = i
        # One real utterance per language: the steps above cover the shape-
        # sensitive sub-modules, this covers the rest of the graph (and builds
        # the lazily-created base-Kokoro pipeline for the non-Swedish languages).
        for lang, text in (("sv", "Hej, nu är jag varm."), ("en", "Hello, I am warmed up now.")):
            try:
                tts_pool.submit(engine.synth, text, None, lang).result()
            except Exception as exc:
                logger.debug("warm-up utterance (%s) failed: %s", lang, exc)
        warm_state["finished"] = True
        logger.info("MPS shape warm-up done: %d shapes in %.0f s", warm_state["done"], time.perf_counter() - t0)

    if vox is not None and stt is not None:
        def _vox_transcribe(pcm: NDArray[np.int16], lang: str) -> str:
            # Beam search and no robot prompt: one-word clips need all the help
            # (greedy base Whisper hears a good "Vale" as "¡Morale!").
            def run() -> str:
                segments, _ = stt._model.transcribe(io.BytesIO(wav), language=lang, beam_size=5)
                return " ".join(seg.text for seg in segments)
            wav, _ = _to_audio_bytes(vox.SAMPLE_RATE, pcm, "wav")
            return stt_pool.submit(run).result()
        vox.transcribe = _vox_transcribe

    def _vox_in_background() -> None:
        """Load VoxCPM2 and render the missing fillers, on the TTS thread one
        job at a time so requests keep being served in between."""
        try:
            tts_pool.submit(vox.load).result()
        except Exception as exc:
            logger.error("VoxCPM2 failed to load, styled lines fall back to Kokoro: %s", exc)
            return
        jobs = vox.filler_jobs()
        t0 = time.perf_counter()
        for job in jobs:
            try:
                tts_pool.submit(job).result()
            except Exception as exc:
                logger.warning("filler failed: %s", exc)
        logger.info("Fillers ready: %s (%d new in %.0f s)",
                    {f"{v}/{l}": len(b) for (v, l), b in vox.fillers.items()}, len(jobs),
                    time.perf_counter() - t0)

    @app.on_event("startup")
    async def _start_warmup() -> None:
        if vox is not None:
            threading.Thread(target=_vox_in_background, name="vox-load", daemon=True).start()
        if warm_shapes:
            threading.Thread(target=_warm_in_background, name="warm-shapes", daemon=True).start()

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
            "speaker": speaker is not None,
            # Which weights serve the non-Swedish languages: "base" = original
            # Kokoro-82M (immune to Swedish training), "swedish" = the fine-tune.
            "other_langs_weights": getattr(engine, "other_langs_weights", None),
            # Swedish voices served by their own fine-tuned weights (--extra-voices).
            "voice_models": getattr(engine, "voice_models", []),
            # Voices VoxCPM2 speaks styled lines in, and whether it has loaded.
            "vox_voices": sorted(vox.voices) if vox is not None else [],
            "vox_loaded": vox is not None and vox.model is not None,
            "fillers": {f"{v}/{l}": len(b) for (v, l), b in vox.fillers.items()} if vox is not None else {},
            # False while MPS shapes are still being warmed: TTS works but a
            # novel sentence may take ~0.7 s instead of ~0.1 s.
            "warm": warm_state["finished"],
            "warm_progress": f"{warm_state['done']}/{warm_state['total']}",
        }

    def _synth_all(body: SpeechRequest, text: str) -> Tuple[int, NDArray[np.int16]]:
        # Chunk long text by sentence: Kokoro caps at ~510 tokens/utterance, and it
        # keeps any single engine call small. Concatenate the PCM back into one WAV.
        # Inline tags ([Viktor], [viskar], [paus 0.5], ...) switch voice, style,
        # speed or insert silence between chunks (parse_tags).
        sample_rate = 24000
        parts = []
        base_speed = max(0.5, min(2.0, body.speed or 1.0))
        # model "voxcpm2" sends untagged lines to VoxCPM2 as well.
        all_vox = (body.model or "").lower().startswith("vox")
        for seg in parse_tags(text):
            if seg[0] == "pause":
                parts.append(np.zeros(int(seg[1] * sample_rate), dtype=np.int16))
                continue
            _, chunk_text, voice, style, speed = seg
            voice = voice or body.voice or getattr(engine, "default_voice", "")
            has_vox = vox is not None and vox.model is not None and vox.has_voice(voice)
            # The fine-tuned Kokoro voices only speak Swedish; anything else would fall
            # back to a stock Kokoro voice (a woman, for English). A voice with a VoxCPM2
            # clone speaks every other language as itself instead.
            foreign = (has_vox and not style and not all_vox and hasattr(engine, "resolve_lang")
                       and engine.resolve_lang(chunk_text, body.language, body.language_hint)
                       not in ("sv", "swedish", "se"))
            use_vox = has_vox and bool(style or all_vox or foreign)
            if style and not use_vox:
                voice = f"{voice}_{style}"
            for chunk in split_sentences(chunk_text) or [chunk_text]:
                if use_vox:
                    sample_rate, pcm = vox.synth(chunk, voice, style)
                else:
                    sample_rate, pcm = engine.synth(chunk, voice, body.language, body.language_hint,
                                                    max(0.5, min(2.0, base_speed * speed)))
                    if mps:
                        release_mps_buffers()
                if len(pcm):
                    parts.append(pcm)
        return sample_rate, (np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16))

    @app.post("/v1/audio/speech")
    async def speech(body: SpeechRequest) -> Response:
        text = (body.input or "").strip()
        if not text:
            return Response(content=b"", media_type="audio/wav")
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        sample_rate, pcm = await loop.run_in_executor(tts_pool, _synth_all, body, text)
        audio, media = _to_audio_bytes(sample_rate, pcm, body.response_format)
        # Per-request timing: the conversation app's latency budget is built from these.
        logger.info(
            "TTS %4.0f ms -> %4.1f s audio  lang=%s hint=%s  %r",
            (time.perf_counter() - t0) * 1000, len(pcm) / sample_rate,
            body.language or "-", body.language_hint or "-", text[:60],
        )
        return Response(content=audio, media_type=media)

    @app.get("/v1/audio/filler")
    def filler(voice: str | None = None, language: str = "sv", response_format: str = "wav") -> Response:
        """A short pre-rendered "hmm…"/"mm, vänta lite…" in *voice* and
        *language*, to play the moment the user stops talking while the reply
        is being thought up and voiced. Served from memory, so it never waits
        for the TTS thread. 204 when there is none (no VoxCPM2, or not ready)."""
        pcm = vox.filler(voice or getattr(engine, "default_voice", ""), _norm_lang(language)) \
            if vox is not None else None
        if pcm is None:
            return Response(status_code=204)
        audio, media = _to_audio_bytes(vox.SAMPLE_RATE, pcm, response_format)
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
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        text, lang = await loop.run_in_executor(stt_pool, stt.transcribe, data, language)
        logger.info(
            "STT %4.0f ms  lang=%s  %r", (time.perf_counter() - t0) * 1000, lang or "-", text[:60]
        )
        return {"text": text, "language": lang}

    @app.post("/v1/audio/speaker")
    async def speaker_embedding(file: UploadFile = File(...)) -> Any:
        """Speaker-verification embedding for one utterance.

        multipart ``file`` (wav/flac/ogg) -> ``{"embedding": [192 floats],
        "dim": 192, "seconds": clip length, "model": name}``. Compare two
        embeddings by cosine similarity (dot product; they are unit vectors).
        Clips under 0.25 s return an empty embedding. Nothing is stored.
        """
        if speaker is None:
            return JSONResponse({"error": "speaker embeddings not enabled (start with --speaker ecapa)"},
                                status_code=501)
        data = await file.read()
        if not data:
            return {"embedding": [], "dim": 0, "seconds": 0.0, "model": speaker.MODEL}
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        vec, seconds = await loop.run_in_executor(spk_pool, speaker.embed, data)
        logger.info("SPEAKER %4.0f ms  clip=%.1fs", (time.perf_counter() - t0) * 1000, seconds)
        return {"embedding": vec, "dim": len(vec), "seconds": round(seconds, 2), "model": speaker.MODEL}

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
        default=os.environ.get("SWEDISH_KOKORO_PATH"),
        help="path to a swedish-kokoro checkout; required for --engine kokoro-sv, "
        "optional for kokoro-svml (default: the kokoro-sv package's vendored g2p)",
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
        "--device",
        default=os.environ.get("KOKORO_DEVICE"),
        help="torch device for the Kokoro engines: cpu, mps or cuda (default: auto — cuda > mps > cpu)",
    )
    p.add_argument(
        "--no-warm-shapes", action="store_true",
        help="skip the background MPS shape warm-up (see warm_shape_steps); "
        "TTS then costs ~0.7 s per never-seen sentence length instead of ~0.1 s",
    )
    p.add_argument(
        "--warm-seconds", type=float, default=12.0,
        help="warm frame counts for utterances up to this many seconds of audio (default 12)",
    )
    p.add_argument(
        "--extra-voices",
        default=os.environ.get("KOKORO_SV_EXTRA_VOICES"),
        help="kokoro-svml: directory of extra Swedish packs (<Name>.pt), checked before the HF "
        "voices repo — for voices not (yet) published. A <Name>.pth beside a pack is that "
        "voice's own fine-tuned model (KModel format)",
    )
    p.add_argument(
        "--other-langs-weights",
        choices=["base", "swedish"],
        default=os.environ.get("KOKORO_OTHER_LANGS_WEIGHTS", "base"),
        help="kokoro-svml: weights for en/fr/es/it/... — 'base' = original Kokoro-82M (default; "
        "unaffected by Swedish fine-tuning, +~330 MB), 'swedish' = share the fine-tuned model",
    )
    p.add_argument(
        "--speaker",
        default=os.environ.get("SPEAKER_MODEL", "off"),
        help="speaker-verification embeddings on /v1/audio/speaker: 'ecapa' or 'off' (default off)",
    )
    p.add_argument(
        "--whisper",
        default=os.environ.get("WHISPER_MODEL", "off"),
        help="faster-whisper size for /v1/audio/transcriptions (tiny/base/small/medium), or 'off'",
    )
    p.add_argument(
        "--vox-voice", action="append", default=[], metavar="NAME=WAV",
        help="speak styled lines ([skrattar], [arg], [viskar], ...) for voice NAME with VoxCPM2, "
        "cloned from the reference clip WAV (repeatable). Needs Apple Silicon and the 'vox' "
        "extra (uv sync --extra vox). Also enables /v1/audio/filler for that voice",
    )
    p.add_argument("--vox-model", default="mlx-community/VoxCPM2-8bit",
                   help="MLX VoxCPM2 weights (default 8-bit; 4-bit garbles short sentences)")
    p.add_argument("--vox-steps", type=int, default=5, help="VoxCPM2 diffusion steps (default 5)")
    p.add_argument("--vox-cache", default=None,
                   help="where rendered fillers are kept (default ~/.cache/kokoro-voice-server/vox)")
    p.add_argument("--filler-langs", default="sv,en,de,fr,es,it",
                   help="languages to render thinking fillers in (default sv,en,de,fr,es,it)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if args.engine == "kokoro-sv":
        if not args.svml_path:
            raise SystemExit("--engine kokoro-sv needs --svml-path (or $SWEDISH_KOKORO_PATH)")
        engine: Any = SwedishKokoroEngine(args.svml_path)
    elif args.engine == "kokoro-svml":
        engine = KokoroSVMLEngine(
            args.svml_path,
            voices_repo=args.voices_repo,
            default_sv_voice=args.voice or "Stina",
            lang="auto" if args.lang == "a" else args.lang,
            allowed_langs=args.langs,
            device=args.device,
            other_langs_weights=args.other_langs_weights,
            extra_voices=args.extra_voices,
        )
    else:
        engine = KokoroEngine(args.voice or "af_heart", args.lang)

    stt = None
    if args.whisper and args.whisper.lower() != "off":
        stt = WhisperSTT(args.whisper)
    speaker = None
    if args.speaker and args.speaker.lower() != "off":
        speaker = SpeakerEncoder()
    logger.info(
        "Voice server: engine=%s whisper=%s on %s:%d", args.engine, args.whisper, args.host, args.port
    )

    vox = None
    if args.vox_voice:
        try:
            voices = dict(v.split("=", 1) for v in args.vox_voice)
        except ValueError:
            raise SystemExit("--vox-voice wants NAME=WAV")
        vox = VoxEngine(voices, repo=args.vox_model, steps=args.vox_steps, cache_dir=args.vox_cache,
                        filler_langs=args.filler_langs)

    import uvicorn

    app = build_app(engine, stt, warm_shapes=not args.no_warm_shapes, warm_seconds=args.warm_seconds,
                    speaker=speaker, vox=vox)
    if args.host in ("::", "dual"):
        # Serve BOTH stacks from one socket. uvicorn's own "::" bind is v6-only
        # here, and that breaks clients that resolve this host to IPv4 — a robot
        # reaching us by mDNS name may pick either family, so refusing one makes
        # the voice server look "unreachable" while other services on the same
        # machine answer fine.
        import socket

        if not socket.has_dualstack_ipv6():
            logger.warning("Dual-stack sockets unsupported here; falling back to IPv4 0.0.0.0")
            uvicorn.run(app, host="0.0.0.0", port=args.port)
            return
        sock = socket.create_server(
            ("::", args.port), family=socket.AF_INET6, dualstack_ipv6=True, reuse_port=False
        )
        logger.info("Listening dual-stack (IPv4 + IPv6) on port %d", args.port)
        uvicorn.Server(uvicorn.Config(app)).run(sockets=[sock])
        return

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

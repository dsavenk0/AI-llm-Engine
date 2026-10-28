# On-device speech-to-text with Whisper on MLX. Optional, like image generation (mflux) and OCR (Vision):
# nothing is pulled in until `./llm setup --audio` installs mlx-whisper, and the model downloads on first use.
#
#   transcribe(path) -> {"text", "language", "segments": [{start, end, text}]}
#
# This is the engine-side primitive; the voice module can later use it in place of the system recogniser, and
# Kokoro TTS / Omni models are the next audio pieces.
import os

# Whisper turbo: large-v3 quality, a few times faster — a good default for a laptop. Overridable per call.
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aiff", ".aif", ".caf",
              ".wma", ".webm", ".mp4", ".mov", ".m4v")


class AudioError(Exception):
    pass


def whisper_available():
    """True when mlx-whisper is installed (./llm setup --audio)."""
    import importlib.util
    return importlib.util.find_spec("mlx_whisper") is not None


def transcribe(path, model=None):
    """Transcribe an audio (or video) file to text with Whisper on MLX. Raises AudioError when mlx-whisper is
    not installed, the file is missing, or decoding fails."""
    if not os.path.isfile(path):
        raise AudioError(f"no such file: {path}")
    if not whisper_available():
        raise AudioError("transcription needs mlx-whisper — install it with ./llm setup --audio")
    import mlx_whisper
    try:
        r = mlx_whisper.transcribe(path, path_or_hf_repo=model or DEFAULT_MODEL)
    except Exception as e:                                 # a bad file, a missing model, a decode failure
        raise AudioError(f"transcription failed: {e}")
    return {
        "text": (r.get("text") or "").strip(),
        "language": r.get("language"),
        "segments": [{"start": s.get("start"), "end": s.get("end"), "text": (s.get("text") or "").strip()}
                     for s in (r.get("segments") or [])],
    }


# ───────────────────────────── text-to-speech (Kokoro on MLX) ─────────────────────────────
KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"
KOKORO_SR = 24000        # Kokoro's output sample rate


def kokoro_available():
    """True when mlx-audio is installed (./llm setup --audio)."""
    import importlib.util
    return importlib.util.find_spec("mlx_audio") is not None


def _write_wav(path, samples, sr):
    """A mono 16-bit WAV, with the stdlib — no soundfile/scipy dependency."""
    import wave
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def synthesize(text, out, voice="af_heart", speed=1.0, lang_code="a", model=None):
    """Speak `text` to a WAV at `out` with Kokoro (MLX). Returns {path, sample_rate, voice, seconds}. Raises
    AudioError when mlx-audio is not installed or nothing was produced."""
    if not text or not text.strip():
        raise AudioError("nothing to say")
    if not kokoro_available():
        raise AudioError("text-to-speech needs mlx-audio — install it with ./llm setup --audio")
    import numpy as np
    from mlx_audio.tts.utils import load_model
    m = load_model(model or KOKORO_MODEL)
    chunks, sr = [], KOKORO_SR
    try:
        for result in m.generate(text=text, voice=voice, speed=speed, lang_code=lang_code):
            chunks.append(np.array(result.audio, dtype=np.float32))
            sr = getattr(result, "sample_rate", sr) or sr
    except Exception as e:
        raise AudioError(f"speech synthesis failed: {e}")
    if not chunks:
        raise AudioError("no audio was produced")
    wav = np.concatenate(chunks)
    _write_wav(out, wav, sr)
    return {"path": out, "sample_rate": int(sr), "voice": voice, "seconds": round(len(wav) / sr, 2)}

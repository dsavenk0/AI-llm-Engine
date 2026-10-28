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

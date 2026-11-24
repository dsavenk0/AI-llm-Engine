"""Neural speech for LLM Voice — an optional helper process, run only while the voice chat is open and only when the
person chose a neural voice in LLMBar's settings (the system voice is the default and needs none of this).

It lives in its own small virtual environment (`./llm setup --voice`: mlx-audio installed without its heavy
extras, next to mlx, numpy, tqdm, tokenizers and scipy), so the engine's own environment and pinned mlx stay as
they are. The model is Qwen3-TTS 0.6B CustomVoice (MLX, 4-bit) — English and eight more languages; Russian replies
stay with the system voice, which reads Russian better.

Protocol on stdin/stdout, one JSON object per line:
    → {"say": "text", "out": "/path/reply-1.wav", "speaker": "aiden", "lang": "english"}
    ← {"ok": "/path/reply-1.wav", "seconds": 2.4, "ms": 610}      or      {"error": "why"}
The first line printed is {"ready": true, "speakers": [...], "languages": [...]} once the model is loaded.
"""
import json
import os
import sys
import time

MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
# Qwen2/Qwen3 pre-tokenizer split (the tokenizer.json of every Qwen model states this exact expression)
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
              r"|\s+(?!\S)|\s+")


class QwenBPE:
    """The model's text tokenizer from its vocab.json + merges.txt with the `tokenizers` library — what
    transformers' AutoTokenizer would build, without transformers. Only `encode` is used by the model."""

    def __init__(self, folder):
        from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
        tok = Tokenizer(models.BPE.from_file(os.path.join(folder, "vocab.json"), os.path.join(folder, "merges.txt")))
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tok.decoder = decoders.ByteLevel()
        try:
            with open(os.path.join(folder, "tokenizer_config.json")) as f:
                cfg = json.load(f)
            from tokenizers import AddedToken
            added = [(int(i), t["content"]) for i, t in (cfg.get("added_tokens_decoder") or {}).items()]
            tok.add_special_tokens([AddedToken(c, special=True, normalized=False) for _, c in sorted(added)])
        except (OSError, ValueError, KeyError):
            pass
        self.tok = tok

    def encode(self, text, add_special_tokens=False):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)


def write_wav(path, samples, sr):
    """Mono 16-bit WAV with the stdlib."""
    import wave
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    tmp = path + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def reply(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve(folder):
    import contextlib
    import numpy as np
    from mlx_audio.tts.utils import load
    with contextlib.redirect_stdout(sys.stderr):      # the library's own notices must not mix into the protocol
        model = load(folder)
    model.tokenizer = QwenBPE(folder)
    cfg = json.load(open(os.path.join(folder, "config.json")))
    talker = cfg.get("talker_config") or {}
    speakers = sorted((talker.get("spk_id") or {}).keys())
    langs = sorted((talker.get("codec_language_id") or {}).keys())
    reply({"ready": True, "speakers": speakers, "languages": langs})
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            reply({"error": "not JSON"})
            continue
        if req.get("quit"):
            break
        text, out = (req.get("say") or "").strip(), req.get("out")
        if not text or not out:
            reply({"error": "say and out are required"})
            continue
        t0 = time.time()
        try:
            chunks, sr = [], 24000
            with contextlib.redirect_stdout(sys.stderr):
                results = list(model.generate(text=text, voice=req.get("speaker") or "aiden",
                                              lang_code=req.get("lang") or "english", verbose=False))
            for r in results:
                chunks.append(np.asarray(r.audio, dtype=np.float32))
                sr = getattr(r, "sample_rate", sr) or sr
            if not chunks:
                raise RuntimeError("no audio was produced")
            audio = np.concatenate(chunks)
            write_wav(out, audio, sr)
            reply({"ok": out, "seconds": round(len(audio) / sr, 2), "ms": int((time.time() - t0) * 1000)})
        except Exception as e:                       # one bad sentence is an error line, not a dead helper
            reply({"error": f"{type(e).__name__}: {e}"[:300]})


if __name__ == "__main__":
    serve(sys.argv[1])

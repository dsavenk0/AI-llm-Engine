"""A tokenizer + chat template built from a GGUF file's own metadata (no llama.cpp, no transformers).

Two kinds of vocabulary are handled: byte-level BPE (the Llama 3 / Qwen / GPT-2 families, only for the
pre-tokenizer rules listed below) and SentencePiece ("llama" in ggml: Gemma). Anything else raises
`Unsupported`, and the caller runs the file on llama.cpp instead (never a silently wrong tokenization).

It offers what engine/compat.py needs from a tokenizer: encode, decode, chat_template, eos_token_ids,
apply_chat_template and the (absent) reasoning / (JSON-in-tags) tool-call markers.
"""
import json

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

from . import gguf

# tokenizer.ggml.pre → the regular expressions llama.cpp splits text with (in order, each one splitting the
# pieces of the previous) before BPE. Every entry here was checked to give llama.cpp's token ids on a few
# hundred lines of English, Russian, CJK, code and markup (see tests/test_gguf_engine.py for the live check).
_CONTRACTIONS = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
_GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_LLAMA3 = (_CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_QWEN2 = (_CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_DIGITS_GPT2 = [r"\p{N}", _GPT2]
PRE = {
    "gpt2": [_GPT2], "default": [_GPT2],
    "llama-bpe": [_LLAMA3], "llama3": [_LLAMA3], "smaug-bpe": [_LLAMA3],
    "qwen2": [_QWEN2], "qwen35": [_QWEN2], "deepseek-r1-qwen": [_QWEN2],
    "smollm": _DIGITS_GPT2, "refact": _DIGITS_GPT2,
}

CONTROL, USER_DEFINED, BYTE = 3, 4, 6
SPIECE = "\u2581"        # ▁, SentencePiece's whitespace


def _spm_merges(tokens, scores, types):
    """SentencePiece merges text greedily by the score of the merged piece (best first, leftmost on ties);
    that is BPE whose merge ranks are the pieces' scores. Every piece that splits into two pieces of the
    vocabulary is one merge."""
    ids = {t: i for i, t in enumerate(tokens)}
    cand = []
    for i, t in enumerate(tokens):
        if len(t) < 2 or (types and i < len(types) and types[i] in (CONTROL, BYTE)):
            continue
        for k in range(1, len(t)):
            a, b = t[:k], t[k:]
            if a in ids and b in ids:
                cand.append((-scores[i], i, a, b))
    cand.sort(key=lambda c: (c[0], c[1]))
    return [(a, b) for _, _, a, b in cand]


class Unsupported(ValueError):
    pass


def _raise(msg):
    raise ValueError(msg)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class GGUFTokenizer:
    """Same surface as the mlx-lm tokenizer wrapper where compat.py touches it."""
    has_thinking = False
    think_start = think_end = None

    def __init__(self, path, meta):
        model = meta.get("tokenizer.ggml.model")
        if model not in ("gpt2", "llama"):
            raise Unsupported(f"tokenizer model {model!r} (byte-level BPE and SentencePiece are built in)")
        tokens = gguf.read_array(path, "tokenizer.ggml.tokens")
        types = gguf.read_array(path, "tokenizer.ggml.token_type") or []
        if not tokens:
            raise Unsupported("the file has no vocabulary")
        vocab = {t: i for i, t in enumerate(tokens)}
        if model == "gpt2":
            pre = meta.get("tokenizer.ggml.pre", "default")
            if pre not in PRE:
                raise Unsupported(f"pre-tokenizer {pre!r}")
            merges = gguf.read_array(path, "tokenizer.ggml.merges")
            if not merges:
                raise Unsupported("the file has no merges")
            pairs = []
            for m in merges:
                a, _, b = m.partition(" ")
                pairs.append((a, b))
            tok = Tokenizer(models.BPE(vocab=vocab, merges=pairs, byte_fallback=False, fuse_unk=False))
            tok.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.Split(Regex(e), behavior="isolated", invert=False) for e in PRE[pre]]
                + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
            tok.decoder = decoders.ByteLevel()
        else:
            scores = gguf.read_array(path, "tokenizer.ggml.scores")
            if not scores or len(scores) != len(tokens):
                raise Unsupported("the SentencePiece vocabulary has no scores")
            if meta.get("tokenizer.ggml.add_space_prefix", True):
                raise Unsupported("SentencePiece with a dummy space prefix")
            unk = tokens[int(meta.get("tokenizer.ggml.unknown_token_id", 0))]
            tok = Tokenizer(models.BPE(vocab=vocab, merges=_spm_merges(tokens, scores, types), unk_token=unk,
                                       byte_fallback=True, fuse_unk=True))
            tok.normalizer = normalizers.Replace(" ", SPIECE)
            tok.decoder = decoders.Sequence([decoders.Replace(SPIECE, " "), decoders.ByteFallback(), decoders.Fuse()])
        specials = [AddedToken(tokens[i], special=True, normalized=False)
                    for i, t in enumerate(types) if t in (CONTROL, USER_DEFINED) and i < len(tokens)]
        if specials:
            tok.add_special_tokens(specials)
        self._tokenizer_obj = tok
        self.vocab_size = len(tokens)

        def tid(key):
            v = meta.get(key)
            return int(v) if isinstance(v, int) else None
        self.bos_id, self.eos_id = tid("tokenizer.ggml.bos_token_id"), tid("tokenizer.ggml.eos_token_id")
        self.bos_token = tokens[self.bos_id] if self.bos_id is not None and self.bos_id < len(tokens) else ""
        self.eos_token = tokens[self.eos_id] if self.eos_id is not None and self.eos_id < len(tokens) else ""
        ends = {self.eos_id, tid("tokenizer.ggml.eot_token_id"), tid("tokenizer.ggml.eom_token_id")}
        for name in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>", "<|eom_id|>", "<end_of_turn>"):
            if name in vocab:
                ends.add(vocab[name])
        self.eos_token_ids = sorted(e for e in ends if e is not None)

        self.chat_template = meta.get("tokenizer.chat_template") or ""
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: __import__("time").strftime(fmt)
        self._template = env.from_string(self.chat_template) if self.chat_template else None

        # tool calls: the Qwen/Hermes convention (JSON inside <tool_call> tags) when the template speaks it
        self.has_tool_calling = "<tool_call>" in self.chat_template
        self.tool_call_start, self.tool_call_end = ("<tool_call>", "</tool_call>") if self.has_tool_calling else (None, None)

    # ---- the tokenizer API
    def encode(self, text, add_special_tokens=False):
        return self._tokenizer_obj.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=False):
        return self._tokenizer_obj.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False, **kw):
        if self._template is None:
            raise ValueError("this GGUF has no chat template")
        text = self._template.render(messages=messages, tools=tools or None,
                                     add_generation_prompt=add_generation_prompt,
                                     bos_token=self.bos_token, eos_token=self.eos_token, **kw)
        return self._tokenizer_obj.encode(text, add_special_tokens=False).ids if tokenize else text

    def tool_parser(self, segment, tools=None):
        """`{"name": ..., "arguments": {...}}` JSON between the tags."""
        data = json.loads(segment)
        return data if isinstance(data, list) else [data]

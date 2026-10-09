# Tokenizer + chat template without transformers: `tokenizers` for BPE,
# a sandboxed jinja2 environment for the HF chat template.
import json
import os

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer as _Tok


class TemplateError(ValueError):
    pass


def _raise(msg):
    raise TemplateError(msg)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class Tokenizer:
    def __init__(self, path):
        self.tok = _Tok.from_file(os.path.join(path, "tokenizer.json"))
        cfg = {}
        try:
            with open(os.path.join(path, "tokenizer_config.json")) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            pass
        tpl = None
        p = os.path.join(path, "chat_template.jinja")
        if os.path.exists(p):
            with open(p) as f:
                tpl = f.read()
        tpl = tpl or cfg.get("chat_template")
        if not tpl:
            raise ValueError("model has no chat template")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(tpl)
        self.eos_token = cfg.get("eos_token")
        self.template_source = tpl

    def token_id(self, s):
        return self.tok.token_to_id(s)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def render(self, messages, tools=None, **kwargs):
        return self.template.render(messages=messages, tools=tools or None,
                                    add_generation_prompt=True, **kwargs)


class StreamDecoder:
    """Incremental detokenization that never splits a multi-byte character."""

    def __init__(self, tok):
        self.tok, self.ids, self.prefix, self.read = tok, [], 0, 0

    def add(self, token):
        self.ids.append(token)
        prev = self.tok.decode(self.ids[self.prefix:self.read])
        cur = self.tok.decode(self.ids[self.prefix:])
        if cur.endswith("�") or len(cur) <= len(prev):
            return ""
        self.prefix, self.read = self.read, len(self.ids)
        return cur[len(prev):]

    def flush(self):
        prev = self.tok.decode(self.ids[self.prefix:self.read])
        cur = self.tok.decode(self.ids[self.prefix:])
        self.prefix = self.read = len(self.ids)
        return cur[len(prev):]

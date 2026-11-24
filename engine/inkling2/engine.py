"""The mlx-lm backend (engine/compat.py) for Inkling 2: its weights load through mlx-lm (registered by
inkling2.register(), the experts streamed from the SSD when the model does not fit); what differs is the chat
format — reasoning is set by the template's `reasoning_effort` ("none" turns it off) and the reply is written in
segments that chat.StreamFilter rewrites for the server's parser."""
import math
import os

from ..compat import CompatEngine, MlxLmRuntime, _json
from . import chat

IMAGE_TOKEN = "<|unused_200054|>"         # the template writes one per image; it becomes one per 40×40 block
IMAGE_MEAN = IMAGE_STD = 0.5


class InklingRuntime(MlxLmRuntime):
    """MlxLmRuntime plus image prompts: the embeddings with the vision tower's rows in place of the image tokens."""
    accepts_embeds = True

    def prefill(self, cache, ids, embeds, pos3, delta, s, e, hidden=False):
        import mlx.core as mx
        if embeds is None:
            return super().prefill(cache, ids, embeds, pos3, delta, s, e, hidden)
        logits = self.model(mx.array([ids[s:e]]), cache=cache, input_embeddings=embeds[:, s:e])
        mx.eval([c.state for c in cache])
        return logits

    def embed_media(self, ids, media):
        import mlx.core as mx
        emb = self.model.embed_inputs(mx.array(ids), pixel_values=media)
        return emb[None], None, 0


def preprocess(image, block, max_pixels):
    """(pixels (3, H', W') padded to whole blocks, number of blocks) — the PR's processor; an image above
    `max_pixels` is scaled down first so one picture cannot take the context or the memory."""
    import mlx.core as mx
    import numpy as np
    w, h = image.size
    if max_pixels and w * h > max_pixels:
        f = math.sqrt(max_pixels / (w * h))
        image = image.resize((max(block, int(w * f)), max(block, int(h * f))))
        w, h = image.size
    rows, cols = math.ceil(h / block), math.ceil(w / block)
    x = np.full((rows * block, cols * block, 3), -1.0, np.float32)    # the padding is -1 on the 0–255 scale
    x[:h, :w] = np.asarray(image.convert("RGB"), np.float32)
    x = (x / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return mx.array(x.transpose(2, 0, 1)), rows * cols


def handles(path):
    """True for an MLX folder of Inkling 2."""
    if not os.path.isdir(path):
        return False
    cfg = _json(os.path.join(path, "config.json"))
    return str(cfg.get("model_type", "")).startswith("inkling2")


class InklingEngine(CompatEngine):
    def __init__(self, path, **kw):
        super().__init__(path, **kw)
        vision = self.config.get("vision_config") or {}
        self.vision_capable = bool(vision) and getattr(self.model.model, "vision_tower", None) is not None
        self.vision_allowed = self.vision_capable and bool(kw.get("vision", True))
        self.image_block = vision.get("native_patch_size", 20) * vision.get("adapter_merge_size", 2)
        self.max_pixels = kw.get("max_pixels") or 1024 * 1024
        # the template thinks unless told reasoning_effort "none" (its default effort is 0.9)
        self.supports_thinking = True
        self.thinking_default = True
        if kw.get("thinking") is None:
            self.thinking = True
        self.efforts = ["low", "medium", "high", "xhigh"]

    def build_prompt(self, messages, tools=None, thinking=None, effort=None, extra_kwargs=None):
        thinking = self.thinking if thinking is None else bool(thinking)
        kw = dict(extra_kwargs or {})
        kw.pop("enable_thinking", None)
        kw["reasoning_effort"] = chat.effort_for(thinking, effort or self.effort)
        images, flat = [], []
        for m in messages:
            m = dict(m)
            if isinstance(m.get("content"), list):
                parts = []                          # the template places each image where it was in the message
                for it in m["content"]:
                    if isinstance(it, dict) and it.get("type") in ("text", "input_text"):
                        parts.append({"type": "text", "text": it.get("text", "")})
                    elif isinstance(it, dict) and it.get("type") in ("image_url", "image", "input_image"):
                        images.append(it)
                        parts.append({"type": "image"})
                m["content"] = parts if images else "".join(p.get("text", "") for p in parts)
            flat.append(m)
        text = self._render(flat, tools, kw)
        # the model opens its reply itself (<|message_model|><|content_thinking|>…); the parser starts in
        # reasoning mode when thinking is on, and the filter closes it if the reply skips its reasoning
        return text, images, thinking

    def _attach_runtime(self):
        self.rt = InklingRuntime(self.model, self._cache_api())

    def encode(self, text, images):
        if not images:
            return super().encode(text, images)
        if not self.vision_allowed:
            raise ValueError("this model/server has no vision support")
        self.ensure_awake()
        from ..vision import load_image
        pixels, counts = [], []
        for ref in images:
            p, n = preprocess(load_image(ref), self.image_block, self.max_pixels)
            pixels.append(p)
            counts.append(n)
        parts = text.split(IMAGE_TOKEN)
        if len(parts) - 1 != len(images):
            raise ValueError(f"the prompt has {len(parts) - 1} image places for {len(images)} images")
        text = parts[0] + "".join(IMAGE_TOKEN * n + part for n, part in zip(counts, parts[1:]))
        return list(self.tok.encode(text, add_special_tokens=False)), pixels

    def _think_closing(self):
        """Reasoning ends with its segment's <|end_message|> — what the budget teacher-forces too."""
        return chat.END, list(self.tok.encode(chat.END, add_special_tokens=False))

    def parser_markers(self):
        return {"think_end": "</think>", "tool_start": "<tool_call>", "tool_end": "</tool_call>"}

    def parse_tool_calls(self, text, tools):
        return chat.parse_tool_calls(text, tools)

    def output_filter(self, thinking_open):
        return chat.StreamFilter(thinking_open)

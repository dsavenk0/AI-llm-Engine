# Copyright © 2026 Apple Inc. — ported from ml-explore/mlx-lm pull request #2086 (MIT License, see NOTICE).

from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

from . import checkpoint
from .config import ModelArgs, TextArgs
from .switch import splice
from .layers import (
    BlockAttentionResidual,
    DecoderLayer,
    FullDecoderLayer,
    MessageInput,
    ShortConvolution,
    SlidingDecoderLayer,
    unweighted_rms_norm,
)
from .vision import VisionModel

__all__ = [
    "Inkling2Model",
    "Model",
    "TextModel",
]


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        # Not in embed_tokens as in transformers, so that it can be quantized.
        self.embed_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [
            (
                SlidingDecoderLayer(args, i)
                if i in args.local_layer_ids
                else FullDecoderLayer(args, i)
            )
            for i in range(args.num_hidden_layers)
        ]
        self.final_input = BlockAttentionResidual(args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.loop_config is not None:
            self.message_input = MessageInput(args.hidden_size, len(args.loop_core))

    def embed(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(self.embed_tokens(inputs))

    @property
    def layer_visits(self) -> list[DecoderLayer]:
        if self.args.loop_config is None:
            return self.layers
        core = self.args.loop_core
        passes = self.args.loop_config["num_passes"]
        layers = self.layers
        return (
            layers[: core.start]
            + layers[core.start : core.stop] * passes
            + layers[core.stop :]
        )

    def _run_loop(self, hidden_states, cache, conv_mask):
        core = self.args.loop_core
        caches = iter(cache)

        def run(layers, hidden_states, block_states, bank=None):
            for i, layer in enumerate(layers):
                messages = None if bank is None else self.message_input(bank, i)
                hidden_states, block_states = layer(
                    hidden_states,
                    block_states,
                    cache=next(caches),
                    conv_mask=conv_mask,
                    messages=messages,
                )
            return hidden_states, block_states

        prelude = run(self.layers[: core.start], hidden_states, None)
        hidden_states, block_states = prelude
        for loop_pass in range(self.args.loop_config["num_passes"]):
            bank = None
            if loop_pass > 0:
                bank = (
                    block_states[prelude[1].shape[0] + 1 :],
                    unweighted_rms_norm(hidden_states.astype(mx.float32)),
                )
                hidden_states, block_states = prelude
            hidden_states, block_states = run(
                self.layers[core.start : core.stop], hidden_states, block_states, bank
            )
        return run(self.layers[core.stop :], hidden_states, block_states)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is None:
            input_embeddings = self.embed(inputs)
        hidden_states = input_embeddings
        dtype = hidden_states.dtype

        if cache is None:
            cache = [None] * len(self.layer_visits)
        conv_mask = None
        if cache[0] is not None:
            conv_mask = cache[0][1].make_mask(hidden_states.shape[1])

        block_states = None
        if self.args.loop_config is None:
            for layer, c in zip(self.layers, cache):
                hidden_states, block_states = layer(
                    hidden_states, block_states, cache=c, conv_mask=conv_mask
                )
        else:
            hidden_states, block_states = self._run_loop(
                hidden_states, cache, conv_mask
            )

        normed_states = unweighted_rms_norm(hidden_states.astype(mx.float32))
        hidden_states = self.final_input(block_states, normed_states)[..., 0, :]
        return self.norm(hidden_states).astype(dtype)


class Inkling2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = TextModel(args.text)
        if args.vision is not None:
            self.vision_tower = VisionModel(args.vision, args.text.hidden_size)

    @property
    def layers(self) -> list[DecoderLayer | None]:
        return self.language_model.layers


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Inkling2Model(args)
        text = args.text
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def encode_image(self, pixel_values: mx.array | list[mx.array]) -> mx.array:
        """pixel_values: one image (num_channels, H, W) or a list of images of
        any size. Returns the rows of the image tokens of all images."""
        if isinstance(pixel_values, mx.array) and pixel_values.ndim == 3:
            pixel_values = [pixel_values]
        assert len(pixel_values), "No images."
        return mx.concatenate([self.model.vision_tower(p) for p in pixel_values])

    def embed_inputs(
        self,
        inputs: mx.array,
        *,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        inputs_embeds = self.model.language_model.embed(inputs)
        if pixel_values is not None:
            image_features = self.encode_image(pixel_values)
            inputs_embeds = splice(
                inputs_embeds,
                inputs,
                rows=image_features,
                token_id=self.args.image_token_id,
            )
        return inputs_embeds

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        if pixel_values is not None:
            assert input_embeddings is None, "Pass pixel_values or input_embeddings."
            input_embeddings = self.embed_inputs(inputs, pixel_values=pixel_values)
        hidden_states = self.model.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )
        logits = self.lm_head(hidden_states)
        text = self.args.text
        softcap = text.final_logit_softcapping
        logits = logits.astype(mx.float32)
        logits = mx.tanh(logits / softcap) * softcap
        return logits[..., : text.unpadded_vocab_size]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]:

        def predicate(path: str, module: nn.Module) -> bool:
            return not path.startswith("model.vision_tower.") and (
                "_message_gate." not in path
            )

        return predicate

    @property
    def cast_predicate(self) -> Callable[[str], bool]:

        def predicate(k: str) -> bool:
            return not k.endswith(
                (
                    ".query",
                    ".mlp.gate.weight",
                    ".mlp.gate.e_score_correction_bias",
                    ".forget_gate_proj.weight",
                    ".message_gate_proj.weight",
                )
            )

        return predicate

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.language_model.layers

    def make_cache(self) -> list[CacheList]:
        window = self.args.text.sliding_window_size
        return [
            CacheList(
                (
                    RotatingKVCache(max_size=window)
                    if isinstance(layer, SlidingDecoderLayer)
                    else KVCache()
                ),
                ArraysCache(4),
            )
            for layer in self.model.language_model.layer_visits
        ]

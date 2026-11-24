"""Inkling 2 (Thinking Machines; model_type "inkling2_loop") on MLX — the looped mixture-of-experts language model
and its vision tower, ported from ml-explore/mlx-lm pull request #2086 (not in a released mlx-lm yet).

mlx-lm loads a model by importing `mlx_lm.models.<model_type>`; `register()` puts this package there, so the
engine's ordinary MLX path (mlx_lm.load, and engine/moe_stream.py when the experts must stream from the SSD)
loads Inkling 2 like any other architecture. What mlx-lm main has and 0.32 lacks is in switch.py; the Metal
kernels (relative attention for prompts, depth attention for every layer) are the PR's.

https://huggingface.co/thinkingmachines/Inkling-2-Mini
"""
import sys

from .config import ModelArgs
from .model import Model

MODEL_TYPES = ("inkling2_loop", "inkling2")

__all__ = ["Model", "ModelArgs", "register", "MODEL_TYPES"]


def register():
    """Make `mlx_lm.models.inkling2_loop` (and `inkling2`) this package, unless mlx-lm has its own already."""
    import importlib
    for name in MODEL_TYPES:
        key = f"mlx_lm.models.{name}"
        if key in sys.modules:
            continue
        try:
            importlib.import_module(key)          # a newer mlx-lm that ships it wins
        except ImportError:
            sys.modules[key] = sys.modules[__name__]

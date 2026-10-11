"""Which models the native engine can run — decided from what the config contains, not from a list of
type names (stdlib only: llm.py uses it without MLX installed).

The native engine implements one architecture: the Qwen3.5 hybrid, i.e. dense blocks that mix
GatedDeltaNet (linear attention with a short conv) and full attention. A checkpoint is native when its
text config carries that architecture's fields — whatever its `model_type` is called (qwen3_5,
prism_hadamard_qwen35, ternel_hadamard_qwen35, a future qwen3_6 …) — and it is not mixture-of-experts.
Anything else goes to the optional mlx-lm backend. Loading still verifies every tensor's shape, so a
look-alike that differs in detail fails loudly instead of running wrong.
"""
import re

# the fields the hybrid block needs: GatedDeltaNet heads/dims/conv + ordinary attention
HYBRID_FIELDS = ("linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
                 "linear_value_head_dim", "linear_conv_kernel_dim", "num_attention_heads", "hidden_size")
# for a hub repo that has not been downloaded only its tags are known: names of the family
FAMILY_NAME = re.compile(r"^(qwen3[_.]?[5-9]\d*(_text)?|prism_hadamard_qwen3\d*|ternel_hadamard_qwen3\d*)$")


def is_moe(cfg, tags=()):
    text = cfg.get("text_config") or {}
    kinds = [cfg.get("model_type"), text.get("model_type"), *tags]
    return bool(text.get("num_experts") or cfg.get("num_experts") or any("moe" in str(k).lower() for k in kinds))


def native_capable(cfg, tags=()):
    """True when the native engine runs this model. `cfg` is config.json (may be {} for a hub repo)."""
    if is_moe(cfg, tags):
        return False
    text = cfg.get("text_config") or cfg
    if all(text.get(k) for k in HYBRID_FIELDS):
        return True
    names = {cfg.get("model_type"), text.get("model_type"), *tags}
    return any(FAMILY_NAME.match(str(n)) for n in names if n)

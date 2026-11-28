"""The Hadamard rotation contract of a checkpoint, from wherever its format declares it (stdlib only: llm.py
reads it to say whether a model will run before MLX is even imported).

Rotated checkpoints (Prism / JANG Bonsai and their repacks) store each weight matrix W as W·D·H — H a blockwise
normalized Walsh-Hadamard transform, D a ±1 sign vector per input width — and the runtime applies H·D to the
activation going in, so the product is unchanged. The token embedding is stored rotated the same way and
transformed back after the lookup. Get any of that wrong and the model still runs and answers fluent nonsense,
so a contract this module cannot read fully is an error, never a guess.

Formats declare the same contract in different places:

    config.json "hadamard": {"block_size": …}         Prism / JANG MLX bundles (schema 1); jang_config.json too
    config.json "hadamard_config": "hadamard.json"    Prism MLX packs, schema 2: `prism.hadamard.*` keys in a file
    GGUF metadata `prism.hadamard.*`                   Prism GGUF (Q2_0 / PQ2_0 / PTQ1_0)

All of them come out as one `Contract`. A new format that declares the same thing somewhere else needs a reader
here, and nothing in the engines.
"""
import json
import os

PRISM = "prism.hadamard."
TRANSFORM = "normalized-sylvester-walsh-hadamard"
AXIS = "input-last-dimension"


class Unsupported(ValueError):
    """The checkpoint declares a rotation this engine cannot reproduce exactly."""


class Contract:
    """block: transform size; signs: {width: [±1, …]} when the declaration carries them (else they ship as
    `.signs` tensors); folded / inverse: the weight names stored rotated / inverse-rotated, or None when the
    format leaves it implicit (schema-1 bundles: every projection and the head, inverse on the embedding);
    gdn_v_grouped: the linear-attention output projection reads value heads grouped by key head."""

    def __init__(self, block, signs=None, folded=None, inverse=None, gdn_v_grouped=None, source=""):
        self.block, self.signs, self.folded, self.inverse = block, signs, folded, inverse
        self.gdn_v_grouped, self.source = gdn_v_grouped, source

    def __repr__(self):
        n = "implicit" if self.folded is None else len(self.folded)
        return f"Contract(block={self.block}, folded={n}, source={self.source!r})"


def _block(v):
    try:
        b = int(v)
    except (TypeError, ValueError):
        raise Unsupported(f"Hadamard block size {v!r} is not a number")
    if b < 2 or b & (b - 1):
        raise Unsupported(f"Hadamard block size {b} is not a power of two")
    return b


def from_prism(get, source):
    """A `prism.hadamard.*` declaration (`get(key)` → value or None, the key without the prefix)."""
    version = get("version")
    if version != 1:
        raise Unsupported(f"Prism Hadamard contract version {version!r} (this engine reads version 1)")
    for key, want in (("transform", TRANSFORM), ("axis", AXIS), ("sign_mode", "explicit")):
        if get(key) != want:
            raise Unsupported(f"Hadamard {key} {get(key)!r} (this engine applies {want!r})")
    block = _block(get("block_size"))
    widths, values = get("sign_widths"), get("sign_values")
    signs = None
    if widths is not None or values is not None:
        widths, values = list(widths or []), list(values or [])
        if sum(int(w) for w in widths) != len(values):
            raise Unsupported(f"Hadamard sign values ({len(values)}) do not add up to the widths {widths}")
        signs, at = {}, 0
        for w in widths:
            w = int(w)
            vec = values[at:at + w]
            at += w
            if w % block:
                raise Unsupported(f"Hadamard block {block} does not divide the width {w}")
            if any(v not in (1, -1, 1.0, -1.0) for v in vec):
                raise Unsupported(f"Hadamard signs of width {w} are not all ±1")
            if w in signs:
                raise Unsupported(f"two Hadamard sign vectors for width {w}")
            signs[w] = [float(v) for v in vec]
    folded = get("weight_names")
    inverse = get("inverse_weight_names")
    if folded is None:
        raise Unsupported("the Hadamard contract does not list which weights are rotated")
    folded, inverse = set(folded), set(inverse or ())
    if folded & inverse:
        raise Unsupported(f"weights listed as both rotated and inverse-rotated: {sorted(folded & inverse)[:3]}")
    grouped = get("gdn_v_grouped")
    return Contract(block, signs, folded, inverse, None if grouped is None else bool(grouped), source)


def bundle(path, config, jang=None):
    """The contract of a safetensors bundle, None when it declares no rotation. Raises Unsupported."""
    jang = jang or {}
    legacy = config.get("hadamard") or jang.get("hadamard")
    if legacy:
        return Contract(_block(legacy.get("block_size", 1024)), source="config")
    ref = config.get("hadamard_config")
    if not ref:
        return None
    if not isinstance(ref, str) or os.path.basename(ref) != ref:
        raise Unsupported(f"hadamard_config {ref!r} is not a file in the model folder")
    try:
        with open(os.path.join(path, ref)) as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        raise Unsupported(f"{ref} unreadable: {e}")
    layout = config.get("gdn_activation_layout")
    if layout not in (None, "grouped"):
        raise Unsupported(f"linear-attention activation layout {layout!r} (this engine runs 'grouped')")
    c = from_prism(lambda k: meta.get(PRISM + k), ref)
    if c.gdn_v_grouped is False:
        raise Unsupported("ungrouped linear-attention output in a rotated bundle")
    return c


def native_set(text_config, layer_range=None):
    """(rotated, inverse-rotated) weight names that engine/model.py applies the transform to, for this config:
    every attention / linear-attention / MLP projection and the head; the embedding inverse. The a/b gate
    projections take the unrotated input."""
    n = int(text_config["num_hidden_layers"])
    every = int(text_config.get("full_attention_interval", 4))
    kinds = text_config.get("layer_types") or ["full_attention" if (i + 1) % every == 0 else "linear_attention"
                                                for i in range(n)]
    pre = "language_model.model.layers."
    out = {"language_model.lm_head.weight"}
    for i, kind in enumerate(kinds):
        if kind == "full_attention":
            names = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
        else:
            names = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
        names += ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
        out.update(f"{pre}{i}.{m}.weight" for m in names)
    return out, {"language_model.model.embed_tokens.weight"}


def _canonical(name):
    if name.startswith(("model.", "lm_head.")) and not name.startswith("model.visual"):
        return "language_model." + name
    return name


def native_mismatch(contract, text_config):
    """"" when the native engine applies exactly the declared rotations, else what differs. A bundle that rotates
    one matrix more or less than the engine would run — and answer nonsense."""
    if contract is None or contract.folded is None:
        return ""
    folded = {_canonical(k) for k in contract.folded}
    inverse = {_canonical(k) for k in contract.inverse}
    want_f, want_i = native_set(text_config)
    for what, got, want in (("rotated", folded, want_f), ("inverse-rotated", inverse, want_i)):
        extra, missing = sorted(got - want), sorted(want - got)
        if extra or missing:
            detail = (f"also {extra[0]}" if extra else f"not {missing[0]}")
            return f"the bundle's {what} weights differ from the engine's ({detail})"
    return ""


def from_gguf(meta, read_array):
    """The contract of a GGUF (`meta`: its scalar header; `read_array(key)` → list or None), None when the file
    declares no rotation. Raises Unsupported."""
    if not any(k.startswith(PRISM) for k in meta):
        return None
    arrays = ("sign_widths", "sign_values", "weight_names", "inverse_weight_names")

    def get(k):
        v = meta.get(PRISM + k)                 # the header keeps small numeric arrays; long ones are read here
        return read_array(PRISM + k) if v is None and k in arrays else v
    c = from_prism(get, "gguf")
    if c.signs is None:
        raise Unsupported("the GGUF declares a Hadamard rotation without its sign vectors")
    bad = sorted(c.inverse - {"token_embd.weight"})
    if bad:
        raise Unsupported(f"inverse rotation on {bad[0]} (this engine applies it to the token embedding only)")
    return c

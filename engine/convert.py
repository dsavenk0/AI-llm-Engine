"""Convert a GGUF file into an MLX model folder (config.json + f16 safetensors + tokenizer.json).

The engine already repacks GGUF weights when it loads them; this writes them out as a plain MLX folder so a
converted model can be opened by mlx-lm (or inspected / shared) without the GGUF around. Weights are
dequantized to float16 — reusing the decoder's own unpacking, so the values are exactly what the engine runs —
and the GGUF tensor names and metadata are mapped to the Hugging Face layout the MLX loaders expect.

Scope: the standard decoder-only families with HF naming — llama (Llama/Mistral/SmolLM2), qwen2, qwen3. Other
architectures (Gemma's sandwich norms, MoE) raise `Unconvertible` for now; the weights still decode in the
engine, this is only the folder export.
"""
import json
import os
import re

import mlx.core as mx

from . import gguf_tensors as T

SUPPORTED = ("llama", "qwen2", "qwen3")


class Unconvertible(ValueError):
    pass


# GGUF per-layer tensor name -> Hugging Face name (reverse of llama.cpp's conversion)
_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias", "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
_TOP = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight"}
_BLK = re.compile(r"blk\.(\d+)\.(.+)")


def hf_name(gguf_name):
    """The Hugging Face name for a GGUF tensor, or None for one we do not map."""
    if gguf_name in _TOP:
        return _TOP[gguf_name]
    m = _BLK.match(gguf_name)
    if m and m.group(2) in _LAYER:
        return f"model.layers.{m.group(1)}.{_LAYER[m.group(2)]}"
    return None


def tensor_f16(gf, info):
    """One GGUF tensor as a float16 MLX array — dequantized exactly as the engine does."""
    if info.type_name in ("F32", "F16", "BF16"):
        return mx.array(T.load_dense(gf, info)).astype(mx.float16)
    w, s, b, bits = T.load_matrix(gf, info)
    return mx.dequantize(mx.array(w), mx.array(s), mx.array(b), group_size=32, bits=bits).astype(mx.float16)


def build_config(gf, vocab):
    m = gf.meta
    arch = m.get("general.architecture")
    g = lambda key, default=None: m.get(f"{arch}.{key}", default)
    n_head = int(g("attention.head_count"))
    n_embd = int(g("embedding_length"))
    head_dim = int(g("attention.key_length", n_embd // n_head))
    cfg = {
        "model_type": arch,
        "architectures": [{"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM",
                           "qwen3": "Qwen3ForCausalLM"}[arch]],
        "hidden_size": n_embd,
        "num_hidden_layers": int(g("block_count")),
        "num_attention_heads": n_head,
        "num_key_value_heads": int(g("attention.head_count_kv", n_head)),
        "intermediate_size": int(g("feed_forward_length")),
        "vocab_size": vocab,
        "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon", 1e-5)),
        "rope_theta": float(g("rope.freq_base", 10000.0)),
        "max_position_embeddings": int(g("context_length", 4096)),
        "head_dim": head_dim,
        "tie_word_embeddings": "output.weight" not in gf.tensors,
        "torch_dtype": "float16",
    }
    if g("attention.head_count_kv", n_head) != n_head or arch in ("qwen2", "qwen3"):
        cfg["num_key_value_heads"] = int(g("attention.head_count_kv", n_head))
    return cfg


def write_tokenizer(gf, path, out_dir):
    """A tokenizer.json rebuilt from the GGUF's embedded BPE vocabulary and merges, plus tokenizer_config.json."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from . import gguf as _g
    tokens = _g.read_array(path, "tokenizer.ggml.tokens")
    merges = _g.read_array(path, "tokenizer.ggml.merges")
    if not tokens or not merges:
        raise Unconvertible("only a byte-level BPE tokenizer is exported so far (no tokens/merges found)")
    vocab = {t: i for i, t in enumerate(tokens)}
    pairs = [tuple(mrg.split(" ", 1)) for mrg in merges if " " in mrg]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=pairs, fuse_unk=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    meta = gf.meta
    tcfg = {"tokenizer_class": "PreTrainedTokenizerFast"}
    for key, field in (("bos_token_id", "bos"), ("eos_token_id", "eos")):
        tid = meta.get(f"tokenizer.ggml.{key}")
        if isinstance(tid, int) and 0 <= tid < len(tokens):
            tcfg[field + "_token"] = tokens[tid]
    if "tokenizer.chat_template" in meta:
        tcfg["chat_template"] = meta["tokenizer.chat_template"]
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tcfg, f, ensure_ascii=False, indent=2)
    return len(tokens)


def convert(gguf_path, out_dir, log=print):
    """Write `gguf_path` as an MLX folder in `out_dir`. Returns the folder. Raises Unconvertible."""
    gf = T.GGUFFile(gguf_path)
    arch = gf.meta.get("general.architecture")
    if arch not in SUPPORTED:
        raise Unconvertible(f"architecture {arch!r} is not exported yet (supported: {', '.join(SUPPORTED)})")
    os.makedirs(out_dir, exist_ok=True)
    weights, skipped = {}, []
    for name, info in gf.tensors.items():
        hf = hf_name(name)
        if hf is None:
            skipped.append(name)
            continue
        weights[hf] = tensor_f16(gf, info)
    if skipped:
        raise Unconvertible(f"unmapped tensor(s): {', '.join(sorted(skipped)[:5])}"
                            + (" …" if len(skipped) > 5 else ""))
    vocab = int(gf.tensors["token_embd.weight"].shape[0])
    mx.eval(list(weights.values()))
    mx.save_safetensors(os.path.join(out_dir, "model.safetensors"), weights, metadata={"format": "mlx"})
    cfg = build_config(gf, vocab)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    ntok = write_tokenizer(gf, gguf_path, out_dir)
    log(f"converted {os.path.basename(gguf_path)} → {out_dir}: {len(weights)} tensors, {ntok} tokens, "
        f"{cfg['num_hidden_layers']} layers")
    return out_dir


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Convert a GGUF file into an MLX model folder")
    ap.add_argument("gguf")
    ap.add_argument("out", nargs="?", default="")
    a = ap.parse_args(argv)
    out = a.out or os.path.splitext(a.gguf)[0] + "-mlx"
    try:
        convert(a.gguf, out)
    except (Unconvertible, T.UnsupportedQuant, OSError, ValueError) as e:
        print(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

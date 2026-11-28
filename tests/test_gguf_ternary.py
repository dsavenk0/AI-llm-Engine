"""Prism Bonsai GGUF on the engine's own decoder (engine/dense.py): ternary weights (Q2_0, PQ2_0, PTQ1_0) in a
blockwise Hadamard basis, against mlx-lm's Qwen3.5 run on the same weights rotated back.

A tiny Qwen3.5 gets random weights; every projection, the head and the embedding are rotated (W·D·H, the
`prism.hadamard.*` contract) and quantized to ternary blocks, and the GGUF is written the way Prism's are: value
heads in ggml's tiled order everywhere except the output projection, whose rotated columns stay grouped by key
head (`gdn_v_grouped`). The reference model gets the dequantized weights rotated back, so both compute the same
function and must give the same logits.

    .venv/bin/python -m unittest tests.test_gguf_ternary -v
"""
import os
import struct
import tempfile
import unittest

import numpy as np

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
except ImportError:                                       # pragma: no cover
    mx = None

from tests import gguf_synth as S
from tests import test_gguf_qwen35 as Q

D, FF, HEADS, KV, HD = 128, 256, 4, 2, 32
BLOCK = 64                                                # the Hadamard block: divides every rotated width
TYPES = {"Q2_0": (42, 64), "PQ2_0": (142, 128), "PTQ1_0": (143, 128)}


def hadamard(x, signs, inverse=False):
    """The normalized blockwise Walsh-Hadamard of the rows of x (numpy, float64), signs first (or last)."""
    n = BLOCK
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    h /= np.sqrt(n)
    rows, width = x.shape
    if not inverse:
        x = x * signs
    y = (x.reshape(rows, width // n, n) @ h).reshape(rows, width)
    return y * signs if inverse else y


def ternary(s, block, rng):
    """Rows of s → (codes 0..2 [rows, cols], d float16 [rows, cols/block]): the nearest ternary grid per block."""
    rows, cols = s.shape
    g = s.reshape(rows, cols // block, block)
    d = (np.abs(g).mean(-1) * 1.4 + 1e-3).astype(np.float16)
    codes = np.clip(np.rint(g / d.astype(np.float32)[..., None]), -1, 1).astype(np.int64) + 1
    return codes.reshape(rows, cols).astype(np.uint8), d


def _trit_byte(digits):
    """Trits (most significant first, 5 or 4 of them) → the PTQ1_0 byte: ceil(v * 256 / 3^n) as ggml packs it."""
    v = 0
    for t in digits:
        v = v * 3 + int(t)
    return -(-v * 256 // 3 ** len(digits))


def encode(kind, codes, d):
    """codes [rows, cols] + d [rows, blocks] → the GGUF bytes of a ternary tensor."""
    rows, cols = codes.shape
    _, block = TYPES[kind]
    out = bytearray()
    for r in range(rows):
        for b in range(cols // block):
            c = codes[r, b * block:(b + 1) * block]
            scale = np.float16(d[r, b]).tobytes()
            if kind in ("Q2_0", "PQ2_0"):
                packed = bytes(int(c[4 * j] | (c[4 * j + 1] << 2) | (c[4 * j + 2] << 4) | (c[4 * j + 3] << 6))
                               for j in range(block // 4))
                out += scale + packed
            else:
                qs = [_trit_byte([c[n * 16 + j] for n in range(5)]) for j in range(16)]
                qs += [_trit_byte([c[80 + n * 8 + j] for n in range(5)]) for j in range(8)]
                qh = [_trit_byte([c[120 + n * 2 + j] for n in range(4)]) for j in range(2)]
                out += bytes(qs + qh) + scale
    return bytes(out)


def build(kind, seed=0):
    """(mlx-lm reference model, {gguf name: (ggml type, dims, bytes)}, metadata) for one ternary type."""
    try:
        from mlx_lm.models import qwen3_5
    except ImportError:
        raise unittest.SkipTest("mlx-lm is not installed: the rotated ternary GGUF was not compared with it")
    tokens, _, _ = S.vocab()
    tc = Q.text_config(moe=False)
    tc.update(hidden_size=D, intermediate_size=FF, num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HD,
              vocab_size=len(tokens))
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": tc}))
    rng = np.random.default_rng(seed)
    signs = {w: rng.choice([-1.0, 1.0], size=w) for w in (D, FF)}
    gtype, block = TYPES[kind]
    p, stored = {}, {}
    for name, v in tree_flatten(model.parameters()):
        shape = v.shape
        if name.endswith("norm.weight"):
            a = 1.0 + rng.standard_normal(shape) * 0.1
        elif name.endswith("A_log"):
            a = rng.standard_normal(shape) * 0.5
        elif name.endswith("dt_bias"):
            a = 1.0 + rng.standard_normal(shape) * 0.2
        else:
            a = rng.standard_normal(shape) * (0.5 if "embed" in name else 0.08)
        a = a.astype(np.float16).astype(np.float32)
        rotated = len(shape) == 2 and not name.endswith(("in_proj_a.weight", "in_proj_b.weight"))
        if rotated:                                   # stored rotated and ternary; the reference gets it back
            codes, d = ternary(hadamard(a.astype(np.float64), signs[shape[1]]), block, rng)
            s_q = (codes.astype(np.float64) - 1) * np.repeat(d.astype(np.float64), block, axis=1)
            a = hadamard(s_q, signs[shape[1]], inverse=True).astype(np.float32)
            stored[name] = (codes, d)
        p[name] = a
    model.load_weights([(k, mx.array(v)) for k, v in p.items()])
    mx.eval(model.parameters())

    gg = Q.to_gguf_tensors(p, moe=False)              # names + ggml's tiled value-head order
    hf = {"token_embd.weight": "language_model.model.embed_tokens.weight", "output.weight": "language_model.lm_head.weight"}
    for i in range(Q.LAYERS):
        L, b = f"language_model.model.layers.{i}.", f"blk.{i}."
        pairs = [("ffn_gate", "mlp.gate_proj"), ("ffn_up", "mlp.up_proj"), ("ffn_down", "mlp.down_proj")]
        pairs += ([("attn_qkv", "linear_attn.in_proj_qkv"), ("attn_gate", "linear_attn.in_proj_z"),
                   ("ssm_out", "linear_attn.out_proj")] if (i + 1) % 4 else
                  [("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj"),
                   ("attn_output", "self_attn.o_proj")])
        hf.update({f"{b}{g}.weight": f"{L}{m}.weight" for g, m in pairs})
    key_dim = Q.NK * Q.DK
    tensors = {}
    for name, a in gg.items():
        if name not in hf:
            tensors[name] = (0, tuple(reversed(a.shape)), a.astype(np.float32).tobytes())
            continue
        codes, d = stored[hf[name]]
        # rows move with the value heads exactly as the float tensor's did; the output projection's rotated
        # columns cannot move, so it stays grouped — what `gdn_v_grouped` declares
        if name.endswith(("attn_qkv.weight", "attn_gate.weight")):
            head_rows = Q._reorder_v(np.arange(codes.shape[0] - (key_dim * 2 if "qkv" in name else 0))[:, None],
                                     0, Q.DK)[:, 0]
            order = np.concatenate([np.arange(2 * key_dim), 2 * key_dim + head_rows]) if "qkv" in name else head_rows
            codes, d = codes[order], d[order]
        tensors[name] = (gtype, tuple(reversed(codes.shape)), encode(kind, codes, d))
    widths = sorted(signs)
    meta = {"prism.hadamard.version": 1, "prism.hadamard.block_size": BLOCK,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension", "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.sign_widths": widths,
            "prism.hadamard.sign_values": [float(v) for w in widths for v in signs[w]],
            "prism.hadamard.weight_names": sorted(n for n in hf if n != "token_embd.weight"),
            "prism.hadamard.inverse_weight_names": ["token_embd.weight"], "prism.hadamard.gdn_v_grouped": True}
    return model, tensors, meta


def write(path, tensors, extra):
    arch = "qwen35"
    tokens, types, merges = S.vocab()
    meta = {
        "general.architecture": arch, "general.name": "tiny-bonsai", "general.file_type": 1,
        f"{arch}.context_length": 512, f"{arch}.embedding_length": D, f"{arch}.block_count": Q.LAYERS,
        f"{arch}.feed_forward_length": FF, f"{arch}.attention.head_count": HEADS, f"{arch}.attention.head_count_kv": KV,
        f"{arch}.attention.key_length": HD, f"{arch}.attention.value_length": HD,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-6, f"{arch}.rope.freq_base": 10000.0,
        f"{arch}.rope.dimension_count": HD // 4, f"{arch}.rope.dimension_sections": [2, 1, 1, 0],
        f"{arch}.ssm.conv_kernel": Q.CONV, f"{arch}.ssm.state_size": Q.DK, f"{arch}.ssm.group_count": Q.NK,
        f"{arch}.ssm.time_step_rank": Q.NV, f"{arch}.ssm.inner_size": Q.NV * Q.DK, f"{arch}.full_attention_interval": 4,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": tokens.index("<|bos|>"), "tokenizer.ggml.eos_token_id": tokens.index("<|im_end|>"),
        "tokenizer.chat_template": S.CHATML, **extra,
    }
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)) + b"".join(S._kv(k, v) for k, v in meta.items())
    infos, offset, blobs = b"", 0, []
    for name, (gtype, dims, raw) in tensors.items():
        nb = name.encode()
        infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<IQ", gtype, offset)
        pad = (-len(raw)) % 32
        blobs.append(raw + b"\0" * pad)
        offset += len(raw) + pad
    pre = head + infos
    pre += b"\0" * ((-len(pre)) % 32)
    with open(path, "wb") as f:
        f.write(pre + b"".join(blobs))


class TestTernaryCodecs(unittest.TestCase):
    """The three block formats decode to exactly the values ggml's dequantizers give."""

    def test_each_format_round_trips_its_codes_and_scales(self):
        from engine import gguf_tensors as T
        rng = np.random.default_rng(1)
        for kind, (_, block) in TYPES.items():
            codes = rng.integers(0, 3, size=(3, 256)).astype(np.uint8)
            d = (rng.random((3, 256 // block)) * 0.05 + 0.01).astype(np.float16)
            raw = np.frombuffer(encode(kind, codes, d), np.uint8)
            size = T.TYPES[TYPES[kind][0]][2]
            words, scale, bias = T.ternary_blocks(kind, raw.reshape(-1, size))
            want = (codes.astype(np.float32) - 1) * np.repeat(d.astype(np.float32), block, axis=1)
            if mx is None:
                continue
            got = mx.dequantize(mx.array(words.reshape(3, -1)), mx.array(scale.reshape(3, -1)),
                                mx.array(bias.reshape(3, -1)), group_size=block, bits=2)
            np.testing.assert_array_equal(np.asarray(got.astype(mx.float32)), want, kind)

    def test_the_gpu_ptq1_unpacking_equals_the_numpy_one(self):
        if mx is None:
            self.skipTest("MLX is not installed")
        from engine import gguf_tensors as T
        blk = np.random.default_rng(3).integers(0, 256, size=(64, 28), dtype=np.uint8)   # every byte value
        codes = T.ptq1_codes(blk).astype(np.uint32).reshape(-1, 8, 16)
        want = np.bitwise_or.reduce(codes << (2 * np.arange(16, dtype=np.uint32)), axis=-1)
        np.testing.assert_array_equal(T._ptq1_words_mlx(blk), want)

    def test_ptq1_matches_the_metal_kernels_decoding(self):
        if mx is None:
            self.skipTest("MLX is not installed")
        from engine import gguf_tensors as T, ptq1
        rng = np.random.default_rng(2)
        codes = rng.integers(0, 3, size=(4, 384)).astype(np.uint8)
        d = (rng.random((4, 3)) * 0.05 + 0.01).astype(np.float16)
        raw = np.frombuffer(encode("PTQ1_0", codes, d), np.uint8)
        ref = ptq1.dequantize_rows(mx.array(raw.reshape(4, -1)), "ptq1")
        words, scale, bias = T.ternary_blocks("PTQ1_0", raw.reshape(-1, 28))
        got = mx.dequantize(mx.array(words.reshape(4, -1)), mx.array(scale.reshape(4, -1)),
                            mx.array(bias.reshape(4, -1)), group_size=128, bits=2)
        np.testing.assert_array_equal(np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32)))


@unittest.skipIf(mx is None, "MLX is not installed")
class TestRotatedTernaryGGUF(unittest.TestCase):
    def check(self, kind):
        from engine import dense, gguf
        ref, tensors, meta = build(kind)
        path = os.path.join(tempfile.mkdtemp(), f"tiny-bonsai-{kind}.gguf")
        write(path, tensors, meta)
        self.assertEqual(gguf.native_support(path), (True, ""))
        model, cfg, _ = dense.build(path, log=lambda *a: None)
        self.addCleanup(model.close)
        self.assertTrue(cfg.rotated and cfg.gdn_v_grouped)
        ca, cb = model.make_cache(), ref.make_cache()
        steps = [[5, 17, 99, 3, 250], [8], [61], [200]]
        for i, ids in enumerate(steps):
            a, b = model(mx.array([ids]), cache=ca), ref(mx.array([ids]), cache=cb)
            mx.eval(a, b)
            got, want = np.asarray(a.astype(mx.float32))[0], np.asarray(b.astype(mx.float32))[0]
            spread = float(np.ptp(want))
            self.assertLess(float(np.abs(got - want).max()), 0.02 * spread, f"{kind} step {i}")
            np.testing.assert_array_equal(got.argmax(-1), want.argmax(-1), f"{kind} step {i}")
        return path, meta, tensors

    def test_q2_0(self):
        self.check("Q2_0")

    def test_pq2_0(self):
        self.check("PQ2_0")

    def test_ptq1_0(self):
        self.check("PTQ1_0")

    def test_without_the_rotation_it_is_not_the_same_model(self):
        # the contract is what makes it work: the same file without it runs, and is wrong
        from engine import dense
        ref, tensors, meta = build("PQ2_0")
        path = os.path.join(tempfile.mkdtemp(), "unrotated.gguf")
        write(path, tensors, {})
        model, _, _ = dense.build(path, log=lambda *a: None)
        self.addCleanup(model.close)
        a, b = model(mx.array([[5, 17, 99, 3, 250]])), ref(mx.array([[5, 17, 99, 3, 250]]))
        self.assertGreater(float(np.abs(np.asarray(a) - np.asarray(b)).max()), 0.1 * float(np.ptp(np.asarray(b))))

    def test_the_model_list_offers_a_prism_ternary_file_as_runnable(self):
        # PQ2_0 / PTQ1_0 are vendor ggml types stock llama.cpp refuses; the own decoder reads them
        import llm
        _, tensors, meta = build("PTQ1_0")
        path = os.path.join(tempfile.mkdtemp(), "tiny-bonsai-PTQ1_0.gguf")
        write(path, tensors, meta)
        info = llm.inspect_gguf(path)
        self.assertTrue(info["supported"], info["problems"])
        write(path, {k: (999 if v[0] == 143 else v[0], *v[1:]) for k, v in tensors.items()}, meta)
        self.assertFalse(llm.inspect_gguf(path)["supported"])     # a vendor type nobody here reads: still flagged

    def test_a_contract_it_cannot_reproduce_is_refused_up_front(self):
        from engine import gguf
        _, tensors, meta = build("PQ2_0")
        d = tempfile.mkdtemp()
        for bad, why in (({"prism.hadamard.transform": "randomized"}, "transform"),
                         ({"prism.hadamard.inverse_weight_names": ["output.weight"]}, "inverse"),
                         ({"prism.hadamard.weight_names": ["blk.0.no_such.weight"]}, "does not have")):
            path = os.path.join(d, f"bad-{why.replace(' ', '_')}.gguf")
            write(path, tensors, {**meta, **bad})
            ok, reason = gguf.native_support(path)
            if why == "does not have":                 # only the decoder knows the file's tensors
                from engine import dense
                self.assertTrue(ok)
                with self.assertRaises(dense.Unsupported) as ctx:
                    dense.build(path, log=lambda *a: None)
                self.assertIn(why, str(ctx.exception))
                continue
            self.assertFalse(ok)
            self.assertIn(why, reason)


if __name__ == "__main__":
    unittest.main()

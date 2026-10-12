"""`llm inspect`: models described from their headers (synthetic safetensors folders and GGUF files).

    .venv/bin/python -m unittest tests.test_modelinfo -v
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import llm  # noqa: E402
from engine import modelinfo as M  # noqa: E402
from tests import gguf_synth as S  # noqa: E402


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, shape)} → a file with that header and zero bytes of data."""
    header, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = M.ST_BYTES[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + bytes(off))


def mlx_folder(extra_cfg=None, tensors=None):
    d = tempfile.mkdtemp()
    cfg = {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
           "num_key_value_heads": 2, "vocab_size": 100, "max_position_embeddings": 512,
           "quantization": {"group_size": 64, "bits": 4}}
    cfg.update(extra_cfg or {})
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    t = tensors or {}
    if not t:
        t["model.embed_tokens.weight"] = ("U32", (100, 8))              # 4-bit: 8 words = 64 columns
        t["model.embed_tokens.scales"] = ("F16", (100, 1))
        t["model.embed_tokens.biases"] = ("F16", (100, 1))
        for i in range(2):
            t[f"model.layers.{i}.mlp.up_proj.weight"] = ("U32", (128, 8))
            t[f"model.layers.{i}.mlp.up_proj.scales"] = ("F16", (128, 1))
            t[f"model.layers.{i}.mlp.up_proj.biases"] = ("F16", (128, 1))
            t[f"model.layers.{i}.input_layernorm.weight"] = ("F16", (64,))
    write_safetensors(os.path.join(d, "model.safetensors"), t)
    return d


class TestMLX(unittest.TestCase):
    def test_groups_params_and_bits_of_a_quantized_folder(self):
        info = M.inspect(mlx_folder())
        self.assertEqual((info["kind"], info["architecture"], info["layers"], info["hidden"]), ("mlx", "llama", 2, 64))
        self.assertEqual(info["tensors"], 11)
        self.assertEqual(info["params"], 100 * 64 + 2 * (128 * 64 + 64))        # real widths, norms counted
        self.assertEqual(info["weights_by_bits"], {"4": 100 * 64 + 2 * 128 * 64})
        names = {g["pattern"]: g for g in info["groups"]}
        up = names["model.layers.N.mlp.up_proj.weight"]
        self.assertEqual((up["count"], up["dtype"], up["bytes"]), (2, "U32", 2 * 128 * 8 * 4))
        self.assertEqual(info["dtypes"]["F16"]["tensors"], 8)
        self.assertEqual(info["size_bytes"], sum(g["bytes"] for g in info["groups"]))
        self.assertEqual(info["groups"][0]["bytes"], max(g["bytes"] for g in info["groups"]))     # largest first
        self.assertIn("no tokenizer file in the folder", info["notes"])

    def test_ptq1_bundles_count_128_weights_per_28_bytes(self):
        d = mlx_folder({"quantization": {"format": "PTQ1_0", "group_size": 128}, "schema": "prism_hadamard_qwen35_ptq1_v1"},
                       {"model.layers.0.mlp.up_proj.weight": ("U8", (256, 56)),     # two blocks per row
                        "model.norm.weight": ("F32", (64,))})
        info = M.inspect(d)
        self.assertEqual(info["params"], 256 * 2 * 128 + 64)
        self.assertEqual(info["weights_by_bits"], {"1.75": 256 * 2 * 128})
        self.assertIn("JANG / Prism bundle", info["notes"])

    def test_text_config_and_experts(self):
        d = mlx_folder({"text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 9, "hidden_size": 32,
                                        "num_experts": 8}})
        info = M.inspect(d)
        self.assertEqual((info["architecture"], info["layers"], info["experts"]), ("qwen3_5_text", 9, 8))

    def test_damaged_and_empty_folders_are_refused_with_a_reason(self):
        d = tempfile.mkdtemp()
        with self.assertRaisesRegex(M.InspectError, "no .safetensors"):
            M.inspect(d)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", 1 << 40) + b"x")
        with self.assertRaisesRegex(M.InspectError, "not plausible"):
            M.inspect(d)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", 5) + b"{oops")
        with self.assertRaisesRegex(M.InspectError, "damaged"):
            M.inspect(d)
        with self.assertRaisesRegex(M.InspectError, "not a model"):
            M.inspect(os.path.join(d, "nothing.bin"))

    def test_grouping_replaces_layer_and_expert_numbers_only(self):
        self.assertEqual(M.group_name("model.layers.12.mlp.experts.7.up_proj.weight"), "model.layers.N.mlp.experts.N.up_proj.weight")
        self.assertEqual(M.group_name("blk.3.ffn_up.weight"), "blk.N.ffn_up.weight")
        self.assertEqual(M.group_name("layer2.weight"), "layer2.weight")               # a digit inside a word stays


class TestGGUF(unittest.TestCase):
    def test_a_gguf_file(self):
        p = os.path.join(tempfile.mkdtemp(), "tiny.gguf")
        S.write(p, quant="Q8_0")
        info = M.inspect(p)
        self.assertEqual((info["kind"], info["architecture"], info["layers"], info["hidden"], info["heads"]),
                         ("gguf", "llama", S.LAYERS, S.HIDDEN, S.HEADS))
        self.assertTrue(info["own_decoder"])
        self.assertEqual(info["tokenizer"], "gpt2")
        self.assertEqual(info["dtypes"]["Q8_0"]["tensors"], 2 + S.LAYERS * 7)         # embedding, output, 7 per layer
        self.assertIn("F32", info["dtypes"])                                           # the norms
        names = {g["pattern"] for g in info["groups"]}
        self.assertIn("blk.N.attn_q.weight", names)
        # sizes come from the offsets: together they cover the data section (alignment padding aside)
        data = os.path.getsize(p) - M.gguf_tensors(p)[2]
        self.assertLessEqual(info["size_bytes"], data)
        self.assertGreater(info["size_bytes"], data - 64 * len(info["groups"]) * 32)
        self.assertEqual(info["params"], sum(__import__("math").prod(g["shape"]) * g["count"] for g in info["groups"]))

    def test_the_reason_a_file_needs_llama_cpp_is_shown(self):
        p = os.path.join(tempfile.mkdtemp(), "odd.gguf")
        S.write(p, arch="phi3")
        info = M.inspect(p)
        self.assertFalse(info["own_decoder"])
        self.assertIn("architecture", info["own_decoder_why"])
        self.assertIn("llama.cpp", M.render(info))


class TestRenderAndCommand(unittest.TestCase):
    def test_render_is_plain_text_with_the_essentials(self):
        text = M.render(M.inspect(mlx_folder()), top=2)
        for want in ("llama", "layers 2", "11 tensors", "quantization: group 64, 4 bits", "weights by bits: 4-bit 100%",
                     "model.layers.N.mlp.up_proj.weight", "more groups"):
            self.assertIn(want, text)

    def test_command_by_path_json_and_unknown(self):
        d = mlx_folder()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertTrue(llm.inspect_cmd({}, [d, "--json"]))
        self.assertEqual(json.loads(buf.getvalue())["architecture"], "llama")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(llm.inspect_cmd({"dirs": [], "model": ""}, ["definitely-not-a-model"]))
        self.assertIn(llm.T("Model not found."), out.getvalue())

    def test_command_by_number_uses_the_model_list(self):
        d = mlx_folder()
        with mock.patch.object(llm, "scan_models", return_value=[{"path": d, "name": "tiny-llama"}]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(llm.inspect_cmd({}, ["1", "--top", "3"]))
            self.assertTrue(llm.inspect_cmd({}, ["tiny"]))
        self.assertEqual(out.getvalue().count("11 tensors"), 2)


if __name__ == "__main__":
    unittest.main()

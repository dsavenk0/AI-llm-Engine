"""GGUF -> MLX folder conversion (engine/convert.py).

    .venv/bin/python -m unittest tests.test_convert -v
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from engine import convert  # noqa: E402
from tests import gguf_synth as S  # noqa: E402


class TestConvert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.gguf = os.path.join(cls.dir, "tiny.gguf")
        cls.values = S.write(cls.gguf, seed=3, quant="Q8_0", arch="llama")      # quantized, to exercise dequant
        cls.out = os.path.join(cls.dir, "mlx")
        convert.convert(cls.gguf, cls.out, log=lambda *_: None)

    def test_config_describes_the_model(self):
        with open(os.path.join(self.out, "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["model_type"], "llama")
        self.assertEqual(cfg["hidden_size"], S.HIDDEN)
        self.assertEqual(cfg["num_hidden_layers"], S.LAYERS)
        self.assertEqual(cfg["num_attention_heads"], S.HEADS)
        self.assertEqual(cfg["num_key_value_heads"], S.KV_HEADS)
        self.assertEqual(cfg["vocab_size"], self.values["token_embd.weight"].shape[0])

    def test_weights_are_hf_named_f16_and_match_the_source(self):
        w = mx.load(os.path.join(self.out, "model.safetensors"))
        self.assertIn("model.embed_tokens.weight", w)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", w)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", w)
        self.assertIn("lm_head.weight", w)
        self.assertEqual(w["model.embed_tokens.weight"].dtype, mx.float16)
        self.assertEqual(w["model.layers.0.self_attn.q_proj.weight"].shape,
                         (S.HEADS * S.HEAD_DIM, S.HIDDEN))
        # the exported embedding is the file's values (Q8_0-dequantized), within fp16
        got = np.asarray(w["model.embed_tokens.weight"].astype(mx.float32))
        np.testing.assert_allclose(got, self.values["token_embd.weight"], atol=0.05, rtol=0)

    def test_tokenizer_round_trips(self):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(self.out, "tokenizer.json"))
        ids = tok.encode("hello world").ids
        self.assertTrue(ids)
        self.assertEqual(tok.decode(ids).strip(), "hello world")
        self.assertTrue(os.path.exists(os.path.join(self.out, "tokenizer_config.json")))

    def test_an_unsupported_architecture_is_refused(self):
        p = os.path.join(self.dir, "granite.gguf")
        S.write(p, seed=1, quant="F16", arch="granite")
        with self.assertRaises(convert.Unconvertible):
            convert.convert(p, os.path.join(self.dir, "granite-mlx"))

    def test_hf_name_mapping(self):
        self.assertEqual(convert.hf_name("token_embd.weight"), "model.embed_tokens.weight")
        self.assertEqual(convert.hf_name("blk.3.attn_q.weight"), "model.layers.3.self_attn.q_proj.weight")
        self.assertEqual(convert.hf_name("blk.0.ffn_down.weight"), "model.layers.0.mlp.down_proj.weight")
        self.assertIsNone(convert.hf_name("something.odd"))


if __name__ == "__main__":
    unittest.main()

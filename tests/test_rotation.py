"""engine/rotation.py: one Hadamard contract from wherever a format declares it, read strictly (stdlib only)."""
import json
import os
import tempfile
import unittest

from engine import rotation

GOOD = {"version": 1, "block_size": 4, "transform": "normalized-sylvester-walsh-hadamard",
        "axis": "input-last-dimension", "sign_mode": "explicit", "sign_widths": [4, 8],
        "sign_values": [1, -1, 1, 1] + [-1] * 8, "weight_names": ["a.weight"], "inverse_weight_names": ["e.weight"],
        "gdn_v_grouped": True}


def prism(**over):
    d = {**GOOD, **over}
    return rotation.from_prism(lambda k: d.get(k), "test")


class TestPrismDeclaration(unittest.TestCase):
    def test_a_complete_declaration_reads(self):
        c = prism()
        self.assertEqual((c.block, sorted(c.signs), c.folded, c.inverse, c.gdn_v_grouped),
                         (4, [4, 8], {"a.weight"}, {"e.weight"}, True))

    def test_each_field_it_cannot_reproduce_is_refused_by_name(self):
        for over, word in (({"version": 2}, "version"), ({"transform": "randomized"}, "transform"),
                           ({"axis": "output"}, "axis"), ({"sign_mode": "seeded"}, "sign_mode"),
                           ({"block_size": 6}, "power of two"), ({"sign_values": [1] * 11}, "add up"),
                           ({"sign_values": [1, 0, 1, 1] + [1] * 8}, "±1"), ({"sign_widths": [4, 6], "sign_values": [1] * 10}, "divide"),
                           ({"weight_names": None}, "which weights"),
                           ({"inverse_weight_names": ["a.weight"]}, "both")):
            with self.subTest(over=over), self.assertRaises(rotation.Unsupported) as ctx:
                prism(**over)
            self.assertIn(word, str(ctx.exception))


class TestBundle(unittest.TestCase):
    def folder(self, contract=None):
        d = tempfile.mkdtemp()
        if contract is not None:
            with open(os.path.join(d, "hadamard.json"), "w") as f:
                json.dump({"prism.hadamard." + k: v for k, v in contract.items()}, f)
        return d

    def test_no_declaration_is_no_rotation(self):
        self.assertIsNone(rotation.bundle(self.folder(), {"model_type": "qwen3_5"}))

    def test_the_schema1_block_reads_from_the_config_or_jang(self):
        self.assertEqual(rotation.bundle(self.folder(), {"hadamard": {"block_size": 512}}).block, 512)
        c = rotation.bundle(self.folder(), {}, {"hadamard": {"block_size": 1024}})
        self.assertEqual((c.block, c.folded), (1024, None))            # which weights: implicit

    def test_the_schema2_file_reads(self):
        c = rotation.bundle(self.folder(GOOD), {"hadamard_config": "hadamard.json"})
        self.assertEqual((c.block, c.source), (4, "hadamard.json"))

    def test_a_pointer_out_of_the_model_folder_is_refused(self):
        with self.assertRaises(rotation.Unsupported):
            rotation.bundle(self.folder(GOOD), {"hadamard_config": "../hadamard.json"})

    def test_a_missing_file_and_an_unknown_activation_layout_are_refused(self):
        with self.assertRaises(rotation.Unsupported):
            rotation.bundle(self.folder(), {"hadamard_config": "hadamard.json"})
        with self.assertRaises(rotation.Unsupported):
            rotation.bundle(self.folder(GOOD), {"hadamard_config": "hadamard.json", "gdn_activation_layout": "tiled"})


class TestNativeSet(unittest.TestCase):
    def test_it_follows_the_layer_kinds_and_leaves_the_gates_unrotated(self):
        folded, inverse = rotation.native_set({"num_hidden_layers": 4, "full_attention_interval": 4})
        self.assertIn("language_model.model.layers.3.self_attn.o_proj.weight", folded)
        self.assertIn("language_model.model.layers.0.linear_attn.out_proj.weight", folded)
        self.assertNotIn("language_model.model.layers.0.linear_attn.in_proj_a.weight", folded)
        self.assertEqual(len(folded), 3 * 6 + 7 + 1)
        self.assertEqual(inverse, {"language_model.model.embed_tokens.weight"})

    def test_names_without_the_language_model_prefix_count_the_same(self):
        tc = {"num_hidden_layers": 4, "full_attention_interval": 4}
        folded, inverse = rotation.native_set(tc)
        short = rotation.Contract(4, folded={k.removeprefix("language_model.") for k in folded},
                                  inverse={k.removeprefix("language_model.") for k in inverse})
        self.assertEqual(rotation.native_mismatch(short, tc), "")


class TestGGUF(unittest.TestCase):
    def meta(self, **over):
        return {"prism.hadamard." + k: v for k, v in {**GOOD, **over}.items()}

    def test_plain_files_have_no_contract(self):
        self.assertIsNone(rotation.from_gguf({"general.architecture": "qwen35"}, lambda k: None))

    def test_long_arrays_come_from_the_reader(self):
        m = self.meta(inverse_weight_names=["token_embd.weight"])
        values = m.pop("prism.hadamard.sign_values")
        c = rotation.from_gguf(m, lambda k: values if k.endswith("sign_values") else None)
        self.assertEqual(sorted(c.signs), [4, 8])

    def test_inverse_rotation_off_the_embedding_and_missing_signs_are_refused(self):
        with self.assertRaises(rotation.Unsupported):
            rotation.from_gguf(self.meta(), lambda k: None)              # inverse on "e.weight"
        m = self.meta(inverse_weight_names=["token_embd.weight"])
        del m["prism.hadamard.sign_widths"], m["prism.hadamard.sign_values"]
        with self.assertRaises(rotation.Unsupported):
            rotation.from_gguf(m, lambda k: None)


if __name__ == "__main__":
    unittest.main()

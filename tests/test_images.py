"""engine/images.py and the /v1/images/generations endpoint, with a fake mflux-generate (no model, no GPU).

    .venv/bin/python -m unittest tests.test_images -v
"""
import base64
import json
import os
import stat
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from engine import images  # noqa: E402

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

FAKE = """#!/bin/sh
# records its arguments and writes a PNG to --output
echo "$@" > "$FAKE_LOG"
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--output" ]; then out="$2"; fi
  shift
done
[ -n "$FAKE_FAIL" ] && { echo "boom: out of memory" 1>&2; exit 3; }
printf '%s' "$FAKE_PNG_B64" | base64 -D > "$out"
"""


def fake_tool(directory, fail=False):
    path = os.path.join(directory, "mflux-generate")
    with open(path, "w") as f:
        f.write(FAKE)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    os.environ["FAKE_LOG"] = os.path.join(directory, "args.txt")
    os.environ["FAKE_PNG_B64"] = base64.b64encode(PNG).decode()
    if fail:
        os.environ["FAKE_FAIL"] = "1"
    else:
        os.environ.pop("FAKE_FAIL", None)
    return path


class TestCommand(unittest.TestCase):
    def test_arguments(self):
        cmd = images.command("/bin/t", "a red fox", "/tmp/o.png", model="z-image-turbo", steps=6, size="1024x768", seed=7)
        self.assertEqual(cmd[:3], ["/bin/t", "--model", "z-image-turbo"])
        self.assertIn("--prompt=a red fox", cmd)
        self.assertEqual(cmd[cmd.index("--width") + 1], "1024")
        self.assertEqual(cmd[cmd.index("--height") + 1], "768")
        self.assertEqual(cmd[cmd.index("--steps") + 1], "6")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "7")
        self.assertEqual(cmd[-2:], ["-q", "8"])

    def test_the_prompt_is_one_argument_and_never_an_option_or_shell(self):
        cmd = images.command("/bin/t", "--output /etc/passwd; rm -rf ~", "/tmp/o.png")
        self.assertIn("--prompt=--output /etc/passwd; rm -rf ~", cmd)
        self.assertEqual(cmd.count("--output"), 1)
        self.assertIn("--prompt=--seed", images.command("/bin/t", "--seed", "/tmp/o.png"))   # no bare option-like value

    def test_bad_input_is_refused(self):
        for kw in ({"prompt": ""}, {"prompt": "x" * 4001}, {"prompt": "x", "model": "../../bin/sh"},
                   {"prompt": "x", "model": "-h"}, {"prompt": "x", "size": "100000x10"}, {"prompt": "x", "size": "big"},
                   {"prompt": "x", "steps": 0}, {"prompt": "x", "steps": 1000}, {"prompt": "x", "quantize": 7}):
            with self.subTest(kw):
                with self.assertRaises(images.ImageError) as ctx:
                    images.command("/bin/t", kw.pop("prompt"), "/tmp/o.png", **kw)
                self.assertEqual(ctx.exception.status, 400)

    def test_sizes_are_rounded_to_multiples_of_16(self):
        self.assertEqual(images.parse_size("1000x700"), (992, 688))
        self.assertEqual(images.parse_size(None), (768, 768))


class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: [os.environ.pop(k, None) for k in ("LLM_IMAGES_BIN", "FAKE_LOG", "FAKE_PNG_B64", "FAKE_FAIL")])

    def test_returns_the_png_the_tool_wrote(self):
        os.environ["LLM_IMAGES_BIN"] = fake_tool(self.dir)
        self.assertEqual(images.generate("a cat", size="512x512"), PNG)
        self.assertIn("a cat", open(os.path.join(self.dir, "args.txt")).read())

    def test_not_installed_says_how_to_install(self):
        os.environ["LLM_IMAGES_BIN"] = os.path.join(self.dir, "missing")
        with self.assertRaises(images.ImageError) as ctx:
            images.generate("a cat")
        self.assertEqual(ctx.exception.status, 501)
        self.assertIn("setup --images", str(ctx.exception))

    def test_a_failing_tool_reports_its_last_lines(self):
        os.environ["LLM_IMAGES_BIN"] = fake_tool(self.dir, fail=True)
        with self.assertRaises(images.ImageError) as ctx:
            images.generate("a cat")
        self.assertEqual(ctx.exception.status, 500)
        self.assertIn("out of memory", str(ctx.exception))

    def test_the_venv_next_to_the_data_folder_is_found(self):
        bindir = os.path.join(self.dir, ".venv-images", "bin")
        os.makedirs(bindir)
        fake_tool(bindir)
        self.assertEqual(images.tool_path(self.dir), os.path.join(bindir, "mflux-generate"))


class TestServerEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_server_native import Server
        from tests.synth import CHATML, make_bundle
        cls.tmp = tempfile.mkdtemp()
        cls.model = os.path.join(cls.tmp, "m")
        make_bundle(cls.model, CHATML, seed=17)
        cls.tool = fake_tool(cls.tmp)
        os.environ["LLM_IMAGES_BIN"] = cls.tool
        cls.on = Server(cls.model, "--images", "on")
        cls.off = Server(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.on.stop()
        cls.off.stop()
        os.environ.pop("LLM_IMAGES_BIN", None)

    def test_off_by_default(self):
        code, r = self.off.json("POST", "/v1/images/generations", {"prompt": "x"})
        self.assertEqual((code, r["error"]["code"]), (501, "images_disabled"))

    def test_an_image_comes_back_as_base64_and_the_text_model_sleeps(self):
        self.assertEqual(self.on.get("/health")["state"], "awake")
        code, r = self.on.json("POST", "/v1/images/generations", {"prompt": "a lighthouse", "size": "256x256"})
        self.assertEqual(code, 200, r)
        self.assertEqual(base64.b64decode(r["data"][0]["b64_json"]), PNG)
        self.assertEqual(self.on.get("/health")["state"], "sleeping")
        code, chat = self.on.chat()                              # …and the chat still works (it wakes)
        self.assertEqual(code, 200, chat)

    def test_validation(self):
        self.assertEqual(self.on.json("POST", "/v1/images/generations", {"prompt": ""})[0], 400)
        self.assertEqual(self.on.json("POST", "/v1/images/generations", {"prompt": "x", "n": 2})[0], 400)
        self.assertEqual(self.on.json("POST", "/v1/images/generations", {"prompt": "x", "size": "9999x9999"})[0], 400)


if __name__ == "__main__":
    unittest.main()

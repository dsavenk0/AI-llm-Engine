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


class TestLimits(unittest.TestCase):
    def cmd(self, **kw):
        return images.command("/bin/t", "x", "/tmp/o.png", **kw)

    def test_size_edges_and_rounding(self):
        self.assertEqual(images.parse_size("256x256"), (256, 256))
        self.assertEqual(images.parse_size("1536x1536"), (1536, 1536))
        self.assertEqual(images.parse_size("1000x777"), (992, 768))        # down to a multiple of 16
        self.assertEqual(images.parse_size(None), (768, 768))
        for bad in ("255x512", "512x255", "1537x512", "512x1537", "0x0", "512"):
            with self.assertRaises(images.ImageError, msg=bad):
                images.parse_size(bad)

    def test_step_and_quantize_edges(self):
        for steps in (1, 100):
            self.assertEqual(self.cmd(steps=steps)[self.cmd(steps=steps).index("--steps") + 1], str(steps))
        for steps in (0, 101, -1):
            with self.assertRaises(images.ImageError, msg=steps):
                self.cmd(steps=steps)
        for q in (3, 4, 5, 6, 8):
            c = self.cmd(quantize=q)
            self.assertEqual(c[-2:], ["-q", str(q)])
        for q in (1, 2, 7, 9):
            with self.assertRaises(images.ImageError, msg=q):
                self.cmd(quantize=q)
        self.assertNotIn("-q", self.cmd(quantize=0))
        self.assertEqual(images.command("/bin/t", "x" * 4000, "/o")[0], "/bin/t")    # 4000 is allowed

    def test_the_tool_is_found_in_the_data_folder_the_environment_or_on_the_path(self):
        from unittest import mock
        d = tempfile.mkdtemp()
        env = {k: v for k, v in os.environ.items() if k != "LLM_IMAGES_BIN"}
        with mock.patch.dict(os.environ, env, clear=True):
            os.environ["PATH"] = ""
            self.assertIsNone(images.tool_path(d))
            os.makedirs(os.path.join(d, ".venv-images", "bin"))
            tool = os.path.join(d, ".venv-images", "bin", "mflux-generate")
            with open(tool, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(tool, 0o755)
            self.assertEqual(images.tool_path(d), tool)
            bin2 = tempfile.mkdtemp()
            on_path = os.path.join(bin2, "mflux-generate")
            with open(on_path, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(on_path, 0o755)
            os.environ["PATH"] = bin2
            self.assertEqual(images.tool_path(None), on_path)
            os.environ["LLM_IMAGES_BIN"] = "/no/such/tool"
            self.assertIsNone(images.tool_path(d))                      # an explicit choice is not second-guessed

    def test_failures_report_the_last_lines_and_the_timeout_default_is_half_an_hour(self):
        import inspect
        self.assertEqual(inspect.signature(images.generate).parameters["timeout"].default, 1800)
        d = tempfile.mkdtemp()
        tool = os.path.join(d, "bad")
        with open(tool, "w") as f:
            f.write("#!/bin/sh\nfor i in 1 2 3 4 5; do echo line$i 1>&2; done\necho " + "z" * 600 + " 1>&2\nexit 3\n")
        os.chmod(tool, 0o755)
        from unittest import mock
        with mock.patch.object(images, "tool_path", return_value=tool):
            with self.assertRaises(images.ImageError) as ctx:
                images.generate("x")
        msg = str(ctx.exception)
        self.assertEqual(ctx.exception.status, 500)
        self.assertTrue(msg.startswith("mflux failed: "))
        self.assertNotIn("line1", msg)                                  # only the last three lines
        self.assertIn("line5", msg)
        self.assertLessEqual(len(msg), len("mflux failed: ") + 400)
        slow = os.path.join(d, "slow")
        with open(slow, "w") as f:
            f.write("#!/bin/sh\nsleep 5\n")
        os.chmod(slow, 0o755)
        with mock.patch.object(images, "tool_path", return_value=slow):
            with self.assertRaises(images.ImageError) as ctx:
                images.generate("x", timeout=1)
        self.assertEqual((ctx.exception.status, str(ctx.exception)), (500, "no image within 1 s"))


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

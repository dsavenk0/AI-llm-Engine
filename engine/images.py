"""Image generation through mflux (FLUX and friends on MLX) — an optional module, stdlib only.

mflux is installed into its own virtual environment (`./llm setup --images`) so the engine's stays small;
nothing here imports it. A request runs `mflux-generate` as a subprocess and returns the PNG, which means
the weights are loaded and freed per image: slow on a cold start, but the Mac's memory is back to the text
model afterwards.
"""
import os
import re
import shutil
import subprocess
import tempfile

# a short list of names the CLI knows; anything matching the pattern is passed through, nothing else is
MODELS = ("schnell", "dev", "z-image-turbo", "z-image", "flux2-klein-4b", "flux2-klein-9b", "qwen-image",
          "krea-dev", "fibo-lite")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
DEFAULT_MODEL = "schnell"
MAX_SIDE = 1536


class ImageError(RuntimeError):
    """`status` is the HTTP code a server should answer with: 400 bad input, 501 not installed, 500 failed."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def tool_path(data_dir=None):
    """The mflux-generate executable: $LLM_IMAGES_BIN, the images venv, or PATH; None when not installed."""
    env = os.environ.get("LLM_IMAGES_BIN")
    if env:
        return env if os.access(env, os.X_OK) else None
    if data_dir:
        p = os.path.join(data_dir, ".venv-images", "bin", "mflux-generate")
        if os.access(p, os.X_OK):
            return p
    return shutil.which("mflux-generate")


def parse_size(size, default=(768, 768)):
    """"WxH" (OpenAI's `size`) -> (w, h), multiples of 16, each 256..MAX_SIDE."""
    if not size:
        return default
    m = re.fullmatch(r"(\d{2,5})x(\d{2,5})", str(size).strip().lower())
    if not m:
        raise ImageError(f"size must look like 1024x1024, got {size!r}")
    w, h = int(m.group(1)), int(m.group(2))
    if not (256 <= w <= MAX_SIDE and 256 <= h <= MAX_SIDE):
        raise ImageError(f"each side must be between 256 and {MAX_SIDE} pixels")
    return (w // 16 * 16, h // 16 * 16)


def command(tool, prompt, out, model=DEFAULT_MODEL, steps=None, size=None, seed=None, quantize=8):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageError("a prompt is required")
    if len(prompt) > 4000:
        raise ImageError("the prompt is too long (4000 characters at most)")
    model = (model or DEFAULT_MODEL).strip().lower()
    if not NAME_RE.match(model):
        raise ImageError(f"not a model name: {model!r}")
    w, h = parse_size(size)
    cmd = [tool, "--model", model, "--prompt", prompt, "--width", str(w), "--height", str(h), "--output", out]
    if steps is not None:
        if not (1 <= int(steps) <= 100):
            raise ImageError("steps must be between 1 and 100")
        cmd += ["--steps", str(int(steps))]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if quantize:
        if int(quantize) not in (3, 4, 5, 6, 8):
            raise ImageError("quantize must be one of 3, 4, 5, 6, 8")
        cmd += ["-q", str(int(quantize))]
    return cmd


def generate(prompt, data_dir=None, timeout=1800, **kw):
    """PNG bytes for `prompt`. Raises ImageError (not installed, bad input, the tool failed or timed out)."""
    tool = tool_path(data_dir)
    if not tool:
        raise ImageError("image generation is not installed — run: ./llm setup --images", 501)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "image.png")
        cmd = command(tool, prompt, out, **kw)
        try:
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                               start_new_session=True)
        except subprocess.TimeoutExpired:
            raise ImageError(f"no image within {timeout} s", 500)
        if p.returncode != 0 or not os.path.isfile(out):
            tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
            raise ImageError("mflux failed: " + " | ".join(tail)[:400], 500)
        with open(out, "rb") as f:
            return f.read()

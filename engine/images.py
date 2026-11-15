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

# MFLUX has a separate entry point for several model families.  Keeping that routing here prevents a model
# name which is valid in MFLUX from accidentally being handed to the FLUX.1 pipeline.  `steps` are the
# distilled models' documented defaults; an API caller can still override them.
MODEL_SPECS = {
    "schnell":        {"binary": "mflux-generate", "steps": 4, "label": "FLUX.1 Schnell", "min_gb": 16},
    "dev":            {"binary": "mflux-generate", "steps": 20, "label": "FLUX.1 Dev", "min_gb": 16},
    "krea-dev":       {"binary": "mflux-generate", "steps": 20, "label": "FLUX.1 Krea Dev", "min_gb": 20},
    "z-image-turbo":  {"binary": "mflux-generate-z-image-turbo", "steps": 9,
                         "label": "Z-Image Turbo", "min_gb": 12},
    "z-image":        {"binary": "mflux-generate-z-image", "steps": 30, "label": "Z-Image", "min_gb": 12},
    "flux2-klein-4b": {"binary": "mflux-generate-flux2", "steps": 4,
                         "label": "FLUX.2 Klein 4B", "min_gb": 8},
    "flux2-klein-9b": {"binary": "mflux-generate-flux2", "steps": 4,
                         "label": "FLUX.2 Klein 9B", "min_gb": 16},
    "qwen-image":     {"binary": "mflux-generate-qwen", "steps": 30, "label": "Qwen Image", "min_gb": 24},
    "fibo-lite":      {"binary": "mflux-generate-fibo", "steps": 8, "label": "FIBO Lite", "min_gb": 20},
}
MODELS = tuple(MODEL_SPECS)
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
DEFAULT_MODEL = "auto"
MAX_SIDE = 1536


class ImageError(RuntimeError):
    """`status` is the HTTP code a server should answer with: 400 bad input, 501 not installed, 500 failed."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def physical_memory_gb():
    """Installed unified memory, without importing MLX (keeps this optional module stdlib-only)."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (AttributeError, OSError, ValueError):
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / 2**30
        except (OSError, ValueError, subprocess.SubprocessError):
            return 16.0


def auto_profile(memory_gb=None):
    """A conservative MFLUX profile which leaves headroom for macOS after the text model is put to sleep."""
    gb = float(memory_gb if memory_gb is not None else physical_memory_gb())
    if gb < 14:
        return {"model": "flux2-klein-4b", "quantize": 4, "size": (512, 512), "low_ram": True}
    if gb < 28:
        return {"model": "flux2-klein-4b", "quantize": 4, "size": (768, 768), "low_ram": gb < 18}
    return {"model": "z-image-turbo", "quantize": 8, "size": (1024, 1024), "low_ram": False}


def _profile(model=None, memory_gb=None):
    auto = not model or str(model).strip().lower() == "auto"
    fit = auto_profile(memory_gb)
    p = fit if auto else {**fit, "model": str(model).strip().lower()}
    if p["model"] not in MODEL_SPECS:
        raise ImageError(f"unsupported image model: {p['model']!r}")
    return p


def tool_path(data_dir=None, model=None):
    """The right MFLUX executable: $LLM_IMAGES_BIN, the images venv, or PATH; None when not installed.

    With no model this deliberately checks the base command, which is also a cheap installation probe.
    """
    env = os.environ.get("LLM_IMAGES_BIN")
    if env:
        return env if os.access(env, os.X_OK) else None
    binary = MODEL_SPECS[_profile(model)["model"]]["binary"] if model else "mflux-generate"
    if data_dir:
        p = os.path.join(data_dir, ".venv-images", "bin", binary)
        if os.access(p, os.X_OK):
            return p
    return shutil.which(binary)


def _cache_roots():
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return [os.path.expanduser(os.environ["HUGGINGFACE_HUB_CACHE"])]
    if os.environ.get("HF_HOME"):
        return [os.path.join(os.path.expanduser(os.environ["HF_HOME"]), "hub")]
    return [os.path.expanduser("~/.cache/huggingface/hub")]


def installed_models():
    """Known image models whose HF cache contains at least one safetensors file.

    This is intentionally read-only and conservative: an interrupted download is not offered in the chooser.
    """
    found = []
    aliases = {
        "flux2-klein-4b": ("flux.2-klein-4b", "flux2-klein-4b", "klein-4b"),
        "flux2-klein-9b": ("flux.2-klein-9b", "flux2-klein-9b", "klein-9b"),
        "z-image-turbo": ("z-image-turbo",), "z-image": ("z-image",),
        "schnell": ("flux.1-schnell", "flux1-schnell"), "dev": ("flux.1-dev", "flux1-dev"),
        "qwen-image": ("qwen-image",), "fibo-lite": ("fibo-lite",), "krea-dev": ("krea-dev",),
    }
    for root in _cache_roots():
        try:
            entries = [os.path.join(root, x) for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
        except OSError:
            continue
        for model, needles in aliases.items():
            candidates = [p for p in entries if any(n in os.path.basename(p).lower() for n in needles)]
            if model == "z-image":
                candidates = [p for p in candidates if "z-image-turbo" not in os.path.basename(p).lower()]
            complete = False
            for base in candidates:
                for _, _, files in os.walk(os.path.join(base, "snapshots")):
                    if any(f.endswith(".safetensors") for f in files):
                        complete = True
                        break
                if complete:
                    break
            if complete and model not in found:
                found.append(model)
    return found


def model_choices(memory_gb=None):
    """Installed choices for the chat dialog, best automatic fit first."""
    gb = float(memory_gb if memory_gb is not None else physical_memory_gb())
    recommended = auto_profile(gb)["model"]
    order = [recommended, "flux2-klein-4b", "z-image-turbo", "flux2-klein-9b", "schnell", "dev",
             "qwen-image", "fibo-lite", "z-image", "krea-dev"]
    have = set(installed_models())
    choices = []
    for m in order:
        if m not in have:
            continue
        fits = gb >= MODEL_SPECS[m]["min_gb"]
        suffix = " · recommended" if m == recommended else ("" if fits else f" · needs ≈{MODEL_SPECS[m]['min_gb']} GB")
        choices.append({"model": m, "label": MODEL_SPECS[m]["label"] + suffix, "fits": fits})
    return sorted(choices, key=lambda x: not x["fits"])


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


def command(tool, prompt, out, model=DEFAULT_MODEL, steps=None, size=None, seed=None, quantize=None,
            low_ram=None, memory_gb=None):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageError("a prompt is required")
    if len(prompt) > 4000:
        raise ImageError("the prompt is too long (4000 characters at most)")
    profile = _profile(model, memory_gb)
    model = profile["model"]
    if not NAME_RE.match(model):
        raise ImageError(f"not a model name: {model!r}")
    w, h = parse_size(size, profile["size"])
    # --prompt=<text> is one argument whatever the text starts with (a bare "--foo" would be read as an option)
    cmd = [tool, "--model", model, f"--prompt={prompt}", "--width", str(w), "--height", str(h), "--output", out]
    steps = MODEL_SPECS[model]["steps"] if steps is None else int(steps)
    if steps is not None:
        if not (1 <= int(steps) <= 100):
            raise ImageError("steps must be between 1 and 100")
        cmd += ["--steps", str(int(steps))]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if profile["low_ram"] if low_ram is None else bool(low_ram):
        cmd += ["--low-ram"]
    quantize = profile["quantize"] if quantize is None else quantize
    if quantize:
        if int(quantize) not in (3, 4, 5, 6, 8):
            raise ImageError("quantize must be one of 3, 4, 5, 6, 8")
        cmd += ["-q", str(int(quantize))]
    return cmd


def generate(prompt, data_dir=None, timeout=1800, **kw):
    """PNG bytes for `prompt`. Raises ImageError (not installed, bad input, the tool failed or timed out)."""
    profile = _profile(kw.get("model"), kw.get("memory_gb"))
    kw["model"] = profile["model"]
    tool = tool_path(data_dir, profile["model"])
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

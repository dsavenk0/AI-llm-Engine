#!/usr/bin/env python3
"""
llm — manager for local MLX models running on the bundled engine (engine/).

  ./llm                      interactive menu
  ./llm open                 new Terminal window: start the model and open chat (= LLM.command)
  ./llm up                   start (if needed), chat right away, then the menu
  ./llm setup [--compat]     create .venv and install dependencies (--compat adds mlx-lm for other model families)
  ./llm models               discovered models and compatibility
  ./llm search <query>       search MLX models on Hugging Face (HF_ENDPOINT / HF_TOKEN respected)
  ./llm pull <org/name>      download a model (resumable) into ~/.cache/huggingface/hub
  ./llm use <#|name part>    select a model (restarts it if running)
  ./llm rm <#|name part> [--yes]   delete a downloaded model from disk
  ./llm start|stop|restart|status|chat|bench|logs|free
  ./llm agent [dir]          terminal coding agent (files, search, shell) in dir or the current folder
  ./llm ui [dir]             native code editor with the assistant (build it once: ./build-editor.sh)
  ./llm connect [client]     register the model in Continue, Kilo Code or opencode (your choice)
  ./llm test                 engine tests (integration tests too when a model is selected)
  ./llm perf [--save]        performance suite vs tests/perf_baseline.json (regression check)
  ./llm eval [--save]        accuracy suite (math/code/sanity) vs tests/eval_baseline.json
  ./llm eval --suite gsm8k,mmlu --limit 100   standard benchmarks (--suite list shows them)
  ./llm calibrate [--apply]  find the largest context that actually prefills on this Mac

The manager itself is stdlib-only; the engine runs from ./.venv.
Language: English by default, Russian via LLM_LANG=ru, the settings menu or a ru system locale.
"""
import glob
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# launched from a GUI app the PATH can be minimal; sysctl / vm_stat / ipconfig live in the system dirs
os.environ["PATH"] = os.environ.get("PATH", "") + ":/usr/bin:/bin:/usr/sbin:/sbin"
HOME = os.path.expanduser("~")
BASE = os.path.dirname(os.path.abspath(__file__))          # code (repo or LLMBar.app/Contents/Resources/llm)
DATA = os.environ.get("LLM_HOME") or BASE                   # state + .venv (LLMBar.app: ~/Library/Application Support/LLMBar)
STATE = os.path.join(DATA, ".llm")
CFG_FILE = os.path.join(STATE, "config.json")
ENGINE_LOG = os.path.join(STATE, "engine.log")
ENGINE_PID = os.path.join(STATE, "engine.pid")
VENV_PY = os.path.join(DATA, ".venv", "bin", "python")
KILO_CFG = os.path.join(HOME, ".config", "kilo", "kilo.jsonc")
RAM_GB = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30

DEFAULTS = {
    "model": "",
    "dirs": [os.path.join(HOME, ".cache/huggingface/hub")],
    "port": 8090,
    "lan": False,
    "api_key": "",
    "auto_free": True,           # on start, quit vMLX/oMLX GUIs and Ollama
    "prefill_step": 1024,
    "speculative": True,         # prompt-lookup speculative decoding
    "idle_sleep": 15,            # minutes of inactivity before the model is unloaded (0 = never)
    "disk_cache_gb": 8,          # SSD prefix cache budget: GB number, "auto" (10% of free disk), or 0 = off
    "disk_cache_dir": "",        # where it lives (empty = ~/.cache/ai-llm-engine/prefix) — e.g. an external SSD
    "advanced": {},              # fine tuning: temperature, top_p, top_k, min_p, presence_penalty,
                                 # draft_max, ngram_min, cache_limit_mb (missing/null = engine default)
    "mem_limit": 88,             # % of system RAM the whole Mac may use; the engine trims/stops above it
    "lang": "auto",              # auto | en | ru
    "theme": "system",           # LLMBar: system | dark | light
    "glass": False,              # LLMBar: optional liquid-glass panel background
    "glass_transparency": 0.55,  # LLMBar: 0.15 (denser) … 0.85 (more see-through)
    "models": {},
}
MODEL_DEFAULTS = {
    "context": 0,                # prompt + output; 0 = auto (from RAM, model size and KV cost)
    "max_output": 8192,          # output cap (larger requests are clamped by the server)
    "thinking": None,            # None = whatever the model's chat template does; True/False = your choice
    "effort": "medium",
    "vision": True,
    "max_pixels": 1024 * 1024,
}

TTY = sys.stdout.isatty()
def c(code, s): return f"\033[{code}m{s}\033[0m" if TTY else str(s)
B, DIM = (lambda s: c("1", s)), (lambda s: c("2", s))
G, R, Y, CY = (lambda s: c("32", s)), (lambda s: c("31", s)), (lambda s: c("33", s)), (lambda s: c("36", s))
MINT = lambda s: c("1;38;2;77;242;194", s)  # the app's accent color (LLMBar, docs/banner.png)


def print_banner():
    """One-time welcome banner for the interactive menu (skipped when not a TTY)."""
    if not TTY:
        return
    title, sub = "  Ai>_   AI-llm-Engine", "  " + T("local MLX engine — fast, minimal, yours")
    width = max(len(title), len(sub)) + 1
    print(MINT("  ╭" + "─" * width + "╮"))
    print(MINT("  │") + B(title.ljust(width)) + MINT("│"))
    print(MINT("  │") + DIM(sub.ljust(width)) + MINT("│"))
    print(MINT("  ╰" + "─" * width + "╯"))


# ───────────────────────────── i18n ─────────────────────────────
_TR = {}


def set_lang(lang):
    global _TR
    if lang == "auto":
        lang = os.environ.get("LLM_LANG", "")
        if not lang:
            loc = subprocess.run(["defaults", "read", "-g", "AppleLocale"], capture_output=True, text=True).stdout
            lang = "ru" if (loc or os.environ.get("LANG", "")).lower().startswith("ru") else "en"
    _TR = {}
    if lang != "en":
        try:
            with open(os.path.join(BASE, "locales", f"{lang}.json"), encoding="utf-8") as f:
                _TR = json.load(f)
        except (OSError, ValueError):
            pass


def T(s):
    """Translate an English UI string (falls back to English)."""
    return _TR.get(s, s)


# ───────────────────────────── config ─────────────────────────────
def load_cfg():
    cfg = json.loads(json.dumps(DEFAULTS))
    try:
        with open(CFG_FILE) as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return cfg


def save_cfg(cfg):
    os.makedirs(STATE, exist_ok=True)
    with open(CFG_FILE, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def mcfg(cfg, path=None):
    m = dict(MODEL_DEFAULTS)
    m.update(cfg["models"].get(path or cfg["model"], {}))
    return m


def set_mcfg(cfg, key, value):
    cfg["models"].setdefault(cfg["model"], {})[key] = value


# ─────────────────────────── models ───────────────────────────
def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


NATIVE_TYPES = ("qwen3_5", "qwen3_5_text", "prism_hadamard_qwen35")
PTQ1_SCHEMA = "prism_hadamard_qwen35_ptq1"
TQ1_TYPE = "ternel_hadamard_qwen35"   # inductiveML 1.75-bit repacks (TQ1_G128): same engine option


def engine_route(cfgj, tags=(), jang=None):
    """Which engine runs this model: "native" (Qwen3.5-hybrid dense family, incl. Prism/JANG
    Hadamard bundles), "ptq1" (the same family in an experimental 1.75-bit ternary format: PTQ1_0
    or TQ1_G128) or "mlxlm"
    (anything else MLX, through the optional mlx-lm backend). `cfgj` is the local config.json;
    for a hub repo that has not been downloaded, pass its tags instead. `jang` is the bundle's
    jang_config.json: JANG's packed ternary storage (the "packed-trits" tag) runs as ptq1 too."""
    text = cfgj.get("text_config") or {}
    kinds = {cfgj.get("model_type"), text.get("model_type"), *tags}
    schema = str(cfgj.get("schema", ""))
    packed = (jang or {}).get("runtime", {}).get("requires_jang_ternary_packed_expansion") or "packed-trits" in kinds
    if schema.startswith(PTQ1_SCHEMA) or PTQ1_SCHEMA in kinds or TQ1_TYPE in kinds or packed:
        return "ptq1"   # checked before "native": a TQ1 repo also carries the qwen3_5 tag
    moe = text.get("num_experts") or cfgj.get("num_experts") or any("moe" in str(k) for k in kinds)
    if kinds & set(NATIVE_TYPES) and not moe:
        return "native"
    return "mlxlm"


def mlxlm_installed():
    sp = glob.glob(os.path.join(os.path.dirname(os.path.dirname(VENV_PY)), "lib", "python*", "site-packages", "mlx_lm"))
    return bool(sp)


def inspect_model(path):
    cfgj = _json(os.path.join(path, "config.json"))
    if not cfgj:
        return None
    files = os.listdir(path)
    weights = [f for f in files if f.endswith(".safetensors")]
    if not weights:
        return None
    size = sum(os.path.getsize(os.path.join(path, f)) for f in weights) / 2**30
    jang = _json(os.path.join(path, "jang_config.json"))
    text = cfgj.get("text_config", cfgj)
    q = cfgj.get("quantization") or cfgj.get("quantization_config") or {}
    mtype = text.get("model_type") or cfgj.get("model_type", "?")

    route = engine_route(cfgj, jang=jang)
    problems = []
    rt = jang.get("runtime", {})
    if rt.get("requires_jang_affine1_expansion"):
        problems.append(T("packed JANG storage"))  # 1-bit affine: no kernel for it

    tpl = ""
    if "chat_template.jinja" in files:
        tpl = open(os.path.join(path, "chat_template.jinja"), errors="replace").read()
    tpl = tpl or str(_json(os.path.join(path, "tokenizer_config.json")).get("chat_template", ""))
    efforts = jang.get("reasoning", {}).get("supported_reasoning_efforts") or \
        sorted(set(re.findall(r"reasoning_effort\s*==\s*'(\w+)'", tpl)))

    name = os.path.basename(path.rstrip("/"))
    parts = path.split(os.sep)
    if "snapshots" in parts:  # standard HF cache: models--org--name/snapshots/<hash>
        name = parts[parts.index("snapshots") - 1].removeprefix("models--").split("--", 1)[-1]
    fmt = f"JANG {jang.get('profile', '')}".strip() if jang else (f"MLX {q['bits']}-bit" if q.get("bits") else "fp")
    if route == "ptq1" and not jang:
        fmt = "TQ1" if cfgj.get("model_type") == TQ1_TYPE else "PTQ1"
    return {"path": path, "name": name, "size": size, "format": fmt, "type": mtype, "route": route,
            "vision": bool(cfgj.get("vision_config")), "thinking": "enable_thinking" in tpl,
            "thinking_default": "enable_thinking" in tpl and template_thinks_by_default(tpl),
            "efforts": efforts, "hadamard": bool(cfgj.get("hadamard") or jang.get("hadamard")),
            "supported": not problems, "problems": problems}


def scan_models(cfg):
    found, seen = [], set()
    for root in cfg["dirs"]:
        root = os.path.expanduser(root)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("blobs", "refs")]
            if dirpath[len(root):].count(os.sep) > 5:
                dirnames[:] = []
                continue
            if "config.json" in filenames:
                info = inspect_model(dirpath)
                if info:
                    key = os.path.realpath(os.path.join(dirpath, "config.json"))
                    if key not in seen:
                        seen.add(key)
                        found.append(info)
                    dirnames[:] = []
    return sorted(found, key=lambda m: (not m["supported"], m["name"].lower()))


def current_model(cfg):
    p = cfg["model"]
    return inspect_model(p) if p and os.path.isdir(p) else None


def template_thinks_by_default(tpl):
    """Does this chat template reason when nobody passes enable_thinking? Bonsai-27B's says
    `enable_thinking is undefined or enable_thinking is true`; the small Qwen3.5 models
    only open a <think> block under `is defined and enable_thinking is true`. (The engine
    renders the template to decide; llm.py is stdlib-only, so this reads the source.)"""
    return not re.search(r"enable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+true", tpl)


def think_enabled(mc, m):
    """The reasoning mode actually in effect for a model: the user's choice, else its default."""
    if not m["thinking"]:
        return False
    return m["thinking_default"] if mc["thinking"] is None else bool(mc["thinking"])


def served_name(m):
    return re.sub(r"[^a-z0-9._-]+", "-", m["name"].lower()).strip("-")


def fit(m):
    need, limit = m["size"] + 2.0, RAM_GB * 0.75
    if need <= limit - 1: return G(T("✓ fits"))
    if need <= limit + 0.5: return Y(T("~ tight"))
    return R(T("✗ won't fit"))


def list_models(cfg, models):
    if not models:
        print(Y(T("No models found in:") + " " + ", ".join(cfg["dirs"]))); return
    print(B(f"\n── {T('Models')} (RAM {RAM_GB:.0f} GB) ──"))
    for i, m in enumerate(models, 1):
        cur = G(" ◀ " + T("selected")) if m["path"] == cfg["model"] else ""
        feats = [f for f, on in (("vision", m["vision"]), ("thinking", m["thinking"]), ("hadamard", m["hadamard"])) if on]
        ok = fit(m) if m["supported"] else R(T("✗ unsupported:") + " " + ", ".join(m["problems"]))
        print(f" {i:>2}) {B(m['name'])}{cur}")
        print(f"     {m['size']:.1f} GB · {m['format']} · {m['type']}{' · ' + ', '.join(feats) if feats else ''} · {ok}")


def choose_model(cfg, arg=None):
    models = scan_models(cfg)
    if arg is None:
        list_models(cfg, models)
        if not models: return False
        arg = input(T("Model number (Enter to cancel)") + " › ").strip()
        if not arg: return False
    if arg.isdigit() and 1 <= int(arg) <= len(models):
        pick = models[int(arg) - 1]
    else:
        hits = [m for m in models if arg.lower() in m["name"].lower()]
        pick = hits[0] if len(hits) == 1 else None
        if len(hits) > 1: print(Y(T("Several matches:") + " " + ", ".join(h["name"] for h in hits)))
    if not pick:
        print(R(T("Model not found."))); return False
    if not pick["supported"]:
        print(R(T("The engine does not support this model:") + " " + ", ".join(pick["problems"]))); return False
    was_running = bool(read_pid(ENGINE_PID))
    cfg["model"] = pick["path"]
    if pick["path"] not in cfg["models"]:
        cfg["models"][pick["path"]] = dict(MODEL_DEFAULTS, effort="medium" if "medium" in pick["efforts"] else "")
    save_cfg(cfg)
    print(G(T("Selected:") + f" {pick['name']}"))
    if was_running:
        restart(cfg, sys.stdin.isatty())
    return True


def model_root(path):
    """The folder that holds a model on disk: models--org--name for the HF cache layout (path is a
    snapshot inside it), else the model folder itself."""
    parts = path.rstrip(os.sep).split(os.sep)
    if "snapshots" in parts:
        return os.sep.join(parts[:parts.index("snapshots")])
    return path


def delete_model(cfg, m, assume_yes=False):
    """Remove a downloaded model from disk. Refuses anything outside the configured model folders,
    and the model the running engine has loaded."""
    root = os.path.realpath(model_root(m["path"]))
    homes = [os.path.realpath(os.path.expanduser(d)) for d in cfg["dirs"]]
    if root in homes or not any(root.startswith(h + os.sep) for h in homes) or not os.path.isdir(root):
        print(R(T("Refusing to delete a folder outside the model directories:") + f" {root}"))
        return False
    if m["path"] == cfg["model"] and read_pid(ENGINE_PID) and health(cfg):
        print(R(T("This model is loaded. Stop the engine first (./llm stop).")))
        return False
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs
               if os.path.isfile(os.path.join(d, f))) / 2**30
    if not assume_yes and not ask_yn(R(T("Delete {name} ({size:.1f} GB) from disk?").format(name=m["name"], size=size)), False):
        return False
    shutil.rmtree(root)
    if cfg["model"] == m["path"]:
        cfg["model"] = ""
    cfg["models"].pop(m["path"], None)
    save_cfg(cfg)
    print(G(T("Deleted: {name} — {size:.1f} GB freed").format(name=m["name"], size=size)))
    return True


def remove_cmd(cfg, args):
    assume_yes = "--yes" in args or "-y" in args
    arg = " ".join(a for a in args if a not in ("--yes", "-y")).strip()
    models = scan_models(cfg)
    if not arg:
        list_models(cfg, models)
        if not models or not sys.stdin.isatty():
            return False
        arg = input(T("Model number to delete (Enter to cancel)") + " › ").strip()
        if not arg:
            return False
    if arg.isdigit() and 1 <= int(arg) <= len(models):
        hits = [models[int(arg) - 1]]
    else:
        hits = [m for m in models if arg == m["path"] or arg.lower() in m["name"].lower()]
    if len(hits) != 1:
        print(R(T("Model not found.")) if not hits else Y(T("Several matches:") + " " + ", ".join(h["name"] for h in hits)))
        return False
    return delete_model(cfg, hits[0], assume_yes)


# ─────────────────────────── processes ───────────────────────────
def read_pid(path):
    try:
        with open(path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def kill_pid(pid, wait=15):
    try: os.killpg(pid, signal.SIGTERM)
    except OSError:
        try: os.kill(pid, signal.SIGTERM)
        except OSError: return
    for _ in range(wait * 10):
        try: os.kill(pid, 0)
        except OSError: return
        time.sleep(0.1)
    try: os.killpg(pid, signal.SIGKILL)
    except OSError: pass


def pgrep(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if int(p) != os.getpid()]


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def http_json(url, data=None, timeout=5, key=""):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def health(cfg, timeout=3):
    try:
        return http_json(f"http://127.0.0.1:{cfg['port']}/health", timeout=timeout)
    except Exception:
        return None


def mem_info():
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", vm).group(1))
    get = lambda k: int(m.group(1)) if (m := re.search(rf"{k}:\s+(\d+)", vm)) else 0
    avail = (get("Pages free") + get("Pages inactive") + get("Pages speculative") + get("Pages purgeable")) * page / 2**30
    m = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
    return avail, (float(m.group(1)) / 1024 if m else 0.0)


def tail(path, n):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


# ─────────────────────────── engine ───────────────────────────
def find_python():
    """A python3 >= 3.10 for the engine venv (the manager itself runs on 3.9+)."""
    for p in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3", shutil.which("python3") or "", sys.executable):
        if p and os.path.exists(p):
            v = subprocess.run([p, "-c", "import sys; print(sys.version_info >= (3, 10))"], capture_output=True, text=True)
            if v.stdout.strip() == "True":
                return p
    raise SystemExit(T("Python 3.10+ is required for the engine (e.g. brew install python)."))


def setup(compat=False):
    """Create/refresh the engine environment; `compat` also installs mlx-lm (the optional backend
    for architectures the native engine lacks — Qwen2/3, GLM, Llama, Gemma, MoE…)."""
    if os.path.exists(VENV_PY):
        print(DIM(T("Environment already exists: .venv")))
    else:
        print(T("Creating .venv…"))
        os.makedirs(DATA, exist_ok=True)
        subprocess.run([find_python(), "-m", "venv", os.path.join(DATA, ".venv")], check=True)
    r = subprocess.run([VENV_PY, "-m", "pip", "install", "-q", "-r", os.path.join(BASE, "requirements.txt")])
    if r.returncode == 0 and compat:
        print(T("Installing mlx-lm…"))
        r = subprocess.run([VENV_PY, "-m", "pip", "install", "-q", "mlx-lm"])
    if r.returncode == 0:
        print(G(T("Dependencies installed.")))
    return r.returncode == 0


def engine_cmd(cfg, m):
    mc = mcfg(cfg, m["path"])
    cmd = [VENV_PY, "-m", "engine.server", "--model", m["path"], "--name", served_name(m),
           "--host", "0.0.0.0" if cfg["lan"] else "127.0.0.1", "--port", str(cfg["port"]),
           "--context", str(mc["context"]), "--max-tokens", str(mc["max_output"]),
           "--prefill-step", str(cfg["prefill_step"]), "--max-pixels", str(mc["max_pixels"]),
           "--mem-limit", str(cfg["mem_limit"] / 100),
           "--stats-file", os.path.join(STATE, "stats.jsonl"),
           "--idle-sleep", str(cfg.get("idle_sleep", 15)), "--disk-cache-gb", str(cfg.get("disk_cache_gb", 8))]
    if m.get("route") == "ptq1":  # picking an experimental-format model is the opt-in; no separate switch
        cmd += ["--ptq1", "on"]
    if mc["thinking"] is not None:
        cmd += ["--thinking", "on" if mc["thinking"] else "off"]
    if cfg.get("disk_cache_dir"):
        cmd += ["--disk-cache-dir", cfg["disk_cache_dir"]]
    if not cfg.get("speculative", True):
        cmd.append("--no-speculative")
    for k, v in (cfg.get("advanced") or {}).items():
        if v is not None and v != "":
            cmd += ["--" + k.replace("_", "-"), str(v)]
    if mc["effort"]:
        cmd += ["--effort", mc["effort"]]
    if not mc["vision"]:
        cmd.append("--no-vision")
    return cmd


def route_blocker(cfg, m, interactive):
    """None when the model's route is ready; else a message (after offering the fix, if interactive).
    Only the optional mlx-lm backend needs anything installed; PTQ1 is enabled by the model itself."""
    route = m.get("route", "native")
    if route == "mlxlm" and not mlxlm_installed():
        if interactive and ask_yn(Y(T("This model runs on the optional mlx-lm backend. Install it into .venv now?")), True):
            if setup(compat=True):
                return None
        return T("This model runs on the optional mlx-lm backend. Install it with:  ./llm setup --compat")
    return None


def start(cfg, interactive=True):
    os.makedirs(STATE, exist_ok=True)
    if not os.path.exists(VENV_PY) and not setup():
        return False
    m = current_model(cfg)
    if not m and not interactive:  # e.g. LLMBar: take the first supported model that fits
        ok = [x for x in scan_models(cfg) if x["supported"] and x["size"] + 2 <= RAM_GB * 0.75]
        if ok and choose_model(cfg, ok[0]["name"]):
            m = current_model(cfg)
    if not m:
        print(Y(T("No model selected.")))
        if not interactive or not choose_model(cfg): return False
        m = current_model(cfg)
    if read_pid(ENGINE_PID) and health(cfg):
        print(Y(T("Engine is already running."))); return True
    why = route_blocker(cfg, m, interactive)
    if why:
        print(R(why)); return False
    if cfg["auto_free"]:
        free_memory(interactive=False, quiet=True)
    if port_busy(cfg["port"]):
        stale = read_pid(ENGINE_PID)  # our own process, just stuck/crashed/still loading — clear it, don't just refuse
        if stale:
            print(Y(T("Port {port} was held by a stuck engine (PID {pid}); stopping it.")
                    .format(port=cfg["port"], pid=stale)))
            kill_pid(stale)
            try: os.remove(ENGINE_PID)
            except OSError: pass
            time.sleep(0.5)
        if port_busy(cfg["port"]):
            print(R(T("Port {port} is busy. Change it in settings.").format(port=cfg["port"]))); return False
    avail, _ = mem_info()
    if m["size"] + 1.5 > avail:
        print(Y(T("⚠ The model needs ~{need:.1f} GB, {avail:.1f} GB available — expect swapping.")
                .format(need=m["size"] + 1.5, avail=avail)))
    cmd = engine_cmd(cfg, m)
    with open(ENGINE_LOG, "a") as log:
        log.write(f"\n===== {time.ctime()} =====\n{' '.join(cmd)}\n")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if cfg["api_key"]:
        env["ENGINE_API_KEY"] = cfg["api_key"]
    with open(ENGINE_LOG, "a") as logf:
        p = subprocess.Popen(cmd, cwd=BASE, stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    with open(ENGINE_PID, "w") as f:
        f.write(str(p.pid))
    print(T("Loading {name} ({size:.1f} GB)").format(name=B(m["name"]), size=m["size"]), end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            print(R("\n" + T("✗ Engine exited (code {code}):").format(code=p.returncode)))
            print(tail(ENGINE_LOG, 15)); return False
        if health(cfg, 1):
            break
        print(".", end="", flush=True); time.sleep(1)
    else:
        kill_pid(p.pid)
        try: os.remove(ENGINE_PID)
        except OSError: pass
        print(R("\n" + T("✗ Load timed out, process stopped."))); return False
    print(G(" " + T("ready in {s:.0f} s").format(s=time.time() - t0)))
    show_endpoints(cfg, m)
    return True


def stop(cfg, quiet=False):
    pid = read_pid(ENGINE_PID)
    if pid:
        kill_pid(pid)
        if not quiet: print(T("Engine stopped (PID {pid})").format(pid=pid))
    elif not quiet:
        print(DIM(T("Nothing is running.")))
    try: os.remove(ENGINE_PID)
    except OSError: pass


def restart(cfg, interactive=True):
    stop(cfg, quiet=True)
    time.sleep(0.5)
    return start(cfg, interactive)


HOGS = [("vMLX", "vMLX.app/Contents/MacOS|vmlx_engine"), ("oMLX", "oMLX.app/Contents/MacOS|omlx-server"),
        ("Ollama", "Ollama.app/Contents/MacOS|ollama serve")]


def free_memory(interactive=True, quiet=False):
    stopped = []
    for name, pat in HOGS:
        pids = pgrep(pat)
        if not pids or (interactive and not ask_yn(T("Stop {name} (PID {pids})?").format(
                name=name, pids=", ".join(map(str, pids))), True)):
            continue
        subprocess.run(["osascript", "-e", f'quit app "{name}"'], capture_output=True, timeout=10)
        time.sleep(1)
        for p in pgrep(pat):
            try: os.kill(p, signal.SIGTERM)
            except OSError: pass
        stopped.append(name)
    if stopped:
        print(G(T("Stopped:") + " " + ", ".join(stopped)))
    elif not quiet:
        print(DIM(T("No memory hogs found.")))
    if not quiet or stopped:
        avail, swap = mem_info()
        print(T("Available RAM: {avail:.1f} GB, swap {swap:.1f} GB").format(avail=avail, swap=swap))


# ─────────────────────────── chat / bench ───────────────────────────
def chat(cfg):
    m = current_model(cfg)
    if not m or not health(cfg):
        print(R(T("The model is not running."))); return
    mc = mcfg(cfg)
    url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"
    think, effort, show = think_enabled(mc, m), mc["effort"], True
    msgs, system = [], ""
    print(B(T("Chat with {name}.").format(name=m["name"])) +
          DIM(" /think on|off  /effort <lvl>  /show  /system <text>  /image <path> <question>"))
    print(DIM("  /clear  /save  /exit   " + T("multi-line input:") + ' """ … """'))
    while True:
        tag = (f"think:{effort}" if think else "no-think") if m["thinking"] else "chat"
        try:
            line = input(CY(f"\n[{tag}] › "))
        except (EOFError, KeyboardInterrupt):
            print(); return
        if line.strip() == '"""':
            buf = []
            while True:
                try: l = input()
                except EOFError: break
                if l.strip() == '"""': break
                buf.append(l)
            line = "\n".join(buf)
        s = line.strip()
        if not s: continue
        if s in ("/exit", "/q"): return
        if s == "/clear": msgs = []; print(DIM(T("History cleared."))); continue
        if s.startswith("/think"): think = not s.endswith("off"); continue
        if s.startswith("/effort"):
            v = s.split()[-1]
            if v in m["efforts"]: effort = v
            else: print(R(" | ".join(m["efforts"]) or T("not supported")))
            continue
        if s == "/show": show = not show; continue
        if s.startswith("/system"): system = s[7:].strip(); continue
        if s == "/save":
            fn = os.path.join(DATA, f"chat-{time.strftime('%Y%m%d-%H%M%S')}.md")
            with open(fn, "w") as f:
                for x in msgs: f.write(f"## {x['role']}\n\n{x['content'] if isinstance(x['content'], str) else '[image]'}\n\n")
            print(DIM(T("Saved:") + f" {fn}")); continue
        content = s
        if s.startswith("/image "):
            _, path, *q = s.split(" ", 2)
            path = os.path.expanduser(path)
            if not os.path.exists(path): print(R(T("File not found"))); continue
            content = [{"type": "image_url", "image_url": {"url": os.path.abspath(path)}},
                       {"type": "text", "text": q[0] if q else T("Describe the image.")}]
        msgs.append({"role": "user", "content": content})
        payload = {"model": served_name(m), "stream": True, "stream_options": {"include_usage": True},
                   "messages": ([{"role": "system", "content": system}] if system else []) + msgs,
                   "chat_template_kwargs": {"enable_thinking": think, **({"reasoning_effort": effort} if effort else {})}}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        answer, reasoning, in_think, t0, first, usage = [], [], False, time.time(), None, None
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                for raw in r:
                    raw = raw.strip()
                    if not raw.startswith(b"data:") or raw == b"data: [DONE]": continue
                    ev = json.loads(raw[5:])
                    usage = ev.get("usage") or usage
                    if ev.get("error"): print(R(ev["error"].get("message"))); break
                    for ch in ev.get("choices", []):
                        d = ch.get("delta", {})
                        if d.get("reasoning_content"):
                            first = first or time.time()
                            reasoning.append(d["reasoning_content"])
                            if show:
                                if not in_think: print(DIM("💭 "), end=""); in_think = True
                                print(DIM(d["reasoning_content"]), end="", flush=True)
                        if d.get("content"):
                            first = first or time.time()
                            if in_think: print("\n"); in_think = False
                            if not answer: print(G("› "), end="")
                            answer.append(d["content"]); print(d["content"], end="", flush=True)
        except KeyboardInterrupt:
            print(Y("\n[" + T("interrupted") + "]"))
        except Exception as e:
            print(R("\n" + T("Error:") + f" {e}")); msgs.pop(); continue
        print()
        if usage:
            dt = time.time() - (first or t0)
            ct = usage.get("completion_tokens", 0)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            print(DIM(T("— {inp} in ({cached} cached) / {out} out · {tps:.1f} tok/s · first token {ttft:.1f} s").format(
                inp=usage.get("prompt_tokens"), cached=cached, out=ct, tps=ct / dt if dt > 0 else 0, ttft=(first or t0) - t0)))
        msgs.append({"role": "assistant", "content": "".join(answer), "reasoning_content": "".join(reasoning)})


def bench(cfg):
    m = current_model(cfg)
    if not m or not health(cfg):
        print(R(T("The model is not running."))); return
    base = f"http://127.0.0.1:{cfg['port']}"
    body = {"model": "x", "max_tokens": 256, "temperature": 0, "enable_thinking": False,
            "messages": [{"role": "user", "content": "Write a detailed story about Japanese bonsai gardens."}]}
    print(T("Generating 256 tokens without reasoning…"))
    http_json(f"{base}/v1/chat/completions", body, timeout=900, key=cfg["api_key"])
    st = (health(cfg) or {}).get("last", {})
    print(G(T("prefill {p} tok/s · decode {d} tok/s · peak {g} GB").format(
        p=st.get("prefill_tps"), d=st.get("decode_tps"), g=st.get("peak_gb"))))


def run_tests():
    env = dict(os.environ)
    m = current_model(load_cfg())
    if m and not read_pid(ENGINE_PID):
        env["ENGINE_TEST_MODEL"] = m["path"]
    elif m:
        print(Y(T("Engine is running — integration tests skipped (two model copies won't fit).")))
    return subprocess.run([VENV_PY, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=BASE, env=env).returncode == 0


def find_editor():
    """LLMEditor.app: next to LLMBar's bundled resources, or built in this folder (./build-editor.sh)."""
    for p in (os.path.join(BASE, "LLMEditor.app"), os.path.join(os.path.dirname(BASE), "LLMEditor.app")):
        if os.path.isdir(p):
            return p
    return None


def run_ui(args=()):
    """Open the editor on a project (default: the folder `./llm ui` was started from)."""
    app = find_editor()
    if not app:
        print(R(T("The editor is not built yet. Build it once with:  ./build-editor.sh"))); return False
    cmd = ["open", "-n", app, "--args", "--llm-root", BASE, "--engine-port", str(load_cfg()["port"])]
    if "--choose" not in args:   # without --root the app reopens its last project, or asks
        root = next((a for a in args if not a.startswith("--")), None) or os.environ.get("LLM_CWD") or os.getcwd()
        cmd += ["--root", os.path.abspath(os.path.join(os.environ.get("LLM_CWD") or os.getcwd(), root))]
    return subprocess.run(cmd).returncode == 0


def run_perf(args=()):
    if read_pid(ENGINE_PID):
        print(Y(T("Stop the engine first: two model copies won't fit."))); return False
    env = dict(os.environ)
    m = current_model(load_cfg())
    if m:
        env["ENGINE_TEST_MODEL"] = m["path"]
    return subprocess.run([VENV_PY, "-m", "tests.perf", *args], cwd=BASE, env=env).returncode == 0


def run_eval(args=()):
    if read_pid(ENGINE_PID):
        print(Y(T("Stop the engine first: two model copies won't fit."))); return False
    env = dict(os.environ)
    m = current_model(load_cfg())
    if m:
        env["ENGINE_TEST_MODEL"] = m["path"]
    return subprocess.run([VENV_PY, "-m", "tests.eval", *args], cwd=BASE, env=env).returncode == 0


def run_calibrate(args=()):
    if read_pid(ENGINE_PID):
        print(Y(T("Stop the engine first: two model copies won't fit."))); return False
    env = dict(os.environ)
    m = current_model(load_cfg())
    if m:
        env["ENGINE_TEST_MODEL"] = m["path"]
    return subprocess.run([VENV_PY, "-m", "tests.calibrate", *args], cwd=BASE, env=env).returncode == 0


# ─────────────────────────── clients ───────────────────────────
def show_endpoints(cfg, m=None):
    m = m or current_model(cfg)
    host = "localhost"
    if cfg["lan"]:
        host = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True).stdout.strip() or host
    mc = dict(mcfg(cfg))
    h = health(cfg)
    if h:
        mc["context"], mc["max_output"] = h.get("context"), h.get("max_tokens")
    print(B("\n── " + T("Connect") + " ──"))
    url = "http://%s:%d/v1" % (host, cfg["port"])
    print(f"OpenAI API : {CY(url)}   model = {CY(served_name(m) if m else '?')} ({T('any name works')})")
    print(DIM(T("context {ctx}, output ≤ {out} (larger max_tokens is clamped by the server)").format(
        ctx=mc["context"], out=mc["max_output"])))


def _read_jsonc(path):
    with open(path) as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except ValueError:
        no_comments = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/', lambda m: m.group(1) or "", txt, flags=re.S)
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", no_comments))


CLIENTS = {
    "continue": ("Continue", os.path.join(HOME, ".continue", "config.yaml")),
    "kilo": ("Kilo Code", KILO_CFG),
    "opencode": ("opencode", os.path.join(HOME, ".config", "opencode", "opencode.json")),
}


def client_limits(cfg, m):
    """Context / output limits for client configs: the running engine's values, else an estimate."""
    mc = dict(mcfg(cfg))
    h = health(cfg)
    if h:
        return h.get("context") or 32768, h.get("max_tokens") or mc["max_output"]
    ctx = mc["context"]
    if not ctx:  # same formula as the engine's auto context
        t = _json(os.path.join(m["path"], "config.json"))
        t = t.get("text_config", t)
        n_full = t.get("num_hidden_layers", 64) // t.get("full_attention_interval", 4)
        kv = n_full * 2 * t.get("num_key_value_heads", 4) * t.get("head_dim", 256) * 2
        budget = RAM_GB * 2**30 * cfg["mem_limit"] / 100 - m["size"] * 2**30 - 4 * 2**30
        ctx = int(max(4096, min(t.get("max_position_embeddings", 32768), budget // kv // 4096 * 4096)))
    return ctx, min(mc["max_output"], ctx // 2)


def _write_opencode_style(path, cfg, m, ctx, out, alias=()):
    """Kilo Code and opencode share the opencode provider schema."""
    vision = mcfg(cfg)["vision"] and m["vision"]
    data = {}
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")
        data = _read_jsonc(path)
    prov = data.setdefault("provider", {})
    opts = {"baseURL": f"http://localhost:{cfg['port']}/v1"}
    if cfg["api_key"]:
        opts["apiKey"] = cfg["api_key"]
    name = served_name(m)
    prov["local_mlx"] = {
        "name": "AI-llm-Engine (local)", "npm": "@ai-sdk/openai-compatible", "options": opts,
        "models": {name: {
            "name": f"{m['name']} (local)", "tool_call": True, "reasoning": m["thinking"], "attachment": vision,
            "modalities": {"input": ["text", "image"] if vision else ["text"], "output": ["text"]},
            "limit": {"context": ctx, "output": out}}},
    }
    for a in alias:  # old provider ids kept as mirrors so already-open editor sessions keep resolving
        prov[a] = prov["local_mlx"]
    data["model"] = f"local_mlx/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


BEGIN, END = "# >>> AI-llm-Engine (managed by ./llm connect continue)", "# <<< AI-llm-Engine"


def _write_continue(path, cfg, m, ctx, out):
    """Insert/replace our model block in ~/.continue/config.yaml without a YAML library."""
    vision = mcfg(cfg)["vision"] and m["vision"]
    caps = "[tool_use, image_input]" if vision else "[tool_use]"
    block = "\n".join([
        f"  {BEGIN}",
        f"  - name: {m['name']} (local)",
        "    provider: openai",
        f"    model: {served_name(m)}",
        f"    apiBase: http://localhost:{cfg['port']}/v1",
        f"    apiKey: {cfg['api_key'] or 'none'}",
        "    roles: [chat, edit, apply]",
        f"    capabilities: {caps}",
        "    defaultCompletionOptions:",
        f"      contextLength: {ctx}",
        f"      maxTokens: {out}",
        f"  {END}",
    ]) + "\n"
    text = ""
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")
        with open(path) as f:
            text = f.read()
    if BEGIN in text and END in text:
        a = text.index(BEGIN)
        a = text.rfind("\n", 0, a) + 1
        b = text.index("\n", text.index(END)) + 1
        text = text[:a] + block + text[b:]
    elif re.search(r"^models:\s*\[\]\s*$", text, re.M):
        text = re.sub(r"^models:\s*\[\]\s*$", "models:\n" + block.rstrip("\n"), text, count=1, flags=re.M)
    elif re.search(r"^models:\s*$", text, re.M):
        mt = re.search(r"^models:\s*$\n?", text, re.M)
        text = text[:mt.end()] + block + text[mt.end():]
    elif text.strip():
        text = text.rstrip("\n") + "\n\nmodels:\n" + block
    else:
        text = "name: Local Config\nversion: 0.0.1\nschema: v1\n\nmodels:\n" + block
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def connect_client(cfg, which=None):
    """Register the model in an editor/agent client of the user's choice."""
    m = current_model(cfg)
    if not m:
        print(R(T("Select a model first."))); return False
    if not which:
        keys = list(CLIENTS)
        for i, k in enumerate(keys, 1):
            print(f"  {i}) {CLIENTS[k][0]}  " + DIM(CLIENTS[k][1]))
        ch = input(T("Client number (Enter to cancel)") + " › ").strip()
        if not ch.isdigit() or not 1 <= int(ch) <= len(keys):
            return False
        which = keys[int(ch) - 1]
    if which not in CLIENTS:
        print(R(T("Unknown client:") + f" {which} ({', '.join(CLIENTS)})")); return False
    title, path = CLIENTS[which]
    ctx, out = client_limits(cfg, m)
    if which == "continue":
        _write_continue(path, cfg, m, ctx, out)
    else:
        _write_opencode_style(path, cfg, m, ctx, out, alias=("local_qwen",) if which == "kilo" else ())
    print(G(T("{client} configured: {name}, context {ctx}, output {out}").format(
        client=title, name=served_name(m), ctx=ctx, out=out)))
    print(DIM(T("File: {path} (backup: .bak) · reload the editor window to apply.").format(path=path)))
    return True


# ─────────────────────────── model hub ───────────────────────────
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")  # e.g. https://hf-mirror.com
PULL_SKIP = (".png", ".jpg", ".jpeg", ".gif", ".mp4", ".gguf", ".bin", ".pt", ".pth", ".onnx", ".h5", ".msgpack")


def _hf(path, timeout=20):
    req = urllib.request.Request(HF_ENDPOINT + path, headers={"User-Agent": "AI-llm-Engine"})
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def hub_route(tags):
    """Engine route for a hub repo from its tags: native / ptq1 / mlxlm, or "none" for what no
    backend here can read (GGUF, non-MLX weights)."""
    t = set(tags or [])
    if "gguf" in t or not (t & {"mlx", "safetensors"}):
        return "none"
    return engine_route({}, t)


def _pull_dest(repo):
    return os.path.join(HOME, ".cache", "huggingface", "hub", *repo.split("/"))


def install_state(repo, siblings=None):
    """"complete" | "partial" | "none" for a repo's download folder. With the hub listing (sizes) every
    wanted file must be there at its full size; without it, a folder with a config, weights and no
    .part files counts as complete. A folder that merely exists is NOT a finished download."""
    dest = _pull_dest(repo)
    if not os.path.isdir(dest):
        return "none"
    if siblings:
        for f in siblings:
            name = f["rfilename"]
            if name.lower().endswith(PULL_SKIP):
                continue
            q = os.path.join(dest, name)
            if not os.path.isfile(q) or (f.get("size") and os.path.getsize(q) != f["size"]):
                return "partial"
        return "complete"
    parts = [n for _, _, fs in os.walk(dest) for n in fs if n.endswith(".part")]
    has = os.path.isfile(os.path.join(dest, "config.json")) and any(
        n.endswith(".safetensors") for n in os.listdir(dest))
    return "complete" if has and not parts else "partial"


def hf_search(query, limit=20):
    """MLX models on the hub, most downloaded first, with size and engine compatibility."""
    from concurrent.futures import ThreadPoolExecutor
    found = _hf(f"/api/models?search={urllib.parse.quote(query)}&filter=mlx&sort=downloads&direction=-1&limit={limit}")

    def probe(repo):
        """(size in GB, storage format the engine can't read yet?, install state) for one repo."""
        try:
            info = _hf(f"/api/models/{repo}?blobs=true", timeout=15)
        except Exception:
            return None, False, install_state(repo)
        names = {f["rfilename"] for f in info.get("siblings", [])}
        size = sum(f.get("size") or 0 for f in info.get("siblings", [])
                   if not f["rfilename"].lower().endswith(PULL_SKIP)) / 2**30
        unsupported = False
        if "jang_config.json" in names:  # JANG bundles may use packed storage that needs expansion
            try:
                rt = _hf(f"/{repo}/resolve/main/jang_config.json", timeout=15).get("runtime", {})
                unsupported = bool(rt.get("requires_jang_affine1_expansion"))  # packed ternary runs (ptq1)
            except Exception:
                pass
        return size, unsupported, install_state(repo, info.get("siblings", []))
    with ThreadPoolExecutor(8) as ex:
        probes = list(ex.map(probe, [m["id"] for m in found]))
    sizes = [p_[0] for p_ in probes]
    return [{"id": m["id"], "downloads": m.get("downloads", 0), "likes": m.get("likes", 0), "size": sz,
             "route": "none" if pr[1] else hub_route(m.get("tags")),
             "compatible": hub_route(m.get("tags")) != "none" and not pr[1],
             "local": pr[2] == "complete", "partial": pr[2] == "partial",
             "fits": sz is not None and sz + 2.0 <= RAM_GB * 0.75} for m, sz, pr in zip(found, sizes, probes)]


def download_file(url, path, size, sha256=None, headers=None, on_chunk=None, retries=6):
    """Fetch `url` into `path`, resuming with Range, and never leave a short or corrupt file
    under the final name. `size` / `sha256` come from the hub listing (0 / None = unknown).

    urllib's read(n) returns b"" on a connection that closed early instead of raising, so a
    clean end of the read loop proves nothing: the size is checked and, when short, the
    transfer is resumed. A server that answers a Range request with 200 (the whole file)
    would otherwise be appended to the partial — start over in that case."""
    on_chunk = on_chunk or (lambda n: None)
    part = path + ".part"
    if os.path.exists(path):
        if (not size or os.path.getsize(path) == size) and not sha256:
            return
        if size and os.path.getsize(path) == size:
            if _sha256(path) == sha256:
                return
            os.remove(path)  # same size, wrong content
        else:
            os.replace(path, part)  # a short file under the final name: resume it
    attempt = 0
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size and have > size:
            os.remove(part)
            have = 0
        if size and have == size:
            break
        req = urllib.request.Request(url, headers=dict(headers or {}))
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and getattr(r, "status", 206) != 206:
                    have = 0
                with open(part, "ab" if have else "wb") as out:
                    while chunk := r.read(1 << 20):
                        out.write(chunk)
                        on_chunk(len(chunk))
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(min(2 ** attempt, 30))
            continue
        got = os.path.getsize(part)
        if not size or got == size:
            break
        attempt += 1
        if attempt > retries:
            raise OSError(f"the connection kept closing early ({got} of {size} bytes)")
    if sha256 and _sha256(part) != sha256:
        os.remove(part)
        raise ValueError("checksum mismatch — the partial download was discarded, run it again")
    os.replace(part, path)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def hf_pull(repo, progress=False):
    """Download a repo into ~/.cache/huggingface/hub/<org>/<name> with resume (Range) and atomic renames."""
    info = _hf(f"/api/models/{repo}?blobs=true")
    files = [f for f in info.get("siblings", []) if not f["rfilename"].lower().endswith(PULL_SKIP)]
    total = sum(f.get("size") or 0 for f in files)
    dest = _pull_dest(repo)
    os.makedirs(dest, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if total > free - 2 * 2**30:
        print(R(T("Not enough disk space: need {need:.1f} GB, free {free:.1f} GB").format(need=total / 2**30, free=free / 2**30)))
        return False
    print(T("Downloading {repo} ({size:.1f} GB) → {dest}").format(repo=repo, size=total / 2**30, dest=dest), flush=True)
    state = {"done": 0, "last": 0.0}
    t0, rev = time.time(), info.get("sha", "main")
    headers = {"User-Agent": "AI-llm-Engine"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]

    def tick(n, name):
        state["done"] += n
        now = time.time()
        if now - state["last"] > 0.5:
            state["last"] = now
            if progress:
                print(f"PROGRESS {min(state['done'], total)} {total} {name}", flush=True)
            else:
                speed = state["done"] / max(1e-6, now - t0) / 2**20
                print(f"\r  {min(state['done'], total) / 2**30:5.2f} / {total / 2**30:.2f} GB  {speed:6.1f} MB/s  "
                      f"{name[:40]:40s}", end="", flush=True)

    for f in files:
        name, size = f["rfilename"], f.get("size") or 0
        path = os.path.join(dest, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for q in (path, path + ".part"):  # what's already on disk counts as progress
            if os.path.exists(q):
                state["done"] += min(os.path.getsize(q), size or os.path.getsize(q))
                break
        url = f"{HF_ENDPOINT}/{repo}/resolve/{rev}/{urllib.parse.quote(name)}"
        try:
            download_file(url, path, size, (f.get("lfs") or {}).get("sha256"), headers,
                          lambda n, name=name: tick(n, name))
        except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError) as e:
            print("\n" + R(T("Download failed: {name}: {err}").format(name=name, err=e)))
            return False
    if progress:
        print(f"PROGRESS {total} {total} done", flush=True)
    print("\n" + G(T("Downloaded: {dest}").format(dest=dest)))
    m = inspect_model(dest)
    if m and not m["supported"]:
        print(Y(T("Note: this model is not supported by the engine:") + " " + ", ".join(m["problems"])))
    return True


def search_cmd(query, as_json=False):
    if not query:
        query = input(T("Search MLX models on Hugging Face") + " › ").strip()
    try:
        res = hf_search(query)
    except Exception as e:
        print(R(T("Search failed:") + f" {e}")); return False
    if as_json:
        print(json.dumps(res, ensure_ascii=False)); return True
    for i, m in enumerate(res, 1):
        size = f"{m['size']:.1f} GB" if m["size"] is not None else "?"
        ok = {"native": G(T("✓ engine")), "ptq1": Y(T("✓ engine (1.75-bit, experimental)")),
              "mlxlm": CY(T("via mlx-lm"))}.get(m["route"], DIM(T("not supported")))
        print(f" {i:>2}) {B(m['id'])}  {size}  ⬇{m['downloads']}  {ok}" + (G("  " + T("downloaded")) if m["local"] else ""))
    if sys.stdin.isatty() and res:
        ch = input(T("Number to download (Enter to cancel)") + " › ").strip()
        if ch.isdigit() and 1 <= int(ch) <= len(res):
            return hf_pull(res[int(ch) - 1]["id"])
    return True


# ─────────────────────────── status / settings ───────────────────────────
def status(cfg):
    avail, swap = mem_info()
    m = current_model(cfg)
    pid, h = read_pid(ENGINE_PID), health(cfg)
    print(B("\n── " + T("Status") + " ──"))
    print(f"{T('Model'):8s}: {B(m['name']) if m else R(T('not selected'))}")
    print(f"{T('Engine'):8s}: {G(T('running')) if h else (Y(T('loading')) if pid else R(T('not running')))}"
          + (DIM(f"  PID {pid}") if pid else ""))
    if h:
        mem, last = h.get("memory_gb", {}), h.get("last", {})
        print(f"{T('Memory'):8s}: " + T("model {m} GB · active {a} GB · peak {p} GB").format(
            m=mem.get("model"), a=mem.get("active"), p=mem.get("peak")))
        print(f"{T('Cache'):8s}: " + T("{n} prefix tokens · {r} requests").format(n=h.get("cached_tokens"), r=h.get("requests")))
        br = h.get("memory_breakdown") or {}
        if br:
            mb = lambda *ks: sum(br.get(k) or 0 for k in ks) / 2**20
            print(f"{'':8s}  " + T("weights {w:.0f} MB · context {kv:.0f} MB · SSM+snapshots {ssm:.0f} MB · buffers {buf:.0f} MB · process {p:.0f} MB").format(
                w=mb("weights", "vision"), kv=mb("kv"), ssm=mb("ssm", "snapshots"), buf=mb("activations", "mlx_cache"), p=mb("process")))
        if h.get("system_ram_used") is not None:
            print(f"{T('Guard'):8s}: " + T("system RAM {u:.0%} of limit {l:.0%}").format(u=h["system_ram_used"], l=h["mem_limit"]))
        if h.get("wired_limit_advice_mb"):
            print(Y(T("⚠ Metal's GPU memory cap is below what this model needs — the Mac may swap/throttle.")))
            print(DIM(f"  sudo sysctl iogpu.wired_limit_mb={h['wired_limit_advice_mb']}") +
                  DIM("  (" + T("resets on reboot") + ")"))
        if last:
            print(f"{T('Last'):8s}: " + T("{pt} in ({ct} cached, {pp} tok/s) → {out} out @ {dt} tok/s").format(
                pt=last.get("prompt_tokens"), ct=last.get("cached_tokens"), pp=last.get("prefill_tps"),
                out=last.get("completion_tokens"), dt=last.get("decode_tps")))
    col = R if avail < 2 else (Y if avail < 4 else G)
    print(f"{'RAM':8s}: " + T("{avail} of {total:.0f} GB available, swap {swap:.1f} GB").format(
        avail=col(f"{avail:.1f}"), total=RAM_GB, swap=swap))
    hogs = [n for n, pat in HOGS if pgrep(pat)]
    if hogs: print(Y(T("⚠ Using memory:") + " " + ", ".join(hogs)))


def settings(cfg):
    on, off = T("on"), T("off")
    while True:
        m = current_model(cfg)
        mc = mcfg(cfg)
        print(B("\n── " + T("Settings") + " ──") + DIM("  (" + T("applied on restart") + ")"))
        items = []
        if m:
            items += [("m:context", f"{T('Context'):26s}: {mc['context'] or 'auto'}", DIM(T("0 = auto, ~64 KB KV per token"))),
                      ("m:max_output", f"{T('Max output tokens'):26s}: {mc['max_output']}", "")]
            if m["thinking"]:
                eff = think_enabled(mc, m)
                note = DIM(" (" + T("model default") + ")") if mc["thinking"] is None else ""
                items.append(("m:thinking", f"{T('Reasoning by default'):26s}: {on if eff else off}", note))
            if m["efforts"]:
                items.append(("m:effort", f"{T('Reasoning effort'):26s}: {mc['effort'] or '—'}", DIM(" / ".join(m["efforts"]))))
            if m["vision"]:
                items.append(("m:vision", f"{T('Images'):26s}: {on if mc['vision'] else off + ' (−0.5 GB)'}", ""))
        items += [("port", f"{T('API port'):26s}: {cfg['port']}", ""),
                  ("lan", f"{T('LAN access'):26s}: {on if cfg['lan'] else off}", ""),
                  ("api_key", f"{T('API key'):26s}: {T('set') if cfg['api_key'] else T('none')}", ""),
                  ("auto_free", f"{T('Quit vMLX/oMLX/Ollama'):26s}: {on if cfg['auto_free'] else off}", ""),
                  ("prefill_step", f"{T('Prefill step'):26s}: {cfg['prefill_step']}", DIM(T("lower = lower peak memory"))),
                  ("mem_limit", f"{T('System RAM limit, %'):26s}: {cfg['mem_limit']}", DIM(T("above it requests are trimmed or stopped"))),
                  ("lang", f"{T('Language'):26s}: {cfg['lang']}", DIM("auto / en / ru")),
                  ("dirs", f"{T('Model folders'):26s}: " + ", ".join(cfg["dirs"]), "")]
        for i, (_, label, hint) in enumerate(items, 1):
            print(f" {i:>2}) {label}  {hint}")
        print("  0) " + T("back"))
        ch = input("› ").strip()
        if ch in ("0", ""): return
        try: key = items[int(ch) - 1][0]
        except (ValueError, IndexError): continue
        if key.startswith("m:"):
            k = key[2:]
            if k == "thinking": set_mcfg(cfg, k, not think_enabled(mc, m))
            elif k == "vision": set_mcfg(cfg, k, not mc[k])
            elif k == "effort":
                o = m["efforts"]; set_mcfg(cfg, k, o[(o.index(mc[k]) + 1) % len(o)] if mc[k] in o else o[0])
            else:
                v = input(T("New value") + f" [{mc[k]}]: ").strip()
                if v.isdigit(): set_mcfg(cfg, k, int(v))
        elif key in ("lan", "auto_free"):
            cfg[key] = not cfg[key]
        elif key == "lang":
            cfg["lang"] = {"auto": "en", "en": "ru", "ru": "auto"}[cfg["lang"]]
            set_lang(cfg["lang"])
            on, off = T("on"), T("off")
        elif key == "api_key":
            cfg[key] = input(T("Key (empty to remove)") + ": ").strip()
        elif key == "dirs":
            v = input(T("Folders, comma-separated") + ": ").strip()
            if v: cfg["dirs"] = [os.path.expanduser(x.strip()) for x in v.split(",") if x.strip()]
        else:
            v = input(T("New value") + f" [{cfg[key]}]: ").strip()
            if v.isdigit(): cfg[key] = int(v)
        save_cfg(cfg)


# ─────────────────────────── menu ───────────────────────────
def ask_yn(q, default=True):
    a = input(f"{q} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not a else a in ("y", "yes", "д", "да")


def logs():
    print(DIM(T("Ctrl+C to go back") + "\n"))
    try: subprocess.run(["tail", "-n", "40", "-f", ENGINE_LOG])
    except KeyboardInterrupt: pass


def menu():
    cfg = load_cfg()
    print_banner()
    actions = [
        ("Start", lambda: start(cfg)), ("Stop", lambda: stop(cfg)),
        ("Restart", lambda: restart(cfg)), ("Select model", lambda: choose_model(cfg)),
        ("Search & download models", lambda: search_cmd("")),
        ("Status", lambda: status(cfg)), ("Chat in terminal", lambda: chat(cfg)),
        ("Coding agent", lambda: __import__("agent").main([])),
        ("Speed test", lambda: bench(cfg)), ("Settings", lambda: settings(cfg)),
        ("Connect an editor (Continue / Kilo / opencode)", lambda: connect_client(cfg)), ("Logs", logs),
        ("Free memory (vMLX, oMLX, Ollama)", lambda: free_memory()),
        ("Engine tests", run_tests),
    ]
    while True:
        h, m = health(cfg, 1), current_model(cfg)
        st = G("● " + T("running")) if h else (Y("● " + T("loading")) if read_pid(ENGINE_PID) else R("● " + T("stopped")))
        avail, swap = mem_info()
        print(B("\n╭─ " + T("Status") + " ─╮"))
        print(f"  {B(m['name']) if m else Y(T('no model selected'))}  {st}")
        print(DIM("  " + T("RAM free {avail:.1f} GB · swap {swap:.1f} GB").format(avail=avail, swap=swap)
                  + (f" · http://localhost:{cfg['port']}/v1" if h else "")))
        for i, (name, _) in enumerate(actions, 1):
            print(f"  {i:>2}) {T(name)}")
        print("   0) " + T("Exit") + DIM(" (" + T("the model keeps running") + ")"))
        try:
            ch = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if ch == "0": return
        if ch.isdigit() and 1 <= int(ch) <= len(actions):
            try: actions[int(ch) - 1][1]()
            except KeyboardInterrupt: print()


def up(cfg):
    """One-step launch: model (if not running) → chat → menu."""
    if not health(cfg) and not start(cfg, sys.stdin.isatty()):
        input(T("Enter to open the menu…"))
    else:
        chat(cfg)
    menu()


def open_window():
    """Open a new Terminal window running `llm up`."""
    cmd = f"cd {json.dumps(BASE)} && LLM_HOME={json.dumps(DATA)} ./llm up"
    script = f'tell application "Terminal" to do script {json.dumps(cmd)}\ntell application "Terminal" to activate'
    return subprocess.run(["osascript", "-e", script], capture_output=True).returncode == 0


def main():
    cmd, arg = (sys.argv[1:] + ["", ""])[:2]
    cfg, tty = load_cfg(), sys.stdin.isatty()
    set_lang(cfg["lang"])
    actions = {
        "setup": lambda: setup(compat="--compat" in sys.argv), "start": lambda: start(cfg, tty), "stop": lambda: stop(cfg),
        "restart": lambda: restart(cfg, tty), "status": lambda: status(cfg),
        "models": lambda: print(json.dumps(scan_models(cfg), ensure_ascii=False)) if arg == "--json"
        else list_models(cfg, scan_models(cfg)), "use": lambda: choose_model(cfg, arg or None),
        "chat": lambda: chat(cfg), "bench": lambda: bench(cfg), "logs": logs,
        "free": lambda: free_memory(tty), "connect": lambda: connect_client(cfg, arg or None), "kilo": lambda: connect_client(cfg, "kilo"), "test": run_tests,
        "rm": lambda: remove_cmd(cfg, sys.argv[2:]),
        "perf": lambda: run_perf(sys.argv[2:]),
        "eval": lambda: run_eval(sys.argv[2:]),
        "calibrate": lambda: run_calibrate(sys.argv[2:]),
        "agent": lambda: __import__("agent").main(sys.argv[2:]),
        "ui": lambda: run_ui(sys.argv[2:]),
        "up": lambda: up(cfg), "open": open_window,
        "search": lambda: search_cmd(" ".join(a for a in sys.argv[2:] if a != "--json"), "--json" in sys.argv),
        "pull": lambda: hf_pull(arg, "--progress" in sys.argv) if arg else print(__doc__), "": menu, "menu": menu,
    }
    if cmd not in actions:
        print(__doc__); sys.exit(2)
    ok = actions[cmd]()
    sys.exit(1 if ok is False else 0)


if __name__ == "__main__":
    main()

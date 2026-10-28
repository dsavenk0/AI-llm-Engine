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
  ./llm fetch <url|magnet> [name]   add a model from a source you provide (.gguf/archive URL, or magnet via aria2c)
  ./llm run <model> [prompt] select (download if only an org/name), start, then chat — or answer one prompt
  ./llm show [<#|name>]      a quick card: format, route, size, context, capabilities
  ./llm ps                   the model loaded right now
  ./llm create <name> --from <model> [--system "…"] [--param k=v]   a named profile (run it with ./llm run <name>)
  ./llm use <#|name part>    select a model (restarts it if running)
  ./llm rm <#|name part> [--yes]   delete a downloaded model from disk
  ./llm start|stop|restart|status|chat|bench|logs|free
  ./llm service install|uninstall|status   optional background service (launchd): up after login, restarts on crash
  ./llm agent [dir]          terminal coding agent (files, search, shell) in dir or the current folder
  ./llm conduct ["<task>"]   optional orchestrator: split a task across models by role (set: conduct set …)
  ./llm convert <m.gguf> [out]   export a GGUF as an MLX folder (config + f16 safetensors + tokenizer)
  ./llm ui [dir]             native code editor with the assistant (a module of LLMBar.app: ./build-app.sh)
  ./llm connect [client]     register the model in Continue, Kilo Code or opencode (your choice)
  ./llm inspect [<#|name|path>] what is inside a model: layers, storage types, bits per weight, biggest tensors
  ./llm web search "<query>" web search (+ the top pages as text) as one JSON line, for the voice chat
  ./llm check [--json]       is everything there to run the engine? (Python, libraries, network, disk, models)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import family, gguf  # noqa: E402  (stdlib only: no MLX needed to route a model)

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
    "gguf_native": True,         # run GGUF files the engine's own MLX decoder can (llama.cpp only for the rest)
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
    "log_level": "info",         # engine log: error | warning | info | debug | trace
    "fallback_model": True,      # a request naming another model is answered by the loaded one (off: 404)
    "prefill_guard": True,       # refuse a prompt that would not fit in free memory before prefilling it
    "images_api": False,         # serve /v1/images/generations through mflux (./llm setup --images)
    "proxy_http": "",            # outgoing HTTP(S) (downloads, hub search, web tools); empty = none
    "proxy_https": "",
    "no_proxy": "",              # comma-separated hosts/CIDRs that bypass the proxy
    "ca_bundle": "",             # PEM file with extra root certificates (TLS-inspecting proxies)
    "favorites": [],             # model paths marked with a star: listed first, and the ones the apps offer to switch to
    "models": {},
    "profiles": {},              # Ollama-style named presets: {name: {from, system, params}} — see ./llm create
}
LOG_LEVELS = ("error", "warning", "info", "debug", "trace")


def network_env(cfg, env=None):
    """`env` (a copy of os.environ by default) with the proxy and CA settings applied. urllib and ssl
    read exactly these variables, so every download and web request in this project follows them."""
    env = dict(os.environ if env is None else env)
    for key, names in (("proxy_http", ("HTTP_PROXY", "http_proxy")), ("proxy_https", ("HTTPS_PROXY", "https_proxy")),
                       ("no_proxy", ("NO_PROXY", "no_proxy")), ("ca_bundle", ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"))):
        v = str(cfg.get(key) or "").strip()
        if v:
            for n in names:
                env[n] = v
    if cfg.get("proxy_http") or cfg.get("proxy_https"):          # the engine itself is always reached directly
        hosts = {h for h in (env.get("NO_PROXY", "") + ",localhost,127.0.0.1,::1").split(",") if h}
        env["NO_PROXY"] = env["no_proxy"] = ",".join(sorted(hosts))
    return env
MODEL_DEFAULTS = {
    "context": 0,                # prompt + output; 0 = auto (from RAM, model size and KV cost)
    "max_output": 8192,          # output cap (larger requests are clamped by the server)
    "thinking": None,            # None = whatever the model's chat template does; True/False = your choice
    "effort": "medium",
    "vision": True,
    "max_pixels": 1024 * 1024,
    "steer": "",                 # an activation edit for this model: a direction file or a folder (./llm steer)
    "steer_alpha": None,         # its strength; None = what the edit's own adapter.json recommends, else 1.0
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
    if family.native_capable(cfgj, tags):   # by the architecture's fields; names only for an undownloaded hub repo
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
            "tool_calls": "<tool_call>" in tpl or "tools" in tpl,
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
    found += scan_gguf(cfg)
    favs = {str(f) for f in (cfg.get("favorites") or [])}
    for m in found:
        m["favorite"] = m["path"] in favs
        m["engine"], m["features"] = route_features(cfg, m)
    return sorted(found, key=lambda m: (not m["favorite"], not m["supported"], m["name"].lower()))


def route_features(cfg, m):
    """(which engine runs the model, what that route can do with it) — shown beside every model so a missing
    feature is explained by the route, not discovered later. Engines: "engine" (native MLX), "engine-ptq1",
    "engine-gguf" (the own GGUF decoder), "mlx-lm", "llama.cpp"."""
    route = m.get("route", "native")
    if route == "gguf":
        which = "engine-gguf" if gguf_native(cfg, m) else "llama.cpp"
    else:
        which = {"native": "engine", "ptq1": "engine-ptq1", "mlxlm": "mlx-lm"}.get(route, route)
    f = []
    if m.get("vision") and which in ("engine", "engine-ptq1", "llama.cpp"):
        f.append("vision")
    if m.get("thinking"):
        f.append("thinking")
    # tool calls: the native engine and llama.cpp parse every convention they know; mlx-lm and the own GGUF
    # decoder follow what the chat template speaks
    if which in ("engine", "engine-ptq1", "llama.cpp") or m.get("tool_calls"):
        f.append("tools")
    if which in ("engine", "engine-ptq1"):
        f.append("ssd-cache")
    if which in ("engine", "engine-gguf"):
        f += ["speculation", "steering"]
    return which, f


_GGUF_CACHE = {}
LMSTUDIO_DIRS = ("~/.lmstudio/models", "~/.cache/lm-studio/models")
_SPLIT_RE = re.compile(r"-0*(\d+)-of-0*(\d+)\.gguf$", re.I)


def inspect_gguf(path, name=None):
    """A model entry (same shape as inspect_model's) for one GGUF file; None when it is not a readable GGUF."""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        if key not in _GGUF_CACHE:
            _GGUF_CACHE[key] = gguf.read_header(path)
        h = _GGUF_CACHE[key]
    except (OSError, gguf.GGUFError):
        return None
    size = st.st_size
    m = _SPLIT_RE.search(path)
    if m:                                     # a split model: llama.cpp opens the other parts itself
        pre, total = path[:m.start()], int(m.group(2))
        parts = [f"{pre}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
        size = sum(os.path.getsize(p) for p in parts if os.path.exists(p))
    tpl = h.get("chat_template") or ""
    base = name or h["name"] or os.path.splitext(os.path.basename(path))[0]
    if h["quant"] and h["quant"].lower() not in base.lower():
        base = f"{base} {h['quant']}"
    return {"path": path, "name": base, "size": size / 2**30, "format": f"GGUF {h['quant']}", "type": h["architecture"],
            "route": "gguf", "vision": bool(gguf.projector_for(path)),
            "thinking": "<think>" in tpl or "enable_thinking" in tpl,
            "thinking_default": "enable_thinking" in tpl and template_thinks_by_default(tpl),
            "efforts": [], "hadamard": False, "supported": True, "problems": [],
            "tool_calls": "<tool_call>" in tpl,
            "context_length": h.get("context_length")}


_NATIVE_CACHE = {}


def gguf_native(cfg, m):
    """True when this GGUF runs on the engine's own MLX decoder: allowed by the setting, a single file without
    an image projector, and an architecture / quantization / tokenizer that decoder implements."""
    if m.get("route") != "gguf" or not cfg.get("gguf_native", True) or m.get("vision"):
        return False
    path = m["path"]
    if _SPLIT_RE.search(path) or not os.path.isfile(path):
        return False
    try:
        st = os.stat(path)
    except OSError:
        return False
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _NATIVE_CACHE:
        _NATIVE_CACHE[key] = gguf.native_support(path)[0]
    return _NATIVE_CACHE[key]


def scan_gguf(cfg):
    """GGUF files in the model folders, LM Studio's folder and Ollama's store (read in place, never copied)."""
    roots = [os.path.expanduser(d) for d in cfg["dirs"]] + [os.path.expanduser(d) for d in LMSTUDIO_DIRS]
    found, seen = [], set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("blobs", "refs")]
            if dirpath[len(root):].count(os.sep) > 6:
                dirnames[:] = []
                continue
            for fn in sorted(filenames):
                if not fn.lower().endswith(".gguf") or gguf.is_projector(fn):
                    continue
                sp = _SPLIT_RE.search(fn)
                if sp and int(sp.group(1)) != 1:
                    continue                      # only the first part names a split model
                p = os.path.join(dirpath, fn)
                real = os.path.realpath(p)
                if real in seen:
                    continue
                seen.add(real)
                info = inspect_gguf(p)
                if info:
                    found.append(info)
    try:
        for label, blob in gguf.ollama_models().items():
            real = os.path.realpath(blob)
            if real not in seen:
                seen.add(real)
                info = inspect_gguf(blob, name=f"{label} (Ollama)")
                if info:
                    found.append(info)
    except OSError:
        pass
    return found


def current_model(cfg):
    p = cfg["model"]
    if p and os.path.isfile(p):
        return inspect_gguf(p)
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
        star = Y("★ ") if m.get("favorite") else "  "
        print(f" {i:>2}) {star}{B(m['name'])}{cur}")
        print(f"     {m['size']:.1f} GB · {m['format']} · {m['type']}{' · ' + ', '.join(feats) if feats else ''} · {ok}")
        if m.get("supported") and m.get("engine"):
            print(DIM(f"     {T('runs on')} {m['engine']}" + (": " + ", ".join(m["features"]) if m.get("features") else "")))


def next_favorite(favs, current):
    """The favourite after `current` in the list (wrapping); the first when the current one is not a favourite;
    None when there is nothing to switch to."""
    if not favs or (len(favs) == 1 and favs[0] == current):
        return None
    if current in favs:
        return favs[(favs.index(current) + 1) % len(favs)]
    return favs[0]


def fav_cmd(cfg, args):
    """`./llm fav` lists the favourites; `./llm fav <number|name|path>` toggles one; --on / --off set it explicitly
    (what the apps send, so two quick clicks never undo each other); `--next` makes the next favourite the active
    model (what the widget's model name does)."""
    flags = [a for a in args if a in ("--on", "--off", "--json", "--next")]
    rest = [a for a in args if a not in flags]
    models = scan_models(cfg)
    favs = [f for f in cfg.get("favorites", []) if isinstance(f, str)]
    if "--next" in flags:
        usable = {m["path"] for m in models if m.get("supported", True)}
        nxt = next_favorite([f for f in favs if f in usable], cfg.get("model", ""))
        if nxt is None:
            print(Y(T("No other favourite model to switch to."))); return False
        return choose_model(cfg, nxt)
    if not rest:
        names = [m for m in models if m["favorite"]]
        if "--json" in flags:
            print(json.dumps([m["path"] for m in names], ensure_ascii=False))
        elif not names:
            print(Y(T("No favourite models yet — ./llm fav <number|name> adds one.")))
        else:
            print(B("\n── " + T("Favourite models") + " ──"))
            for m in names:
                print(f"  {Y('★')} {B(m['name'])}  {m['size']:.1f} GB")
        return True
    key = rest[0]
    pick = next((m for m in models if m["path"] == key), None)
    if pick is None and key.isdigit() and 1 <= int(key) <= len(models):
        pick = models[int(key) - 1]
    if pick is None:
        hits = [m for m in models if key.lower() in m["name"].lower()]
        if len(hits) > 1:
            print(Y(T("Several matches:") + " " + ", ".join(h["name"] for h in hits)))
        pick = hits[0] if len(hits) == 1 else None
    if pick is None:
        print(R(T("Model not found."))); return False
    want = ("--on" in flags) or ("--off" not in flags and pick["path"] not in favs)
    favs = [f for f in favs if f != pick["path"]]
    if want:
        favs.append(pick["path"])
    cfg["favorites"] = favs
    save_cfg(cfg)
    print(G((T("★ Added to favourites:") if want else T("Removed from favourites:")) + f" {pick['name']}"))
    return True


def resolve_model(models, arg):
    """Find the one model `arg` names — by exact path, 1-based list number, or a unique name substring.
    Returns (pick, matches): pick is None when nothing or several things match (matches lists them, for a
    "be more specific" message)."""
    arg = str(arg)
    exact = next((m for m in models if m["path"] == arg), None)
    if exact:
        return exact, [exact]
    if arg.isdigit() and 1 <= int(arg) <= len(models):
        return models[int(arg) - 1], [models[int(arg) - 1]]
    hits = [m for m in models if arg.lower() in m["name"].lower()]
    return (hits[0] if len(hits) == 1 else None), hits


def profile_params(params):
    """Keep only the sampling keys the server honours from a profile's params (so a stray key can't break
    a request). `max_tokens` is allowed through as well."""
    allow = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "max_tokens", "seed")
    return {k: v for k, v in (params or {}).items() if k in allow}


def run_cmd(cfg, args):
    """`./llm run <model|org/name|profile> [prompt]` — Ollama-style: select the model (download it first if
    only an org/name is given and it isn't installed), start the engine, then chat. A named profile (see
    `./llm create`) brings its base model, system prompt and sampling. A trailing prompt prints one answer."""
    args = [a for a in args if a and not a.startswith("--")]
    if not args:
        print(T("Usage: ./llm run <model|org/name|profile> [prompt]")); return False
    name, prompt = args[0], " ".join(args[1:]).strip()
    prof = cfg.get("profiles", {}).get(name)
    target = prof["from"] if prof else name           # a profile runs on top of its base model
    system = prof.get("system", "") if prof else ""
    params = prof.get("params", {}) if prof else {}
    pick, hits = resolve_model(scan_models(cfg), target)
    if not pick and len(hits) > 1:
        print(Y(T("Several matches:") + " " + ", ".join(h["name"] for h in hits))); return False
    if not pick and "/" in target:                    # not installed, but looks like a hub id → download it
        if not hf_pull(target, progress=True):
            return False
        pick, _ = resolve_model(scan_models(cfg), target)
    if not pick:
        print(R(T("Model not found. Give an installed name/number, or an org/name to download."))); return False
    if pick["path"] != cfg.get("model") and not choose_model(cfg, pick["path"]):
        return False
    if not (read_pid(ENGINE_PID) and health(cfg)) and not start(cfg, sys.stdin.isatty()):
        print(R(T("The engine did not start."))); return False
    if prof:
        print(DIM(T("Profile {name}: {base}").format(name=name, base=pick["name"])
                  + (" · system" if system else "") + (f" · {len(params)} params" if params else "")))
    if prompt:
        return run_once(cfg, prompt, system=system, params=params)
    chat(cfg, system=system, params=params)
    return True


def run_once(cfg, prompt, system="", params=None):
    """Send one prompt and stream the answer to stdout (for `llm run <model> <prompt>`)."""
    m = current_model(cfg)
    if not m or not health(cfg):
        print(R(T("The model is not running."))); return False
    payload = {"model": served_name(m), "stream": True,
               "messages": ([{"role": "system", "content": system}] if system else [])
               + [{"role": "user", "content": prompt}],
               "chat_template_kwargs": {"enable_thinking": think_enabled(mcfg(cfg), m)},
               **profile_params(params)}
    req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                raw = raw.strip()
                if not raw.startswith(b"data:") or raw == b"data: [DONE]":
                    continue
                ev = json.loads(raw[5:])
                if ev.get("error"):
                    print(R(ev["error"].get("message"))); return False
                for ch in ev.get("choices", []):
                    piece = ch.get("delta", {}).get("content")
                    if piece:
                        print(piece, end="", flush=True)
        print()
        return True
    except KeyboardInterrupt:
        print(Y("\n[" + T("interrupted") + "]")); return False
    except Exception as e:
        print(R("\n" + T("Error:") + f" {e}")); return False


def _coerce(v):
    """A `--param` value as its natural JSON type: bool, int, float, else the string."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def create_cmd(cfg, args):
    """`./llm create <name> --from <model> [--system "..."] [--param key=value ...]` — a named profile
    (Ollama-style): a base model plus a system prompt and sampling, run later with `./llm run <name>`.
    `./llm create` lists profiles; `./llm create <name> --rm` deletes one."""
    profiles = cfg.setdefault("profiles", {})
    bare = [a for a in args if not a.startswith("--")]
    if not args:
        if not profiles:
            print(DIM(T("No profiles yet. Make one: ./llm create <name> --from <model> --system \"...\"")))
            return True
        for n, p in profiles.items():
            print(f"{B(n)}  ← {p['from']}" + (f"  · {len(p.get('params', {}))} params" if p.get("params") else ""))
        return True
    name = bare[0] if bare else ""
    if not name:
        print(T("Usage: ./llm create <name> --from <model> [--system \"…\"] [--param k=v]")); return False
    if "--rm" in args:
        if profiles.pop(name, None) is None:
            print(R(T("Profile not found."))); return False
        save_cfg(cfg); print(G(T("Deleted profile: {name}").format(name=name))); return True
    base, system, params, i = "", "", {}, 1
    while i < len(args):
        a = args[i]
        if a == "--from" and i + 1 < len(args):
            base = args[i + 1]; i += 2
        elif a == "--system" and i + 1 < len(args):
            system = args[i + 1]; i += 2
        elif a == "--param" and i + 1 < len(args):
            k, _, v = args[i + 1].partition("=")
            if k.strip():
                params[k.strip()] = _coerce(v.strip())
            i += 2
        else:
            i += 1
    if not base:
        print(R(T("Give a base model: ./llm create {name} --from <model>").format(name=name))); return False
    profiles[name] = {"from": base, "system": system, "params": params}
    save_cfg(cfg)
    print(G(T("Created profile {name} ← {base}").format(name=name, base=base))
          + (f"  · {len(params)} params" if params else ""))
    return True


def show_cmd(cfg, args):
    """`./llm show [model|profile] [--json]` — a quick card for a model (format, route, size, context when
    loaded, thinking/vision/tools, and any reason the engine cannot run it) or a profile. Defaults to the
    selected model."""
    as_json = "--json" in args
    key = next((a for a in args if not a.startswith("--")), "")
    prof = cfg.get("profiles", {}).get(key)
    if prof:
        if as_json:
            print(json.dumps({"profile": key, **prof}, ensure_ascii=False)); return True
        print(B(key) + DIM(f"  ← {prof['from']}"))
        if prof.get("system"):
            print(DIM("  system: " + prof["system"][:200]))
        if prof.get("params"):
            print(DIM("  params: " + ", ".join(f"{k}={v}" for k, v in prof["params"].items())))
        return True
    models = scan_models(cfg)
    if key:
        pick, hits = resolve_model(models, key)
        if not pick:
            print(Y(T("Several matches:") + " " + ", ".join(h["name"] for h in hits)) if len(hits) > 1
                  else R(T("Model not found.")))
            return False
    else:
        pick = next((m for m in models if m["path"] == cfg.get("model")), None)
        if not pick:
            print(R(T("No model selected."))); return False
    ctx = None
    if pick["path"] == cfg.get("model") and (h := health(cfg)):
        ctx = h.get("context")
    caps = [c for c, on in (("thinking", pick["thinking"]), ("vision", pick["vision"]),
                            ("tools", pick["tool_calls"])) if on]
    if as_json:
        print(json.dumps({"name": pick["name"], "path": pick["path"], "format": pick["format"],
                          "route": pick["route"], "size_gb": round(pick["size"], 2), "context": ctx,
                          "capabilities": caps, "efforts": pick["efforts"], "supported": pick["supported"],
                          "problems": pick["problems"]}, ensure_ascii=False))
        return True
    print(B(pick["name"]))
    print(DIM(f"  {pick['format']} · {pick['route']} · {pick['size']:.1f} GB" + (f" · ctx {ctx}" if ctx else "")))
    print(DIM("  " + (", ".join(caps) or "—")))
    if not pick["supported"]:
        print(R(T("The engine does not support this model:") + " " + ", ".join(pick["problems"])))
    return True


def ps_cmd(cfg, args):
    """`./llm ps [--json]` — the model loaded right now (Ollama-style)."""
    as_json = "--json" in args
    h = health(cfg) if read_pid(ENGINE_PID) else None
    m = current_model(cfg) if h and h.get("status") == "ok" else None
    rows = []
    if m:
        rows = [{"name": m["name"], "state": h.get("state"), "size_gb": (h.get("memory_gb") or {}).get("model"),
                 "context": h.get("context"), "port": cfg["port"]}]
    if as_json:
        print(json.dumps({"models": rows}, ensure_ascii=False)); return True
    if not rows:
        print(DIM(T("No model is loaded."))); return True
    r = rows[0]
    print(f"{r['name']}  ·  {r['state']}  ·  {r['size_gb']} GB  ·  ctx {r['context']}  ·  :{r['port']}")
    return True


def choose_model(cfg, arg=None):
    models = scan_models(cfg)
    if arg is None:
        list_models(cfg, models)
        if not models: return False
        arg = input(T("Model number (Enter to cancel)") + " › ").strip()
        if not arg: return False
    pick, hits = resolve_model(models, arg)
    if not pick and len(hits) > 1:
        print(Y(T("Several matches:") + " " + ", ".join(h["name"] for h in hits)))
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
    try:
        sync_clients(cfg)
    except Exception:
        pass
    if was_running:
        restart(cfg, sys.stdin.isatty())
    return True


def model_root(path):
    """The folder that holds a model on disk: models--org--name for the HF cache layout (path is a
    snapshot inside it), else the model folder itself. A GGUF is a file: its own folder is not ours to remove."""
    if os.path.isfile(path):
        return path
    parts = path.rstrip(os.sep).split(os.sep)
    if "snapshots" in parts:
        return os.sep.join(parts[:parts.index("snapshots")])
    return path


def delete_model(cfg, m, assume_yes=False):
    """Remove a downloaded model from disk. Refuses anything outside the configured model folders,
    and the model the running engine has loaded."""
    if m.get("route") == "gguf" or os.path.isfile(m["path"]):
        return delete_gguf(cfg, m, assume_yes)
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


def set_vision(cfg, arg="on"):
    """`./llm vision on|off`: image input for the selected model (a setting that needs an engine restart,
    which this does when the engine is running). The editor's "Images" switch calls it."""
    m = current_model(cfg)
    if not m:
        print(R(T("Select a model first."))); return False
    if not m["vision"]:
        print(R(T("This model has no vision tower."))); return False
    on = str(arg or "on").lower() not in ("off", "0", "no", "false")
    set_mcfg(cfg, "vision", on)
    save_cfg(cfg)
    print(G(T("Images: {state} for {name}").format(state=T("on") if on else T("off"), name=m["name"])))
    if read_pid(ENGINE_PID):
        return restart(cfg, False)
    return True


REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def steer_base(path):
    """(base repo id, pinned revision) an edit folder names — direction.json's "model" and the adapter's
    "base_revision"; (None, None) when it names none."""
    d = _json(os.path.join(path, "direction.json"))
    a = _json(os.path.join(path, "runtime-adapter", "adapter.json")) or _json(os.path.join(path, "adapter.json"))
    repo = d.get("model") or a.get("base_model")
    if not isinstance(repo, str) or not REPO_ID_RE.match(repo):
        repo = None
    rev = a.get("base_revision")
    return repo, (rev if isinstance(rev, str) and re.match(r"^[0-9a-f]{7,40}$", rev) else None)


def find_installed(cfg, repo):
    """The scanned model that is hub repo `org/name`, wherever it was downloaded: our own flat folder
    (…/hub/org/name) or the Hugging Face cache layout (…/hub/models--org--name/snapshots/<rev>). None if absent."""
    org, name = repo.split("/", 1)
    flat = os.sep + os.path.join(org, name)
    layered = f"models--{org}--{name}"
    for m in scan_models(cfg):
        path = os.path.realpath(m["path"])
        if path.endswith(flat) or (layered in path.split(os.sep)):
            return m
    return None


def steer_cmd(cfg, args):
    """`./llm steer`                       what is set for the selected model
       `./llm steer <folder|file|org/name> [--alpha A]`   use an activation edit (a direction shipped next to a
                                           base model, see engine/steering.py). A hub repo is downloaded first,
                                           and so is the base model it names (the exact revision it was made
                                           for) when that is not installed; that model is then selected.
       `./llm steer off`                   back to the unmodified model
    Per model, native route only; the engine restarts when it is running."""
    from engine import steering
    m = current_model(cfg)
    rest = [a for a in args if not a.startswith("--alpha")]
    alpha = None
    for i, a in enumerate(args):
        if a.startswith("--alpha"):
            raw = a.split("=", 1)[1] if "=" in a else (args[i + 1] if i + 1 < len(args) else "")
            try:
                alpha = float(raw)
            except ValueError:
                print(R(T("--alpha needs a number"))); return False
            rest = [x for x in rest if x != raw]
    if not rest or rest[0].lower() in ("off", "none", "-"):
        if not m:
            print(R(T("Select a model first."))); return False
        mc = mcfg(cfg, m["path"])
        if not rest:
            if mc.get("steer"):
                print(T("Activation edit for {name}: {path}  (alpha {a})").format(
                    name=m["name"], path=mc["steer"], a=mc.get("steer_alpha") if mc.get("steer_alpha") is not None else "auto"))
            else:
                print(DIM(T("No activation edit for {name}.").format(name=m["name"])))
            return True
        set_mcfg(cfg, "steer", ""); set_mcfg(cfg, "steer_alpha", None)
        save_cfg(cfg)
        print(G(T("Activation edit removed for {name}").format(name=m["name"])))
        return restart(cfg, False) if read_pid(ENGINE_PID) else True

    target = rest[0]
    path = os.path.expanduser(target)
    if not os.path.exists(path):
        if not REPO_ID_RE.match(target):
            print(R(T("{p}: no such file or folder").format(p=target))); return False
        if not hf_pull(target):
            return False
        path = _pull_dest(target)
    try:
        file, rec = steering.resolve(path)
    except steering.SteerError as e:
        print(R(str(e))); return False

    # the model the edit belongs to: its own base when it names one (fetched if missing), else the selected one
    base, rev = steer_base(path)
    chosen = m
    if base:
        chosen = find_installed(cfg, base)
        if chosen is None:
            print(T("This edit was made for {base}; it is not installed, downloading it first.").format(base=base))
            if not hf_pull(base, revision=rev):
                return False
            chosen = find_installed(cfg, base)
        if chosen is None:
            print(R(T("{base} is downloaded but the engine does not list it as a model.").format(base=base)))
            return False
    elif not m:
        print(R(T("Select a model first."))); return False
    if not chosen.get("supported", True):
        print(R(T("The engine does not support this model:") + " " + ", ".join(chosen.get("problems", [])))); return False
    if chosen.get("route", "native") != "native" and not gguf_native(cfg, chosen):
        print(R(T("Activation edits work on the native engine route only (this model runs on: {r}).")
                .format(r=chosen.get("route")))); return False

    cfg["models"].setdefault(chosen["path"], {})
    cfg["models"][chosen["path"]]["steer"] = os.path.realpath(path)
    cfg["models"][chosen["path"]]["steer_alpha"] = alpha
    save_cfg(cfg)
    used = alpha if alpha is not None else (rec if rec is not None else 1.0)
    print(G(T("Activation edit set for {name}: {file}, alpha {a}").format(
        name=chosen["name"], file=os.path.basename(file), a=used)))
    print(DIM(T("It must match the base model it was made for; a wrong size is refused at start.")))
    if not m or os.path.realpath(chosen["path"]) != os.path.realpath(m["path"]):
        models = scan_models(cfg)
        idx = next(i for i, x in enumerate(models, 1) if os.path.realpath(x["path"]) == os.path.realpath(chosen["path"]))
        return choose_model(cfg, str(idx))          # selects it (and restarts a running engine on it)
    return restart(cfg, False) if read_pid(ENGINE_PID) else True


def delete_gguf(cfg, m, assume_yes=False):
    """Remove a GGUF file (and its projector / split parts) from a model folder or LM Studio's folder.
    Ollama's store is not ours: it is shared by its own tool, so the person is pointed at `ollama rm`."""
    path = os.path.realpath(m["path"])
    ollama = os.path.realpath(os.path.expanduser("~/.ollama"))
    if path.startswith(ollama + os.sep):
        print(R(T("This model belongs to Ollama; remove it there:  ollama rm <name>")))
        return False
    homes = [os.path.realpath(os.path.expanduser(d)) for d in list(cfg["dirs"]) + list(LMSTUDIO_DIRS)]
    if not any(path.startswith(h + os.sep) for h in homes) or not os.path.isfile(path):
        print(R(T("Refusing to delete a folder outside the model directories:") + f" {path}"))
        return False
    if path == os.path.realpath(cfg["model"] or "") and read_pid(ENGINE_PID) and health(cfg):
        print(R(T("This model is loaded. Stop the engine first (./llm stop).")))
        return False
    victims = [path]
    sp = _SPLIT_RE.search(path)
    if sp:
        total, pre = int(sp.group(2)), path[:sp.start()]
        victims = [f"{pre}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
    proj = gguf.projector_for(path)
    d = os.path.dirname(path)
    others = [n for n in os.listdir(d) if n.lower().endswith(".gguf") and not gguf.is_projector(n)
              and os.path.join(d, n) not in victims]
    if proj and not others:                                  # the projector belongs to this model alone
        victims.append(proj)
    victims = [v for v in victims if os.path.isfile(v)]
    size = sum(os.path.getsize(v) for v in victims) / 2**30
    if not assume_yes and not ask_yn(R(T("Delete {name} ({size:.1f} GB) from disk?").format(name=m["name"], size=size)), False):
        return False
    for v in victims:
        os.remove(v)
    try:
        os.rmdir(d)                                          # a pulled repo's folder, once it is empty
    except OSError:
        pass
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
    if arg in cfg.get("profiles", {}):                # a named profile is removed here too, not deleted from disk
        cfg["profiles"].pop(arg); save_cfg(cfg)
        print(G(T("Deleted profile: {name}").format(name=arg))); return True
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
    """The PID in `path` when that process is alive AND is ours: a pid file outlives a crash or a reboot,
    and its number may since belong to any other program — which `./llm stop` must never signal.
    The file holds the pid and, on a second line, what was started (a fragment of its command line)."""
    try:
        with open(path) as f:
            parts = f.read().split()
        pid = int(parts[0])
        expect = parts[1] if len(parts) > 1 else "engine.server"
        os.kill(pid, 0)
    except (OSError, ValueError, IndexError):
        return None
    if expect:
        try:
            cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True,
                                 timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return pid                     # cannot tell: keep the old behaviour
        if expect not in cmd:
            try: os.remove(path)           # stale: somebody else's process now
            except OSError: pass
            return None
    return pid


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


def port_holder(port):
    """(pid, command name) of the process listening on the port, or None when it cannot be told."""
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True,
                             text=True, timeout=5).stdout.split()
        pid = int(out[0])
        name = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True,
                              timeout=5).stdout.strip()
        return pid, os.path.basename(name) or "?"
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


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
# An offline build (./build-app.sh --offline) carries a Python and the engine's wheels inside the app.
_RES = os.path.dirname(BASE) if os.path.basename(os.path.dirname(BASE)) == "Resources" else None
BUNDLED_PY = os.path.join(_RES, "python", "bin", "python3") if _RES else None
BUNDLED_WHEELS = os.path.join(_RES, "wheels") if _RES else None


def bundled_wheels():
    """The folder of wheels inside the app, or None (a normal build, or a checkout)."""
    if BUNDLED_WHEELS and os.path.isdir(BUNDLED_WHEELS) and any(n.endswith(".whl") for n in os.listdir(BUNDLED_WHEELS)):
        return BUNDLED_WHEELS
    return None


def _python_ok(p):
    """Is `p` a python >= 3.10? (/usr/bin/python3 is skipped without the Command Line Tools: running it
    would open the macOS install dialog.)"""
    if not p or not os.path.exists(p):
        return False
    if p == "/usr/bin/python3" and subprocess.run(["xcode-select", "-p"], capture_output=True).returncode != 0:
        return False
    try:
        v = subprocess.run([p, "-c", "import sys; print(sys.version_info >= (3, 10))"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return v.stdout.strip() == "True"


def find_python():
    """A python3 >= 3.10 for the engine venv (the manager itself runs on 3.9+). With none on the Mac, a
    private copy is downloaded once into the data folder — a clean laptop needs nothing installed first."""
    from engine import pyruntime
    found = _python_probe()
    if found:
        return found
    print(Y(T("Python 3.10+ was not found — downloading a private copy (about 35 MB, once).")))
    try:
        p = pyruntime.install(DATA, log=lambda s: print(DIM("  " + s)))
    except (pyruntime.PythonInstallError, OSError, urllib.error.URLError, ValueError) as e:
        offline = isinstance(e, (urllib.error.URLError, OSError)) and not isinstance(e, pyruntime.PythonInstallError)
        raise SystemExit((T("No internet connection — Python (about 35 MB) must be downloaded.") + f" {e}\n"
                          if offline else T("Could not install Python:") + f" {e}\n") +
                         T("Python 3.10+ is required for the engine (e.g. brew install python)."))
    if not _python_ok(p):
        raise SystemExit(T("Python 3.10+ is required for the engine (e.g. brew install python)."))
    print(G(T("Python installed:") + f" {p}"))
    return p


def _python_probe():
    """A usable python >= 3.10 without downloading anything (None when there is none)."""
    from engine import pyruntime
    for p in (BUNDLED_PY or "", "/opt/homebrew/bin/python3", "/usr/local/bin/python3", pyruntime.python_path(DATA),
              shutil.which("python3") or "", sys.executable):
        if _python_ok(p):
            return p
    return None


def inspect_cmd(cfg, args):
    """`./llm inspect <number|name|path> [--json] [--top N]`: what is inside a model, from its headers only
    (layers, storage types, bits per weight, the biggest tensor groups, and which engine route runs it)."""
    from engine import modelinfo
    flags = {"--json"}
    top, rest, i = 14, [], 0
    while i < len(args):
        if args[i] == "--top" and i + 1 < len(args):
            try:
                top = max(1, int(args[i + 1]))
            except ValueError:
                pass
            i += 2
        else:
            rest.append(args[i])
            i += 1
    key = next((a for a in rest if a not in flags), "")
    path = None
    if key:
        if os.path.exists(os.path.expanduser(key)):
            path = os.path.expanduser(key)
        else:
            models = scan_models(cfg)
            if key.isdigit() and 1 <= int(key) <= len(models):
                path = models[int(key) - 1]["path"]
            else:
                hits = [m for m in models if key.lower() in m["name"].lower()]
                if len(hits) == 1:
                    path = hits[0]["path"]
                elif hits:
                    print(Y(T("Several matches:") + " " + ", ".join(h["name"] for h in hits))); return False
    elif cfg.get("model"):
        path = cfg["model"]
    if not path:
        print(R(T("Model not found."))); return False
    try:
        info = modelinfo.inspect(path)
    except (modelinfo.InspectError, OSError, ValueError) as e:
        print(R(str(e))); return False
    if info["kind"] == "mlx":
        try:
            info["route"] = engine_route(_json(os.path.join(path, "config.json")), jang=_json(os.path.join(path, "jang_config.json")))
        except Exception:
            info["route"] = None
    if "--json" in rest:
        print(json.dumps(info, ensure_ascii=False))
    else:
        print(modelinfo.render(info, top=top))
        if info.get("route"):
            print(f"  engine route: {info['route']}")
    return True


def web_cmd(args):
    """`./llm web search "<query>" [--n 4] [--read 2]`: one JSON line {"query", "results": [{title, url, snippet}],
    "pages": [{url, text}]} — what the voice chat adds to a question that needs fresh facts. Failures come
    back as {"error": ...}, never as a traceback: the caller is an app."""
    from concurrent.futures import ThreadPoolExecutor
    from harness import web
    words, opts, i = [], {}, 0
    while i < len(args):
        if args[i] in ("--n", "--read") and i + 1 < len(args):
            opts[args[i][2:]] = args[i + 1]
            i += 2
        else:
            words.append(args[i])
            i += 1
    if not words or words[0] != "search" or len(words) < 2:
        print(json.dumps({"error": 'usage: llm web search "<query>" [--n 4] [--read 2]'}))
        return False
    query = " ".join(words[1:]).strip()[:300]
    try:
        n = max(1, min(8, int(opts.get("n", 4))))
        read = max(0, min(3, int(opts.get("read", 2))))
    except ValueError:
        n, read = 4, 2
    out = {"query": query, "results": [], "pages": []}
    try:
        out["results"] = web.search(query, n=n, timeout=12)
    except Exception as e:                                  # offline, blocked, parser change…
        out["error"] = str(e)[:200]
        print(json.dumps(out, ensure_ascii=False))
        return False

    def page(hit):
        try:
            return {"url": hit["url"], "text": web.fetch(hit["url"], max_chars=2500, timeout=8)}
        except Exception:
            return None
    if read and out["results"]:
        with ThreadPoolExecutor(max_workers=read) as pool:
            out["pages"] = [p for p in pool.map(page, out["results"][:read]) if p]
    print(json.dumps(out, ensure_ascii=False))
    return True


def check_cmd(cfg, as_json=False):
    """`./llm check [--json]`: is everything there to run the engine? Exit status 1 when it cannot start."""
    from engine import preflight
    try:
        models = len(scan_models(cfg))
    except Exception:
        models = None
    res = preflight.summary(preflight.run(DATA, VENV_PY, _python_probe, models, env_ok=lambda _p: env_ready(),
                                              wheels_bundled=bool(bundled_wheels())))
    if as_json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        for i in res["items"]:
            mark = {"ok": G("✓"), "warn": Y("!"), "error": R("✗")}[i["level"]]
            print(f"{mark} {i['title']}" + (DIM(f"  {i['detail']}") if i["detail"] else ""))
    return res["ok"]


ENV_READY = os.path.join(DATA, ".venv", ".ready")      # written once the libraries are in


def _ready_stamp():
    """What the "ready" mark vouches for: these requirements, this interpreter. A new app version with other
    requirements (or a venv rebuilt on another Python) is checked again, once."""
    import hashlib
    try:
        with open(os.path.join(BASE, "requirements.txt"), "rb") as f:
            reqs = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        reqs = ""
    return f"{reqs}\n{os.path.realpath(VENV_PY)}\n"


def env_ready():
    """Is the engine environment usable? A venv alone is not: it also exists after an install that failed
    half way (no network). Checked once by importing the libraries; the result is remembered, so every later
    launch is a file read instead of importing MLX."""
    if not os.path.exists(VENV_PY):
        return False
    stamp = _ready_stamp()
    try:
        with open(ENV_READY) as f:
            if f.read() == stamp:
                return True
    except OSError:
        pass
    from engine import preflight
    if not preflight.environment_ok(VENV_PY):
        return False
    try:
        with open(ENV_READY, "w") as f:
            f.write(stamp)
    except OSError:
        pass
    return True


def setup(compat=False):
    """Create/refresh the engine environment; `compat` also installs mlx-lm (the optional backend
    for architectures the native engine lacks — Qwen2/3, GLM, Llama, Gemma, MoE…)."""
    env = network_env(load_cfg())                        # pip follows the proxy and CA settings too
    from engine import preflight
    wheels = bundled_wheels()
    if (not env_ready() and not wheels) or compat:       # something is to be downloaded: say so before touching disk
        ok, why = preflight.reachable(preflight.HOSTS["pypi"])
        if not ok:
            print(R(T("No internet connection — the engine libraries must be downloaded (~300 MB).")) + f"\n  {why}\n" +
                  T("Connect to the network (or set the proxy in Settings) and start again."))
            return False
    if os.path.exists(VENV_PY):
        print(DIM(T("Environment already exists: .venv")))
    else:
        print(T("Creating .venv…"))
        os.makedirs(DATA, exist_ok=True)
        subprocess.run([find_python(), "-m", "venv", os.path.join(DATA, ".venv")], check=True)
    offline = ["--no-index", "--find-links", wheels] if wheels else []     # the offline build installs from the app itself
    r = subprocess.run([VENV_PY, "-m", "pip", "install", "-q", *offline, "-r", os.path.join(BASE, "requirements.txt")], env=env)
    if r.returncode == 0 and compat:
        print(T("Installing mlx-lm…"))
        r = subprocess.run([VENV_PY, "-m", "pip", "install", "-q", "mlx-lm"], env=env)
    if r.returncode == 0:
        try:
            with open(ENV_READY, "w") as f:
                f.write(_ready_stamp())
        except OSError:
            pass
        print(G(T("Dependencies installed.")))
    return r.returncode == 0


IMAGES_VENV = os.path.join(DATA, ".venv-images")


def setup_images():
    """`./llm setup --images`: mflux (FLUX & co on MLX) in its own environment, so the engine's stays small.
    The first image downloads its weights from Hugging Face (several GB; the model name decides)."""
    py = os.path.join(IMAGES_VENV, "bin", "python")
    if not os.path.exists(py):
        print(T("Creating the image environment…"))
        os.makedirs(DATA, exist_ok=True)
        r = subprocess.run([find_python(), "-m", "venv", IMAGES_VENV])
        if r.returncode != 0:
            return False
    print(T("Installing mflux…"))
    r = subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "mflux"], env=network_env(load_cfg()))
    if r.returncode == 0:
        print(G(T("Image generation installed. Try:  ./llm image \"a lighthouse at dusk\"")))
    return r.returncode == 0


def image_cmd(cfg, args):
    """`./llm image "prompt" [--model schnell] [--size 1024x1024] [--steps N] [--seed N] [--out file.png]`."""
    from engine import images
    opts, words, i = {}, [], 0
    while i < len(args):
        a = args[i]
        if a in ("--model", "--size", "--steps", "--seed", "--out") and i + 1 < len(args):
            opts[a[2:]] = args[i + 1]
            i += 2
        else:
            words.append(a)
            i += 1
    prompt = " ".join(words).strip()
    if not prompt:
        prompt = input(T("Describe the image") + " › ").strip()
    if not images.tool_path(DATA):
        print(R(T("Image generation is not installed. Run:  ./llm setup --images"))); return False
    if read_pid(ENGINE_PID) and RAM_GB < 24:
        print(Y(T("The engine is running; on this Mac the language model and an image model do not fit together. "
                  "Stop it first (./llm stop).")))
        return False
    out = os.path.abspath(opts.get("out") or os.path.join(os.environ.get("LLM_CWD") or os.getcwd(),
                                                          f"image-{int(time.time())}.png"))
    try:
        kw = {"model": opts.get("model", images.DEFAULT_MODEL), "size": opts.get("size")}
        if "steps" in opts: kw["steps"] = int(opts["steps"])
        if "seed" in opts: kw["seed"] = int(opts["seed"])
        print(DIM(T("Generating… (the first image downloads the model weights)")))
        png = images.generate(prompt, data_dir=DATA, **kw)
    except (images.ImageError, ValueError) as e:
        print(R(str(e))); return False
    with open(out, "wb") as f:
        f.write(png)
    print(G(T("Saved:") + f" {out}"))
    return True


def gguf_cmd(cfg, m):
    """The engine for a GGUF model: engine.gguf_server in front of llama.cpp's llama-server."""
    mc = mcfg(cfg, m["path"])
    cmd = [VENV_PY, "-m", "engine.gguf_server", "--model", m["path"], "--name", served_name(m),
           "--host", "0.0.0.0" if cfg["lan"] else "127.0.0.1", "--port", str(cfg["port"]),
           "--context", str(mc["context"]), "--max-tokens", str(mc["max_output"]),
           "--idle-sleep", str(cfg.get("idle_sleep", 15)), "--inner-log", os.path.join(STATE, "llama-server.log"),
           "--ram-gb", f"{RAM_GB:.1f}"]
    if os.environ.get("LLM_PARENT_PID"):   # the app sets this so the engine (and llama-server) stop if it crashes
        cmd += ["--parent-pid", os.environ["LLM_PARENT_PID"]]
    srv = gguf.server_path(DATA)
    if srv:
        cmd += ["--llama-server", srv]
    if cfg.get("log_level", "info") != "info":
        cmd += ["--log-level", cfg["log_level"]]
    if not cfg.get("fallback_model", True):
        cmd += ["--fallback-model", "off"]
    if not mc["vision"]:
        cmd.append("--no-vision")
    if not cfg.get("speculative", True):
        cmd.append("--no-speculative")
    return cmd


def setup_gguf():
    """`./llm setup --gguf`: llama.cpp's own macOS build into the data folder (once, ~40 MB)."""
    if gguf.server_path(DATA):
        print(DIM(T("llama.cpp is already installed:") + f" {gguf.server_path(DATA)}"))
        return True
    print(T("Installing llama.cpp (the runtime for GGUF models, one-time download)…"))
    try:
        path = gguf.install(DATA, log=lambda s: print(DIM("  " + s)))
    except (gguf.GGUFError, OSError, urllib.error.URLError, ValueError) as e:
        print(R(T("Could not install llama.cpp:") + f" {e}"))
        return False
    print(G(T("llama.cpp installed:") + f" {path}"))
    return True


def engine_cmd(cfg, m):
    native_gguf = gguf_native(cfg, m)
    if m.get("route") == "gguf" and not native_gguf:
        return gguf_cmd(cfg, m)
    mc = mcfg(cfg, m["path"])
    cmd = [VENV_PY, "-m", "engine.server", "--model", m["path"], "--name", served_name(m),
           "--host", "0.0.0.0" if cfg["lan"] else "127.0.0.1", "--port", str(cfg["port"]),
           "--context", str(mc["context"]), "--max-tokens", str(mc["max_output"]),
           "--prefill-step", str(cfg["prefill_step"]), "--max-pixels", str(mc["max_pixels"]),
           "--mem-limit", str(cfg["mem_limit"] / 100),
           "--stats-file", os.path.join(STATE, "stats.jsonl"),
           "--idle-sleep", str(cfg.get("idle_sleep", 15)), "--disk-cache-gb", str(cfg.get("disk_cache_gb", 8))]
    if os.environ.get("LLM_PARENT_PID"):   # the app sets this so the engine stops if the app crashes
        cmd += ["--parent-pid", os.environ["LLM_PARENT_PID"]]
    if m.get("route") == "ptq1":  # picking an experimental-format model is the opt-in; no separate switch
        cmd += ["--ptq1", "on"]
    if native_gguf:
        cmd += ["--backend", "gguf"]
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
    if mc.get("steer") and (m.get("route", "native") == "native" or native_gguf):
        cmd += ["--steer", mc["steer"]]
        if mc.get("steer_alpha") is not None:
            cmd += ["--steer-alpha", str(mc["steer_alpha"])]
    if cfg.get("log_level", "info") in LOG_LEVELS and cfg.get("log_level") != "info":
        cmd += ["--log-level", cfg["log_level"]]
    if not cfg.get("fallback_model", True):
        cmd += ["--fallback-model", "off"]
    if not cfg.get("prefill_guard", True):
        cmd += ["--prefill-guard", "off"]
    if cfg.get("images_api"):
        cmd += ["--images", "on"]
    return cmd


SERVICE_LABEL = "com.ai-llm-engine"
LAUNCH_AGENT = os.path.join(HOME, "Library", "LaunchAgents", SERVICE_LABEL + ".plist")


def serve_cmd(cfg):
    """Run the engine in the FOREGROUND for the selected model and replace this process with it (`os.execv`),
    so launchd — or you, by hand — supervises the real engine. It reads the current config, so a (re)start
    always serves the model you have selected now."""
    m = current_model(cfg)
    if not m:
        print(R(T("No model selected. Pick one with ./llm use <#|name>"))); return False
    blocker = route_blocker(cfg, m, interactive=False)
    if blocker:
        print(R(blocker)); return False
    cmd = engine_cmd(cfg, m)
    os.execv(cmd[0], cmd)                 # launchd (or the shell) now supervises the engine directly


def service_plist():
    """The LaunchAgent that keeps the engine available after login and restarts it on a crash. It runs
    `./llm serve` (which reads the config each time), not a baked-in command, so switching model/port in
    Settings needs no reinstall. The model still unloads when idle, so an idle service costs little RAM."""
    import plistlib
    return plistlib.dumps({
        "Label": SERVICE_LABEL,
        "ProgramArguments": [os.path.join(BASE, "llm"), "serve"],
        "EnvironmentVariables": {"LLM_HOME": DATA,
                                 "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
        "RunAtLoad": True,           # start at login
        "KeepAlive": True,           # restart if it crashes
        "ThrottleInterval": 10,      # …but not in a tight loop
        "WorkingDirectory": BASE,
        "StandardOutPath": os.path.join(STATE, "service.log"),
        "StandardErrorPath": os.path.join(STATE, "service.log"),
    })


def service_cmd(cfg, args):
    """`./llm service install | uninstall | status` — an optional background service (off by default). It is
    independent of the menu-bar app, so the local API is up for coding agents even with the app closed."""
    action = (args[0].lower() if args else "status")
    uid = os.getuid()
    if action == "install":
        os.makedirs(os.path.dirname(LAUNCH_AGENT), exist_ok=True)
        os.makedirs(STATE, exist_ok=True)
        with open(LAUNCH_AGENT, "wb") as f:
            f.write(service_plist())
        subprocess.run(["launchctl", "bootout", f"gui/{uid}/{SERVICE_LABEL}"], capture_output=True)   # idempotent
        r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", LAUNCH_AGENT], capture_output=True, text=True)
        if r.returncode != 0:                      # older macOS: load -w
            r = subprocess.run(["launchctl", "load", "-w", LAUNCH_AGENT], capture_output=True, text=True)
        if r.returncode == 0:
            print(G(T("Background service installed — it starts at login and restarts on a crash.")))
            print(DIM(T("The model still unloads when idle. Remove it with: ./llm service uninstall")))
            return True
        print(R(T("Could not start the service:") + " " + (r.stderr or r.stdout).strip())); return False
    if action == "uninstall":
        subprocess.run(["launchctl", "bootout", f"gui/{uid}/{SERVICE_LABEL}"], capture_output=True)
        subprocess.run(["launchctl", "unload", LAUNCH_AGENT], capture_output=True)
        try:
            os.remove(LAUNCH_AGENT)
        except OSError:
            pass
        print(G(T("Background service removed."))); return True
    if not os.path.exists(LAUNCH_AGENT):
        print(T("Background service: not installed. Install it with ./llm service install")); return True
    running = subprocess.run(["launchctl", "list", SERVICE_LABEL], capture_output=True).returncode == 0
    print(G(T("Background service: installed and running")) if running
          else Y(T("Background service: installed, not running")))
    return True


def route_blocker(cfg, m, interactive):
    """None when the model's route is ready; else a message (after offering the fix, if interactive).
    Only the optional mlx-lm backend needs anything installed; PTQ1 is enabled by the model itself."""
    route = m.get("route", "native")
    if route == "gguf" and not gguf_native(cfg, m) and not gguf.server_path(DATA):
        # a GGUF model was chosen on purpose and llama.cpp is its only runtime: fetch it (printing what happens)
        if interactive and not ask_yn(Y(T("This model runs on llama.cpp, which is not installed. Download it now (~40 MB)?")), True):
            return T("GGUF models need llama.cpp. Install it with:  ./llm setup --gguf")
        return None if setup_gguf() else T("Could not install llama.cpp. Try:  ./llm setup --gguf")
    if route == "mlxlm" and not mlxlm_installed():
        if interactive and ask_yn(Y(T("This model runs on the optional mlx-lm backend. Install it into .venv now?")), True):
            if setup(compat=True):
                return None
        return T("This model runs on the optional mlx-lm backend. Install it with:  ./llm setup --compat")
    return None


def start(cfg, interactive=True):
    os.makedirs(STATE, exist_ok=True)
    if not env_ready() and not setup():
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
    loading = read_pid(ENGINE_PID)
    if loading and health(cfg):
        print(Y(T("Engine is already running."))); return True
    if loading and not port_busy(cfg["port"]):
        # an engine of ours is alive but not listening yet: the port is bound only once the model is in memory.
        # The panel, the editor, the voice chat and the widget all call `start`: a second call waits for the one
        # that is loading instead of loading a second copy of the model beside it
        print(Y(T("The engine is still loading (PID {pid}) — waiting for it.").format(pid=loading)), end="", flush=True)
        t0 = time.time()
        while time.time() - t0 < 600:
            if not read_pid(ENGINE_PID):
                print(); break                      # it exited: start a fresh one below
            if health(cfg, 1):
                print(G(" " + T("ready in {s:.0f} s").format(s=time.time() - t0)))
                return True
            print(".", end="", flush=True); time.sleep(1)
        else:
            kill_pid(loading)
            try: os.remove(ENGINE_PID)
            except OSError: pass
            print(R("\n" + T("✗ Load timed out, process stopped."))); return False
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
            # somebody else's process: never cleared on its own. Say who it is; ask only when a person is here
            who = port_holder(cfg["port"])
            if who and interactive and sys.stdin.isatty() and ask_yn(
                    Y(T("Port {port} is held by {name} (PID {pid}), not by this engine. Stop it?")
                      .format(port=cfg["port"], name=who[1], pid=who[0])), False):
                kill_pid(who[0])
                time.sleep(0.5)
            if port_busy(cfg["port"]):
                held = T(" (held by {name}, PID {pid})").format(name=who[1], pid=who[0]) if who else ""
                print(R(T("Port {port} is busy{held}. Free it yourself or change the port in settings.")
                        .format(port=cfg["port"], held=held))); return False
    avail, _ = mem_info()
    if m["size"] + 1.5 > avail:
        print(Y(T("⚠ The model needs ~{need:.1f} GB, {avail:.1f} GB available — expect swapping.")
                .format(need=m["size"] + 1.5, avail=avail)))
    cmd = engine_cmd(cfg, m)
    with open(ENGINE_LOG, "a") as log:
        log.write(f"\n===== {time.ctime()} =====\n{' '.join(cmd)}\n")
    env = dict(network_env(cfg), PYTHONUNBUFFERED="1")
    if cfg["api_key"]:
        env["ENGINE_API_KEY"] = cfg["api_key"]
    with open(ENGINE_LOG, "a") as logf:
        p = subprocess.Popen(cmd, cwd=BASE, stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    # the pid file also names what was started (module or script): see read_pid
    what = next((a for a in cmd[1:] if not a.startswith("-")), "engine.server")
    with open(ENGINE_PID, "w") as f:
        f.write(f"{p.pid}\n{what}")
    print(T("Loading {name} ({size:.1f} GB)").format(name=B(m["name"]), size=m["size"]), end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            print(R("\n" + T("✗ Engine exited (code {code}):").format(code=p.returncode)))
            print(tail(ENGINE_LOG, 15))
            if gguf_native(cfg, m):          # the built-in decoder could not take this file: llama.cpp can
                print(Y(T("The built-in decoder could not load this file; trying llama.cpp.")))
                try: os.remove(ENGINE_PID)
                except OSError: pass
                return start(dict(cfg, gguf_native=False), interactive)
            return False
        if health(cfg, 1):
            break
        print(".", end="", flush=True); time.sleep(1)
    else:
        kill_pid(p.pid)
        try: os.remove(ENGINE_PID)
        except OSError: pass
        print(R("\n" + T("✗ Load timed out, process stopped."))); return False
    print(G(" " + T("ready in {s:.0f} s").format(s=time.time() - t0)))
    try:
        sync_clients(cfg)      # a connected Kilo/opencode learns about the installed models
    except Exception:
        pass
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
def chat(cfg, system="", params=None):
    m = current_model(cfg)
    if not m or not health(cfg):
        print(R(T("The model is not running."))); return
    mc = mcfg(cfg)
    url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"
    think, effort, show = think_enabled(mc, m), mc["effort"], True
    params = profile_params(params)                 # a named profile may pre-set the system prompt + sampling
    msgs = []
    # the history is the code editor's: the same file per project, so a chat continues in either place
    from harness import chats as store
    root = os.getcwd()
    sessions, session = store.load(root), store.new_session()

    def remember(question, attached, reply, reasoning):
        store.add_turn(session, "user", question, attached=attached)
        store.add_turn(session, "assistant", reply, reasoning)
        if session not in sessions: sessions.insert(0, session)
        try: store.save(root, sessions)
        except OSError as e: print(R(T("Error:") + f" {e}"))

    print(B(T("Chat with {name}.").format(name=m["name"])) +
          DIM(" /think on|off  /effort <lvl>  /show  /system <text>  /image <path> <question>"))
    print(DIM("  /clear  /save  /exit   " + T("multi-line input:") + ' """ … """'))
    print(DIM("  /history  /open <N>  /rename <N> <title>  /delete <N>  /new   " +
              T("(shared with the code editor)")))
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
        if s in ("/clear", "/new"):
            msgs, session = [], store.new_session()
            print(DIM(T("History cleared.") if s == "/clear" else T("New chat."))); continue
        if s == "/history":
            lines = store.listing(sessions, session["id"])
            print("\n".join(DIM(l) for l in lines) if lines else DIM(T("No saved chats yet."))); continue
        if s.split()[0] in ("/open", "/rename", "/delete"):
            cmd, _, rest = s.partition(" ")
            ref, _, title = rest.strip().partition(" ")
            target = store.find(sessions, ref) if ref else None
            if cmd == "/rename" and not title and ref and target is None:
                target, title = (session if session in sessions else None), rest      # rename the open chat
            if target is None:
                print(R(T("No such chat. /history lists them."))); continue
            if cmd == "/open":
                session, msgs = target, store.messages(target)
                print(DIM(T("Opened:") + f" {target['title']} ({len(msgs)})"))
                if msgs: print(DIM("› ") + str(msgs[-1]["content"])[-400:])
            elif cmd == "/rename":
                if not store.rename(target, title): print(R(T("Give the chat a name."))); continue
                store.save(root, sessions); print(DIM(T("Renamed:") + f" {target['title']}"))
            else:
                if input(Y(T("Delete “{title}”? [y/N] ").format(title=target["title"]))).strip().lower() not in ("y", "yes", "д", "да"):
                    continue
                store.delete(sessions, target); store.save(root, sessions)
                if target["id"] == session["id"]: msgs, session = [], store.new_session()
                print(DIM(T("Deleted.")))
            continue
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
        content, question, attached = s, s, []
        if s.startswith("/image "):
            _, path, *q = s.split(" ", 2)
            path = os.path.expanduser(path)
            if not os.path.exists(path): print(R(T("File not found"))); continue
            question, attached = (q[0] if q else T("Describe the image.")), [os.path.abspath(path)]
            content = [{"type": "image_url", "image_url": {"url": attached[0]}},
                       {"type": "text", "text": question}]
        msgs.append({"role": "user", "content": content})
        payload = {"model": served_name(m), "stream": True, "stream_options": {"include_usage": True},
                   "messages": ([{"role": "system", "content": system}] if system else []) + msgs,
                   "chat_template_kwargs": {"enable_thinking": think, **({"reasoning_effort": effort} if effort else {})},
                   **params}
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
        if answer: remember(question, attached, "".join(answer), "".join(reasoning))


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


def find_app():
    """LLMBar.app — it holds the editor (and the voice chat) as modules: the bundle this CLI runs from
    (…/LLMBar.app/Contents/Resources/llm), or the one built in this folder (./build-app.sh), or /Applications."""
    here = os.path.abspath(BASE)
    parts = here.split(os.sep)
    if "Resources" in parts and any(p.endswith(".app") for p in parts):
        bundle = os.sep.join(parts[:max(i for i, p in enumerate(parts) if p.endswith(".app")) + 1])
        if os.path.isdir(bundle):
            return bundle
    for p in (os.path.join(here, "LLMBar.app"), os.path.join(os.path.dirname(here), "LLMBar.app"),
              "/Applications/LLMBar.app"):
        if os.path.isdir(p):
            return p
    return None


def run_ui(args=()):
    """Open the editor on a project (default: the folder `./llm ui` was started from), in the one LLMBar app:
    the llmbar://editor link starts the app if it is not running, or adds a window to the running one."""
    app = find_app()
    if not app:
        print(R(T("The app is not built yet. Build it once with:  ./build-app.sh"))); return False
    url = "llmbar://editor"
    if "--choose" not in args:   # without a root the editor reopens its last project, or asks
        root = next((a for a in args if not a.startswith("--")), None) or os.environ.get("LLM_CWD") or os.getcwd()
        root = os.path.abspath(os.path.join(os.environ.get("LLM_CWD") or os.getcwd(), root))
        url += "?root=" + urllib.parse.quote(root, safe="")
    return subprocess.run(["open", "-a", app, url]).returncode == 0


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


def client_limits(cfg, m, running=True):
    """Context / output limits for client configs: the running engine's values (when `m` is the loaded
    model), else an estimate."""
    mc = dict(mcfg(cfg, m["path"]))
    h = health(cfg) if running else None
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


def _legacy_model_id(m):
    """The id earlier versions registered for a hub download: "<org>/<name>" from the folder layout
    (~/.cache/huggingface/hub/<org>/<name>). Editor sessions that still name it must resolve."""
    root = os.path.normpath(os.path.join(HOME, ".cache", "huggingface", "hub"))
    rel = os.path.relpath(os.path.normpath(m["path"]), root)
    bits = rel.split(os.sep)
    return "/".join(bits) if len(bits) == 2 and not rel.startswith("..") else None


def _write_opencode_style(path, cfg, m, ctx, out, alias=(), others=()):
    """Kilo Code and opencode share the opencode provider schema. `m` is the selected model; `others`
    are (model, context, output) for the rest of the installed ones, so the client can offer them all
    (the engine serves one at a time and ignores the name, so picking another one only labels it)."""
    data = {}
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")
        data = _read_jsonc(path)
    prov = data.setdefault("provider", {})
    opts = {"baseURL": f"http://localhost:{cfg['port']}/v1"}
    if cfg["api_key"]:
        opts["apiKey"] = cfg["api_key"]
    models = {}
    for mm, c, o in [(m, ctx, out), *others]:
        vision = mcfg(cfg, mm["path"])["vision"] and mm["vision"]
        entry = {"name": f"{mm['name']} (local)", "tool_call": True, "reasoning": mm["thinking"],
                 "attachment": vision,
                 "modalities": {"input": ["text", "image"] if vision else ["text"], "output": ["text"]},
                 "limit": {"context": c, "output": o}}
        models[served_name(mm)] = entry
        legacy = _legacy_model_id(mm)
        if legacy:
            models.setdefault(legacy, entry)
    name = served_name(m)
    prov["local_mlx"] = {"name": "AI-llm-Engine (local)", "npm": "@ai-sdk/openai-compatible", "options": opts,
                         "models": models}
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
        _write_opencode_style(path, cfg, m, ctx, out, alias=("local_qwen",) if which == "kilo" else (),
                              others=_other_models(cfg, m))
    print(G(T("{client} configured: {name}, context {ctx}, output {out}").format(
        client=title, name=served_name(m), ctx=ctx, out=out)))
    print(DIM(T("File: {path} (backup: .bak) · reload the editor window to apply.").format(path=path)))
    return True


def _other_models(cfg, current):
    """(model, context, output) of every other supported, installed model."""
    return [(x, *client_limits(cfg, x, running=False)) for x in scan_models(cfg)
            if x["supported"] and x["path"] != current["path"]]


def sync_clients(cfg):
    """Refresh Kilo Code / opencode when they were connected before: the installed models changed or a
    different one was selected. Never creates a client config that is not there already."""
    m = current_model(cfg)
    if not m:
        return []
    done = []
    for which in ("kilo", "opencode"):
        title, path = CLIENTS[which]
        try:
            if not os.path.exists(path) or "local_mlx" not in open(path, errors="replace").read():
                continue
            ctx, out = client_limits(cfg, m)
            _write_opencode_style(path, cfg, m, ctx, out, alias=("local_qwen",) if which == "kilo" else (),
                                  others=_other_models(cfg, m))
            done.append(title)
        except (OSError, ValueError):
            pass
    return done


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
    if "mlx" in t or ("safetensors" in t and "gguf" not in t):
        return engine_route({}, t)          # a repo that has MLX weights is pulled (and run) as MLX
    if "gguf" in t:
        return "gguf"                        # llama.cpp runs it
    return "none"


MS_ENDPOINT = os.environ.get("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn").rstrip("/")
MS_PREFIX = "ms:"       # a repo id from ModelScope carries this prefix: "ms:mlx-community/Qwen3-0.6B-4bit"


def _split_source(repo):
    """("hf" | "modelscope", id without the prefix)"""
    return ("modelscope", repo[len(MS_PREFIX):]) if repo.startswith(MS_PREFIX) else ("hf", repo)


def _ms(path, data=None, method="GET", timeout=20):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(MS_ENDPOINT + path, data=body, method=method,
                                 headers={"User-Agent": "AI-llm-Engine", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    if out.get("Code") not in (200, None) or out.get("Success") is False:
        raise OSError(out.get("Message") or "ModelScope error")
    return out.get("Data", out)


def ms_files(repo):
    """The repo's files in the shape the hub listing has: [{rfilename, size, lfs: {sha256}}]."""
    data = _ms(f"/api/v1/models/{urllib.parse.quote(repo)}/repo/files?Revision=master&Recursive=true")
    return [{"rfilename": f["Path"], "size": f.get("Size") or 0, "lfs": {"sha256": f.get("Sha256") or None}}
            for f in data.get("Files", []) if f.get("Type") == "blob"]


def ms_search(query, limit=20):
    """MLX models on ModelScope (a mirror that is reachable where huggingface.co is not), same shape as
    hf_search; ids carry the "ms:" prefix so a pull knows where to go."""
    data = _ms("/api/v1/dolphin/models", method="PUT", data={
        "PageSize": limit, "PageNumber": 1, "SortBy": "Default", "Target": "", "SingleCriterion": [],
        "Name": f"{query} mlx" if "mlx" not in query.lower() else query})
    out = []
    for m in data.get("Model", {}).get("Models", []):
        tags = list(dict.fromkeys([*(m.get("Libraries") or []), *(m.get("Tags") or [])]))
        if "mlx" not in [t.lower() for t in tags]:
            continue
        rid = f"{m.get('Path')}/{m.get('Name')}"
        size = (m.get("StorageSize") or 0) / 2**30 or None
        state = install_state(MS_PREFIX + rid)
        route = hub_route([*tags, "safetensors"])
        out.append({"id": MS_PREFIX + rid, "downloads": m.get("Downloads", 0), "likes": m.get("Stars", 0), "size": size,
                    "route": route, "compatible": route != "none", "local": state == "complete",
                    "partial": state == "partial", "fits": size is not None and size + 2.0 <= RAM_GB * 0.75})
    return out


def _pull_dest(repo):
    return os.path.join(HOME, ".cache", "huggingface", "hub", *_split_source(repo)[1].split("/"))


def install_state(repo, siblings=None):
    """"complete" | "partial" | "none" for a repo's download folder. With the hub listing (sizes) every
    wanted file must be there at its full size; without it, a folder with a config, weights and no
    .part files counts as complete. A folder that merely exists is NOT a finished download."""
    dest = _pull_dest(repo)
    if not os.path.isdir(dest):
        return "none"
    if siblings and any(f["rfilename"].lower().endswith(".gguf") for f in siblings) and not any(
            f["rfilename"].lower().endswith(".safetensors") for f in siblings):
        files, _, _ = gguf.pick_files(siblings)
        have = [f for f in files if os.path.isfile(os.path.join(dest, f["rfilename"]))
                and (not f.get("size") or os.path.getsize(os.path.join(dest, f["rfilename"])) == f["size"])]
        any_part = any(n.endswith(".part") for _, _, fs in os.walk(dest) for n in fs)
        return "complete" if any(not gguf.is_projector(f["rfilename"]) for f in have) and not any_part else "partial"
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


def hf_search(query, limit=20, source="hf", fmt="mlx"):
    """MLX (or, with fmt="gguf", GGUF) models on the hub, most downloaded first, with size and engine
    compatibility. source="modelscope" searches ModelScope instead (ids come back with the "ms:" prefix)."""
    if source == "modelscope":
        return ms_search(query, limit)
    from concurrent.futures import ThreadPoolExecutor
    found = _hf(f"/api/models?search={urllib.parse.quote(query)}&filter={'gguf' if fmt == 'gguf' else 'mlx'}"
                f"&sort=downloads&direction=-1&limit={limit}")

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


class _SameHostAuth(urllib.request.HTTPRedirectHandler):
    """A redirect to another host (the CDN behind a hub's /resolve/ URL) must not carry the hub's token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            for h in list(new.headers):
                if h.lower() == "authorization":
                    del new.headers[h]
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_DOWNLOAD_OPENER = urllib.request.build_opener(_SameHostAuth)


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
            with _DOWNLOAD_OPENER.open(req, timeout=60) as r:
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


def hf_pull(repo, progress=False, revision=None):
    """Download a repo into ~/.cache/huggingface/hub/<org>/<name> with resume (Range) and atomic renames.
    A GGUF repo holds many quantizations: "org/name:Q4_K_M" takes one (default: the best-balanced on offer)."""
    repo, quant = gguf.split_repo_quant(repo)
    source, rid = _split_source(repo)
    if not REPO_ID_RE.match(rid):
        print(R(T("Download failed: {name}: {err}").format(name=repo, err=T("not a repository id (org/name)"))))
        return False
    if source == "modelscope":
        try:
            info = {"siblings": ms_files(rid), "sha": "master"}
        except (OSError, urllib.error.URLError, ValueError, KeyError) as e:
            print(R(T("Download failed: {name}: {err}").format(name=repo, err=e))); return False
    elif revision and re.match(r"^[0-9a-f]{7,40}$", revision):       # the exact commit an edit was made for
        info = _hf(f"/api/models/{repo}/revision/{revision}?blobs=true")
    else:
        info = _hf(f"/api/models/{repo}?blobs=true")
    files = [f for f in info.get("siblings", []) if not f["rfilename"].lower().endswith(PULL_SKIP)]
    gg = [f for f in info.get("siblings", []) if f["rfilename"].lower().endswith(".gguf")]
    if gg and not any(f["rfilename"].lower().endswith(".safetensors") for f in info.get("siblings", [])):
        try:
            files, chosen, offered = gguf.pick_files(gg, quant)
        except gguf.GGUFError as e:
            print(R(str(e))); return False
        print(DIM(T("GGUF quantization: {q}  (on offer: {all})").format(q=chosen, all=", ".join(offered))), flush=True)
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
    if os.environ.get("HF_TOKEN") and source == "hf":      # the Hugging Face token goes to Hugging Face only
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
        path = os.path.realpath(os.path.join(dest, name))
        if not path.startswith(os.path.realpath(dest) + os.sep):   # "../x" or an absolute name from the listing
            print("\n" + R(T("Download failed: {name}: {err}").format(name=name, err=T("unsafe file name"))))
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for q in (path, path + ".part"):  # what's already on disk counts as progress
            if os.path.exists(q):
                state["done"] += min(os.path.getsize(q), size or os.path.getsize(q))
                break
        if source == "modelscope":
            url = (f"{MS_ENDPOINT}/api/v1/models/{urllib.parse.quote(rid)}/repo?Revision={rev}"
                   f"&FilePath={urllib.parse.quote(name)}")
        else:
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


ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def _fetch_basename(url):
    """The file name a URL ends in, without the query string."""
    path = urllib.parse.urlsplit(url).path
    return urllib.parse.unquote(os.path.basename(path.rstrip("/"))) or "download"


def _fetch_name(url):
    """A folder name for a fetched source: the magnet's display name, or the URL's file name without its
    archive extension."""
    if url.startswith("magnet:"):
        dn = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("dn", [""])[0]
        return (dn or "custom").replace("/", "-").strip() or "custom"
    base = _fetch_basename(url)
    low = base.lower()
    for ext in sorted(ARCHIVE_EXTS, key=len, reverse=True):
        if low.endswith(ext):
            return base[: -len(ext)] or "custom"
    return os.path.splitext(base)[0] or base


def _safe_extract(archive, dest):
    """Extract a .zip or .tar* into `dest`, refusing any member that would escape it (absolute paths,
    ``..`` or a symlink pointing outside)."""
    import tarfile
    import zipfile
    root = os.path.realpath(dest)

    def inside(name):
        return os.path.realpath(os.path.join(dest, name)).startswith(root + os.sep)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if not inside(n):
                    raise ValueError(T("Unsafe path in archive: {name}").format(name=n))
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if (m.issym() or m.islnk()) or not inside(m.name):
                    raise ValueError(T("Unsafe path in archive: {name}").format(name=m.name))
            t.extractall(dest)
    else:
        raise ValueError(T("Not a .zip or .tar archive"))


def _remote_size(url, headers):
    """Content-Length for a URL, or 0 when the server does not say."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=20) as r:
            return int(r.headers.get("Content-Length") or 0)
    except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError):
        return 0


def fetch_cmd(cfg, args):
    """`./llm fetch <url|magnet> [name] [--progress]` — add a model from a source **you** provide: a direct
    http(s) link to a `.gguf` or an archive (.zip/.tar*), or a magnet/.torrent (needs `aria2c`). It only
    downloads what you point it at, into your models folder; you are responsible for the source's licence."""
    progress = "--progress" in args
    rest = [a for a in args if not a.startswith("--")]
    if not rest:
        print(T("Usage: ./llm fetch <url|magnet> [name]")); return False
    url = rest[0]
    name = (rest[1] if len(rest) > 1 else _fetch_name(url)).strip("/") or "custom"
    base = os.path.expanduser(cfg["dirs"][0])
    dest = os.path.join(base, "custom", name)
    if url.startswith("magnet:") or url.lower().endswith(".torrent"):
        return _fetch_torrent(url, dest, progress)
    return _fetch_http(url, dest, progress)


def _fetch_torrent(url, dest, progress):
    aria = shutil.which("aria2c")
    if not aria:
        print(R(T("Install aria2 to fetch magnet/torrent links: brew install aria2"))); return False
    os.makedirs(dest, exist_ok=True)
    print(T("Fetching {url} → {dest}").format(url=url[:60], dest=dest), flush=True)
    p = subprocess.Popen([aria, "--dir", dest, "--seed-time=0", "--summary-interval=1",
                          "--console-log-level=warn", url],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = ""
    for line in p.stdout:                             # aria2c prints "... (53%) ..." — forward it as PROGRESS
        m = re.search(r"\((\d+)%\)", line)
        if m and progress:
            pct = m.group(1)
            if pct != last:
                last = pct; print(f"PROGRESS {pct} 100 {os.path.basename(dest)}", flush=True)
        elif not progress:
            print("  " + line.rstrip(), flush=True)
    ok = p.wait() == 0
    if ok and progress:
        print("PROGRESS 100 100 done", flush=True)
    print((G(T("Fetched: {dest}")) if ok else R(T("Download failed"))).format(dest=dest))
    return ok


def _fetch_http(url, dest, progress):
    os.makedirs(dest, exist_ok=True)
    fname = _fetch_basename(url)
    is_archive = fname.lower().endswith(ARCHIVE_EXTS)
    headers = {"User-Agent": "AI-llm-Engine"}
    total = _remote_size(url, headers)
    free = shutil.disk_usage(dest).free
    if total and total > free - 2 * 2**30:
        print(R(T("Not enough disk space: need {need:.1f} GB, free {free:.1f} GB")
                .format(need=total / 2**30, free=free / 2**30))); return False
    target = os.path.realpath(os.path.join(dest, fname))
    if not target.startswith(os.path.realpath(dest) + os.sep):
        print(R(T("Unsafe path in archive: {name}").format(name=fname))); return False
    print(T("Fetching {url} → {dest}").format(url=url[:60], dest=dest), flush=True)
    state = {"done": 0, "last": 0.0}
    t0 = time.time()

    def tick(n):
        state["done"] += n
        now = time.time()
        if now - state["last"] > 0.5:
            state["last"] = now
            if progress and total:
                print(f"PROGRESS {min(state['done'], total)} {total} {fname}", flush=True)
            elif not progress:
                speed = state["done"] / max(1e-6, now - t0) / 2**20
                print(f"\r  {state['done'] / 2**30:5.2f} GB  {speed:6.1f} MB/s  {fname[:40]:40s}", end="", flush=True)
    try:
        download_file(url, target, total, None, headers, lambda n: tick(n))
    except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError) as e:
        print("\n" + R(T("Download failed: {name}: {err}").format(name=fname, err=e))); return False
    if is_archive:
        try:
            _safe_extract(target, dest)
            os.remove(target)
        except (ValueError, OSError) as e:
            print("\n" + R(T("Download failed: {name}: {err}").format(name=fname, err=e))); return False
    if progress and total:
        print(f"PROGRESS {total} {total} done", flush=True)
    print("\n" + G(T("Fetched: {dest}").format(dest=dest)))
    return True


def search_cmd(query, as_json=False, source="hf"):
    if not query:
        query = input(T("Search MLX models on Hugging Face") + " › ").strip()
    try:
        res = hf_search(query, source=source)
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
                  ("fallback_model", f"{T('Model fallback'):26s}: {on if cfg['fallback_model'] else off}",
                   DIM(T("another model name is answered by the loaded one instead of 404"))),
                  ("prefill_guard", f"{T('Prefill memory guard'):26s}: {on if cfg['prefill_guard'] else off}",
                   DIM(T("refuse prompts that would not fit in free memory"))),
                  ("log_level", f"{T('Log level'):26s}: {cfg['log_level']}", DIM(" / ".join(LOG_LEVELS))),
                  ("proxy_http", f"{T('HTTP proxy'):26s}: {cfg['proxy_http'] or '—'}", ""),
                  ("proxy_https", f"{T('HTTPS proxy'):26s}: {cfg['proxy_https'] or '—'}", ""),
                  ("no_proxy", f"{T('No proxy for'):26s}: {cfg['no_proxy'] or '—'}", DIM(T("hosts / CIDRs, comma-separated"))),
                  ("ca_bundle", f"{T('CA bundle (PEM)'):26s}: {cfg['ca_bundle'] or '—'}", ""),
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
        elif key in ("lan", "auto_free", "fallback_model", "prefill_guard"):
            cfg[key] = not cfg[key]
        elif key == "log_level":
            cfg[key] = LOG_LEVELS[(LOG_LEVELS.index(cfg[key]) + 1) % len(LOG_LEVELS)] if cfg[key] in LOG_LEVELS else "info"
        elif key in ("proxy_http", "proxy_https", "no_proxy", "ca_bundle"):
            cfg[key] = input(T("New value") + f" [{cfg[key]}] (" + T("empty to clear") + "): ").strip()
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


# ───────────────────────────── the Conductor (optional) ─────────────────────────────
def _conduct_local_fallback(cfg):
    m = current_model(cfg)
    return {"local": m["name"]} if m else None


def _conduct_switch(cfg):
    """A `switch_local(model)` for the Conductor: load `model` on the one engine, wait until it serves, and
    return its local OpenAI endpoint. Reuses the normal select + (re)start + health path."""
    def switch(model):
        if not choose_model(cfg, model):
            raise RuntimeError(f"cannot select model {model!r}")
        if not (read_pid(ENGINE_PID) and health(cfg)) and not start(cfg, interactive=False):
            raise RuntimeError(f"the engine did not start on {model!r}")
        if not health(cfg, timeout=600):
            raise RuntimeError(f"the engine is not answering on {model!r}")
        return {"type": "openai", "url": f"http://127.0.0.1:{cfg['port']}/v1", "model": "local"}
    return switch


def _conduct_status(cfg, conductor, root):
    from harness import delegate
    roles, dropped = conductor.resolve_roles(root, local=_conduct_local_fallback(cfg))
    print(B(T("Conductor") + " — " + T("optional; working with one model is unaffected")))
    print(DIM(conductor.config_path(root)))
    for r in conductor.ROLES:
        if r in roles:
            kind = T("local, takes turns") if conductor.is_local(roles[r]) else T("remote, parallel")
            print(f"  {r:<5} → {conductor.where(roles[r])}   [{kind}]")
        elif r in dropped:
            print(f"  {r:<5} → " + Y(T("unusable")) + f": {dropped[r]}")
        else:
            print(f"  {r:<5} → " + DIM(T("not set — uses the selected model")))
    installed = [m["name"] for m in scan_models(cfg) if m["supported"]]
    print("\n" + T("Installed models:") + " " + (", ".join(installed) or T("none")))
    workers = delegate.load(root)
    print(T("Workers:") + " " + (", ".join(workers) or T("none")))
    if "heavy" not in roles:
        print(Y("• " + T("set 'heavy' (it plans and synthesizes): llm conduct set heavy <model|worker>")))
    if roles and len({conductor.where(roles[r]) for r in roles}) == 1:
        print(DIM("• " + T("all roles use one model — the Conductor adds little, which is fine")))
    print(DIM("\n" + T("set:") + " llm conduct set <heavy|code|fast> <model|worker>    "
              + T("run:") + " llm conduct \"<task>\""))
    return True


def _conduct_set(cfg, conductor, root, role, value):
    from harness import delegate
    if role not in conductor.ROLES:
        print(R(T("role must be one of") + " " + ", ".join(conductor.ROLES))); return False
    kind, name = None, value
    if value.startswith("local:"):
        kind, name = "local", value[6:].strip()
    elif value.startswith("worker:"):
        kind, name = "worker", value[7:].strip()
    workers = delegate.load(root)
    if kind in (None, "local"):
        hits = [m for m in scan_models(cfg) if m["supported"]
                and (name.lower() in m["name"].lower() or m["path"] == name)]
        if len(hits) > 1 and kind != "local":
            print(Y(T("Several models:") + " " + ", ".join(h["name"] for h in hits))); return False
        if hits:
            conductor.set_role(root, role, {"local": hits[0]["name"]})
            print(G(f"{role} → {hits[0]['name']} " + T("(local)"))); return True
        if kind == "local":
            print(R(T("No installed model matches") + f" {name!r}")); return False
    if kind in (None, "worker") and name in workers:
        conductor.set_role(root, role, name)
        print(G(f"{role} → {name} " + T("(worker)"))); return True
    print(R(T("No installed model or worker matches") + f" {value!r}")); return False


def _conduct_status_json(cfg, conductor, root):
    """The role mapping and the choices a UI offers, as one JSON line (for the editor's Conductor panel)."""
    from harness import delegate
    roles, dropped = conductor.resolve_roles(root, local=_conduct_local_fallback(cfg))
    raw = conductor.load_mapping(root)
    out = {"roles": {}, "installed": [m["name"] for m in scan_models(cfg) if m["supported"]],
           "workers": list(delegate.load(root)),
           "configPath": conductor.config_path(root)}
    for r in conductor.ROLES:
        if r in roles:
            out["roles"][r] = {"set": r in raw, "ok": True, "local": conductor.is_local(roles[r]),
                               "label": conductor.where(roles[r])}
        else:
            out["roles"][r] = {"set": r in raw, "ok": False, "local": False,
                               "label": dropped.get(r, "not set — uses the selected model")}
    print(json.dumps(out, ensure_ascii=False))
    return True


def conduct_cmd(cfg, args):
    """Optional orchestrator: split a task across models by role. Opt-in — single-model work is unaffected.
    `--root DIR` points at the project (the editor passes it); `--json` prints the status for a UI."""
    from harness import conductor
    root, as_json, rest = os.getcwd(), False, []
    i = 0
    while i < len(args):
        if args[i] == "--root" and i + 1 < len(args):
            root = args[i + 1]; i += 2; continue
        if args[i] == "--json":
            as_json = True; i += 1; continue
        rest.append(args[i]); i += 1
    args = rest
    sub = args[0] if args else "status"
    if as_json or (sub in ("status", "roles", "show") and len(args) <= 1):
        return _conduct_status_json(cfg, conductor, root) if as_json else _conduct_status(cfg, conductor, root)
    if sub == "set":
        if len(args) < 3:
            print(R("usage: llm conduct set <heavy|code|fast> <model|worker>")); return False
        return _conduct_set(cfg, conductor, root, args[1], " ".join(args[2:]))
    if sub == "unset" and len(args) == 2:
        conductor.set_role(root, args[1], None); print(G(f"{args[1]} cleared")); return True
    task = " ".join(args).strip()
    if not task:
        return _conduct_status(cfg, conductor, root)
    try:
        answer = conductor.conduct(task, root, local=_conduct_local_fallback(cfg),
                                   switch_local=_conduct_switch(cfg), emit=lambda line: print(DIM(line)))
    except RuntimeError as e:
        print(R(str(e))); return False
    print("\n" + (answer or "(empty)"))
    return True


def convert_cmd(cfg, args):
    """`./llm convert <model.gguf> [out_dir]` — export a GGUF as an MLX folder (config.json, f16 safetensors,
    tokenizer.json), so a converted model opens with mlx-lm without the GGUF around."""
    args = [a for a in args if a and not a.startswith("-")]
    if not args:
        print("usage: ./llm convert <model.gguf> [out_dir]")
        return False
    src = args[0]
    if not os.path.isfile(src) or not src.lower().endswith(".gguf"):
        print(R(f"not a .gguf file: {src}"))
        return False
    if not env_ready() and not setup():
        return False
    out = args[1] if len(args) > 1 else os.path.splitext(src)[0] + "-mlx"
    return subprocess.run([VENV_PY, "-m", "engine.convert", src, out], cwd=BASE).returncode == 0


def main():
    cmd, arg = (sys.argv[1:] + ["", ""])[:2]
    cfg, tty = load_cfg(), sys.stdin.isatty()
    set_lang(cfg["lang"])
    os.environ.update(network_env(cfg))      # hub search, downloads and the agent's web tools follow the proxy
    actions = {
        "check": lambda: check_cmd(cfg, "--json" in sys.argv),
        "web": lambda: web_cmd(sys.argv[2:]),
        "inspect": lambda: inspect_cmd(cfg, sys.argv[2:]),
        "setup": lambda: setup_images() if "--images" in sys.argv else (
            setup_gguf() if "--gguf" in sys.argv else setup(compat="--compat" in sys.argv)),
        "image": lambda: image_cmd(cfg, sys.argv[2:]),
        "start": lambda: start(cfg, tty), "stop": lambda: stop(cfg),
        "restart": lambda: restart(cfg, tty), "status": lambda: status(cfg),
        "models": lambda: print(json.dumps(scan_models(cfg), ensure_ascii=False)) if arg == "--json"
        else list_models(cfg, scan_models(cfg)), "use": lambda: choose_model(cfg, arg or None),
        "run": lambda: run_cmd(cfg, sys.argv[2:]),
        "show": lambda: show_cmd(cfg, sys.argv[2:]),
        "ps": lambda: ps_cmd(cfg, sys.argv[2:]),
        "create": lambda: create_cmd(cfg, sys.argv[2:]),
        "fav": lambda: fav_cmd(cfg, sys.argv[2:]),
        "chat": lambda: chat(cfg), "bench": lambda: bench(cfg), "logs": logs,
        "free": lambda: free_memory(tty), "connect": lambda: connect_client(cfg, arg or None), "kilo": lambda: connect_client(cfg, "kilo"), "test": run_tests,
        "rm": lambda: remove_cmd(cfg, sys.argv[2:]),
        "vision": lambda: set_vision(cfg, arg or "on"),
        "steer": lambda: steer_cmd(cfg, sys.argv[2:]),
        "perf": lambda: run_perf(sys.argv[2:]),
        "eval": lambda: run_eval(sys.argv[2:]),
        "calibrate": lambda: run_calibrate(sys.argv[2:]),
        "agent": lambda: __import__("agent").main(sys.argv[2:]),
        "conduct": lambda: conduct_cmd(cfg, sys.argv[2:]),
        "convert": lambda: convert_cmd(cfg, sys.argv[2:]),
        "ui": lambda: run_ui(sys.argv[2:]),
        "up": lambda: up(cfg), "open": open_window,
        "serve": lambda: serve_cmd(cfg), "service": lambda: service_cmd(cfg, sys.argv[2:]),
        "search": lambda: search_cmd(" ".join(a for a in sys.argv[2:] if not a.startswith("--") and a != "modelscope"),
                                     "--json" in sys.argv, "modelscope" if "--source" in sys.argv and "modelscope" in sys.argv else "hf"),
        "pull": lambda: hf_pull(arg, "--progress" in sys.argv) if arg else print(__doc__),
        "fetch": lambda: fetch_cmd(cfg, sys.argv[2:]), "": menu, "menu": menu,
    }
    if cmd not in actions:
        print(__doc__); sys.exit(2)
    ok = actions[cmd]()
    sys.exit(1 if ok is False else 0)


if __name__ == "__main__":
    main()

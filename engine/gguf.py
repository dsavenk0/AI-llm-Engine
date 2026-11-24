"""GGUF files: read their header, find them on disk, and fetch the llama.cpp server that runs them.

GGUF is what most people download (Hugging Face, Ollama, LM Studio). The engine does not reimplement those
architectures: it hands the file to llama.cpp's own `llama-server`, installed on demand into the data folder
(`./llm setup --gguf`), and keeps the same OpenAI-compatible front. Stdlib only.
"""
import json
import os
import re
import struct
import subprocess
import tarfile
import tempfile
import urllib.request
from functools import lru_cache

MAGIC = b"GGUF"
# value types of the metadata section
_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# llama_ftype -> the quantization name people use
FTYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
         12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
         19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
         26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16"}
WANTED = ("general.architecture", "general.name", "general.file_type", "general.size_label", "general.basename",
          "tokenizer.chat_template", "general.quantization_version")


class GGUFError(ValueError):
    pass


def _read(f, fmt):
    size = struct.calcsize(fmt)
    b = f.read(size)
    if len(b) != size:
        raise GGUFError("truncated GGUF header")
    return struct.unpack(fmt, b)[0]


def _string(f, keep=True, limit=1 << 20):
    n = _read(f, "<Q")
    if n > (1 << 28):
        raise GGUFError("implausible string length")
    if keep and n <= limit:
        return f.read(n).decode("utf-8", "replace")
    f.seek(n, 1)
    return None


def _value(f, t, keep):
    if t in _FIXED:
        v = _read(f, _FIXED[t])
        return v
    if t == 8:
        return _string(f, keep)
    if t == 9:                                   # array: element type, count, elements
        et, n = _read(f, "<I"), _read(f, "<Q")
        if et in _FIXED and n <= 64:             # small numeric arrays (rope sections, image mean/std) are kept
            return [_read(f, _FIXED[et]) for _ in range(n)]
        if et in _FIXED:
            f.seek(n * struct.calcsize(_FIXED[et]), 1)
        elif et == 8:
            for _ in range(n):                   # token lists run to 150k strings: walk the lengths
                f.seek(_read(f, "<Q"), 1)
        else:
            raise GGUFError(f"unsupported array element type {et}")
        return None
    raise GGUFError(f"unsupported metadata type {t}")


def read_header(path):
    """The few metadata fields the engine cares about, plus the architecture's context length.
    Raises GGUFError for anything that is not a readable GGUF."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise GGUFError("not a GGUF file")
        version = _read(f, "<I")
        if version < 2:
            raise GGUFError(f"GGUF version {version} is not supported")
        _read(f, "<Q")                           # tensor count
        kv = _read(f, "<Q")
        out, arch = {"version": version}, None
        for _ in range(kv):
            key = _string(f)
            t = _read(f, "<I")
            want = key in WANTED or (arch and key.startswith(arch + ".") and key.endswith(
                (".context_length", ".block_count", ".expert_count", ".expert_used_count")))
            v = _value(f, t, want)
            if key == "general.architecture":
                arch = v
            if want and v is not None:
                out[key] = v
    ft = out.get("general.file_type")
    out["architecture"] = out.get("general.architecture", "?")
    out["name"] = out.get("general.name") or out.get("general.basename") or os.path.basename(path)
    out["quant"] = FTYPE.get(ft, f"ftype {ft}") if ft is not None else guess_quant(os.path.basename(path))
    out["context_length"] = out.get(f"{arch}.context_length") if arch else None
    out["expert_count"] = out.get(f"{arch}.expert_count") if arch else None
    out["expert_used_count"] = out.get(f"{arch}.expert_used_count") if arch else None
    out["chat_template"] = out.get("tokenizer.chat_template", "")
    return out


def read_header_full(path):
    """Every scalar/string metadata value (arrays skipped), plus where the tensor directory starts."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise GGUFError("not a GGUF file")
        version = _read(f, "<I")
        count = _read(f, "<Q")
        kv = _read(f, "<Q")
        out = {"_version": version, "_tensor_count": count}
        for _ in range(kv):
            key = _string(f)
            t = _read(f, "<I")
            v = _value(f, t, True)
            if v is not None:
                out[key] = v
            elif t == 9:                      # arrays: keep the small ones (rope scaling lists etc. are numeric)
                pass
        out["_tensor_info_offset"] = f.tell()
    return out


def read_array(path, key, limit=1 << 24):
    """One metadata array (e.g. tokenizer.ggml.tokens) as a Python list; None when absent."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise GGUFError("not a GGUF file")
        _read(f, "<I")
        _read(f, "<Q")
        kv = _read(f, "<Q")
        for _ in range(kv):
            k = _string(f)
            t = _read(f, "<I")
            if k == key and t == 9:
                et, n = _read(f, "<I"), _read(f, "<Q")
                if n > limit:
                    raise GGUFError("array too large")
                if et == 8:
                    return [_string(f, True, 1 << 20) or "" for _ in range(n)]
                fmt = _FIXED[et]
                size = struct.calcsize(fmt)
                data = f.read(n * size)
                return list(struct.unpack(f"<{n}{fmt[1]}", data))
            _value(f, t, False)
    return None


def has_array(path, key):
    """Is `key` a metadata array of the file? (Nothing is read: a vocabulary can be hundreds of thousands of entries.)"""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise GGUFError("not a GGUF file")
        _read(f, "<I")
        _read(f, "<Q")
        kv = _read(f, "<Q")
        for _ in range(kv):
            k = _string(f)
            t = _read(f, "<I")
            if k == key and t == 9:
                return True
            _value(f, t, False)
    return False


# ggml tensor types the own decoder reads (engine/gguf_tensors.py): F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0
# Q2_K Q3_K Q4_K Q5_K Q6_K IQ4_NL IQ4_XS and the codebook i-quants IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S
NATIVE_TYPES = {0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23, 30,
                16, 17, 18, 19, 21, 22, 29}          # + IQ2_XXS IQ2_XS IQ3_XXS IQ1_S IQ3_S IQ2_S IQ1_M
NATIVE_ARCHS = ("llama", "qwen2", "qwen3", "gemma2", "gemma3", "granite", "granitemoe", "qwen3moe", "minicpm",
                "qwen35", "qwen35moe")
# pre-tokenizer rules verified against llama.cpp (engine/gguf_tokenizer.py keeps the expressions)
NATIVE_PRE = ("default", "gpt2", "llama-bpe", "llama3", "smaug-bpe", "qwen2", "qwen35", "deepseek-r1-qwen",
              "smollm", "refact")


def tensor_types(path):
    """The set of ggml type ids of every tensor in the file (stdlib; reads only the tensor directory)."""
    meta = read_header_full(path)
    types = set()
    with open(path, "rb") as f:
        f.seek(meta["_tensor_info_offset"])
        for _ in range(meta["_tensor_count"]):
            n = _read(f, "<Q")
            f.seek(n, 1)
            nd = _read(f, "<I")
            f.seek(8 * nd, 1)
            types.add(_read(f, "<I"))
            f.seek(8, 1)
    return types


def tensor_names(path):
    """The names of every tensor in the file (stdlib)."""
    meta = read_header_full(path)
    names = set()
    with open(path, "rb") as f:
        f.seek(meta["_tensor_info_offset"])
        for _ in range(meta["_tensor_count"]):
            n = _read(f, "<Q")
            names.add(f.read(n).decode("utf-8", "replace"))
            nd = _read(f, "<I")
            f.seek(8 * nd + 4 + 8, 1)
    return names


def native_support(path):
    """(True, "") when the engine's own MLX decoder can run this GGUF, else (False, why). Cheap and
    stdlib-only: the answer decides whether llama.cpp has to be installed at all."""
    try:
        meta = read_header_full(path)
        arch = meta.get("general.architecture")
        if arch not in NATIVE_ARCHS:
            return False, f"architecture {arch!r}"
        if meta.get(f"{arch}.expert_count"):
            names = tensor_names(path)
            if "blk.0.ffn_gate_exps.weight" not in names or "blk.0.ffn_gate_inp.weight" not in names:
                return False, "mixture of experts stored per expert"
            if "blk.0.ffn_gate_shexp.weight" in names and arch != "qwen35moe":
                return False, "mixture of experts with a shared expert"
        tk = meta.get("tokenizer.ggml.model")
        if tk == "llama":                              # SentencePiece (Gemma, Mistral, MiniCPM): scored pieces
            if not has_array(path, "tokenizer.ggml.scores"):
                return False, "tokenizer: SentencePiece without scores"
        elif tk != "gpt2":
            return False, f"tokenizer {tk!r}"
        elif meta.get("tokenizer.ggml.pre", "default") not in NATIVE_PRE:
            return False, f"pre-tokenizer {meta.get('tokenizer.ggml.pre')!r}"
        scaling = meta.get(f"{arch}.rope.scaling.type")
        longrope = scaling == "longrope" and arch == "minicpm"
        if scaling not in (None, "none", "llama3") and not (scaling == "linear" and arch == "gemma3") and not longrope:
            return False, f"rope scaling {scaling!r}"
        if longrope:                  # only the short factors are read: fine while the context stays within the original
            orig = meta.get(f"{arch}.rope.scaling.original_context_length")
            if "rope_factors_short.weight" not in tensor_names(path):
                return False, "longrope scaling without its short factors"
            if orig and orig < meta.get(f"{arch}.context_length", orig):
                return False, "longrope scaling with a longer context than the original"
        if meta.get(f"{arch}.attention.sliding_window") and arch not in ("gemma2", "gemma3"):
            return False, "sliding-window attention"
        if scaling == "llama3" and "rope_freqs.weight" not in tensor_names(path):
            return False, "llama3 rope scaling without its frequency table"
        bad = tensor_types(path) - NATIVE_TYPES
        if bad:
            return False, f"quantization type(s) {sorted(bad)}"
    except (OSError, GGUFError, struct.error, KeyError) as e:
        return False, str(e)
    return True, ""


GGML_TYPE_COUNT = 43    # upstream ggml types are [0, 43); anything at or above is a vendor extension


def vendor_types(path):
    """ggml tensor types outside upstream's range — a vendor extension (Prism's ternary PTQ1_0 is type 143).
    Neither this engine's own decoder nor stock llama.cpp can read them: llama.cpp refuses with
    "invalid ggml type N, should be in [0, 43)". Empty set when the file uses only standard types."""
    return {t for t in tensor_types(path) if t >= GGML_TYPE_COUNT}


_QUANT_RE = re.compile(r"(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z]+|Q\d_K(?:_[A-Z]+)?|Q\d_\d|MXFP4(?:_MOE)?|BF16|F16|F32)(?:_XL|_L)?)"
                       r"(?=[-_.]|$)", re.I)


def guess_quant(filename):
    """The quantization a file name says, as written: "Q4_K_M", "IQ4_XS", and unsloth's dynamic variants whole —
    "UD-Q2_K_XL", "UD-IQ4_NL_XL", "MXFP4_MOE" (they used to come out as "Q2_K" or "?")."""
    name = os.path.basename(filename)
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    part = _PART_RE.match(name + ".gguf")              # "…-BF16-00002-of-00002": the parts share one quantization
    if part:
        name = part.group(1)
    hits = _QUANT_RE.findall(name)
    return hits[-1].upper() if hits else "?"


def _quant_key(q):
    """How a quantization is matched to the user's spelling: case, '-' or '_' do not matter."""
    return q.upper().replace("-", "_")


def _quant_rank(q):
    """Preference of a quantization; a dynamic (UD-…, …_XL) variant ranks with the one it builds on."""
    base = re.sub(r"^UD-", "", q.upper())
    plain = re.sub(r"_XL$", "", base)
    # Q4_K_XL ranks as Q4_K_M; Q2_K_XL (no _M variant) as Q2_K; IQ4_NL_XL as IQ4_NL
    for cand in (base, plain + "_M" if base.endswith("_XL") else base, plain, re.sub(r"_[SML]$", "", plain)):
        if cand in QUANT_PREFERENCE:
            return QUANT_PREFERENCE.index(cand)
    return 99


# ───────────────────────────── finding files ─────────────────────────────
def is_projector(path):
    """vision projector files (mmproj-*.gguf) travel next to a model; they are not models themselves"""
    return "mmproj" in os.path.basename(path).lower()


def is_gguf_path(path):
    return os.path.isfile(path) and path.lower().endswith(".gguf")


def projector_for(model_path):
    d = os.path.dirname(model_path)
    try:
        cands = sorted(n for n in os.listdir(d) if n.lower().endswith(".gguf") and "mmproj" in n.lower())
    except OSError:
        return None
    return os.path.join(d, cands[0]) if cands else None


def ollama_models(root=None):
    """{name: gguf blob path} from Ollama's store: manifests name a model layer whose blob IS the GGUF."""
    root = root or os.path.expanduser("~/.ollama/models")
    out = {}
    man = os.path.join(root, "manifests")
    for dirpath, _, files in os.walk(man):
        for fn in files:
            p = os.path.join(dirpath, fn)
            try:
                with open(p) as f:
                    m = json.load(f)
                layer = next(l for l in m.get("layers", []) if l.get("mediaType", "").endswith("image.model"))
                blob = os.path.join(root, "blobs", layer["digest"].replace(":", "-"))
            except (OSError, ValueError, StopIteration, KeyError):
                continue
            if os.path.isfile(blob):
                rel = os.path.relpath(p, man).split(os.sep)           # registry.ollama.ai/library/llama3.2/latest
                out[f"{rel[-2]}:{rel[-1]}"] = blob
    return out


# ───────────────────────────── the llama.cpp server ─────────────────────────────
def server_path(data_dir=None):
    """llama-server: $LLM_LLAMA_SERVER, the copy installed into the data folder, or PATH (Homebrew)."""
    import shutil
    env = os.environ.get("LLM_LLAMA_SERVER")
    if env:
        return env if os.access(env, os.X_OK) else None
    if data_dir:
        p = os.path.join(data_dir, "llama.cpp", "llama-server")
        if os.access(p, os.X_OK):
            return p
    return shutil.which("llama-server")


# image projectors (mmproj) the built-in decoder runs itself: the Qwen3-VL tower of Qwen3.5 / 3.6 (engine/vision.py)
NATIVE_PROJECTORS = ("qwen3vl_merger",)
NATIVE_VISION_ARCHS = ("qwen35", "qwen35moe")


def projector_type(path):
    """The projector an mmproj GGUF declares ("qwen3vl_merger" …), or None (stdlib, header only)."""
    try:
        return read_header_full(path).get("clip.projector_type")
    except (OSError, GGUFError, struct.error, KeyError):
        return None


_PROJ_CACHE = {}


def native_vision(model_path):
    """True when the projector next to this GGUF is one the built-in decoder runs (images without llama.cpp).
    Cached per projector file (path, mtime, size): model lists ask for every model on every refresh."""
    proj = projector_for(model_path)
    if not proj:
        return False
    try:
        st = os.stat(proj)
    except OSError:
        return False
    key = (proj, st.st_mtime_ns, st.st_size)
    if key not in _PROJ_CACHE:
        _PROJ_CACHE[key] = projector_type(proj) in NATIVE_PROJECTORS
    return _PROJ_CACHE[key]


# draft models for speculative decoding: llama.cpp loads them, but only next to the model they draft for
DRAFT_ARCHS = frozenset({"dflash", "eagle", "eagle3", "medusa"})
_LIB_ARCHS = {}


def llamacpp_knows(arch, data_dir=None):
    """Whether llama.cpp runs GGUF architecture `arch`: True/False from the installed library's own names (each is
    stored there null-terminated), None when llama.cpp is not installed — it is fetched fresh on first use, so a
    name this engine has never heard of is not assumed to be missing from it."""
    import glob
    libs = sorted(glob.glob(os.path.join(data_dir, "llama.cpp", "libllama*.dylib")), key=len) if data_dir else []
    if not libs:
        return None
    try:
        st = os.stat(libs[0])
        key = (libs[0], st.st_mtime_ns)
        if key not in _LIB_ARCHS:
            with open(libs[0], "rb") as f:
                _LIB_ARCHS.clear()
                _LIB_ARCHS[key] = f.read(64 * 2**20)
        return (b"\0" + arch.encode() + b"\0") in _LIB_ARCHS[key]
    except OSError:
        return None


def hub_gguf_route(arch, tags=(), projector=False, data_dir=None):
    """(route, why) for a GGUF repo on the hub, from the architecture it declares: "native" when this engine's
    decoder reads it — with its image projector too when that is one the decoder runs (Qwen3.5 / 3.6); "gguf" when
    llama.cpp runs it (another projector goes to llama.cpp at once); "none" for a draft model or an architecture
    nothing here runs (the installed llama.cpp lacks it, or only InferSSD knows it)."""
    if not arch:
        return "gguf", ""                                  # the hub did not say: llama.cpp is the general route
    if arch in DRAFT_ARCHS or "draft-model" in set(tags or ()):
        return "none", "draft model for speculative decoding"
    if arch in NATIVE_ARCHS and (not projector or arch in NATIVE_VISION_ARCHS):
        return "native", ""
    known = llamacpp_knows(arch, data_dir)
    if known is False or (known is None and arch in INFERSSD_PROFILES):
        return "none", f"architecture {arch} is not supported"
    return "gguf", ""


# Profiles declared by InferSSD's typed launcher at the reviewed 0.1.4/e91c7d7 revision. Keep this exact: a
# family resemblance is not enough to send an untested checkpoint to a specialised SSD executor.
INFERSSD_PROFILES = {"deepseek41": "dsv", "muse-spark": "muse",
                     "qwen35moe": "qwen35moe", "qwen4exp": "qwen4exp"}
INFERSSD_ARCHS = frozenset(INFERSSD_PROFILES)


def inferssd_path(data_dir=None):
    """An already installed InferSSD command. AI-llm-Engine never downloads this optional third-party runtime."""
    import shutil
    explicit = os.environ.get("LLM_INFERSSD")
    if explicit:
        return explicit if os.access(explicit, os.X_OK) else None
    candidates = [os.path.expanduser("~/.local/bin/inferssd")]
    if data_dir:
        candidates[:0] = [os.path.join(data_dir, "inferssd", "bin", "inferssd")]
    return next((p for p in candidates if os.access(p, os.X_OK)), None) or shutil.which("inferssd")


def inferssd_support(model):
    """True only for a routed-MoE GGUF whose architecture has an explicit InferSSD launch profile."""
    info = read_header(model) if isinstance(model, (str, os.PathLike)) else model
    return bool(info and info.get("architecture") in INFERSSD_ARCHS and info.get("expert_count"))


@lru_cache(maxsize=8)
def _inferssd_capabilities(binary, mtime_ns, size):
    """Validated capabilities of one installed build; file identity makes an in-place upgrade re-probe it."""
    try:
        result = subprocess.run([binary, "--version", "--json"], capture_output=True, text=True, timeout=5)
        value = json.loads(result.stdout) if result.returncode == 0 else None
        version = tuple(int(x) for x in str(value.get("version", "")).split("."))
        profiles = (value.get("capabilities") or {}).get("model_profiles")
        if len(version) < 3 or not isinstance(profiles, list) or not all(isinstance(x, str) for x in profiles):
            return None
        return version, frozenset(profiles)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError):
        return None


def inferssd_runtime_support(binary, model):
    """The installed command implements the reviewed 0.1.4 contract and advertises this model's profile."""
    try:
        info = read_header(model) if isinstance(model, (str, os.PathLike)) else model
        if not inferssd_support(info):
            return False
        st = os.stat(binary)
        found = _inferssd_capabilities(binary, st.st_mtime_ns, st.st_size)
        return bool(found and found[0] >= (0, 1, 4) and INFERSSD_PROFILES[info["architecture"]] in found[1])
    except (OSError, KeyError, TypeError):
        return False


RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8"


def pick_asset(releases, arch="arm64"):
    """(tag, name, url) of the newest release that has a macOS tarball for this architecture."""
    for r in releases:
        for a in r.get("assets", []):
            n = a["name"].lower()
            if "macos" in n and arch in n and n.endswith(".tar.gz") and "kleidiai" not in n:
                return r["tag_name"], a["name"], a["browser_download_url"]
    return None


def install(data_dir, log=print, opener=urllib.request.urlopen):
    """Download llama.cpp's own macOS build into <data>/llama.cpp. Returns the llama-server path."""
    import platform
    arch = "arm64" if platform.machine() == "arm64" else "x64"
    with opener(urllib.request.Request(RELEASES, headers={"User-Agent": "AI-llm-Engine"}), timeout=30) as r:
        found = pick_asset(json.load(r), arch)
    if not found:
        raise GGUFError("no macOS build of llama.cpp found in the latest releases")
    tag, name, url = found
    dest = os.path.join(data_dir, "llama.cpp")
    log(f"llama.cpp {tag}: {name}")
    with tempfile.TemporaryDirectory() as tmp:
        tgz = os.path.join(tmp, name)
        with opener(urllib.request.Request(url, headers={"User-Agent": "AI-llm-Engine"}), timeout=120) as r, open(tgz, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        with tarfile.open(tgz) as t:
            for m in t.getmembers():                       # nothing may land outside the folder
                if m.name.startswith(("/", "..")) or ".." in m.name.split("/") or m.islnk() and m.linkname.startswith(("/", "..")):
                    raise GGUFError(f"unsafe path in the archive: {m.name}")
            staging = os.path.join(tmp, "x")
            t.extractall(staging)
        # the archive has one top folder (build/ or llama-bNNNN/): the binary and its dylibs live inside it
        found_bin = None
        for dp, _, fs in os.walk(staging):
            if "llama-server" in fs:
                found_bin = dp
                break
        if not found_bin:
            raise GGUFError("llama-server is not in the downloaded archive")
        import shutil
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(found_bin, dest, symlinks=True)
    path = os.path.join(dest, "llama-server")
    os.chmod(path, 0o755)
    with open(os.path.join(dest, "VERSION"), "w") as f:
        f.write(tag + "\n")
    return path


# ───────────────────────────── choosing files of a hub repo ─────────────────────────────
QUANT_PREFERENCE = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0", "Q6_K", "Q8_0", "IQ4_XS", "IQ4_NL", "Q4_1",
                    "Q5_0", "Q5_1", "Q3_K_M", "Q3_K_L", "Q3_K_S", "IQ3_M", "IQ3_S", "Q2_K", "BF16", "F16")
_PART_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)


def split_repo_quant(repo):
    """"org/name:Q4_K_M" -> ("org/name", "Q4_K_M"); no suffix -> (repo, None)."""
    prefix = "ms:" if repo.startswith("ms:") else ""
    body = repo[len(prefix):]
    if ":" in body and "/" in body.split(":")[0]:
        r, q = body.rsplit(":", 1)
        return prefix + r, q or None
    return repo, None


def pick_files(siblings, quant=None):
    """(files to download, quant chosen, quants on offer) for a GGUF repo listing [{rfilename, size}].
    One quantization (all of its split parts) plus the smallest of the preferred vision projectors."""
    models, projectors = {}, []
    for f in siblings:
        name = f["rfilename"]
        if not name.lower().endswith(".gguf"):
            continue
        if "mmproj" in name.lower():
            projectors.append(f)
            continue
        if "imatrix" in os.path.basename(name).lower():     # an importance matrix for quantizing, not a model
            continue
        m = _PART_RE.match(os.path.basename(name))
        group = (os.path.dirname(name), m.group(1) if m else os.path.basename(name)[:-5])
        models.setdefault(group, {"quant": guess_quant(os.path.basename(name)), "files": []})["files"].append(f)
    if not models:
        return [], None, []
    offered = sorted({g["quant"] for g in models.values()},
                     key=lambda q: (_quant_rank(q), q))
    chosen = None
    if quant:
        want = _quant_key(quant)
        chosen = next((q for q in offered if _quant_key(q) == want), None)
        if chosen is None:
            raise GGUFError(f"quantization {quant!r} is not in this repo (on offer: {', '.join(offered)})")
    else:
        chosen = offered[0]
    group = next(g for g in models.values() if g["quant"] == chosen)
    files = sorted(group["files"], key=lambda f: f["rfilename"])
    proj = None
    for pref in ("F16", "BF16", "F32"):
        hits = [p for p in projectors if pref in p["rfilename"].upper()]
        if hits:
            proj = min(hits, key=lambda p: p.get("size") or 0)
            break
    if proj is None and projectors:
        proj = min(projectors, key=lambda p: p.get("size") or 0)
    return files + ([proj] if proj else []), chosen, offered

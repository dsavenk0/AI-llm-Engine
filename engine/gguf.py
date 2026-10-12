"""GGUF files: read their header, find them on disk, and fetch the llama.cpp server that runs them.

GGUF is what most people download (Hugging Face, Ollama, LM Studio). The engine does not reimplement those
architectures: it hands the file to llama.cpp's own `llama-server`, installed on demand into the data folder
(`./llm setup --gguf`), and keeps the same OpenAI-compatible front. Stdlib only.
"""
import json
import os
import re
import struct
import tarfile
import tempfile
import urllib.request

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
            want = key in WANTED or (arch and key.startswith(arch + ".") and key.endswith((".context_length", ".block_count")))
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
# Q2_K Q3_K Q4_K Q5_K Q6_K IQ4_NL IQ4_XS
NATIVE_TYPES = {0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23, 30}
NATIVE_ARCHS = ("llama", "qwen2", "qwen3", "gemma3", "granite", "granitemoe", "qwen3moe", "minicpm")
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
            if "blk.0.ffn_gate_shexp.weight" in names:
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
        if meta.get(f"{arch}.attention.sliding_window") and arch != "gemma3":
            return False, "sliding-window attention"
        if scaling == "llama3" and "rope_freqs.weight" not in tensor_names(path):
            return False, "llama3 rope scaling without its frequency table"
        bad = tensor_types(path) - NATIVE_TYPES
        if bad:
            return False, f"quantization type(s) {sorted(bad)}"
    except (OSError, GGUFError, struct.error, KeyError) as e:
        return False, str(e)
    return True, ""


def guess_quant(filename):
    m = re.search(r"(IQ\d_[A-Z]+|Q\d_K(?:_[SML])?|Q\d_\d|BF16|F16|F32|Q8_0)", filename, re.I)
    return m.group(1).upper() if m else "?"


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
        m = _PART_RE.match(os.path.basename(name))
        group = (os.path.dirname(name), m.group(1) if m else os.path.basename(name)[:-5])
        models.setdefault(group, {"quant": guess_quant(os.path.basename(name)), "files": []})["files"].append(f)
    if not models:
        return [], None, []
    offered = sorted({g["quant"] for g in models.values()},
                     key=lambda q: QUANT_PREFERENCE.index(q) if q in QUANT_PREFERENCE else 99)
    chosen = None
    if quant:
        want = quant.upper().replace("-", "_")
        chosen = next((q for q in offered if q.upper() == want), None)
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

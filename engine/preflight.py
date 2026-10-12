"""Is this Mac ready to run the engine? A quick check made at launch (`llm check [--json]`).

Each item is {id, level: ok|warn|error, title, detail}. `error` means the engine cannot start as things
are; `warn` means something will be downloaded or is worth knowing. Nothing here needs MLX: it runs
before the environment exists. The network is only probed when something has to be downloaded.
"""
import os
import platform
import shutil
import socket
import subprocess
import urllib.error
import urllib.request

NEED_FREE_GB = 1.5            # Python (~0.1 GB) + MLX and the other wheels (~0.3 GB), with room to unpack
LOW_FREE_GB = 5               # a model is several GB: below this say so
IMPORTS = "import mlx.core, numpy, tokenizers, jinja2, PIL"
HOSTS = {"pypi": "https://pypi.org/simple/mlx/", "python": "https://api.github.com/repos/astral-sh/python-build-standalone"}


def _item(id, level, title, detail=""):
    return {"id": id, "level": level, "title": title, "detail": detail}


def reachable(url, env=None, timeout=6):
    """(True, "") when the URL answers at all (any HTTP status); (False, reason) otherwise."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "AI-llm-Engine"}),
                                    timeout=timeout):
            return True, ""
    except urllib.error.HTTPError:
        return True, ""                                # the server answered: the network works
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        return False, str(getattr(e, "reason", e))


def environment_ok(venv_py, timeout=30):
    if not os.path.exists(venv_py):
        return False
    try:
        return subprocess.run([venv_py, "-c", IMPORTS], capture_output=True, timeout=timeout).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def run(data_dir, venv_py, python_found, model_count=None, probe=reachable, env_ok=environment_ok,
        machine=None, mac_ver=None, free_gb=None, wheels_bundled=False):
    items = []
    machine = machine or platform.machine()
    if machine != "arm64":
        items.append(_item("apple-silicon", "error", "Apple Silicon is required",
                           "The engine runs on MLX, which needs an M-series Mac."))
    ver = mac_ver or platform.mac_ver()[0]
    try:
        if ver and int(ver.split(".")[0]) < 14:
            items.append(_item("macos", "error", f"macOS {ver} is too old", "MLX needs macOS 14 or newer."))
    except ValueError:
        pass

    env_good = env_ok(venv_py)
    need_download = []
    if not env_good:
        py = python_found()                            # only looked for when the environment has to be built
        if not wheels_bundled:                         # an offline build installs from files inside the app
            need_download.append(("pypi", "the engine libraries (~300 MB)"))
        if not py:
            need_download.append(("python", "Python (~35 MB)"))
        what = " and ".join(t for _, t in need_download)
        items.append(_item("environment", "warn",
                           "First launch: the engine environment is not installed yet" if not os.path.exists(venv_py)
                           else "The engine environment is damaged and will be reinstalled",
                           f"The first START downloads {what}." if need_download
                           else "The first START installs it from the files inside the app (no network needed)."))
    else:
        items.append(_item("environment", "ok", "Engine environment is ready"))

    for key, what in need_download:                    # offline is only a problem when there is something to fetch
        ok, why = probe(HOSTS[key])
        if not ok:
            items.append(_item("network", "error", "No internet connection",
                               f"Needed to download {what}: {why}. Connect, or set the proxy in Settings, then retry."))
            break

    free = free_gb if free_gb is not None else shutil.disk_usage(
        data_dir if os.path.isdir(data_dir) else os.path.dirname(data_dir.rstrip("/")) or "/").free / 2**30
    if need_download and free < NEED_FREE_GB:
        items.append(_item("disk", "error", "Not enough free disk space",
                           f"{free:.1f} GB free; the first launch needs about {NEED_FREE_GB} GB."))
    elif free < LOW_FREE_GB:
        items.append(_item("disk", "warn", "Little free disk space",
                           f"{free:.1f} GB free: a model takes several GB."))
    if model_count == 0:
        items.append(_item("models", "warn", "No models yet", "Pick one in the Models tab, or:  ./llm pull <org/name>"))
    return items


def summary(items):
    return {"ok": not any(i["level"] == "error" for i in items), "items": items}

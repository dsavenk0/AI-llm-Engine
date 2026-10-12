"""A private Python for the engine when the Mac has none that is new enough (3.10+).

macOS ships python3 3.9 (and only behind the Command Line Tools prompt), so a clean laptop usually
cannot create the engine's environment. This downloads a relocatable CPython build
(astral-sh/python-build-standalone, ~35 MB) into <data>/python, once, and checks its SHA-256 first.
Stdlib only — the manager imports it before any environment exists.
"""
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request

RELEASES = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
PREFERRED = ("3.12", "3.13", "3.11")          # the engine's wheels (mlx & co) exist for all of these
_NAME = re.compile(r"^cpython-(\d+\.\d+)\.\d+\+\d+-(aarch64|x86_64)-apple-darwin-install_only\.tar\.gz$")


class PythonInstallError(RuntimeError):
    pass


def python_path(data_dir):
    return os.path.join(data_dir, "python", "bin", "python3")


def pick_asset(release, arch="aarch64"):
    """(tag, name, url, sha256 or None, sums_url or None) of the best install_only build in a release."""
    best = None
    sums = next((a["browser_download_url"] for a in release.get("assets", []) if a.get("name") == "SHA256SUMS"), None)
    for a in release.get("assets", []):
        m = _NAME.match(a.get("name", ""))
        if not m or m.group(2) != arch or m.group(1) not in PREFERRED:
            continue
        rank = PREFERRED.index(m.group(1))
        if best is None or rank < best[0]:
            digest = a.get("digest") or ""
            best = (rank, a["name"], a["browser_download_url"],
                    digest.split(":", 1)[1] if digest.startswith("sha256:") else None)
    if not best:
        return None
    return release.get("tag_name", ""), best[1], best[2], best[3], sums


def _expected_sha(name, digest, sums_url, opener):
    if digest:
        return digest.lower()
    if not sums_url:
        raise PythonInstallError("the release publishes no checksum for the Python build; refusing to install it")
    with opener(urllib.request.Request(sums_url, headers={"User-Agent": "AI-llm-Engine"}), timeout=60) as r:
        for line in r.read().decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == name:
                return parts[0].lower()
    raise PythonInstallError(f"no checksum for {name} in SHA256SUMS")


def _safe_extract(tgz, dest):
    with tarfile.open(tgz) as t:
        for m in t.getmembers():                       # nothing may land outside the folder
            parts = m.name.split("/")
            if m.name.startswith("/") or ".." in parts:
                raise PythonInstallError(f"unsafe path in the archive: {m.name}")
            if (m.issym() or m.islnk()) and (m.linkname.startswith("/") or ".." in m.linkname.split("/")):
                raise PythonInstallError(f"unsafe link in the archive: {m.name} -> {m.linkname}")
            if not (m.isfile() or m.isdir() or m.issym() or m.islnk()):
                raise PythonInstallError(f"unexpected entry in the archive: {m.name}")
        try:
            t.extractall(dest, filter="data")
        except TypeError:                              # Python < 3.12 without extraction filters
            t.extractall(dest)


def install(data_dir, log=print, opener=urllib.request.urlopen, machine=None):
    """Download and unpack the Python build into <data>/python; returns the python3 path."""
    import platform
    machine = machine or platform.machine()
    if machine not in ("arm64", "aarch64"):
        raise PythonInstallError("the engine needs an Apple Silicon Mac")
    with opener(urllib.request.Request(RELEASES, headers={"User-Agent": "AI-llm-Engine"}), timeout=30) as r:
        found = pick_asset(json.load(r))
    if not found:
        raise PythonInstallError("no macOS Python build found in the latest python-build-standalone release")
    tag, name, url, digest, sums = found
    want = _expected_sha(name, digest, sums, opener)
    log(f"Python {name.split('-')[1].split('+')[0]} ({tag})")
    os.makedirs(data_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        tgz, h, done = os.path.join(tmp, name), hashlib.sha256(), 0
        with opener(urllib.request.Request(url, headers={"User-Agent": "AI-llm-Engine"}), timeout=120) as r, open(tgz, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0) if hasattr(r, "headers") else 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if total:
                    log(f"  {done >> 20} / {total >> 20} MB")
        if h.hexdigest() != want:
            raise PythonInstallError("the downloaded Python does not match its checksum")
        staging = os.path.join(tmp, "x")
        _safe_extract(tgz, staging)
        src = os.path.join(staging, "python")
        if not os.path.isfile(os.path.join(src, "bin", "python3")):
            raise PythonInstallError("python3 is not in the downloaded archive")
        dest = os.path.join(data_dir, "python")
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.move(src, dest)                         # same volume: one rename, nothing half-installed
    return python_path(data_dir)

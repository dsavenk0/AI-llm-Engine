#!/usr/bin/env python3
"""Puts everything the engine needs INSIDE LLMBar.app, so a clean Mac works with no network at first launch:

  LLMBar.app/Contents/Resources/python/   a relocatable CPython (python-build-standalone, SHA-256 checked)
  LLMBar.app/Contents/Resources/wheels/   the wheels of requirements.txt (+ their dependencies), for this Python

At run time llm.py prefers that Python and installs from those wheels with `pip --no-index`.
Used by `./build-app.sh --offline` (and `./build-dmg.sh --offline`); the plain build downloads both at first launch.
Downloads are cached in <repo>/.build-cache so repeated builds do not fetch them again.

    tools/bundle-deps.py LLMBar.app [--cache DIR]
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import pyruntime  # noqa: E402  (stdlib only)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    app, cache = argv[0], os.path.join(ROOT, ".build-cache")
    if "--cache" in argv:
        cache = argv[argv.index("--cache") + 1]
    res = os.path.join(app, "Contents", "Resources")
    if not os.path.isdir(res):
        print(f"not an app bundle: {app}", file=sys.stderr)
        return 1
    os.makedirs(cache, exist_ok=True)

    py = pyruntime.python_path(cache)
    if not os.path.exists(py):
        print("▸ Downloading Python…")
        py = pyruntime.install(cache, log=lambda s: print("  " + s))
    else:
        print("▸ Python (cached)")

    wheels = os.path.join(cache, "wheels")
    shutil.rmtree(wheels, ignore_errors=True)          # always resolved afresh: requirements may have changed
    print("▸ Downloading wheels…")
    subprocess.run([py, "-m", "pip", "download", "-q", "--only-binary=:all:", "-r", os.path.join(ROOT, "requirements.txt"),
                    "-d", wheels], check=True)
    if not any(n.endswith(".whl") for n in os.listdir(wheels)):
        print("no wheels were downloaded", file=sys.stderr)
        return 1

    print("▸ Copying into the app…")
    for name, src in (("python", os.path.join(cache, "python")), ("wheels", wheels)):
        dst = os.path.join(res, name)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True)
    for dp, dn, _ in os.walk(os.path.join(res, "python")):   # bytecode caches only add weight (and are rebuilt)
        for d in list(dn):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(dp, d), ignore_errors=True)
                dn.remove(d)
    size = sum(os.path.getsize(os.path.join(dp, f)) for n in ("python", "wheels")
               for dp, _, fs in os.walk(os.path.join(res, n)) for f in fs if os.path.isfile(os.path.join(dp, f)))
    print(f"✓ Bundled Python and {len(os.listdir(wheels))} wheels (+{size / 2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

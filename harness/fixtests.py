"""The "run tests → fix → repeat" loop's non-model parts: pick a test command, run it, and cut the
output down to what the model needs (the failing tests and their tracebacks, not the whole log).

The agent drives the loop (agent.py: /fix-tests) so every edit still goes through its normal
confirmations; this module only knows about tests.
"""
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys

MAX_EXCERPT = 6000
TIMEOUT = 900


def detect_command(root):
    """The project's test command, or None. `.llm/test_cmd` (one line) always wins."""
    p = os.path.join(root, ".llm", "test_cmd")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            line = f.read().strip()
        if line:
            return line
    has = lambda *n: any(os.path.exists(os.path.join(root, x)) for x in n)
    if has("package.json"):
        try:
            with open(os.path.join(root, "package.json"), encoding="utf-8") as f:
                if (json.load(f).get("scripts") or {}).get("test"):
                    return "npm test --silent"
        except (OSError, ValueError):
            pass
    if has("Cargo.toml"):
        return "cargo test"
    if has("go.mod"):
        return "go test ./..."
    if has("Package.swift"):
        return "swift test"
    if has("pytest.ini", "tox.ini", "conftest.py") or _mentions(root, "pyproject.toml", "pytest") \
            or _mentions(root, "setup.cfg", "pytest"):
        return f"{shlex.quote(sys.executable)} -m pytest -x -q"
    if os.path.isdir(os.path.join(root, "tests")) or os.path.isdir(os.path.join(root, "test")):
        top = "tests" if os.path.isdir(os.path.join(root, "tests")) else "test"
        pkg = os.path.exists(os.path.join(root, top, "__init__.py"))   # a package needs the project as top level
        return f"{shlex.quote(sys.executable)} -B -m unittest discover -s {top}" + (" -t ." if pkg else "")
    if _mentions(root, "Makefile", "\ntest:") or _mentions(root, "makefile", "\ntest:"):
        return "make test"
    return None


def _mentions(root, name, text):
    try:
        with open(os.path.join(root, name), encoding="utf-8", errors="replace") as f:
            return text in "\n" + f.read() if text.startswith("\n") else text in f.read()
    except OSError:
        return False


def run(command, root, timeout=TIMEOUT):
    """(exit_code, combined output). A timeout is a failure with an explanatory tail."""
    try:
        p = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "")) if e else ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return 124, out + f"\n[timed out after {timeout} s]"


def failures(output, limit=MAX_EXCERPT):
    """The failing tests with tracebacks plus the summary line. Understands unittest and pytest;
    anything else falls back to the tail of the log."""
    out = output.replace("\r", "")
    blocks = []
    # unittest: "======\nFAIL: test (mod.Class.test)\n------\n<traceback>"
    for m in re.finditer(r"^={20,}\n((?:FAIL|ERROR): .*?)(?=^={20,}\n|\Z)", out, re.S | re.M):
        blocks.append(m.group(1).strip())
    # pytest: "____ test_name ____" sections up to the next one or the short summary
    if not blocks:
        for m in re.finditer(r"^_{3,} (.+?) _{3,}\n(.*?)(?=^_{3,} .+? _{3,}\n|^={3,} .*? ={3,}\n|\Z)", out, re.S | re.M):
            blocks.append(f"FAILED: {m.group(1)}\n{m.group(2).strip()}")
    summary = ""
    for pat in (r"^FAILED \(.*\)$", r"^=+ .*(?:failed|error).* =+$", r"^Ran \d+ tests?.*$",
                r"^Tests?:.*$", r"^test result:.*$", r"^FAIL\b.*$"):
        ms = re.findall(pat, out, re.M | re.I)
        if ms:
            summary = ms[-1]
            break
    if not blocks:
        tail = "\n".join(out.strip().split("\n")[-60:])
        return tail[-limit:]
    text, used = [], 0
    for b in blocks:
        if used + len(b) > limit and text:
            text.append(f"… and {len(blocks) - len(text)} more failing test(s) not shown")
            break
        b = b if len(b) <= limit else b[:limit // 2] + "\n…\n" + b[-limit // 2:]
        text.append(b)
        used += len(b)
    return "\n\n".join(text) + (f"\n\n{summary}" if summary else "")


def signature(excerpt):
    """Stable fingerprint of a failure set (paths, line numbers and addresses normalised away),
    used to notice that a round of edits changed nothing."""
    norm = re.sub(r"0x[0-9a-f]+|\d+(\.\d+)?s\b|line \d+|:\d+", "N", excerpt)
    return hashlib.sha1(norm.encode()).hexdigest()[:12]


def prompt(command, excerpt, round_no, previous=None):
    lead = (f"Tests are failing (`{command}`). Fix the code so they pass — change the implementation "
            f"unless a test is clearly wrong. Make the smallest correct edit with edit_file; do not "
            f"run the tests yourself, they will be re-run automatically.")
    if previous:
        lead += " Your previous attempt did not fix it; read the code again before editing."
    return f"{lead}\n\nRound {round_no}. Failing output:\n\n{excerpt}"

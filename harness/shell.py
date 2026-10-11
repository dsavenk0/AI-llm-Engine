"""Run a command with a deadline and leave nothing behind (stdlib only).

`subprocess.run(..., timeout=)` kills only the process it started: a shell's children (a dev server,
`sleep`, a test runner's workers) keep running after the timeout and hold their ports. The command
gets its own session here, and the whole process group is ended on timeout.
"""
import os
import signal
import subprocess
from types import SimpleNamespace


def _kill_group(proc, grace=2.0):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(grace)
            break
        except subprocess.TimeoutExpired:
            continue


MAX_OUTPUT = 4_000_000          # bytes kept per stream: `yes` or `cat` of a huge file must not fill the Mac's memory


def _drain(stream, parts, over):
    """Read a pipe to its end, keeping at most MAX_OUTPUT bytes; flags `over` and stops when it is exceeded."""
    total = 0
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            room = MAX_OUTPUT - total
            if room > 0:
                parts.append(chunk[:room])
            total += len(chunk)
            if total > MAX_OUTPUT:
                over.set()
                return
    except (OSError, ValueError):
        return


def run(command, cwd=None, timeout=120, shell=True, env=None, stdin=subprocess.DEVNULL):
    """-> SimpleNamespace(returncode, stdout, stderr, timed_out, truncated). On timeout the output produced so
    far is returned and returncode is 124 (like timeout(1)); the command's process group is gone. A command
    that writes more than MAX_OUTPUT bytes to a stream is stopped too (truncated=True, returncode 125)."""
    import threading
    import time
    proc = subprocess.Popen(command, shell=shell, cwd=cwd, env=env, stdin=stdin, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True)
    over = threading.Event()
    out_parts, err_parts = [], []
    readers = [threading.Thread(target=_drain, args=(proc.stdout, out_parts, over), daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, err_parts, over), daemon=True)]
    for t in readers:
        t.start()
    deadline = time.time() + timeout
    timed_out = False
    while proc.poll() is None:
        if over.is_set():
            break
        if time.time() >= deadline:
            timed_out = True
            break
        time.sleep(0.02)
    truncated = over.is_set() and not timed_out
    if timed_out or truncated:
        _kill_group(proc)
    else:
        proc.wait()
    for t in readers:
        t.join(timeout=2)
    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except OSError:
            pass
    decode = lambda parts: b"".join(parts).decode("utf-8", "replace")
    code = 124 if timed_out else (125 if truncated else proc.returncode)
    return SimpleNamespace(returncode=code, stdout=decode(out_parts), stderr=decode(err_parts),
                           timed_out=timed_out, truncated=truncated)

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


def run(command, cwd=None, timeout=120, shell=True, env=None, stdin=subprocess.DEVNULL):
    """-> SimpleNamespace(returncode, stdout, stderr, timed_out). On timeout the output produced so far is
    returned and returncode is 124 (like timeout(1)); the command's process group is gone."""
    proc = subprocess.Popen(command, shell=shell, cwd=cwd, env=env, stdin=stdin, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return SimpleNamespace(returncode=proc.returncode, stdout=out or "", stderr=err or "", timed_out=False)
    except subprocess.TimeoutExpired as e:
        partial_out, partial_err = e.stdout, e.stderr
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=2)       # whatever was written before the kill
        except subprocess.TimeoutExpired:
            out, err = partial_out, partial_err
        def text(x):
            return x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")
        return SimpleNamespace(returncode=124, stdout=text(out), stderr=text(err), timed_out=True)

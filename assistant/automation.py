"""Private client for LLMBar Automation.app.

The browser extension deliberately cannot reach this process.  Only the authenticated assistant service talks to
the owner-only Unix socket, and every request is a fixed click/type/key operation bound to an issued ticket.
"""
import atexit
import json
import os
import secrets
import socket
import subprocess
import time


class AutomationError(RuntimeError):
    pass


class Automation:
    def __init__(self, root, home):
        self.root, self.home = root, home
        self.proc = None
        self.secret = secrets.token_urlsafe(32)
        self.socket_path = os.path.join(home, f"automation-{os.getpid()}.sock")
        self._trust = (0.0, False)
        atexit.register(self.close)

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired: self.proc.kill()
        try: os.unlink(self.socket_path)
        except OSError: pass

    def binary(self):
        override = os.environ.get("LLMBAR_AUTOMATION")
        paths = [override] if override else []
        paths += [
            os.path.join(self.root, "automation", "LLMBar Automation.app", "Contents", "MacOS", "LLMBarAutomation"),
            os.path.join(self.root, "LLMBar.app", "Contents", "Resources", "llm", "automation", "LLMBar Automation.app", "Contents", "MacOS", "LLMBarAutomation"),
            "/Applications/LLMBar.app/Contents/Resources/llm/automation/LLMBar Automation.app/Contents/MacOS/LLMBarAutomation",
            os.path.join(os.path.dirname(self.root), "LLMBar Automation.app", "Contents", "MacOS", "LLMBarAutomation"),
        ]
        return next((p for p in paths if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)

    def status(self):
        binary = self.binary()
        checked, trusted = self._trust
        if binary and time.time() - checked > 5:
            try: trusted = subprocess.run([binary, "--trusted"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          timeout=2).returncode == 0
            except (OSError, subprocess.SubprocessError): trusted = False
            self._trust = (time.time(), trusted)
        return {"available": bool(binary), "trusted": bool(trusted), "running": bool(self.proc and self.proc.poll() is None),
                "socket": bool(os.path.exists(self.socket_path))}

    def open_settings(self):
        subprocess.Popen(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         close_fds=True)
        return {"ok": True}

    def _start(self):
        if self.proc and self.proc.poll() is None and os.path.exists(self.socket_path):
            return
        binary = self.binary()
        if not binary:
            raise AutomationError("LLMBar Automation is not installed; rebuild LLMBar to enable real browser input")
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
        os.makedirs(self.home, mode=0o700, exist_ok=True)
        self.proc = subprocess.Popen([binary, "--serve", self.socket_path, self.secret], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        for _ in range(30):
            if os.path.exists(self.socket_path): return
            if self.proc.poll() is not None: break
            time.sleep(0.05)
        raise AutomationError("LLMBar Automation did not start")

    def request(self, body):
        self._start()
        packet = (json.dumps({**body, "secret": self.secret}, separators=(",", ":")) + "\n").encode()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect(self.socket_path)
                s.sendall(packet)
                chunks, size = [], 0
                while True:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > 32 * 1024:
                        raise AutomationError("LLMBar Automation returned an oversized response")
                    if b"\n" in chunk:
                        break
                data = b"".join(chunks).split(b"\n", 1)[0]
        except OSError as e:
            raise AutomationError(f"could not reach LLMBar Automation: {e}") from None
        try:
            out = json.loads(data)
        except ValueError:
            raise AutomationError("LLMBar Automation returned an invalid response") from None
        if not isinstance(out, dict): raise AutomationError("LLMBar Automation returned an invalid response")
        return out

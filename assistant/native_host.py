"""The Chromium browsers' half of pairing: a native-messaging host that hands the extension the service's token, so
nobody pastes it. (Safari gets the same answer from the handler inside LLMBar.app.)

Chrome starts this program when the extension calls `chrome.runtime.sendNativeMessage("local.llm_engine.assistant",
{"kind": "token"})`, but only for the extension ID named in the host manifest (`allowed_origins`), which
`./llm assistant setup` writes into each installed Chromium browser's NativeMessagingHosts folder. Messages are
JSON preceded by their length (4 bytes, native byte order). It answers "token" and nothing else, then exits.
"""
import json
import os
import struct
import sys

NAME = "local.llm_engine.assistant"
EXTENSION_ID = "acoaldoejdbigghamllanabnpamijfnh"     # fixed by the "key" in browser-assistant/manifests/chromium.json
MAX_MESSAGE = 64 * 1024


def token_file():
    home = os.environ.get("ASSISTANT_HOME") or os.path.expanduser("~/Library/Application Support/AI-llm-Engine")
    return os.path.join(home, "browser-assistant.json")


def answer(message, path=None):
    """The reply to one message from the extension."""
    if not isinstance(message, dict) or message.get("kind") != "token":
        return {"ok": False, "error": "unknown request"}
    try:
        with open(path or token_file()) as f:
            tok = json.load(f).get("token")
    except (OSError, ValueError):
        tok = None
    if isinstance(tok, str) and len(tok) >= 24:
        return {"ok": True, "token": tok}
    return {"ok": False, "error": "the assistant service has not been set up yet"}


def read_message(stream):
    head = stream.read(4)
    if len(head) < 4:
        return None
    (n,) = struct.unpack("=I", head)
    if n > MAX_MESSAGE:
        return None
    return json.loads(stream.read(n) or b"null")


def write_message(stream, obj):
    data = json.dumps(obj).encode()
    stream.write(struct.pack("=I", len(data)) + data)
    stream.flush()


def manifest(executable):
    """The host manifest a Chromium browser reads to find this program, and which extension may start it."""
    return {"name": NAME, "description": "LLMBar browser assistant: pairing", "path": executable, "type": "stdio",
            "allowed_origins": [f"chrome-extension://{EXTENSION_ID}/"]}


# where each Chromium browser looks for host manifests (only browsers whose profile folder exists are written)
BROWSER_DIRS = ("Google/Chrome", "Google/Chrome Beta", "Chromium", "Microsoft Edge", "BraveSoftware/Brave-Browser",
                "Vivaldi")


def install(home, python=None, support=None):
    """Write the launcher and the host manifest for every installed Chromium browser; returns their names."""
    support = support or os.path.expanduser("~/Library/Application Support")
    os.makedirs(home, exist_ok=True)
    launcher = os.path.join(home, "native-host")
    with open(launcher, "w") as f:
        f.write("#!/bin/sh\n"
                f"ASSISTANT_HOME={json.dumps(os.path.dirname(home.rstrip('/')))} "
                f"exec {json.dumps(python or sys.executable)} {json.dumps(os.path.abspath(__file__))}\n")
    os.chmod(launcher, 0o755)
    done = []
    for browser in BROWSER_DIRS:
        base = os.path.join(support, browser)
        if not os.path.isdir(base):
            continue
        hosts = os.path.join(base, "NativeMessagingHosts")
        os.makedirs(hosts, exist_ok=True)
        with open(os.path.join(hosts, NAME + ".json"), "w") as f:
            json.dump(manifest(launcher), f, indent=2)
        done.append(browser.split("/")[-1])
    return done


def main():
    msg = read_message(sys.stdin.buffer)
    write_message(sys.stdout.buffer, answer(msg))


if __name__ == "__main__":
    main()

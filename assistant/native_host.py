"""The Chromium and Firefox half of pairing: a native-messaging host that hands the extension the service's token.
Safari gets the same answer from the handler inside LLMBar.app.

Chrome starts this program when the extension calls `chrome.runtime.sendNativeMessage("local.llm_engine.assistant",
{"kind": "token"})`, but only for the extension ID named in the host manifest (`allowed_origins` for Chromium,
`allowed_extensions` for Firefox), which `./llm assistant setup` writes into each installed browser's
NativeMessagingHosts folder. Messages are JSON preceded by their length (4 bytes, native byte order). It answers
"token" and nothing else, then exits.
"""
import json
import os
import struct
import sys

NAME = "local.llm_engine.assistant"
EXTENSION_ID = "acoaldoejdbigghamllanabnpamijfnh"     # fixed by the "key" in browser-assistant/manifests/chromium.json
FIREFOX_EXTENSION_ID = "llmbar-assistant@local.llm-engine"  # fixed by browser_specific_settings.gecko.id
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


def manifest(executable, browser="chromium"):
    """A browser's host manifest: where to find this app, and which one extension may start it."""
    allowed = ({"allowed_extensions": [FIREFOX_EXTENSION_ID]} if browser == "firefox"
               else {"allowed_origins": [f"chrome-extension://{EXTENSION_ID}/"]})
    return {"name": NAME, "description": "LLMBar browser assistant: pairing", "path": executable,
            "type": "stdio", **allowed}


# where each Chromium browser looks for host manifests (only browsers whose profile folder exists are written)
BROWSER_DIRS = ("Google/Chrome", "Google/Chrome Beta", "Chromium", "Microsoft Edge", "BraveSoftware/Brave-Browser",
                "Vivaldi")


def install(home, python=None, support=None, firefox=False):
    """Write the launcher and host manifest for each installed browser; returns their names."""
    default_support = support is None
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
    # Firefox's host manifests live in Mozilla's application-support directory, separately from its profile.
    firefox_app = default_support and any(os.path.exists(p) for p in (
        "/Applications/Firefox.app", os.path.expanduser("~/Applications/Firefox.app")))
    if firefox or firefox_app or os.path.isdir(os.path.join(support, "Firefox", "Profiles")):
        hosts = os.path.join(support, "Mozilla", "NativeMessagingHosts")
        os.makedirs(hosts, exist_ok=True)
        with open(os.path.join(hosts, NAME + ".json"), "w") as f:
            json.dump(manifest(launcher, "firefox"), f, indent=2)
        done.append("Firefox")
    return done


def main():
    msg = read_message(sys.stdin.buffer)
    write_message(sys.stdout.buffer, answer(msg))


if __name__ == "__main__":
    main()

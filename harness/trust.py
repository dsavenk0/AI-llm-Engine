"""Workspace configs that start programs or send secrets need the person's yes once.

`.llm/mcp.json` and `.llm/workers.json` can sit in a project folder — including one just cloned from
somewhere. They name commands to run and environment variables to put into requests, so an untrusted
copy must never act on its own. A config is trusted per (path, content hash): it is asked about the
first time and again whenever the file changes. Configs in the user's home folder are their own and are
not asked about.
"""
import hashlib
import json
import os

STORE = os.path.expanduser("~/.llm/trusted.json")


def _hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def is_workspace_config(path, root):
    real, base = os.path.realpath(path), os.path.realpath(root)
    return real.startswith(base + os.sep)


def _load():
    try:
        with open(STORE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def is_trusted(path, root):
    """True for the user's own configs and for workspace configs that were approved in this exact form."""
    if not os.path.isfile(path) or not is_workspace_config(path, root):
        return True
    return _load().get(os.path.realpath(path)) == _hash(path)


def remember(path):
    data = _load()
    data[os.path.realpath(path)] = _hash(path)
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    tmp = STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, STORE)

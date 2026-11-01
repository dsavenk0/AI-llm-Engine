"""Constrained decoding: make the model's output valid JSON, and make it match a schema.

The cheap way to "support" `response_format` is to ask the model nicely in the prompt and hope. This does
the real thing instead: a character-level pushdown automaton that knows which characters may come next, and
a sampler that will not pick a token which would break it. A reply that comes back is parseable, or the
request fails — it never returns prose to a client that asked for JSON.

Checking every one of a vocabulary's ~250k tokens against the automaton on every step would cost more than
the forward pass. It is not needed: the sampler already works with a short candidate list (the top-k it was
going to choose from, or the single argmax), so only those are tested, in the order the model prefers them.
In practice the model's first choice is already valid and one test is paid per token.

What `schema` understands, which is the part of JSON Schema that `json_schema` responses actually use:
`type` (object, array, string, number, integer, boolean, null), `properties`, `required`,
`additionalProperties`, `items`, and `enum`. Anything else in the schema is ignored rather than
half-enforced, and `supported()` says so before a request starts.
"""
import json

WS = " \t\n\r"
DIGITS = "0123456789"


def _number_prefix(s):
    r"""True when `s` could still grow into a JSON number.

    Python's float() is not the right judge here: it happily reads "01", "1_0" and "nan", none of which
    JSON allows. The grammar is -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)? and this walks it.
    """
    i, n = 0, len(s)
    if i < n and s[i] == "-":
        i += 1
    if i == n:
        return True
    if s[i] == "0":
        i += 1
    elif s[i] in DIGITS:
        while i < n and s[i] in DIGITS:
            i += 1
    else:
        return False
    if i == n:
        return True
    if s[i] == ".":
        i += 1
        if i == n:
            return True
        if s[i] not in DIGITS:
            return False
        while i < n and s[i] in DIGITS:
            i += 1
        if i == n:
            return True
    if s[i] in "eE":
        i += 1
        if i == n:
            return True
        if s[i] in "+-":
            i += 1
            if i == n:
                return True
        if s[i] not in DIGITS:
            return False
        while i < n and s[i] in DIGITS:
            i += 1
        return i == n
    return False


def _number_whole(s):
    """True when `s` is a finished JSON number, not just a promising start ("1." and "1e" are not)."""
    return bool(s) and _number_prefix(s) and s[-1] in DIGITS


class Invalid(Exception):
    """The text cannot be continued into a valid document."""


class _Frame:
    """One open container (or the root), and what the grammar still expects inside it."""
    __slots__ = ("kind", "schema", "want", "seen", "key", "buf", "lit", "esc", "unicode", "in_key")

    def __init__(self, kind, schema=None, want="value"):
        self.kind = kind          # "root" | "object" | "array"
        self.schema = schema      # the schema node this container must satisfy, or None
        self.want = want          # value | key | colon | comma-or-close | string | number | literal | done
        self.seen = set()         # object: keys already written
        self.key = ""             # object: the key being read, then the key just finished
        self.buf = ""             # string/number being read
        self.lit = ""             # the remainder of true/false/null still to come
        self.esc = False          # inside a string escape
        self.unicode = 0          # how many hex digits of a \\uXXXX escape are still expected
        self.in_key = False       # the string being read is an object key, not a value

    def clone(self):
        f = _Frame.__new__(_Frame)
        for s in _Frame.__slots__:
            setattr(f, s, getattr(self, s))
        f.seen = set(self.seen)
        return f


def supported(schema):
    """None, or a sentence naming the part of this schema that would not be enforced.

    Said before generation starts, so a client is never told a reply is schema-checked when half of its
    schema was quietly dropped.
    """
    known = {"type", "properties", "required", "additionalProperties", "items", "enum",
             "title", "description", "$schema", "default", "examples"}
    def walk(node, path):
        if not isinstance(node, dict):
            return f"{path or 'schema'} must be an object"
        extra = sorted(set(node) - known)
        if extra:
            return f"{path or 'schema'}: {', '.join(extra)} is not enforced by this engine"
        t = node.get("type")
        if t is not None and (not isinstance(t, str) or
                              t not in ("object", "array", "string", "number", "integer", "boolean", "null")):
            return f"{path or 'schema'}: type {t!r} is not supported"
        for k, v in (node.get("properties") or {}).items():
            bad = walk(v, f"{path}.{k}" if path else k)
            if bad:
                return bad
        if "items" in node:
            return walk(node["items"], f"{path}[]" if path else "items")
        return None
    return walk(schema, "")


class Cursor:
    """Where the grammar is, after the text written so far. Cheap to clone, so candidates can be tried."""

    def __init__(self, schema=None, root="any"):
        """`root` is "object" for response_format json_object, "any" when a schema decides the shape."""
        if schema is not None:
            root = schema.get("type", "any")
        self.stack = [_Frame("root", schema, "value")]
        self.root = root
        self.done = False

    def clone(self):
        c = Cursor.__new__(Cursor)
        c.stack = [f.clone() for f in self.stack]
        c.root = self.root
        c.done = self.done
        return c

    # ── the automaton ────────────────────────────────────────────────────────
    def feed(self, text):
        for ch in text:
            self._char(ch)

    def accepts(self, text):
        """True when `text` can follow what has been written without breaking the grammar."""
        probe = self.clone()
        try:
            probe.feed(text)
        except Invalid:
            return False
        return True

    def complete(self):
        """True when what has been written is already a whole document, so stopping here is legal.

        A number at the root is the one value with no closing character: "42" is finished, but the automaton
        only learns that from whatever follows it, which at the end of a reply is nothing.
        """
        if self.done:
            return True
        root = self.stack[0]
        if len(self.stack) == 1 and root.want == "number":
            return _number_whole(root.buf)
        return False

    def _char(self, ch):
        if self.done:
            if ch in WS:
                return
            raise Invalid("trailing text after the document")
        f = self.stack[-1]
        getattr(self, "_in_" + f.want.replace("-", "_"))(f, ch)

    # ── scalars ──────────────────────────────────────────────────────────────
    def _in_string(self, f, ch):
        if f.unicode:
            if ch not in "0123456789abcdefABCDEF":
                raise Invalid("bad \\u escape")
            f.unicode -= 1
            f.buf += ch
            return
        if f.esc:
            if ch == "u":
                f.unicode = 4
            elif ch not in '"\\/bfnrt':
                raise Invalid("bad escape")
            f.esc = False
            f.buf += ch
            return
        if ch == "\\":
            if f.in_key and self._keys(f) is not None:
                # the schema's keys are literal names; an escape could only spell one the long way round
                # and would defeat the prefix check below, so it is refused outright
                raise Invalid("an escape cannot appear in a schema key")
            f.esc = True
            f.buf += ch
            return
        if ch == '"':
            self._finish_string(f)
            return
        if ch < " ":
            raise Invalid("raw control character in a string")   # JSON forbids every char below 0x20
        f.buf += ch
        allowed = self._keys(f) if f.in_key else self._enum_strings(f)
        if allowed is not None and not any(k.startswith(f.buf) for k in allowed):
            raise Invalid(f"nothing the schema allows here starts with {f.buf!r}")

    def _finish_string(self, f):
        if f.in_key:
            key = f.buf
            if key in f.seen:
                raise Invalid("duplicate key")
            allowed = self._keys(f)
            if allowed is not None and key not in allowed:
                raise Invalid(f"key {key!r} is not in the schema")
            f.key, f.buf, f.want, f.in_key = key, "", "colon", False
            return
        value = f.buf
        f.buf = ""
        node = self._value_schema(f)
        if node and node.get("enum") is not None and value not in node["enum"]:
            raise Invalid("value is not one of the enum")
        self._value_done(f)

    def _in_number(self, f, ch):
        nxt = f.buf + ch
        if self._number_prefix(nxt, f):
            f.buf = nxt
            return
        self._number_end(f)
        self._char(ch)                               # the character belongs to whatever follows

    def _number_prefix(self, s, f):
        node = self._value_schema(f)
        if node and node.get("type") == "integer" and ("." in s or "e" in s or "E" in s):
            return False
        return _number_prefix(s)

    def _number_end(self, f):
        if not _number_whole(f.buf):
            raise Invalid("incomplete number")
        f.buf = ""
        self._value_done(f)

    def _in_literal(self, f, ch):
        if not f.lit.startswith(ch):
            raise Invalid("bad literal")
        f.lit = f.lit[1:]
        if not f.lit:
            self._value_done(f)

    # ── structure ────────────────────────────────────────────────────────────
    def _in_value(self, f, ch):
        if ch in WS:
            return
        if self.root != "any" and len(self.stack) == 1 and ch not in "{[":
            raise Invalid(f"the reply must be a JSON {self.root}")   # a scalar never reaches _open()
        node = self._value_schema(f)
        t = node.get("type") if node else None
        enum = node.get("enum") if node else None
        if ch == "{":
            self._open("object", node, t, enum)
        elif ch == "[":
            self._open("array", node, t, enum)
        elif ch == '"':
            self._want(f, t, ("string",), ch)
            f.want, f.buf, f.esc, f.unicode = "string", "", False, 0
        elif ch == "-" or ch in DIGITS:
            self._want(f, t, ("number", "integer"), ch)
            f.want, f.buf = "number", ch
        elif ch in "tfn":
            word = {"t": "true", "f": "false", "n": "null"}[ch]
            self._want(f, t, ("boolean",) if ch != "n" else ("null",), ch)
            if enum is not None and json.loads(word) not in enum:
                raise Invalid("value is not one of the enum")
            f.want, f.lit = "literal", word[1:]
        else:
            raise Invalid(f"{ch!r} cannot start a value")

    def _want(self, f, t, ok, ch):
        if t is not None and t not in ok:
            raise Invalid(f"{ch!r} starts a {ok[0]}, but the schema wants {t}")

    def _open(self, kind, node, t, enum):
        if t is not None and t != kind:
            raise Invalid(f"the schema wants {t} here, not {kind}")
        if self.root != "any" and len(self.stack) == 1 and kind != self.root:
            raise Invalid(f"the reply must be a JSON {self.root}")
        if enum is not None:
            raise Invalid("an enum value cannot be a container")
        self.stack[-1].want = "comma-or-close"       # the parent resumes here once this closes
        self.stack.append(_Frame(kind, node, "first-key" if kind == "object" else "first-value"))

    def _in_first_key(self, f, ch):
        if ch in WS:
            return
        if ch == "}":
            self._close(f)
            return
        self._in_key(f, ch)

    def _in_key(self, f, ch):
        if ch in WS:
            return
        if ch != '"':
            raise Invalid("an object key must be a string")      # no trailing comma, no bare key
        f.want, f.buf, f.key, f.esc, f.unicode, f.in_key = "string", "", "", False, 0, True

    def _in_colon(self, f, ch):
        if ch in WS:
            return
        if ch != ":":
            raise Invalid("expected :")
        f.want = "value"

    def _in_first_value(self, f, ch):
        if ch in WS:
            return
        if ch == "]":
            self._close(f)
            return
        f.want = "value"
        self._char(ch)

    def _in_comma_or_close(self, f, ch):
        if ch in WS:
            return
        if ch == ",":
            if f.kind == "object" and self._keys(f) == set():
                raise Invalid("the schema has no further keys for this object")   # it must close instead
            f.want = "key" if f.kind == "object" else "value"
            f.key = ""
            return
        if (ch == "}" and f.kind == "object") or (ch == "]" and f.kind == "array"):
            self._close(f)
            return
        raise Invalid(f"expected , or the end of the {f.kind}")

    def _in_done(self, f, ch):
        if ch in WS:
            return
        raise Invalid("trailing text after the document")

    def _close(self, f):
        if f.kind == "object":
            missing = set((f.schema or {}).get("required") or []) - f.seen
            if missing:
                raise Invalid(f"missing required: {', '.join(sorted(missing))}")
        if f.buf:
            raise Invalid("unterminated value")
        self.stack.pop()
        parent = self.stack[-1]
        self._value_done(parent, closed=True)

    def _value_done(self, f, closed=False):
        """A value finished inside frame `f` (the container holding it, or the root)."""
        if f.kind == "object":
            f.seen.add(f.key)
            f.key = ""
            f.want = "comma-or-close"
        elif f.kind == "array":
            f.want = "comma-or-close"
        else:
            f.want = "done"
            self.done = True

    # ── schema lookup ────────────────────────────────────────────────────────
    def _enum_strings(self, f):
        """The string values this position may still take, or None when it is not a string enum.

        Checked while the value is being typed, not when the quote closes: by then the model has committed
        to a word the schema forbids and has no legal way out, which is how it ends up looping.
        """
        node = self._value_schema(f)
        if not node or node.get("enum") is None:
            return None
        vals = [v for v in node["enum"] if isinstance(v, str)]
        return vals if vals else None

    def _keys(self, f):
        """The key names this object may still use, or None when anything goes."""
        node = f.schema
        if not node or not isinstance(node.get("properties"), dict):
            return None
        if node.get("additionalProperties", False):
            return None
        return set(node["properties"]) - f.seen

    def _value_schema(self, f):
        """The schema node the value now being written must satisfy."""
        node = f.schema
        if not node:
            return None
        if f.kind == "object":
            return (node.get("properties") or {}).get(f.key)
        if f.kind == "array":
            return node.get("items")
        return node


def finished(text, schema=None, root="any"):
    """True when `text` is a complete document the schema accepts. Used to check a finished reply."""
    c = Cursor(schema, root)
    try:
        c.feed(text)
    except Invalid:
        return False
    return c.complete()

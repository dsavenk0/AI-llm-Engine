"""The data plane's framing: a large payload as checksummed chunks, sent a bounded window at a time
(docs/adar/NETWORKING.md, "Data plane").

Four properties, each of which a test holds it to:

* **bounded memory** — the sender reads its source one chunk at a time and never holds more than `window`
  unacknowledged chunks; a slow receiver slows the reading, it does not grow a buffer;
* **a corrupt chunk is never accepted** — each frame carries the SHA-256 of its header fields and payload,
  and the receiver asks again for one that does not match;
* **the whole is checked too** — a final digest over the stream, so a chunk that is valid alone but in the
  wrong place, or a truncated transfer, fails at the end instead of passing;
* **resumable** — the receiver says how many chunks it has durably; a new sender starts from there.

The receiver takes chunks in order only (go-back-N): that keeps its own memory at one chunk, at the cost
of resending what arrives ahead of a gap — the right trade on a LAN, where loss is rare.

Frames are binary with a fixed header; nothing here goes through the control plane's JSON or its 1 MB
limit, and a transfer never shares a queue with a heartbeat or a cancellation.
"""
import hashlib
import struct

MAGIC = b"ADRD"
_HEAD = struct.Struct(">4sB16sII32s")      # magic, kind, transfer id, chunk index, payload length, checksum
DATA, ACK, NACK, FIN = 1, 2, 3, 4
CHUNK = 64 * 1024


class TransferError(ValueError):
    pass


def _sum(kind, tid, index, payload):
    return hashlib.sha256(struct.pack(">B16sI", kind, tid, index) + payload).digest()


def frame(kind, tid, index, payload=b""):
    """The checksum covers the header fields as well as the payload: a chunk whose *index* was damaged is
    as wrong as one whose bytes were, and would otherwise be written in the wrong place."""
    return _HEAD.pack(MAGIC, kind, tid, index, len(payload), _sum(kind, tid, index, payload)) + payload


def parse(raw):
    if len(raw) < _HEAD.size or raw[:4] != MAGIC:
        raise TransferError("not a data-plane frame")
    _m, kind, tid, index, length, digest = _HEAD.unpack_from(raw)
    payload = raw[_HEAD.size:]
    if len(payload) != length:
        raise TransferError("frame length does not match its header")
    if digest != _sum(kind, tid, index, payload):
        raise TransferError("frame checksum does not match")
    return kind, tid, index, payload


def is_frame(raw):
    return raw[:4] == MAGIC


class Sender:
    """`read(offset, n) -> bytes` supplies the payload (a file, a generator behind a cursor — never the whole
    thing in memory). `start` is the chunk to begin at when resuming."""

    def __init__(self, tid, read, total, *, chunk=CHUNK, window=8, start=0, prefix_digest=None):
        self.tid, self.read, self.total, self.chunk, self.window = tid, read, total, chunk, window
        self.count = (total + chunk - 1) // chunk
        self.base = start                 # first chunk not yet acknowledged
        self.next = start                 # next chunk to read
        self.held = {}                    # index -> payload, at most `window` of them
        self.peak_held = 0                # bytes, for the bounded-memory test
        self.digest = prefix_digest or hashlib.sha256()
        self.hashed = start               # chunks folded into the stream digest so far
        self.done = False
        self.fin_sent = False

    def frames(self):
        """Frames that may go out now: new chunks up to the window, then the final digest."""
        out = []
        while self.next < self.count and self.next - self.base < self.window:
            payload = self.read(self.next * self.chunk, min(self.chunk, self.total - self.next * self.chunk))
            if self.next == self.hashed:
                self.digest.update(payload)
                self.hashed += 1
            self.held[self.next] = payload
            out.append(frame(DATA, self.tid, self.next, payload))
            self.next += 1
        self.peak_held = max(self.peak_held, sum(len(p) for p in self.held.values()))
        if self.base == self.count and not self.fin_sent:
            self.fin_sent = True
            out.append(frame(FIN, self.tid, self.count, self.digest.digest()))
        return out

    def resend(self):
        """Everything unacknowledged again (after a timeout, or when the receiver reports a gap)."""
        if self.base == self.count:
            return [frame(FIN, self.tid, self.count, self.digest.digest())]
        return [frame(DATA, self.tid, i, self.held[i]) for i in sorted(self.held)]

    def on_frame(self, raw):
        try:
            kind, tid, index, _payload = parse(raw)
        except TransferError:
            return []                     # a damaged acknowledgment: the retransmit timer covers it
        if tid != self.tid:
            return []
        if kind == ACK:                   # cumulative: everything below `index` is safely written
            for i in [i for i in self.held if i < index]:
                del self.held[i]
            self.base = max(self.base, index)
            if index > self.count:        # the acknowledgment of FIN
                self.done = True
                return []
            return self.frames()
        if kind == NACK:                  # the receiver wants chunk `index` (corrupt, or a gap before it)
            return self.resend()
        return []


class Receiver:
    """`write(offset, payload)` stores a chunk durably. `have` = chunks already stored (to resume), and
    `prefix_digest` the running SHA-256 over them."""

    def __init__(self, tid, write, total, *, chunk=CHUNK, have=0, prefix_digest=None):
        self.tid, self.write, self.total, self.chunk = tid, write, total, chunk
        self.count = (total + chunk - 1) // chunk
        self.have = have
        self.digest = prefix_digest or hashlib.sha256()
        self.complete = False
        self.failed = None
        self.rejected = 0                 # corrupt chunks turned away

    def on_frame(self, raw):
        """The frame to send back (an ACK or a NACK), or None."""
        try:
            kind, tid, index, payload = parse(raw)
        except TransferError:             # damaged anywhere — header or payload: never used, asked for again
            self.rejected += 1
            return frame(NACK, self.tid, self.have)
        if tid != self.tid or self.complete:
            return frame(ACK, self.tid, self.count + 1) if self.complete and tid == self.tid else None
        if kind == FIN:                   # its payload is the SHA-256 of the whole stream
            if self.have != self.count:
                return frame(NACK, self.tid, self.have)
            if payload != self.digest.digest():
                self.failed = "the transfer's final checksum does not match"
                return None
            self.complete = True
            return frame(ACK, self.tid, self.count + 1)
        if kind != DATA:
            return None
        if index != self.have:            # ahead of a gap, or a repeat of something already stored
            return frame(NACK if index > self.have else ACK, self.tid, self.have)
        self.write(index * self.chunk, payload)
        self.digest.update(payload)
        self.have += 1
        return frame(ACK, self.tid, self.have)

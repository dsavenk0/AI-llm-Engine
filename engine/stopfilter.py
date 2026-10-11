"""Stop strings, the OpenAI way: the stop sequence is never part of the answer, and text that might
still turn into one is held back instead of being streamed. Shared by the native engine and the
mlx-lm backend so both behave identically."""


class StopFilter:
    def __init__(self, stop):
        self.stop = [s for s in (stop or []) if s]
        self.text = ""        # everything fed so far
        self.sent = 0         # characters of `text` already released
        self.cut = False      # a stop string was found

    def feed(self, piece):
        """-> (hit, text to release now). `hit` means a stop string was completed: the text before it
        is released and generation should end."""
        if not self.stop:
            return False, piece
        self.text += piece
        hits = [i for i in (self.text.find(s, max(0, self.sent - len(s) + 1)) for s in self.stop) if i >= 0]
        if hits:
            cut = min(hits)
            out = self.text[self.sent:cut]
            self.sent, self.cut = cut, True
            return True, out
        # the tail could be the beginning of a stop string: keep it back until it is decided
        hold = max((k for s in self.stop for k in range(min(len(s) - 1, len(self.text)), 0, -1)
                    if self.text.endswith(s[:k])), default=0)
        end = len(self.text) - hold
        out = self.text[self.sent:end]
        self.sent = max(self.sent, end)
        return False, out

    def flush(self):
        """Text still held back when generation ended for another reason (it never completed a stop)."""
        if not self.stop or self.cut:
            return ""
        out = self.text[self.sent:]
        self.sent = len(self.text)
        return out

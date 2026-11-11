// Server-sent events over fetch. EventSource cannot send an Authorization header, so the stream is read by
// hand: `data:` lines joined until a blank line, comments (": keep-alive") ignored.

export class SseParser {
  private buf = "";
  private data: string[] = [];

  /** Feed a decoded chunk; returns the complete events it finished, as parsed JSON. */
  push(chunk: string): unknown[] {
    this.buf += chunk;
    const out: unknown[] = [];
    let nl: number;
    while ((nl = this.buf.indexOf("\n")) >= 0) {
      let line = this.buf.slice(0, nl);
      this.buf = this.buf.slice(nl + 1);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      if (line === "") {
        if (this.data.length) {
          const text = this.data.join("\n");
          this.data = [];
          try {
            out.push(JSON.parse(text));
          } catch {
            // a malformed event is skipped, never fatal
          }
        }
      } else if (line.startsWith("data:")) {
        this.data.push(line.slice(5).replace(/^ /, ""));
      }
    }
    return out;
  }
}

/** Read an SSE response body, calling `onEvent` for each event; resolves when the stream ends. */
export async function readSse(body: ReadableStream<Uint8Array>, onEvent: (ev: unknown) => void): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const ev of parser.push(decoder.decode(value, { stream: true }))) onEvent(ev);
  }
  for (const ev of parser.push(decoder.decode() + "\n\n")) onEvent(ev);
}

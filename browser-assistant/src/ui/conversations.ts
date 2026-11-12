// The side panel's conversations, kept in the extension's own storage so they outlive the panel, a reload of
// the extension and a restart of the browser — and can be opened again from the history, as in LLMBar's voice
// chat. Nothing here leaves the Mac. The storage is passed in, so the logic is tested without a browser.

export interface Turn { q: string; a: string }
export interface Conversation {
  id: string;
  title: string;              // the first question, until renamed
  host: string;               // the site it started on
  url: string;
  updated: number;            // ms since epoch
  turns: Turn[];
}

export interface Store {
  get(keys: string[]): Promise<Record<string, unknown>>;
  set(items: Record<string, unknown>): Promise<void>;
}

export const MAX_CONVERSATIONS = 60;
export const MAX_TURNS_KEPT = 40;
const MAX_TEXT = 8000;

export class Conversations {
  constructor(private store: Store, private now: () => number = Date.now) {}

  async list(): Promise<Conversation[]> {
    const all = ((await this.store.get(["conversations"])).conversations as Conversation[] | undefined) || [];
    return [...all].sort((a, b) => b.updated - a.updated);
  }

  async get(id: string): Promise<Conversation | undefined> {
    return (await this.list()).find((c) => c.id === id);
  }

  private async write(all: Conversation[]): Promise<void> {
    const kept = [...all].sort((a, b) => b.updated - a.updated).slice(0, MAX_CONVERSATIONS);
    await this.store.set({ conversations: kept });
  }

  /** A finished exchange. Starts the conversation when `id` is not stored yet; returns its id. */
  async add(id: string, turn: Turn, page: { url: string }): Promise<string> {
    const all = await this.list();
    let c = all.find((x) => x.id === id);
    if (!c) {
      let host = "";
      try { host = new URL(page.url).host; } catch { host = ""; }
      c = { id, title: turn.q.slice(0, 80) || "Conversation", host, url: page.url, updated: this.now(), turns: [] };
      all.push(c);
    }
    c.turns.push({ q: turn.q.slice(0, MAX_TEXT), a: turn.a.slice(0, MAX_TEXT) });
    c.turns = c.turns.slice(-MAX_TURNS_KEPT);
    c.updated = this.now();
    await this.write(all);
    return c.id;
  }

  async rename(id: string, title: string): Promise<void> {
    const all = await this.list();
    const c = all.find((x) => x.id === id);
    if (!c || !title.trim()) return;
    c.title = title.trim().slice(0, 80);
    await this.write(all);
  }

  async remove(id: string): Promise<void> {
    await this.write((await this.list()).filter((c) => c.id !== id));
    const map = await this.tabMap();
    for (const k of Object.keys(map)) if (map[k] === id) delete map[k];
    await this.store.set({ tabConversations: map });
  }

  // ── which conversation each tab shows (so reopening the panel brings it back) ──
  private async tabMap(): Promise<Record<string, string>> {
    return ((await this.store.get(["tabConversations"])).tabConversations as Record<string, string> | undefined) || {};
  }

  async forTab(tabId: number): Promise<string | undefined> {
    return (await this.tabMap())[String(tabId)];
  }

  async bind(tabId: number, id: string | null): Promise<void> {
    const map = await this.tabMap();
    if (id) map[String(tabId)] = id; else delete map[String(tabId)];
    await this.store.set({ tabConversations: map });
  }
}

/** The last exchanges to send with a new question (the service keeps what fits). */
export function historyFor(c: Conversation | undefined, n = 8): Turn[] {
  return c ? c.turns.slice(-n) : [];
}

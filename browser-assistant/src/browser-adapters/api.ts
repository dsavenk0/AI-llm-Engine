// The one place that knows which browser this is. Safari and Firefox expose `browser.*` with promises;
// Chromium exposes `chrome.*`, promise-returning in Manifest V3 for everything used here. Only the calls the
// extension makes are typed — narrow on purpose, so a missing API is a compile error, not a guess.

export interface Tab { id?: number; url?: string; title?: string; windowId?: number }
export interface Port {
  name: string;
  sender?: { tab?: Tab; id?: string };
  postMessage(msg: unknown): void;
  disconnect(): void;
  onMessage: { addListener(fn: (msg: any, port: Port) => void): void };
  onDisconnect: { addListener(fn: (port: Port) => void): void };
}
interface Event<F> { addListener(fn: F): void }

export interface BrowserApi {
  runtime: {
    id: string;
    connect(info?: { name?: string }): Port;
    onConnect: Event<(port: Port) => void>;
    onInstalled: Event<(details: { reason: string }) => void>;
    openOptionsPage(): Promise<void>;
    getURL(path: string): string;
    /** Safari: a message to the native handler of the app the extension ships in (LLMBar). */
    sendNativeMessage?(application: string, message: unknown): Promise<any>;
  };
  action?: { onClicked: Event<(tab: Tab) => void> };
  storage: {
    local: {
      get(keys: string | string[]): Promise<Record<string, any>>;
      set(items: Record<string, unknown>): Promise<void>;
    };
  };
  tabs: {
    query(q: { active?: boolean; currentWindow?: boolean }): Promise<Tab[]>;
    get(id: number): Promise<Tab & { status?: string }>;
    update(id: number, props: { url?: string }): Promise<Tab>;
    onUpdated: Event<(id: number, change: { status?: string; url?: string; title?: string }, tab: Tab) => void>;
    onRemoved: Event<(id: number) => void>;
    onActivated: Event<(info: { tabId: number; windowId: number }) => void>;
    /** A picture of the visible part of the window's active tab (needs <all_urls> or activeTab). */
    captureVisibleTab(windowId: number | undefined, options: { format: "jpeg" | "png"; quality?: number }): Promise<string>;
  };
  permissions: {
    contains(p: { origins: string[] }): Promise<boolean>;
    request(p: { origins: string[] }): Promise<boolean>;
  };
  scripting: {
    executeScript(details: { target: { tabId: number }; func?: (...a: any[]) => unknown; files?: string[];
      args?: unknown[] }): Promise<{ result?: unknown }[]>;
  };
  /** Chromium only (Chrome 114+): the docked side panel. Elsewhere the same page is docked into the tab. */
  sidePanel?: {
    setPanelBehavior(b: { openPanelOnActionClick: boolean }): Promise<void>;
  };
  contextMenus: {
    create(props: { id: string; title: string; contexts: string[] }): void;
    removeAll(): Promise<void>;
    onClicked: Event<(info: { menuItemId: string | number; selectionText?: string }, tab?: Tab) => void>;
  };
}

const g = globalThis as unknown as { browser?: BrowserApi; chrome?: BrowserApi };

export const api: BrowserApi = (g.browser ?? g.chrome) as BrowserApi;

export const browserName: "safari" | "chromium" | "firefox" | "unknown" =
  typeof navigator === "undefined" ? "unknown"
    : /Firefox\//.test(navigator.userAgent) ? "firefox"
    : /Chrome\//.test(navigator.userAgent) ? "chromium"
    : /Safari\//.test(navigator.userAgent) ? "safari" : "unknown";

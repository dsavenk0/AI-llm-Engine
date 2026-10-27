# Legacy: VS Code (Open VSX) extensions

Archived on 2026-10-27. The editor no longer installs or uses VS Code extensions; its place is
taken by the **Assistant Tools (MCP)** panel (connect MCP servers, with confirmation and safe
launch — the backend is `harness/mcp.py`).

This folder keeps the removed work for reference. **None of it is built or tested** by the app or
the test suite (`build-app.sh` compiles `app/**` only; the suite runs under `tests/`).

- `extensions.py` — the Python backend: search/install/enable/remove on open-vsx.org, and the
  declarative-only extraction of colour themes and snippets (extension code was never run).
- `test_extensions.py` — its unit tests (import `extensions` from this folder, not `ui.extensions`).
- `EditorExtensions.swift` — the Swift pieces that lived in the editor: the `ExtensionsSheet` UI,
  the response structs, the `ThemeOverride` colour theme, and the VS Code snippet engine
  (`SnippetExpander` / `SnippetMatch`, prefix + Tab). It references live app types and does not
  build on its own.

What was removed from the active tree (restore from git `82bdac1` if ever needed):

- `ui/extensions.py` and its ~10 `api_extensions_*` routes in `ui/server.py`.
- `app/editor/` — the `.extensions` module, `ExtensionsSheet` (Panels), the extension structs
  (Backend), `ThemeOverride` + custom-theme support (Theme), the snippet engine (Logic), snippet
  loading (Workspace) and Tab expansion (CodeEditor), plus the menu/sheet wiring (Shell).
- The snippet/theme checks in `tests/swift/main.swift`.

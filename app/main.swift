// LLMBar — one app, several modules: the menu-bar panel (engine control), the code editor and the voice chat.
// See app/core/Host.swift for how they share a process, and docs in README ("LLMBar").
import AppKit

// Child engines (started through ./llm) inherit this and stop themselves if the app goes away — so a crash
// never leaves an mlx engine or a llama-server orphaned, holding the model and the port. See engine/server.py.
setenv("LLM_PARENT_PID", "\(ProcessInfo.processInfo.processIdentifier)", 1)

MainActor.assumeIsolated {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    AppHost.installBaseMenu()
    app.run()
}

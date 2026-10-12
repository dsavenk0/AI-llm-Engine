// LLMBar — one app, several modules: the menu-bar panel (engine control), the code editor and the voice chat.
// See app/core/Host.swift for how they share a process, and docs in README ("LLMBar").
import AppKit

MainActor.assumeIsolated {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    AppHost.installBaseMenu()
    app.run()
}

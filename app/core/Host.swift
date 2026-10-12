// The host of the modules: what turns "one process" into "one app with parts". It decides when the app has a Dock
// icon, which main menu is showing, and where a link (the widget's, `./llm ui`) goes. The modules themselves never
// know about each other.
import AppKit

@MainActor
enum AppHost {
    private static var baseMenu: NSMenu?
    private static var observing = false

    // MARK: modules and their switches

    enum Module: String, CaseIterable {
        case editor, voice
        var key: String { "module.\(rawValue).enabled" }
    }

    /// Switched on by default; a module that is off has no button, no link and no window.
    static func enabled(_ m: Module) -> Bool { UserDefaults.standard.object(forKey: m.key) as? Bool ?? true }
    static func setEnabled(_ m: Module, _ on: Bool) { UserDefaults.standard.set(on, forKey: m.key) }

    // MARK: opening

    static func openEditor(project: String?) {
        guard enabled(.editor) else { return }
        install()
        EditorModule.shared.open(project: project)
        windowsChanged()
    }

    static func openVoice(listen: Bool = true) {
        guard enabled(.voice) else { return }
        install()
        VoiceModule.shared.show(listen: listen)
        windowsChanged()
    }

    static func openTerminal(_ command: String = "./llm") { CLI.terminal(command) }

    /// The model on or off, in the background — what the widget's power button asks for.
    static func toggleModel() {
        Task { await VoiceModule.shared.voice.toggleModelInBackground() }
    }

    // MARK: links

    /// True once any llmbar:// link was handled — a launch that came from a link does not also pop the panel open.
    static var openedByLink = false

    /// `llmbar://voice` · `llmbar://editor?root=/path` · `llmbar://cli` · `llmbar://toggle`.
    /// Opening a module this way also starts loading the model, so it is ready when the window is.
    static func handle(_ url: URL) {
        guard url.scheme == "llmbar" || url.scheme == "llmvoice" else { return }
        openedByLink = true
        let query = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        let q = { (name: String) in query.first { $0.name == name }?.value }
        var what = url.host ?? ""
        if what == "launch" || what == "listen" { what = q("target") ?? "voice" }          // the widget's older links
        switch what {
        case "toggle":
            toggleModel()
        case "editor":
            openEditor(project: q("root"))
            preloadModel()
        case "cli":
            openTerminal("./llm")
            preloadModel()
        default:
            openVoice()                                            // listening starts the model itself
        }
    }

    private static func preloadModel() {
        Task { _ = await VoiceModule.shared.voice.ensureModel(); }
    }

    // MARK: Dock icon and menus

    /// The app is a menu-bar app (no Dock icon) until one of its windows opens; then it is an ordinary app, so a
    /// minimised window can be found again. It goes back when the last of them closes.
    static func windowsChanged() {
        DispatchQueue.main.async {
            let any = EditorModule.shared.hasWindows || VoiceModule.shared.isOpen
            NSApp.setActivationPolicy(any ? .regular : .accessory)
            if any { NSApp.activate(ignoringOtherApps: true) }
        }
    }

    static func install() {
        guard !observing else { return }
        observing = true
        NotificationCenter.default.addObserver(forName: NSWindow.didBecomeKeyNotification, object: nil, queue: .main) { n in
            MainActor.assumeIsolated {
                guard let w = n.object as? NSWindow else { return }
                if EditorModule.shared.owns(w) { NSApp.mainMenu = EditorModule.shared.mainMenu }
                else if VoiceModule.shared.owns(w) || w is PanelWindow { NSApp.mainMenu = baseMenu }
            }
        }
    }

    static func installBaseMenu() {
        baseMenu = makeBaseMenu()
        NSApp.mainMenu = baseMenu
    }

    // MARK: development and tests

    /// `LLMBar --module editor …` / `--module voice …`: one module alone (self-test, snapshot) without the panel.
    /// Returns true when it took over.
    static func runModuleFromCommandLine() -> Bool {
        let a = CommandLine.arguments
        guard let i = a.firstIndex(of: "--module"), i + 1 < a.count else { return false }
        install()
        switch a[i + 1] {
        case "editor":
            NSApp.setActivationPolicy(a.contains("--selftest") ? .prohibited : .regular)
            if !EditorModule.shared.runCommandLineModes() { openEditor(project: nil) }
        case "voice":
            NSApp.setActivationPolicy(.regular)
            VoiceModule.shared.renderSnapshot(a)
        default:
            return false
        }
        return true
    }
}

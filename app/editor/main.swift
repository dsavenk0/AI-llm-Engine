// LLM Editor — a native, on-demand code editor for the local engine. Started by `./llm ui` or LLMBar;
// every window owns one project, its chats and a small Python backend (ui/server.py).
import AppKit
import SwiftUI

/// What LLMBar posts to a running editor to get one more window.
let newWindowNotification = Notification.Name("local.llm-engine.editor.new-window")

@MainActor
final class EditorWindowController: NSObject, NSWindowDelegate {
    let ws: Workspace
    let chat: ChatStore
    private(set) var window: NSWindow!
    weak var owner: AppDelegate?

    init(llmRoot: String, enginePort: Int) {
        let w = Workspace(llmRoot: llmRoot, enginePort: enginePort)
        ws = w
        chat = ChatStore(ws: w)
        super.init()
    }

    func show(near other: NSWindow?) {
        let host = NSHostingView(rootView: RootView(ws: ws, chat: chat, modules: ws.modules))
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1240, height: 780),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "LLM Editor"
        window.contentView = host
        window.isReleasedWhenClosed = false
        window.appearance = ThemeSettings.shared.appearance
        window.backgroundColor = Palette.windowNS(isDark: ThemeSettings.shared.isDark)
        window.delegate = self
        if let o = other {                                     // a new window cascades from the one before it
            window.setFrame(o.frame, display: false)
            let p = window.cascadeTopLeft(from: NSPoint(x: o.frame.minX, y: o.frame.maxY))
            window.setFrameTopLeftPoint(p)
        } else {
            window.center()
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func open(_ path: String) {
        let p = (path as NSString).standardizingPath
        UserDefaults.standard.set(p, forKey: "lastProject")
        Workspace.remember(project: p)
        window.title = "LLM Editor — \((p as NSString).lastPathComponent)"
        ws.openProject(p)
        chat.load(root: p)
    }

    /// The folder picker as a sheet on this window — never a blocking modal at launch (an app that opens with
    /// nothing but a panel looks like it did not start).
    func chooseProjectSheet() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Open"
        panel.message = "Choose the project folder"
        if let last = UserDefaults.standard.string(forKey: "lastProject") {
            panel.directoryURL = URL(fileURLWithPath: last).deletingLastPathComponent()
        }
        panel.beginSheetModal(for: window) { [weak self] r in
            if r == .OK, let url = panel.url { self?.open(url.path) }
        }
    }

    /// Ask for a folder; false when the user cancelled.
    @discardableResult
    func chooseProject() -> Bool {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Open"
        panel.message = "Choose the project folder"
        if let last = UserDefaults.standard.string(forKey: "lastProject") {
            panel.directoryURL = URL(fileURLWithPath: last).deletingLastPathComponent()
        }
        guard panel.runModal() == .OK, let url = panel.url else { return false }
        // switching the project replaces every open buffer: unsaved work must not vanish silently
        let dirty = ws.open.filter(\.dirty)
        if !dirty.isEmpty {
            let a = NSAlert()
            a.messageText = "\(dirty.count) file(s) have unsaved changes"
            a.informativeText = dirty.map { $0.path }.joined(separator: "\n")
            a.addButton(withTitle: "Save and switch")
            a.addButton(withTitle: "Discard and switch")
            a.addButton(withTitle: "Cancel")
            switch a.runModal() {
            case .alertFirstButtonReturn:
                Task { @MainActor in
                    if await self.ws.saveAllAndWait() { self.open(url.path) }
                    else { self.ws.say("Could not save every file; the project was not switched", bad: true) }
                }
                return true
            case .alertSecondButtonReturn: break
            default: return false
            }
        }
        open(url.path)
        return true
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        let dirty = ws.open.filter(\.dirty)
        if dirty.isEmpty { return true }
        let alert = NSAlert()
        alert.messageText = "\(dirty.count) file(s) have unsaved changes"
        alert.informativeText = dirty.map { $0.path }.joined(separator: "\n")
        alert.addButton(withTitle: "Save all and close")
        alert.addButton(withTitle: "Discard")
        alert.addButton(withTitle: "Cancel")
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            ws.saveAll()
            return false                                   // closes on the next attempt, once saved
        case .alertSecondButtonReturn: return true
        default: return false
        }
    }

    func windowWillClose(_ n: Notification) {
        ws.shutdown()                                      // this window's backend and shell go with it
        owner?.remove(self)
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuItemValidation {
    private var controllers: [EditorWindowController] = []
    private var llmRoot = Bundle.main.object(forInfoDictionaryKey: "LLMRoot") as? String ?? FileManager.default.currentDirectoryPath
    private var enginePort = 8090
    private var startRoot: String?

    override init() {
        let a = CommandLine.arguments
        var i = 1
        super.init()
        while i < a.count {
            switch a[i] {
            case "--root": if i + 1 < a.count { startRoot = a[i + 1]; i += 1 }
            case "--llm-root": if i + 1 < a.count { llmRoot = a[i + 1]; i += 1 }
            case "--engine-port": if i + 1 < a.count { enginePort = Int(a[i + 1]) ?? 8090; i += 1 }
            default: break
            }
            i += 1
        }
        if startRoot == nil { startRoot = UserDefaults.standard.string(forKey: "lastProject") }
    }

    /// The window that has the focus (or the newest one).
    var current: EditorWindowController? { controllers.first { $0.window?.isKeyWindow == true } ?? controllers.last }

    func remove(_ c: EditorWindowController) { controllers.removeAll { $0 === c } }

    func applicationDidFinishLaunching(_ n: Notification) {
        if CommandLine.arguments.contains("--selftest") {
            NSApp.setActivationPolicy(.prohibited)             // no window, no focus: a logic check
            let ws = Workspace(llmRoot: llmRoot, enginePort: enginePort)
            runSelfTest(ws, root: startRoot ?? FileManager.default.currentDirectoryPath)
            return
        }
        if CommandLine.arguments.contains("--selftest-notify") {     // the LLMBar → editor "new window" contract
            NSApp.setActivationPolicy(.prohibited)
            DistributedNotificationCenter.default().addObserver(forName: newWindowNotification, object: nil, queue: .main) { n in
                print("NEW-WINDOW \(n.userInfo?["root"] as? String ?? "")"); fflush(stdout); exit(0)
            }
            print("READY"); fflush(stdout)
            return
        }
        buildMenus()
        DistributedNotificationCenter.default().addObserver(
            self, selector: #selector(newWindowRequested(_:)), name: newWindowNotification, object: nil)
        newWindow(project: startRoot.flatMap { FileManager.default.fileExists(atPath: $0) ? $0 : nil })
        if let out = value(after: "--snapshot") { scheduleSnapshot(to: out) }
    }

    private func value(after flag: String) -> String? {
        let a = CommandLine.arguments
        guard let i = a.firstIndex(of: flag), i + 1 < a.count else { return nil }
        return a[i + 1]
    }

    /// `--snapshot out.png [--layout ide|chat] [--open-file rel] [--theme light|dark] [--bottom tab]`: renders the
    /// window's own view hierarchy to a PNG (no screen-recording permission needed) and quits. For checking the
    /// look of a build without a person at the screen.
    private func scheduleSnapshot(to path: String) {
        if let t = value(after: "--theme"), let m = ThemeMode(rawValue: t) { ThemeSettings.shared.mode = m }
        Task { @MainActor in
            guard let c = self.controllers.last else { exit(2) }
            for _ in 0..<150 where c.ws.starting { try? await Task.sleep(nanoseconds: 100_000_000) }
            if let l = self.value(after: "--layout") { c.ws.mode = l == "ide" ? .ide : .chat }
            if let f = self.value(after: "--open-file") { c.ws.openFile(f) }
            if let b = self.value(after: "--bottom") { c.ws.showPanel(b) }
            if CommandLine.arguments.contains("--all-modules") { ModuleID.allCases.forEach { if !ModuleSettings.shared.isOn($0) { ModuleSettings.shared.toggle($0) } } }
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            guard let view = c.window.contentView,
                  let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { exit(3) }
            if ProcessInfo.processInfo.environment["LLM_EDITOR_DEBUG_VIEWS"] != nil {
                func walk(_ v: NSView, _ chain: [String]) {
                    let me = chain + ["\(type(of: v))(a=\(v.alphaValue))"]
                    if v is NSTextView && !(v is ComposerNSTextView) {
                        print("TEXTVIEW", me.suffix(8).joined(separator: " > "))
                        if let tv = v as? NSTextView {
                            let attrs = tv.textStorage?.attributes(at: 40, effectiveRange: nil) ?? [:]
                            print("  attrs:", attrs[.foregroundColor] as Any, "typing:", tv.typingAttributes[.foregroundColor] as Any,
                                  "textColor:", tv.textColor as Any, "layer:", tv.layer?.opacity as Any, "appearance:", tv.effectiveAppearance.name.rawValue)
                        }
                    }
                    v.subviews.forEach { walk($0, me) }
                }
                walk(view, [])
            }
            view.cacheDisplay(in: view.bounds, to: rep)
            guard let png = rep.representation(using: .png, properties: [:]) else { exit(4) }
            try? png.write(to: URL(fileURLWithPath: path))
            print("SNAPSHOT \(path) \(Int(view.bounds.width))x\(Int(view.bounds.height))")
            fflush(stdout)
            c.ws.shutdown()
            exit(0)
        }
    }

    /// Another window, on `project` — or, without one, on a folder the user picks.
    func newWindow(project: String?) {
        let c = EditorWindowController(llmRoot: llmRoot, enginePort: enginePort)
        c.owner = self
        c.show(near: current?.window)
        controllers.append(c)
        c.ws.onChooseProject = { [weak c] in c?.chooseProjectSheet() }
        c.ws.onOpenProject = { [weak c] path in c?.open(path) }
        if let p = project { c.open(p) }
        else { c.ws.needsProject = true; c.ws.starting = false }         // the welcome screen, with the picker on top
    }

    @objc func newWindowRequested(_ n: Notification) {
        let root = (n.userInfo?["root"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        newWindow(project: root)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }

    /// ⌘Q does not go through windowShouldClose: ask here, once, about every window's unsaved files.
    func applicationShouldTerminate(_ app: NSApplication) -> NSApplication.TerminateReply {
        let dirty = controllers.flatMap { c in c.ws.open.filter(\.dirty).map { "\(c.ws.projectName)/\($0.path)" } }
        if dirty.isEmpty { return .terminateNow }
        let a = NSAlert()
        a.messageText = "\(dirty.count) file(s) have unsaved changes"
        a.informativeText = dirty.prefix(12).joined(separator: "\n") + (dirty.count > 12 ? "\n…" : "")
        a.addButton(withTitle: "Save all and quit")
        a.addButton(withTitle: "Quit without saving")
        a.addButton(withTitle: "Cancel")
        switch a.runModal() {
        case .alertFirstButtonReturn:
            Task { @MainActor in
                var ok = true
                for c in self.controllers { ok = await c.ws.saveAllAndWait() && ok }
                NSApp.reply(toApplicationShouldTerminate: ok)
                if !ok { self.controllers.first?.ws.say("Some files could not be saved; the editor stays open", bad: true) }
            }
            return .terminateLater
        case .alertSecondButtonReturn: return .terminateNow
        default: return .terminateCancel
        }
    }
    func applicationWillTerminate(_ n: Notification) { controllers.forEach { $0.ws.shutdown() } }

    // MARK: menus (act on the focused window)

    @objc func newEditorWindow() { newWindow(project: nil) }
    @objc func chooseProject() { current?.chooseProject() }
    @objc func save() { current?.ws.save() }
    @objc func saveAll() { current?.ws.saveAll() }
    @objc func closeTab() { if let ws = current?.ws, let p = ws.activePath { ws.closeFile(p) } }
    @objc func toggleLayout() { if let ws = current?.ws { ws.mode = ws.mode == .chat ? .ide : .chat } }
    @objc func newChat() { current?.chat.newChat(); current?.ws.mode = .chat }
    @objc func togglePanel() { current?.ws.bottomVisible.toggle() }
    @objc func showTerminal() { current?.ws.showPanel("terminal") }
    @objc func showProblems() { current?.ws.showPanel("problems") }
    @objc func toggleModule(_ sender: NSMenuItem) {
        if let m = sender.representedObject as? ModuleID { ModuleSettings.shared.toggle(m) }
    }
    @objc func showAbout() {
        NSApp.orderFrontStandardAboutPanel(options: [
            .applicationName: "LLM Editor",
            .credits: NSAttributedString(string: "A light, native code editor with a local assistant.\nPart of AI-llm-Engine · Apache-2.0")])
    }

    func validateMenuItem(_ item: NSMenuItem) -> Bool {
        if let m = item.representedObject as? ModuleID {
            item.state = ModuleSettings.shared.isOn(m) ? .on : .off
            return m != .files
        }
        if item.action == #selector(save) || item.action == #selector(closeTab) { return current?.ws.active != nil }
        if item.action == #selector(chooseProject) || item.action == #selector(newChat) { return current != nil }
        return true
    }

    private func buildMenus() {
        let main = NSMenu()
        func add(_ title: String, _ items: [NSMenuItem]) {
            let top = NSMenuItem(); main.addItem(top)
            let m = NSMenu(title: title); items.forEach(m.addItem); top.submenu = m
        }
        func item(_ title: String, _ action: Selector?, _ key: String = "", _ mods: NSEvent.ModifierFlags = .command) -> NSMenuItem {
            let i = NSMenuItem(title: title, action: action, keyEquivalent: key)
            i.keyEquivalentModifierMask = mods
            if action != nil, action != #selector(NSApplication.terminate(_:)) { i.target = self }
            return i
        }
        let quit = NSMenuItem(title: "Quit LLM Editor", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        add("LLM Editor", [item("About LLM Editor", #selector(showAbout)), .separator(), quit])
        add("File", [item("New Window…", #selector(newEditorWindow), "n", [.command, .shift]),
                     item("Open Project…", #selector(chooseProject), "o"), .separator(),
                     item("Save", #selector(save), "s"), item("Save All", #selector(saveAll), "s", [.command, .option]),
                     item("Close Tab", #selector(closeTab), "w"), .separator(), item("New Chat", #selector(newChat), "n")])
        let edit: [NSMenuItem] = [
            NSMenuItem(title: "Undo", action: Selector(("undo:")), keyEquivalent: "z"),
            NSMenuItem(title: "Redo", action: Selector(("redo:")), keyEquivalent: "Z"), .separator(),
            NSMenuItem(title: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x"),
            NSMenuItem(title: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c"),
            NSMenuItem(title: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v"),
            NSMenuItem(title: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a"),
        ]
        add("Edit", edit)
        var view = [item("Switch Chat / Editor", #selector(toggleLayout), "e"),
                    item("Show / Hide Panel", #selector(togglePanel), "j"),
                    item("Problems", #selector(showProblems), "m", [.command, .shift]),
                    item("Terminal", #selector(showTerminal), "`", .control), .separator()]
        for m in ModuleID.allCases {
            let i = item(m.title, #selector(toggleModule(_:)))
            i.representedObject = m
            view.append(i)
        }
        add("View", view)
        add("Window", [NSMenuItem(title: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")])
        NSApp.mainMenu = main
    }
}

MainActor.assumeIsolated {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    app.setActivationPolicy(.regular)
    app.run()
}

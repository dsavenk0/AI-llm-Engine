// The editor module of LLMBar — a native code editor for the local engine. Opened from the panel, the tool row,
// the llm://editor link or `./llm ui`; every window owns one project, its chats and a small Python backend
// (ui/server.py). It lives in the same process as the other modules and is created only when first used.
import AppKit
import SwiftUI

@MainActor
final class EditorWindowController: NSObject, NSWindowDelegate {
    let ws: Workspace
    let chat: ChatStore
    private(set) var window: NSWindow!
    weak var owner: EditorModule?

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

/// The editor as a module of the one app: owns the editor windows, their menu and the unsaved-files check.
@MainActor
final class EditorModule: NSObject, NSMenuItemValidation {
    static let shared = EditorModule()

    private var controllers: [EditorWindowController] = []
    private var llmRoot: String { Paths.shared.root }
    private var enginePort: Int { Config.port }
    private var startRoot: String?
    private var menu: NSMenu?

    private override init() {
        super.init()
        let a = CommandLine.arguments
        if let i = a.firstIndex(of: "--root"), i + 1 < a.count { startRoot = a[i + 1] }
        if startRoot == nil { startRoot = UserDefaults.standard.string(forKey: "lastProject") }
    }

    var hasWindows: Bool { !controllers.isEmpty }
    func owns(_ window: NSWindow?) -> Bool { controllers.contains { $0.window === window } }

    /// The editor's own main menu, shown while one of its windows is the key window.
    var mainMenu: NSMenu {
        if menu == nil { buildMenus() }
        return menu!
    }

    /// Opens a window on `project` — or on the last project, or the welcome screen when there is none.
    func open(project: String? = nil) {
        let target = project ?? startRoot.flatMap { FileManager.default.fileExists(atPath: $0) ? $0 : nil }
        newWindow(project: target)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Driven from the command line (`--module editor --selftest | --snapshot …`): returns true when it handled it.
    func runCommandLineModes() -> Bool {
        if CommandLine.arguments.contains("--selftest") {
            let ws = Workspace(llmRoot: llmRoot, enginePort: enginePort)
            runSelfTest(ws, root: startRoot ?? FileManager.default.currentDirectoryPath)
            return true
        }
        if let out = value(after: "--snapshot") {
            open()
            scheduleSnapshot(to: out)
            return true
        }
        return false
    }

    private func value(after flag: String) -> String? {
        let a = CommandLine.arguments
        guard let i = a.firstIndex(of: flag), i + 1 < a.count else { return nil }
        return a[i + 1]
    }

    /// The window that has the focus (or the newest one).
    var current: EditorWindowController? { controllers.first { $0.window?.isKeyWindow == true } ?? controllers.last }

    func remove(_ c: EditorWindowController) { controllers.removeAll { $0 === c } }

    /// `--snapshot out.png [--layout ide|chat] [--open-file rel] [--theme light|dark] [--bottom tab] [--size WxH] [--resize WxH]`: renders the
    /// window's own view hierarchy to a PNG (no screen-recording permission needed) and quits. For checking the
    /// look of a build without a person at the screen.
    private func scheduleSnapshot(to path: String) {
        if let t = value(after: "--theme"), let m = ThemeMode(rawValue: t) { ThemeSettings.shared.mode = m }
        Task { @MainActor in
            guard let c = self.controllers.last else { exit(2) }
            for _ in 0..<150 where c.ws.starting { try? await Task.sleep(nanoseconds: 100_000_000) }
            if let z = self.value(after: "--size") {                        // WxH of the content, in points
                let p = z.split(separator: "x").compactMap { Double($0) }
                if p.count == 2 { c.window.setContentSize(NSSize(width: p[0], height: p[1])) }
            }
            if let l = self.value(after: "--layout") { c.ws.mode = l == "ide" ? .ide : .chat }
            if let f = self.value(after: "--open-file") { c.ws.openFile(f) }
            if let b = self.value(after: "--bottom") { c.ws.showPanel(b) }
            if CommandLine.arguments.contains("--all-modules") { ModuleID.allCases.forEach { if !ModuleSettings.shared.isOn($0) { ModuleSettings.shared.toggle($0) } } }
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            if let z = self.value(after: "--resize") {                      // widen/narrow after the file is shown
                let p = z.split(separator: "x").compactMap { Double($0) }
                if p.count == 2 { c.window.setContentSize(NSSize(width: p[0], height: p[1])) }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            if CommandLine.arguments.contains("--modes-panel") {           // the Modes popover on its own
                let store = ChatStore(ws: c.ws)
                let host = NSHostingView(rootView: ModesPanel(ws: c.ws, store: store, shown: .constant(true))
                    .background(Palette.panel).environment(\.colorScheme, ThemeSettings.shared.isDark ? .dark : .light))
                host.frame = NSRect(origin: .zero, size: host.fittingSize)
                c.window.contentView = host
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
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
                                  "textColor:", tv.textColor as Any, "textkit2:", tv.textLayoutManager != nil, "layer:", tv.layer?.opacity as Any, "appearance:", tv.effectiveAppearance.name.rawValue)
                        }
                    }
                    v.subviews.forEach { walk($0, me) }
                }
                walk(view, [])
            }
            if ProcessInfo.processInfo.environment["LLM_EDITOR_DEBUG_FRAMES"] != nil {
                func dump(_ v: NSView, _ depth: Int) {
                    let r = v.convert(v.bounds, to: nil)
                    let name = "\(type(of: v))"
                    if let ruler = v as? NSRulerView {
                        print("  RULER frame=\(ruler.frame) bounds=\(ruler.bounds) thickness=\(ruler.ruleThickness) "
                              + "required=\(ruler.requiredThickness)")
                    }
                    if let clip = v as? NSClipView, clip.documentView is NSTextView {
                        print("  CLIP bounds=\(clip.bounds) frame=\(clip.frame) doc=\(clip.documentView!.frame)")
                    }
                    if depth <= 30, r.width > 60, !name.hasPrefix("_NSInspectorBar") {
                        print(String(repeating: " ", count: min(depth, 24)), name.prefix(70),
                              "x=\(Int(r.minX))..\(Int(r.maxX)) y=\(Int(r.minY))..\(Int(r.maxY)) clip=\(v.clipsToBounds)")
                    }
                    v.subviews.forEach { dump($0, depth + 1) }
                }
                dump(view, 0)
            }
            // what the compositor really shows (layer tiles included), for the window's own pixels only —
            // cacheDisplay below redraws everything and so hides stale-layer defects
            if let out = ProcessInfo.processInfo.environment["LLM_EDITOR_WINDOW_CAPTURE"],
               let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "CGWindowListCreateImage") {
                typealias Capture = @convention(c) (CGRect, UInt32, UInt32, UInt32) -> Unmanaged<CGImage>?
                c.window.orderFrontRegardless()
                try? await Task.sleep(nanoseconds: 600_000_000)
                let f = unsafeBitCast(sym, to: Capture.self)
                if let img = f(CGRect.null, 8, UInt32(c.window.windowNumber), 1)?.takeRetainedValue(),
                   let png = NSBitmapImageRep(cgImage: img).representation(using: .png, properties: [:]) {
                    try? png.write(to: URL(fileURLWithPath: out))
                    print("WINDOW-CAPTURE \(out) \(img.width)x\(img.height)")
                } else { print("WINDOW-CAPTURE failed") }
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

    /// ⌘Q does not go through windowShouldClose: the host asks here, once, about every window's unsaved files.
    func shouldTerminate() -> NSApplication.TerminateReply {
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
    func willTerminate() { controllers.forEach { $0.ws.shutdown() } }

    // MARK: menus (act on the focused window)

    @objc func newEditorWindow() { newWindow(project: nil) }
    @objc func chooseProject() { current?.chooseProject() }
    @objc func save() { current?.ws.save() }
    @objc func saveAll() { current?.ws.saveAll() }
    @objc func closeTab() { if let ws = current?.ws, let p = ws.activePath { ws.closeFile(p) } }
    @objc func toggleLayout() { if let ws = current?.ws { ws.mode = ws.mode == .chat ? .ide : .chat } }
    @objc func newChat() { current?.chat.newChat(); current?.ws.mode = .chat }
    @objc func togglePanel() { current?.ws.bottomVisible.toggle() }
    @objc func showTerminal() {
        if !ModuleSettings.shared.isOn(.terminal) { ModuleSettings.shared.toggle(.terminal) }
        current?.ws.showPanel("terminal")
    }
    @objc func showProblems() { current?.ws.showPanel("problems") }
    @objc func showBrowser() {                                  // the opt-in Preview module: switching it on shows it
        if !ModuleSettings.shared.isOn(.preview) { ModuleSettings.shared.toggle(.preview) }
        current?.ws.showPanel("preview")
    }
    @objc func showFiles() {
        guard let c = current else { return }
        c.ws.mode = .ide
        c.ws.leftTab = 0
    }
    @objc func showChanges() {
        guard let c = current else { return }
        if !ModuleSettings.shared.isOn(.git) { ModuleSettings.shared.toggle(.git) }
        c.ws.mode = .ide
        c.ws.leftTab = 1
    }
    @objc func toggleStatusBar() { ViewPrefs.shared.statusBar.toggle() }
    @objc func zoomIn() { ViewPrefs.shared.zoom += 0.1 }
    @objc func zoomOut() { ViewPrefs.shared.zoom -= 0.1 }
    @objc func zoomReset() { ViewPrefs.shared.zoom = 1.0 }
    @objc func openVoiceChat() { AppHost.openVoice() }
    @objc func openPanel() { AppHost.openPanel() }
    @objc func toggleModule(_ sender: NSMenuItem) {
        if let m = sender.representedObject as? ModuleID { ModuleSettings.shared.toggle(m) }
    }

    func validateMenuItem(_ item: NSMenuItem) -> Bool {
        if let m = item.representedObject as? ModuleID {
            item.state = ModuleSettings.shared.isOn(m) ? .on : .off
            return m != .files
        }
        if item.action == #selector(toggleStatusBar) { item.state = ViewPrefs.shared.statusBar ? .on : .off }
        if item.action == #selector(zoomIn) { return ViewPrefs.shared.zoom < 2.0 }
        if item.action == #selector(zoomOut) { return ViewPrefs.shared.zoom > 0.7 }
        if item.action == #selector(save) || item.action == #selector(closeTab) { return current?.ws.active != nil }
        if item.action == #selector(chooseProject) || item.action == #selector(newChat) { return current != nil }
        return true
    }

    /// App · File · Edit · View · Window · Help — the same set of menus a Mac editor has, so nothing a person
    /// looks for in the menu bar is missing. Edit is titled "Edit" so the system adds Autofill, Dictation and Emoji.
    private func buildMenus() {
        let main = NSMenu()
        func add(_ title: String, _ items: [NSMenuItem]) {
            let top = NSMenuItem(); main.addItem(top)
            let m = NSMenu(title: title); items.forEach(m.addItem); top.submenu = m
        }
        func item(_ title: String, _ action: Selector?, _ key: String = "", _ mods: NSEvent.ModifierFlags = .command) -> NSMenuItem {
            let i = NSMenuItem(title: title, action: action, keyEquivalent: key)
            i.keyEquivalentModifierMask = mods
            if action != nil { i.target = self }
            return i
        }
        main.addItem(AppMenus.applicationMenu())
        add("File", [item("New Chat", #selector(newChat), "n"),
                     item("New Window", #selector(newEditorWindow), "n", [.command, .shift]),
                     item("Open Folder…", #selector(chooseProject), "o"), .separator(),
                     item("New Terminal", #selector(showTerminal)),
                     item("New Browser", #selector(showBrowser)), .separator(),
                     item("Save", #selector(save), "s"), item("Save All", #selector(saveAll), "s", [.command, .option]),
                     item("Close Tab", #selector(closeTab), "w"), .separator(),
                     item("Open Voice Chat", #selector(openVoiceChat)),
                     item("Open LLMBar Panel", #selector(openPanel))])
        add("Edit", [
            NSMenuItem(title: "Undo", action: Selector(("undo:")), keyEquivalent: "z"),
            { let r = NSMenuItem(title: "Redo", action: Selector(("redo:")), keyEquivalent: "z")
              r.keyEquivalentModifierMask = [.command, .shift]; return r }(), .separator(),
            NSMenuItem(title: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x"),
            NSMenuItem(title: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c"),
            NSMenuItem(title: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v"),
            .separator(),
            NSMenuItem(title: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a"),
        ])
        var view = [item("Chat ⇄ Editor", #selector(toggleLayout), "e"),
                    item("Files", #selector(showFiles), "e", [.command, .shift]),
                    item("Changes", #selector(showChanges), "g", [.control, .shift]),
                    item("Browser", #selector(showBrowser), "b", [.command, .shift]),
                    item("Terminal", #selector(showTerminal), "`", .control),
                    item("Problems", #selector(showProblems), "m", [.command, .shift]),
                    item("Show / Hide Panel", #selector(togglePanel), "j"), .separator(),
                    item("Status Bar", #selector(toggleStatusBar)), .separator(),
                    item("Zoom In", #selector(zoomIn), "+"), item("Zoom Out", #selector(zoomOut), "-"),
                    item("Reset Zoom", #selector(zoomReset), "0"), .separator()]
        let modulesItem = NSMenuItem(title: "Modules", action: nil, keyEquivalent: "")
        let modulesMenu = NSMenu(title: "Modules")
        for m in ModuleID.allCases {
            let i = item(m.title, #selector(toggleModule(_:)))
            i.representedObject = m
            modulesMenu.addItem(i)
        }
        modulesItem.submenu = modulesMenu
        view.append(modulesItem)
        add("View", view)
        main.addItem(AppMenus.windowMenu())
        main.addItem(AppMenus.helpMenu())
        menu = main
    }
}

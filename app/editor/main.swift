// LLM Editor — a native, on-demand code editor for the local engine. Started by `./llm ui` or LLMBar;
// it owns a small Python backend (ui/server.py) for highlighting, structure, git and the agent loop.
import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSMenuItemValidation {
    var window: NSWindow!
    let ws: Workspace
    let chat: ChatStore
    private var startRoot: String?

    override init() {
        var root: String?
        var llmRoot = Bundle.main.object(forInfoDictionaryKey: "LLMRoot") as? String ?? FileManager.default.currentDirectoryPath
        var port = 8090
        let a = CommandLine.arguments
        var i = 1
        while i < a.count {
            switch a[i] {
            case "--root": if i + 1 < a.count { root = a[i + 1]; i += 1 }
            case "--llm-root": if i + 1 < a.count { llmRoot = a[i + 1]; i += 1 }
            case "--engine-port": if i + 1 < a.count { port = Int(a[i + 1]) ?? 8090; i += 1 }
            default: break
            }
            i += 1
        }
        let w = Workspace(llmRoot: llmRoot, enginePort: port)
        ws = w
        chat = ChatStore(ws: w)
        startRoot = root ?? UserDefaults.standard.string(forKey: "lastProject")
        super.init()
    }

    func applicationDidFinishLaunching(_ n: Notification) {
        if CommandLine.arguments.contains("--selftest") {
            NSApp.setActivationPolicy(.prohibited)             // no window, no focus: a logic check
            runSelfTest(ws, root: startRoot ?? FileManager.default.currentDirectoryPath)
            return
        }
        buildMenus()
        let host = NSHostingView(rootView: RootView(ws: ws, chat: chat, modules: ws.modules))
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1240, height: 780),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "LLM Editor"
        window.contentView = host
        window.center()
        window.delegate = self
        window.setFrameAutosaveName("LLMEditorMain")
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        if let r = startRoot, FileManager.default.fileExists(atPath: r) { open(r) } else { chooseProject() }
    }

    func open(_ path: String) {
        let p = (path as NSString).standardizingPath
        UserDefaults.standard.set(p, forKey: "lastProject")
        window.title = "LLM Editor — \((p as NSString).lastPathComponent)"
        ws.openProject(p)
        chat.load(root: p)
    }

    @objc func chooseProject() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Open"
        panel.message = "Choose the project folder"
        if panel.runModal() == .OK, let url = panel.url { open(url.path) }
        else if ws.root.isEmpty { NSApp.terminate(nil) }
    }

    func applicationWillTerminate(_ n: Notification) { ws.shutdown() }
    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        let dirty = ws.open.filter(\.dirty)
        if dirty.isEmpty { return true }
        let alert = NSAlert()
        alert.messageText = "\(dirty.count) file(s) have unsaved changes"
        alert.informativeText = dirty.map { $0.path }.joined(separator: "\n")
        alert.addButton(withTitle: "Save all and quit")
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

    // MARK: menus

    @objc func save() { ws.save() }
    @objc func saveAll() { ws.saveAll() }
    @objc func closeTab() { if let p = ws.activePath { ws.closeFile(p) } }
    @objc func toggleLayout() { ws.mode = ws.mode == .chat ? .ide : .chat }
    @objc func newChat() { chat.newChat(); ws.mode = .chat }
    @objc func togglePanel() { ws.bottomVisible.toggle() }
    @objc func showTerminal() { ws.showPanel("terminal") }
    @objc func showProblems() { ws.showPanel("problems") }
    @objc func toggleModule(_ sender: NSMenuItem) {
        if let m = sender.representedObject as? ModuleID { ws.modules.toggle(m) }
    }
    @objc func showAbout() {
        NSApp.orderFrontStandardAboutPanel(options: [
            .applicationName: "LLM Editor",
            .credits: NSAttributedString(string: "A light, native code editor with a local assistant.\nPart of AI-llm-Engine · Apache-2.0")])
    }

    func validateMenuItem(_ item: NSMenuItem) -> Bool {
        if let m = item.representedObject as? ModuleID { item.state = ws.modules.isOn(m) ? .on : .off; return m != .files }
        if item.action == #selector(save) || item.action == #selector(closeTab) { return ws.active != nil }
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
        add("File", [item("Open Project…", #selector(chooseProject), "o"), .separator(),
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

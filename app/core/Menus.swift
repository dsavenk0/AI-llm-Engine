// The parts of the main menu every module shares: the application menu (About, Settings, Services, Hide, Quit),
// Window (the system fills it with Fill / Center / Move & Resize / Tile and the window list) and Help (the system
// adds the search field). The editor and the base menu add their own File / Edit / View between them.
import AppKit

@MainActor
final class MenuActions: NSObject {
    static let shared = MenuActions()
    static let repo = "https://github.com/dsavenk0/AI-llm-Engine"

    @objc func about() {
        NSApp.orderFrontStandardAboutPanel(options: [
            .applicationName: "LLMBar",
            .credits: NSAttributedString(string: "A local model hub for the Mac: the engine panel, a code editor and a voice chat in one app.\nAI-llm-Engine · Apache-2.0")])
        NSApp.activate(ignoringOtherApps: true)
    }
    @objc func checkForUpdates() { open(MenuActions.repo + "/releases") }
    @objc func settings() { AppHost.openPanel() }
    @objc func docs() { open(MenuActions.repo + "#readme") }
    @objc func issues() { open(MenuActions.repo + "/issues") }
    @objc func license() {
        let local = Paths.shared.root + "/LICENSE"
        if FileManager.default.fileExists(atPath: local) { NSWorkspace.shared.open(URL(fileURLWithPath: local)) }
        else { open(MenuActions.repo + "/blob/main/LICENSE") }
    }
    @objc func shortcuts() {
        let a = NSAlert()
        a.messageText = "Keyboard shortcuts"
        a.informativeText = """
        ⌘N  new chat          ⇧⌘N  new window          ⌘O  open folder
        ⌘S  save              ⌥⌘S  save all            ⌘W  close tab
        ⌘E  chat ⇄ editor     ⌘J  show / hide panel    ⌃`  terminal
        ⇧⌘M  problems         ⇧⌘B  preview (browser)   ⇧⌘E  files
        ⌃⇧G  changes          ⌘+ / ⌘− / ⌘0  zoom       ⇧Tab  next agent mode
        Enter  send           ⇧Enter  new line
        """
        a.runModal()
    }
    private func open(_ s: String) { if let u = URL(string: s) { NSWorkspace.shared.open(u) } }
}

@MainActor
enum AppMenus {
    private static func item(_ title: String, _ action: Selector, _ key: String = "", _ mods: NSEvent.ModifierFlags = .command,
                             target: AnyObject? = nil, own: Bool = true) -> NSMenuItem {
        let i = NSMenuItem(title: title, action: action, keyEquivalent: key)
        i.keyEquivalentModifierMask = mods
        i.target = own ? (target ?? MenuActions.shared) : nil
        return i
    }

    /// About · Check for Updates · Settings · Services · Hide · Hide Others · Show All · Quit.
    static func applicationMenu() -> NSMenuItem {
        let top = NSMenuItem()
        let m = NSMenu(title: "LLMBar")
        m.addItem(item("About LLMBar", #selector(MenuActions.about)))
        m.addItem(item("Check for Updates…", #selector(MenuActions.checkForUpdates)))
        m.addItem(.separator())
        m.addItem(item("Settings…", #selector(MenuActions.settings), ","))
        m.addItem(.separator())
        let services = NSMenuItem(title: "Services", action: nil, keyEquivalent: "")
        let servicesMenu = NSMenu(title: "Services")
        services.submenu = servicesMenu
        NSApp.servicesMenu = servicesMenu
        m.addItem(services)
        m.addItem(.separator())
        m.addItem(item("Hide LLMBar", #selector(NSApplication.hide(_:)), "h", own: false))
        m.addItem(item("Hide Others", #selector(NSApplication.hideOtherApplications(_:)), "h", [.command, .option], own: false))
        m.addItem(item("Show All", #selector(NSApplication.unhideAllApplications(_:)), own: false))
        m.addItem(.separator())
        m.addItem(item("Quit LLMBar", #selector(NSApplication.terminate(_:)), "q", own: false))
        top.submenu = m
        return top
    }

    static func windowMenu() -> NSMenuItem {
        let top = NSMenuItem()
        let m = NSMenu(title: "Window")
        m.addItem(item("Minimize", #selector(NSWindow.performMiniaturize(_:)), "m", own: false))
        m.addItem(item("Zoom", #selector(NSWindow.performZoom(_:)), own: false))
        m.addItem(.separator())
        m.addItem(item("Bring All to Front", #selector(NSApplication.arrangeInFront(_:)), own: false))
        top.submenu = m
        NSApp.windowsMenu = m                                    // the system adds the window list and the tiling items
        return top
    }

    static func helpMenu(extra: [NSMenuItem] = []) -> NSMenuItem {
        let top = NSMenuItem()
        let m = NSMenu(title: "Help")
        extra.forEach(m.addItem)
        if !extra.isEmpty { m.addItem(.separator()) }
        m.addItem(item("Keyboard Shortcuts", #selector(MenuActions.shortcuts)))
        m.addItem(item("Documentation", #selector(MenuActions.docs)))
        m.addItem(item("Report an Issue…", #selector(MenuActions.issues)))
        m.addItem(.separator())
        m.addItem(item("View License", #selector(MenuActions.license)))
        top.submenu = m
        NSApp.helpMenu = m                                       // the system adds the search field
        return top
    }
}

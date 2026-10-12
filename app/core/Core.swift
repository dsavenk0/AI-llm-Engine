// Shared by every module of the app: where the config lives, what it says, and the one theme setting.
// The Bar panel, the editor and the voice chat are modules of one process; this is the only place that reads
// or writes `.llm/config.json` outside the Bar's own settings screens, so they cannot disagree.
import AppKit

extension Notification.Name {
    /// Posted after the theme setting changed (from LLMBar's settings or from any module's own control).
    static let appThemeChanged = Notification.Name("local.llm-engine.theme-changed")
}

/// `.llm/config.json` — the file `./llm` and the Bar share.
enum Config {
    static func read() -> [String: Any] {
        guard let d = FileManager.default.contents(atPath: Paths.shared.config),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return [:] }
        return j
    }

    /// Read-modify-write of one top-level key; everything else in the file is kept.
    static func set(_ key: String, _ value: Any?) {
        var j = read()
        j[key] = value
        try? FileManager.default.createDirectory(atPath: Paths.shared.home + "/.llm", withIntermediateDirectories: true)
        if let d = try? JSONSerialization.data(withJSONObject: j, options: [.prettyPrinted, .sortedKeys]) {
            try? d.write(to: URL(fileURLWithPath: Paths.shared.config), options: .atomic)
        }
    }

    /// The engine's port: `--engine-port` on the command line (tests, development), else the shared config.
    static var port: Int {
        let a = CommandLine.arguments
        if let i = a.firstIndex(of: "--engine-port"), i + 1 < a.count, let p = Int(a[i + 1]) { return p }
        return read()["port"] as? Int ?? 8090
    }
    static var apiKey: String { read()["api_key"] as? String ?? "" }
    static var selectedModel: String { read()["model"] as? String ?? "" }
    /// LLMBar's transparency slider (0.15 … 0.85): how much of the desktop shows through the windows.
    static var transparency: Double { min(0.85, max(0.15, read()["glass_transparency"] as? Double ?? 0.55)) }
}

/// One theme for the whole app: the "theme" setting of LLMBar (system / dark / light). The editor's picker and the
/// voice chat follow it; changing it anywhere changes it everywhere.
enum AppTheme {
    static var setting: String { Config.read()["theme"] as? String ?? "system" }

    static var appearance: NSAppearance? {
        switch setting {
        case "light": return NSAppearance(named: .aqua)
        case "dark": return NSAppearance(named: .darkAqua)
        default: return nil
        }
    }

    static var isLight: Bool {
        switch setting {
        case "light": return true
        case "dark": return false
        default: return NSApp?.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .aqua
        }
    }

    /// Recomputes what depends on the setting and tells everyone (windows, SwiftUI views, the editor's palette).
    static func refresh() {
        lightTheme = isLight
        NSApp?.appearance = appearance
        NotificationCenter.default.post(name: .appThemeChanged, object: nil)
    }

    static func set(_ value: String) {
        guard value != setting else { return }
        Config.set("theme", value)
        refresh()
    }
}

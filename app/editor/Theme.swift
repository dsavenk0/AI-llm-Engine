// Appearance: System / Light / Dark, and a palette related to LLMBar's (graphite-blue ground, cyan accent)
// that stays opaque — the editor holds text people read for hours, so no see-through backgrounds.
import AppKit
import SwiftUI

enum ThemeMode: String, CaseIterable, Identifiable {
    case system, light, dark
    var id: String { rawValue }
    var title: String {
        switch self { case .system: return "System"; case .light: return "Light"; case .dark: return "Dark" }
    }
}

/// One setting for every window of the editor (like the module switches).
final class ThemeSettings: ObservableObject {
    static let shared = ThemeSettings()
    @Published var mode: ThemeMode {
        didSet {
            UserDefaults.standard.set(mode.rawValue, forKey: "themeMode")
            apply()
        }
    }

    init() {
        mode = ThemeMode(rawValue: UserDefaults.standard.string(forKey: "themeMode") ?? "") ?? .dark
    }

    /// nil = follow the system.
    var appearance: NSAppearance? {
        switch mode {
        case .system: return nil
        case .light: return NSAppearance(named: .aqua)
        case .dark: return NSAppearance(named: .darkAqua)
        }
    }

    var colorScheme: ColorScheme? {
        switch mode { case .system: return nil; case .light: return .light; case .dark: return .dark }
    }

    var isDark: Bool {
        switch mode {
        case .dark: return true
        case .light: return false
        case .system: return NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        }
    }

    /// Pushes the choice to every window (AppKit controls and the text views' dynamic colours follow it).
    func apply() {
        NSApp.appearance = appearance
        for w in NSApp.windows { w.appearance = appearance; w.backgroundColor = Palette.windowNS(isDark: isDark) }
        NotificationCenter.default.post(name: .themeChanged, object: nil)
    }
}

extension Notification.Name {
    static let themeChanged = Notification.Name("local.llm-engine.editor.theme-changed")
}

enum Palette {
    private static var dark: Bool { ThemeSettings.shared.isDark }
    private static func c(_ l: (Double, Double, Double), _ d: (Double, Double, Double)) -> Color {
        let v = dark ? d : l
        return Color(red: v.0, green: v.1, blue: v.2)
    }

    static var window: Color { c((0.965, 0.972, 0.980), (0.055, 0.075, 0.100)) }
    static var panel: Color { c((1.0, 1.0, 1.0), (0.082, 0.108, 0.140)) }
    static var field: Color { c((0.935, 0.945, 0.955), (0.110, 0.140, 0.180)) }
    /// LLMBar's cyan, deepened on light so it stays readable.
    static var accent: Color { c((0.0, 0.45, 0.62), (0.0, 0.82, 0.95)) }
    /// The Send button: a plain strong blue, not the accent.
    static var send: Color { c((0.0, 0.40, 0.95), (0.20, 0.50, 1.0)) }
    static var hairline: Color { c((0.0, 0.0, 0.0), (0.45, 0.80, 0.92)).opacity(dark ? 0.16 : 0.10) }

    /// The code area is opaque on purpose: AppKit renders text on a non-opaque text view without font
    /// smoothing, which is what made the code look thin and washed out.
    static func editorNS(isDark: Bool) -> NSColor {
        isDark ? NSColor(srgbRed: 0.066, green: 0.088, blue: 0.115, alpha: 1)
               : NSColor(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 1)
    }

    /// LLMBar's small accents, for details only (labels, marks, counters) — never large fills.
    static var yellow: Color { c((0.70, 0.56, 0.0), (1.0, 0.94, 0.0)) }
    static var magenta: Color { c((0.82, 0.08, 0.40), (1.0, 0.30, 0.62)) }
    static var phosphor: Color { c((0.0, 0.56, 0.45), (0.30, 0.95, 0.76)) }

    static func windowNS(isDark: Bool) -> NSColor {
        isDark ? NSColor(srgbRed: 0.055, green: 0.075, blue: 0.100, alpha: 1)
               : NSColor(srgbRed: 0.965, green: 0.972, blue: 0.980, alpha: 1)
    }
}


/// A small mono caption with a coloured bar: LLMBar's way of naming a section.
struct AccentCaption: View {
    let text: String
    var color: Color = Palette.accent

    var body: some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 1).fill(color).frame(width: 2.5, height: 11)
            Text(text.uppercased()).font(.system(size: 10, weight: .semibold, design: .monospaced)).tracking(1.6)
                .foregroundColor(color)
        }
    }
}

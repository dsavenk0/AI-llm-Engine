// Which system voice reads a reply aloud — shared by LLM Voice (which speaks) and LLMBar's settings (where the
// person chooses). A Mac ships compact voices that sound robotic; the Enhanced and Premium ones (a free download in
// System Settings ▸ Accessibility ▸ Spoken Content) sound natural. Without a choice the best installed voice of
// the reply's language is used, so downloading a better one improves the voice with nothing else to change.
import AVFoundation
import AppKit

enum SpeechVoices {
    /// The voices of one language (two-letter code), best first: Premium, Enhanced, then compact.
    static func installed(_ lang: String) -> [AVSpeechSynthesisVoice] {
        let code = String(lang.prefix(2)).lowercased()
        return AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.lowercased().hasPrefix(code) && !novelty.contains($0.name) }
            .sorted { rank($0) != rank($1) ? rank($0) > rank($1) : $0.name < $1.name }
    }

    /// The voice for `lang`: the person's choice when it is installed, else the best installed one.
    static func voice(for lang: String) -> AVSpeechSynthesisVoice? {
        let all = installed(lang)
        if let id = UserDefaults.standard.string(forKey: key(lang)), let chosen = all.first(where: { $0.identifier == id }) {
            return chosen
        }
        return all.first ?? AVSpeechSynthesisVoice(language: lang)
    }

    static func choose(_ identifier: String?, for lang: String) {
        if let identifier { UserDefaults.standard.set(identifier, forKey: key(lang)) }
        else { UserDefaults.standard.removeObject(forKey: key(lang)) }
    }

    static func chosen(_ lang: String) -> String? { UserDefaults.standard.string(forKey: key(lang)) }

    static func quality(_ v: AVSpeechSynthesisVoice, russian: Bool) -> String {
        switch v.quality {
        case .premium: return russian ? "премиум" : "premium"
        case .enhanced: return russian ? "улучшенный" : "enhanced"
        default: return russian ? "компактный" : "compact"
        }
    }

    /// Opens the system pane where Enhanced / Premium voices are downloaded.
    static func openDownloads() {
        let panes = ["x-apple.systempreferences:com.apple.Accessibility-Settings.extension?SpokenContent",
                     "x-apple.systempreferences:com.apple.preference.universalaccess?SpokenContent"]
        for p in panes { if let u = URL(string: p), NSWorkspace.shared.open(u) { return } }
    }

    private static func key(_ lang: String) -> String { "speechVoice." + String(lang.prefix(2)).lowercased() }

    private static func rank(_ v: AVSpeechSynthesisVoice) -> Int {
        switch v.quality { case .premium: return 2; case .enhanced: return 1; default: return 0 }
    }

    // macOS's joke voices (Bells, Bubbles, Zarvox …) are never what a reply should sound like
    private static let novelty: Set<String> = ["Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos",
                                               "Good News", "Jester", "Organ", "Superstar", "Trinoids", "Whisper",
                                               "Wobble", "Zarvox", "Deranged", "Hysterical", "Pipe Organ"]
}

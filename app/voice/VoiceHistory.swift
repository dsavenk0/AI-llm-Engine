// The voice chat's saved conversations: pure data and file handling (Foundation only, so tests/swift can run it).
import Foundation

/// One line of a saved conversation. Pictures are not kept (a data: URL is megabytes); their names are.
struct StoredTurn: Codable, Equatable {
    var role: String
    var text: String
    var files: [String] = []
    var fileText = ""
    var spoken = false
}

struct VoiceChat: Codable, Identifiable, Equatable {
    var id = UUID()
    var title = ""
    var custom = false              // the person named it: the first message no longer renames it
    var updated = Date()
    var turns: [StoredTurn] = []
}

struct VoiceHistory: Codable, Equatable {
    var chats: [VoiceChat] = []
    static let keep = 60            // the oldest go when there are more

    /// The first thing the person said, trimmed to a short line.
    static func title(for turns: [StoredTurn]) -> String {
        guard let first = turns.first(where: { $0.role == "user" }) else { return "New chat" }
        let line = first.text.split(whereSeparator: \.isNewline).first.map(String.init) ?? first.text
        let t = line.trimmingCharacters(in: .whitespaces)
        if t.isEmpty { return first.files.first ?? "New chat" }
        return t.count > 48 ? String(t.prefix(47)) + "…" : t
    }

    /// Saves `turns` as chat `id` (created on the first call). An empty conversation is not worth a row.
    mutating func upsert(id: UUID, turns: [StoredTurn], now: Date = Date()) {
        guard !turns.isEmpty else { return }
        if let i = chats.firstIndex(where: { $0.id == id }) {
            chats[i].turns = turns
            chats[i].updated = now
            if !chats[i].custom { chats[i].title = VoiceHistory.title(for: turns) }
        } else {
            chats.append(VoiceChat(id: id, title: VoiceHistory.title(for: turns), custom: false, updated: now, turns: turns))
        }
        if chats.count > VoiceHistory.keep {
            chats = Array(chats.sorted { $0.updated > $1.updated }.prefix(VoiceHistory.keep))
        }
    }

    mutating func rename(_ id: UUID, to title: String) {
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, let i = chats.firstIndex(where: { $0.id == id }) else { return }
        chats[i].title = String(t.prefix(80))
        chats[i].custom = true
    }

    mutating func delete(_ id: UUID) { chats.removeAll { $0.id == id } }

    /// Newest first.
    var sorted: [VoiceChat] { chats.sorted { $0.updated > $1.updated } }

    func chat(_ id: UUID?) -> VoiceChat? { id.flatMap { i in chats.first { $0.id == i } } }

    // MARK: file

    static func load(from url: URL) -> VoiceHistory {
        guard let data = try? Data(contentsOf: url) else { return VoiceHistory() }
        return (try? JSONDecoder().decode(VoiceHistory.self, from: data)) ?? VoiceHistory()
    }

    func save(to url: URL) {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(self) { try? data.write(to: url, options: .atomic) }
    }
}

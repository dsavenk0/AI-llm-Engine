// Knowledge, the pure part: what `./llm knowledge … --json` answers, decoded, and the small decisions made on it.
// Foundation only, so tests/swift compiles and checks it on its own. The index itself is Python
// (engine/knowledge.py); the app never opens the database — it asks the CLI, which runs, answers and exits.
import Foundation

struct KnowledgeLibrary: Decodable, Identifiable, Equatable, Hashable {
    let id: Int
    let name: String
    let documents: Int
    let ready: Int
}

struct KnowledgeDocument: Decodable, Identifiable, Equatable {
    let id: Int
    let name: String
    let path: String
    let kind: String
    let state: String            // indexing · ready · failed
    let pages: Int
    let anchored: Bool
    let note: String
    let chunks: Int
    let changed: Bool
    let missing: Bool

    /// One line under the name: what is there, and anything the person should know.
    var detail: String {
        var parts: [String] = []
        if state == "ready" {
            parts.append(pages > 0 ? "\(pages) p. · \(chunks) chunks" : "\(chunks) chunks")
        } else {
            parts.append(state)
        }
        if missing { parts.append("original is missing") } else if changed { parts.append("changed since indexing") }
        if !note.isEmpty { parts.append(note) }
        return parts.joined(separator: " · ")
    }
}

/// One result of `add`/`attach`.
struct KnowledgeAddResult: Decodable, Equatable {
    let name: String
    let state: String            // ready · failed · skipped · removed
    let note: String
    let reused: Bool?
}

struct KnowledgeHit: Decodable, Identifiable, Equatable {
    let chunk: Int
    let name: String
    let anchor: String
    let text: String
    let score: Double
    var id: Int { chunk }
    var label: String { anchor.isEmpty ? name : "\(name), \(anchor)" }
}

/// The prompt block (`search … --context --json`): the text to add and the sources it cites.
struct KnowledgeContext: Decodable, Equatable {
    struct Source: Decodable, Equatable {
        let n: Int
        let name: String
        let anchor: String
        var label: String { anchor.isEmpty ? name : "\(name), \(anchor)" }
    }
    let text: String
    let sources: [Source]
}

enum KnowledgeJSON {
    /// The CLI prints progress lines before the JSON when asked to; the JSON is the last top-level value.
    /// `null` (nothing found) and `{"error": …}` decode as nil.
    static func decode<T: Decodable>(_ type: T.Type, from output: String) -> T? {
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        var candidates = [trimmed]
        if let i = startOfLastValue(trimmed), i != trimmed.startIndex { candidates.append(String(trimmed[i...])) }
        for c in candidates {
            if let data = c.data(using: .utf8), let v = try? JSONDecoder().decode(T.self, from: data) { return v }
        }
        return nil
    }

    /// The error text of `{"error": "…"}`, if that is what the CLI answered.
    static func error(in output: String) -> String? {
        guard let data = output.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj["error"] as? String
    }

    private static func startOfLastValue(_ s: String) -> String.Index? {
        // the last line that opens a JSON array or object at column 0
        var idx: String.Index?
        var lineStart = s.startIndex
        for i in s.indices {
            if i == lineStart, s[i] == "[" || s[i] == "{" { idx = i }
            if s[i] == "\n" { lineStart = s.index(after: i) }
        }
        return idx
    }
}

/// What a chat has chosen: persistent libraries (ids) and whether it has documents of its own. Both empty means
/// Knowledge is off for it — nothing is sent, nothing is searched.
struct KnowledgeChoice: Equatable {
    var libraries: [Int] = []
    var ownDocuments = false

    var isOff: Bool { libraries.isEmpty && !ownDocuments }

    mutating func toggle(_ id: Int) {
        if let i = libraries.firstIndex(of: id) { libraries.remove(at: i) } else { libraries.append(id) }
    }

    /// Libraries that no longer exist are dropped (deleted elsewhere), so a stale choice cannot linger.
    mutating func keepOnly(existing: [KnowledgeLibrary]) {
        let ids = Set(existing.map(\.id))
        libraries.removeAll { !ids.contains($0) }
    }

    /// The `knowledge` field of an editor request, or nil when Knowledge is off for this chat.
    func requestField(chatID: String) -> [String: Any]? {
        if isOff { return nil }
        var f: [String: Any] = ["libraries": libraries]
        if ownDocuments { f["chat"] = chatID }
        return f
    }

    /// The compact label next to the books icon: nothing when off, else how many sources are on.
    func badge(libraryNames: [Int: String]) -> String {
        if isOff { return "" }
        let names = libraries.compactMap { libraryNames[$0] }
        if names.count == 1, !ownDocuments { return names[0] }
        return "\(names.count + (ownDocuments ? 1 : 0))"
    }
}

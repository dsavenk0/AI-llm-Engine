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

// MARK: - what the model is sent, and when the microphone opens again (pure, tested)

/// The conversation as the API takes it. The whole spoken or typed exchange so far goes along, not just the
/// last question: oldest turns are dropped first (never the newest), the window starts on a user turn, empty
/// turns are skipped, and two turns of the same role in a row (a question asked while the answer was cut off)
/// are joined — chat templates reject or mangle those.
enum VoiceContext {
    static func window(_ turns: [(role: String, text: String)], maxTurns: Int = 24, maxChars: Int = 16000)
        -> [(role: String, text: String)] {
        var out: [(role: String, text: String)] = []
        for t in turns where !t.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if let last = out.last, last.role == t.role {
                out[out.count - 1].text += "\n" + t.text
            } else {
                out.append(t)
            }
        }
        while out.count > maxTurns { out.removeFirst() }
        while out.count > 1, out.reduce(0, { $0 + $1.text.count }) > maxChars { out.removeFirst() }
        while let first = out.first, first.role != "user", out.count > 1 { out.removeFirst() }
        return out
    }
}

enum DialogPolicy {
    /// Opens the microphone again after an answer: only in a running dialog, when the person did not stop it,
    /// and never from a failure.
    static func shouldResume(dialogMode: Bool, dialogActive: Bool, failed: Bool) -> Bool {
        dialogMode && dialogActive && !failed
    }
}

// MARK: - looking things up on the web (pure, tested)

enum WebLookup {
    struct Hit: Equatable { var title: String, url: String, snippet: String }
    struct Page: Equatable { var url: String, text: String }
    struct Found: Equatable { var hits: [Hit], pages: [Page], error: String? }

    /// What the model answered when asked whether a lookup is needed: the search query, or nil for "no".
    static func query(from reply: String) -> String? {
        guard var line = reply.split(whereSeparator: \.isNewline).map({ $0.trimmingCharacters(in: .whitespaces) })
            .first(where: { !$0.isEmpty }) else { return nil }
        let junk = CharacterSet(charactersIn: "\"'`«»“”*").union(.whitespaces)
        line = line.trimmingCharacters(in: junk)
        for prefix in ["search query:", "search:", "query:", "поиск:", "запрос:"] where line.lowercased().hasPrefix(prefix) {
            line = String(line.dropFirst(prefix.count))
            break
        }
        line = line.trimmingCharacters(in: junk)
        let lower = line.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ".!"))
        let words = lower.split(separator: " ")
        if lower.isEmpty || ["no", "none", "n/a", "нет", "не нужно", "no search", "no search needed"].contains(lower) { return nil }
        if words.first == "no" && words.count <= 3 { return nil }
        if line.count > 120 { return nil }                  // it started answering instead of searching
        return line
    }

    private static let freshWords: [String] = [
        // English
        "today", "tonight", "tomorrow", "yesterday", "right now", "currently", "current ", "latest", "newest", "recent",
        "news", "weather", "forecast", "temperature in", "price of", "prices", "stock", "exchange rate", "score",
        "who won", "winner", "standings", "schedule", "release date", "when does", "when is the next", "this week",
        "this month", "this year", "breaking", "trending", "live ", "who is the president", "who is the prime minister",
        "who is the ceo", "who is the mayor", "how much is", "how much does",
        // Russian
        "сегодня", "сейчас", "завтра", "вчера", "послед", "новост", "погод", "прогноз", "курс ", "цена", "цены",
        "стоимость", "сколько стоит", "кто выиграл", "счёт", "счет матча", "результат", "расписание", "когда выйдет",
        "в этом году", "на этой неделе", "текущ", "актуальн", "сегодняшн", "кто президент", "кто премьер",
        // German, French, Spanish
        "heute", "aktuell", "wetter", "nachrichten", "preis", "aujourd'hui", "actuel", "météo", "actualités", "prix",
        "hoy", "actual", "noticias", "clima", "precio", "mañana",
    ]

    /// Does the question ask for something that changes (news, weather, prices, scores, "today", this or last
    /// year)? A small local model is unreliable at deciding this alone, so these signs decide too.
    static func needsFresh(_ question: String, year: Int = Calendar.current.component(.year, from: Date())) -> Bool {
        let q = " " + question.lowercased() + " "
        if freshWords.contains(where: { q.contains($0) }) { return true }
        for y in (year - 1)...(year + 1) where q.contains(String(y)) { return true }
        return false
    }

    /// The JSON line `llm web search` prints.
    static func parse(_ json: String) -> Found? {
        guard let line = json.split(separator: "\n").last(where: { $0.hasPrefix("{") }),
              let obj = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else { return nil }
        let hits = (obj["results"] as? [[String: Any]] ?? []).compactMap { r -> Hit? in
            guard let url = r["url"] as? String else { return nil }
            return Hit(title: r["title"] as? String ?? "", url: url, snippet: r["snippet"] as? String ?? "")
        }
        let pages = (obj["pages"] as? [[String: Any]] ?? []).compactMap { p -> Page? in
            guard let url = p["url"] as? String, let text = p["text"] as? String else { return nil }
            return Page(url: url, text: text)
        }
        return Found(hits: hits, pages: pages, error: obj["error"] as? String)
    }

    /// Text appended to the system prompt: the results, or an honest note that the lookup failed.
    static func context(query: String, found: Found?, maxChars: Int = 5000) -> String {
        guard let f = found, f.error == nil, !f.hits.isEmpty else {
            let why = found?.error.map { " (\($0))" } ?? ""
            return "A web search for \"\(query)\" was tried but gave nothing\(why). Say briefly that you could not "
                + "check the internet; answer from memory only if you are sure, and say that it may be out of date."
        }
        var out = "Web search results for \"\(query)\", fetched just now. They are more current than your memory: use "
            + "them to answer. Name the source in a few words when you rely on one. If they do not contain the "
            + "answer, say so.\n"
        for (i, h) in f.hits.enumerated() {
            out += "\n\(i + 1). \(h.title) — \(h.snippet) [\(h.url)]"
        }
        for p in f.pages {
            out += "\n\nPage \(p.url):\n\(p.text)"
        }
        return out.count > maxChars ? String(out.prefix(maxChars)) + " …" : out
    }
}

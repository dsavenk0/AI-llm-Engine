// Pure logic of the editor app — Foundation only, no UI, so it compiles and is tested on its own
// (tests/swift/main.swift, run by tests/test_swift_logic.py).
import Foundation

// MARK: - agent events (the backend streams one JSON object per SSE line)

enum AgentEvent: Equatable {
    case run(id: String)
    case delta(String)
    case reasoning(String)
    case tool(id: String, name: String, category: String, args: [String: String])
    case approval(id: String, kind: String, path: String?, old: String?, new: String?, command: String?, dangerous: String?)
    case question(id: String, question: String, options: [String])
    case toolResult(id: String, name: String, ok: Bool, summary: String)
    case done
    case error(String)

    /// One `data: {...}` line → event. Anything else (comments, [DONE], malformed JSON) → nil.
    static func parse(line: String) -> AgentEvent? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("data:") else { return nil }
        let payload = trimmed.dropFirst(5).trimmingCharacters(in: .whitespaces)
        guard payload != "[DONE]", let data = payload.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = obj["type"] as? String else { return nil }
        func s(_ k: String) -> String? { obj[k] as? String }
        switch type {
        case "run": return s("id").map { .run(id: $0) }
        case "delta": return s("text").map { .delta($0) }
        case "reasoning": return s("text").map { .reasoning($0) }
        case "tool":
            let raw = obj["args"] as? [String: Any] ?? [:]
            return .tool(id: s("id") ?? "", name: s("name") ?? "?", category: s("category") ?? "other",
                         args: raw.mapValues { "\($0)" })
        case "approval":
            return .approval(id: s("id") ?? "", kind: s("kind") ?? "edit", path: s("path"), old: s("old"),
                             new: s("new"), command: s("command"),
                             dangerous: (obj["dangerous"] as? String).flatMap { $0.isEmpty ? nil : $0 })
        case "question":
            return .question(id: s("id") ?? "", question: s("question") ?? "", options: obj["options"] as? [String] ?? [])
        case "tool_result":
            return .toolResult(id: s("id") ?? "", name: s("name") ?? "", ok: obj["ok"] as? Bool ?? true,
                               summary: s("summary") ?? "")
        case "done": return .done
        case "error": return .error(s("message") ?? "error")
        default: return nil
        }
    }
}

// MARK: - "Explored 3 searches, read 2 files, ran 1 command"

struct ActivityCounts: Equatable {
    var searches = 0, explores = 0, reads = 0, edits = 0, commands = 0, questions = 0

    mutating func add(category: String) {
        switch category {
        case "search": searches += 1
        case "explore": explores += 1
        case "read": reads += 1
        case "edit": edits += 1
        case "command": commands += 1
        case "question": questions += 1
        default: break
        }
    }

    var isEmpty: Bool { self == ActivityCounts() }

    /// English summary; the UI localises around it.
    func summary() -> String {
        func n(_ c: Int, _ one: String, _ many: String) -> String? { c == 0 ? nil : "\(c) \(c == 1 ? one : many)" }
        let looked = [n(searches, "search", "searches"), n(explores, "folder", "folders"), n(reads, "file", "files")]
            .compactMap { $0 }
        var parts: [String] = []
        if !looked.isEmpty { parts.append("Explored " + looked.joined(separator: ", ")) }
        if let e = n(edits, "edit", "edits") { parts.append("made " + e) }
        if let c = n(commands, "command", "commands") { parts.append("ran " + c) }
        if parts.isEmpty { return "" }
        let text = parts.joined(separator: ", ")
        return text.prefix(1).uppercased() + text.dropFirst()
    }
}

// MARK: - markdown blocks (text / fenced code), with a file hint for "apply to file"

enum MDBlock: Equatable {
    case text(String)
    case code(lang: String, code: String, hint: String?)
}

enum Markdown {
    static func split(_ md: String) -> [MDBlock] {
        var blocks: [MDBlock] = []
        var text: [String] = []
        var lines = md.components(separatedBy: "\n")
        var i = 0
        func flush() {
            let joined = text.joined(separator: "\n").trimmingCharacters(in: .newlines)
            if !joined.isEmpty { blocks.append(.text(joined)) }
            text = []
        }
        while i < lines.count {
            let line = lines[i]
            if let fence = openingFence(line) {
                let before = text
                var body: [String] = []
                var j = i + 1
                while j < lines.count, !lines[j].hasPrefix(fence.marker) { body.append(lines[j]); j += 1 }
                // a path named on the line(s) just before the fence is the target file
                var hint: String? = nil
                for prev in before.suffix(2).reversed() { if let h = pathIn(prev) { hint = h; break } }
                if hint == nil, let first = body.first { hint = pathInComment(first) }
                flush()
                blocks.append(.code(lang: fence.lang, code: body.joined(separator: "\n"), hint: hint))
                i = j + 1
                continue
            }
            text.append(line)
            i += 1
        }
        flush()
        lines = []
        return blocks
    }

    private static func openingFence(_ line: String) -> (marker: String, lang: String)? {
        let t = line.trimmingCharacters(in: .whitespaces)
        guard t.hasPrefix("```") else { return nil }
        let marker = String(t.prefix(while: { $0 == "`" }))
        let lang = t.dropFirst(marker.count).trimmingCharacters(in: .whitespaces)
        guard !lang.contains(" ") && !lang.contains("`") else { return nil }
        return (marker, lang)
    }

    private static let pathRx = try! NSRegularExpression(pattern: #"`([\w./@+-]+\.\w+)`|^\**([\w./@+-]+\.\w+)\**:?\s*$"#)
    private static func pathIn(_ s: String) -> String? {
        let r = NSRange(s.startIndex..., in: s)
        guard let m = pathRx.firstMatch(in: s, range: r) else { return nil }
        for g in 1...2 where m.range(at: g).location != NSNotFound {
            return (s as NSString).substring(with: m.range(at: g))
        }
        return nil
    }
    private static let commentRx = try! NSRegularExpression(
        pattern: #"^(?://|#|--|/\*)\s*(?:file(?:name)?:\s*)?([\w./@+-]+\.\w+)\s*(?:\*/)?$"#, options: [.caseInsensitive])
    private static func pathInComment(_ s: String) -> String? {
        let t = s.trimmingCharacters(in: .whitespaces)
        guard let m = commentRx.firstMatch(in: t, range: NSRange(t.startIndex..., in: t)) else { return nil }
        return (t as NSString).substring(with: m.range(at: 1))
    }
}

// MARK: - line diff (common prefix/suffix trimmed, LCS in between)

struct DiffLine: Equatable {
    enum Kind: String { case same = " ", added = "+", removed = "-" }
    let kind: Kind
    let text: String
}

enum LineDiff {
    static func diff(old: String, new: String) -> [DiffLine] {
        let a = old.components(separatedBy: "\n"), b = new.components(separatedBy: "\n")
        var s = 0
        while s < a.count, s < b.count, a[s] == b[s] { s += 1 }
        var ea = a.count, eb = b.count
        while ea > s, eb > s, a[ea - 1] == b[eb - 1] { ea -= 1; eb -= 1 }
        let A = Array(a[s..<ea]), B = Array(b[s..<eb])
        var out = a[..<s].map { DiffLine(kind: .same, text: $0) }
        if A.count * B.count > 4_000_000 {
            out += A.map { DiffLine(kind: .removed, text: $0) } + B.map { DiffLine(kind: .added, text: $0) }
        } else {
            let n = A.count, m = B.count
            var L = [[UInt32]](repeating: [UInt32](repeating: 0, count: m + 1), count: n + 1)
            if n > 0 && m > 0 {
                for i in stride(from: n - 1, through: 0, by: -1) {
                    for j in stride(from: m - 1, through: 0, by: -1) {
                        L[i][j] = A[i] == B[j] ? L[i + 1][j + 1] + 1 : max(L[i + 1][j], L[i][j + 1])
                    }
                }
            }
            var i = 0, j = 0
            while i < n && j < m {
                if A[i] == B[j] { out.append(DiffLine(kind: .same, text: A[i])); i += 1; j += 1 }
                else if L[i + 1][j] >= L[i][j + 1] { out.append(DiffLine(kind: .removed, text: A[i])); i += 1 }
                else { out.append(DiffLine(kind: .added, text: B[j])); j += 1 }
            }
            while i < n { out.append(DiffLine(kind: .removed, text: A[i])); i += 1 }
            while j < m { out.append(DiffLine(kind: .added, text: B[j])); j += 1 }
        }
        out += a[ea...].map { DiffLine(kind: .same, text: $0) }
        return out
    }

    static func stats(_ d: [DiffLine]) -> (added: Int, removed: Int) {
        (d.filter { $0.kind == .added }.count, d.filter { $0.kind == .removed }.count)
    }
}

// MARK: - the backend counts characters as Unicode code points, AppKit as UTF-16 units

struct OffsetMap {
    private let utf16At: [Int]          // code point index → UTF-16 offset; one extra entry for the end

    init(_ text: String) {
        var map = [Int](); map.reserveCapacity(text.unicodeScalars.count + 1)
        var u = 0
        for scalar in text.unicodeScalars { map.append(u); u += scalar.value > 0xFFFF ? 2 : 1 }
        map.append(u)
        utf16At = map
    }

    var scalarCount: Int { utf16At.count - 1 }

    /// NSRange for a [start, end) span of code points, clamped to the text.
    func range(start: Int, end: Int) -> NSRange? {
        guard start >= 0, start < end, start <= scalarCount else { return nil }
        let e = min(end, scalarCount)
        return NSRange(location: utf16At[start], length: utf16At[e] - utf16At[start])
    }

    /// Code point index → UTF-16 offset of the start of a 1-based line.
    func lineStart(_ line: Int, in text: String) -> Int {
        if line <= 1 { return 0 }
        var seen = 1, u = 0
        for scalar in text.unicodeScalars {
            u += scalar.value > 0xFFFF ? 2 : 1
            if scalar == "\n" { seen += 1; if seen == line { return u } }
        }
        return u
    }
}

// MARK: - small helpers

enum Paths {
    static func displayName(_ path: String) -> String { (path as NSString).lastPathComponent }
    static func isInside(_ path: String, root: String) -> Bool {
        let p = (path as NSString).standardizingPath, r = (root as NSString).standardizingPath
        return p == r || p.hasPrefix(r + "/")
    }
}

/// Reads the URL the backend prints once it listens: "  http://127.0.0.1:PORT/?token=TOKEN".
func parseBackendURL(_ line: String) -> (port: Int, token: String)? {
    guard let rx = try? NSRegularExpression(pattern: #"http://127\.0\.0\.1:(\d+)/\?token=([\w-]+)"#),
          let m = rx.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
          let p = Range(m.range(at: 1), in: line), let t = Range(m.range(at: 2), in: line),
          let port = Int(line[p]) else { return nil }
    return (port, String(line[t]))
}

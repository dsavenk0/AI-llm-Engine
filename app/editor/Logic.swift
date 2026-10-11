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

struct ActivityCounts: Equatable, Codable {
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

// MARK: - diagnostics (problems)

struct Diagnostic: Codable, Equatable, Identifiable {
    let line: Int, col: Int
    let end_col: Int?
    let severity: String, message: String, source: String
    var id: String { "\(line):\(col):\(message)" }
    var isError: Bool { severity == "error" }
}

extension OffsetMap {
    /// UTF-16 offset of (1-based line, 1-based column counted in code points); clamped to the line.
    static func position(line: Int, col: Int, in text: String) -> Int {
        var u = 0, curLine = 1, curCol = 1
        for scalar in text.unicodeScalars {
            if curLine == line && curCol == col { return u }
            if scalar == "\n" {
                if curLine == line { return u }                  // the column is past the end of its line
                curLine += 1; curCol = 1
            } else { curCol += 1 }
            u += scalar.value > 0xFFFF ? 2 : 1
        }
        return u
    }
}

// MARK: - a small terminal screen: UTF-8 text + the ANSI sequences ordinary tools emit

enum TermColor: Equatable {
    case index(Int)                       // 0-15 ANSI, 16-255 the xterm palette
    case rgb(UInt8, UInt8, UInt8)
}

struct TermStyle: Equatable {
    var fg: TermColor?
    var bg: TermColor?
    var bold = false, dim = false, italic = false, underline = false, inverse = false
}

struct TermCell: Equatable {
    var ch: Character
    var style: TermStyle
}

final class TermBuffer {
    private(set) var lines: [[TermCell]] = [[]]
    private(set) var row = 0
    private(set) var col = 0
    private var style = TermStyle()
    let maxLines: Int
    let screenRows: Int

    private enum State { case ground, esc, csi, osc, oscEsc, charset }
    private var state = State.ground
    private var params = ""

    init(maxLines: Int = 5000, screenRows: Int = 24) {
        self.maxLines = maxLines
        self.screenRows = screenRows
    }

    var text: String { lines.map { String($0.map(\.ch)) }.joined(separator: "\n") }
    func lineText(_ i: Int) -> String { i < lines.count ? String(lines[i].map(\.ch)) : "" }

    // MARK: feeding

    func feed(_ s: String) {
        for ch in s.unicodeScalars { step(ch) }
    }

    private func step(_ u: Unicode.Scalar) {
        switch state {
        case .ground:
            switch u {
            case "\u{1B}": state = .esc
            case "\n": lineFeed()
            case "\r": col = 0
            case "\u{08}": col = max(0, col - 1)
            case "\t": put(" "); while col % 8 != 0 { put(" ") }
            case "\u{07}", "\u{00}": break
            default:
                if u.value >= 0x20 && u.value != 0x7F { put(Character(u)) }
            }
        case .esc:
            switch u {
            case "[": state = .csi; params = ""
            case "]": state = .osc
            case "(", ")", "*", "+": state = .charset
            default: state = .ground                             // ESC =, ESC >, ESC 7/8 …: ignored
            }
        case .charset: state = .ground
        case .osc:
            if u == "\u{07}" { state = .ground } else if u == "\u{1B}" { state = .oscEsc }
        case .oscEsc: state = .ground
        case .csi:
            if (0x30...0x3F).contains(u.value) || u == " " { params.unicodeScalars.append(u); return }   // parameter bytes
            state = .ground
            if (0x40...0x7E).contains(u.value) { csi(final: Character(u)) }
        }
    }

    // MARK: operations

    private func ensure(row r: Int) { while lines.count <= r { lines.append([]) } }

    private func lineFeed() {
        row += 1
        ensure(row: row)
        if lines.count > maxLines {
            let drop = lines.count - maxLines
            lines.removeFirst(drop)
            row = max(0, row - drop)
        }
    }

    private func put(_ ch: Character) {
        ensure(row: row)
        let cell = TermCell(ch: ch, style: style)
        if col < lines[row].count { lines[row][col] = cell }
        else {
            while lines[row].count < col { lines[row].append(TermCell(ch: " ", style: TermStyle())) }
            lines[row].append(cell)
        }
        col += 1
    }

    private var screenTop: Int { max(0, lines.count - screenRows) }

    private func nums(_ defaults: Int = 0) -> [Int] {
        let clean = params.filter { $0.isNumber || $0 == ";" }
        if clean.isEmpty { return [defaults] }
        return clean.split(separator: ";", omittingEmptySubsequences: false).map { Int($0) ?? defaults }
    }

    private func csi(final f: Character) {
        if params.hasPrefix("?") || params.hasPrefix(">") || params.hasPrefix("=") { return }   // private modes
        let n = nums()
        switch f {
        case "m": sgr(n)
        case "K":
            ensure(row: row)
            switch n[0] {
            case 1:                                             // start of line up to AND INCLUDING the cursor
                for i in 0..<min(col + 1, lines[row].count) { lines[row][i] = TermCell(ch: " ", style: TermStyle()) }
            case 2: lines[row] = []
            default: if col < lines[row].count { lines[row].removeSubrange(col...) }
            }
        case "J":
            switch n[0] {
            case 0:
                ensure(row: row)
                if col < lines[row].count { lines[row].removeSubrange(col...) }
                if row + 1 < lines.count { lines.removeSubrange((row + 1)...) }
            case 1: for r in screenTop..<row where r < lines.count { lines[r] = [] }
            default:                                            // 2/3: clear the screen (scrollback kept for 2)
                if n[0] == 3 { lines = [[]]; row = 0 } else {
                    for r in screenTop..<lines.count { lines[r] = [] }
                    row = screenTop
                }
                col = 0
            }
        case "A": row = max(0, row - max(1, n[0]))
        case "B": row += max(1, n[0]); ensure(row: row)
        case "C": col += max(1, n[0])
        case "D": col = max(0, col - max(1, n[0]))
        case "E": row += max(1, n[0]); ensure(row: row); col = 0
        case "F": row = max(0, row - max(1, n[0])); col = 0
        case "G": col = max(0, max(1, n[0]) - 1)
        case "H", "f":
            let r = max(1, n[0]), c = n.count > 1 ? max(1, n[1]) : 1
            row = screenTop + r - 1
            ensure(row: row)
            col = c - 1
        case "P":                                               // delete characters
            ensure(row: row)
            if col < lines[row].count { lines[row].removeSubrange(col..<min(lines[row].count, col + max(1, n[0]))) }
        case "X":                                               // erase characters
            ensure(row: row)
            for i in col..<min(lines[row].count, col + max(1, n[0])) { lines[row][i] = TermCell(ch: " ", style: TermStyle()) }
        case "@":                                               // insert blanks
            ensure(row: row)
            if col <= lines[row].count {
                lines[row].insert(contentsOf: Array(repeating: TermCell(ch: " ", style: style), count: max(1, n[0])), at: col)
            }
        default: break
        }
    }

    private func sgr(_ n: [Int]) {
        var i = 0
        let list = n.isEmpty ? [0] : n
        while i < list.count {
            let c = list[i]
            switch c {
            case 0: style = TermStyle()
            case 1: style.bold = true
            case 2: style.dim = true
            case 3: style.italic = true
            case 4: style.underline = true
            case 7: style.inverse = true
            case 22: style.bold = false; style.dim = false
            case 23: style.italic = false
            case 24: style.underline = false
            case 27: style.inverse = false
            case 30...37: style.fg = .index(c - 30)
            case 39: style.fg = nil
            case 40...47: style.bg = .index(c - 40)
            case 49: style.bg = nil
            case 90...97: style.fg = .index(c - 90 + 8)
            case 100...107: style.bg = .index(c - 100 + 8)
            case 38, 48:
                var color: TermColor?
                if i + 2 < list.count, list[i + 1] == 5 { color = .index(max(0, min(255, list[i + 2]))); i += 2 }
                else if i + 4 < list.count, list[i + 1] == 2 {
                    color = .rgb(UInt8(max(0, min(255, list[i + 2]))), UInt8(max(0, min(255, list[i + 3]))),
                                 UInt8(max(0, min(255, list[i + 4])))); i += 4
                }
                if c == 38 { style.fg = color } else { style.bg = color }
            default: break
            }
            i += 1
        }
    }
}

extension TermColor {
    /// sRGB components 0…1 for the standard xterm palette.
    func components() -> (Double, Double, Double) {
        switch self {
        case .rgb(let r, let g, let b): return (Double(r) / 255, Double(g) / 255, Double(b) / 255)
        case .index(let i):
            let base: [(Double, Double, Double)] = [
                (0, 0, 0), (0.80, 0.19, 0.19), (0.05, 0.74, 0.47), (0.90, 0.90, 0.06),
                (0.14, 0.45, 0.78), (0.74, 0.25, 0.74), (0.07, 0.66, 0.80), (0.90, 0.90, 0.90),
                (0.40, 0.40, 0.40), (0.95, 0.30, 0.30), (0.14, 0.82, 0.55), (0.96, 0.96, 0.26),
                (0.23, 0.56, 0.92), (0.84, 0.44, 0.84), (0.16, 0.72, 0.86), (1, 1, 1)]
            if i < 16 { return base[max(0, i)] }
            if i < 232 {
                let v = i - 16
                let lv: (Int) -> Double = { $0 == 0 ? 0 : (Double($0) * 40 + 55) / 255 }
                return (lv(v / 36), lv((v / 6) % 6), lv(v % 6))
            }
            let g = Double((i - 232) * 10 + 8) / 255
            return (g, g, g)
        }
    }
}

/// Splits a byte stream into valid UTF-8 text, holding back an incomplete trailing sequence.
struct UTF8Stream {
    private var pending = Data()
    mutating func decode(_ chunk: Data) -> String {
        pending.append(chunk)
        var end = pending.count
        // an incomplete sequence at the end: up to 3 continuation bytes after a lead byte
        var back = 0
        while back < min(3, pending.count) {
            let b = pending[pending.count - 1 - back]
            if b & 0xC0 == 0x80 { back += 1; continue }
            let need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1
            if need > back + 1 { end = pending.count - 1 - back }
            break
        }
        let head = pending.prefix(end)
        pending = Data(pending.suffix(from: pending.startIndex + end))
        return String(decoding: head, as: UTF8.self)
    }
}

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
    case image(path: String, prompt: String)          // a picture the assistant generated, saved in the project
    case note(String)                                 // a remark for the person (not part of the conversation)
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
        case "image": return s("path").map { .image(path: $0, prompt: s("prompt") ?? "") }
        case "note": return s("text").map { .note($0) }
        case "done": return .done
        case "error": return .error(s("message") ?? "error")
        default: return nil
        }
    }
}

// MARK: - "Explored 3 searches, read 2 files, ran 1 command"

struct ActivityCounts: Equatable, Codable {
    var searches = 0, explores = 0, reads = 0, edits = 0, commands = 0, questions = 0, webs = 0, images = 0

    mutating func add(category: String) {
        switch category {
        case "search": searches += 1
        case "explore": explores += 1
        case "read": reads += 1
        case "edit": edits += 1
        case "command": commands += 1
        case "question": questions += 1
        case "web": webs += 1
        case "image": images += 1
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
        if let w = n(webs, "web request", "web requests") { parts.append("made " + w) }
        if let i = n(images, "image", "images") { parts.append("generated " + i) }
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

enum PathTools {
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
    private(set) var screenRows: Int
    private(set) var screenCols: Int

    /// Modes a program can switch with ESC [ ? … h / l; the view reads them.
    private(set) var cursorVisible = true
    private(set) var applicationCursorKeys = false      // ?1: arrows send ESC O A instead of ESC [ A
    private(set) var bracketedPaste = false             // ?2004
    /// Full-screen programs (vim, htop, less) switch to a second screen and back (?1049 / ?47 / ?1047).
    private(set) var alternateScreen = false
    private var mainLines: [[TermCell]] = [[]]
    private var mainRow = 0, mainCol = 0
    private var scrollTop = 0, scrollBottom = 0          // the scroll region, inclusive (full screen by default)
    private var savedCursor: (row: Int, col: Int, style: TermStyle)?
    private var g0Graphics = false, g1Graphics = false, shifted = false   // DEC line drawing (ESC ( 0)

    private enum State { case ground, esc, csi, osc, oscEsc, charset }
    private var state = State.ground
    private var params = ""
    private var charsetSlot: Character = "("

    init(maxLines: Int = 5000, screenRows: Int = 24, screenCols: Int = 110) {
        self.maxLines = maxLines
        self.screenRows = screenRows
        self.screenCols = screenCols
        scrollBottom = screenRows - 1
    }

    var text: String { lines.map { String($0.map(\.ch)) }.joined(separator: "\n") }
    func lineText(_ i: Int) -> String { i < lines.count ? String(lines[i].map(\.ch)) : "" }

    /// The window changed size: the screen model follows (a full-screen program is told by the pty itself).
    func resize(rows: Int, cols: Int) {
        let r = max(2, rows), c = max(10, cols)
        screenRows = r
        screenCols = c
        if alternateScreen {
            if lines.count > r { lines.removeFirst(lines.count - r) }
            while lines.count < r { lines.append([]) }
            for k in 0..<lines.count where lines[k].count > c { lines[k].removeSubrange(c...) }
            row = min(row, r - 1)
            col = min(col, c - 1)
        }
        scrollTop = 0
        scrollBottom = r - 1
    }

    // MARK: feeding

    func feed(_ s: String) {
        for ch in s.unicodeScalars { step(ch) }
    }

    private static let lineDrawing: [Character: Character] = [
        "q": "─", "x": "│", "l": "┌", "k": "┐", "m": "└", "j": "┘", "t": "├", "u": "┤", "v": "┴", "w": "┬",
        "n": "┼", "a": "▒", "`": "◆", "f": "°", "g": "±", "o": "⎺", "s": "⎽", "~": "·", "y": "≤", "z": "≥",
        "{": "π", "|": "≠", "}": "£", "h": "▒", "p": "⎻", "r": "⎼", "0": "█"]

    private func step(_ u: Unicode.Scalar) {
        switch state {
        case .ground:
            switch u {
            case "\u{1B}": state = .esc
            case "\n", "\u{0B}", "\u{0C}": lineFeed()
            case "\r": col = 0
            case "\u{08}": col = max(0, col - 1)
            case "\t":
                if alternateScreen { col = min(screenCols - 1, (col / 8 + 1) * 8) }
                else { put(" "); while col % 8 != 0 { put(" ") } }
            case "\u{0E}": shifted = true                       // SO: the G1 set
            case "\u{0F}": shifted = false                      // SI: back to G0
            case "\u{07}", "\u{00}": break
            default:
                if u.value >= 0x20 && u.value != 0x7F {
                    var ch = Character(u)
                    if (shifted ? g1Graphics : g0Graphics), let g = TermBuffer.lineDrawing[ch] { ch = g }
                    put(ch)
                }
            }
        case .esc:
            state = .ground
            switch u {
            case "[": state = .csi; params = ""
            case "]": state = .osc
            case "(", ")", "*", "+": state = .charset; charsetSlot = Character(u)
            case "7": savedCursor = (row, col, style)
            case "8": restoreCursor()
            case "M": reverseIndex()
            case "D": lineFeed()
            case "E": col = 0; lineFeed()
            case "c": style = TermStyle(); g0Graphics = false; g1Graphics = false
            default: break                                       // ESC = / > …: ignored
            }
        case .charset:
            state = .ground
            let graphics = u == "0"
            if charsetSlot == "(" { g0Graphics = graphics } else if charsetSlot == ")" { g1Graphics = graphics }
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

    private func ensure(row r: Int) {
        if alternateScreen { return }                            // a fixed grid
        while lines.count <= r { lines.append([]) }
    }

    private func lineFeed() {
        if alternateScreen {
            if row == scrollBottom { scrollUp(1) } else if row < screenRows - 1 { row += 1 }
            return
        }
        row += 1
        ensure(row: row)
        if lines.count > maxLines {
            let drop = lines.count - maxLines
            lines.removeFirst(drop)
            row = max(0, row - drop)
        }
    }

    private func reverseIndex() {
        if alternateScreen {
            if row == scrollTop { scrollDown(1) } else if row > 0 { row -= 1 }
        } else { row = max(0, row - 1) }
    }

    private func scrollUp(_ n: Int) {
        guard alternateScreen, scrollTop < lines.count else { return }
        for _ in 0..<max(1, n) {
            lines.remove(at: scrollTop)
            lines.insert([], at: min(scrollBottom, lines.count))
        }
    }

    private func scrollDown(_ n: Int) {
        guard alternateScreen, scrollBottom < lines.count else { return }
        for _ in 0..<max(1, n) {
            lines.remove(at: scrollBottom)
            lines.insert([], at: scrollTop)
        }
    }

    private func restoreCursor() {
        guard let s = savedCursor else { return }
        row = alternateScreen ? min(s.row, screenRows - 1) : s.row
        col = s.col
        style = s.style
        ensure(row: row)
    }

    private func put(_ ch: Character) {
        if alternateScreen && col >= screenCols { col = 0; lineFeed() }        // autowrap
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

    private func enterAlternate(clear: Bool = true) {
        guard !alternateScreen else { return }
        mainLines = lines; mainRow = row; mainCol = col
        alternateScreen = true
        lines = Array(repeating: [], count: screenRows)
        row = 0; col = 0
        scrollTop = 0; scrollBottom = screenRows - 1
    }

    private func leaveAlternate() {
        guard alternateScreen else { return }
        alternateScreen = false
        lines = mainLines.isEmpty ? [[]] : mainLines
        row = min(mainRow, lines.count - 1)
        col = mainCol
        mainLines = [[]]
    }

    private func privateMode(_ on: Bool) {
        let list = params.dropFirst().split(separator: ";").compactMap { Int($0) }
        for m in list {
            switch m {
            case 1: applicationCursorKeys = on
            case 25: cursorVisible = on
            case 2004: bracketedPaste = on
            case 47, 1047, 1049: if on { enterAlternate() } else { leaveAlternate() }
            default: break
            }
        }
    }

    private func csi(final f: Character) {
        if params.hasPrefix("?") {
            if f == "h" { privateMode(true) } else if f == "l" { privateMode(false) }
            return
        }
        if params.hasPrefix(">") || params.hasPrefix("=") { return }
        let n = nums()
        let amount = max(1, n[0])
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
            if alternateScreen {
                switch n[0] {
                case 0:
                    if col < lines[row].count { lines[row].removeSubrange(col...) }
                    for r in (row + 1)..<max(row + 1, lines.count) { lines[r] = [] }
                case 1:
                    for r in 0..<row { lines[r] = [] }
                    for i in 0..<min(col + 1, lines[row].count) { lines[row][i] = TermCell(ch: " ", style: TermStyle()) }
                default:
                    for r in 0..<lines.count { lines[r] = [] }
                }
                return
            }
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
        case "A": row = max(alternateScreen ? 0 : 0, row - amount)
        case "B":
            row += amount
            if alternateScreen { row = min(row, screenRows - 1) } else { ensure(row: row) }
        case "C": col = min(alternateScreen ? screenCols - 1 : Int.max / 2, col + amount)
        case "D": col = max(0, col - amount)
        case "E": row += amount; if alternateScreen { row = min(row, screenRows - 1) } else { ensure(row: row) }; col = 0
        case "F": row = max(0, row - amount); col = 0
        case "G", "`": col = min(alternateScreen ? screenCols - 1 : Int.max / 2, max(0, amount - 1))
        case "d":                                               // line position absolute
            row = alternateScreen ? min(screenRows - 1, amount - 1) : screenTop + amount - 1
            ensure(row: row)
        case "H", "f":
            let r = max(1, n[0]), c = n.count > 1 ? max(1, n[1]) : 1
            if alternateScreen {
                row = min(screenRows - 1, r - 1)
                col = min(screenCols - 1, c - 1)
            } else {
                row = screenTop + r - 1
                ensure(row: row)
                col = c - 1
            }
        case "r":                                               // DECSTBM: the scroll region
            guard alternateScreen else { return }
            let top = max(1, n[0]), bottom = n.count > 1 && n[1] > 0 ? n[1] : screenRows
            if top < bottom && bottom <= screenRows { scrollTop = top - 1; scrollBottom = bottom - 1 }
            else { scrollTop = 0; scrollBottom = screenRows - 1 }
            row = 0; col = 0
        case "L":                                               // insert lines at the cursor, inside the region
            guard alternateScreen, row >= scrollTop, row <= scrollBottom else { return }
            for _ in 0..<min(amount, scrollBottom - row + 1) {
                lines.remove(at: scrollBottom)
                lines.insert([], at: row)
            }
            col = 0
        case "M":                                               // delete lines
            guard alternateScreen, row >= scrollTop, row <= scrollBottom else { return }
            for _ in 0..<min(amount, scrollBottom - row + 1) {
                lines.remove(at: row)
                lines.insert([], at: scrollBottom)
            }
            col = 0
        case "S": scrollUp(amount)
        case "T": scrollDown(amount)
        case "s": savedCursor = (row, col, style)
        case "u": restoreCursor()
        case "P":                                               // delete characters
            ensure(row: row)
            if col < lines[row].count { lines[row].removeSubrange(col..<min(lines[row].count, col + amount)) }
        case "X":                                               // erase characters
            ensure(row: row)
            for i in col..<max(col, min(lines[row].count, col + amount)) { lines[row][i] = TermCell(ch: " ", style: TermStyle()) }
        case "@":                                               // insert blanks
            ensure(row: row)
            if col <= lines[row].count {
                lines[row].insert(contentsOf: Array(repeating: TermCell(ch: " ", style: style), count: amount), at: col)
                if alternateScreen && lines[row].count > screenCols { lines[row].removeSubrange(screenCols...) }
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

// MARK: - extensions from Open VSX: a colour theme and snippets (declarative parts only)

/// A VS Code theme converted by the backend: background/foreground and a colour per token kind.
struct ThemeOverride: Codable, Equatable {
    var label: String
    var id: String
    var dark: Bool
    var background: String
    var foreground: String
    var kinds: [String: String]

    /// "#rrggbb" → 0…1 components; nil for anything else.
    static func rgb(_ hex: String) -> (Double, Double, Double)? {
        guard hex.count == 7, hex.hasPrefix("#"), let v = Int(hex.dropFirst(), radix: 16) else { return nil }
        return (Double((v >> 16) & 0xFF) / 255, Double((v >> 8) & 0xFF) / 255, Double(v & 0xFF) / 255)
    }
}

struct Snippet: Codable, Equatable, Hashable {
    let prefix: String
    let body: [String]
    let description: String
}

enum SnippetExpander {
    /// A VS Code snippet body → plain text plus the caret position (UTF-16 offset into the text): at the first
    /// tab stop ($1, ${1:default}), else at $0, else at the end. Variables ($TM_FILENAME…) expand to nothing —
    /// or to their default. `indent` is put in front of every line after the first.
    static func expand(_ body: [String], indent: String = "") -> (text: String, caret: Int) {
        let src = Array(body.joined(separator: "\n" + indent))
        var out = "", stops: [Int: Int] = [:], i = 0
        parse(src, &i, nil, &out, &stops, 0)
        let first = stops.keys.filter { $0 > 0 }.min()
        return (out, first.flatMap { stops[$0] } ?? stops[0] ?? out.utf16.count)
    }

    private static func parse(_ s: [Character], _ i: inout Int, _ close: Character?, _ out: inout String,
                              _ stops: inout [Int: Int], _ depth: Int) {
        func stop(_ n: Int) { if stops[n] == nil { stops[n] = out.utf16.count } }
        while i < s.count {
            let c = s[i]
            if let close, c == close { return }
            if c == "\\", i + 1 < s.count, "$}\\".contains(s[i + 1]) { out.append(s[i + 1]); i += 2; continue }
            guard c == "$", i + 1 < s.count else { out.append(c); i += 1; continue }
            let n = s[i + 1]
            if n.isASCII, n.isNumber {                                  // $1
                var j = i + 1, num = ""
                while j < s.count, s[j].isASCII, s[j].isNumber { num.append(s[j]); j += 1 }
                if let k = Int(num) { stop(k) }
                i = j
            } else if n == "{" {
                var j = i + 2, name = ""
                while j < s.count, s[j].isLetter || s[j].isNumber || s[j] == "_" { name.append(s[j]); j += 1 }
                if j < s.count, s[j] == "}" {                           // ${1} / ${VAR}
                    if let k = Int(name) { stop(k) }
                    i = j + 1
                } else if j < s.count, s[j] == ":" {                    // ${1:default}
                    if let k = Int(name) { stop(k) }
                    var k = j + 1
                    if depth < 8 { parse(s, &k, "}", &out, &stops, depth + 1) }
                    i = k < s.count ? k + 1 : k
                } else if j < s.count, s[j] == "|" {                    // ${1|one,two|}: the first choice
                    if let k = Int(name) { stop(k) }
                    j += 1
                    var first = "", seenComma = false
                    while j < s.count, s[j] != "|" {
                        if s[j] == "," { seenComma = true } else if !seenComma { first.append(s[j]) }
                        j += 1
                    }
                    out += first
                    while j < s.count, s[j] != "}" { j += 1 }
                    i = min(j + 1, s.count)
                } else { out.append(c); i += 1 }
            } else if n.isLetter || n == "_" {                          // $TM_FILENAME …
                var j = i + 1
                while j < s.count, s[j].isLetter || s[j].isNumber || s[j] == "_" { j += 1 }
                i = j
            } else { out.append(c); i += 1 }
        }
    }
}

enum SnippetMatch {
    private static func isWord(_ c: Character) -> Bool { c.isLetter || c.isNumber || c == "_" }

    /// The snippet (the longest prefix wins) that ends `lineBeforeCaret` as a whole word.
    static func find(in lineBeforeCaret: String, among snippets: [Snippet]) -> Snippet? {
        let chars = Array(lineBeforeCaret)
        var best: Snippet?
        for s in snippets where !s.prefix.isEmpty && lineBeforeCaret.hasSuffix(s.prefix) {
            let start = chars.count - s.prefix.count
            if start > 0, isWord(chars[start - 1]), let f = s.prefix.first, isWord(f) { continue }
            if best == nil || s.prefix.count > best!.prefix.count { best = s }
        }
        return best
    }
}

// MARK: - how much the assistant may do on its own, and how hard it thinks

enum AgentMode: String, CaseIterable, Codable, Identifiable {
    case manual, edit, plan, auto
    var id: String { rawValue }

    var title: String {
        switch self {
        case .manual: return "Manual"
        case .edit: return "Edit automatically"
        case .plan: return "Plan"
        case .auto: return "Auto"
        }
    }
    var short: String { self == .edit ? "Edit" : title }
    var detail: String {
        switch self {
        case .manual: return "Asks for approval before every edit and command."
        case .edit: return "Applies file edits right away; commands still ask."
        case .plan: return "Explores the project and presents a plan; changes nothing until you switch mode."
        case .auto: return "Runs edits and commands that pass a safety check; pauses for anything risky."
        }
    }
    var symbol: String {
        switch self {
        case .manual: return "hand.raised"
        case .edit: return "chevron.left.forwardslash.chevron.right"
        case .plan: return "list.bullet.clipboard"
        case .auto: return "bolt"
        }
    }
    /// Shift+Tab
    var next: AgentMode {
        let all = AgentMode.allCases
        return all[(all.firstIndex(of: self)! + 1) % all.count]
    }
    /// File edits are applied without asking.
    var autoEdit: Bool { self == .edit || self == .auto }
}

/// The Effort control: "off", then the model's levels (or a single "on" when it names none).
enum EffortScale {
    static func positions(levels: [String], supportsThinking: Bool) -> [String] {
        guard supportsThinking else { return [] }
        return ["off"] + (levels.isEmpty ? ["on"] : levels)
    }

    static func setting(for position: String) -> (think: Bool, effort: String?) {
        switch position {
        case "off": return (false, nil)
        case "on": return (true, nil)
        default: return (true, position)
        }
    }

    static func position(think: Bool, effort: String?, among positions: [String]) -> String {
        guard think else { return "off" }
        if let e = effort, positions.contains(e) { return e }
        return positions.contains("on") ? "on" : (positions.dropFirst().first ?? "on")
    }
}

// MARK: - how full the context window is

enum ContextMeter {
    /// The system prompt and the tool definitions every request carries.
    static let overhead = 1800

    /// A rough token count without a tokenizer: ~3.5 characters per token for Latin text and code, ~1.6 for
    /// Cyrillic, CJK and the like.
    static func tokens(of text: String) -> Int {
        var ascii = 0, other = 0
        for u in text.unicodeScalars { if u.isASCII { ascii += 1 } else { other += 1 } }
        return Int((Double(ascii) / 3.5 + Double(other) / 1.6).rounded(.up))
    }

    static func fraction(used: Int, limit: Int?) -> Double? {
        guard let l = limit, l > 0 else { return nil }
        return min(1.0, Double(used) / Double(l))
    }

    enum Level { case ok, warn, full }
    static func level(_ f: Double) -> Level { f >= 0.85 ? .full : (f >= 0.6 ? .warn : .ok) }

    /// "12.3k" / "840"
    static func short(_ n: Int) -> String {
        n >= 1000 ? String(format: "%.1fk", Double(n) / 1000) : "\(n)"
    }
}

// MARK: - several terminals in the bottom panel

/// The tabs of the integrated terminal: which exist, which is shown, and what they are called. Each tab keeps a
/// number that is never reused while the panel lives, so closing "zsh 2" and opening another does not make two
/// tabs with the same name, and a session can be found again by it.
struct TerminalTabs: Equatable {
    private(set) var serials: [Int] = [1]
    private(set) var active = 0
    private var next = 2

    var count: Int { serials.count }
    var activeSerial: Int { serials[active] }

    mutating func add() {
        serials.append(next)
        next += 1
        active = serials.count - 1
    }

    mutating func select(_ i: Int) {
        if serials.indices.contains(i) { active = i }
    }

    /// Closes tab `i`; false (nothing changes) for the last one — a panel always has a shell.
    @discardableResult
    mutating func close(_ i: Int) -> Bool {
        guard serials.count > 1, serials.indices.contains(i) else { return false }
        serials.remove(at: i)
        if active > i { active -= 1 } else if active >= serials.count { active = serials.count - 1 }
        return true
    }

    func title(_ i: Int, shell: String) -> String {
        let name = (shell as NSString).lastPathComponent
        guard serials.indices.contains(i) else { return name }
        return serials[i] == 1 ? name : "\(name) \(serials[i])"
    }
}

// MARK: - the command palette: fuzzy matching of what is typed against command names

enum PaletteMatch {
    /// nil when the typed letters do not occur in the title in that order; otherwise higher = better: letters
    /// that follow each other, letters that start a word, and shorter titles win.
    static func score(_ query: String, in title: String) -> Int? {
        let q = Array(query.lowercased().filter { !$0.isWhitespace })
        if q.isEmpty { return 0 }
        let t = Array(title.lowercased())
        var score = 0, ti = 0, last = -2
        for ch in q {
            var found = -1
            var j = ti
            while j < t.count { if t[j] == ch { found = j; break }; j += 1 }
            if found < 0 { return nil }
            score += 1
            if found == last + 1 { score += 8 }                                       // consecutive
            if found == 0 || " ›-_/".contains(t[found - 1]) { score += 10 }             // starts a word
            score -= min(3, found - ti)                                                // a small price for gaps
            last = found
            ti = found + 1
        }
        return score * 4 - min(t.count, 60) / 4
    }

    /// Indices of `titles` that match, best first (ties keep their order). An empty query keeps everything.
    static func rank(_ titles: [String], query: String) -> [Int] {
        var scored: [(Int, Int)] = []
        for (i, t) in titles.enumerated() { if let s = score(query, in: t) { scored.append((i, s)) } }
        if query.trimmingCharacters(in: .whitespaces).isEmpty { return scored.map { $0.0 } }
        return scored.sorted { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }.map { $0.0 }
    }
}

// MARK: - "check this code": when a message is about the file that is open

enum OpenFileIntent {
    private static let words = [
        "this code", "the code", "this file", "the file", "this function", "this class", "this method", "this test",
        "review", "check", "fix", "bug", "error", "refactor", "explain", "what does", "what is wrong", "why does",
        "optimi", "improve", "document", "add test", "write test", "typo", "lint", "cleanup", "clean up",
        "код", "файл", "проверь", "проверить", "исправ", "ошибк", "баг", "тест", "объясни", "рефактор", "что делает",
        "почему", "оптимиз", "улучш", "документ", "перепиши", "доработ", "найди", "посмотри", "глянь",
    ]

    /// Does the message talk about the open file without naming one? Then the file goes along with it (the way
    /// editors with an assistant do); a plain "hello" or a general question does not drag a file in.
    static func refers(to message: String) -> Bool {
        let m = " " + message.lowercased() + " "
        if m.contains("`") || m.contains(".swift") || m.contains(".py") || m.contains(".js") || m.contains(".ts") ||
            m.contains(".rs") || m.contains(".go") || m.contains(".md") { return true }
        return words.contains { m.contains($0) }
    }
}

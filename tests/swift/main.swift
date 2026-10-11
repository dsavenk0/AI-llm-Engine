// Tests for app/editor/Logic.swift. Compiled and run by tests/test_swift_logic.py:
//   swiftc app/editor/Logic.swift tests/swift/main.swift -o <bin> && <bin>
import Foundation

var failures = 0
var checks = 0
func check(_ cond: Bool, _ what: String, line: Int = #line) {
    checks += 1
    if !cond { failures += 1; print("FAIL (line \(line)): \(what)") }
}
func eq<T: Equatable>(_ a: T, _ b: T, _ what: String, line: Int = #line) {
    checks += 1
    if a != b { failures += 1; print("FAIL (line \(line)): \(what)\n   got:      \(a)\n   expected: \(b)") }
}

// ── events
eq(AgentEvent.parse(line: #"data: {"type":"run","id":"abc"}"#), .run(id: "abc"), "run")
eq(AgentEvent.parse(line: #"data: {"type":"delta","text":"Hel"}"#), .delta("Hel"), "delta")
eq(AgentEvent.parse(line: #"data: {"type":"reasoning","text":"hmm"}"#), .reasoning("hmm"), "reasoning")
eq(AgentEvent.parse(line: #"data: {"type":"tool","id":"c1","name":"search","category":"search","args":{"pattern":"def x","n":3}}"#),
   .tool(id: "c1", name: "search", category: "search", args: ["pattern": "def x", "n": "3"]), "tool with mixed arg types")
eq(AgentEvent.parse(line: #"data: {"type":"approval","id":"c2","kind":"edit","path":"a.py","old":"x","new":"y"}"#),
   .approval(id: "c2", kind: "edit", path: "a.py", old: "x", new: "y", command: nil, dangerous: nil), "edit approval")
eq(AgentEvent.parse(line: #"data: {"type":"approval","id":"c3","kind":"run","command":"git push","dangerous":"pushes code"}"#),
   .approval(id: "c3", kind: "run", path: nil, old: nil, new: nil, command: "git push", dangerous: "pushes code"), "dangerous run")
eq(AgentEvent.parse(line: #"data: {"type":"approval","id":"c4","kind":"run","command":"ls","dangerous":null}"#),
   .approval(id: "c4", kind: "run", path: nil, old: nil, new: nil, command: "ls", dangerous: nil), "null danger")
eq(AgentEvent.parse(line: #"data: {"type":"question","id":"q","question":"Which?","options":["a","b"]}"#),
   .question(id: "q", question: "Which?", options: ["a", "b"]), "question")
eq(AgentEvent.parse(line: #"data: {"type":"tool_result","id":"c1","name":"search","ok":false,"summary":"error: x"}"#),
   .toolResult(id: "c1", name: "search", ok: false, summary: "error: x"), "tool_result")
eq(AgentEvent.parse(line: #"data: {"type":"done"}"#), .done, "done")
eq(AgentEvent.parse(line: #"data: {"type":"error","message":"boom"}"#), .error("boom"), "error")
check(AgentEvent.parse(line: "data: [DONE]") == nil, "[DONE] is not an event")
check(AgentEvent.parse(line: ": keep-alive") == nil, "comment")
check(AgentEvent.parse(line: "data: {broken") == nil, "malformed JSON")
check(AgentEvent.parse(line: #"data: {"type":"mystery"}"#) == nil, "unknown type")
check(AgentEvent.parse(line: "") == nil, "empty line")

// ── activity summary
var c = ActivityCounts()
check(c.isEmpty && c.summary() == "", "empty counts")
["search", "search", "search", "read", "read", "command"].forEach { c.add(category: $0) }
eq(c.summary(), "Explored 3 searches, 2 files, ran 1 command", "mixed activity")
var one = ActivityCounts(); one.add(category: "read")
eq(one.summary(), "Explored 1 file", "singular")
var edits = ActivityCounts(); edits.add(category: "edit"); edits.add(category: "edit")
eq(edits.summary(), "Made 2 edits", "edits only")
var q = ActivityCounts(); q.add(category: "question"); q.add(category: "weird")
eq(q.summary(), "", "a question alone is not activity text")

// ── markdown blocks
let md = """
Here is the fix for `src/app.py`:

```python
def run():
    return 2
```

And a config:
src/config.json
```json
{"a": 1}
```
```swift
// File: App/Main.swift
let x = 1
```
tail text
"""
let blocks = Markdown.split(md)
eq(blocks.count, 6, "block count")
if blocks.count == 6 {
    eq(blocks[0], .text("Here is the fix for `src/app.py`:"), "intro text")
    eq(blocks[1], .code(lang: "python", code: "def run():\n    return 2", hint: "src/app.py"), "hint from a backticked path")
    eq(blocks[3], .code(lang: "json", code: "{\"a\": 1}", hint: "src/config.json"), "hint from a bare path line")
    eq(blocks[4], .code(lang: "swift", code: "// File: App/Main.swift\nlet x = 1", hint: "App/Main.swift"), "hint from a first-line comment")
    eq(blocks[5], .text("tail text"), "tail text")
}
eq(Markdown.split("no code at all"), [.text("no code at all")], "plain text")
eq(Markdown.split("```\nunterminated"), [.code(lang: "", code: "unterminated", hint: nil)], "an open fence while streaming")
eq(Markdown.split("````md\n```\ninner\n```\n````"), [.code(lang: "md", code: "```\ninner\n```", hint: nil)], "longer fence wraps a shorter one")
eq(Markdown.split(""), [], "empty")

// ── diff
let d1 = LineDiff.diff(old: "a\nb\nc", new: "a\nB\nc")
eq(d1.map { $0.kind.rawValue + $0.text }, [" a", "-b", "+B", " c"], "one line changed")
eq(LineDiff.stats(d1).added, 1, "stats added"); eq(LineDiff.stats(d1).removed, 1, "stats removed")
eq(LineDiff.diff(old: "x", new: "x").map { $0.kind }, [.same], "identical")
let d2 = LineDiff.diff(old: "1\n2\n3\n4", new: "1\n3\n4\n5")
eq(d2.map { $0.kind.rawValue + $0.text }, [" 1", "-2", " 3", " 4", "+5"], "delete and append")
eq(LineDiff.diff(old: "", new: "a\nb").map { $0.kind.rawValue + $0.text }, ["-", "+a", "+b"], "from empty")
let big = (0..<2500).map { "line \($0)" }.joined(separator: "\n")
let bigNew = big.replacingOccurrences(of: "line 1000", with: "CHANGED")
let d3 = LineDiff.diff(old: big, new: bigNew)
eq(LineDiff.stats(d3).added, 1, "large file: one added")
eq(LineDiff.stats(d3).removed, 1, "large file: one removed")

// ── code point → UTF-16 mapping
let emoji = "a😀b\nline2 é\nlast"
let map = OffsetMap(emoji)
eq(map.scalarCount, 16, "scalar count")                         // a 😀 b \n l i n e 2 ␠ é \n l a s t
eq(map.range(start: 0, end: 1), NSRange(location: 0, length: 1), "plain char")
eq(map.range(start: 1, end: 2), NSRange(location: 1, length: 2), "an emoji is two UTF-16 units")
eq(map.range(start: 2, end: 3), NSRange(location: 3, length: 1), "after the emoji everything shifts by one")
eq(map.range(start: 14, end: 99), NSRange(location: 15, length: 2), "end is clamped to the last two letters")
check(map.range(start: 5, end: 5) == nil && map.range(start: -1, end: 2) == nil && map.range(start: 99, end: 100) == nil,
      "empty, negative and out-of-range spans are rejected")
eq(map.lineStart(1, in: emoji), 0, "line 1"); eq(map.lineStart(2, in: emoji), 5, "line 2 starts after the emoji + newline")
eq((emoji as NSString).substring(from: map.lineStart(3, in: emoji)), "last", "line 3")
eq(map.lineStart(9, in: emoji), (emoji as NSString).length, "past the end")

// ── paths and backend url
check(Paths.isInside("/a/b/c.txt", root: "/a/b"), "inside")
check(!Paths.isInside("/a/bc/c.txt", root: "/a/b"), "sibling with a shared prefix is outside")
check(Paths.isInside("/a/b/../b/x", root: "/a/b"), "dots are standardised")
eq(Paths.displayName("src/ui/app.py"), "app.py", "display name")
let u = parseBackendURL("Editor UI for /tmp/x\n  http://127.0.0.1:53114/?token=abc_DEF-123\n")
check(u?.port == 53114 && u?.token == "abc_DEF-123", "backend url")
check(parseBackendURL("nothing here") == nil, "no url")


// ── diagnostics positions
let src = "let a = 1\nlet 😀b = (\nend"
eq(OffsetMap.position(line: 1, col: 1, in: src), 0, "line 1 col 1")
eq(OffsetMap.position(line: 2, col: 1, in: src), 10, "line 2 starts at 10")
eq(OffsetMap.position(line: 2, col: 5, in: src), 14, "column counted in code points before the emoji")
eq(OffsetMap.position(line: 2, col: 6, in: src), 16, "after the emoji: two UTF-16 units")
eq(OffsetMap.position(line: 2, col: 99, in: src), 21, "past the end of the line clamps to its end (the newline)")
eq(OffsetMap.position(line: 9, col: 1, in: src), (src as NSString).length, "past the end of the file")
let dj = try! JSONDecoder().decode(Diagnostic.self, from: #"{"line":3,"col":4,"end_col":6,"severity":"error","message":"bad","source":"python"}"#.data(using: .utf8)!)
check(dj.isError && dj.line == 3 && dj.end_col == 6, "diagnostic decodes")

// ── terminal buffer
func term(_ s: String, rows: Int = 24) -> TermBuffer { let t = TermBuffer(screenRows: rows); t.feed(s); return t }
eq(term("hello\r\nworld").text, "hello\nworld", "CRLF")
eq(term("abc\rX").text, "Xbc", "carriage return overwrites")
eq(term("10%\r50%\r100%").text, "100%", "a progress bar stays on one line")
eq(term("abc\u{08}\u{08}X").text, "aXc", "backspace then write")
eq(term("hello world\u{1B}[6D\u{1B}[K").text, "hello", "move left, erase to end of line")
eq(term("abcdef\u{1B}[3G\u{1B}[1K").text, "   def", "erase to start of line keeps the tail")
eq(term("one\r\ntwo\u{1B}[2K\r\nthree").text, "one\n\nthree", "erase whole line")
eq(term("a\tb").text, "a       b", "tab stops")
eq(term("x\u{1B}]0;window title\u{07}y").text, "xy", "OSC title is swallowed")
eq(term("x\u{1B}]8;;http://a\u{1B}\\link\u{1B}]8;;\u{1B}\\").text, "xlink", "OSC with ST terminator")
eq(term("a\u{1B}[?2004hb\u{1B}[?25lc").text, "abc", "private modes are ignored")
eq(term("\u{1B}(Bplain").text, "plain", "charset selection is swallowed")
let split = TermBuffer(); split.feed("a\u{1B}["); split.feed("31mred\u{1B}"); split.feed("[0m ok")
eq(split.text, "ared ok", "an escape sequence split across chunks")
eq(split.lines[0][1].style.fg, TermColor.index(1), "the split sequence still applied")
eq(split.lines[0][5].style, TermStyle(), "and reset restored the style")
let col = term("\u{1B}[1;31mERR\u{1B}[0m \u{1B}[38;5;208morange\u{1B}[0m \u{1B}[38;2;1;2;3mrgb\u{1B}[0m \u{1B}[42mbg\u{1B}[0m")
eq(col.lines[0][0].style, TermStyle(fg: .index(1), bold: true), "bold red")
eq(col.lines[0][4].style.fg, TermColor.index(208), "256-colour")
eq(col.lines[0][11].style.fg, TermColor.rgb(1, 2, 3), "true colour")
eq(col.lines[0][15].style.bg, TermColor.index(2), "background")
eq(term("\u{1B}[91mhi").lines[0][0].style.fg, TermColor.index(9), "bright foreground")
let clear = term("old line\r\nanother\u{1B}[2J\u{1B}[Hnew")
eq(clear.lineText(0), "new", "clear screen and home: output restarts at the top"); check(!clear.text.contains("old") && !clear.text.contains("another"), "the screen was really cleared")
let cup = term("\u{1B}[3;5Hx", rows: 5)
eq(cup.lineText(2), "    x", "cursor addressing")
let many = TermBuffer(maxLines: 100)
for i in 0..<500 { many.feed("line \(i)\r\n") }
check(many.lines.count <= 100 && many.lineText(many.lines.count - 2) == "line 499", "scrollback is capped, newest kept")
eq(term("日本語 ok ☃").text, "日本語 ok ☃", "non-ASCII text")
eq(term("a\u{1B}[3Pb").text, "ab", "delete characters")
eq(term("abc\r\u{1B}[2@XY").text, "XYabc", "insert blanks then overwrite")
let (r, g, b) = TermColor.index(232).components()
check(r == g && g == b && r < 0.1, "grayscale ramp")
check(TermColor.index(196).components().0 > 0.9 && TermColor.index(196).components().1 == 0, "cube red")

// ── UTF-8 across chunk boundaries
var u8 = UTF8Stream()
let bytes = Array("héllo 😀 ok".utf8)
var out = ""
for chunk in stride(from: 0, to: bytes.count, by: 2).map({ Data(bytes[$0..<min($0 + 2, bytes.count)]) }) { out += u8.decode(chunk) }
eq(out, "héllo 😀 ok", "UTF-8 split every two bytes decodes without replacement characters")

print("swift logic: \(checks - failures)/\(checks) checks passed")
exit(failures == 0 ? 0 : 1)

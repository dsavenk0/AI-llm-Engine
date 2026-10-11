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

print("swift logic: \(checks - failures)/\(checks) checks passed")
exit(failures == 0 ? 0 : 1)

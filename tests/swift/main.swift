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

var web = ActivityCounts(); web.add(category: "web"); web.add(category: "web"); web.add(category: "read")
eq(web.summary(), "Explored 1 file, made 2 web requests", "web requests are counted apart from files")

let img = AgentEvent.parse(line: #"data: {"type":"image","path":".llm/images/a.png","prompt":"a fox"}"#)
check(img == .image(path: ".llm/images/a.png", prompt: "a fox"), "image event")
check(AgentEvent.parse(line: #"data: {"type":"image"}"#) == nil, "image event without a path")
var pics = ActivityCounts(); pics.add(category: "image")
eq(pics.summary(), "Generated 1 image", "images are counted")

// ── modes and effort
eq(AgentMode.manual.next, .edit, "shift+tab goes to the next mode")
eq(AgentMode.auto.next, .manual, "…and wraps around")
check(AgentMode.edit.autoEdit && AgentMode.auto.autoEdit && !AgentMode.manual.autoEdit && !AgentMode.plan.autoEdit, "who applies edits unasked")
eq(EffortScale.positions(levels: [], supportsThinking: true), ["off", "on"], "no named levels: off / on")
eq(EffortScale.positions(levels: ["low", "medium", "high"], supportsThinking: true), ["off", "low", "medium", "high"], "named levels")
eq(EffortScale.positions(levels: ["low"], supportsThinking: false), [], "a model that does not reason has no control")
check(EffortScale.setting(for: "off") == (false, nil), "off")
check(EffortScale.setting(for: "high") == (true, "high"), "a level")
eq(EffortScale.position(think: true, effort: nil, among: ["off", "low", "medium"]), "low", "reasoning on without a level shows the first level")
eq(EffortScale.position(think: false, effort: "low", among: ["off", "low"]), "off", "thinking off wins")

// ── context meter
eq(ContextMeter.tokens(of: "abcdefg"), 2, "latin text: ~3.5 characters per token")
eq(ContextMeter.tokens(of: "привет"), 4, "cyrillic costs more tokens per character")
check(ContextMeter.fraction(used: 500, limit: 1000) == 0.5, "fraction")
check(ContextMeter.fraction(used: 5000, limit: 1000) == 1.0, "clamped at full")
check(ContextMeter.fraction(used: 5, limit: nil) == nil && ContextMeter.fraction(used: 5, limit: 0) == nil, "unknown limit")
check(ContextMeter.level(0.3) == .ok && ContextMeter.level(0.7) == .warn && ContextMeter.level(0.9) == .full, "levels")
eq(ContextMeter.short(12345), "12.3k", "short count")
eq(ContextMeter.short(840), "840", "small count")

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
check(PathTools.isInside("/a/b/c.txt", root: "/a/b"), "inside")
check(!PathTools.isInside("/a/bc/c.txt", root: "/a/b"), "sibling with a shared prefix is outside")
check(PathTools.isInside("/a/b/../b/x", root: "/a/b"), "dots are standardised")
eq(PathTools.displayName("src/ui/app.py"), "app.py", "display name")
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

// ── voice chat history (app/voice/VoiceHistory.swift)
func turn(_ role: String, _ text: String, files: [String] = []) -> StoredTurn { StoredTurn(role: role, text: text, files: files) }
eq(VoiceHistory.title(for: [turn("user", "What is the weather like in Lisbon in May? And what to pack?")]),
   "What is the weather like in Lisbon in May? And …", "a long first message is cut at 47 characters with an ellipsis")
eq(VoiceHistory.title(for: [turn("assistant", "hi"), turn("user", "  hello  \nsecond line")]), "hello", "the first user line, trimmed")
eq(VoiceHistory.title(for: [turn("user", "", files: ["notes.md"])]), "notes.md", "a file-only message is named by the file")
eq(VoiceHistory.title(for: []), "New chat", "no messages")
var hist = VoiceHistory()
let idA = UUID(), idB = UUID()
hist.upsert(id: idA, turns: [], now: Date(timeIntervalSince1970: 1))
eq(hist.chats.count, 0, "an empty conversation is not saved")
hist.upsert(id: idA, turns: [turn("user", "first")], now: Date(timeIntervalSince1970: 10))
hist.upsert(id: idB, turns: [turn("user", "second")], now: Date(timeIntervalSince1970: 20))
eq(hist.sorted.map(\.title), ["second", "first"], "newest first")
hist.upsert(id: idA, turns: [turn("user", "first"), turn("assistant", "ok")], now: Date(timeIntervalSince1970: 30))
eq(hist.sorted.first?.id, idA, "an updated chat moves to the top")
eq(hist.chat(idA)?.turns.count, 2, "its messages are replaced, not appended twice")
hist.rename(idA, to: "  My trip  ")
eq(hist.chat(idA)?.title, "My trip", "renamed and trimmed")
hist.upsert(id: idA, turns: [turn("user", "something else entirely")], now: Date(timeIntervalSince1970: 40))
eq(hist.chat(idA)?.title, "My trip", "a custom name survives later messages")
hist.rename(idA, to: "   ")
eq(hist.chat(idA)?.title, "My trip", "an empty name is ignored")
hist.delete(idB)
eq(hist.chats.map(\.id), [idA], "deleted")
for i in 0..<70 { hist.upsert(id: UUID(), turns: [turn("user", "m\(i)")], now: Date(timeIntervalSince1970: Double(100 + i))) }
eq(hist.chats.count, VoiceHistory.keep, "only the newest 60 are kept")
check(hist.chats.contains { $0.title == "m69" } && !hist.chats.contains { $0.title == "m0" }, "the oldest go first")
let file = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("voice-hist-\(UUID().uuidString)/chats.json")
hist.save(to: file)
eq(VoiceHistory.load(from: file), hist, "saved and loaded back unchanged")
eq(VoiceHistory.load(from: file.deletingLastPathComponent().appendingPathComponent("none.json")), VoiceHistory(), "a missing file is an empty history")
try? "{broken".write(to: file, atomically: true, encoding: .utf8)
eq(VoiceHistory.load(from: file), VoiceHistory(), "a damaged file is an empty history, not a crash")

// what the model is sent
func ctx(_ pairs: [(String, String)], turns: Int = 24, chars: Int = 16000) -> [String] {
    VoiceContext.window(pairs.map { (role: $0.0, text: $0.1) }, maxTurns: turns, maxChars: chars).map { "\($0.role):\($0.text)" }
}
eq(ctx([("user", "capital of France?"), ("assistant", "Paris."), ("user", "and its population?")]),
   ["user:capital of France?", "assistant:Paris.", "user:and its population?"], "the whole exchange goes along, newest last")
eq(ctx([("user", "a"), ("assistant", ""), ("user", "b")]), ["user:a\nb"], "an answer that never came: the two questions are one turn")
eq(ctx([("user", "  "), ("user", "hi"), ("assistant", "hello")]), ["user:hi", "assistant:hello"], "empty turns are skipped")
eq(ctx([("assistant", "stray"), ("user", "q")]), ["user:q"], "the window starts on a user turn")
let manyTurns = (0..<30).flatMap { [("user", "q\($0)"), ("assistant", "a\($0)")] }
let kept = ctx(manyTurns, turns: 6)
eq(kept.count, 6, "the newest turns are kept")
eq(kept.first, "user:q27", "…starting on a question")
eq(kept.last, "assistant:a29", "…and ending with the latest answer")
let long = ctx([("user", String(repeating: "x", count: 100)), ("assistant", String(repeating: "y", count: 100)), ("user", "now?")], chars: 150)
eq(long, ["user:now?"], "over the budget the oldest go first, the newest question always stays")
check(DialogPolicy.shouldResume(dialogMode: true, dialogActive: true, failed: false), "a running dialog goes on")
check(!DialogPolicy.shouldResume(dialogMode: false, dialogActive: true, failed: false), "dialog mode off: one question at a time")
check(!DialogPolicy.shouldResume(dialogMode: true, dialogActive: false, failed: false), "the person stopped it")
check(!DialogPolicy.shouldResume(dialogMode: true, dialogActive: true, failed: true), "never from a failure")

// web lookups
eq(WebLookup.query(from: "NO"), nil, "no lookup")
eq(WebLookup.query(from: "No."), nil, "no lookup, with a full stop")
eq(WebLookup.query(from: "no search needed"), nil, "no lookup, in words")
eq(WebLookup.query(from: "Нет"), nil, "no lookup, Russian")
eq(WebLookup.query(from: "weather Lisbon today"), "weather Lisbon today", "a query")
eq(WebLookup.query(from: "\"Search: Benfica score tonight\"\nbecause the user asks about a match"), "Benfica score tonight", "quotes, a prefix and a second line are dropped")
eq(WebLookup.query(from: "курс евро к доллару сегодня"), "курс евро к доллару сегодня", "a Russian query")
eq(WebLookup.query(from: String(repeating: "word ", count: 40)), nil, "an answer instead of a query is not searched")
eq(WebLookup.query(from: "   \n  "), nil, "an empty reply")
eq(WebLookup.query(from: "North Korea news"), "North Korea news", "a query that starts with 'no' letters is still a query")
let json = #"{"query":"q","results":[{"title":"T1","url":"https://a.example/x","snippet":"S1"},{"url":"https://b.example/y"},{"title":"bad"}],"pages":[{"url":"https://a.example/x","text":"page text"}]}"#
let found = WebLookup.parse("noise\n" + json)
eq(found?.hits.map(\.url), ["https://a.example/x", "https://b.example/y"], "hits without a url are dropped")
eq(found?.pages.first?.text, "page text", "pages")
eq(WebLookup.parse(#"{"query":"q","results":[],"pages":[],"error":"offline"}"#)?.error, "offline", "an error is carried")
eq(WebLookup.parse("not json"), nil, "garbage is not a result")
let ctxText = WebLookup.context(query: "q", found: found)
check(ctxText.contains("1. T1 — S1 [https://a.example/x]") && ctxText.contains("Page https://a.example/x:"), "results and pages are in the text")
check(WebLookup.context(query: "q", found: WebLookup.Found(hits: [], pages: [], error: "offline")).contains("could not check the internet"), "a failed lookup is said honestly")
check(WebLookup.context(query: "q", found: nil).contains("could not check the internet"), "no answer at all, the same")
check(WebLookup.context(query: "q", found: found, maxChars: 80).count <= 84, "the text is capped")

for q in ["What's the weather in Lisbon right now?", "What is the current price of bitcoin?", "Какая сегодня погода в Москве?",
          "Кто выиграл вчера матч Реал Мадрид?", "Who is the prime minister of the UK?", "Wetter in Berlin",
          "What happened in 2026 in space exploration?", "курс евро"] {
    check(WebLookup.needsFresh(q, year: 2026), "needs the web: \(q)")
}
for q in ["What is two plus two?", "Tell me a joke.", "Explain how photosynthesis works.", "How tall is Mount Everest?",
          "Write a haiku about the sea", "Why is the sky blue?", "Что такое фотосинтез?"] {
    check(!WebLookup.needsFresh(q, year: 2026), "general knowledge: \(q)")
}

// terminal tabs
var tabs = TerminalTabs()
eq(tabs.count, 1, "one terminal to begin with")
eq(tabs.title(0, shell: "/bin/zsh"), "zsh", "the first is just the shell's name")
tabs.add(); tabs.add()
eq(tabs.serials, [1, 2, 3], "numbers")
eq(tabs.active, 2, "a new terminal is shown")
eq(tabs.title(2, shell: "/bin/zsh"), "zsh 3", "later ones are numbered")
tabs.select(0)
eq(tabs.active, 0, "selected")
tabs.select(9)
eq(tabs.active, 0, "an unknown tab is ignored")
tabs.select(2)
check(tabs.close(1), "closing a middle tab works")
eq(tabs.serials, [1, 3], "the others stay")
eq(tabs.active, 1, "the shown tab is still the same terminal (serial 3)")
eq(tabs.activeSerial, 3, "…by its number")
check(tabs.close(1), "closing the shown last tab works")
eq(tabs.active, 0, "…and the one before it is shown")
check(!tabs.close(0), "the last terminal cannot be closed")
eq(tabs.count, 1, "…and stays")
tabs.add()
eq(tabs.serials, [1, 4], "numbers are not reused")
tabs.select(1)
check(tabs.close(0), "closing a tab before the shown one")
eq(tabs.activeSerial, 4, "keeps the shown terminal")
eq(tabs.active, 0, "…whose index moved")

// command palette
let cmds = ["File › New Chat", "File › New Window", "File › Open Folder…", "File › New Terminal", "View › Zoom In",
            "View › Zoom Out", "View › Reset Zoom", "View › Terminal", "View › Show / Hide Panel", "View › Modules › Preview"]
func top(_ q: String) -> String? { PaletteMatch.rank(cmds, query: q).first.map { cmds[$0] } }
eq(PaletteMatch.rank(cmds, query: "").count, cmds.count, "an empty query keeps everything")
eq(top("zoom in"), "View › Zoom In", "the words as typed")
eq(top("zi"), "View › Zoom In", "initials of the words")
eq(top("term"), "View › Terminal", "a prefix of a word beats a letter inside another")
eq(top("newt"), "File › New Terminal", "letters across words")
eq(top("rz"), "View › Reset Zoom", "initials of two words")
eq(PaletteMatch.rank(cmds, query: "qqq"), [], "nothing matches")
eq(top("PREV"), "View › Modules › Preview", "case does not matter")
check(PaletteMatch.score("zoom", in: "View › Zoom In")! > PaletteMatch.score("zoom", in: "View › Reset Zoom")!, "the word start scores higher")
eq(PaletteMatch.score("xyz", in: "abc"), nil, "no match is nil")
eq(PaletteMatch.rank(["b", "a"], query: "").map { $0 }, [0, 1], "stable for an empty query")

// full-screen programs: the alternate screen, scroll regions, line insertion, saved cursor, modes
func screen(_ s: String, rows: Int = 5, cols: Int = 12) -> TermBuffer {
    let t = TermBuffer(screenRows: rows, screenCols: cols); t.feed(s); return t
}
let ESC = "\u{1B}"
let fs_alt = screen("main1\r\nmain2\r\n\(ESC)[?1049h\(ESC)[HALT\(ESC)[3;1Hrow3")
check(fs_alt.alternateScreen, "?1049h switches to the alternate screen")
eq(fs_alt.lines.count, 5, "the alternate screen is a fixed grid of the window's rows")
eq(fs_alt.lineText(0), "ALT", "text at home")
eq(fs_alt.lineText(2), "row3", "cursor addressing on the grid")
fs_alt.feed("\(ESC)[?1049l")
check(!fs_alt.alternateScreen, "?1049l comes back")
eq(fs_alt.text, "main1\nmain2\n", "the scrollback is exactly as it was; nothing of the full-screen program is left")
let fs_ow = screen("\(ESC)[?1049hone\r\ntwo\r\nthree\r\nfour\r\nfive\r\nsix")
eq((0..<5).map { fs_ow.lineText($0) }, ["two", "three", "four", "five", "six"], "a line past the bottom scrolls the grid")
eq(fs_ow.lines.count, 5, "…and the grid keeps its size")
let fs_region = screen("\(ESC)[?1049hA\r\nB\r\nC\r\nD\r\nE\(ESC)[2;4r\(ESC)[4;1H\nZ")
eq((0..<5).map { fs_region.lineText($0) }, ["A", "C", "D", "Z", "E"], "scrolling inside a fs_region leaves the lines outside it")
eq(fs_region.row, 3, "the cursor stays on the region's bottom line")
let fs_il = screen("\(ESC)[?1049hA\r\nB\r\nC\r\nD\r\nE\(ESC)[2;1H\(ESC)[L")
eq((0..<5).map { fs_il.lineText($0) }, ["A", "", "B", "C", "D"], "insert line pushes the rest down")
let fs_dl = screen("\(ESC)[?1049hA\r\nB\r\nC\r\nD\r\nE\(ESC)[2;1H\(ESC)[M")
eq((0..<5).map { fs_dl.lineText($0) }, ["A", "C", "D", "E", ""], "delete line pulls the rest up")
let fs_ri = screen("\(ESC)[?1049hA\r\nB\(ESC)[H\(ESC)M")
eq((0..<5).map { fs_ri.lineText($0) }, ["", "A", "B", "", ""], "reverse index at the top scrolls down")
let fs_sv = screen("\(ESC)[?1049h\(ESC)[2;3H\(ESC)7\(ESC)[5;9Hx\(ESC)8y")
eq(fs_sv.lineText(1).trimmingCharacters(in: .whitespaces), "y", "save and restore the cursor (ESC 7 / 8)")
let fs_md = screen("\(ESC)[?25l\(ESC)[?1h\(ESC)[?2004h")
check(!fs_md.cursorVisible && fs_md.applicationCursorKeys && fs_md.bracketedPaste, "cursor hidden, application keys, bracketed paste")
fs_md.feed("\(ESC)[?25h\(ESC)[?1l\(ESC)[?2004l")
check(fs_md.cursorVisible && !fs_md.applicationCursorKeys && !fs_md.bracketedPaste, "and switched off again")
eq(screen("\(ESC)(0lqqk\(ESC)(B lqk").lineText(0), "┌──┐ lqk", "DEC line drawing, then ASCII again")
let fs_wrap = screen("\(ESC)[?1049habcdefghijklmnop", rows: 3, cols: 12)
eq(fs_wrap.lineText(0), "abcdefghijkl", "autowrap at the last column")
eq(fs_wrap.lineText(1), "mnop", "…continues on the next line")
let fs_clr = screen("\(ESC)[?1049hjunk\r\nmore\(ESC)[2J")
eq(fs_clr.lines.count, 5, "clear screen keeps the grid")
eq(fs_clr.text.replacingOccurrences(of: "\n", with: ""), "", "…and empties it")
let fs_rs = screen("\(ESC)[?1049h\(ESC)[5;12Hx")
fs_rs.resize(rows: 3, cols: 10)
eq(fs_rs.lines.count, 3, "resize shrinks the grid")
check(fs_rs.row <= 2 && fs_rs.col <= 9, "…and the cursor stays inside it")
let fs_tab = screen("\(ESC)[?1049ha\tb")
eq(fs_tab.lineText(0), "a       b", "a fs_tab moves to the next stop on the grid")

eq(AgentEvent.parse(line: #"data: {"type":"note","text":"Small model"}"#), .note("Small model"), "a note event")
// when a message is about the open file
for q in ["проверь код", "Check this code", "fix the bug", "объясни, что делает эта функция", "review Tests/GreeterTests.swift",
          "почему тест падает?", "can you refactor it", "what does `run` return"] {
    check(OpenFileIntent.refers(to: q), "about the open file: \(q)")
}
for q in ["привет", "hello", "what is the capital of France?", "расскажи анекдот", "thanks", "спасибо большое"] {
    check(!OpenFileIntent.refers(to: q), "not about it: \(q)")
}

// a lookup only on a clear request or a clear sign
for q in ["привет", "hello", "как дела?", "спасибо", "расскажи анекдот", "что такое фотосинтез?", "write a haiku about the sea",
          "объясни последовательность Фибоначчи", "thanks a lot", "what is a noun?", "hoyo"] {
    check(!WebLookup.wantsSearch(q, year: 2026), "no lookup: \(q)")
}
for q in ["найди в интернете рецепт борща", "поищи отзывы о MacBook Air M5", "search the web for swift concurrency",
          "look it up please", "какая погода сегодня в Лиссабоне?", "курс доллара", "who won the match yesterday",
          "последние новости", "загугли это", "what happened in 2026"] {
    check(WebLookup.wantsSearch(q, year: 2026), "lookup: \(q)")
}
check(WebLookup.explicitRequest("Search online for it") && !WebLookup.explicitRequest("it is online"), "explicit request words")

print("swift logic: \(checks - failures)/\(checks) checks passed")
exit(failures == 0 ? 0 : 1)

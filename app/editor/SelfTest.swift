// `LLMEditor --selftest --root <dir> --llm-root <dir>`: drives the Workspace against a real backend with
// no window, prints what it checked, and exits 0/1. Used by tests/test_editor_app.py.
import AppKit

@MainActor
func runSelfTest(_ ws: Workspace, root: String) {
    var failures: [String] = []
    func check(_ ok: Bool, _ what: String) { print((ok ? "ok   " : "FAIL ") + what); if !ok { failures.append(what) } }
    Task { @MainActor in
        ws.openProject(root)
        for _ in 0..<150 where ws.starting { try? await Task.sleep(nanoseconds: 100_000_000) }
        check(!ws.starting && ws.fatal == nil, "backend started (\(ws.fatal ?? "no error"))")
        check(!ws.projectName.isEmpty, "project name: \(ws.projectName)")
        check((ws.tree[""] ?? []).contains { $0.name == "hello.py" }, "tree lists hello.py")
        check(ws.git?.repo == true && ws.git?.branch != nil, "git status decoded (\(ws.git?.branch ?? "?"))")

        ws.openFile("hello.py")
        for _ in 0..<50 where ws.active == nil { try? await Task.sleep(nanoseconds: 100_000_000) }
        let f = ws.active
        check(f?.path == "hello.py", "file opened")
        check(f?.language == "python", "language detected: \(f?.language ?? "?")")
        check((f?.outline.first?.name) == "Greeter", "outline decoded: \(f?.outline.map(\.name) ?? [])")
        check((f?.spans.contains { $0.kind == "keyword" }) == true, "spans decoded: \(f?.spans.count ?? 0)")
        check((f?.outline.first?.children.first?.name) == "hello", "method nested under the class")

        if let f {                                        // edit → analyze → save → on disk
            f.text += "\ndef added():\n    return 'é😀'\n"
            f.version += 1
            await ws.analyze(f)
            check(f.outline.contains { $0.name == "added" }, "re-analysis sees the new function")
            let map = OffsetMap(f.text)
            let emoji = f.spans.first { $0.kind == "string" && (map.range(start: $0.start, end: $0.end).map { (f.text as NSString).substring(with: $0).contains("😀") } ?? false) }
            check(emoji != nil, "a span around an emoji maps back to the right UTF-16 range")
            ws.save()
            for _ in 0..<50 where f.dirty { try? await Task.sleep(nanoseconds: 100_000_000) }
            let onDisk = (try? String(contentsOfFile: root + "/hello.py", encoding: .utf8)) ?? ""
            check(!f.dirty && onDisk.contains("def added"), "saved to disk")
        }
        await ws.refreshGit()
        check(ws.git?.files?.contains { $0.path == "hello.py" && $0.status == "modified" } == true, "git sees the modification")

        let hits: SearchResponse? = try? await ws.backend.get("/api/search?q=Greeter")
        check((hits?.hits.first?.path) == "hello.py", "search finds Greeter")
        await ws.loadAllFiles()
        check(ws.allFiles.contains("hello.py"), "quick-open file list")

        // the agent endpoint answers (the model is not running here: an error event is the proof it is wired)
        var sawError = false
        do {
            for try await ev in ws.backend.agent(messages: [["role": "user", "content": "hi"]], attachments: ["hello.py"],
                                                 think: false, autoEdit: false) {
                if case .error = ev { sawError = true }
            }
        } catch {}
        check(sawError, "agent stream delivers events (error when no model runs)")

        // problems: a file with a syntax error is flagged, counted, and can be sent to the assistant
        ws.openFile("broken.py")
        for _ in 0..<50 where !ws.open.contains(where: { $0.path == "broken.py" }) { try? await Task.sleep(nanoseconds: 100_000_000) }
        if let b = ws.open.first(where: { $0.path == "broken.py" }) {
            for _ in 0..<30 where b.diagnostics.isEmpty { try? await Task.sleep(nanoseconds: 100_000_000) }
            check(b.diagnostics.first?.line == 2 && b.diagnostics.first?.isError == true,
                  "syntax error found on line 2: \(b.diagnostics.first?.message ?? "none")")
            check(ws.problemSummary.errors >= 1, "problem counter: \(ws.problemSummary.errors)")
            if let d = b.diagnostics.first {
                var asked = false
                ws.onAsk = { asked = true }
                ws.askAboutProblem(d, in: b)
                check(asked && ws.draft.contains("broken.py") && ws.draft.contains("line 2") && ws.draft.contains(d.message),
                      "the problem is in the assistant's composer")
                check(ws.attachments.contains("broken.py"), "and the file is attached")
            }
        } else { check(false, "broken.py opened") }

        // terminal: a real shell on a pty, started lazily
        let term = ws.terminalSession()
        check(!term.running, "no shell before the panel is shown")
        term.start()
        term.send("echo pty_$((6*7))_ok\r")
        for _ in 0..<80 where !term.buffer.text.contains("pty_42_ok") { try? await Task.sleep(nanoseconds: 100_000_000) }
        check(term.buffer.text.contains("pty_42_ok"), "the shell ran a command and its output reached the buffer")
        term.send("printf '\\033[31mred\\033[0m\\n'\r")
        for _ in 0..<50 where !term.buffer.lines.contains(where: { $0.contains { $0.style.fg == .index(1) } }) {
            try? await Task.sleep(nanoseconds: 100_000_000) }
        check(term.buffer.lines.contains { $0.contains { $0.style.fg == .index(1) } }, "ANSI colour from the shell was parsed")
        ws.askAboutTerminal(term.buffer.text)
        check(ws.draft.contains("pty_42_ok") && ws.draft.contains("terminal"), "terminal output can be sent to the assistant")

        // attachments: in the project → by path; elsewhere → its text; a binary file or an unreadable one → refused
        let realRoot = (root as NSString).resolvingSymlinksInPath
        let outside = NSTemporaryDirectory() + "llm-selftest-outside.txt"
        try? "text from elsewhere".write(toFile: outside, atomically: true, encoding: .utf8)
        if case .project(let rel)? = try? Attach.make(URL(fileURLWithPath: realRoot + "/hello.py"), root: realRoot) {
            check(rel == "hello.py", "a file in the project is attached by path")
        } else { check(false, "a file in the project is attached by path") }
        if case .inline(let f)? = try? Attach.make(URL(fileURLWithPath: outside), root: realRoot) {
            check(f.text == "text from elsewhere", "a file from elsewhere travels as text")
        } else { check(false, "a file from elsewhere travels as text") }
        let bin = NSTemporaryDirectory() + "llm-selftest.bin"
        FileManager.default.createFile(atPath: bin, contents: Data([0, 1, 2, 3, 0]))
        check((try? Attach.make(URL(fileURLWithPath: bin), root: realRoot)) == nil, "a binary file is refused")
        ws.addAttachment(from: URL(fileURLWithPath: outside))
        check(ws.inline.count == 1, "addAttachment puts it in the composer's list")

        // branches: list, create + switch, come back
        await ws.loadBranches()
        let original = ws.branches?.current ?? ""
        check(!original.isEmpty && ws.branches?.local.contains(original) == true, "branches listed (current \(original))")
        ws.checkout("selftest-branch", create: true)
        for _ in 0..<50 where ws.branches?.current != "selftest-branch" { try? await Task.sleep(nanoseconds: 100_000_000) }
        check(ws.branches?.current == "selftest-branch" && ws.git?.branch == "selftest-branch", "created and switched to a branch")
        ws.checkout(original)
        for _ in 0..<50 where ws.branches?.current != original { try? await Task.sleep(nanoseconds: 100_000_000) }
        check(ws.branches?.current == original, "switched back")

        ws.shutdown()
        print(failures.isEmpty ? "SELFTEST OK" : "SELFTEST FAILED: \(failures.count)")
        exit(failures.isEmpty ? 0 : 1)
    }
}

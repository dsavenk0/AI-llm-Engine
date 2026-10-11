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

        ws.shutdown()
        print(failures.isEmpty ? "SELFTEST OK" : "SELFTEST FAILED: \(failures.count)")
        exit(failures.isEmpty ? 0 : 1)
    }
}

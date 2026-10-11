// Project state shared by the modules: open files, the tree, git, attachments, layout and which
// modules are switched on.
import AppKit
import SwiftUI

enum LayoutMode: String { case chat, ide }

/// The optional building blocks of the window. Each one is a separate view; switching one off removes it
/// and leaves the others working (e.g. no chat = a plain editor; no git = no Changes tab).
enum ModuleID: String, CaseIterable, Identifiable {
    case files, outline, git, chat, problems, terminal, github, preview
    var id: String { rawValue }
    /// Modules that start switched off: they bring a web view / network, so they are opt-in.
    static let offByDefault: Set<ModuleID> = [.preview]
    var title: String {
        switch self {
        case .github: return "GitHub"
        case .preview: return "Preview (browser)"
        case .files: return "Files"
        case .outline: return "Outline"
        case .git: return "Changes (git)"
        case .chat: return "Assistant"
        case .problems: return "Problems"
        case .terminal: return "Terminal"
        }
    }
}

final class ModuleSettings: ObservableObject {
    /// One set of switches for every window of the editor.
    static let shared = ModuleSettings()
    @Published var enabled: Set<ModuleID>
    private let key = "enabledModules"

    init() {
        if let raw = UserDefaults.standard.array(forKey: key) as? [String] {
            enabled = Set(raw.compactMap(ModuleID.init(rawValue:)))
        } else {
            enabled = Set(ModuleID.allCases).subtracting(ModuleID.offByDefault)
        }
        if !enabled.contains(.files) { enabled.insert(.files) }       // a window without files is not a window
    }
    func isOn(_ m: ModuleID) -> Bool { enabled.contains(m) }
    func toggle(_ m: ModuleID) {
        if m == .files { return }
        if enabled.contains(m) { enabled.remove(m) } else { enabled.insert(m) }
        UserDefaults.standard.set(enabled.map(\.rawValue), forKey: key)
    }
}

final class OpenFile: ObservableObject, Identifiable {
    let path: String
    @Published var text: String
    @Published var saved: String
    @Published var mtime: Int64
    @Published var language: String
    @Published var spans: [Span]
    @Published var outline: [Symbol]
    @Published var diagnostics: [Diagnostic] = []     // built-in checks + the language's parser, when run
    @Published var jumpLine: Int?          // set by the outline / search, consumed by the editor
    var deepDiagnostics: [Diagnostic] = []
    var version = 0                        // bumped on every edit; stale analysis results are dropped
    var id: String { path }
    var dirty: Bool { text != saved }

    init(path: String, text: String, mtime: Int64, language: String, spans: [Span], outline: [Symbol]) {
        self.path = path; self.text = text; self.saved = text; self.mtime = mtime
        self.language = language; self.spans = spans; self.outline = outline
    }
}

@MainActor
final class Workspace: ObservableObject {
    let backend = Backend()
    let modules = ModuleSettings.shared
    var llmRoot: String
    var enginePort: Int

    @Published var root = ""
    @Published var projectName = ""
    @Published var starting = true
    @Published var fatal: String?
    @Published var engine: EngineInfo?
    @Published var mode: LayoutMode = .chat
    @Published var tree: [String: [TreeEntry]] = [:]
    @Published var expanded: Set<String> = [""]
    @Published var open: [OpenFile] = []
    @Published var activePath: String?
    @Published var git: GitStatus?
    @Published var attachments: [String] = []        // project files, by path
    @Published var inline: [InlineFile] = []         // files from elsewhere: their text travels with the message
    @Published var pictures: [Picture] = []
    @Published var visionSwitching = false
    @Published var remoteURL: String?                 // set while the assistant talks to another server
    @Published var saveTick = 0                       // bumped after every save: the preview reloads on it
    @Published var branches: BranchesResponse?
    @Published var toast: String?
    @Published var toastBad = false
    @Published var cursor = (line: 1, col: 1)
    @Published var pendingApply: (code: String, hint: String?)?
    @Published var allFiles: [String] = []
    @Published var problemSummary = (errors: 0, warnings: 0)
    @Published var draft = ""                 // the assistant's composer text (shared so other modules can fill it)
    var onAsk: (() -> Void)?                  // set by the shell: switch to the assistant and focus the composer
    @Published var bottomVisible = false
    @Published var bottomTab = "problems"
    private var terminal: TerminalSession?

    /// The shell is created the first time the panel asks for it (and only then started).
    func terminalSession() -> TerminalSession {
        if let t = terminal, t.cwd == root { return t }
        terminal?.stop()
        let t = TerminalSession(cwd: root)
        terminal = t
        return t
    }
    func showPanel(_ tab: String) { bottomTab = tab; bottomVisible = true }
    func recountProblems() {
        let all = open.flatMap(\.diagnostics)
        problemSummary = (all.filter(\.isError).count, all.filter { !$0.isError }.count)
    }

    /// Put a question in the assistant's composer (with the file attached) and bring the assistant forward.
    func ask(_ prompt: String, attach path: String? = nil) {
        if let path { attach(path) }
        draft = prompt
        onAsk?()
    }

    /// "Fix this problem": the message, the code around it and the file.
    func askAboutProblem(_ d: Diagnostic, in f: OpenFile) {
        let lines = f.text.components(separatedBy: "\n")
        let lo = max(1, d.line - 3), hi = min(lines.count, d.line + 3)
        let excerpt = (lo...max(lo, hi)).map { n in
            (n == d.line ? "› " : "  ") + String(format: "%4d  ", n) + (n - 1 < lines.count ? lines[n - 1] : "")
        }.joined(separator: "\n")
        ask("Fix this \(d.severity) in `\(f.path)` (line \(d.line), \(d.source)): \(d.message)\n\n```\n\(excerpt)\n```",
            attach: f.path)
    }

    func askAboutProblems(of f: OpenFile) {
        let list = f.diagnostics.prefix(15).map { "- line \($0.line): \($0.message) (\($0.source))" }.joined(separator: "\n")
        ask("`\(f.path)` has these problems — please fix them:\n\n\(list)", attach: f.path)
    }

    func askAboutTerminal(_ output: String) {
        let tail = output.components(separatedBy: "\n").suffix(60).joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        ask("This came out of the terminal in `\((root as NSString).lastPathComponent)`. What is wrong and how do I fix it?\n\n```\n\(tail)\n```",
            attach: active?.path)
    }

    private var pollTask: Task<Void, Never>?

    init(llmRoot: String, enginePort: Int) {
        self.llmRoot = llmRoot
        self.enginePort = enginePort
    }

    var active: OpenFile? { open.first { $0.path == activePath } }

    func say(_ text: String, bad: Bool = false) {
        toast = text; toastBad = bad
        let current = text
        Task { try? await Task.sleep(nanoseconds: bad ? 6_000_000_000 : 2_500_000_000)
               if self.toast == current { self.toast = nil } }
    }

    func guarded(_ what: String = "", _ work: @escaping () async throws -> Void) {
        Task { do { try await work() } catch { say((what.isEmpty ? "" : what + ": ") + error.localizedDescription, bad: true) } }
    }

    // MARK: project

    func openProject(_ path: String) {
        starting = true
        fatal = nil
        open = []; activePath = nil; tree = [:]; expanded = [""]; attachments = []; inline = []; pictures = []
        git = nil; branches = nil
        root = path
        Task {
            do {
                try await backend.start(root: path, llmRoot: llmRoot, enginePort: enginePort)
                let info: InfoResponse = try await backend.get("/api/info")
                projectName = info.name; engine = info.engine; git = info.git; remoteURL = info.endpoint
                await restoreEndpoint()
                await loadBranches()
                await loadDir("")
                starting = false
                startPolling()
                Task { await loadAllFiles() }
            } catch {
                fatal = error.localizedDescription
                starting = false
            }
        }
    }

    func shutdown() { pollTask?.cancel(); terminal?.stop(); backend.stop() }

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 8_000_000_000)
                guard let self else { return }
                if let info: InfoResponse = try? await self.backend.get("/api/info") {
                    self.engine = info.engine
                    self.git = info.git
                    self.remoteURL = info.endpoint
                } else { self.engine = nil }
            }
        }
    }

    // MARK: tree / files

    func loadDir(_ path: String) async {
        if let r: TreeResponse = try? await backend.get("/api/tree?path=\(backend.encoded(path))") { tree[path] = r.entries }
    }

    func loadAllFiles() async {
        if let r: FilesResponse = try? await backend.get("/api/files") { allFiles = r.files }
    }

    func toggleDir(_ path: String) {
        if expanded.contains(path) { expanded.remove(path) } else {
            expanded.insert(path)
            Task { await loadDir(path) }
        }
    }

    func openFile(_ path: String, line: Int? = nil) {
        if let f = open.first(where: { $0.path == path }) {
            activePath = path
            if let line { f.jumpLine = line }
            return
        }
        guarded("Open \(Paths.displayName(path))") {
            let r: FileResponse = try await self.backend.get("/api/file?path=\(self.backend.encoded(path))")
            if r.binary == true { self.say("\(path) is a binary file", bad: true); return }
            if r.too_large == true { self.say("\(path) is too large to open", bad: true); return }
            if r.not_utf8 == true { self.say("\(path) is not UTF-8 text; it is not opened (saving would damage it)", bad: true); return }
            if let again = self.open.first(where: { $0.path == r.path }) { self.activePath = again.path; return }  // opened twice quickly
            let f = OpenFile(path: r.path, text: r.text ?? "", mtime: r.mtime ?? 0, language: r.language ?? "plaintext",
                             spans: r.spans ?? [], outline: r.outline ?? [])
            f.jumpLine = line
            self.open.append(f)
            self.activePath = f.path
            await self.analyze(f)                  // diagnostics for a file the moment it opens
        }
    }

    func closeFile(_ path: String) {
        guard let f = open.first(where: { $0.path == path }) else { return }
        if f.dirty {
            let a = NSAlert()
            a.messageText = "Discard unsaved changes in \(Paths.displayName(path))?"
            a.addButton(withTitle: "Discard"); a.addButton(withTitle: "Cancel")
            if a.runModal() != .alertFirstButtonReturn { return }
        }
        open.removeAll { $0.path == path }
        if activePath == path { activePath = open.last?.path }
        recountProblems()
    }

    func save() {
        guard let f = active else { return }
        guarded("Save") {
            _ = try await self.write(f)
            self.say("Saved \(Paths.displayName(f.path))")
            await self.refreshGit()
        }
    }

    /// Writes the buffer as it is *now*; marks that exact text as saved (keystrokes made while the request
    /// was in flight stay unsaved).
    @discardableResult
    private func write(_ f: OpenFile) async throws -> Bool {
        let snapshot = f.text
        let r: SaveResponse = try await backend.request("PUT", "/api/file", body: [
            "path": f.path, "text": snapshot, "expected_mtime": f.mtime])
        f.saved = snapshot
        f.mtime = r.mtime
        saveTick += 1
        return true
    }

    func saveAll() {
        for f in open where f.dirty { activePath = f.path; save() }
    }

    /// Saves every changed buffer and reports whether all of them reached the disk.
    func saveAllAndWait() async -> Bool {
        var ok = true
        for f in open where f.dirty {
            do { try await write(f) } catch { say("Save \(Paths.displayName(f.path)): \(error.localizedDescription)", bad: true); ok = false }
        }
        await refreshGit()
        return ok
    }

    func newFile(_ rel: String) {
        guarded("New file") {
            let _: SaveResponse = try await self.backend.request("PUT", "/api/file", body: ["path": rel, "text": ""])
            await self.loadDir((rel as NSString).deletingLastPathComponent)
            await self.loadAllFiles()
            self.openFile(rel)
        }
    }

    /// Re-run highlighting/outline for the buffer (debounced by the editor).
    func analyze(_ f: OpenFile) async {
        let v = f.version, text = f.text
        guard let r: AnalyzeResponse = try? await backend.request("POST", "/api/analyze", body: ["path": f.path, "text": text]),
              f.version == v else { return }
        f.spans = r.spans; f.outline = r.outline; f.language = r.language
        f.diagnostics = (r.diagnostics ?? []) + f.deepDiagnostics.filter { d in
            !(r.diagnostics ?? []).contains { $0.line == d.line } }
        recountProblems()
        scheduleDeepCheck(f)
    }

    /// The language's own parser (swiftc/node/bash…), run once the buffer has been quiet for a moment.
    private func scheduleDeepCheck(_ f: OpenFile) {
        let v = f.version
        Task {
            try? await Task.sleep(nanoseconds: 900_000_000)
            guard f.version == v, f.text.utf8.count < 400_000 else { return }
            guard let r: DiagnosticsResponse = try? await self.backend.request(
                "POST", "/api/diagnostics", body: ["path": f.path, "text": f.text, "deep": true]), f.version == v else { return }
            f.deepDiagnostics = r.diagnostics
            f.diagnostics = r.diagnostics
            self.recountProblems()
        }
    }

    func refreshGit() async {
        if let g: GitStatus = try? await backend.get("/api/git/status") { git = g }
    }

    // MARK: attachments

    func attach(_ path: String) { if !attachments.contains(path) { attachments.append(path) } }
    func detach(_ path: String) { attachments.removeAll { $0 == path } }

    /// A file picked, dropped or pasted into the composer.
    func addAttachment(from url: URL) {
        do {
            switch try Attach.make(url, root: (root as NSString).resolvingSymlinksInPath) {
            case .project(let rel): attach(rel)
            case .inline(let f): if !inline.contains(where: { $0.name == f.name && $0.text == f.text }) { inline.append(f) }
            case .picture(let p): add(picture: p)
            }
        } catch { say(error.localizedDescription, bad: true) }
    }

    func addPastedImage(_ img: NSImage) {
        do { add(picture: try Attach.picture(img, name: "pasted-image-\(pictures.count + 1).jpg")) }
        catch { say(error.localizedDescription, bad: true) }
    }

    private func add(picture p: Picture) {
        pictures.append(p)
        if engine?.vision_capable == true, engine?.vision != true {
            say("Images are switched off for this model — tick “Images” to enable them", bad: true)
        } else if engine?.vision_capable != true, engine != nil {
            say("This model cannot read images; the picture is sent but ignored", bad: true)
        }
    }

    // MARK: image input switch (a per-model setting that needs the model restarted)

    func setVision(_ on: Bool) {
        guard !visionSwitching else { return }
        visionSwitching = true
        Task {
            let ok = await Task.detached { [llmRoot] () -> Bool in
                let p = Process()
                p.executableURL = URL(fileURLWithPath: llmRoot + "/llm")
                p.arguments = ["vision", on ? "on" : "off"]
                p.currentDirectoryURL = URL(fileURLWithPath: llmRoot)
                p.standardOutput = FileHandle.nullDevice
                p.standardError = FileHandle.nullDevice
                do { try p.run() } catch { return false }
                p.waitUntilExit()
                return p.terminationStatus == 0
            }.value
            // the restart is asynchronous: poll until the engine answers again
            for _ in 0..<60 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if let info: InfoResponse = try? await backend.get("/api/info"), let e = info.engine {
                    engine = e
                    if e.vision == on || !ok { break }
                }
            }
            visionSwitching = false
            say(ok ? (on ? "Images are on" : "Images are off") : "Could not change the image setting", bad: !ok)
        }
    }

    // MARK: branches

    func loadBranches() async {
        if let b: BranchesResponse = try? await backend.get("/api/git/branches") { branches = b }
    }

    func checkout(_ branch: String, create: Bool = false) {
        let dirtyOpen = open.filter(\.dirty)
        guarded(create ? "New branch" : "Switch branch") {
            let r: CheckoutResponse = try await self.backend.request("POST", "/api/git/checkout",
                                                                     body: ["branch": branch, "create": create])
            self.branches = r.branches
            self.git = r.git
            self.say("On branch \(r.branches.current ?? branch)")
            // files on disk may differ now: reload the open ones that have no unsaved work, then the tree
            for f in self.open where !dirtyOpen.contains(where: { $0.path == f.path }) {
                if let fr: FileResponse = try? await self.backend.get("/api/file?path=\(self.backend.encoded(f.path))"),
                   let t = fr.text { f.text = t; f.saved = t; f.mtime = fr.mtime ?? f.mtime }
                else { self.open.removeAll { $0.path == f.path } }        // gone on this branch
            }
            if let a = self.activePath, !self.open.contains(where: { $0.path == a }) { self.activePath = self.open.last?.path }
            self.tree = [:]
            await self.loadDir("")
            for d in self.expanded where !d.isEmpty { await self.loadDir(d) }
            await self.loadAllFiles()
        }
    }

    func newBranch() {
        let a = NSAlert()
        a.messageText = "New branch"
        a.informativeText = "Created from the current commit and switched to."
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24))
        field.placeholderString = "feature/name"
        a.accessoryView = field
        a.addButton(withTitle: "Create"); a.addButton(withTitle: "Cancel")
        guard a.runModal() == .alertFirstButtonReturn else { return }
        let name = field.stringValue.trimmingCharacters(in: .whitespaces)
        if !name.isEmpty { checkout(name, create: true) }
    }
}

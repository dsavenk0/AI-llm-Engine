// Project state shared by the modules: open files, the tree, git, attachments, layout and which
// modules are switched on.
import AppKit
import SwiftUI

enum LayoutMode: String { case chat, ide }

/// The optional building blocks of the window. Each one is a separate view; switching one off removes it
/// and leaves the others working (e.g. no chat = a plain editor; no git = no Changes tab).
enum ModuleID: String, CaseIterable, Identifiable {
    case files, outline, git, chat
    var id: String { rawValue }
    var title: String {
        switch self {
        case .files: return "Files"
        case .outline: return "Outline"
        case .git: return "Changes (git)"
        case .chat: return "Assistant"
        }
    }
}

final class ModuleSettings: ObservableObject {
    @Published var enabled: Set<ModuleID>
    private let key = "enabledModules"

    init() {
        if let raw = UserDefaults.standard.array(forKey: key) as? [String] {
            enabled = Set(raw.compactMap(ModuleID.init(rawValue:)))
        } else {
            enabled = Set(ModuleID.allCases)
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
    @Published var jumpLine: Int?          // set by the outline / search, consumed by the editor
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
    let modules = ModuleSettings()
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
    @Published var attachments: [String] = []
    @Published var toast: String?
    @Published var toastBad = false
    @Published var cursor = (line: 1, col: 1)
    @Published var pendingApply: (code: String, hint: String?)?
    @Published var allFiles: [String] = []

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
        open = []; activePath = nil; tree = [:]; expanded = [""]; attachments = []; git = nil
        root = path
        Task {
            do {
                try await backend.start(root: path, llmRoot: llmRoot, enginePort: enginePort)
                let info: InfoResponse = try await backend.get("/api/info")
                projectName = info.name; engine = info.engine; git = info.git
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

    func shutdown() { pollTask?.cancel(); backend.stop() }

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 8_000_000_000)
                guard let self else { return }
                if let info: InfoResponse = try? await self.backend.get("/api/info") {
                    self.engine = info.engine
                    self.git = info.git
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
            let f = OpenFile(path: r.path, text: r.text ?? "", mtime: r.mtime ?? 0, language: r.language ?? "plaintext",
                             spans: r.spans ?? [], outline: r.outline ?? [])
            f.jumpLine = line
            self.open.append(f)
            self.activePath = f.path
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
    }

    func save() {
        guard let f = active else { return }
        guarded("Save") {
            let r: SaveResponse = try await self.backend.request("PUT", "/api/file", body: [
                "path": f.path, "text": f.text, "expected_mtime": f.mtime])
            f.saved = f.text; f.mtime = r.mtime
            self.say("Saved \(Paths.displayName(f.path))")
            await self.refreshGit()
        }
    }

    func saveAll() {
        for f in open where f.dirty { activePath = f.path; save() }
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
    }

    func refreshGit() async {
        if let g: GitStatus = try? await backend.get("/api/git/status") { git = g }
    }

    // MARK: attachments

    func attach(_ path: String) { if !attachments.contains(path) { attachments.append(path) } }
    func detach(_ path: String) { attachments.removeAll { $0 == path } }
}

// Knowledge, the app side: the one client for the index, its shared state, and the picker the chats share.
// The libraries page lives in LLMBar ▸ Tools (app/bar/KnowledgePage.swift).
//
// The index is engine/knowledge.py; this file never opens it. Every question goes to `./llm knowledge … --json`,
// a short process that answers and exits — so with Knowledge unused nothing of it runs, and the app holds no
// database, no cache and no model for it. Until a knowledge database exists on disk, not even the CLI is run.
import AppKit
import SwiftUI

extension Notification.Name {
    /// Posted after libraries or documents changed (so every open picker and window shows the same thing).
    static let knowledgeChanged = Notification.Name("local.llm-engine.knowledge-changed")
}

// MARK: - the CLI

enum KnowledgeCLI {
    /// The database the CLI uses by default (engine/knowledge.py `default_path`, which LLM_KNOWLEDGE_DB overrides —
    /// the CLI inherits this process's environment, so both always look at the same file).
    static var databasePath: String {
        if let p = ProcessInfo.processInfo.environment["LLM_KNOWLEDGE_DB"], !p.isEmpty { return p }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/LLMBar/knowledge/knowledge.db").path
    }
    static var databaseExists: Bool { FileManager.default.fileExists(atPath: databasePath) }

    /// A running command, so a long indexing run can be stopped (the CLI finishes the current file as stopped).
    final class Job: @unchecked Sendable {
        private let lock = NSLock()
        private var process: Process?
        func attach(_ p: Process) { lock.lock(); process = p; lock.unlock() }
        func stop() { lock.lock(); process?.terminate(); lock.unlock() }
    }

    /// stdout of `./llm knowledge <args> --json` (stderr — progress — is dropped).
    static func run(_ args: [String], job: Job? = nil) async -> String {
        let root = Paths.shared.root, home = Paths.shared.home
        return await Task.detached(priority: .utility) {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: root + "/llm")
            p.arguments = ["knowledge"] + args + ["--json"]
            p.currentDirectoryURL = URL(fileURLWithPath: root)
            var env = ProcessInfo.processInfo.environment
            env["LLM_HOME"] = home
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            p.environment = env
            p.standardInput = FileHandle.nullDevice
            p.standardError = FileHandle.nullDevice
            let out = Pipe()
            p.standardOutput = out
            do { try p.run() } catch { return "" }
            job?.attach(p)
            let data = out.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            return String(decoding: data, as: UTF8.self)
        }.value
    }
}

// MARK: - shared state

@MainActor
final class KnowledgeStore: ObservableObject {
    static let shared = KnowledgeStore()

    @Published private(set) var libraries: [KnowledgeLibrary] = []
    @Published var busy: String?                  // "Indexing 3 items…" while a command runs
    @Published var message: String?              // the last outcome or error, for the window's status line
    private var loadedAt: Date?
    private var job: KnowledgeCLI.Job?

    var names: [Int: String] { Dictionary(uniqueKeysWithValues: libraries.map { ($0.id, $0.name) }) }

    /// Libraries, re-read at most every 20 s unless `force`. No database on disk → none, without running anything.
    func refresh(force: Bool = false) async {
        guard KnowledgeCLI.databaseExists else { libraries = []; return }
        if !force, let t = loadedAt, Date().timeIntervalSince(t) < 20 { return }
        let out = await KnowledgeCLI.run(["libraries"])
        if let libs = KnowledgeJSON.decode([KnowledgeLibrary].self, from: out) { libraries = libs }
        loadedAt = Date()
    }

    private func changed() async {
        await refresh(force: true)
        NotificationCenter.default.post(name: .knowledgeChanged, object: nil)
    }

    /// Runs a changing command; its error (if any) becomes `message`. True on success.
    private func mutate(_ args: [String], busy label: String? = nil, done: String? = nil) async -> Bool {
        if let label { busy = label }
        let out = await KnowledgeCLI.run(args)
        busy = nil
        if let err = KnowledgeJSON.error(in: out) { message = err; return false }
        if out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            message = L("The knowledge command did not answer (is the engine installed?)", "Команда knowledge не ответила (движок установлен?)")
            return false
        }
        message = done
        await changed()
        return true
    }

    func create(_ name: String) async -> Bool { await mutate(["create", name], done: L("Library “\(name)” created", "Библиотека «\(name)» создана")) }
    func rename(_ id: Int, to name: String) async -> Bool { await mutate(["rename", String(id), name]) }
    func delete(_ id: Int) async -> Bool { await mutate(["delete", String(id), "--yes"], done: L("Library deleted — your files are untouched", "Библиотека удалена — ваши файлы не тронуты")) }
    func remove(doc: Int, library: Int) async -> Bool { await mutate(["remove", String(library), String(doc)]) }
    func detach(doc: Int, chat: String) async -> Bool { await mutate(["detach", chat, String(doc)]) }
    func promote(doc: Int, library: Int) async -> Bool { await mutate(["promote", String(doc), String(library)]) }

    /// A deleted chat's own documents leave the index (what a library also holds stays).
    func forgetChat(_ chat: String) async {
        guard KnowledgeCLI.databaseExists else { return }
        _ = await KnowledgeCLI.run(["forget-chat", chat])
        NotificationCenter.default.post(name: .knowledgeChanged, object: nil)
    }

    /// Indexes files/folders into a library (`library`) or for one chat (`chat`). Returns the per-file results.
    func add(_ paths: [String], library: Int? = nil, chat: String? = nil) async -> [KnowledgeAddResult] {
        guard !paths.isEmpty, library != nil || chat != nil else { return [] }
        let j = KnowledgeCLI.Job()
        job = j
        busy = L("Indexing \(paths.count) item\(paths.count == 1 ? "" : "s")…", "Индексация: \(paths.count)…")
        let args = library.map { ["add", String($0)] } ?? ["attach", chat!]
        let out = await KnowledgeCLI.run(args + paths, job: j)
        busy = nil
        job = nil
        if let err = KnowledgeJSON.error(in: out) { message = err; return [] }
        let results = KnowledgeJSON.decode([KnowledgeAddResult].self, from: out) ?? []
        let ok = results.filter { $0.state == "ready" }.count
        let failed = results.filter { $0.state != "ready" }
        message = L("\(ok) of \(results.count) indexed", "Проиндексировано \(ok) из \(results.count)")
            + (failed.isEmpty ? "" : L(" · not indexed: ", " · не проиндексировано: ")
            + failed.prefix(3).map { "\($0.name) (\($0.note))" }.joined(separator: ", "))
        await changed()
        return results
    }

    func stopIndexing() { job?.stop() }

    func documents(library: Int? = nil, chat: String? = nil) async -> [KnowledgeDocument] {
        guard KnowledgeCLI.databaseExists else { return [] }
        var args = ["docs"]
        if let library { args.append(String(library)) }
        if let chat { args += ["--chat", chat] }
        return KnowledgeJSON.decode([KnowledgeDocument].self, from: await KnowledgeCLI.run(args)) ?? []
    }

    func search(_ query: String, libraries: [Int]) async -> [KnowledgeHit] {
        guard KnowledgeCLI.databaseExists, !libraries.isEmpty else { return [] }
        let libs = libraries.map(String.init).joined(separator: ",")
        return KnowledgeJSON.decode([KnowledgeHit].self,
                                    from: await KnowledgeCLI.run(["search", libs, query, "--limit", "12"])) ?? []
    }

    /// The prompt block for one question, or nil (Knowledge off, nothing indexed, nothing matched).
    func context(_ query: String, libraries: [Int], chat: String? = nil) async -> KnowledgeContext? {
        guard KnowledgeCLI.databaseExists, !libraries.isEmpty || chat != nil else { return nil }
        let libs = libraries.isEmpty ? "-" : libraries.map(String.init).joined(separator: ",")
        var args = ["search", libs, String(query.prefix(2000)), "--context"]
        if let chat { args += ["--chat", chat] }
        return KnowledgeJSON.decode(KnowledgeContext.self, from: await KnowledgeCLI.run(args))
    }
}

// MARK: - the picker (editor composer, quick chat)

/// The books button next to a chat's input: which libraries this chat reads. Off (nothing ticked) is the default,
/// and then the chat sends exactly what it sent before Knowledge existed.
struct KnowledgePicker: View {
    @ObservedObject private var store = KnowledgeStore.shared
    @Binding var choice: KnowledgeChoice
    var tint: Color = .accentColor
    var dim: Color = .secondary
    /// Offered when the chat can hold documents of its own (the editor). nil hides those items.
    var addToChat: (() -> Void)? = nil
    var clearChat: (() -> Void)? = nil

    var body: some View {
        Menu {
            if store.libraries.isEmpty {
                Text(L("No libraries yet", "Библиотек пока нет"))
            } else {
                Section(L("Search in", "Искать в")) {
                    ForEach(store.libraries) { lib in
                        Toggle(isOn: Binding(get: { choice.libraries.contains(lib.id) }, set: { _ in choice.toggle(lib.id) })) {
                            Text(lib.ready == 0 ? "\(lib.name) — " + L("empty", "пусто") : "\(lib.name) (\(lib.ready))")
                        }
                    }
                }
            }
            if let addToChat {
                Divider()
                Button(L("Add files to this chat…", "Добавить файлы в этот чат…"), action: addToChat)
                if choice.ownDocuments, let clearChat {
                    Button(L("Remove this chat’s documents", "Убрать документы этого чата"), action: clearChat)
                }
            }
            Divider()
            Button(L("Manage Knowledge…", "Управлять Knowledge…")) { AppHost.openKnowledge() }
        } label: {
            HStack(spacing: 3) {
                Image(systemName: choice.isOff ? "books.vertical" : "books.vertical.fill")
                let badge = choice.badge(libraryNames: store.names)
                if !badge.isEmpty { Text(badge).font(.caption).lineLimit(1) }
            }
            .foregroundColor(choice.isOff ? dim : tint)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(choice.isOff ? L("Knowledge is off for this chat: pick libraries to search",
                              "Knowledge в этом чате выключен: выберите библиотеки для поиска")
                           : L("This chat searches the ticked libraries before answering",
                               "Этот чат ищет в отмеченных библиотеках перед ответом"))
        .task { await store.refresh() }
        .onReceive(NotificationCenter.default.publisher(for: .knowledgeChanged)) { _ in
            choice.keepOnly(existing: store.libraries)
        }
    }
}

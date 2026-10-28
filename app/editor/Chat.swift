// The assistant module: sessions, the streaming agent transcript, question and approval cards.
import AppKit
import SwiftUI

struct ApprovalInfo: Codable, Equatable {
    var id: String, kind: String
    var path: String?, old: String?, new: String?, command: String?, dangerous: String?
}
struct QuestionInfo: Codable, Equatable {
    var id: String, question: String, options: [String]
}

struct ChatItem: Identifiable, Codable, Equatable {
    enum Kind: String, Codable { case user, assistant, activity, approval, question, error, image }
    var id = UUID()
    var kind: Kind
    var text = ""
    var reasoning = ""
    var attached: [String] = []
    var steps: [String] = []              // activity: "search: def run" …
    var counts = ActivityCounts()
    var approval: ApprovalInfo?
    var question: QuestionInfo?
    var resolved: String?                 // what the user decided / answered
    var imagePath: String?                // image: project-relative file of a generated picture
}

struct ChatSession: Identifiable, Codable {
    var id = UUID()
    var title = "New chat"
    var updated = Date()
    var items: [ChatItem] = []
}

@MainActor
final class ChatStore: ObservableObject {
    @Published var sessions: [ChatSession] = []
    @Published var currentID: UUID?
    @Published var busy = false
    @Published var think = false
    @Published var effort: String?        // nil = the model's own default level
    @Published var mode: AgentMode {
        didSet { UserDefaults.standard.set(mode.rawValue, forKey: "agentMode") }
    }
    var autoEdit: Bool { mode.autoEdit }
    @Published var web = false            // the globe: web search and page reading for the agent, off by default
    private var runID: String?
    private var task: Task<Void, Never>?
    private var file: URL?
    private unowned let ws: Workspace

    init(ws: Workspace) {
        self.ws = ws
        mode = AgentMode(rawValue: UserDefaults.standard.string(forKey: "agentMode") ?? "") ?? .manual
    }

    var current: ChatSession? { sessions.first { $0.id == currentID } }
    var items: [ChatItem] { current?.items ?? [] }

    // MARK: persistence (one JSON file per project)

    func load(root: String) {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LLMBar/editor-chats", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let key = String(root.utf8.reduce(UInt64(5381)) { ($0 &* 33) &+ UInt64($1) }, radix: 16)
        file = dir.appendingPathComponent("\((root as NSString).lastPathComponent)-\(key).json")
        if let f = file, let data = try? Data(contentsOf: f),
           let s = try? JSONDecoder().decode([ChatSession].self, from: data) { sessions = s }
        else { sessions = [] }
        currentID = sessions.first?.id
        if sessions.isEmpty { newChat() }
    }

    private func save() {
        guard let f = file else { return }
        lastSave = Date()
        let keep = sessions.sorted { $0.updated > $1.updated }.prefix(40)
        try? JSONEncoder().encode(Array(keep)).write(to: f, options: .atomic)
    }

    private var lastSave = Date.distantPast
    /// Persist at most once a second while a reply streams in, so a crash mid-generation keeps the request
    /// and the partial answer on disk (the unfinished reply can be asked again next launch).
    private func saveThrottled() {
        if Date().timeIntervalSince(lastSave) > 1.0 { save() }
    }

    func newChat() {
        if let c = current, c.items.isEmpty { return }
        let s = ChatSession()
        sessions.insert(s, at: 0)
        currentID = s.id
    }

    func delete(_ id: UUID) {
        sessions.removeAll { $0.id == id }
        if currentID == id { currentID = sessions.first?.id; if currentID == nil { newChat() } }
        save()
    }

    func rename(_ id: UUID, to title: String) {
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, let i = sessions.firstIndex(where: { $0.id == id }) else { return }
        sessions[i].title = String(t.prefix(80))
        save()
    }

    /// Asks for a new name (a small dialog, like the new-branch one).
    func promptRename(_ id: UUID) {
        guard let s = sessions.first(where: { $0.id == id }) else { return }
        let a = NSAlert()
        a.messageText = "Rename chat"
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 280, height: 24))
        field.stringValue = s.title
        a.accessoryView = field
        a.addButton(withTitle: "Rename"); a.addButton(withTitle: "Cancel")
        a.window.initialFirstResponder = field
        if a.runModal() == .alertFirstButtonReturn { rename(id, to: field.stringValue) }
    }

    /// Deleting a chat cannot be undone, so it asks (unless the chat is empty).
    func confirmDelete(_ id: UUID) {
        guard let s = sessions.first(where: { $0.id == id }) else { return }
        if !s.items.isEmpty {
            let a = NSAlert()
            a.alertStyle = .warning
            a.messageText = "Delete “\(s.title)”?"
            a.informativeText = "The conversation is removed from this project's chats. This cannot be undone."
            a.addButton(withTitle: "Delete"); a.addButton(withTitle: "Cancel")
            a.buttons.first?.hasDestructiveAction = true
            guard a.runModal() == .alertFirstButtonReturn else { return }
        }
        delete(id)
    }

    private func mutate(_ body: (inout ChatSession) -> Void) {
        guard let i = sessions.firstIndex(where: { $0.id == currentID }) else { return }
        body(&sessions[i])
        sessions[i].updated = Date()
    }

    // MARK: context

    /// The model sees the plain conversation (activity cards and approvals are for the human).
    private func plainHistory() -> [[String: String]] {
        items.compactMap { item -> [String: String]? in
            switch item.kind {
            case .user: return ["role": "user", "content": item.text]
            case .assistant: return item.text.isEmpty ? nil : ["role": "assistant", "content": item.text]
            default: return nil
            }
        }
    }

    /// What the next request will carry, roughly (see ContextMeter).
    var contextTokens: Int {
        ContextMeter.overhead + plainHistory().reduce(0) { $0 + ContextMeter.tokens(of: $1["content"] ?? "") }
    }

    @Published var compacting = false

    /// Replaces everything before the last two user turns with a short note the model writes.
    func compact() {
        guard !busy, !compacting else { return }
        let users = items.indices.filter { items[$0].kind == .user }
        guard users.count > 2 else { ws.say("The conversation is still short — nothing to compact"); return }
        let tailStart = users[users.count - 2]
        let history = plainHistory()
        compacting = true
        Task {
            defer { compacting = false }
            struct Result: Decodable { let summary: String; let folded: Int }
            do {
                let r: Result = try await ws.backend.request("POST", "/api/compact",
                                                              body: ["messages": history, "keep_user_turns": 2])
                guard !r.summary.isEmpty else { ws.say("Nothing to compact"); return }
                mutate { s in
                    guard tailStart <= s.items.count else { return }
                    let note = ChatItem(kind: .assistant, text: "**Earlier conversation (compacted)**\n\n" + r.summary)
                    s.items.replaceSubrange(0..<tailStart, with: [note])
                }
                save()
                ws.say("Compacted \(r.folded) messages")
            } catch { ws.say("Could not compact: \(error.localizedDescription)", bad: true) }
        }
    }

    // MARK: sending

    func send(_ text: String) {
        let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !busy else { return }
        var attached = ws.attachments
        let inline = ws.inline, pictures = ws.pictures
        // "check this code" with nothing attached: the open file goes along (a small, writable one, only to the
        // local model), shown as the usual attachment chip so it is plain what was sent
        if attached.isEmpty, inline.isEmpty, pictures.isEmpty, ws.remoteURL == nil, OpenFileIntent.refers(to: prompt),
           let f = ws.active, f.readOnlyNote == nil, f.text.utf8.count <= 40_000 {
            attached = [f.path]
        }
        if let f = ws.active, f.dirty, attached.contains(f.path) { ws.save() }
        ws.attachments = []; ws.inline = []; ws.pictures = []
        let names = attached + inline.map(\.name) + pictures.map { "🖼 " + $0.name }
        mutate { s in
            if s.items.isEmpty { s.title = String(prompt.prefix(48)) }
            s.items.append(ChatItem(kind: .user, text: prompt, attached: names))
        }
        save()                        // persist the request before generation starts (survives a crash)
        let history = plainHistory()
        // what the person is looking at: "check this code" means the open file (and where the cursor is)
        let active: [String: Any]? = ws.active.map { ["file": $0.path, "line": ws.cursor.0] }
        busy = true
        task = Task { await self.run(history: history, attached: attached, inline: inline, pictures: pictures, active: active) }
    }

    private func run(history: [[String: String]], attached: [String], inline: [InlineFile], pictures: [Picture],
                     active: [String: Any]? = nil) async {
        defer { busy = false; runID = nil; save() }
        do {
            let stream = ws.backend.agent(messages: history, attachments: attached,
                                          inline: inline.map { ["name": $0.name, "text": $0.text] },
                                          images: pictures.map { ["name": $0.name, "data": $0.dataURL] },
                                          think: think, autoEdit: autoEdit, web: web,
                                          mode: mode.rawValue, effort: think ? effort : nil, active: active)
            for try await ev in stream {
                apply(ev)
            }
        } catch is CancellationError {
        } catch {
            if !Task.isCancelled { mutate { $0.items.append(ChatItem(kind: .error, text: error.localizedDescription)) } }
        }
    }

    func stop() { task?.cancel() }

    /// A turn whose reply did not complete — stopped, failed, or cut off by a crash (the last item is the
    /// user's message or an error, with no finished answer after it). Such a turn can be asked again.
    var canRetry: Bool {
        guard !busy, let last = current?.items.last else { return false }
        return last.kind == .user || last.kind == .error
    }

    /// Ask the last user message again: drop the unfinished reply after it and regenerate from the history.
    func retry() {
        guard !busy, let items = current?.items, let i = items.lastIndex(where: { $0.kind == .user }) else { return }
        mutate { s in if i + 1 < s.items.count { s.items.removeSubrange((i + 1)...) } }
        save()
        busy = true
        let history = plainHistory()
        task = Task { await self.run(history: history, attached: [], inline: [], pictures: [], active: nil) }
    }

    private func apply(_ ev: AgentEvent) {
        mutate { s in
            func lastIs(_ k: ChatItem.Kind) -> Bool { s.items.last?.kind == k }
            switch ev {
            case .run(let id): runID = id
            case .delta(let t):
                if !lastIs(.assistant) { s.items.append(ChatItem(kind: .assistant)) }
                s.items[s.items.count - 1].text += t
            case .reasoning(let t):
                if !lastIs(.assistant) { s.items.append(ChatItem(kind: .assistant)) }
                s.items[s.items.count - 1].reasoning += t
            case .tool(_, let name, let category, let args):
                if !lastIs(.activity) { s.items.append(ChatItem(kind: .activity)) }
                let n = s.items.count - 1
                s.items[n].counts.add(category: category)
                let detail = args.values.first.map { String($0.prefix(60)) } ?? ""
                s.items[n].steps.append(detail.isEmpty ? name : "\(name): \(detail)")
            case .image(let path, let prompt):
                s.items.append(ChatItem(kind: .image, text: prompt, imagePath: path))
            case .approval(let id, let kind, let path, let old, let new, let command, let dangerous):
                s.items.append(ChatItem(kind: .approval, approval: ApprovalInfo(id: id, kind: kind, path: path, old: old,
                                                                                new: new, command: command, dangerous: dangerous)))
            case .question(let id, let q, let options):
                s.items.append(ChatItem(kind: .question, question: QuestionInfo(id: id, question: q, options: options)))
            case .toolResult(_, _, let ok, let summary):
                if !ok, !lastIs(.error) { s.items.append(ChatItem(kind: .error, text: summary)) }
            case .note(let t): ws.say(t)
            case .done: break
            case .error(let m): s.items.append(ChatItem(kind: .error, text: m))
            }
        }
        saveThrottled()               // keep the partial answer on disk as it streams in
    }

    // MARK: answering cards

    func decide(_ item: ChatItem, approve: Bool) {
        guard let a = item.approval, let run = runID else { return }
        mark(item.id, approve ? "Approved" : "Rejected")
        ws.guarded("Answer") { try await self.ws.backend.post("/api/agent/answer",
                                                              body: ["run": run, "kind": "approval", "id": a.id, "approve": approve]) }
        if approve, a.kind != "run" { Task { try? await Task.sleep(nanoseconds: 700_000_000)
                                        await ws.refreshGit(); reloadOpen(a.path) } }
    }

    func answer(_ item: ChatItem, text: String) {
        guard let q = item.question, let run = runID else { return }
        mark(item.id, text.isEmpty ? "Skipped" : text)
        ws.guarded("Answer") { try await self.ws.backend.post("/api/agent/answer",
                                                              body: ["run": run, "kind": "question", "id": q.id, "text": text]) }
    }

    private func mark(_ id: UUID, _ text: String) {
        mutate { s in if let i = s.items.firstIndex(where: { $0.id == id }) { s.items[i].resolved = text } }
    }

    /// An approved edit changed a file on disk: refresh the open buffer if it has no unsaved work.
    private func reloadOpen(_ path: String?) {
        guard let path, let f = ws.open.first(where: { $0.path == path }), !f.dirty else { return }
        ws.guarded("Reload") {
            let r: FileResponse = try await self.ws.backend.get("/api/file?path=\(self.ws.backend.encoded(path))")
            if let t = r.text { f.text = t; f.saved = t; f.mtime = r.mtime ?? f.mtime; f.spans = r.spans ?? []; f.outline = r.outline ?? [] }
        }
    }
}

// MARK: - views

/// ⋯ on a chat row: rename or delete without knowing about the right-click menu.
struct ChatRowMenu: View {
    @ObservedObject var store: ChatStore
    let id: UUID
    var body: some View {
        Menu {
            Button("Rename…") { store.promptRename(id) }
            Button("Delete…") { store.confirmDelete(id) }
        } label: { Image(systemName: "ellipsis").foregroundColor(.secondary) }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().help("Rename or delete this chat")
    }
}

/// The editor layout has no chat list: this bar names the current chat and switches, renames or deletes.
struct ChatSwitcher: View {
    @ObservedObject var store: ChatStore
    var body: some View {
        HStack(spacing: 6) {
            Menu {
                Button("New chat") { store.newChat() }
                Divider()
                ForEach(store.sessions.sorted { $0.updated > $1.updated }.prefix(15)) { s in
                    Button { store.currentID = s.id } label: {
                        if s.id == store.currentID { Label(s.title, systemImage: "checkmark") } else { Text(s.title) }
                    }
                }
                if let id = store.currentID {
                    Divider()
                    Button("Rename this chat…") { store.promptRename(id) }
                    Button("Delete this chat…") { store.confirmDelete(id) }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(store.current?.title ?? "Chat").lineLimit(1).font(.callout)
                }.foregroundColor(.secondary)
            }.menuStyle(.borderlessButton).fixedSize()
            Spacer()
            Button { store.newChat() } label: { Image(systemName: "square.and.pencil") }
                .buttonStyle(.plain).help("New chat")
        }.padding(.horizontal, 10).padding(.vertical, 5)
    }
}

struct ChatModuleView: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var store: ChatStore
    var showSwitcher = false

    var body: some View {
        VStack(spacing: 0) {
            if showSwitcher { ChatSwitcher(store: store); Divider() }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        if store.items.isEmpty { EmptyChat() }
                        ForEach(store.items) { item in ChatItemView(item: item, ws: ws, store: store).id(item.id) }
                        if store.busy { ProgressView().controlSize(.small).padding(.leading, 4) }
                        Color.clear.frame(height: 1).id("end")
                    }.padding(12)
                }
                .onChange(of: store.items.last?.text) { _ in proxy.scrollTo("end") }
                .onChange(of: store.items.count) { _ in proxy.scrollTo("end") }
            }
            Divider()
            Composer(ws: ws, store: store, draft: $ws.draft)
        }
    }
}

private struct EmptyChat: View {
    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: "sparkles").font(.largeTitle).foregroundColor(.accentColor)
            Text("Ask about this project").font(.headline)
            Text("The assistant can search and read files, ask you questions, and edit after your approval.")
                .font(.callout).foregroundColor(.secondary).multilineTextAlignment(.center)
        }.frame(maxWidth: .infinity).padding(.top, 60)
    }
}

struct ChatItemView: View {
    let item: ChatItem
    @ObservedObject var ws: Workspace
    @ObservedObject var store: ChatStore
    @State private var showSteps = false

    var body: some View {
        switch item.kind {
        case .user:
            VStack(alignment: .leading, spacing: 4) {
                Text(item.text).textSelection(.enabled)
                if !item.attached.isEmpty {
                    Label(item.attached.map(PathTools.displayName).joined(separator: ", "), systemImage: "paperclip")
                        .font(.caption).foregroundColor(.secondary)
                }
            }
            .padding(10).frame(maxWidth: .infinity, alignment: .leading)
            .background(Palette.accent.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
            .overlay(alignment: .leading) { RoundedRectangle(cornerRadius: 1).fill(Palette.accent).frame(width: 3).padding(.vertical, 6) }
        case .assistant:
            VStack(alignment: .leading, spacing: 8) {
                if !item.reasoning.isEmpty {
                    Collapsible(title: "Reasoning") {
                        Text(item.reasoning).font(.callout).foregroundColor(.secondary).textSelection(.enabled)
                    }
                }
                ForEach(Array(Markdown.split(item.text).enumerated()), id: \.offset) { _, block in
                    switch block {
                    case .text(let t): MarkdownText(t)
                    case .code(let lang, let code, let hint): CodeBlock(lang: lang, code: code, hint: hint, ws: ws)
                    }
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        case .activity:
            VStack(alignment: .leading, spacing: 2) {
                Button { showSteps.toggle() } label: {
                    HStack(spacing: 4) {
                        Text(item.counts.summary()).foregroundColor(.secondary)
                        Image(systemName: showSteps ? "chevron.down" : "chevron.right").font(.caption2).foregroundColor(.secondary)
                    }
                }.buttonStyle(.plain)
                if showSteps {
                    ForEach(Array(item.steps.enumerated()), id: \.offset) { _, s in
                        Text(s).font(.system(size: 11, design: .monospaced)).foregroundColor(.secondary).lineLimit(1)
                    }.padding(.leading, 8)
                }
            }.font(.callout)
        case .approval:
            if let a = item.approval { ApprovalCard(item: item, a: a, store: store) }
        case .question:
            if let q = item.question { QuestionCard(item: item, q: q, store: store) }
        case .error:
            Label(item.text, systemImage: "exclamationmark.triangle").foregroundColor(.red).font(.callout)
        case .image:
            GeneratedImage(item: item, ws: ws)
        }
    }
}

/// A picture the assistant drew: shown inline, kept in the project under .llm/images, a click opens it.
struct GeneratedImage: View {
    let item: ChatItem
    @ObservedObject var ws: Workspace

    private var url: URL? { item.imagePath.map { URL(fileURLWithPath: ws.root).appendingPathComponent($0) } }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let u = url, let img = NSImage(contentsOf: u) {
                Image(nsImage: img).resizable().scaledToFit()
                    .frame(maxWidth: 420, maxHeight: 420, alignment: .leading)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Palette.hairline))
                    .onTapGesture { NSWorkspace.shared.open(u) }
                    .help("Open the image")
            } else {
                Label("The image is no longer on disk", systemImage: "photo").foregroundColor(.secondary)
            }
            Text((item.imagePath ?? "") + (item.text.isEmpty ? "" : "  ·  " + item.text))
                .font(.caption).foregroundColor(.secondary).lineLimit(2).textSelection(.enabled)
        }
    }
}

/// A disclosure whose whole header row is the click target (the stock one reacts to its chevron only).
struct Collapsible<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content
    @State private var open = false
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button { withAnimation(.easeOut(duration: 0.12)) { open.toggle() } } label: {
                HStack(spacing: 6) {
                    Image(systemName: open ? "chevron.down" : "chevron.right").font(.caption2).frame(width: 12)
                    Text(title)
                    Spacer()
                }.foregroundColor(.secondary).contentShape(Rectangle())
            }.buttonStyle(.plain)
            if open { content.padding(.leading, 18) }
        }
    }
}

struct MarkdownText: View {
    let text: String
    init(_ t: String) { text = t }
    var body: some View {
        let opts = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        if let a = try? AttributedString(markdown: text, options: opts) {
            Text(a).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        } else {
            Text(text).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct CodeBlock: View {
    let lang: String, code: String, hint: String?
    @ObservedObject var ws: Workspace
    @State private var copied = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(lang.isEmpty ? "code" : lang).font(.caption).foregroundColor(.secondary)
                if let h = hint { Text("→ \(h)").font(.caption).foregroundColor(.accentColor) }
                Spacer()
                Button(copied ? "Copied" : "Copy") {
                    NSPasteboard.general.clearContents(); NSPasteboard.general.setString(code, forType: .string)
                    copied = true
                }.buttonStyle(.borderless).font(.caption)
                Button("Apply…") { ws.pendingApply = (code, hint) }.buttonStyle(.borderless).font(.caption)
            }.padding(.horizontal, 8).padding(.vertical, 4).background(Color.primary.opacity(0.07))
            ScrollView(.horizontal) {
                Text(code).font(.system(size: 12, design: .monospaced)).textSelection(.enabled).padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(Color.primary.opacity(0.04))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.1)))
    }
}

struct QuestionCard: View {
    let item: ChatItem
    let q: QuestionInfo
    @ObservedObject var store: ChatStore
    @State private var other = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Question").font(.caption).foregroundColor(.secondary)
            Text(q.question).font(.headline)
            if let r = item.resolved {
                Label(r, systemImage: "checkmark.circle").foregroundColor(.secondary)
            } else {
                ForEach(Array(q.options.enumerated()), id: \.offset) { i, o in
                    Button { store.answer(item, text: o) } label: {
                        HStack { Text(["A", "B", "C", "D"][min(i, 3)]).font(.caption).bold().frame(width: 20, height: 20)
                                .background(Color.primary.opacity(0.08), in: RoundedRectangle(cornerRadius: 4))
                            Text(o); Spacer() }
                            .padding(6).contentShape(Rectangle())
                    }.buttonStyle(.plain)
                }
                HStack {
                    TextField("Other…", text: $other).textFieldStyle(.roundedBorder)
                        .onSubmit { if !other.isEmpty { store.answer(item, text: other) } }
                    Button("Skip") { store.answer(item, text: "") }
                    Button("Continue") { store.answer(item, text: other) }.disabled(other.isEmpty)
                        .keyboardShortcut(.return, modifiers: [])
                }
            }
        }
        .padding(12).background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.primary.opacity(0.12)))
    }
}

struct ApprovalCard: View {
    let item: ChatItem
    let a: ApprovalInfo
    @ObservedObject var store: ChatStore

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: a.kind == "run" ? "terminal" : "pencil.line")
                Text(a.kind == "run" ? "Run a command?" : "\(a.kind == "write" ? "Write" : "Edit") \(a.path ?? "")?").font(.headline)
                Spacer()
                if let d = a.dangerous { Text("⚠ \(d)").font(.caption).bold().foregroundColor(.red) }
            }
            if a.kind == "run" {
                Text(a.command ?? "").font(.system(size: 12, design: .monospaced)).padding(6)
                    .frame(maxWidth: .infinity, alignment: .leading).background(Color.black.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
            } else {
                DiffView(lines: LineDiff.diff(old: a.old ?? "", new: a.new ?? ""))
            }
            if let r = item.resolved {
                Label(r, systemImage: r == "Approved" ? "checkmark.circle" : "xmark.circle")
                    .foregroundColor(r == "Approved" ? .green : .secondary)
            } else {
                HStack {
                    Button("Approve") { store.decide(item, approve: true) }.keyboardShortcut(.return, modifiers: .command)
                        .tint(a.dangerous == nil ? .accentColor : .red)
                    Button("Reject") { store.decide(item, approve: false) }.keyboardShortcut(.escape, modifiers: [])
                }
            }
        }
        .padding(12).background(Palette.yellow.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Palette.yellow.opacity(0.35)))
    }
}

struct DiffView: View {
    let lines: [DiffLine]
    var body: some View {
        let shown = lines.count > 60 ? Array(lines.filter { $0.kind != .same }.prefix(60)) : lines
        ScrollView([.vertical, .horizontal]) {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(shown.enumerated()), id: \.offset) { _, l in
                    Text("\(l.kind.rawValue) \(l.text)").font(.system(size: 11.5, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(l.kind == .added ? Color.green.opacity(0.18) : l.kind == .removed ? Color.red.opacity(0.18) : .clear)
                }
            }
        }.frame(maxHeight: 200).background(Color.primary.opacity(0.04)).clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

// MARK: - sessions list (the chat-mode sidebar)

struct SessionsModuleView: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var store: ChatStore

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button { store.newChat() } label: { Label("New chat", systemImage: "square.and.pencil").frame(maxWidth: .infinity, alignment: .leading) }
                .buttonStyle(.plain).padding(10)
            Divider()
            AccentCaption(text: "Chats", color: Palette.accent).padding(.horizontal, 10).padding(.top, 10).padding(.bottom, 2)
            List(selection: Binding(get: { store.currentID }, set: { store.currentID = $0 })) {
                ForEach(store.sessions.sorted { $0.updated > $1.updated }) { s in
                    HStack {
                        Circle().fill(s.id == store.currentID ? Color.accentColor : Color.clear).frame(width: 6, height: 6)
                        VStack(alignment: .leading) {
                            Text(s.title).lineLimit(1)
                            Text(s.updated, style: .relative).font(.caption2).foregroundColor(.secondary)
                        }
                        Spacer()
                        ChatRowMenu(store: store, id: s.id)
                    }
                    .tag(s.id)
                    .contextMenu {
                        Button("Rename…") { store.promptRename(s.id) }
                        Button("Delete…") { store.confirmDelete(s.id) }
                    }
                }
            }.listStyle(.sidebar).scrollContentBackground(.hidden)
        }.background(Palette.panel)
    }
}

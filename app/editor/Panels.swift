// Files, outline and git — three independent modules (each only needs the Workspace).
import SwiftUI

// MARK: - Files

struct FilesModuleView: View {
    @ObservedObject var ws: Workspace
    @State private var query = ""
    @State private var contentHits: [SearchHit] = []
    @State private var searching = false
    @State private var newName = ""
    @State private var showNew = false

    private var nameMatches: [String] {
        guard !query.isEmpty else { return [] }
        let q = query.lowercased()
        return ws.allFiles.filter { $0.lowercased().contains(q) }.prefix(60).map { $0 }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass").foregroundColor(.secondary)
                TextField("Files, text…", text: $query)
                    .textFieldStyle(.plain)
                    .onSubmit { runContentSearch() }
                if !query.isEmpty { Button { query = ""; contentHits = [] } label: { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.plain).foregroundColor(.secondary) }
                Button { showNew = true } label: { Image(systemName: "doc.badge.plus") }
                    .buttonStyle(.plain).help("New file")
            }
            .padding(8)
            .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
            .padding(8)
            Divider()
            if query.isEmpty {
                ScrollView { LazyVStack(alignment: .leading, spacing: 0) { TreeRows(ws: ws, path: "", depth: 0) } }
            } else {
                List {
                    if !nameMatches.isEmpty {
                        Section("Files") {
                            ForEach(nameMatches, id: \.self) { p in
                                Button { ws.openFile(p) } label: {
                                    VStack(alignment: .leading) {
                                        Text(Paths.displayName(p))
                                        Text(p).font(.caption).foregroundColor(.secondary)
                                    }
                                }.buttonStyle(.plain)
                            }
                        }
                    }
                    Section(searching ? "Searching…" : "Text (press Return)") {
                        ForEach(contentHits) { h in
                            Button { ws.openFile(h.path, line: Int(h.line)) } label: {
                                VStack(alignment: .leading) {
                                    Text("\(h.path):\(h.line)").font(.caption).foregroundColor(.secondary)
                                    Text(h.text.trimmingCharacters(in: .whitespaces)).lineLimit(1).font(.system(.body, design: .monospaced))
                                }
                            }.buttonStyle(.plain)
                        }
                    }
                }.listStyle(.plain)
            }
        }
        .alert("New file", isPresented: $showNew) {
            TextField("path/relative/to/project.txt", text: $newName)
            Button("Create") { if !newName.isEmpty { ws.newFile(newName); newName = "" } }
            Button("Cancel", role: .cancel) { newName = "" }
        }
    }

    private func runContentSearch() {
        guard !query.isEmpty else { return }
        searching = true
        ws.guarded("Search") {
            let r: SearchResponse = try await ws.backend.get("/api/search?q=\(ws.backend.encoded(query))")
            contentHits = r.hits
            searching = false
        }
    }
}

struct TreeRows: View {
    @ObservedObject var ws: Workspace
    let path: String
    let depth: Int

    var body: some View {
        ForEach(ws.tree[path] ?? []) { e in
            if e.isDir {
                Button { ws.toggleDir(e.path) } label: {
                    HStack(spacing: 4) {
                        Image(systemName: ws.expanded.contains(e.path) ? "chevron.down" : "chevron.right").font(.caption2).frame(width: 12)
                        Image(systemName: "folder").foregroundColor(.accentColor)
                        Text(e.name).lineLimit(1)
                        Spacer()
                    }.padding(.leading, CGFloat(depth) * 14 + 8).padding(.vertical, 3).contentShape(Rectangle())
                }.buttonStyle(.plain)
                if ws.expanded.contains(e.path) { TreeRows(ws: ws, path: e.path, depth: depth + 1) }
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "doc.text").foregroundColor(.secondary).frame(width: 16)
                    Text(e.name).lineLimit(1)
                    Spacer()
                    Button { ws.attach(e.path) } label: { Image(systemName: "paperclip") }
                        .buttonStyle(.plain).foregroundColor(.secondary).help("Attach to the next message")
                }
                .padding(.leading, CGFloat(depth) * 14 + 20).padding(.trailing, 8).padding(.vertical, 3)
                .background(ws.activePath == e.path ? Color.accentColor.opacity(0.18) : Color.clear)
                .contentShape(Rectangle())
                .onTapGesture { ws.openFile(e.path) }
            }
        }
    }
}

// MARK: - Outline

struct OutlineModuleView: View {
    @ObservedObject var ws: Workspace

    var body: some View {
        if let f = ws.active { OutlineList(file: f, ws: ws) }
        else { Text("Open a file to see its structure").foregroundColor(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity) }
    }
}

private struct OutlineList: View {
    @ObservedObject var file: OpenFile
    @ObservedObject var ws: Workspace

    private func rows(_ items: [Symbol], depth: Int = 0) -> [(Symbol, Int)] {
        items.flatMap { [($0, depth)] + rows($0.children, depth: depth + 1) }
    }

    var body: some View {
        let flat = rows(file.outline)
        if flat.isEmpty {
            Text("No symbols").foregroundColor(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            List(flat, id: \.0.id) { sym, depth in
                Button { file.jumpLine = sym.line } label: {
                    HStack(spacing: 6) {
                        Text(String(sym.kind.prefix(3)).uppercased())
                            .font(.system(size: 9, weight: .semibold)).padding(.horizontal, 4)
                            .background(Color.primary.opacity(0.08), in: RoundedRectangle(cornerRadius: 3))
                        Text(sym.name).lineLimit(1)
                        Spacer()
                        Text("\(sym.line)").font(.caption).foregroundColor(.secondary)
                    }.padding(.leading, CGFloat(depth) * 12)
                }.buttonStyle(.plain)
            }.listStyle(.plain)
        }
    }
}

// MARK: - Git

struct GitModuleView: View {
    @ObservedObject var ws: Workspace
    @State private var selected: Set<String> = []
    @State private var message = ""
    @State private var diff: (path: String, text: String)?
    @State private var busy = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let g = ws.git, g.repo {
                HStack {
                    Image(systemName: "arrow.triangle.branch")
                    Text(g.branch ?? "?").bold()
                    if let a = g.ahead, a > 0 { Text("↑\(a)").foregroundColor(.secondary) }
                    if let b = g.behind, b > 0 { Text("↓\(b)").foregroundColor(.secondary) }
                    Spacer()
                    Button { Task { await ws.refreshGit() } } label: { Image(systemName: "arrow.clockwise") }.buttonStyle(.plain)
                }
                if (g.files ?? []).isEmpty {
                    Text("Working tree clean").foregroundColor(.secondary).frame(maxWidth: .infinity, minHeight: 60)
                } else {
                    List(g.files ?? []) { f in
                        HStack {
                            Toggle("", isOn: Binding(get: { selected.contains(f.path) },
                                                     set: { on in if on { selected.insert(f.path) } else { selected.remove(f.path) } }))
                                .labelsHidden()
                            Text(label(f.status)).bold().foregroundColor(color(f.status)).frame(width: 14)
                            Button { showDiff(f.path) } label: { Text(f.path).lineLimit(1) }.buttonStyle(.plain)
                            Spacer()
                        }
                    }.listStyle(.plain)
                }
                TextField("Commit message", text: $message, axis: .vertical).lineLimit(1...4).textFieldStyle(.roundedBorder)
                HStack {
                    Button("Commit selected") { commit() }.disabled(message.trimmingCharacters(in: .whitespaces).isEmpty || selected.isEmpty || busy)
                    Spacer()
                    Button("Push \(g.branch ?? "")…") { push(g) }.disabled(busy || (g.remotes ?? []).isEmpty)
                        .help((g.remotes ?? []).isEmpty ? "No remote configured" : "Send commits to the remote")
                }
            } else {
                Text("Not a git repository").foregroundColor(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding(8)
        .onChange(of: ws.git?.files?.map(\.path) ?? []) { paths in selected = Set(paths) }
        .onAppear { selected = Set(ws.git?.files?.map(\.path) ?? []) }
        .sheet(isPresented: Binding(get: { diff != nil }, set: { if !$0 { diff = nil } })) {
            VStack(alignment: .leading) {
                Text(diff?.path ?? "").font(.headline)
                ScrollView([.vertical, .horizontal]) {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(Array((diff?.text ?? "").components(separatedBy: "\n").enumerated()), id: \.offset) { _, l in
                            Text(l.isEmpty ? " " : l).font(.system(size: 12, design: .monospaced))
                                .foregroundColor(l.hasPrefix("+") ? .green : l.hasPrefix("-") ? .red : .primary)
                        }
                    }.textSelection(.enabled)
                }
                HStack { Spacer(); Button("Close") { diff = nil }.keyboardShortcut(.cancelAction) }
            }.padding().frame(minWidth: 640, minHeight: 420)
        }
    }

    private func label(_ s: String) -> String { ["modified": "M", "added": "A", "deleted": "D", "renamed": "R", "untracked": "?"][s] ?? s }
    private func color(_ s: String) -> Color { s == "deleted" ? .red : s == "modified" ? .orange : .green }

    private func showDiff(_ path: String) {
        ws.guarded("Diff") {
            let r: DiffResponse = try await ws.backend.get("/api/git/diff?path=\(ws.backend.encoded(path))")
            diff = (path, r.diff.isEmpty ? "No changes" : r.diff)
        }
    }

    private func commit() {
        busy = true
        let paths = Array(selected), msg = message
        ws.guarded("Commit") {
            defer { busy = false }
            let r: CommitResponse = try await ws.backend.request("POST", "/api/git/commit", body: ["message": msg, "paths": paths])
            ws.git = r.git; message = ""
            ws.say("Committed: \(r.commit)")
        }
    }

    private func push(_ g: GitStatus) {
        let a = NSAlert()
        a.messageText = "Push \(g.branch ?? "") to \(g.remotes?.first ?? "origin")?"
        a.informativeText = "This sends your commits to the remote using your local git credentials. It cannot be undone from here."
        a.addButton(withTitle: "Push"); a.addButton(withTitle: "Cancel")
        guard a.runModal() == .alertFirstButtonReturn else { return }
        busy = true
        ws.guarded("Push") {
            defer { busy = false }
            let r: PushResponse = try await ws.backend.request("POST", "/api/git/push", body: ["confirm": true])
            ws.git = r.git
            ws.say("Pushed")
        }
    }
}

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
            HStack { AccentCaption(text: "Files", color: Palette.accent); Spacer()
                     Text("\(ws.allFiles.count)").font(.system(size: 10, design: .monospaced)).foregroundColor(.secondary) }
                .padding(.horizontal, 10).padding(.bottom, 4)
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
                                        Text(PathTools.displayName(p))
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
        VStack(spacing: 0) {
            HStack { AccentCaption(text: "Outline", color: Palette.phosphor); Spacer() }
                .padding(.horizontal, 10).padding(.vertical, 6)
            Divider()
            if let f = ws.active { OutlineList(file: f, ws: ws) }
            else { Text("Open a file to see its structure").foregroundColor(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity) }
        }
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
            VStack(spacing: 4) {
                Text("No functions, classes or headings found").foregroundColor(.secondary)
                Text("The outline lists a file's structure — it fills in for source code and Markdown.")
                    .font(.caption).foregroundColor(.secondary).multilineTextAlignment(.center)
            }.padding(20).frame(maxWidth: .infinity, maxHeight: .infinity)
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
            }.listStyle(.plain).clipped()
        }
    }
}

// MARK: - GitHub (through the `gh` command-line tool: its login, its credentials)

struct GitHubSection: View {
    @ObservedObject var ws: Workspace
    @State private var status: GitHubStatus?
    @State private var prs: [PullRequest] = []
    @State private var expanded = false
    @State private var title = ""
    @State private var body_ = ""
    @State private var draft = false
    @State private var busy = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 8) {
                if let s = status {
                    if !s.installed {
                        Text("The GitHub command-line tool is not installed.").foregroundColor(.secondary)
                        Button("Copy “brew install gh”") {
                            NSPasteboard.general.clearContents(); NSPasteboard.general.setString("brew install gh", forType: .string)
                        }
                    } else if !s.authenticated {
                        Text("Not signed in to GitHub.").foregroundColor(.secondary)
                        Button("Sign in in the terminal") {
                            ws.showPanel("terminal")
                            let t = ws.terminalSession(); t.start()
                            t.send("gh auth login --web --git-protocol https\r")
                        }
                        Button("I have signed in — check again") { reload() }
                    } else {
                        Label("Signed in as \(s.user ?? "?")" + (s.repo.map { " · \($0)" } ?? ""), systemImage: "checkmark.seal")
                            .foregroundColor(.secondary)
                        if s.repo == nil { createRepo() } else { pullRequests(s) }
                    }
                } else { ProgressView().controlSize(.small) }
            }.padding(.top, 6)
        } label: {
            Label("GitHub", systemImage: "network").font(.callout)
        }
        .onChange(of: expanded) { open in if open { reload() } }
    }

    private func reload() {
        Task {
            status = try? await ws.backend.get("/api/github/status")
            if status?.authenticated == true, status?.repo != nil {
                let r: PullRequestsResponse? = try? await ws.backend.get("/api/github/prs")
                prs = r?.prs ?? []
            }
        }
    }

    @ViewBuilder private func pullRequests(_ s: GitHubStatus) -> some View {
        if prs.isEmpty { Text("No open pull requests").font(.caption).foregroundColor(.secondary) }
        ForEach(prs) { pr in
            Button { if let u = URL(string: pr.url) { NSWorkspace.shared.open(u) } } label: {
                HStack { Text("#\(pr.number)").foregroundColor(.secondary); Text(pr.title).lineLimit(1); Spacer()
                         Text(pr.headRefName).font(.caption).foregroundColor(.secondary) }
            }.buttonStyle(.plain)
        }
        Divider()
        Text("New pull request from \(ws.git?.branch ?? "this branch")").font(.caption).foregroundColor(.secondary)
        TextField("Title", text: $title).textFieldStyle(.roundedBorder)
        TextField("Description", text: $body_, axis: .vertical).lineLimit(2...5).textFieldStyle(.roundedBorder)
        HStack {
            Toggle("Draft", isOn: $draft).toggleStyle(.checkbox)
            Spacer()
            Button("Create pull request…") { createPR() }
                .disabled(title.trimmingCharacters(in: .whitespaces).isEmpty || busy)
        }
    }

    private func createRepo() -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("This project has no GitHub remote.").font(.caption).foregroundColor(.secondary)
            Button("Create a private repository from this folder…") { newRepo() }.disabled(busy)
        }
    }

    private func createPR() {
        let a = NSAlert()
        a.messageText = "Open a pull request?"
        a.informativeText = "“\(title)” from \(ws.git?.branch ?? "this branch") — visible on GitHub as soon as it is created. Push the branch first."
        a.addButton(withTitle: "Create"); a.addButton(withTitle: "Cancel")
        guard a.runModal() == .alertFirstButtonReturn else { return }
        busy = true
        let t = title, b = body_, d = draft
        ws.guarded("Pull request") {
            defer { busy = false }
            let r: URLResponseBody = try await ws.backend.request("POST", "/api/github/pr", body: [
                "title": t, "body": b, "draft": d, "confirm": true])
            title = ""; body_ = ""
            ws.say("Pull request created")
            if let u = URL(string: r.url) { NSWorkspace.shared.open(u) }
            reload()
        }
    }

    private func newRepo() {
        let a = NSAlert()
        a.messageText = "Create a private GitHub repository"
        a.informativeText = "The folder is published as a new private repository under your account and the current branch is pushed."
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24))
        field.stringValue = ws.projectName
        a.accessoryView = field
        a.addButton(withTitle: "Create and push"); a.addButton(withTitle: "Cancel")
        guard a.runModal() == .alertFirstButtonReturn else { return }
        busy = true
        let name = field.stringValue
        ws.guarded("Create repository") {
            defer { busy = false }
            let r: CreateRepoResponse = try await ws.backend.request("POST", "/api/github/create_repo", body: [
                "name": name, "private": true, "confirm": true])
            ws.git = r.git
            ws.say("Repository created")
            if let u = URL(string: r.url) { NSWorkspace.shared.open(u) }
            reload()
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
                AccentCaption(text: "Changes", color: Palette.yellow)
                HStack {
                    Image(systemName: "arrow.triangle.branch").foregroundColor(Palette.accent)
                    Text(g.branch ?? "?").bold()
                    if let a = g.ahead, a > 0 { Text("↑\(a)").foregroundColor(.secondary) }
                    if let b = g.behind, b > 0 { Text("↓\(b)").foregroundColor(.secondary) }
                    Spacer()
                    if !(g.files ?? []).isEmpty {
                        Button { ws.discardChanges(nil) } label: { Image(systemName: "arrow.uturn.backward") }
                            .buttonStyle(.plain).help("Discard all changes…")
                    }
                    Button { Task { await ws.refreshGit() } } label: { Image(systemName: "arrow.clockwise") }
                        .buttonStyle(.plain).help("Refresh")
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
                            Button { ws.discardChanges([f.path]) } label: { Image(systemName: "arrow.uturn.backward") }
                                .buttonStyle(.plain).foregroundColor(.secondary).help("Discard changes in this file…")
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
                if ModuleSettings.shared.isOn(.github) { Divider(); GitHubSection(ws: ws) }
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
    private func color(_ s: String) -> Color { s == "deleted" ? .red : s == "modified" ? Palette.yellow : Palette.phosphor }

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

// MARK: - Extensions (Open VSX): search, install, use a theme. Only themes and snippets are used.

struct ExtensionsSheet: View {
    @ObservedObject var ws: Workspace
    @Binding var shown: Bool
    @ObservedObject private var theme = ThemeSettings.shared
    @State private var query = ""
    @State private var hits: [ExtensionHit] = []
    @State private var installed: [InstalledExtension] = []
    @State private var busy: String?
    @State private var note: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                AccentCaption(text: "Extensions · Open VSX")
                Spacer()
                if theme.custom != nil {
                    Button("Back to the built-in colours") { theme.custom = nil }.buttonStyle(.borderless)
                }
                Button("Close") { shown = false }.keyboardShortcut(.cancelAction)
            }
            Text("Colour themes and snippets are used (type a snippet's prefix and press Tab). An extension's own code is never run here.")
                .font(.caption).foregroundColor(.secondary)
            HStack {
                TextField("Search open-vsx.org (a theme name, a language…)", text: $query)
                    .textFieldStyle(.roundedBorder).onSubmit(search)
                Button("Search", action: search).disabled(query.trimmingCharacters(in: .whitespaces).isEmpty || busy != nil)
            }
            if let n = note { Text(n).font(.caption).foregroundColor(Palette.yellow) }
            List {
                if !installed.isEmpty {
                    Section("Installed") {
                        ForEach(installed) { e in installedRow(e) }
                    }
                }
                if !hits.isEmpty {
                    Section("Search results") {
                        ForEach(hits) { h in hitRow(h) }
                    }
                }
            }.listStyle(.inset)
            if let b = busy { HStack { ProgressView().controlSize(.small); Text(b).font(.caption) } }
        }
        .padding(16).frame(minWidth: 640, minHeight: 480)
        .onAppear { Task { await reload() } }
    }

    private func hitRow(_ h: ExtensionHit) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(h.displayName).bold()
                    Text(h.id + " " + h.version).font(.caption).foregroundColor(.secondary)
                    if h.verified { Image(systemName: "checkmark.seal").font(.caption).foregroundColor(Palette.accent) }
                }
                Text(h.description).font(.caption).foregroundColor(.secondary).lineLimit(2)
                Text("\(h.downloads) downloads").font(.caption2).foregroundColor(.secondary)
            }
            Spacer()
            if installed.contains(where: { $0.id == h.id }) {
                Text("Installed").font(.caption).foregroundColor(.secondary)
            } else {
                Button("Install") { install(h) }.disabled(busy != nil)
            }
        }
    }

    private func installedRow(_ e: InstalledExtension) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(e.displayName).bold()
                Text(e.id + " " + e.version).font(.caption).foregroundColor(.secondary)
                Spacer()
                Button("Remove") { remove(e) }.buttonStyle(.borderless).disabled(busy != nil)
            }
            Text(e.detail).font(.caption).foregroundColor(.secondary)
            if e.has_code || !e.skipped.isEmpty {
                let what = ([e.has_code ? "its code" : nil] + e.skipped.map { Optional($0) }).compactMap { $0 }
                Text("Not used: " + what.joined(separator: ", ")).font(.caption2).foregroundColor(.secondary)
            }
            if !e.themes.isEmpty {
                HStack {
                    ForEach(e.themes, id: \.self) { t in
                        Button(t.label) { ws.applyExtensionTheme(id: e.id, label: t.label); shown = false }
                            .buttonStyle(.bordered).controlSize(.small)
                    }
                }
            }
        }.padding(.vertical, 2)
    }

    private func reload() async {
        if let r: InstalledExtensionsResponse = try? await ws.backend.get("/api/extensions/installed") { installed = r.extensions }
    }

    private func search() {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty, busy == nil else { return }
        busy = "Searching open-vsx.org…"; note = nil
        Task {
            defer { busy = nil }
            do {
                let r: ExtensionSearchResponse = try await ws.backend.get("/api/extensions/search?q=\(ws.backend.encoded(q))")
                hits = r.results
                if r.results.isEmpty { note = "Nothing found" }
            } catch { note = error.localizedDescription }
        }
    }

    private func install(_ h: ExtensionHit) {
        busy = "Installing \(h.displayName)…"; note = nil
        Task {
            defer { busy = nil }
            do {
                try await ws.backend.post("/api/extensions/install",
                                          body: ["namespace": h.namespace, "name": h.name, "version": h.version, "confirm": true])
                await reload()
                ws.resetSnippets()
                let e = installed.first { $0.id == h.id }
                note = e.map { "\($0.displayName): " + $0.detail } ?? "Installed"
            } catch { note = error.localizedDescription }
        }
    }

    private func remove(_ e: InstalledExtension) {
        Task {
            try? await ws.backend.post("/api/extensions/remove", body: ["id": e.id])
            if theme.custom?.id == e.id { theme.custom = nil }
            ws.resetSnippets()
            await reload()
        }
    }
}

// The window: toolbar, the two layouts (chat-first and IDE), status bar, "apply suggestion" sheet.
import AppKit
import SwiftUI

struct RootView: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var chat: ChatStore
    @ObservedObject var modules: ModuleSettings
    @ObservedObject private var theme = ThemeSettings.shared

    var body: some View {
        Group {
            if ws.starting {
                VStack(spacing: 12) { ProgressView(); Text("Starting the editor backend…").foregroundColor(.secondary) }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let err = ws.fatal {
                VStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundColor(.orange)
                    Text(err)
                    Button("Try again") { ws.openProject(ws.root) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 0) {
                    Toolbar(ws: ws, modules: modules).background(Palette.panel)
                    Divider()
                    VSplitView {
                        Group {
                            if ws.mode == .chat { ChatLayout(ws: ws, chat: chat, modules: modules) }
                            else { IDELayout(ws: ws, chat: chat, modules: modules) }
                        }.frame(minHeight: 200)
                        if ws.bottomVisible {
                            BottomPanel(ws: ws, modules: modules).frame(minHeight: 120, idealHeight: 240)
                        }
                    }
                    Divider()
                    StatusBar(ws: ws)
                }
            }
        }
        .overlay(alignment: .bottomTrailing) {
            if let t = ws.toast {
                Text(t).padding(.horizontal, 12).padding(.vertical, 8)
                    .background(ws.toastBad ? Color.red.opacity(0.9) : Color.accentColor.opacity(0.9), in: RoundedRectangle(cornerRadius: 8))
                    .foregroundColor(.white).padding(16).transition(.opacity)
            }
        }
        .background(Palette.window)
        .tint(Palette.accent)
        .preferredColorScheme(theme.colorScheme)
        .animation(.easeOut(duration: 0.15), value: ws.toast)
        .onAppear {
            ws.onAsk = {                                  // a problem or terminal output was sent to the assistant
                if !modules.isOn(.chat) { modules.toggle(.chat) }
            }
        }
        .sheet(isPresented: Binding(get: { ws.pendingApply != nil }, set: { if !$0 { ws.pendingApply = nil } })) {
            ApplySheet(ws: ws)
        }
    }
}

private struct Toolbar: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var modules: ModuleSettings
    @State private var showRemote = false

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "folder").foregroundColor(.secondary)
            Text(ws.projectName).bold()
            if ws.git?.repo == true { BranchMenu(ws: ws) }
            Spacer()
            Picker("", selection: $ws.mode) {
                Text("Chat").tag(LayoutMode.chat)
                Text("Editor").tag(LayoutMode.ide)
            }.pickerStyle(.segmented).frame(width: 150).labelsHidden()
            Spacer()
            Menu {
                Button("This Mac's engine") { ws.useLocalEngine() }.disabled(ws.remoteURL == nil)
                Button("Another server (OpenRouter, other computer)…") { showRemote = true }
                if let u = ws.remoteURL { Divider(); Text("Now: \(u)") }
            } label: {
                HStack(spacing: 5) {
                    Circle().fill(ws.engine == nil ? Color.red : Color.green).frame(width: 8, height: 8)
                    if ws.remoteURL != nil { Image(systemName: "cloud").font(.caption) }
                    Text(ws.engine.map { ($0.model ?? "model") + ($0.state == "sleeping" ? " · sleeping" : "") } ?? "model not running")
                        .font(.callout).lineLimit(1)
                }.foregroundColor(.secondary)
            }.menuStyle(.borderlessButton).fixedSize().help("Where the assistant's model runs")
            .sheet(isPresented: $showRemote) { RemoteSheet(ws: ws, shown: $showRemote) }
            Button { ws.bottomVisible.toggle() } label: { Image(systemName: "rectangle.bottomthird.inset.filled") }
                .buttonStyle(.plain).help("Problems and terminal (⌘J)")
            Menu {
                ForEach(ModuleID.allCases) { m in
                    Toggle(m.title, isOn: Binding(get: { modules.isOn(m) }, set: { _ in
                        modules.toggle(m)
                        if m == .preview, modules.isOn(.preview) { ws.showPanel("preview") }      // switching it on shows it
                    })).disabled(m == .files)
                }
            } label: { Image(systemName: "square.grid.2x2") }.menuStyle(.borderlessButton).fixedSize().help("Modules")
            Menu {
                Picker("Theme", selection: Binding(get: { ThemeSettings.shared.mode }, set: { ThemeSettings.shared.mode = $0 })) {
                    ForEach(ThemeMode.allCases) { Text($0.title).tag($0) }
                }.pickerStyle(.inline)
            } label: { Image(systemName: "circle.lefthalf.filled") }.menuStyle(.borderlessButton).fixedSize().help("Theme")
        }.padding(.horizontal, 12).padding(.vertical, 7)
    }
}

/// The current branch; a click lists the others (local, then remote ones that have no local copy yet).
private struct BranchMenu: View {
    @ObservedObject var ws: Workspace

    var body: some View {
        Menu {
            if let b = ws.branches {
                ForEach(b.local, id: \.self) { name in
                    Button { if name != b.current { ws.checkout(name) } } label: {
                        if name == b.current { Label(name, systemImage: "checkmark") } else { Text(name) }
                    }
                }
                let extra = b.remote.filter { r in !b.local.contains { r.hasSuffix("/" + $0) } }
                if !extra.isEmpty {
                    Divider()
                    ForEach(extra, id: \.self) { r in Button(r) { ws.checkout(r) } }
                }
            }
            Divider()
            Button("New branch…") { ws.newBranch() }
        } label: {
            Label(ws.git?.branch ?? ws.branches?.current ?? "branch", systemImage: "arrow.triangle.branch")
        }
        .menuStyle(.borderlessButton).fixedSize().font(.callout)
        .onAppear { Task { await ws.loadBranches() } }
        .help("Switch or create a branch")
    }
}

private struct StatusBar: View {
    @ObservedObject var ws: Workspace
    var body: some View {
        HStack(spacing: 14) {
            if let f = ws.active { FileStatus(file: f, ws: ws) }
            Button { ws.showPanel("problems") } label: { ProblemSummary(ws: ws) }.buttonStyle(.plain)
            Spacer()
            let dirty = ws.open.filter(\.dirty).count
            if dirty > 0 { Text("\(dirty) unsaved").foregroundColor(.orange) }
        }.font(.caption).foregroundColor(.secondary).padding(.horizontal, 12).frame(height: 22)
    }
}

struct ProblemSummary: View {
    @ObservedObject var ws: Workspace
    var body: some View {
        let s = ws.problemSummary
        HStack(spacing: 8) {
            Label("\(s.errors)", systemImage: "xmark.octagon").foregroundColor(s.errors > 0 ? .red : .secondary)
            Label("\(s.warnings)", systemImage: "exclamationmark.triangle").foregroundColor(s.warnings > 0 ? .yellow : .secondary)
        }.labelStyle(.titleAndIcon)
    }
}

// MARK: - bottom panel: problems + terminal (each only when its module is on)

private struct BottomPanel: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var modules: ModuleSettings

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                if modules.isOn(.problems) {
                    tab("problems", "Problems" + (ws.problemSummary.errors + ws.problemSummary.warnings > 0
                                                  ? " \(ws.problemSummary.errors + ws.problemSummary.warnings)" : ""))
                }
                if modules.isOn(.terminal) { tab("terminal", "Terminal") }
                if modules.isOn(.preview) { tab("preview", "Preview") }
                Spacer()
                Button { ws.bottomVisible = false } label: { Image(systemName: "xmark") }.buttonStyle(.plain)
            }.padding(.horizontal, 10).padding(.vertical, 5).background(Color.primary.opacity(0.04))
            Divider()
            if ws.bottomTab == "terminal", modules.isOn(.terminal) {
                TerminalPanel(ws: ws)
            } else if ws.bottomTab == "preview", modules.isOn(.preview) {
                PreviewPanel(ws: ws)
            } else if modules.isOn(.problems) {
                ProblemsView(ws: ws)
            } else {
                Text("Problems and Terminal are switched off (Modules menu)").foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private func tab(_ id: String, _ title: String) -> some View {
        Button { ws.bottomTab = id } label: {
            Text(title).fontWeight(ws.bottomTab == id ? .semibold : .regular)
                .foregroundColor(ws.bottomTab == id ? .primary : .secondary)
        }.buttonStyle(.plain)
    }
}

private struct TerminalPanel: View {
    @ObservedObject var ws: Workspace
    var body: some View {
        let s = ws.terminalSession()
        VStack(spacing: 0) {
            TerminalModuleView(session: s)
            Divider()
            HStack {
                Spacer()
                Button { ws.askAboutTerminal(s.buffer.text) } label: { Label("Ask the assistant about this output", systemImage: "sparkles") }
                    .buttonStyle(.borderless)
            }.font(.caption).padding(.horizontal, 10).padding(.vertical, 4)
        }
    }
}

struct ProblemsView: View {
    @ObservedObject var ws: Workspace

    var body: some View {
        let total = ws.problemSummary.errors + ws.problemSummary.warnings
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                if ws.open.isEmpty {
                    Text("Open a file to see its problems").foregroundColor(.secondary).padding(14)
                } else if total == 0 {
                    Label("No problems in the open files", systemImage: "checkmark.circle").foregroundColor(.green).padding(14)
                }
                ForEach(ws.open) { f in FileProblems(ws: ws, file: f) }
            }
        }
    }
}

private struct FileProblems: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var file: OpenFile

    var body: some View {
        if !file.diagnostics.isEmpty {
            HStack {
                Image(systemName: "doc.text")
                Text(file.path).bold()
                Text("\(file.diagnostics.count)").foregroundColor(.secondary)
                Spacer()
                Button { ws.askAboutProblems(of: file) } label: { Label("Ask the assistant", systemImage: "sparkles") }
                    .buttonStyle(.borderless)
            }.font(.callout).padding(.horizontal, 10).padding(.top, 8)
            ForEach(file.diagnostics) { d in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: d.isError ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                        .foregroundColor(d.isError ? .red : .yellow)
                    Text(d.message).lineLimit(2)
                    Spacer()
                    Text("\(d.source) · \(d.line):\(d.col)").font(.caption).foregroundColor(.secondary)
                    Button { ws.askAboutProblem(d, in: file) } label: { Image(systemName: "sparkles") }
                        .buttonStyle(.borderless).help("Send this problem to the assistant")
                }
                .padding(.horizontal, 10).padding(.vertical, 3).contentShape(Rectangle())
                .onTapGesture { ws.activePath = file.path; file.jumpLine = d.line }
            }
        }
    }
}

private struct FileStatus: View {
    @ObservedObject var file: OpenFile
    @ObservedObject var ws: Workspace
    var body: some View {
        Text("\(file.language)  ·  \(ws.cursor.line):\(ws.cursor.col)")
    }
}

// MARK: - layouts

private struct ChatLayout: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var chat: ChatStore
    @ObservedObject var modules: ModuleSettings

    var body: some View {
        HSplitView {
            SessionsModuleView(ws: ws, store: chat).frame(minWidth: 170, idealWidth: 210, maxWidth: 300)
            Group {
                if modules.isOn(.chat) { ChatModuleView(ws: ws, store: chat) }
                else { Text("The assistant module is switched off (Modules menu)").foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity) }
            }.frame(minWidth: 360)
            ChatRightPanel(ws: ws, modules: modules).frame(minWidth: 260, idealWidth: 380)
        }
    }
}

private struct ChatRightPanel: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var modules: ModuleSettings

    var body: some View {
        if let f = ws.active {
            VStack(spacing: 0) {
                HStack {
                    Button { ws.activePath = nil } label: { Image(systemName: "chevron.left") }.buttonStyle(.plain)
                    Text(Paths.displayName(f.path)).bold().lineLimit(1)
                    DirtyMark(file: f)
                    Spacer()
                    Button { ws.save() } label: { Image(systemName: "square.and.arrow.down") }.buttonStyle(.plain).help("Save (⌘S)")
                    Button { ws.attach(f.path) } label: { Image(systemName: "paperclip") }.buttonStyle(.plain).help("Attach to the next message")
                    Button { ws.mode = .ide } label: { Image(systemName: "arrow.up.right.square") }.buttonStyle(.plain).help("Open in the editor layout")
                    Button { ws.closeFile(f.path) } label: { Image(systemName: "xmark") }.buttonStyle(.plain)
                }.padding(8)
                Divider()
                EditorSurface(ws: ws, file: f)
            }
        } else {
            FilesModuleView(ws: ws)
        }
    }
}

private struct DirtyMark: View {
    @ObservedObject var file: OpenFile
    var body: some View { Text(file.dirty ? "●" : "").foregroundColor(.orange) }
}

private struct IDELayout: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var chat: ChatStore
    @ObservedObject var modules: ModuleSettings
    @State private var leftTab = 0

    var body: some View {
        HSplitView {
            VStack(spacing: 0) {
                if modules.isOn(.git) {
                    Picker("", selection: $leftTab) { Text("Files").tag(0); Text("Changes").tag(1) }
                        .pickerStyle(.segmented).labelsHidden().padding(6)
                    Divider()
                }
                if leftTab == 0 || !modules.isOn(.git) { FilesModuleView(ws: ws) } else { GitModuleView(ws: ws) }
            }.frame(minWidth: 190, idealWidth: 250, maxWidth: 400)
            EditorArea(ws: ws).frame(minWidth: 380)
            if modules.isOn(.outline) || modules.isOn(.chat) {
                VStack(spacing: 0) {
                    if modules.isOn(.outline) {
                        OutlineModuleView(ws: ws).frame(minHeight: 120, idealHeight: 220)
                        Divider()
                    }
                    if modules.isOn(.chat) { ChatModuleView(ws: ws, store: chat) }
                }.frame(minWidth: 280, idealWidth: 360)
            }
        }
    }
}

private struct EditorArea: View {
    @ObservedObject var ws: Workspace

    var body: some View {
        VStack(spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 0) {
                    ForEach(ws.open) { f in TabButton(file: f, ws: ws) }
                }
            }.frame(height: 30).background(Color.primary.opacity(0.04))
            Divider()
            if let f = ws.active { EditorSurface(ws: ws, file: f) }
            else { Text("Pick a file on the left").foregroundColor(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity) }
        }
    }
}

private struct TabButton: View {
    @ObservedObject var file: OpenFile
    @ObservedObject var ws: Workspace
    var body: some View {
        HStack(spacing: 6) {
            Text(Paths.displayName(file.path) + (file.dirty ? " ●" : "")).lineLimit(1)
            Button { ws.closeFile(file.path) } label: { Image(systemName: "xmark").font(.caption2) }.buttonStyle(.plain)
        }
        .padding(.horizontal, 10).frame(height: 30)
        .background(ws.activePath == file.path ? Color.accentColor.opacity(0.18) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture { ws.activePath = file.path }
    }
}

private struct EditorSurface: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var file: OpenFile
    var body: some View {
        CodeEditor(file: file,
                   onEdit: { f in Task { await ws.analyze(f) } },
                   onCursor: { l, c in ws.cursor = (l, c) })
            .id(file.path)
    }
}

// MARK: - applying a suggested code block

private struct ApplySheet: View {
    @ObservedObject var ws: Workspace
    @State private var target: String = ""

    private var file: OpenFile? { ws.open.first { $0.path == target } }

    var body: some View {
        let pending = ws.pendingApply
        VStack(alignment: .leading, spacing: 10) {
            Text("Apply suggestion").font(.headline)
            HStack {
                Text("File:")
                Picker("", selection: $target) {
                    ForEach(ws.open) { Text($0.path).tag($0.path) }
                }.labelsHidden()
            }
            if let f = file, let p = pending {
                let d = LineDiff.diff(old: f.text, new: p.code), st = LineDiff.stats(d)
                Text("Replaces the whole file (\(st.added) added, \(st.removed) removed). Nothing is written to disk until you save.")
                    .font(.callout).foregroundColor(.secondary)
                DiffView(lines: d)
            } else {
                Text("Open the file this code belongs to first.").foregroundColor(.secondary)
            }
            HStack {
                Spacer()
                Button("Cancel") { ws.pendingApply = nil }.keyboardShortcut(.cancelAction)
                Button("Apply to editor") {
                    if let f = file, let p = pending { f.text = p.code; f.version += 1; ws.activePath = f.path
                        Task { await ws.analyze(f) } }
                    ws.pendingApply = nil
                }.keyboardShortcut(.defaultAction).disabled(file == nil)
            }
        }
        .padding(16).frame(minWidth: 620, minHeight: 360)
        .onAppear {
            if let h = pending?.hint, ws.open.contains(where: { $0.path == h }) { target = h }
            else if let h = pending?.hint { ws.openFile(h); target = ws.activePath ?? "" }
            else { target = ws.activePath ?? "" }
        }
    }
}

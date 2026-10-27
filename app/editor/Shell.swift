// The window: toolbar, the two layouts (chat-first and IDE), status bar, "apply suggestion" sheet.
import AppKit
import SwiftUI

struct RootView: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var chat: ChatStore
    @ObservedObject var modules: ModuleSettings
    @ObservedObject private var theme = ThemeSettings.shared
    @ObservedObject private var prefs = ViewPrefs.shared
    @State private var bottomHeight = SplitPrefs.get("editor.bottom.height", 240)

    var body: some View {
        Group {
            if ws.needsProject {
                WelcomeView(ws: ws)
            } else if ws.starting {
                VStack(spacing: 12) { ProgressView(); Text("Starting the editor backend…").foregroundColor(.secondary) }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let err = ws.fatal {
                VStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundColor(Palette.yellow)
                    Text(err)
                    Button("Try again") { ws.openProject(ws.root) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 0) {
                    Toolbar(ws: ws, modules: modules).background(Palette.panel)
                    Divider()
                    GeometryReader { g in
                        VStack(spacing: 0) {
                            Group {
                                if ws.mode == .chat { ChatLayout(ws: ws, chat: chat, modules: modules) }
                                else { IDELayout(ws: ws, chat: chat, modules: modules) }
                            }.frame(maxWidth: .infinity, maxHeight: .infinity)
                            if ws.bottomVisible {
                                SplitHandle(axis: .rows, size: $bottomHeight, range: 120...max(120, g.size.height - 200),
                                            growsWithDrag: false, key: "editor.bottom.height")
                                BottomPanel(ws: ws, modules: modules)
                                    .frame(height: SplitPrefs.clamp(bottomHeight, 120, g.size.height - 200))
                            }
                        }
                    }
                    if prefs.statusBar {
                        Divider()
                        StatusBar(ws: ws)
                    }
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
        .sheet(isPresented: Binding(get: { ws.palette != nil }, set: { if !$0 { ws.palette = nil } })) {
            PaletteSheet(ws: ws)
        }
        .sheet(isPresented: Binding(get: { ws.pendingApply != nil }, set: { if !$0 { ws.pendingApply = nil } })) {
            ApplySheet(ws: ws)
        }
    }
}

/// What a window with no folder shows (a plain launch, or a window opened from LLMBar without a project):
/// the name, a button, and the folders used before.
struct WelcomeView: View {
    @ObservedObject var ws: Workspace

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "chevron.left.forwardslash.chevron.right")
                .font(.system(size: 34, weight: .semibold)).foregroundColor(Palette.accent)
            Text("LLM Editor").font(.system(size: 22, weight: .semibold))
            Text("A native editor for your local model — open a folder to begin.")
                .foregroundColor(.secondary)
            Button { ws.onChooseProject?() } label: {
                Label("Open a folder…", systemImage: "folder").padding(.horizontal, 10).padding(.vertical, 3)
            }.buttonStyle(.borderedProminent).tint(Palette.send).keyboardShortcut("o")
            let recent = Workspace.recentProjects
            if !recent.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    AccentCaption(text: "Recent", color: Palette.accent)
                    ForEach(recent.prefix(6), id: \.self) { p in
                        Button { ws.onOpenProject?(p) } label: {
                            HStack {
                                Image(systemName: "folder.fill").foregroundColor(Palette.accent)
                                VStack(alignment: .leading, spacing: 0) {
                                    Text((p as NSString).lastPathComponent)
                                    Text((p as NSString).abbreviatingWithTildeInPath).font(.caption).foregroundColor(.secondary)
                                }
                                Spacer()
                            }.padding(8).contentShape(Rectangle())
                        }.buttonStyle(.plain)
                            .background(Palette.field, in: RoundedRectangle(cornerRadius: 8))
                    }
                }.frame(width: 380).padding(.top, 8)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            // a first launch: offer the picker right away, as a sheet on this window
            if Workspace.recentProjects.isEmpty { DispatchQueue.main.async { ws.onChooseProject?() } }
        }
    }
}

private struct Toolbar: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var modules: ModuleSettings
    @State private var showRemote = false
    @State private var showMCP = false

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "folder.fill").foregroundColor(Palette.accent)
            Text(ws.projectName).bold()
            if ws.git?.repo == true { BranchMenu(ws: ws) }
            Spacer()
            Picker("", selection: $ws.mode) {
                Text("Chat").tag(LayoutMode.chat)
                Text("Editor").tag(LayoutMode.ide)
            }.pickerStyle(.segmented).frame(width: 150).labelsHidden()
            Spacer()
            Menu {
                if ws.remoteURL == nil {
                    if ws.engine == nil {
                        Button("Start the model") { ws.startModel() }.disabled(ws.engineBusy != nil)
                    } else {
                        Button("Stop the model") { ws.stopModel() }.disabled(ws.engineBusy != nil)
                    }
                    let favourites = ws.installed.filter { $0.favorite == true }
                    if !favourites.isEmpty {
                        Section("Favourites") {
                            ForEach(favourites) { m in
                                Button { ws.startModel(m.path ?? m.name) } label: {
                                    Label(m.name, systemImage: m.name == ws.engine?.model ? "checkmark" : "star.fill")
                                }
                            }
                        }.disabled(ws.engineBusy != nil)
                    }
                    Menu("Choose a model") {
                        if ws.installed.isEmpty { Text("No models found") }
                        ForEach(ws.installed) { m in
                            Button { ws.startModel(m.path ?? m.name) } label: {
                                let size = m.size.map { String(format: "  ·  %.1f GB", $0) } ?? ""
                                let star = m.favorite == true ? "★ " : ""
                                if m.name == ws.engine?.model { Label(star + m.name + size, systemImage: "checkmark") }
                                else { Text(star + m.name + size) }
                            }
                        }
                    }.disabled(ws.engineBusy != nil)
                    Menu("Favourites…") {
                        ForEach(ws.installed) { m in
                            Toggle(m.name, isOn: Binding(get: { m.favorite == true }, set: { ws.setFavorite(m, $0) }))
                        }
                    }
                    Divider()
                }
                Button("This Mac's engine") { ws.useLocalEngine() }.disabled(ws.remoteURL == nil)
                Button("Another server (OpenRouter, other computer)…") { showRemote = true }
                if let u = ws.remoteURL { Divider(); Text("Now: \(u)") }
            } label: {
                HStack(spacing: 5) {
                    // filled while the model answers, an empty ring while it is off
                    ZStack {
                        Circle().strokeBorder(Palette.magenta, lineWidth: 1.3).opacity(ws.engine == nil ? 1 : 0)
                        Circle().fill(Palette.phosphor).shadow(color: Palette.phosphor.opacity(0.7), radius: 3)
                            .opacity(ws.engine == nil ? 0 : 1)
                    }.frame(width: 8, height: 8)
                    if ws.remoteURL != nil { Image(systemName: "cloud").font(.caption) }
                    if ws.engineBusy != nil { ProgressView().controlSize(.mini) }
                    Text(ws.engineBusy ?? ws.engine.map { ($0.model ?? "model") + ($0.state == "sleeping" ? " · sleeping" : "") }
                         ?? "model not running")
                        .font(.callout).lineLimit(1)
                }.foregroundColor(.secondary)
            }.menuStyle(.borderlessButton).fixedSize().help("Where the assistant's model runs")
            .sheet(isPresented: $showRemote) { RemoteSheet(ws: ws, shown: $showRemote) }
            .onAppear { Task { await ws.loadInstalled() } }
            Button { ws.bottomVisible.toggle() } label: { Image(systemName: "rectangle.bottomthird.inset.filled") }
                .buttonStyle(.plain).foregroundColor(Palette.accent)          // the same accent as the Modules menu beside it
                .help("Problems and terminal (⌘J)")
            Menu {
                ForEach(ModuleID.allCases) { m in
                    Toggle(m.title, isOn: Binding(get: { modules.isOn(m) }, set: { _ in
                        modules.toggle(m)
                        if m == .preview, modules.isOn(.preview) { ws.showPanel("preview") }      // switching it on shows it
                    })).disabled(m == .files)
                }
                if modules.isOn(.mcp) {
                    Divider()
                    Button("Assistant Tools (MCP)…") { showMCP = true }
                }
            } label: { Image(systemName: "square.grid.2x2") }.menuStyle(.borderlessButton).fixedSize().help("Modules")
            .sheet(isPresented: $showMCP) { MCPSheet(ws: ws, shown: $showMCP) }
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
            if dirty > 0 { Text("\(dirty) unsaved").foregroundColor(Palette.yellow) }
        }.font(.caption).foregroundColor(.secondary).padding(.horizontal, 12).frame(height: 22)
    }
}

struct ProblemSummary: View {
    @ObservedObject var ws: Workspace
    var body: some View {
        let s = ws.problemSummary
        HStack(spacing: 8) {
            Label("\(s.errors)", systemImage: "xmark.octagon").foregroundColor(s.errors > 0 ? Palette.magenta : .secondary)
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
                                                  ? " \(ws.problemSummary.errors + ws.problemSummary.warnings)" : ""),
                        color: ws.problemSummary.errors > 0 ? Palette.magenta : Palette.accent)
                }
                if modules.isOn(.terminal) { tab("terminal", "Terminal", color: Palette.phosphor) }
                if modules.isOn(.preview) { tab("preview", "Preview", color: Palette.accent) }
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

    private func tab(_ id: String, _ title: String, color: Color = Palette.accent) -> some View {
        Button { ws.bottomTab = id } label: {
            Text(title.uppercased()).font(.system(size: 10, weight: .semibold, design: .monospaced)).tracking(1.4)
                .foregroundColor(ws.bottomTab == id ? color : .secondary)
                .padding(.vertical, 3)
                .overlay(alignment: .bottom) { Rectangle().fill(ws.bottomTab == id ? color : .clear).frame(height: 1.5) }
        }.buttonStyle(.plain)
    }
}

private struct TerminalPanel: View {
    @ObservedObject var ws: Workspace
    private var shell: String { ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/zsh" }

    /// A slim strip of terminals: one per tab, the shown one underlined, and a plus for another shell.
    private var tabs: some View {
        HStack(spacing: 4) {
            ForEach(Array(ws.terminalTabs.serials.enumerated()), id: \.element) { i, _ in
                let on = i == ws.terminalTabs.active
                HStack(spacing: 4) {
                    Text(ws.terminalTabs.title(i, shell: shell)).font(.system(size: 10, design: .monospaced))
                        .foregroundColor(on ? .primary : .secondary)
                    if ws.terminalTabs.count > 1 {
                        Button { ws.closeTerminal(at: i) } label: { Image(systemName: "xmark").font(.system(size: 8, weight: .bold)) }
                            .buttonStyle(.plain).foregroundColor(.secondary).help("Close this terminal")
                    }
                }
                .padding(.horizontal, 8).padding(.vertical, 3)
                .background(RoundedRectangle(cornerRadius: 5).fill(on ? Color.primary.opacity(0.08) : .clear))
                .contentShape(Rectangle())
                .onTapGesture { ws.terminalTabs.select(i) }
            }
            Button { ws.newTerminal() } label: { Image(systemName: "plus").font(.system(size: 10, weight: .semibold)) }
                .buttonStyle(.plain).foregroundColor(.secondary).help("New terminal")
            Spacer()
        }.padding(.horizontal, 8).padding(.vertical, 3)
    }

    var body: some View {
        let s = ws.terminalSession()
        VStack(spacing: 0) {
            tabs
            Divider()
            TerminalModuleView(session: s).id(ws.terminalTabs.activeSerial)
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
                    Label("No problems in the open files", systemImage: "checkmark.circle")
                        .foregroundColor(.primary).padding(14)
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
                        .foregroundColor(d.isError ? Palette.magenta : Palette.yellow)
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
    @State private var leftWidth = SplitPrefs.get("editor.chat.left", 210)
    @State private var rightWidth = SplitPrefs.get("editor.chat.right", 380)

    var body: some View {
        GeometryReader { g in
            let left = SplitPrefs.clamp(leftWidth, 150, min(320, g.size.width * 0.3))
            let right = SplitPrefs.clamp(rightWidth, 240, g.size.width - left - 360 - 14)
            HStack(spacing: 0) {
                SessionsModuleView(ws: ws, store: chat).frame(width: left)
                SplitHandle(axis: .columns, size: $leftWidth, range: 150...max(150, min(320, g.size.width * 0.3)),
                            key: "editor.chat.left")
                Group {
                    if modules.isOn(.chat) { ChatModuleView(ws: ws, store: chat) }
                    else { Text("The assistant module is switched off (Modules menu)").foregroundColor(.secondary)
                            .frame(maxWidth: .infinity, maxHeight: .infinity) }
                }.frame(minWidth: 360, maxWidth: .infinity)
                SplitHandle(axis: .columns, size: $rightWidth, range: 240...max(240, g.size.width - left - 360 - 14),
                            growsWithDrag: false, key: "editor.chat.right")
                ChatRightPanel(ws: ws, modules: modules).frame(width: right)
            }
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
                    Text(PathTools.displayName(f.path)).bold().lineLimit(1)
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
    var body: some View { Text(file.dirty ? "●" : "").foregroundColor(Palette.yellow) }
}

private struct IDELayout: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var chat: ChatStore
    @ObservedObject var modules: ModuleSettings
    @State private var leftWidth = SplitPrefs.get("editor.ide.left", 250)
    @State private var rightWidth = SplitPrefs.get("editor.ide.right", 360)
    @State private var outlineHeight = SplitPrefs.get("editor.ide.outline", 150)

    var body: some View {
        GeometryReader { g in
            let hasRight = modules.isOn(.outline) || modules.isOn(.chat)
            let left = SplitPrefs.clamp(leftWidth, 190, min(400, g.size.width * 0.3))
            let right = SplitPrefs.clamp(rightWidth, 280, g.size.width - left - 380 - 14)
            HStack(spacing: 0) {
                VStack(spacing: 0) {
                    if modules.isOn(.git) {
                        Picker("", selection: $ws.leftTab) { Text("Files").tag(0); Text("Changes").tag(1) }
                            .pickerStyle(.segmented).labelsHidden().padding(6)
                        Divider()
                    }
                    if ws.leftTab == 0 || !modules.isOn(.git) { FilesModuleView(ws: ws) } else { GitModuleView(ws: ws) }
                }.frame(width: left)
                SplitHandle(axis: .columns, size: $leftWidth, range: 190...max(190, min(400, g.size.width * 0.3)),
                            key: "editor.ide.left")
                EditorArea(ws: ws).frame(minWidth: 380, maxWidth: .infinity)
                if hasRight {
                    SplitHandle(axis: .columns, size: $rightWidth, range: 280...max(280, g.size.width - left - 380 - 14),
                                growsWithDrag: false, key: "editor.ide.right")
                    GeometryReader { rg in
                        VStack(spacing: 0) {
                            if modules.isOn(.outline) {
                                OutlineModuleView(ws: ws)
                                    .frame(height: modules.isOn(.chat)
                                           ? SplitPrefs.clamp(outlineHeight, 70, rg.size.height - 160) : rg.size.height)
                                    .clipped()
                            }
                            if modules.isOn(.outline) && modules.isOn(.chat) {
                                SplitHandle(axis: .rows, size: $outlineHeight, range: 70...max(70, rg.size.height - 160),
                                            key: "editor.ide.outline")
                            }
                            if modules.isOn(.chat) {
                                ChatModuleView(ws: ws, store: chat, showSwitcher: true)
                                    .frame(maxWidth: .infinity, maxHeight: .infinity).clipped()
                            }
                        }
                    }.frame(width: right)
                }
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
            Text(PathTools.displayName(file.path) + (file.dirty ? " ●" : "")).lineLimit(1)
            Button { ws.closeFile(file.path) } label: { Image(systemName: "xmark").font(.caption2) }.buttonStyle(.plain)
        }
        .padding(.horizontal, 10).frame(height: 30)
        .background(ws.activePath == file.path ? Palette.accent.opacity(0.14) : Color.clear)
        .overlay(alignment: .bottom) {
            Rectangle().fill(ws.activePath == file.path ? Palette.accent : Color.clear).frame(height: 2)
        }
        .contentShape(Rectangle())
        .onTapGesture { ws.activePath = file.path }
    }
}

private struct EditorSurface: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var file: OpenFile
    var body: some View {
        VStack(spacing: 0) {
            if let note = file.readOnlyNote {
                HStack(spacing: 6) {
                    Image(systemName: "lock")
                    Text(note)
                    Spacer()
                    Button("Open in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: ws.root).appendingPathComponent(file.path)])
                    }.buttonStyle(.borderless)
                }.font(.caption).padding(.horizontal, 10).padding(.vertical, 4).background(Palette.accent.opacity(0.10))
                Divider()
            }
            CodeEditor(file: file,
                       onEdit: { f in Task { await ws.analyze(f) } },
                       onCursor: { l, c in ws.cursor = (l, c) })
                .id(file.path)
        }
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

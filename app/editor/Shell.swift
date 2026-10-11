// The window: toolbar, the two layouts (chat-first and IDE), status bar, "apply suggestion" sheet.
import AppKit
import SwiftUI

struct RootView: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var chat: ChatStore
    @ObservedObject var modules: ModuleSettings

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
                    Toolbar(ws: ws, modules: modules)
                    Divider()
                    if ws.mode == .chat { ChatLayout(ws: ws, chat: chat, modules: modules) }
                    else { IDELayout(ws: ws, chat: chat, modules: modules) }
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
        .animation(.easeOut(duration: 0.15), value: ws.toast)
        .sheet(isPresented: Binding(get: { ws.pendingApply != nil }, set: { if !$0 { ws.pendingApply = nil } })) {
            ApplySheet(ws: ws)
        }
    }
}

private struct Toolbar: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var modules: ModuleSettings

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "folder").foregroundColor(.secondary)
            Text(ws.projectName).bold()
            if let b = ws.git?.branch, ws.git?.repo == true {
                Label(b, systemImage: "arrow.triangle.branch").font(.callout).foregroundColor(.secondary)
            }
            Spacer()
            Picker("", selection: $ws.mode) {
                Text("Chat").tag(LayoutMode.chat)
                Text("Editor").tag(LayoutMode.ide)
            }.pickerStyle(.segmented).frame(width: 150).labelsHidden()
            Spacer()
            HStack(spacing: 5) {
                Circle().fill(ws.engine == nil ? Color.red : Color.green).frame(width: 8, height: 8)
                Text(ws.engine.map { ($0.model ?? "model") + ($0.state == "sleeping" ? " · sleeping" : "") } ?? "model not running")
                    .font(.callout).foregroundColor(.secondary).lineLimit(1)
            }
            Menu {
                ForEach(ModuleID.allCases) { m in
                    Toggle(m.title, isOn: Binding(get: { modules.isOn(m) }, set: { _ in modules.toggle(m) })).disabled(m == .files)
                }
            } label: { Image(systemName: "square.grid.2x2") }.menuStyle(.borderlessButton).fixedSize().help("Modules")
        }.padding(.horizontal, 12).padding(.vertical, 7)
    }
}

private struct StatusBar: View {
    @ObservedObject var ws: Workspace
    var body: some View {
        HStack(spacing: 14) {
            if let f = ws.active { FileStatus(file: f, ws: ws) }
            Spacer()
            let dirty = ws.open.filter(\.dirty).count
            if dirty > 0 { Text("\(dirty) unsaved").foregroundColor(.orange) }
        }.font(.caption).foregroundColor(.secondary).padding(.horizontal, 12).frame(height: 22)
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

// LLMBar ▸ Tools ▸ Knowledge: the document libraries, managed inside the panel — create, open, add files or a
// folder, remove a document, rename or delete a library, and try a question to see what a chat would be given.
// Everything goes through KnowledgeStore (app/knowledge), i.e. the `llm knowledge` CLI; the page holds nothing
// once it is closed, and with no knowledge database on disk it runs nothing at all.
import AppKit
import SwiftUI

struct KnowledgePage: View {
    @ObservedObject var e: EngineState
    /// Snapshots only (`--module panel --tab 4 --page knowledge --library … --query …`): what to show opened.
    static var preset: (library: String, query: String)?

    @ObservedObject private var store = KnowledgeStore.shared
    @State private var openID: Int?
    @State private var docs: [KnowledgeDocument] = []
    @State private var query = ""
    @State private var hits: [KnowledgeHit] = []
    @State private var searched = false
    @State private var loaded = false

    private var library: KnowledgeLibrary? { store.libraries.first { $0.id == openID } }

    var body: some View {
        VStack(spacing: 10) {
            Card {
                HStack {
                    Button(action: goBack) {
                        HStack(spacing: 4) {
                            Image(systemName: "chevron.left").font(.system(size: 10, weight: .semibold))
                            Text(library == nil ? L("Tools", "Тулзы") : L("Libraries", "Библиотеки")).font(HUD.mono(10, .semibold))
                        }
                        .padding(.vertical, 4).padding(.trailing, 10)
                        .contentShape(Rectangle())           // the whole label is clickable, not only the glyphs
                    }.buttonStyle(.plain).foregroundColor(HUD.cyan)
                    Spacer()
                }
                Caption(text: library?.name ?? "Knowledge")
                status
                if let lib = library { libraryView(lib) } else { librariesView }
            }
            if let lib = library, lib.ready > 0 { tryQuestion(lib) }
        }
        .task {
            await store.refresh(force: true)
            if let p = KnowledgePage.preset {
                openID = store.libraries.first { $0.name == p.library }?.id
                query = p.query
            }
            loaded = true
            await reload()
            if !query.isEmpty, let id = openID { hits = await store.search(query, libraries: [id]); searched = true }
        }
        .onChange(of: openID) { _ in hits = []; searched = false; Task { await reload() } }
        .onReceive(NotificationCenter.default.publisher(for: .knowledgeChanged)) { _ in Task { await reload() } }
    }

    /// One level up from what is shown: an open library → the list; the list → the Tools rows. Decided by what is
    /// on screen (`library`), so a remembered id of a library that is gone cannot swallow the click.
    private func goBack() {
        if library != nil {
            openID = nil
        } else {
            openID = nil
            e.toolsPage = nil
        }
    }

    // ── status: indexing (with Stop) or the last outcome
    @ViewBuilder private var status: some View {
        if let busy = store.busy {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(busy).font(.system(size: 11)).foregroundColor(HUD.text)
                Spacer()
                Button(L("Stop", "Стоп")) { store.stopIndexing() }.buttonStyle(.plain).font(HUD.mono(10, .semibold))
                    .foregroundColor(HUD.magenta)
            }
        } else if let m = store.message {
            Text(m).font(.system(size: 10.5)).foregroundColor(HUD.dim).lineLimit(3).fixedSize(horizontal: false, vertical: true)
        }
    }

    // ── all libraries
    @ViewBuilder private var librariesView: some View {
        if store.libraries.isEmpty {
            Text(loaded ? L("No libraries yet. A library is a set of your documents that a chat can search — PDFs, notes, "
                            + "code. Nothing is indexed, searched or loaded until you add some, and your files are never changed.",
                            "Библиотек пока нет. Библиотека — это набор ваших документов, в которых может искать чат: PDF, "
                            + "заметки, код. Пока вы их не добавили, ничего не индексируется, не ищется и не загружается, "
                            + "а ваши файлы никогда не меняются.") : "…")
                .font(.system(size: 11)).foregroundColor(HUD.dim).fixedSize(horizontal: false, vertical: true)
        } else {
            ForEach(store.libraries) { lib in
                Button(action: { openID = lib.id }) {
                    HStack(spacing: 10) {
                        Image(systemName: "books.vertical").font(.system(size: 14)).foregroundColor(HUD.cyan).frame(width: 22)
                            .shadow(color: HUD.cyan.opacity(0.7), radius: 3)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(lib.name).font(.system(size: 13, weight: .medium)).foregroundColor(HUD.text).lineLimit(1)
                            Text(lib.documents == 0 ? L("empty", "пусто")
                                 : L("\(lib.ready) of \(lib.documents) documents ready", "готово \(lib.ready) из \(lib.documents) документов"))
                                .font(.system(size: 10)).foregroundColor(HUD.dim).lineLimit(1)
                        }
                        Spacer()
                        Image(systemName: "chevron.right").font(.system(size: 10)).foregroundColor(HUD.dim)
                    }.contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .contextMenu {
                    Button(L("Rename…", "Переименовать…")) { rename(lib) }
                    Button(L("Delete…", "Удалить…")) { delete(lib) }
                }
            }
        }
        HStack {
            Button(action: newLibrary) {
                Label(L("New library", "Новая библиотека"), systemImage: "plus").font(HUD.mono(10, .semibold))
                    .padding(.horizontal, 10).padding(.vertical, 5)
            }.buttonStyle(NeonButton(color: HUD.cyan)).fixedSize()
            Spacer()
        }.padding(.top, 2)
    }

    // ── one library: its documents and what can be done with it
    @ViewBuilder private func libraryView(_ lib: KnowledgeLibrary) -> some View {
        HStack(spacing: 6) {
            neon(L("+ Files", "+ Файлы"), HUD.cyan) { pick(folders: false, into: lib.id) }
            neon(L("+ Folder", "+ Папка"), HUD.cyan) { pick(folders: true, into: lib.id) }
            Spacer()
            Button(action: { rename(lib) }) {
                Image(systemName: "pencil").font(.system(size: 11, weight: .semibold)).padding(.horizontal, 8).padding(.vertical, 5)
            }.buttonStyle(NeonButton(color: HUD.dim)).fixedSize().help(L("Rename the library", "Переименовать библиотеку"))
            neon(L("Delete", "Удалить"), HUD.magenta) { delete(lib) }
        }
        Text(lib.documents == 0 ? L("No documents yet: PDF, Markdown, text and source files, or a whole folder. "
                                    + "Credentials (.env, keys) and ignored files are skipped; scanned pages are read "
                                    + "with on-device OCR.",
                                    "Документов пока нет: PDF, Markdown, текст и исходники или целая папка. Учётные данные "
                                    + "(.env, ключи) и игнорируемые файлы пропускаются; сканы читаются OCR на устройстве.")
                                 : L("\(lib.ready) of \(lib.documents) documents ready", "готово \(lib.ready) из \(lib.documents) документов"))
            .font(.system(size: 10.5)).foregroundColor(HUD.dim).fixedSize(horizontal: false, vertical: true)
        ForEach(docs) { d in documentRow(d, library: lib.id) }
    }

    private func documentRow(_ d: KnowledgeDocument, library: Int) -> some View {
        HStack(spacing: 8) {
            Image(systemName: d.state == "ready" ? (d.changed ? "exclamationmark.circle" : "doc.text")
                                                 : d.state == "failed" ? "xmark.octagon" : "hourglass")
                .font(.system(size: 12))
                .foregroundColor(d.state == "failed" ? HUD.magenta : d.changed ? HUD.amber : HUD.cyan)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 1) {
                Text(d.name).font(.system(size: 12)).foregroundColor(HUD.text).lineLimit(1).truncationMode(.middle)
                Text(detail(d)).font(.system(size: 9.5)).foregroundColor(HUD.dim).lineLimit(2)
            }
            Spacer(minLength: 4)
            Button(action: { Task { _ = await store.remove(doc: d.id, library: library) } }) {
                Image(systemName: "xmark").font(.system(size: 9, weight: .semibold))
            }.buttonStyle(.plain).foregroundColor(HUD.dim)
                .help(L("Remove from this library (the file stays)", "Убрать из библиотеки (файл останется)"))
        }
        .contextMenu {
            Button(L("Show in Finder", "Показать в Finder")) {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: d.path)])
            }.disabled(d.missing)
            if d.changed && !d.missing {
                Button(L("Index again", "Проиндексировать заново")) { Task { _ = await store.add([d.path], library: library) } }
            }
            Button(L("Remove from library", "Убрать из библиотеки")) { Task { _ = await store.remove(doc: d.id, library: library) } }
        }
    }

    /// The line under a document's name, in the panel's language (the index's own notes stay as written).
    private func detail(_ d: KnowledgeDocument) -> String {
        var parts: [String] = []
        switch d.state {
        case "ready":
            parts.append(d.pages > 0 ? L("\(d.pages) p. · \(d.chunks) chunks", "\(d.pages) стр. · \(d.chunks) фрагм.")
                                     : L("\(d.chunks) chunks", "\(d.chunks) фрагм."))
        case "failed": parts.append(L("not indexed", "не проиндексирован"))
        default: parts.append(L("indexing…", "индексируется…"))
        }
        if d.missing { parts.append(L("original is missing", "исходного файла нет")) }
        else if d.changed { parts.append(L("changed since indexing", "изменён после индексации")) }
        if !d.note.isEmpty { parts.append(d.note) }
        return parts.joined(separator: " · ")
    }

    /// What a chat would be given for a question, and from where — the way to check a library works.
    private func tryQuestion(_ lib: KnowledgeLibrary) -> some View {
        Card(glow: HUD.phosphor) {
            Caption(text: L("Try a question", "Проверить вопрос"), color: HUD.phosphor)
            TextField(L("What would a chat be given?", "Что получит чат?"), text: $query)
                .textFieldStyle(.plain).font(.system(size: 12)).foregroundColor(HUD.text)
                .padding(.horizontal, 8).padding(.vertical, 5)
                .background(RoundedRectangle(cornerRadius: 7).fill(HUD.track))
                .onSubmit { Task { hits = await store.search(query, libraries: [lib.id]); searched = true } }
            if searched {
                if hits.isEmpty {
                    Text(L("Nothing matches — a chat would get no extra context.", "Совпадений нет — чат не получит дополнительного контекста."))
                        .font(.system(size: 10.5)).foregroundColor(HUD.dim)
                } else {
                    ForEach(hits.prefix(3)) { h in
                        VStack(alignment: .leading, spacing: 1) {
                            Text(h.label).font(HUD.mono(9.5, .semibold)).foregroundColor(HUD.phosphor).lineLimit(1)
                            Text(h.text.replacingOccurrences(of: "\n", with: " ")).font(.system(size: 10.5))
                                .foregroundColor(HUD.text).lineLimit(3)
                        }
                    }
                }
            }
        }
    }

    // ── actions
    private func neon(_ title: String, _ color: Color, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title).font(HUD.mono(10, .semibold)).lineLimit(1).padding(.horizontal, 8).padding(.vertical, 5)
        }.buttonStyle(NeonButton(color: color)).fixedSize()
    }

    private func reload() async {
        guard let id = openID else { docs = []; return }
        docs = await store.documents(library: id)
    }

    /// The panel hides when another window takes focus (a file dialog, an alert); it comes back afterwards so
    /// the outcome — and indexing progress — is in front of the person.
    private func comeBack() { AppHost.openPanel() }

    private func pick(folders: Bool, into library: Int) {
        let p = NSOpenPanel()
        p.canChooseFiles = !folders
        p.canChooseDirectories = folders
        p.allowsMultipleSelection = true
        p.prompt = L("Add", "Добавить")
        let ok = p.runModal() == .OK
        comeBack()
        guard ok else { return }
        let paths = p.urls.map(\.path)
        Task { _ = await store.add(paths, library: library) }
    }

    private func ask(_ title: String, _ initial: String = "", button: String) -> String? {
        let a = NSAlert()
        a.messageText = title
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24))
        field.stringValue = initial
        a.accessoryView = field
        a.addButton(withTitle: button)
        a.addButton(withTitle: L("Cancel", "Отмена"))
        a.window.initialFirstResponder = field
        let ok = a.runModal() == .alertFirstButtonReturn
        comeBack()
        let t = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        return ok && !t.isEmpty ? t : nil
    }

    private func newLibrary() {
        guard let name = ask(L("New library", "Новая библиотека"), button: L("Create", "Создать")) else { return }
        Task { if await store.create(name) { openID = store.libraries.first { $0.name == name }?.id } }
    }

    private func rename(_ lib: KnowledgeLibrary) {
        guard let name = ask(L("Rename library", "Переименовать библиотеку"), lib.name, button: L("Rename", "Переименовать"))
        else { return }
        Task { _ = await store.rename(lib.id, to: name) }
    }

    private func delete(_ lib: KnowledgeLibrary) {
        let a = NSAlert()
        a.alertStyle = .warning
        a.messageText = L("Delete “\(lib.name)”?", "Удалить «\(lib.name)»?")
        a.informativeText = L("Its index is removed. Your original files stay where they are; chats that searched it stop doing so.",
                              "Удаляется её индекс. Исходные файлы остаются на месте; чаты, которые в ней искали, перестанут.")
        a.addButton(withTitle: L("Delete", "Удалить"))
        a.addButton(withTitle: L("Cancel", "Отмена"))
        let ok = a.runModal() == .alertFirstButtonReturn
        comeBack()
        guard ok else { return }
        Task { if await store.delete(lib.id) { openID = nil } }
    }
}

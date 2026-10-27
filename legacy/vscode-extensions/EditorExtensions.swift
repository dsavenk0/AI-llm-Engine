// ARCHIVED — legacy VS Code (Open VSX) extensions UI, removed from the active editor on 2026-10-27
// and replaced by the MCP tools panel. This file is NOT compiled (build-app.sh builds app/** only).
// Kept as reference; the Python backend is legacy/vscode-extensions/extensions.py. The types below
// reference live app types (Workspace, Palette, ThemeSettings, backend) and will not build standalone.

// ─────────── from app/editor/Backend.swift (response/model structs) ───────────
struct ExtensionHit: Decodable, Identifiable, Hashable {
    let id: String, namespace: String, name: String, version: String
    let displayName: String, description: String
    let downloads: Int
    let verified: Bool
}
struct ExtensionSearchResponse: Decodable { let results: [ExtensionHit] }
struct ExtensionTheme: Decodable, Hashable { let label: String, uiTheme: String }
struct ExtensionSnippetFile: Decodable, Hashable { let language: String }
struct ExtensionSnippetItem: Decodable, Hashable { let language: String, prefix: String, description: String }
struct ExtensionDetail: Decodable {
    let themes: [String], languages: [String]
    let snippets: [ExtensionSnippetItem]
    let snippet_total: Int
    let homepage: String?
    let folder: String
}
struct ExtensionUpdate: Decodable, Hashable { let id: String, installed: String, latest: String }
struct ExtensionUpdatesResponse: Decodable { let updates: [ExtensionUpdate] }

struct InstalledExtension: Decodable, Identifiable, Hashable {
    let id: String, name: String, displayName: String, version: String, description: String
    let themes: [ExtensionTheme], snippets: [ExtensionSnippetFile]
    let skipped: [String]
    let has_code: Bool
    var enabled: Bool? = nil
    var usable: Bool? = nil
    var homepage: String? = nil
    var isEnabled: Bool { enabled ?? true }
    var detail: String {
        var parts: [String] = []
        if !themes.isEmpty { parts.append("\(themes.count) theme\(themes.count == 1 ? "" : "s")") }
        if !snippets.isEmpty { parts.append("snippets: " + Set(snippets.map(\.language)).sorted().joined(separator: ", ")) }
        if parts.isEmpty { parts.append("nothing this editor can use") }
        return parts.joined(separator: " · ")
    }
}
struct InstalledExtensionsResponse: Decodable { let extensions: [InstalledExtension] }
struct InstallExtensionResponse: Decodable { let extension_: InstalledExtension
    enum CodingKeys: String, CodingKey { case extension_ = "extension" } }
struct SnippetsResponse: Decodable { let snippets: [Snippet] }

// ─────────── from app/editor/Logic.swift (theme override + snippet engine) ───────────
// MARK: - extensions from Open VSX: a colour theme and snippets (declarative parts only)

/// A VS Code theme converted by the backend: background/foreground and a colour per token kind.
struct ThemeOverride: Codable, Equatable {
    var label: String
    var id: String
    var dark: Bool
    var background: String
    var foreground: String
    var kinds: [String: String]

    /// "#rrggbb" → 0…1 components; nil for anything else.
    static func rgb(_ hex: String) -> (Double, Double, Double)? {
        guard hex.count == 7, hex.hasPrefix("#"), let v = Int(hex.dropFirst(), radix: 16) else { return nil }
        return (Double((v >> 16) & 0xFF) / 255, Double((v >> 8) & 0xFF) / 255, Double(v & 0xFF) / 255)
    }
}

struct Snippet: Codable, Equatable, Hashable {
    let prefix: String
    let body: [String]
    let description: String
}

enum SnippetExpander {
    /// A VS Code snippet body → plain text plus the caret position (UTF-16 offset into the text): at the first
    /// tab stop ($1, ${1:default}), else at $0, else at the end. Variables ($TM_FILENAME…) expand to nothing —
    /// or to their default. `indent` is put in front of every line after the first.
    static func expand(_ body: [String], indent: String = "") -> (text: String, caret: Int) {
        let src = Array(body.joined(separator: "\n" + indent))
        var out = "", stops: [Int: Int] = [:], i = 0
        parse(src, &i, nil, &out, &stops, 0)
        let first = stops.keys.filter { $0 > 0 }.min()
        return (out, first.flatMap { stops[$0] } ?? stops[0] ?? out.utf16.count)
    }

    private static func parse(_ s: [Character], _ i: inout Int, _ close: Character?, _ out: inout String,
                              _ stops: inout [Int: Int], _ depth: Int) {
        func stop(_ n: Int) { if stops[n] == nil { stops[n] = out.utf16.count } }
        while i < s.count {
            let c = s[i]
            if let close, c == close { return }
            if c == "\\", i + 1 < s.count, "$}\\".contains(s[i + 1]) { out.append(s[i + 1]); i += 2; continue }
            guard c == "$", i + 1 < s.count else { out.append(c); i += 1; continue }
            let n = s[i + 1]
            if n.isASCII, n.isNumber {                                  // $1
                var j = i + 1, num = ""
                while j < s.count, s[j].isASCII, s[j].isNumber { num.append(s[j]); j += 1 }
                if let k = Int(num) { stop(k) }
                i = j
            } else if n == "{" {
                var j = i + 2, name = ""
                while j < s.count, s[j].isLetter || s[j].isNumber || s[j] == "_" { name.append(s[j]); j += 1 }
                if j < s.count, s[j] == "}" {                           // ${1} / ${VAR}
                    if let k = Int(name) { stop(k) }
                    i = j + 1
                } else if j < s.count, s[j] == ":" {                    // ${1:default}
                    if let k = Int(name) { stop(k) }
                    var k = j + 1
                    if depth < 8 { parse(s, &k, "}", &out, &stops, depth + 1) }
                    i = k < s.count ? k + 1 : k
                } else if j < s.count, s[j] == "|" {                    // ${1|one,two|}: the first choice
                    if let k = Int(name) { stop(k) }
                    j += 1
                    var first = "", seenComma = false
                    while j < s.count, s[j] != "|" {
                        if s[j] == "," { seenComma = true } else if !seenComma { first.append(s[j]) }
                        j += 1
                    }
                    out += first
                    while j < s.count, s[j] != "}" { j += 1 }
                    i = min(j + 1, s.count)
                } else { out.append(c); i += 1 }
            } else if n.isLetter || n == "_" {                          // $TM_FILENAME …
                var j = i + 1
                while j < s.count, s[j].isLetter || s[j].isNumber || s[j] == "_" { j += 1 }
                i = j
            } else { out.append(c); i += 1 }
        }
    }
}

enum SnippetMatch {
    private static func isWord(_ c: Character) -> Bool { c.isLetter || c.isNumber || c == "_" }

    /// The snippet (the longest prefix wins) that ends `lineBeforeCaret` as a whole word.
    static func find(in lineBeforeCaret: String, among snippets: [Snippet]) -> Snippet? {
        let chars = Array(lineBeforeCaret)
        var best: Snippet?
        for s in snippets where !s.prefix.isEmpty && lineBeforeCaret.hasSuffix(s.prefix) {
            let start = chars.count - s.prefix.count
            if start > 0, isWord(chars[start - 1]), let f = s.prefix.first, isWord(f) { continue }
            if best == nil || s.prefix.count > best!.prefix.count { best = s }
        }
        return best
    }
}

// ─────────── from app/editor/Workspace.swift (snippet/theme loaders) ───────────
    // MARK: extensions (Open VSX): themes and snippets

    /// Snippets for a language, asked once per language while the module is on.
    func loadSnippets(for language: String) {
        guard ModuleSettings.shared.isOn(.extensions), !language.isEmpty, !snippetsAsked.contains(language) else { return }
        snippetsAsked.insert(language)
        Task {
            if let r: SnippetsResponse = try? await backend.get("/api/extensions/snippets?language=\(backend.encoded(language))") {
                snippets[language] = r.snippets
            }
        }
    }

    /// Forget what was loaded (after an install or removal) so the next file asks again.
    func resetSnippets() { snippets = [:]; snippetsAsked = [] }

    func applyExtensionTheme(id: String, label: String) {
        guarded("Theme") {
            let t: ThemeOverride = try await self.backend.get(
                "/api/extensions/theme?id=\(self.backend.encoded(id))&label=\(self.backend.encoded(label))")
            ThemeSettings.shared.mode = t.dark ? .dark : .light
            ThemeSettings.shared.custom = t
            self.say("Theme: \(t.label)")
        }
    }

// ─────────── from app/editor/CodeEditor.swift (Tab snippet expansion) ───────────
        /// A snippet prefix before the caret + Tab → the snippet body, caret at its first tab stop.
        private func expandSnippet(in tv: NSTextView) -> Bool {
            let snippets = parent.snippets
            let sel = tv.selectedRange()
            guard !snippets.isEmpty, sel.length == 0 else { return false }
            let ns = tv.string as NSString
            let lineStart = ns.lineRange(for: NSRange(location: sel.location, length: 0)).location
            let before = ns.substring(with: NSRange(location: lineStart, length: sel.location - lineStart))
            guard let snip = SnippetMatch.find(in: before, among: snippets) else { return false }
            let indent = String(before.prefix { $0 == " " || $0 == "\t" })
            let (text, caret) = SnippetExpander.expand(snip.body, indent: indent)
            let prefixLen = (snip.prefix as NSString).length
            let range = NSRange(location: sel.location - prefixLen, length: prefixLen)
            guard tv.shouldChangeText(in: range, replacementString: text) else { return false }
            tv.replaceCharacters(in: range, with: text)
            tv.didChangeText()
            tv.setSelectedRange(NSRange(location: range.location + caret, length: 0))
            return true
        }

// ─────────── from app/editor/Panels.swift (the Extensions sheet UI) ───────────
// MARK: - Extensions (Open VSX): search, install, use a theme. Only themes and snippets are used.

struct ExtensionsSheet: View {
    @ObservedObject var ws: Workspace
    @Binding var shown: Bool
    @ObservedObject private var theme = ThemeSettings.shared
    @State private var query = ""
    @State private var category = ""                       // "", Themes, Snippets, Programming Languages
    @State private var hits: [ExtensionHit] = []
    @State private var installed: [InstalledExtension] = []
    @State private var busy: String?
    @State private var note: String?
    @State private var updates: [String: String] = [:]     // id -> the newer version on Open VSX
    @State private var open: Set<String> = []              // installed extensions whose details are shown
    @State private var details: [String: ExtensionDetail] = [:]

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
            VStack(alignment: .leading, spacing: 4) {
                Text("Here an extension gives its colour themes and snippets (type a snippet's prefix and press Tab) — its own code is never run.")
                    .font(.caption).foregroundColor(.secondary)
                Text("Extensions that bring their own code (agents, debuggers) cannot run here — the assistant of this "
                     + "editor does that job; what an extension adds as themes and snippets works.")
                    .font(.caption).foregroundColor(.secondary)
            }
            HStack {
                TextField("Search open-vsx.org (a theme name, a language…)", text: $query)
                    .textFieldStyle(.roundedBorder).onSubmit(search)
                Picker("", selection: $category) {
                    Text("All").tag("")
                    Text("Themes").tag("Themes")
                    Text("Snippets").tag("Snippets")
                    Text("Languages").tag("Programming Languages")
                }.pickerStyle(.segmented).labelsHidden().frame(width: 270)
                Button("Search", action: search).disabled(query.trimmingCharacters(in: .whitespaces).isEmpty || busy != nil)
            }
            if let n = note { Text(n).font(.caption).foregroundColor(Palette.yellow) }
            List {
                if !installed.isEmpty {
                    Section {
                        ForEach(installed) { e in installedRow(e) }
                    } header: {
                        HStack {
                            Text("Installed")
                            Spacer()
                            Button("Check for updates", action: checkUpdates).buttonStyle(.borderless).disabled(busy != nil)
                        }
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
        .padding(16).frame(minWidth: 700, minHeight: 520)
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
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Toggle("", isOn: Binding(get: { e.isEnabled }, set: { setEnabled(e, $0) }))
                    .toggleStyle(.switch).labelsHidden().controlSize(.small)
                    .help(e.isEnabled ? "On: its themes and snippets are available" : "Off: kept installed, not used")
                Text(e.displayName).bold().foregroundColor(e.isEnabled ? .primary : .secondary)
                Text(e.id + " " + e.version).font(.caption).foregroundColor(.secondary)
                Spacer()
                if let latest = updates[e.id] {
                    Button("Update to \(latest)") { update(e) }.buttonStyle(.bordered).controlSize(.small).disabled(busy != nil)
                }
                Button(open.contains(e.id) ? "Hide" : "Details") { toggleDetails(e) }.buttonStyle(.borderless)
                Button("Remove") { remove(e) }.buttonStyle(.borderless).disabled(busy != nil)
            }
            Text(e.detail).font(.caption).foregroundColor(.secondary)
            if e.usable == false {
                Text("Nothing here this editor can use — it is made of its own code" + (e.has_code ? "" : " or UI") +
                     ". Remove it, or use it in VS Code.").font(.caption2).foregroundColor(Palette.yellow)
            } else if e.has_code || !e.skipped.isEmpty {
                let what = ([e.has_code ? "its code" : nil] + e.skipped.map { Optional($0) }).compactMap { $0 }
                Text("Not used: " + what.joined(separator: ", ")).font(.caption2).foregroundColor(.secondary)
            }
            if e.isEnabled, !e.themes.isEmpty {
                HStack {
                    ForEach(e.themes, id: \.self) { t in
                        Button(t.label) { ws.applyExtensionTheme(id: e.id, label: t.label); shown = false }
                            .buttonStyle(.bordered).controlSize(.small)
                    }
                }
            }
            if open.contains(e.id) { detailView(e) }
        }.padding(.vertical, 2)
    }

    private func detailView(_ e: InstalledExtension) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let d = details[e.id] {
                if !d.themes.isEmpty { Text("Themes: " + d.themes.joined(separator: ", ")).font(.caption) }
                if !d.languages.isEmpty { Text("Languages: " + d.languages.joined(separator: ", ")).font(.caption) }
                if d.snippets.isEmpty { Text("No snippets").font(.caption).foregroundColor(.secondary) }
                else {
                    Text("Snippets — type the prefix, press Tab (\(d.snippet_total)):").font(.caption)
                    ForEach(Array(d.snippets.prefix(25).enumerated()), id: \.offset) { _, sn in
                        HStack(spacing: 6) {
                            Text(sn.prefix).font(.system(size: 11, design: .monospaced)).foregroundColor(Palette.accent)
                            Text(sn.language).font(.caption2).foregroundColor(.secondary)
                            Text(sn.description).font(.caption2).foregroundColor(.secondary).lineLimit(1)
                        }
                    }
                    if d.snippet_total > 25 { Text("… and \(d.snippet_total - 25) more").font(.caption2).foregroundColor(.secondary) }
                }
                HStack(spacing: 12) {
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: d.folder)])
                    }.buttonStyle(.borderless).font(.caption)
                    if let h = d.homepage, let url = URL(string: h) {
                        Button("Open its page") { NSWorkspace.shared.open(url) }.buttonStyle(.borderless).font(.caption)
                    }
                }
            } else {
                ProgressView().controlSize(.small)
            }
        }
        .padding(.leading, 40).padding(.top, 2)
    }

    private func reload() async {
        if let r: InstalledExtensionsResponse = try? await ws.backend.get("/api/extensions/installed") { installed = r.extensions }
    }

    private func toggleDetails(_ e: InstalledExtension) {
        if open.contains(e.id) { open.remove(e.id); return }
        open.insert(e.id)
        Task {
            if let d: ExtensionDetail = try? await ws.backend.get("/api/extensions/detail?id=\(ws.backend.encoded(e.id))") {
                details[e.id] = d
            }
        }
    }

    private func setEnabled(_ e: InstalledExtension, _ on: Bool) {
        Task {
            try? await ws.backend.post("/api/extensions/enable", body: ["id": e.id, "enabled": on])
            if !on, theme.custom?.id == e.id { theme.custom = nil }          // its theme goes with it
            ws.resetSnippets()
            await reload()
        }
    }

    private func checkUpdates() {
        busy = "Asking open-vsx.org for newer versions…"; note = nil
        Task {
            defer { busy = nil }
            do {
                let r: ExtensionUpdatesResponse = try await ws.backend.get("/api/extensions/updates")
                updates = Dictionary(uniqueKeysWithValues: r.updates.map { ($0.id, $0.latest) })
                note = r.updates.isEmpty ? "Everything is up to date" : nil
            } catch { note = error.localizedDescription }
        }
    }

    private func update(_ e: InstalledExtension) {
        let parts = e.id.split(separator: ".", maxSplits: 1).map(String.init)
        guard parts.count == 2, let latest = updates[e.id] else { return }
        busy = "Updating \(e.displayName)…"; note = nil
        Task {
            defer { busy = nil }
            do {
                try await ws.backend.post("/api/extensions/install",
                                          body: ["namespace": parts[0], "name": parts[1], "version": latest, "confirm": true])
                updates[e.id] = nil
                details[e.id] = nil
                await reload()
                ws.resetSnippets()
                if theme.custom?.id == e.id { theme.custom = nil }
                note = "\(e.displayName) is now \(latest)"
            } catch { note = error.localizedDescription }
        }
    }

    private func search() {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty, busy == nil else { return }
        busy = "Searching open-vsx.org…"; note = nil
        Task {
            defer { busy = nil }
            do {
                let cat = category.isEmpty ? "" : "&category=\(ws.backend.encoded(category))"
                let r: ExtensionSearchResponse = try await ws.backend.get("/api/extensions/search?q=\(ws.backend.encoded(q))\(cat)")
                hits = r.results
                if r.results.isEmpty { note = "Nothing found" + (category.isEmpty ? "" : " in this category") }
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
            open.remove(e.id); details[e.id] = nil; updates[e.id] = nil
            ws.resetSnippets()
            await reload()
        }
    }
}

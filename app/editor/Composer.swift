// The assistant's message box: Enter sends, Shift+Enter breaks the line; files from anywhere (or the
// open one), images (picked, pasted or dropped), and dictation. Attachments are plain values the
// Workspace holds until the message is sent.
import AppKit
import AVFoundation
import Speech
import SwiftUI
import UniformTypeIdentifiers

// MARK: - attachments

struct InlineFile: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let text: String
}

struct Picture: Identifiable {
    let id = UUID()
    let name: String
    let dataURL: String
    let thumb: NSImage
}

enum Attachment {
    case project(String)
    case inline(InlineFile)
    case picture(Picture)
}

struct AttachError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

enum Attach {
    static let maxText = 200_000
    static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "gif", "webp", "heic", "tiff", "bmp"]
    private static let direct: [String: String] = ["png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                                                   "gif": "image/gif", "webp": "image/webp"]

    /// A file the user picked, dropped or pasted: inside the project it is sent by path (the backend
    /// reads it), anywhere else its text travels with the message; images become data URLs.
    static func make(_ url: URL, root: String) throws -> Attachment {
        let ext = url.pathExtension.lowercased()
        if imageExtensions.contains(ext) { return .picture(try picture(url)) }
        let real = url.resolvingSymlinksInPath().path
        if real.hasPrefix(root + "/") {
            let rel = String(real.dropFirst(root.count + 1))
            if !rel.split(separator: "/").contains(".git") { return .project(rel) }
        }
        guard let data = try? Data(contentsOf: url) else { throw AttachError(message: "Cannot read \(url.lastPathComponent)") }
        if data.count > maxText { throw AttachError(message: "\(url.lastPathComponent) is larger than \(maxText / 1000) KB") }
        if data.prefix(8192).contains(0) { throw AttachError(message: "\(url.lastPathComponent) is a binary file") }
        guard let text = String(data: data, encoding: .utf8) else {
            throw AttachError(message: "\(url.lastPathComponent) is not UTF-8 text")
        }
        return .inline(InlineFile(name: url.lastPathComponent, text: text))
    }

    static func picture(_ url: URL) throws -> Picture {
        guard let img = NSImage(contentsOf: url) else { throw AttachError(message: "\(url.lastPathComponent) is not an image") }
        let ext = url.pathExtension.lowercased()
        if let mime = direct[ext], let raw = try? Data(contentsOf: url), raw.count <= 4_000_000 {
            return Picture(name: url.lastPathComponent, dataURL: "data:\(mime);base64,\(raw.base64EncodedString())",
                           thumb: thumbnail(img))
        }
        return try picture(img, name: url.deletingPathExtension().lastPathComponent + ".jpg")
    }

    /// Re-encoded as JPEG no wider than 2048 px: the engine downsizes anyway and the request stays small.
    static func picture(_ img: NSImage, name: String) throws -> Picture {
        let maxSide: CGFloat = 2048
        var size = img.size
        if max(size.width, size.height) > maxSide {
            let k = maxSide / max(size.width, size.height)
            size = NSSize(width: size.width * k, height: size.height * k)
        }
        let target = NSImage(size: size)
        target.lockFocus()
        img.draw(in: NSRect(origin: .zero, size: size), from: .zero, operation: .copy, fraction: 1)
        target.unlockFocus()
        guard let tiff = target.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
              let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.88]) else {
            throw AttachError(message: "Cannot encode \(name)")
        }
        return Picture(name: name, dataURL: "data:image/jpeg;base64,\(jpeg.base64EncodedString())", thumb: thumbnail(img))
    }

    static func thumbnail(_ img: NSImage) -> NSImage {
        let h: CGFloat = 40
        let w = max(1, img.size.width / max(1, img.size.height) * h)
        let t = NSImage(size: NSSize(width: w, height: h))
        t.lockFocus()
        img.draw(in: NSRect(x: 0, y: 0, width: w, height: h), from: .zero, operation: .copy, fraction: 1)
        t.unlockFocus()
        return t
    }
}

// MARK: - the text box

final class ComposerNSTextView: NSTextView {
    var onSend: (() -> Void)?
    var onImage: ((NSImage) -> Void)?
    var onFiles: (([URL]) -> Void)?
    var onCycleMode: (() -> Void)?
    var placeholder = ""

    override func keyDown(with e: NSEvent) {
        if e.keyCode == 48, e.modifierFlags.intersection([.shift, .option, .control, .command]) == .shift {
            onCycleMode?()                                    // Shift+Tab: the next mode
            return
        }
        let isReturn = e.keyCode == 36 || e.keyCode == 76
        if isReturn, !hasMarkedText() {                       // not while an input method is composing
            let mods = e.modifierFlags.intersection([.shift, .option, .control, .command])
            if mods.isEmpty || mods == .command { onSend?(); return }
            insertNewlineIgnoringFieldEditor(nil)             // Shift+Enter / Option+Enter: a new line
            return
        }
        super.keyDown(with: e)
    }

    override func paste(_ sender: Any?) {
        let pb = NSPasteboard.general
        if let urls = pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL], !urls.isEmpty {
            onFiles?(urls); return
        }
        if pb.string(forType: .string) == nil, let img = NSImage(pasteboard: pb) { onImage?(img); return }
        super.paste(sender)
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard string.isEmpty, !placeholder.isEmpty else { return }
        placeholder.draw(at: NSPoint(x: textContainerInset.width + 5, y: textContainerInset.height),
                         withAttributes: [.font: font ?? NSFont.systemFont(ofSize: 13),
                                          .foregroundColor: NSColor.secondaryLabelColor])
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        if let urls = sender.draggingPasteboard.readObjects(forClasses: [NSURL.self],
                                                            options: [.urlReadingFileURLsOnly: true]) as? [URL], !urls.isEmpty {
            onFiles?(urls); return true
        }
        return super.performDragOperation(sender)
    }
}

struct ComposerTextView: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String
    var onSend: () -> Void
    var onImage: (NSImage) -> Void
    var onFiles: ([URL]) -> Void
    var onCycleMode: () -> Void = {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        let tv = ComposerNSTextView()
        tv.isRichText = false
        tv.allowsUndo = true
        tv.font = .systemFont(ofSize: 13.5)
        tv.drawsBackground = false
        tv.textContainerInset = NSSize(width: 4, height: 6)
        tv.isHorizontallyResizable = false
        tv.isVerticallyResizable = true
        tv.autoresizingMask = [.width]
        tv.textContainer?.widthTracksTextView = true
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isAutomaticDashSubstitutionEnabled = false
        tv.placeholder = placeholder
        tv.registerForDraggedTypes([.fileURL])
        tv.delegate = context.coordinator
        scroll.documentView = tv
        context.coordinator.tv = tv
        wire(tv, context.coordinator)
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let tv = context.coordinator.tv else { return }
        wire(tv, context.coordinator)
        if tv.string != text { tv.string = text; tv.needsDisplay = true }
    }

    private func wire(_ tv: ComposerNSTextView, _ c: Coordinator) {
        tv.onSend = { [weak c] in c?.parent.onSend() }
        tv.onImage = { [weak c] in c?.parent.onImage($0) }
        tv.onFiles = { [weak c] in c?.parent.onFiles($0) }
        tv.onCycleMode = { [weak c] in c?.parent.onCycleMode() }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ComposerTextView
        weak var tv: ComposerNSTextView?
        init(_ p: ComposerTextView) { parent = p }
        func textDidChange(_ n: Notification) {
            guard let tv else { return }
            parent.text = tv.string
            tv.needsDisplay = true
        }
    }
}

// MARK: - dictation

/// Speech to text through the system recogniser (on-device when the language supports it). Nothing is
/// started until the microphone button is pressed.
@MainActor
final class Dictation: ObservableObject {
    @Published var listening = false
    @Published var note: String?
    private var engine: AVAudioEngine?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    static var localeID: String {
        get { UserDefaults.standard.string(forKey: "dictationLocale") ?? (Locale.preferredLanguages.first ?? "en-US") }
        set { UserDefaults.standard.set(newValue, forKey: "dictationLocale") }
    }

    func toggle(base: String, onText: @escaping (String) -> Void) {
        if listening { stop(); return }
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                guard status == .authorized else {
                    self.note = "Speech recognition is not allowed (System Settings → Privacy & Security)"; return
                }
                AVCaptureDevice.requestAccess(for: .audio) { ok in
                    DispatchQueue.main.async {
                        if ok { self.start(base: base, onText: onText) }
                        else { self.note = "Microphone access is not allowed (System Settings → Privacy & Security)" }
                    }
                }
            }
        }
    }

    private func start(base: String, onText: @escaping (String) -> Void) {
        let locale = Locale(identifier: Dictation.localeID)
        guard let rec = SFSpeechRecognizer(locale: locale), rec.isAvailable else {
            note = "Speech recognition is not available for \(locale.identifier)"; return
        }
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        if rec.supportsOnDeviceRecognition { req.requiresOnDeviceRecognition = true }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buf, _ in req.append(buf) }
        do { try engine.start() } catch { note = "Cannot use the microphone: \(error.localizedDescription)"; return }
        self.engine = engine
        request = req
        listening = true
        note = nil
        let prefix = base.isEmpty || base.hasSuffix(" ") || base.hasSuffix("\n") ? base : base + " "
        task = rec.recognitionTask(with: req) { [weak self] result, error in
            DispatchQueue.main.async {
                if let r = result { onText(prefix + r.bestTranscription.formattedString) }
                if error != nil || result?.isFinal == true { self?.stop() }
            }
        }
    }

    func stop() {
        engine?.stop()
        engine?.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        engine = nil; request = nil; task = nil
        listening = false
    }
}

// MARK: - the composer

struct Composer: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var store: ChatStore
    @Binding var draft: String
    @StateObject private var dictation = Dictation()

    private var canSend: Bool { !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            chips
            ComposerTextView(text: $draft, placeholder: "Just chat, or describe a task…",
                             onSend: send, onImage: { addImage($0) }, onFiles: { addFiles($0) },
                             onCycleMode: { store.mode = store.mode.next })
                .frame(height: 74)
                .help("Enter — send · Shift+Enter — new line · Shift+Tab — next mode")
                .padding(4)
                .background(Palette.field, in: RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Palette.hairline))
            HStack(spacing: 12) {
                Menu {
                    Button("File from disk…") { pickFiles(images: false) }
                    Button("Image…") { pickFiles(images: true) }
                    Button("Current file" + (ws.activePath.map { " — " + PathTools.displayName($0) } ?? "")) {
                        if let p = ws.activePath { ws.attach(p) }
                    }.disabled(ws.activePath == nil)
                } label: { Label("Attach", systemImage: "paperclip") }
                    .menuStyle(.borderlessButton).fixedSize()
                Button { dictation.toggle(base: draft) { draft = $0 } } label: {
                    Image(systemName: dictation.listening ? "mic.fill" : "mic")
                        .foregroundColor(dictation.listening ? .red : .secondary)
                }.buttonStyle(.plain).help(dictation.listening ? "Stop dictation" : "Dictate (\(Dictation.localeID))")
                Button { store.web.toggle() } label: {
                    Image(systemName: "globe").foregroundColor(store.web ? Palette.accent : .secondary)
                }.buttonStyle(.plain)
                    .help(store.web ? "Web search is on: the assistant may search the internet and read pages (click to turn off)"
                                    : "Let the assistant search the internet and read web pages")
                Spacer()
                contextButton
                modeButton
                if store.busy {
                    Button("Stop") { store.stop() }.buttonStyle(.bordered)
                } else {
                    Button(action: send) {
                        Label("Send", systemImage: "arrow.up").labelStyle(.titleAndIcon)
                    }
                    .buttonStyle(.borderedProminent).tint(Palette.send).disabled(!canSend)
                }
            }.font(.callout)
            if let n = dictation.note { Text(n).font(.caption).foregroundColor(Palette.yellow) }
        }.padding(10).background(Palette.panel)
    }

    // MARK: the modes control (replaces the Think / Auto / Images checkboxes)

    @State private var showModes = false

    /// A ring for how full the context window is; a click compacts the conversation.
    @ViewBuilder private var contextButton: some View {
        let used = store.contextTokens
        if let f = ContextMeter.fraction(used: used, limit: ws.engine?.context) {
            let color: Color = { switch ContextMeter.level(f) {
                case .ok: return Palette.accent
                case .warn: return Palette.yellow
                case .full: return Palette.magenta } }()
            Button { store.compact() } label: {
                ZStack {
                    Circle().stroke(Color.primary.opacity(0.18), lineWidth: 2.5)
                    Circle().trim(from: 0, to: max(0.02, f)).stroke(color, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                    if store.compacting { ProgressView().controlSize(.mini).scaleEffect(0.6) }
                }.frame(width: 17, height: 17)
            }
            .buttonStyle(.plain).disabled(store.busy || store.compacting)
            .help("Context ≈ \(ContextMeter.short(used)) of \(ContextMeter.short(ws.engine?.context ?? 0)) tokens "
                  + "(\(Int((f * 100).rounded())) %). Click to compact the earlier conversation into a short note.")
        }
    }

    private var modeButton: some View {
        Button { showModes.toggle() } label: {
            HStack(spacing: 5) {
                Image(systemName: store.mode.symbol)
                Text(store.mode.short)
            }
            .font(.callout).padding(.horizontal, 9).padding(.vertical, 4)
            .background(Palette.field, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Palette.hairline))
        }
        .buttonStyle(.plain)
        .help("Mode and reasoning effort (Shift+Tab switches the mode)")
        .popover(isPresented: $showModes, arrowEdge: .top) { ModesPanel(ws: ws, store: store, shown: $showModes) }
    }

    private var chips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(ws.attachments, id: \.self) { p in
                    chip(icon: "doc.text", title: PathTools.displayName(p)) { ws.detach(p) }
                }
                ForEach(ws.inline) { f in chip(icon: "doc.badge.plus", title: f.name) { ws.inline.removeAll { $0.id == f.id } } }
                ForEach(ws.pictures) { p in
                    HStack(spacing: 4) {
                        Image(nsImage: p.thumb).resizable().scaledToFit().frame(height: 22).clipShape(RoundedRectangle(cornerRadius: 3))
                        Text(p.name).font(.caption).lineLimit(1)
                        Button { ws.pictures.removeAll { $0.id == p.id } } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain)
                    }.padding(.horizontal, 6).padding(.vertical, 3).background(Palette.accent.opacity(0.15), in: Capsule())
                }
            }
        }.frame(height: ws.attachments.isEmpty && ws.inline.isEmpty && ws.pictures.isEmpty ? 0 : 28)
    }

    private func chip(icon: String, title: String, remove: @escaping () -> Void) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon).font(.caption)
            Text(title).font(.caption).lineLimit(1)
            Button(action: remove) { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain)
        }.padding(.horizontal, 8).padding(.vertical, 3).background(Palette.accent.opacity(0.15), in: Capsule())
    }

    private func send() {
        guard canSend, !store.busy else { return }
        if dictation.listening { dictation.stop() }
        store.send(draft)
        draft = ""
    }

    private func addFiles(_ urls: [URL]) { for u in urls { ws.addAttachment(from: u) } }
    private func addImage(_ img: NSImage) { ws.addPastedImage(img) }

    private func pickFiles(images: Bool) {
        let p = NSOpenPanel()
        p.allowsMultipleSelection = true
        p.canChooseDirectories = false
        p.prompt = "Attach"
        if images { p.allowedContentTypes = [.png, .jpeg, .gif, .webP, .heic, .tiff, .bmp] }
        if p.runModal() == .OK { addFiles(p.urls) }
    }
}

// MARK: - the Modes popover

private struct KeyCap: View {
    let text: String
    var body: some View {
        Text(text).font(.system(size: 11, weight: .medium, design: .rounded))
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background(Palette.field, in: RoundedRectangle(cornerRadius: 4))
            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Palette.hairline))
    }
}

struct ModesPanel: View {
    @ObservedObject var ws: Workspace
    @ObservedObject var store: ChatStore
    @Binding var shown: Bool

    private var positions: [String] {
        let levels = ws.remoteURL == nil ? (ws.engine?.efforts ?? []) : []
        let thinking = ws.engine?.thinking_supported ?? true
        return EffortScale.positions(levels: levels, supportsThinking: thinking)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Modes").foregroundColor(.secondary)
                Spacer()
                KeyCap(text: "⇧"); Text("+").foregroundColor(.secondary); KeyCap(text: "tab")
                Text("to switch").foregroundColor(.secondary)
            }.font(.callout).padding(.horizontal, 10).padding(.top, 6).padding(.bottom, 4)

            ForEach(AgentMode.allCases) { m in
                Button { store.mode = m; shown = false } label: {
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: m.symbol).frame(width: 22).padding(.top, 2)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(m.title)
                            Text(m.detail).font(.caption).foregroundColor(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Spacer(minLength: 4)
                        if m == store.mode { Image(systemName: "checkmark").foregroundColor(.primary) }
                    }
                    .padding(.horizontal, 10).padding(.vertical, 7).contentShape(Rectangle())
                    .background(m == store.mode ? Color.primary.opacity(0.10) : Color.clear, in: RoundedRectangle(cornerRadius: 8))
                }.buttonStyle(.plain)
            }

            if !positions.isEmpty {
                Divider().padding(.vertical, 4)
                effortRow
            }
            if ws.engine?.vision_capable == true {
                Divider().padding(.vertical, 4)
                HStack {
                    Image(systemName: "photo").frame(width: 22)
                    Text("Images")
                    Spacer()
                    if ws.visionSwitching { ProgressView().controlSize(.mini) }
                    Toggle("", isOn: Binding(get: { ws.engine?.vision == true }, set: { ws.setVision($0) }))
                        .toggleStyle(.switch).labelsHidden().controlSize(.small).disabled(ws.visionSwitching)
                }
                .padding(.horizontal, 10).padding(.vertical, 4)
                .help("Image input for this model (switching restarts the model, ~10 s)")
            }
        }
        .padding(8).frame(width: 340)
    }

    private var effortRow: some View {
        let pos = positions
        let current = EffortScale.position(think: store.think, effort: store.effort, among: pos)
        return HStack(spacing: 10) {
            Image(systemName: "brain").frame(width: 22)
            Text("Effort")
            Text("(\(current.prefix(1).uppercased() + current.dropFirst()))").foregroundColor(.secondary)
            Spacer()
            HStack(spacing: 14) {
                ForEach(pos, id: \.self) { p in
                    Button {
                        let s = EffortScale.setting(for: p)
                        store.think = s.think; store.effort = s.effort
                    } label: {
                        Circle().fill(p == current ? Color.primary.opacity(0.85) : Color.primary.opacity(0.25))
                            .frame(width: p == current ? 14 : 7, height: p == current ? 14 : 7)
                            .frame(width: 16, height: 16)
                    }.buttonStyle(.plain).help(p == "off" ? "No reasoning" : "Reasoning: \(p)")
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(Palette.field, in: Capsule())
        }
        .padding(.horizontal, 10).padding(.vertical, 4)
    }
}

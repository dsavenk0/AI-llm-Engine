// The integrated terminal: a real shell on a pty (via /usr/bin/script, so no fork() is needed), a small
// screen model (TermBuffer, tested separately) and a native text view. Started lazily on first show.
import AppKit
import SwiftUI

@MainActor
final class TerminalSession: ObservableObject {
    @Published var revision = 0
    @Published var running = false
    let buffer = TermBuffer(maxLines: 2000, screenRows: 30)
    private var process: Process?
    private var input: FileHandle?
    private var utf8 = UTF8Stream()
    private var dirty = false
    let cwd: String

    init(cwd: String) { self.cwd = cwd }

    func start() {
        guard process == nil else { return }
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/zsh"
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/script")
        p.arguments = ["-q", "/dev/null", shell, "-i"]
        p.currentDirectoryURL = URL(fileURLWithPath: cwd)
        var env = ProcessInfo.processInfo.environment
        env["TERM"] = "xterm-256color"
        env["COLUMNS"] = "110"; env["LINES"] = "30"
        env["PROMPT_EOL_MARK"] = ""                          // zsh's '%' marker for a missing newline is noise here
        env["LLM_EDITOR_TERMINAL"] = "1"
        p.environment = env
        let out = Pipe(), inp = Pipe()
        p.standardOutput = out
        p.standardError = out
        p.standardInput = inp
        input = inp.fileHandleForWriting
        out.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            if data.isEmpty { h.readabilityHandler = nil; return }
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.receive(data) } }
        }
        p.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async { MainActor.assumeIsolated {
                self?.buffer.feed("\r\n[process exited]\r\n")
                self?.running = false
                self?.process = nil
                self?.touch()
            } }
        }
        do { try p.run(); process = p; running = true } catch {
            buffer.feed("could not start \(shell): \(error.localizedDescription)\r\n")
            touch()
        }
    }

    private func receive(_ data: Data) {
        buffer.feed(utf8.decode(data))
        touch()
    }

    /// Coalesce bursts of output into at most ~30 redraws a second.
    private func touch() {
        if dirty { return }
        dirty = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.033) {
            MainActor.assumeIsolated { self.dirty = false; self.revision += 1 }
        }
    }

    func send(_ s: String) { if running, let d = s.data(using: .utf8) { try? input?.write(contentsOf: d) } }

    func restart() { stop(); utf8 = UTF8Stream(); start() }

    func stop() {
        process?.terminationHandler = nil
        process?.terminate()
        process = nil
        running = false
    }
}

final class TerminalTextView: NSTextView {
    var onInput: ((String) -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with e: NSEvent) {
        let flags = e.modifierFlags
        if flags.contains(.command) { super.keyDown(with: e); return }
        var out: String?
        switch e.keyCode {
        case 36, 76: out = "\r"
        case 51: out = "\u{7F}"
        case 117: out = "\u{1B}[3~"
        case 48: out = flags.contains(.shift) ? "\u{1B}[Z" : "\t"
        case 53: out = "\u{1B}"
        case 123: out = "\u{1B}[D"
        case 124: out = "\u{1B}[C"
        case 125: out = "\u{1B}[B"
        case 126: out = "\u{1B}[A"
        case 115: out = "\u{1B}[H"
        case 119: out = "\u{1B}[F"
        default:
            if flags.contains(.control), let c = e.charactersIgnoringModifiers?.unicodeScalars.first {
                let v = c.value
                if (0x40...0x7E).contains(v) || (0x61...0x7A).contains(v) { out = String(UnicodeScalar(UInt8(v & 0x1F))) }
            } else if flags.contains(.option), let c = e.charactersIgnoringModifiers { out = "\u{1B}" + c }
            else { out = e.characters }
        }
        if let out, !out.isEmpty { onInput?(out) }
    }

    override func paste(_ sender: Any?) {
        if let s = NSPasteboard.general.string(forType: .string) { onInput?(s.replacingOccurrences(of: "\n", with: "\r")) }
    }
    override func insertText(_ string: Any, replacementRange: NSRange) {}      // typing goes through keyDown only
}

struct TerminalView: NSViewRepresentable {
    @ObservedObject var session: TerminalSession

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        let tv = TerminalTextView(frame: .zero)
        tv.isEditable = false
        tv.isSelectable = true
        tv.isRichText = true
        tv.drawsBackground = false
        tv.font = Syntax.font
        tv.textContainerInset = NSSize(width: 6, height: 6)
        tv.isHorizontallyResizable = false
        tv.autoresizingMask = [.width]
        tv.onInput = { [weak session] s in session?.send(s) }
        scroll.documentView = tv
        scroll.drawsBackground = false
        context.coordinator.tv = tv
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        _ = session.revision
        guard let tv = context.coordinator.tv, let storage = tv.textStorage else { return }
        let nearBottom = scroll.contentView.bounds.maxY >= (scroll.documentView?.frame.maxY ?? 0) - 40
        storage.setAttributedString(render(session.buffer))
        if nearBottom { tv.scrollToEndOfDocument(nil) }
        if tv.window?.firstResponder !== tv, tv.window?.firstResponder is NSTextView == false { tv.window?.makeFirstResponder(tv) }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { weak var tv: TerminalTextView? }

    private func render(_ b: TermBuffer) -> NSAttributedString {
        let out = NSMutableAttributedString()
        let regular = Syntax.font
        let bold = NSFontManager.shared.convert(regular, toHaveTrait: .boldFontMask)
        func attrs(_ s: TermStyle, cursor: Bool) -> [NSAttributedString.Key: Any] {
            var fg: NSColor = .labelColor, bg: NSColor? = nil
            if let c = s.fg { let (r, g, bl) = c.components(); fg = NSColor(srgbRed: r, green: g, blue: bl, alpha: 1) }
            if let c = s.bg { let (r, g, bl) = c.components(); bg = NSColor(srgbRed: r, green: g, blue: bl, alpha: 1) }
            if s.dim { fg = fg.withAlphaComponent(0.6) }
            if s.inverse || cursor { let f = bg ?? NSColor.textBackgroundColor; bg = s.inverse && cursor ? nil : fg; fg = f }
            var a: [NSAttributedString.Key: Any] = [.font: s.bold ? bold : regular, .foregroundColor: fg]
            if let bg { a[.backgroundColor] = bg }
            if s.underline { a[.underlineStyle] = NSUnderlineStyle.single.rawValue }
            return a
        }
        for (i, line) in b.lines.enumerated() {
            var j = 0
            while j < line.count {
                let isCursor = i == b.row && j == b.col
                var run = String(line[j].ch)
                var k = j + 1
                while k < line.count, line[k].style == line[j].style, !(i == b.row && k == b.col) { run.append(line[k].ch); k += 1 }
                out.append(NSAttributedString(string: run, attributes: attrs(line[j].style, cursor: isCursor)))
                j = k
            }
            if i == b.row && b.col >= line.count {             // the cursor sits after the last character
                let pad = String(repeating: " ", count: b.col - line.count)
                out.append(NSAttributedString(string: pad, attributes: attrs(TermStyle(), cursor: false)))
                out.append(NSAttributedString(string: " ", attributes: attrs(TermStyle(), cursor: true)))
            }
            if i < b.lines.count - 1 { out.append(NSAttributedString(string: "\n", attributes: attrs(TermStyle(), cursor: false))) }
        }
        return out
    }
}

struct TerminalModuleView: View {
    @ObservedObject var session: TerminalSession

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "terminal")
                Text(session.cwd).font(.caption).foregroundColor(.secondary).lineLimit(1).truncationMode(.head)
                Spacer()
                Button("Clear") { session.send("clear\r") }.buttonStyle(.borderless)
                Button("Restart") { session.restart() }.buttonStyle(.borderless)
                Button("Open in Terminal.app") {
                    NSWorkspace.shared.open([URL(fileURLWithPath: session.cwd)],
                                            withApplicationAt: URL(fileURLWithPath: "/System/Applications/Utilities/Terminal.app"),
                                            configuration: NSWorkspace.OpenConfiguration())
                }.buttonStyle(.borderless).help("For full-screen programs (vim, htop…)")
            }.font(.caption).padding(.horizontal, 10).padding(.vertical, 4).background(Color.primary.opacity(0.04))
            Divider()
            TerminalView(session: session)
        }
        .onAppear { session.start() }                      // lazily: no shell until the panel is first shown
    }
}

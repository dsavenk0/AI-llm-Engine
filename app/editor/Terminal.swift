// The integrated terminal: a real shell on a pty (via /usr/bin/script, so no fork() is needed), a small
// screen model (TermBuffer, tested separately) and a native text view. Started lazily on first show.
import AppKit
import SwiftUI

@MainActor
final class TerminalSession: ObservableObject {
    @Published var revision = 0
    @Published var running = false
    let buffer = TermBuffer(maxLines: 2000, screenRows: 30, screenCols: 110)
    private(set) var cols = 110, rows = 30
    private var resizePipe: String?                       // a named pipe the shell's helper reads "rows cols" from
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
        // The pty starts at 0x0: set its size first, then keep a small helper that applies later sizes from the
        // named pipe (stty on the controlling tty makes the kernel send SIGWINCH to vim, htop, less…). The helper
        // ends with the shell, or on "q".
        let fifo = NSTemporaryDirectory() + "llmbar-term-\(getpid())-\(UInt32.random(in: 0...UInt32.max)).fifo"
        mkfifo(fifo, 0o600)
        resizePipe = fifo
        let wrapper = """
        stty rows "$3" cols "$4" 2>/dev/null
        ( me=$$; exec 9<>"$2"
          trap '' TTOU TTIN                  # this helper is a background job: the kernel refuses a window-size
          while kill -0 $me 2>/dev/null; do  # change from one (EIO) unless the signal is ignored
            if read -t 1 -u 9 r c; then
              [ "$r" = q ] && break
              stty rows "$r" cols "$c" </dev/tty 2>/dev/null
            fi
          done ) &
        exec "$1" -i
        """
        p.arguments = ["-q", "/dev/null", "/bin/bash", "-c", wrapper, "bash", shell, fifo, String(rows), String(cols)]
        p.currentDirectoryURL = URL(fileURLWithPath: cwd)
        var env = ProcessInfo.processInfo.environment
        env["TERM"] = "xterm-256color"
        env["COLUMNS"] = String(cols); env["LINES"] = String(rows)
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
                self?.closePipe()
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

    /// The panel changed size: the screen model and the pty follow, so full-screen programs redraw to fit.
    func resize(rows r: Int, cols c: Int) {
        let r = max(5, min(r, 200)), c = max(20, min(c, 400))
        guard r != rows || c != cols else { return }
        rows = r; cols = c
        buffer.resize(rows: r, cols: c)
        if running, let fifo = resizePipe {
            let fd = open(fifo, O_WRONLY | O_NONBLOCK)
            if fd >= 0 {
                let msg = "\(r) \(c)\n"
                _ = msg.withCString { write(fd, $0, strlen($0)) }
                close(fd)
            }
        }
    }

    private func closePipe() {
        guard let fifo = resizePipe else { return }
        let fd = open(fifo, O_WRONLY | O_NONBLOCK)
        if fd >= 0 { _ = "q\n".withCString { write(fd, $0, 2) }; close(fd) }
        unlink(fifo)
        resizePipe = nil
    }

    func restart() { stop(); utf8 = UTF8Stream(); start() }

    func stop() {
        closePipe()
        process?.terminationHandler = nil
        process?.terminate()
        process = nil
        running = false
    }
}

final class TerminalTextView: NSTextView {
    var onInput: ((String) -> Void)?
    var applicationCursorKeys = false      // the program asked for ESC O A style arrows (?1)
    var bracketedPaste = false             // ... and for pasted text to be wrapped (?2004)

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
        case 123: out = applicationCursorKeys ? "\u{1B}OD" : "\u{1B}[D"
        case 124: out = applicationCursorKeys ? "\u{1B}OC" : "\u{1B}[C"
        case 125: out = applicationCursorKeys ? "\u{1B}OB" : "\u{1B}[B"
        case 126: out = applicationCursorKeys ? "\u{1B}OA" : "\u{1B}[A"
        case 115: out = applicationCursorKeys ? "\u{1B}OH" : "\u{1B}[H"
        case 119: out = applicationCursorKeys ? "\u{1B}OF" : "\u{1B}[F"
        case 116: out = "\u{1B}[5~"                          // page up
        case 121: out = "\u{1B}[6~"                          // page down
        case 122: out = "\u{1B}OP"                           // F1 … F4
        case 120: out = "\u{1B}OQ"
        case 99: out = "\u{1B}OR"
        case 118: out = "\u{1B}OS"
        case 96: out = "\u{1B}[15~"                          // F5 …
        case 97: out = "\u{1B}[17~"
        case 98: out = "\u{1B}[18~"
        case 100: out = "\u{1B}[19~"
        case 101: out = "\u{1B}[20~"
        case 109: out = "\u{1B}[21~"
        case 103: out = "\u{1B}[23~"
        case 111: out = "\u{1B}[24~"
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
        guard let s = NSPasteboard.general.string(forType: .string) else { return }
        let text = s.replacingOccurrences(of: "\n", with: "\r")
        onInput?(bracketedPaste ? "\u{1B}[200~" + text + "\u{1B}[201~" : text)
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
        tv.applicationCursorKeys = session.buffer.applicationCursorKeys
        tv.bracketedPaste = session.buffer.bracketedPaste
        // the character cell, from the font: how many columns and rows the visible area holds
        let cell = ("M" as NSString).size(withAttributes: [.font: Syntax.font])
        let lineH = (Syntax.font.ascender - Syntax.font.descender + Syntax.font.leading).rounded(.up)
        let size = scroll.contentSize
        if cell.width > 1, lineH > 1, size.width > 100, size.height > 40 {
            session.resize(rows: Int((size.height - 12) / lineH), cols: Int((size.width - 12) / cell.width))
        }
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
                let isCursor = b.cursorVisible && i == b.row && j == b.col
                var run = String(line[j].ch)
                var k = j + 1
                while k < line.count, line[k].style == line[j].style, !(i == b.row && k == b.col) { run.append(line[k].ch); k += 1 }
                out.append(NSAttributedString(string: run, attributes: attrs(line[j].style, cursor: isCursor)))
                j = k
            }
            if i == b.row && b.col >= line.count && b.cursorVisible {      // the cursor sits after the last character
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

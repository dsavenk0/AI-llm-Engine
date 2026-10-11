// The text editor: a native NSTextView with a line-number ruler. Colours come from the backend's spans.
import AppKit
import SwiftUI

enum Syntax {
    static let font = NSFont.monospacedSystemFont(ofSize: 12.5, weight: .regular)

    private static func dyn(_ light: (CGFloat, CGFloat, CGFloat), _ dark: (CGFloat, CGFloat, CGFloat)) -> NSColor {
        NSColor(name: nil) { a in
            let c = a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: c.0, green: c.1, blue: c.2, alpha: 1)
        }
    }
    static let colors: [String: NSColor] = [
        "keyword": dyn((0.67, 0.10, 0.45), (1.00, 0.48, 0.71)),
        "string": dyn((0.10, 0.45, 0.15), (0.61, 0.89, 0.48)),
        "comment": dyn((0.42, 0.47, 0.52), (0.45, 0.54, 0.61)),
        "number": dyn((0.70, 0.35, 0.00), (1.00, 0.72, 0.42)),
        "function": dyn((0.00, 0.40, 0.65), (0.15, 0.83, 1.00)),
        "type": dyn((0.00, 0.50, 0.40), (0.30, 0.95, 0.76)),
        "decorator": dyn((0.50, 0.25, 0.70), (0.78, 0.57, 0.92)),
        "property": dyn((0.15, 0.30, 0.75), (0.51, 0.67, 1.00)),
        "constant": dyn((0.75, 0.30, 0.10), (0.97, 0.55, 0.42)),
        "tag": dyn((0.67, 0.10, 0.45), (1.00, 0.48, 0.71)),
        "heading": dyn((0.00, 0.50, 0.40), (0.30, 0.95, 0.76)),
    ]
}

/// Draws 1-based line numbers next to the text view.
final class LineNumberRuler: NSRulerView {
    private weak var textView: NSTextView?

    init(textView: NSTextView, scrollView: NSScrollView) {
        self.textView = textView
        super.init(scrollView: scrollView, orientation: .verticalRuler)
        clientView = textView
        ruleThickness = 44
        NotificationCenter.default.addObserver(self, selector: #selector(redraw), name: NSText.didChangeNotification, object: textView)
        NotificationCenter.default.addObserver(self, selector: #selector(redraw), name: NSView.boundsDidChangeNotification,
                                               object: scrollView.contentView)
    }
    required init(coder: NSCoder) { fatalError() }
    @objc private func redraw() { needsDisplay = true }

    override func drawHashMarksAndLabels(in rect: NSRect) {
        guard let tv = textView, let lm = tv.layoutManager, let tc = tv.textContainer else { return }
        NSColor.windowBackgroundColor.withAlphaComponent(0.6).setFill()
        rect.fill()
        let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.monospacedDigitSystemFont(ofSize: 10.5, weight: .regular),
                                                    .foregroundColor: NSColor.tertiaryLabelColor]
        let visible = lm.glyphRange(forBoundingRect: tv.visibleRect, in: tc)
        let text = tv.string as NSString
        var line = 1
        if visible.location > 0 {                                       // count the lines above the viewport
            let upto = lm.characterRange(forGlyphRange: NSRange(location: 0, length: visible.location), actualGlyphRange: nil)
            text.enumerateSubstrings(in: NSRange(location: 0, length: upto.length), options: [.byLines, .substringNotRequired]) { _, _, _, _ in line += 1 }
            if upto.length > 0, text.character(at: upto.length - 1) != 10 { line -= 1 }
        }
        var index = lm.characterIndexForGlyph(at: visible.location)
        let origin = tv.textContainerOrigin
        let end = NSMaxRange(lm.characterRange(forGlyphRange: visible, actualGlyphRange: nil))
        while index <= end && index <= text.length {
            let lineRange = text.lineRange(for: NSRange(location: min(index, text.length), length: 0))
            let glyph = lm.glyphIndexForCharacter(at: min(lineRange.location, max(text.length - 1, 0)))
            var frag = lm.lineFragmentRect(forGlyphAt: min(glyph, max(lm.numberOfGlyphs - 1, 0)), effectiveRange: nil)
            if text.length == 0 { frag = lm.extraLineFragmentRect }
            let y = frag.minY + origin.y - tv.visibleRect.minY
            let label = "\(line)" as NSString
            let size = label.size(withAttributes: attrs)
            label.draw(at: NSPoint(x: ruleThickness - size.width - 8, y: y + (frag.height - size.height) / 2), withAttributes: attrs)
            line += 1
            let next = NSMaxRange(lineRange)
            if next <= index { break }
            index = next
            if index == text.length && text.length > 0 && text.character(at: text.length - 1) != 10 { break }
        }
        if text.length > 0, text.character(at: text.length - 1) == 10, index >= text.length {      // the empty last line
            let frag = lm.extraLineFragmentRect
            let y = frag.minY + origin.y - tv.visibleRect.minY
            let label = "\(line)" as NSString
            let size = label.size(withAttributes: attrs)
            label.draw(at: NSPoint(x: ruleThickness - size.width - 8, y: y + (frag.height - size.height) / 2), withAttributes: attrs)
        }
    }
}

struct CodeEditor: NSViewRepresentable {
    @ObservedObject var file: OpenFile
    var onEdit: (OpenFile) -> Void
    var onCursor: (Int, Int) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = true
        scroll.drawsBackground = false
        let tv = NSTextView()
        tv.isRichText = false
        tv.allowsUndo = true
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isAutomaticDashSubstitutionEnabled = false
        tv.isAutomaticTextReplacementEnabled = false
        tv.isAutomaticSpellingCorrectionEnabled = false
        tv.isContinuousSpellCheckingEnabled = false
        tv.font = Syntax.font
        tv.textColor = .labelColor
        tv.drawsBackground = false
        tv.textContainerInset = NSSize(width: 6, height: 8)
        tv.isHorizontallyResizable = true
        tv.isVerticallyResizable = true
        tv.autoresizingMask = [.width]
        tv.textContainer?.widthTracksTextView = false
        tv.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        tv.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        tv.delegate = context.coordinator
        scroll.documentView = tv
        scroll.hasVerticalRuler = true
        scroll.rulersVisible = true
        scroll.verticalRulerView = LineNumberRuler(textView: tv, scrollView: scroll)
        context.coordinator.textView = tv
        context.coordinator.load(file)
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.sync(file)
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: CodeEditor
        weak var textView: NSTextView?
        private var shownPath = ""
        private var appliedSpans = -1
        private var applying = false
        private var timer: Timer?

        init(_ p: CodeEditor) { parent = p }

        func load(_ f: OpenFile) {
            guard let tv = textView else { return }
            applying = true
            tv.string = f.text
            applying = false
            shownPath = f.path
            appliedSpans = -1
            paint(f)
        }

        func sync(_ f: OpenFile) {
            guard let tv = textView else { return }
            if shownPath != f.path { load(f) }
            else if tv.string != f.text {                    // changed from outside (e.g. an applied suggestion)
                applying = true
                let sel = tv.selectedRange()
                tv.string = f.text
                tv.setSelectedRange(NSRange(location: min(sel.location, (f.text as NSString).length), length: 0))
                applying = false
                paint(f)
            } else if appliedSpans != f.spans.count || f.spans.isEmpty { paint(f) }
            if let line = f.jumpLine {
                f.jumpLine = nil
                let off = OffsetMap(f.text).lineStart(line, in: f.text)
                tv.setSelectedRange(NSRange(location: off, length: 0))
                tv.scrollRangeToVisible(NSRange(location: off, length: 0))
                tv.window?.makeFirstResponder(tv)
            }
        }

        /// Base attributes for everything, then a colour per span (code points → UTF-16 ranges).
        func paint(_ f: OpenFile) {
            guard let tv = textView, let storage = tv.textStorage else { return }
            let full = NSRange(location: 0, length: storage.length)
            storage.beginEditing()
            storage.addAttributes([.font: Syntax.font, .foregroundColor: NSColor.labelColor], range: full)
            let map = OffsetMap(f.text)
            for s in f.spans {
                guard let r = map.range(start: s.start, end: s.end), NSMaxRange(r) <= storage.length,
                      let c = Syntax.colors[s.kind] else { continue }
                storage.addAttribute(.foregroundColor, value: c, range: r)
                if s.kind == "comment" {
                    storage.addAttribute(.font, value: NSFontManager.shared.convert(Syntax.font, toHaveTrait: .italicFontMask), range: r)
                }
            }
            storage.endEditing()
            appliedSpans = f.spans.count
        }

        func textDidChange(_ n: Notification) {
            guard !applying, let tv = textView else { return }
            let f = parent.file
            f.text = tv.string
            f.version += 1
            timer?.invalidate()
            timer = Timer.scheduledTimer(withTimeInterval: 0.15, repeats: false) { [weak self] _ in
                guard let self else { return }
                Task { @MainActor in self.parent.onEdit(f) }
            }
        }

        func textViewDidChangeSelection(_ n: Notification) {
            guard let tv = textView else { return }
            let loc = tv.selectedRange().location
            let ns = tv.string as NSString
            let before = ns.substring(to: min(loc, ns.length))
            let line = before.components(separatedBy: "\n").count
            let col = loc - (before.lastIndex(of: "\n").map { before.distance(from: before.startIndex, to: $0) + 1 } ?? 0) + 1
            parent.onCursor(line, col)
        }

        func textView(_ tv: NSTextView, doCommandBy sel: Selector) -> Bool {
            if sel == #selector(NSResponder.insertTab(_:)) { tv.insertText("    ", replacementRange: tv.selectedRange()); return true }
            return false
        }
    }
}

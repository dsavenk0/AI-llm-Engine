// The text editor: a native NSTextView with a line-number ruler. Colours come from the backend's spans.
import AppKit
import SwiftUI

enum Syntax {
    /// Medium weight: regular strokes of a monospaced face are ~1 px wide on a retina display and read as
    /// washed out once anti-aliased, whatever the colour.
    static let font = NSFont.monospacedSystemFont(ofSize: 13, weight: .medium)

    private static func ns(_ c: (CGFloat, CGFloat, CGFloat)) -> NSColor { NSColor(srgbRed: c.0, green: c.1, blue: c.2, alpha: 1) }

    /// Chosen by the theme setting itself (a dynamic colour resolved to the wrong variant in a hosted view
    /// and came out as pale pastels on a light background). Light values are saturated and dark enough
    /// for ~7:1 contrast on white; dark values are bright on the graphite ground.
    private static let light: [String: (CGFloat, CGFloat, CGFloat)] = [
        "keyword": (0.55, 0.00, 0.45), "string": (0.00, 0.34, 0.05), "comment": (0.30, 0.34, 0.40),
        "number": (0.60, 0.22, 0.00), "function": (0.00, 0.26, 0.56), "type": (0.00, 0.36, 0.28),
        "decorator": (0.40, 0.12, 0.60), "property": (0.05, 0.18, 0.62), "constant": (0.64, 0.18, 0.00),
        "tag": (0.55, 0.00, 0.45), "heading": (0.00, 0.36, 0.28),
    ]
    private static let dark: [String: (CGFloat, CGFloat, CGFloat)] = [
        "keyword": (1.00, 0.45, 0.70), "string": (0.60, 0.92, 0.45), "comment": (0.50, 0.60, 0.68),
        "number": (1.00, 0.72, 0.40), "function": (0.30, 0.85, 1.00), "type": (0.35, 0.96, 0.78),
        "decorator": (0.80, 0.60, 0.95), "property": (0.55, 0.72, 1.00), "constant": (0.98, 0.58, 0.45),
        "tag": (1.00, 0.45, 0.70), "heading": (0.35, 0.96, 0.78),
    ]

    static func color(_ kind: String, dark isDark: Bool) -> NSColor? {
        if let t = ThemeSettings.shared.custom {                // a theme from an extension: its colours, kind by kind
            if let hex = t.kinds[kind], let v = ThemeOverride.rgb(hex) { return ns((CGFloat(v.0), CGFloat(v.1), CGFloat(v.2))) }
            return nil
        }
        return (isDark ? dark : light)[kind].map(ns)
    }
    static func text(dark isDark: Bool) -> NSColor {
        if let t = ThemeSettings.shared.custom, let v = ThemeOverride.rgb(t.foreground) {
            return ns((CGFloat(v.0), CGFloat(v.1), CGFloat(v.2)))
        }
        return isDark ? ns((0.92, 0.96, 0.98)) : ns((0.02, 0.04, 0.07))
    }
}

/// Draws 1-based line numbers next to the text view.
final class LineNumberRuler: NSRulerView {
    private weak var textView: NSTextView?
    var marks: [Int: Bool] = [:] { didSet { needsDisplay = true } }      // line → has an error (else a warning)

    init(textView: NSTextView, scrollView: NSScrollView) {
        self.textView = textView
        super.init(scrollView: scrollView, orientation: .verticalRuler)
        clientView = textView
        clipToFrame()
        ruleThickness = 44
        NotificationCenter.default.addObserver(self, selector: #selector(redraw), name: NSText.didChangeNotification, object: textView)
        NotificationCenter.default.addObserver(self, selector: #selector(redraw), name: NSView.boundsDidChangeNotification,
                                               object: scrollView.contentView)
    }
    required init(coder: NSCoder) { fatalError() }
    @objc private func redraw() { needsDisplay = true }

    override func drawHashMarksAndLabels(in rect: NSRect) {
        guard let tv = textView, let lm = tv.layoutManager, let tc = tv.textContainer else { return }
        // Only the gutter itself, and opaque: this used to be a 60 % window-coloured fill of the whole `rect`,
        // which (views are not clipped to their frame since macOS 14) dimmed the code beside the numbers —
        // an "overlay" that kept the pane's first width when the window was widened.
        Palette.editorNS(isDark: ThemeSettings.shared.isDark).setFill()
        rect.intersection(bounds).fill()
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
        while index <= end && (index < text.length || (text.length == 0 && index == 0)) {
            let lineRange = text.lineRange(for: NSRange(location: min(index, text.length), length: 0))
            let glyph = lm.glyphIndexForCharacter(at: min(lineRange.location, max(text.length - 1, 0)))
            var frag = lm.lineFragmentRect(forGlyphAt: min(glyph, max(lm.numberOfGlyphs - 1, 0)), effectiveRange: nil)
            if text.length == 0 { frag = lm.extraLineFragmentRect }
            let y = frag.minY + origin.y - tv.visibleRect.minY
            let label = "\(line)" as NSString
            let size = label.size(withAttributes: attrs)
            label.draw(at: NSPoint(x: ruleThickness - size.width - 8, y: y + (frag.height - size.height) / 2), withAttributes: attrs)
            if let isError = marks[line] {
                (isError ? NSColor.systemRed : NSColor.systemYellow).setFill()
                NSBezierPath(ovalIn: NSRect(x: 5, y: y + (frag.height - 7) / 2, width: 7, height: 7)).fill()
            }
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

/// The text view is as wide as its longest line but never narrower than the visible area; without that the
/// part of the pane the text view did not cover was painted by something else (a lighter, uncoloured band
/// that kept its old width when the window was widened).
final class CodeScrollView: NSScrollView {
    private var gutterOverlap: CGFloat = -1
    private var boundsObserver: NSObjectProtocol?

    /// With a TextKit 1 text view the line-number ruler lies over the first 44 pt of the text instead of beside it.
    /// Whether the text then starts under the ruler depends on where the clip view rests: at bounds.x = -44 the
    /// system has already made room, at 0 (what it settles on when the document is narrower than the pane, e.g.
    /// after the window was widened) the text sits under the gutter and loses its first characters. So the inset
    /// follows the clip view's position — measured, not assumed — and is re-checked whenever that position changes.
    private func updateGutterInset() {
        guard let tv = documentView as? NSTextView, let r = verticalRulerView, rulersVisible,
              contentView.bounds.origin.x <= 0 else { return }
        let docOriginOnScreen = contentView.frame.minX - contentView.bounds.origin.x
        let overlap = max(0, (r.frame.maxX - docOriginOnScreen).rounded())
        if overlap != gutterOverlap {
            gutterOverlap = overlap
            tv.textContainerInset = NSSize(width: 6 + overlap, height: 8)
        }
    }

    override func layout() {
        super.layout()
        guard let tv = documentView as? NSTextView else { return }
        if boundsObserver == nil {
            contentView.postsBoundsChangedNotifications = true
            boundsObserver = NotificationCenter.default.addObserver(
                forName: NSView.boundsDidChangeNotification, object: contentView, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.updateGutterInset() }
            }
        }
        updateGutterInset()
        let size = contentSize
        tv.minSize = NSSize(width: size.width, height: size.height)
        if tv.frame.width < size.width { tv.setFrameSize(NSSize(width: size.width, height: tv.frame.height)) }
        if lastSize != size {                  // the pane was resized: nothing the layer cached is valid any more
            lastSize = size
            tv.layoutManager?.ensureLayout(for: tv.textContainer!)
            tv.setNeedsDisplay(tv.bounds)
            contentView.setNeedsDisplay(contentView.bounds)
        }
    }
    private var lastSize = NSSize.zero
}

extension NSView {
    /// macOS 14+ no longer clips a view's drawing to its frame by default.
    func clipToFrame() {
        if #available(macOS 14.0, *) { clipsToBounds = true }
    }
}

struct CodeEditor: NSViewRepresentable {
    @ObservedObject var file: OpenFile
    var snippets: [Snippet] = []                // from installed extensions: a prefix + Tab expands
    var onEdit: (OpenFile) -> Void
    var onCursor: (Int, Int) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = CodeScrollView()
        scroll.clipToFrame()
        scroll.contentView.clipToFrame()
        scroll.contentView.layerContentsRedrawPolicy = .duringViewResize
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = true
        scroll.drawsBackground = true
        scroll.backgroundColor = Palette.editorNS(isDark: ThemeSettings.shared.isDark)
        // TextKit 1 from the start: the line-number ruler needs the layout manager, and touching it on a default
        // (TextKit 2) view switches it over mid-life, leaving a band of text the old engine drew once, at its
        // original width, in the default colour.
        let tv = NSTextView(usingTextLayoutManager: false)
        tv.isRichText = false
        tv.allowsUndo = true
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isAutomaticDashSubstitutionEnabled = false
        tv.isAutomaticTextReplacementEnabled = false
        tv.isAutomaticSpellingCorrectionEnabled = false
        tv.isContinuousSpellCheckingEnabled = false
        tv.font = Syntax.font
        tv.textColor = Syntax.text(dark: ThemeSettings.shared.isDark)
        tv.insertionPointColor = ThemeSettings.shared.isDark ? NSColor(srgbRed: 0, green: 0.9, blue: 1, alpha: 1) : .black
        tv.drawsBackground = true
        tv.backgroundColor = Palette.editorNS(isDark: ThemeSettings.shared.isDark)
        tv.textContainerInset = NSSize(width: 6, height: 8)
        tv.isHorizontallyResizable = true
        tv.isVerticallyResizable = true
        tv.autoresizingMask = [.width]
        tv.textContainer?.widthTracksTextView = false
        tv.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        tv.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        tv.delegate = context.coordinator
        tv.clipToFrame()
        tv.layerContentsRedrawPolicy = .duringViewResize
        scroll.documentView = tv
        scroll.hasVerticalRuler = true
        scroll.rulersVisible = true
        scroll.verticalRulerView = LineNumberRuler(textView: tv, scrollView: scroll)
        context.coordinator.textView = tv
        context.coordinator.load(file)
        context.coordinator.themeObserver = NotificationCenter.default.addObserver(
            forName: .themeChanged, object: nil, queue: .main) { [weak coordinator = context.coordinator] _ in
            MainActor.assumeIsolated { coordinator?.themeChanged() }
        }
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
        private var appliedDiag = -1
        private var applying = false
        private var timer: Timer?
        var themeObserver: NSObjectProtocol?

        init(_ p: CodeEditor) { parent = p }
        deinit { if let o = themeObserver { NotificationCenter.default.removeObserver(o) } }

        func themeChanged() {
            guard let tv = textView else { return }
            let dark = ThemeSettings.shared.isDark
            tv.textColor = Syntax.text(dark: dark)
            tv.backgroundColor = Palette.editorNS(isDark: dark)
            tv.enclosingScrollView?.backgroundColor = Palette.editorNS(isDark: dark)
            tv.insertionPointColor = dark ? NSColor(srgbRed: 0, green: 0.9, blue: 1, alpha: 1) : .black
            appliedSpans = -1
            paint(parent.file)
        }

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
            } else if appliedSpans != f.spans.count || f.spans.isEmpty || appliedDiag != diagSignature(f) { paint(f) }
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
            let dark = ThemeSettings.shared.isDark
            storage.addAttributes([.font: Syntax.font, .foregroundColor: Syntax.text(dark: dark)], range: full)
            let map = OffsetMap(f.text)
            for s in f.spans {
                guard let r = map.range(start: s.start, end: s.end), NSMaxRange(r) <= storage.length,
                      let c = Syntax.color(s.kind, dark: dark) else { continue }
                storage.addAttribute(.foregroundColor, value: c, range: r)
                if s.kind == "comment" {
                    storage.addAttribute(.font, value: NSFontManager.shared.convert(Syntax.font, toHaveTrait: .italicFontMask), range: r)
                }
            }
            // problems: a dotted underline in the severity colour, the message as a tooltip
            var marks: [Int: Bool] = [:]
            for d in f.diagnostics {
                let start = OffsetMap.position(line: d.line, col: d.col, in: f.text)
                var end = OffsetMap.position(line: d.line, col: max(d.col + 1, d.end_col ?? d.col + 1), in: f.text)
                if end <= start { end = min(storage.length, start + 1) }
                guard start < storage.length else { continue }
                let r = NSRange(location: start, length: min(end, storage.length) - start)
                let color = d.isError ? NSColor.systemRed : NSColor.systemYellow
                storage.addAttributes([.underlineStyle: NSUnderlineStyle.single.rawValue | NSUnderlineStyle.patternDot.rawValue,
                                       .underlineColor: color, .toolTip: "\(d.message)  (\(d.source))"], range: r)
                marks[d.line] = (marks[d.line] ?? false) || d.isError
            }
            (tv.enclosingScrollView?.verticalRulerView as? LineNumberRuler)?.marks = marks
            storage.endEditing()
            appliedSpans = f.spans.count
            appliedDiag = diagSignature(f)
        }

        private func diagSignature(_ f: OpenFile) -> Int {
            var h = Hasher()
            for d in f.diagnostics { h.combine(d.id) }
            return h.finalize()
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

        func textView(_ tv: NSTextView, doCommandBy sel: Selector) -> Bool {
            if sel == #selector(NSResponder.insertTab(_:)) {
                if expandSnippet(in: tv) { return true }
                tv.insertText("    ", replacementRange: tv.selectedRange())
                return true
            }
            return false
        }
    }
}

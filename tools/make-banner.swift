import AppKit

// README banner (1600×560 PNG): app icon, title, tagline and headline numbers.
// Usage: swift make-banner.swift <output.png> <icon-1024.png>

let outPath = CommandLine.arguments.dropFirst().first ?? "docs/banner.png"
let W: CGFloat = 1600
let H: CGFloat = 560

guard let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: Int(W), pixelsHigh: Int(H),
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
) else { exit(1) }

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

let accent = NSColor(srgbRed: 0.30, green: 0.95, blue: 0.76, alpha: 1)
let text = NSColor(srgbRed: 0.92, green: 0.96, blue: 0.96, alpha: 1)
let muted = NSColor(srgbRed: 0.55, green: 0.63, blue: 0.65, alpha: 1)

func font(_ size: CGFloat, bold: Bool = false) -> NSFont {
    NSFont(name: bold ? "Menlo-Bold" : "Menlo-Regular", size: size)
        ?? NSFont.monospacedSystemFont(ofSize: size, weight: bold ? .bold : .regular)
}
func width(_ s: String, _ f: NSFont) -> CGFloat { (s as NSString).size(withAttributes: [.font: f]).width }
func draw(_ s: String, _ p: NSPoint, _ f: NSFont, _ c: NSColor) {
    NSAttributedString(string: s, attributes: [.font: f, .foregroundColor: c]).draw(at: p)
}

// background
let bg = NSBezierPath(roundedRect: NSRect(x: 0, y: 0, width: W, height: H), xRadius: 36, yRadius: 36)
NSGradient(colors: [NSColor(srgbRed: 0.055, green: 0.085, blue: 0.095, alpha: 1),
                    NSColor(srgbRed: 0.015, green: 0.025, blue: 0.03, alpha: 1)])?.draw(in: bg, angle: -40)
bg.addClip()
let grid = NSBezierPath()
for x in stride(from: CGFloat(-200), through: W + 200, by: 80) {
    grid.move(to: NSPoint(x: x, y: 0)); grid.line(to: NSPoint(x: x + 300, y: H))
}
grid.lineWidth = 1
accent.withAlphaComponent(0.04).setStroke()
grid.stroke()
accent.withAlphaComponent(0.9).setFill()
NSBezierPath(rect: NSRect(x: 0, y: 0, width: W, height: 8)).fill()

// app icon (rendered by make-icon.swift, passed as the 2nd argument)
let iconPath = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "/tmp/llm-icon-1024.png"
if let icon = NSImage(contentsOfFile: iconPath) {
    icon.draw(in: NSRect(x: 70, y: 120, width: 320, height: 320))
}

// title + tagline
let x0: CGFloat = 420
let title = font(84, bold: true)
draw("AI-llm-", NSPoint(x: x0, y: 350), title, text)
draw("Engine", NSPoint(x: x0 + width("AI-llm-", title), y: 350), title, accent)
draw("Fast local Qwen3.5 / JANG inference on Apple Silicon", NSPoint(x: x0, y: 300), font(27), muted)

// headline numbers
let stats = [("2.6×", "code edits"), ("+37%", "decode vs vMLX"), ("168", "prefill tok/s"), ("no 413", "clamped max_tokens")]
var x = x0
for (big, small) in stats {
    let card = NSRect(x: x, y: 110, width: 255, height: 140)
    let p = NSBezierPath(roundedRect: card, xRadius: 22, yRadius: 22)
    NSColor(srgbRed: 0.06, green: 0.09, blue: 0.10, alpha: 0.9).setFill(); p.fill()
    accent.withAlphaComponent(0.22).setStroke(); p.lineWidth = 2; p.stroke()
    draw(big, NSPoint(x: x + 24, y: 180), font(40, bold: true), accent)
    draw(small, NSPoint(x: x + 24, y: 136), font(19), muted)
    x += 275
}

NSGraphicsContext.restoreGraphicsState()
guard let png = rep.representation(using: .png, properties: [:]) else { exit(1) }
try FileManager.default.createDirectory(at: URL(fileURLWithPath: outPath).deletingLastPathComponent(),
                                        withIntermediateDirectories: true)
try png.write(to: URL(fileURLWithPath: outPath))

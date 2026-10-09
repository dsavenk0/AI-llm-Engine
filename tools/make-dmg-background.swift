import AppKit

// Retina background for the installer DMG Finder window: 680 × 440 pt @2x.
// Usage: swift make-dmg-background.swift <output.png>

let outPath = CommandLine.arguments.dropFirst().first ?? "/tmp/llmbar-dmg-background.png"
let W: CGFloat = 1360
let H: CGFloat = 880

guard let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: Int(W), pixelsHigh: Int(H),
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
) else { exit(1) }

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

let accent = NSColor(srgbRed: 0.30, green: 0.95, blue: 0.76, alpha: 1)
let ink = NSColor(srgbRed: 0.02, green: 0.035, blue: 0.04, alpha: 1)
let panel = NSColor(srgbRed: 0.06, green: 0.09, blue: 0.10, alpha: 0.88)
let text = NSColor(srgbRed: 0.91, green: 0.95, blue: 0.95, alpha: 1)
let muted = NSColor(srgbRed: 0.52, green: 0.60, blue: 0.62, alpha: 1)

func font(_ size: CGFloat, bold: Bool = false) -> NSFont {
    NSFont(name: bold ? "Menlo-Bold" : "Menlo-Regular", size: size)
        ?? NSFont.monospacedSystemFont(ofSize: size, weight: bold ? .bold : .regular)
}

func draw(_ s: String, _ p: NSPoint, _ f: NSFont, _ c: NSColor) {
    NSAttributedString(string: s, attributes: [.font: f, .foregroundColor: c]).draw(at: p)
}

NSGradient(colors: [NSColor(srgbRed: 0.05, green: 0.08, blue: 0.09, alpha: 1), ink])?
    .draw(in: NSBezierPath(rect: NSRect(x: 0, y: 0, width: W, height: H)), angle: -58)

let grid = NSBezierPath()
for x in stride(from: CGFloat(-40), through: W + 80, by: 80) {
    grid.move(to: NSPoint(x: x, y: 0))
    grid.line(to: NSPoint(x: x + 250, y: H))
}
grid.lineWidth = 1
accent.withAlphaComponent(0.035).setStroke()
grid.stroke()

accent.withAlphaComponent(0.95).setFill()
NSBezierPath(rect: NSRect(x: 0, y: 0, width: W, height: 8)).fill()

let title = font(58, bold: true)
draw("llm", NSPoint(x: 80, y: 738), title, text)
let w1 = ("llm" as NSString).size(withAttributes: [.font: title]).width
draw("Bar", NSPoint(x: 80 + w1, y: 738), title, accent)
draw("Drag to Applications to install", NSPoint(x: 80, y: 680), font(25), muted)

for x in [80 as CGFloat, 830] {
    let card = NSBezierPath(roundedRect: NSRect(x: x, y: 160, width: 450, height: 430), xRadius: 34, yRadius: 34)
    panel.setFill(); card.fill()
    accent.withAlphaComponent(0.20).setStroke()
    card.lineWidth = 2; card.stroke()
}
draw("1", NSPoint(x: 116, y: 530), font(22, bold: true), accent)
draw("LLMBar", NSPoint(x: 164, y: 530), font(22, bold: true), text)
draw("2", NSPoint(x: 866, y: 530), font(22, bold: true), accent)
draw("Applications", NSPoint(x: 914, y: 530), font(22, bold: true), text)

let glow = NSShadow()
glow.shadowColor = accent.withAlphaComponent(0.6)
glow.shadowBlurRadius = 22
glow.shadowOffset = .zero
glow.set()
let y: CGFloat = 370
let arrow = NSBezierPath()
arrow.move(to: NSPoint(x: 570, y: y)); arrow.line(to: NSPoint(x: 760, y: y))
arrow.move(to: NSPoint(x: 720, y: y + 38)); arrow.line(to: NSPoint(x: 762, y: y)); arrow.line(to: NSPoint(x: 720, y: y - 38))
arrow.lineWidth = 12
arrow.lineCapStyle = .round
arrow.lineJoinStyle = .round
accent.setStroke()
arrow.stroke()

NSGraphicsContext.restoreGraphicsState()
guard let png = rep.representation(using: .png, properties: [:]) else { exit(1) }
try png.write(to: URL(fileURLWithPath: outPath))

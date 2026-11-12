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
let ink = NSColor(srgbRed: 0.91, green: 0.95, blue: 0.94, alpha: 1)
let text = NSColor(srgbRed: 0.035, green: 0.075, blue: 0.08, alpha: 1)
let muted = NSColor(srgbRed: 0.24, green: 0.34, blue: 0.35, alpha: 1)

func font(_ size: CGFloat, bold: Bool = false) -> NSFont {
    NSFont(name: bold ? "Menlo-Bold" : "Menlo-Regular", size: size)
        ?? NSFont.monospacedSystemFont(ofSize: size, weight: bold ? .bold : .regular)
}

func draw(_ s: String, _ p: NSPoint, _ f: NSFont, _ c: NSColor) {
    NSAttributedString(string: s, attributes: [.font: f, .foregroundColor: c]).draw(at: p)
}

NSGradient(colors: [NSColor(srgbRed: 0.99, green: 0.995, blue: 0.99, alpha: 1), ink])?
    .draw(in: NSBezierPath(rect: NSRect(x: 0, y: 0, width: W, height: H)), angle: -58)

let grid = NSBezierPath()
for x in stride(from: CGFloat(-40), through: W + 80, by: 80) {
    grid.move(to: NSPoint(x: x, y: 0))
    grid.line(to: NSPoint(x: x + 250, y: H))
}
grid.lineWidth = 1
accent.withAlphaComponent(0.055).setStroke()
grid.stroke()

accent.withAlphaComponent(0.95).setFill()
NSBezierPath(rect: NSRect(x: 0, y: 0, width: W, height: 8)).fill()

let title = font(58, bold: true)
draw("llm", NSPoint(x: 80, y: 738), title, text)
let w1 = ("llm" as NSString).size(withAttributes: [.font: title]).width
draw("Bar", NSPoint(x: 80 + w1, y: 738), title, accent)
draw("Drag to Applications to install", NSPoint(x: 80, y: 680), font(25), muted)

// Finder draws icon names in black, so keep the whole installer background light for readable labels.

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

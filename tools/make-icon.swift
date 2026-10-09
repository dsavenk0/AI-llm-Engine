import AppKit

// App icon (1024×1024 PNG): a little CRT terminal — dark-teal bezel, black
// screen with a soft vignette and a mint "Ai>_" prompt that glows.
// Usage: swift make-icon.swift <output.png>

let outPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/tmp/llm-icon-1024.png"
let size: CGFloat = 1024

guard let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: Int(size), pixelsHigh: Int(size),
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
) else { exit(1) }

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

let phosphor = NSColor(srgbRed: 0.30, green: 0.95, blue: 0.76, alpha: 1.0)

// --- bezel: dark-teal squircle with a lit top edge ---
let inset: CGFloat = 92
let outer = NSRect(x: inset, y: inset, width: size - 2 * inset, height: size - 2 * inset)
let bezel = NSBezierPath(roundedRect: outer, xRadius: 190, yRadius: 190)
NSGradient(colors: [
    NSColor(srgbRed: 0.16, green: 0.30, blue: 0.27, alpha: 1),
    NSColor(srgbRed: 0.05, green: 0.13, blue: 0.11, alpha: 1),
])?.draw(in: bezel, angle: -90)
NSColor(srgbRed: 0.30, green: 0.52, blue: 0.46, alpha: 0.55).setStroke()
bezel.lineWidth = 4
bezel.stroke()

// --- screen: black with a soft mint vignette ---
let screenRect = outer.insetBy(dx: 46, dy: 46)
let screen = NSBezierPath(roundedRect: screenRect, xRadius: 150, yRadius: 150)
NSGraphicsContext.saveGraphicsState()
let inner = NSShadow()
inner.shadowColor = NSColor.black.withAlphaComponent(0.9)
inner.shadowBlurRadius = 24
inner.shadowOffset = NSSize(width: 0, height: -6)
inner.set()
NSColor(srgbRed: 0.02, green: 0.03, blue: 0.025, alpha: 1).setFill()
screen.fill()
NSGraphicsContext.restoreGraphicsState()
NSGraphicsContext.saveGraphicsState()
screen.addClip()
NSGradient(colors: [
    NSColor(srgbRed: 0.08, green: 0.17, blue: 0.15, alpha: 0.55),
    NSColor(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.0),
])?.draw(in: screen, relativeCenterPosition: .zero)
// faint scanlines
NSColor.black.withAlphaComponent(0.18).setFill()
for y in stride(from: screenRect.minY, to: screenRect.maxY, by: 9) {
    NSBezierPath(rect: NSRect(x: screenRect.minX, y: y, width: screenRect.width, height: 3)).fill()
}
NSGraphicsContext.restoreGraphicsState()

// --- glowing prompt ---
let font = NSFont(name: "Menlo-Regular", size: 250) ?? NSFont.monospacedSystemFont(ofSize: 250, weight: .regular)
let prompt = NSAttributedString(string: "Ai>_", attributes: [.font: font, .foregroundColor: phosphor])
let ts = prompt.size()
let origin = NSPoint(x: (size - ts.width) / 2, y: (size - ts.height) / 2 - 10)
for (blur, alpha) in [(CGFloat(60), CGFloat(0.55)), (28, 0.7), (8, 0.9)] {
    NSGraphicsContext.saveGraphicsState()
    let glow = NSShadow()
    glow.shadowColor = phosphor.withAlphaComponent(alpha)
    glow.shadowBlurRadius = blur
    glow.shadowOffset = .zero
    glow.set()
    prompt.draw(at: origin)
    NSGraphicsContext.restoreGraphicsState()
}
prompt.draw(at: origin)

NSGraphicsContext.restoreGraphicsState()
guard let png = rep.representation(using: .png, properties: [:]) else { exit(1) }
try png.write(to: URL(fileURLWithPath: outPath))

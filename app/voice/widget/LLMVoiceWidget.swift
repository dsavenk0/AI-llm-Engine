// The desktop / Notification Centre widget of LLM Voice: one tap opens the quick chat, which listens at once.
// A WidgetKit extension cannot talk to the engine itself; it only draws the orb and carries the link.
import SwiftUI
import WidgetKit

struct OrbEntry: TimelineEntry { let date: Date }

struct OrbProvider: TimelineProvider {
    func placeholder(in context: Context) -> OrbEntry { OrbEntry(date: Date()) }
    func getSnapshot(in context: Context, completion: @escaping (OrbEntry) -> Void) { completion(OrbEntry(date: Date())) }
    func getTimeline(in context: Context, completion: @escaping (Timeline<OrbEntry>) -> Void) {
        completion(Timeline(entries: [OrbEntry(date: Date())], policy: .never))      // nothing ever changes
    }
}

/// A burst of thin curling filaments around a white-hot core — the same picture as the app's sphere, still.
struct StaticOrb: View {
    private func rnd(_ i: Int, _ salt: Double) -> Double {
        let v = sin(Double(i) * 12.9898 + salt * 78.233) * 43758.5453
        return v - v.rounded(.down)
    }

    var body: some View {
        Canvas { ctx, size in
            let c = CGPoint(x: size.width / 2, y: size.height / 2)
            let R = min(size.width, size.height) * 0.42
            let colors: [Color] = [Color(red: 0, green: 0.9, blue: 1), Color(red: 0.3, green: 0.95, blue: 0.76),
                                   Color(red: 0.2, green: 0.55, blue: 1), Color(red: 0.85, green: 0.98, blue: 1)]
            for i in 0..<44 {
                let a = rnd(i, 1) * .pi * 2
                let len = R * (0.45 + 0.55 * rnd(i, 2))
                let curl = (rnd(i, 3) - 0.5) * 1.6
                var path = Path()
                path.move(to: c)
                for k in 1...24 {
                    let t = Double(k) / 24
                    let ang = a + curl * t * t
                    let r = len * t
                    let wob = sin(t * 10 + Double(i) * 1.9) * R * 0.04 * t
                    path.addLine(to: CGPoint(x: c.x + cos(ang) * r + cos(ang + 1.57) * wob,
                                             y: c.y + sin(ang) * r + sin(ang + 1.57) * wob))
                }
                let col = colors[i % colors.count]
                ctx.stroke(path, with: .color(col.opacity(0.25)), lineWidth: 4.5)
                ctx.stroke(path, with: .color(col.opacity(0.9)), lineWidth: 1.2)
            }
            let core = CGRect(x: c.x - R * 0.12, y: c.y - R * 0.12, width: R * 0.24, height: R * 0.24)
            ctx.fill(Path(ellipseIn: core.insetBy(dx: -R * 0.1, dy: -R * 0.1)), with: .color(Color.white.opacity(0.25)))
            ctx.fill(Path(ellipseIn: core), with: .color(.white))
        }
    }
}

struct OrbWidgetView: View {
    @Environment(\.widgetFamily) private var family

    var body: some View {
        ZStack {
            StaticOrb()
            VStack {
                Spacer()
                Text(family == .systemSmall ? "SPEAK" : "TAP TO SPEAK OR TYPE")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .tracking(1.6)
                    .foregroundColor(Color(red: 0.0, green: 0.9, blue: 1.0).opacity(0.9))
            }.padding(.bottom, 2)
        }
        .widgetURL(URL(string: "llmvoice://listen"))
        .containerBackground(for: .widget) {
            LinearGradient(colors: [Color(red: 0.02, green: 0.05, blue: 0.08), Color(red: 0.04, green: 0.10, blue: 0.14)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }
}

struct OrbWidget: Widget {
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: "local.llm-engine.voice.widget", provider: OrbProvider()) { _ in OrbWidgetView() }
            .configurationDisplayName("LLM Voice")
            .description("One tap opens the quick chat — it listens at once.")
            .supportedFamilies([.systemSmall, .systemMedium])
    }
}

@main
struct LLMVoiceWidgetBundle: WidgetBundle {
    var body: some Widget { OrbWidget() }
}

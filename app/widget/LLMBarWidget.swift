// The desktop / Notification Centre widget of LLM Voice: one tap opens the quick chat — the app starts the model
// and listens at once.
// A WidgetKit extension cannot talk to the engine itself; it only draws the orb and carries the link.
import SwiftUI
import WidgetKit

struct OrbEntry: TimelineEntry {
    let date: Date
    let model: String            // the model chosen in LLMBar / `./llm use`, "" when none
    let running: Bool            // the engine answers on its port
    let russian: Bool
}

/// What the widget may know: LLMBar's and the CLI's shared config (the build grants the sandboxed extension read
/// access to exactly that folder, path in Info.plist as LLMHome) and the engine's /health on localhost.
enum Status {
    private static func config() -> [String: Any] {
        guard let home = Bundle.main.object(forInfoDictionaryKey: "LLMHome") as? String,
              let data = try? Data(contentsOf: URL(fileURLWithPath: home + "/.llm/config.json")),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return obj
    }

    static func entry(completion: @escaping (OrbEntry) -> Void) {
        let c = config()
        let path = c["model"] as? String ?? ""
        let name = (path as NSString).lastPathComponent
        let ru = (c["lang"] as? String) == "ru"
        let port = c["port"] as? Int ?? 8090
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/health")!)
        req.timeoutInterval = 1.2
        if let key = c["api_key"] as? String, !key.isEmpty { req.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
        URLSession.shared.dataTask(with: req) { _, resp, _ in
            let up = (resp as? HTTPURLResponse)?.statusCode == 200
            completion(OrbEntry(date: Date(), model: name, running: up, russian: ru))
        }.resume()
    }
}

struct OrbProvider: TimelineProvider {
    func placeholder(in context: Context) -> OrbEntry {
        OrbEntry(date: Date(), model: "qwen3.5-0.8b-4bit", running: true, russian: false)
    }
    func getSnapshot(in context: Context, completion: @escaping (OrbEntry) -> Void) {
        if context.isPreview { completion(placeholder(in: context)) } else { Status.entry(completion: completion) }
    }
    func getTimeline(in context: Context, completion: @escaping (Timeline<OrbEntry>) -> Void) {
        // the app asks for a reload whenever the model changes state; this is the fallback
        Status.entry { entry in
            let next = Calendar.current.date(byAdding: .minute, value: 5, to: Date()) ?? Date().addingTimeInterval(300)
            completion(Timeline(entries: [entry], policy: .after(next)))
        }
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
    let entry: OrbEntry

    private let cyan = Color(red: 0.0, green: 0.9, blue: 1.0)
    private let mint = Color(red: 0.3, green: 0.95, blue: 0.76)
    private let pink = Color(red: 1, green: 0.18, blue: 0.53)

    private var toggleLabel: String {
        if entry.russian { return entry.running ? "Выключить" : "Включить" }
        return entry.running ? "Turn off" : "Turn on"
    }

    private func launcher(_ symbol: String, _ en: String, _ ru: String, _ target: String) -> some View {
        Link(destination: URL(string: "llmbar://launch?target=\(target)")!) {
            VStack(spacing: 2) {
                Image(systemName: symbol).font(.system(size: 14, weight: .semibold))
                Text((entry.russian ? ru : en).uppercased())
                    .font(.system(size: 8, weight: .semibold, design: .monospaced)).tracking(0.8)
            }
            .foregroundColor(cyan)
            .frame(width: 70, height: 32)
            .background(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(cyan.opacity(0.35), lineWidth: 1))
        }
    }

    var body: some View {
        Group {
            if family == .systemMedium {
                VStack(spacing: 3) {
                    HStack(spacing: 5) {                                      // which model, and whether it is up
                        ZStack {                                           // filled while the model is on, an empty ring off
                            Circle().strokeBorder(Color.white.opacity(0.45), lineWidth: 1.2).opacity(entry.running ? 0 : 1)
                            Circle().fill(mint).opacity(entry.running ? 1 : 0)
                        }.frame(width: 8, height: 8)
                        Text(entry.model.isEmpty ? (entry.russian ? "МОДЕЛЬ НЕ ВЫБРАНА" : "NO MODEL SELECTED") : entry.model)
                            .lineLimit(1).truncationMode(.middle)
                        Spacer(minLength: 4)
                        Link(destination: URL(string: "llmbar://toggle")!) {      // the model on / off, in the background
                            HStack(spacing: 4) {
                                Image(systemName: "power").font(.system(size: 10, weight: .bold))
                                Text(toggleLabel.uppercased())
                                    .font(.system(size: 9, weight: .semibold, design: .monospaced)).tracking(0.8)
                            }
                            .foregroundColor(entry.running ? pink : cyan)
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(Capsule().strokeBorder((entry.running ? pink : cyan).opacity(0.75), lineWidth: 1))
                        }
                    }
                    .font(.system(size: 10.5, weight: .semibold, design: .monospaced)).tracking(0.4)
                    .foregroundColor(entry.model.isEmpty ? pink : .white.opacity(0.92))
                    StaticOrb().scaleEffect(1.3).frame(maxHeight: .infinity)  // the big icon: a tap opens the voice chat
                    HStack(spacing: 8) {                                     // each of these loads the model ahead of use
                        launcher("waveform", "Voice", "Голос", "voice")
                        launcher("chevron.left.forwardslash.chevron.right", "Editor", "Редактор", "editor")
                        launcher("terminal", "CLI", "CLI", "cli")
                    }
                }
            } else {
                ZStack {
                    StaticOrb()
                    VStack {
                        Spacer()
                        Text("LLM VOICE")
                            .font(.system(size: 9, weight: .semibold, design: .monospaced)).tracking(1.6)
                            .foregroundColor(cyan.opacity(0.9))
                    }.padding(.bottom, 2)
                }
            }
        }
        .widgetURL(URL(string: "llmbar://launch?target=voice"))
        .containerBackground(for: .widget) {
            LinearGradient(colors: [Color(red: 0.02, green: 0.05, blue: 0.08), Color(red: 0.04, green: 0.10, blue: 0.14)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }
}

struct OrbWidget: Widget {
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: "local.llm-engine.llmbar.widget", provider: OrbProvider()) { OrbWidgetView(entry: $0) }
            .configurationDisplayName("LLM Voice")
            .description("The selected model, a power button that turns it on and off in the background, and the voice chat, editor and terminal — each one loads the model ahead of use.")
            .supportedFamilies([.systemSmall, .systemMedium])
    }
}

@main
struct LLMBarWidgetBundle: WidgetBundle {
    var body: some Widget { OrbWidget() }
}

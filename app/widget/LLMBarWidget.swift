// The desktop / Notification Centre widget of LLMBar: the selected model, a power button, the three modules and —
// while the model is on — GPU load and RAM use on either side of the orb. A WidgetKit extension cannot run the
// engine itself; it draws, reads a few numbers, and carries `llmbar://` links into the app.
import IOKit
import SwiftUI
import WidgetKit

struct OrbEntry: TimelineEntry {
    let date: Date
    let model: String            // the model chosen in LLMBar / `./llm use`, "" when none
    let running: Bool            // the engine answers on its port
    let russian: Bool
    var asleep = false           // the engine runs but the weights are unloaded (idle sleep)
    var phase = 0.0              // which frame of the slow animation this entry is
    var gpu: Double = 0          // 0…1, GPU device utilisation (IOAccelerator)
    var ramUsed: Double = 0      // bytes in use: wired + active + compressed
    var ramTotal: Double = Double(ProcessInfo.processInfo.physicalMemory)
    /// The model is loaded and answering: the orb moves, the gauges show. Off or asleep: still, a ring.
    var active: Bool { running && !asleep }
}

/// The two numbers next to the orb. Both are read the way LLMBar's panel reads them.
enum Metrics {
    static func gpu() -> Double {
        var it: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) == KERN_SUCCESS else { return 0 }
        defer { IOObjectRelease(it) }
        var best = 0.0
        var entry = IOIteratorNext(it)
        while entry != 0 {
            if let raw = IORegistryEntryCreateCFProperty(entry, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0),
               let p = raw.takeRetainedValue() as? [String: Any] {
                best = max(best, Double(p["Device Utilization %"] as? Int ?? 0) / 100)
            }
            IOObjectRelease(entry)
            entry = IOIteratorNext(it)
        }
        return best
    }

    static func ramUsed() -> Double {
        var s = vm_statistics64()
        var c = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &s) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(c)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &c) }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        let page = Double(vm_kernel_page_size)
        return (Double(s.wire_count) + Double(s.active_count) + Double(s.compressor_page_count)) * page
    }
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
        URLSession.shared.dataTask(with: req) { data, resp, _ in
            let up = (resp as? HTTPURLResponse)?.statusCode == 200
            let health = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            let asleep = (health?["state"] as? String) == "sleeping"
            completion(OrbEntry(date: Date(), model: name, running: up, russian: ru, asleep: asleep,
                                gpu: up && !asleep ? Metrics.gpu() : 0, ramUsed: Metrics.ramUsed()))
        }.resume()
    }
}

struct OrbProvider: TimelineProvider {
    func placeholder(in context: Context) -> OrbEntry {
        OrbEntry(date: Date(), model: "qwen3.5-0.8b-4bit", running: true, russian: false, gpu: 0.42, ramUsed: 9.4e9)
    }
    func getSnapshot(in context: Context, completion: @escaping (OrbEntry) -> Void) {
        if context.isPreview { completion(placeholder(in: context)) } else { Status.entry(completion: completion) }
    }
    func getTimeline(in context: Context, completion: @escaping (Timeline<OrbEntry>) -> Void) {
        // the app asks for a reload whenever the model changes state. While the model is on the numbers next to the
        // orb are worth refreshing — as often as the system's widget budget lets it (a snapshot, not a live graph).
        Status.entry { entry in
            guard entry.active else {                                   // off or asleep: one still picture, no animation
                let next = Date().addingTimeInterval(5 * 60)
                completion(Timeline(entries: [entry], policy: .after(next)))
                return
            }
            // While the model is on: a slow animation, one frame every 10 s for two minutes (a widget cannot run
            // a continuous one; it can be given frames in advance). Then the timeline is rebuilt with fresh numbers.
            var frames: [OrbEntry] = []
            for i in 0..<12 {
                var f = entry
                f = OrbEntry(date: Date().addingTimeInterval(Double(i) * 10), model: entry.model, running: true,
                             russian: entry.russian, asleep: false, phase: Double(i) * 0.45,
                             gpu: entry.gpu, ramUsed: entry.ramUsed, ramTotal: entry.ramTotal)
                frames.append(f)
            }
            completion(Timeline(entries: frames, policy: .after(Date().addingTimeInterval(120))))
        }
    }
}

/// A burst of thin curling filaments around a white-hot core — the same picture as the app's sphere, still.
struct StaticOrb: View {
    /// The desktop draws an unfocused widget in one tint (`.accented`): colours collapse to white, so the soft
    /// halo behind each filament becomes a grey band. There the filaments are one thin line and the core a dot.
    var reduced = false
    var phase = 0.0                  // the animation frame: nudges every filament a little

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
                let a = rnd(i, 1) * .pi * 2 + phase * 0.06 * (rnd(i, 9) - 0.5)
                let len = R * (0.45 + 0.55 * rnd(i, 2)) * (1 + 0.05 * sin(phase + Double(i)))
                let curl = (rnd(i, 3) - 0.5) * 1.6 + 0.12 * sin(phase * 0.8 + Double(i) * 0.7)
                var path = Path()
                path.move(to: c)
                for k in 1...24 {
                    let t = Double(k) / 24
                    let ang = a + curl * t * t
                    let r = len * t
                    let wob = sin(t * 10 + Double(i) * 1.9 + phase) * R * 0.04 * t
                    path.addLine(to: CGPoint(x: c.x + cos(ang) * r + cos(ang + 1.57) * wob,
                                             y: c.y + sin(ang) * r + sin(ang + 1.57) * wob))
                }
                let col = colors[i % colors.count]
                if !reduced { ctx.stroke(path, with: .color(col.opacity(0.25)), lineWidth: 4.5) }
                ctx.stroke(path, with: .color(col.opacity(reduced ? 0.8 : 0.9)), lineWidth: reduced ? 0.9 : 1.2)
            }
            let core = CGRect(x: c.x - R * 0.12, y: c.y - R * 0.12, width: R * 0.24, height: R * 0.24)
            if !reduced {
                ctx.fill(Path(ellipseIn: core.insetBy(dx: -R * 0.1, dy: -R * 0.1)), with: .color(Color.white.opacity(0.25)))
            }
            ctx.fill(Path(ellipseIn: core), with: .color(.white))
        }
    }
}

struct OrbWidgetView: View {
    @Environment(\.widgetFamily) private var family
    @Environment(\.widgetRenderingMode) private var rendering
    let entry: OrbEntry
    private var reduced: Bool { rendering != .fullColor }

    private let cyan = Color(red: 0.0, green: 0.9, blue: 1.0)
    private let mint = Color(red: 0.3, green: 0.95, blue: 0.76)
    private let pink = Color(red: 1, green: 0.18, blue: 0.53)

    private var toggleLabel: String {
        if entry.russian { return entry.running ? "Выключить" : "Включить" }
        return entry.running ? "Turn off" : "Turn on"
    }

    /// A vertical gauge: label above, the bar, the value below. Pink above 85 %.
    private func gauge(_ label: String, _ fraction: Double, _ value: String, tint: Color) -> some View {
        let f = max(0.02, min(1, fraction))
        let color = f > 0.85 ? pink : tint
        return VStack(spacing: 3) {
            Text(label).font(.system(size: 10.5, weight: .bold, design: .monospaced)).tracking(0.8)
                .foregroundColor(.white.opacity(0.75))
            ZStack(alignment: .bottom) {
                Capsule().fill(Color.white.opacity(0.12)).frame(width: 7)
                GeometryReader { g in
                    Capsule().fill(color).frame(width: 7, height: g.size.height * f)
                        .frame(maxHeight: .infinity, alignment: .bottom)
                        .widgetAccentable()
                }.frame(width: 7)
            }.frame(height: 44)
            Text(value).font(.system(size: 12.5, weight: .bold, design: .monospaced))
                .foregroundColor(.white).lineLimit(1).minimumScaleFactor(0.75)
        }.frame(width: 62)
    }

    @ViewBuilder private var orbRow: some View {
        if entry.active {
            HStack(spacing: 4) {
                gauge("GPU", entry.gpu, "\(Int((entry.gpu * 100).rounded()))%", tint: cyan)          // one side …
                StaticOrb(reduced: reduced, phase: entry.phase).scaleEffect(1.15).frame(maxHeight: .infinity)
                gauge("RAM", entry.ramTotal > 0 ? entry.ramUsed / entry.ramTotal : 0,                // … and the other
                      String(format: "%.1f GB", entry.ramUsed / 1e9), tint: mint)
            }
        } else {
            StaticOrb(reduced: reduced).scaleEffect(1.3).opacity(entry.running ? 0.75 : 1).frame(maxHeight: .infinity)
        }
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
                            Circle().strokeBorder(Color.white.opacity(0.45), lineWidth: 1.2).opacity(entry.active ? 0 : 1)
                            Circle().fill(mint).opacity(entry.active ? 1 : 0)
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
                    orbRow                                                    // the big icon: a tap opens the voice chat
                    HStack(spacing: 8) {                                     // each of these loads the model ahead of use
                        launcher("waveform", "Voice", "Голос", "voice")
                        launcher("chevron.left.forwardslash.chevron.right", "Editor", "Редактор", "editor")
                        launcher("terminal", "CLI", "CLI", "cli")
                    }
                }
            } else {
                ZStack {
                    StaticOrb(reduced: reduced)
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

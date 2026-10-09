// LLMBar — the lightweight UI for AI-llm-Engine.
//
// Menu-bar icon → panel (or a regular window on double-click / when the notch
// hides the icon). Tabs: Engine (status, token statistics, RAM guard), Monitor
// (CPU E/P cores, GPU, memory), Settings (incl. advanced tuning) and Tools.
// Look: Apple-grade restraint with a Ghost-in-the-Shell HUD — dark glass,
// hairline neon frames, monospaced telemetry, cyan / phosphor glow.
// All engine work goes through the `llm` CLI and the HTTP API; the app only
// polls and edits .llm/config.json. Build: ./build-app.sh
import AppKit
import Darwin
import IOKit
import SwiftUI

// MARK: - language

var useRussian: Bool = Locale.preferredLanguages.first?.hasPrefix("ru") ?? false
func L(_ en: String, _ ru: String) -> String { useRussian ? ru : en }

// MARK: - theme

/// Light/dark is resolved from the "theme" setting (system / dark / light).
var lightTheme = false

enum HUD {
    static var cyan: Color { lightTheme ? Color(red: 0.0, green: 0.45, blue: 0.62) : Color(red: 0.0, green: 0.90, blue: 1.0) }
    static var phosphor: Color { lightTheme ? Color(red: 0.0, green: 0.56, blue: 0.45) : Color(red: 0.30, green: 0.95, blue: 0.76) }  // mint
    static var magenta: Color { lightTheme ? Color(red: 0.82, green: 0.08, blue: 0.40) : Color(red: 1.0, green: 0.18, blue: 0.53) }
    static var amber: Color { lightTheme ? Color(red: 0.78, green: 0.42, blue: 0.0) : Color(red: 1.0, green: 0.64, blue: 0.16) }
    static var ink: Color { lightTheme ? Color(red: 0.97, green: 0.98, blue: 0.99) : Color(red: 0.02, green: 0.04, blue: 0.06) }
    static var text: Color { lightTheme ? Color(red: 0.05, green: 0.09, blue: 0.13) : Color(red: 0.86, green: 0.95, blue: 0.98) }
    static var dim: Color { lightTheme ? Color(red: 0.36, green: 0.43, blue: 0.48) : Color(red: 0.45, green: 0.58, blue: 0.64) }
    static var glow: Double { lightTheme ? 0.25 : 1.0 }   // neon glow is for dark glass only
    static var panel: Color { lightTheme ? Color.white.opacity(0.42) : Color.black.opacity(0.20) }
    static var track: Color { lightTheme ? Color.black.opacity(0.07) : Color.white.opacity(0.06) }

    static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

/// Hairline-framed glass card.
struct Card<Content: View>: View {
    var glow: Color = HUD.cyan
    @ViewBuilder let content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 7) { content }
            .padding(11)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(HUD.panel)
                    .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(LinearGradient(colors: [glow.opacity(0.45), glow.opacity(0.08)],
                                                     startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 0.8))
            )
    }
}

/// "// SECTION" HUD caption.
struct Caption: View {
    let text: String
    var color: Color = HUD.cyan
    var body: some View {
        HStack(spacing: 6) {
            Rectangle().fill(color).frame(width: 2, height: 10).shadow(color: color, radius: 3)
            Text(text.uppercased()).font(HUD.mono(10, .semibold)).tracking(2.2).foregroundColor(color)
            Rectangle().fill(color.opacity(0.18)).frame(height: 0.6)
        }
    }
}

struct Bar: View {
    let value: Double
    var color: Color = HUD.cyan
    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(HUD.track)
                Capsule().fill(LinearGradient(colors: [color.opacity(0.55), color], startPoint: .leading, endPoint: .trailing))
                    .frame(width: max(3, g.size.width * min(1, max(0, value))))
                    .shadow(color: color.opacity(0.8 * HUD.glow), radius: 4)
            }
        }.frame(height: 5)
    }
}

struct Sparkline: View {
    let series: [([Double], Color)]
    var height: CGFloat = 38
    var body: some View {
        GeometryReader { g in
            ZStack {
                // faint HUD grid
                Path { p in
                    for i in 1..<4 { let y = g.size.height * CGFloat(i) / 4; p.move(to: CGPoint(x: 0, y: y)); p.addLine(to: CGPoint(x: g.size.width, y: y)) }
                }.stroke(Color.white.opacity(0.05), lineWidth: 0.5)
                ForEach(0..<series.count, id: \.self) { i in
                    let (vals, color) = series[i]
                    let pts = points(vals, g.size)
                    if pts.count > 1 {
                        Path { p in
                            p.move(to: CGPoint(x: pts[0].x, y: g.size.height))
                            pts.forEach { p.addLine(to: $0) }
                            p.addLine(to: CGPoint(x: pts.last!.x, y: g.size.height))
                        }.fill(LinearGradient(colors: [color.opacity(0.28), color.opacity(0)], startPoint: .top, endPoint: .bottom))
                        Path { p in p.move(to: pts[0]); pts.dropFirst().forEach { p.addLine(to: $0) } }
                            .stroke(color, style: StrokeStyle(lineWidth: 1.4, lineJoin: .round))
                            .shadow(color: color.opacity(0.9), radius: 3)
                    }
                }
            }
        }.frame(height: height)
    }

    func points(_ vals: [Double], _ size: CGSize) -> [CGPoint] {
        guard vals.count > 1 else { return [] }
        let step = size.width / 59
        let x0 = size.width - step * CGFloat(vals.count - 1)
        return vals.enumerated().map { j, v in
            CGPoint(x: x0 + step * CGFloat(j), y: size.height * (1 - CGFloat(min(1, max(0, v)))) )
        }
    }
}

struct Row: View {
    let label: String
    let value: String
    var hint: String = ""
    var color: Color = HUD.text
    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).font(.system(size: 12)).foregroundColor(HUD.dim)
            Spacer(minLength: 8)
            Text(value).font(HUD.mono(12, .medium)).foregroundColor(color)
        }.help(hint)
    }
}

/// Small "ⓘ" line under a control.
struct Hint: View {
    let text: String
    var body: some View { Text(text).font(.system(size: 10)).foregroundColor(HUD.dim.opacity(0.85)).fixedSize(horizontal: false, vertical: true) }
}

func gb(_ b: Double) -> String { String(format: "%.1f GB", b / 1_073_741_824) }
func num(_ v: Any?) -> String {
    guard let v, !(v is NSNull) else { return "—" }
    if let i = v as? Int { return i >= 100_000 ? String(format: "%.1fK", Double(i) / 1000) : "\(i)" }
    if let d = v as? Double { return d == d.rounded() && abs(d) < 1e6 ? "\(Int(d))" : String(format: "%.2f", d) }
    return "\(v)"
}

// MARK: - paths & CLI

struct Paths {
    let root: String   // llm / llm.py / engine
    let home: String   // LLM_HOME: .llm/config.json, logs, .venv

    static let shared: Paths = {
        let fm = FileManager.default
        if let repo = Bundle.main.object(forInfoDictionaryKey: "LLMRoot") as? String,
           fm.fileExists(atPath: repo + "/llm.py") {
            return Paths(root: repo, home: repo)  // built from a checkout: share its .venv and config
        }
        let home = fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/LLMBar").path
        try? fm.createDirectory(atPath: home, withIntermediateDirectories: true)
        return Paths(root: Bundle.main.resourceURL!.appendingPathComponent("llm").path, home: home)
    }()

    var config: String { home + "/.llm/config.json" }
}

enum CLI {
    static func run(_ args: [String], done: @escaping (Int32, String) -> Void) {
        DispatchQueue.global().async {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: Paths.shared.root + "/llm")
            p.arguments = args
            p.currentDirectoryURL = URL(fileURLWithPath: Paths.shared.root)
            var env = ProcessInfo.processInfo.environment
            env["LLM_HOME"] = Paths.shared.home
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            p.environment = env
            let pipe = Pipe()
            p.standardOutput = pipe
            p.standardError = pipe
            p.standardInput = FileHandle.nullDevice
            try? p.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            DispatchQueue.main.async { done(p.terminationStatus, String(decoding: data, as: UTF8.self)) }
        }
    }

    /// Run a shell command in a new Terminal window. Uses a throwaway .command file opened by
    /// Terminal — no Apple Events / automation permission needed (AppleScript is silently denied
    /// for apps without NSAppleEventsUsageDescription).
    static func terminal(_ command: String) {
        let r = Paths.shared.root, h = Paths.shared.home
        let dir = h + "/.llm"
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let file = dir + "/run-\(Int(Date().timeIntervalSince1970 * 1000)).command"
        let body = "#!/bin/sh\ncd '\(r)' || exit 1\nexport LLM_HOME='\(h)'\nrm -f '\(file)'\n\(command)\n"
        do {
            try body.write(toFile: file, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: file)
        } catch { return }
        let terminal = URL(fileURLWithPath: "/System/Applications/Utilities/Terminal.app")
        NSWorkspace.shared.open([URL(fileURLWithPath: file)], withApplicationAt: terminal,
                                configuration: NSWorkspace.OpenConfiguration())
    }
}

// MARK: - engine state

struct ModelInfo: Identifiable, Hashable {
    let path, name, format: String
    let size: Double
    let vision, thinking, supported: Bool
    let efforts: [String]
    var id: String { path }
}

final class EngineState: ObservableObject {
    @Published var health: [String: Any]? = nil
    @Published var stats: [String: Any]? = nil
    @Published var busy: String? = nil
    @Published var message: String? = nil
    @Published var models: [ModelInfo] = []
    @Published var config: [String: Any] = [:]
    var onTheme: (() -> Void)? = nil

    var running: Bool { health != nil }
    var sleeping: Bool { (health?["state"] as? String) == "sleeping" }
    var port: Int { config["port"] as? Int ?? 8090 }
    var apiURL: String { "http://localhost:\(port)/v1" }

    init() { loadConfig() }

    func loadConfig() {
        if let d = FileManager.default.contents(atPath: Paths.shared.config),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] { config = j }
        let theme = config["theme"] as? String ?? "system"
        lightTheme = theme == "light" || (theme == "system" && NSApp?.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .aqua)
        let lang = config["lang"] as? String ?? "auto"
        useRussian = lang == "ru" || (lang == "auto" && (Locale.preferredLanguages.first?.hasPrefix("ru") ?? false))
    }

    func saveConfig() {
        try? FileManager.default.createDirectory(atPath: Paths.shared.home + "/.llm", withIntermediateDirectories: true)
        if let d = try? JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys]) {
            try? d.write(to: URL(fileURLWithPath: Paths.shared.config))
        }
    }

    func g<T>(_ k: String, _ def: T) -> T { config[k] as? T ?? def }
    func setG(_ k: String, _ v: Any?) { config[k] = v; saveConfig() }

    var modelPath: String { config["model"] as? String ?? "" }
    func m<T>(_ k: String, _ def: T) -> T {
        ((config["models"] as? [String: Any])?[modelPath] as? [String: Any])?[k] as? T ?? def
    }
    func setM(_ k: String, _ v: Any) {
        var all = config["models"] as? [String: Any] ?? [:]
        var mine = all[modelPath] as? [String: Any] ?? [:]
        mine[k] = v; all[modelPath] = mine; config["models"] = all; saveConfig()
    }
    func adv(_ k: String) -> String {
        guard let v = (config["advanced"] as? [String: Any])?[k], !(v is NSNull) else { return "" }
        return "\(v)"
    }
    func setAdv(_ k: String, _ text: String) {
        var a = config["advanced"] as? [String: Any] ?? [:]
        let t = text.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")
        if t.isEmpty { a.removeValue(forKey: k) } else if let i = Int(t) { a[k] = i } else if let f = Double(t) { a[k] = f }
        config["advanced"] = a; saveConfig()
    }

    func poll() {
        get("/health") { j in self.health = j }
        if running { get("/v1/stats") { j in self.stats = j } }
    }

    private func get(_ path: String, _ done: @escaping ([String: Any]?) -> Void) {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        req.timeoutInterval = 1.5
        URLSession.shared.dataTask(with: req) { data, _, _ in
            let j = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            DispatchQueue.main.async { done(j) }
        }.resume()
    }

    func loadModels() {
        CLI.run(["models", "--json"]) { _, out in
            guard let d = out.data(using: .utf8),
                  let arr = try? JSONSerialization.jsonObject(with: d) as? [[String: Any]] else { return }
            self.models = arr.map {
                ModelInfo(path: $0["path"] as? String ?? "", name: $0["name"] as? String ?? "?",
                          format: $0["format"] as? String ?? "", size: $0["size"] as? Double ?? 0,
                          vision: $0["vision"] as? Bool ?? false, thinking: $0["thinking"] as? Bool ?? false,
                          supported: $0["supported"] as? Bool ?? false, efforts: $0["efforts"] as? [String] ?? [])
            }
        }
    }

    func command(_ args: [String], label: String) {
        busy = label
        message = nil
        CLI.run(args) { code, out in
            self.busy = nil
            if code != 0 {
                let tail = out.split(separator: "\n").suffix(3).joined(separator: "\n")
                self.message = L("llm \(label) failed: ", "llm \(label): ошибка: ") + tail
            }
            self.loadConfig()
            self.poll()
        }
    }
}

// MARK: - system monitor

final class SystemMonitor: ObservableObject {
    @Published var eLoad = 0.0
    @Published var pLoad = 0.0
    @Published var eHist: [Double] = []
    @Published var pHist: [Double] = []
    @Published var gpu = 0.0
    @Published var gpuMem = 0.0
    @Published var gpuHist: [Double] = []
    @Published var gpuMemHist: [Double] = []
    @Published var wired = 0.0
    @Published var active = 0.0
    @Published var compressed = 0.0
    @Published var total = Double(ProcessInfo.processInfo.physicalMemory)
    @Published var thermal = ProcessInfo.ThermalState.nominal
    @Published var load: [Double] = [0, 0, 0]

    private var prev: [(busy: UInt64, total: UInt64)] = []
    private let eCores: Int = {
        var v: Int32 = 0
        var size = MemoryLayout<Int32>.size
        return sysctlbyname("hw.perflevel1.logicalcpu", &v, &size, nil, 0) == 0 ? Int(v) : 0
    }()

    var used: Double { wired + active + compressed }

    func sample() {
        sampleCPU(); sampleGPU(); sampleMemory()
        thermal = ProcessInfo.processInfo.thermalState
        var la = [Double](repeating: 0, count: 3)
        getloadavg(&la, 3)
        load = la
    }

    private func push(_ v: Double, _ arr: inout [Double]) {
        arr.append(v)
        if arr.count > 60 { arr.removeFirst(arr.count - 60) }
    }

    private func sampleCPU() {
        var n: natural_t = 0
        var info: processor_info_array_t?
        var count: mach_msg_type_number_t = 0
        guard host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &n, &info, &count) == KERN_SUCCESS,
              let info else { return }
        defer { vm_deallocate(mach_task_self_, vm_address_t(bitPattern: info), vm_size_t(Int(count) * MemoryLayout<integer_t>.stride)) }
        var ticks: [(busy: UInt64, total: UInt64)] = []
        for i in 0..<Int(n) {
            let b = Int(CPU_STATE_MAX) * i
            let busy = UInt64(info[b + Int(CPU_STATE_USER)]) + UInt64(info[b + Int(CPU_STATE_SYSTEM)]) + UInt64(info[b + Int(CPU_STATE_NICE)])
            ticks.append((busy, busy + UInt64(info[b + Int(CPU_STATE_IDLE)])))
        }
        if prev.count == ticks.count {
            var eB = 0.0, eT = 0.0, pB = 0.0, pT = 0.0
            for (i, t) in ticks.enumerated() {
                let db = Double(t.busy &- prev[i].busy), dt = Double(t.total &- prev[i].total)
                if i < eCores { eB += db; eT += dt } else { pB += db; pT += dt }
            }
            eLoad = eT > 0 ? eB / eT : 0
            pLoad = pT > 0 ? pB / pT : 0
            push(eLoad, &eHist); push(pLoad, &pHist)
        }
        prev = ticks
    }

    private func sampleGPU() {
        var it: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) == KERN_SUCCESS else { return }
        defer { IOObjectRelease(it) }
        var entry = IOIteratorNext(it)
        while entry != 0 {
            if let raw = IORegistryEntryCreateCFProperty(entry, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0),
               let p = raw.takeRetainedValue() as? [String: Any] {
                gpu = Double(p["Device Utilization %"] as? Int ?? 0) / 100
                gpuMem = Double(p["In use system memory"] as? Int ?? 0)
            }
            IOObjectRelease(entry)
            entry = IOIteratorNext(it)
        }
        push(gpu, &gpuHist)
        push(total > 0 ? gpuMem / total : 0, &gpuMemHist)
    }

    private func sampleMemory() {
        var s = vm_statistics64()
        var c = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &s) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(c)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &c) }
        }
        guard kr == KERN_SUCCESS else { return }
        let page = Double(vm_kernel_page_size)
        wired = Double(s.wire_count) * page
        active = Double(s.active_count) * page
        compressed = Double(s.compressor_page_count) * page
    }
}

// MARK: - Engine tab

struct EngineTab: View {
    @ObservedObject var e: EngineState
    @ObservedObject var mon: SystemMonitor

    var body: some View {
        VStack(spacing: 10) {
            Card {
                Caption(text: L("Core", "Ядро"))
                if let h = e.health {
                    Row(label: L("Context · output", "Контекст · ответ"), value: "\(num(h["context"])) · \(num(h["max_tokens"]))",
                        hint: L("Max prompt+output tokens · output cap. Larger max_tokens are clamped, never rejected.",
                                "Максимум токенов промпта+ответа · потолок ответа. Больший max_tokens урезается, а не отклоняется."))
                    Row(label: L("Vision", "Зрение"), value: "\(h["vision"] as? String ?? L("off", "выкл"))",
                        hint: L("Loaded on the first image request (~0.5 GB)", "Загружается при первой картинке (~0,5 ГБ)"))
                } else {
                    Text(L("Offline. Press Start — the model wakes up in seconds.", "Не в сети. Нажмите «Запустить» — модель проснётся за секунды."))
                        .font(.system(size: 12)).foregroundColor(HUD.dim)
                }
                guardMeter
            }
            memoryCard
            if let last = e.health?["last"] as? [String: Any], !last.isEmpty {
                Card(glow: HUD.phosphor) {
                    Caption(text: L("Last request", "Последний запрос"), color: HUD.phosphor)
                    HStack(spacing: 0) {
                        metric(num(last["decode_tps"]), L("tok/s gen", "ток/с ген."), HUD.phosphor)
                        metric(num(last["prefill_tps"]), L("tok/s prefill", "ток/с prefill"), HUD.cyan)
                        metric(num(last["completion_tokens"]), L("tokens out", "токенов"), HUD.text)
                    }
                    Row(label: L("Prompt · from cache", "Промпт · из кэша"), value: "\(num(last["prompt_tokens"])) · \(num(last["cached_tokens"]))")
                    if let a = last["draft_accept"] as? Double {
                        Row(label: L("Drafts accepted", "Принято черновиков"), value: "\(Int(a * 100))%",
                            hint: L("Prompt-lookup speculative decoding acceptance", "Доля принятых токенов спекулятивного черновика"))
                    }
                }
            }
            tokens
        }
    }

    func metric(_ v: String, _ label: String, _ c: Color) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(v).font(HUD.mono(20, .semibold)).foregroundColor(c).shadow(color: c.opacity(0.6), radius: 6)
            Text(label.uppercased()).font(HUD.mono(8, .medium)).tracking(1.2).foregroundColor(HUD.dim)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Where the Mac's memory goes: the engine's parts, other apps and what is free.
    var memoryCard: some View {
        let b = e.health?["memory_breakdown"] as? [String: Any] ?? [:]
        func v(_ k: String) -> Double { Double((b[k] as? Int) ?? 0) }
        let engineTotal = Double((b["footprint"] as? Int) ?? 0)
        let others = max(0, mon.used - engineTotal)
        let free = max(0, mon.total - mon.used)
        let parts: [(String, Double, Color, String)] = [
            (L("Model weights", "Веса модели"), v("weights") + v("vision"), HUD.phosphor,
             L("Quantized weights (+ vision tower when loaded); pinned in RAM", "Квантованные веса (+ vision-башня, если загружена); закреплены в RAM")),
            (L("Context (KV cache)", "Контекст (KV-кэш)"), v("kv"), HUD.cyan,
             L("Attention keys/values of the current conversation; grows ~64 KB per token", "Ключи и значения attention текущего диалога; растёт на ~64 КБ на токен")),
            (L("Recurrent state + snapshots", "Состояние SSM + снимки"), v("ssm") + v("snapshots"), Color(red: 0.35, green: 0.55, blue: 1.0),
             L("GatedDeltaNet state and prefix-cache restore points (fixed size, not per token)", "Состояние GatedDeltaNet и точки восстановления prefix-кэша (фиксированный размер)")),
            (L("Engine buffers", "Буферы движка"), v("activations") + v("mlx_cache"), HUD.amber,
             L("Activations of a running request and the MLX allocator cache", "Активации текущего запроса и кэш аллокатора MLX")),
            (L("Engine process", "Процесс движка"), v("process"), Color.purple,
             L("Python, tokenizer, HTTP server (CPU memory)", "Python, токенизатор, HTTP-сервер (память CPU)")),
            (L("Other apps & macOS", "Другие приложения и macOS"), others, HUD.dim,
             L("Everything else in use: apps, system, compressed pages", "Всё остальное: приложения, система, сжатые страницы")),
        ]
        let limit = (e.health?["mem_limit"] as? Double) ?? Double(e.g("mem_limit", 88)) / 100
        return Card(glow: HUD.phosphor) {
            Caption(text: L("Memory", "Память"), color: HUD.phosphor)
            HStack(alignment: .firstTextBaseline) {
                Text(e.running ? L("Engine ", "Движок ") + gb(engineTotal) : L("Engine stopped", "Движок остановлен"))
                    .font(HUD.mono(14, .semibold)).foregroundColor(HUD.text)
                Spacer()
                Text("\(gb(mon.used)) / \(gb(mon.total))").font(HUD.mono(11)).foregroundColor(HUD.dim)
            }
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    HStack(spacing: 1) {
                        ForEach(0..<parts.count, id: \.self) { i in
                            let w = mon.total > 0 ? g.size.width * CGFloat(parts[i].1 / mon.total) : 0
                            if w >= 1 { Rectangle().fill(parts[i].2).frame(width: w) }
                        }
                        Rectangle().fill(HUD.track)
                    }.clipShape(Capsule())
                    Rectangle().fill(HUD.magenta).frame(width: 1.5, height: 12).offset(x: g.size.width * limit - 0.75)
                        .shadow(color: HUD.magenta.opacity(HUD.glow), radius: 3)
                }
            }.frame(height: 9)
            .help(L("Whole-Mac memory; the magenta tick is the RAM guard limit", "Память всего Mac; малиновая отметка — предел сторожа RAM"))
            ForEach(0..<parts.count, id: \.self) { i in
                if parts[i].1 > 0 || i == 0 {
                    memRow(parts[i].0, parts[i].1, parts[i].2, parts[i].3)
                }
            }
            memRow(L("Free", "Свободно"), free, HUD.track, L("Available without swapping", "Доступно без свопа"))
        }
    }

    func memRow(_ name: String, _ bytes: Double, _ c: Color, _ hint: String) -> some View {
        HStack {
            RoundedRectangle(cornerRadius: 2).fill(c).frame(width: 9, height: 9)
                .overlay(RoundedRectangle(cornerRadius: 2).strokeBorder(HUD.dim.opacity(0.5), lineWidth: 0.6))
            Text(name).font(.system(size: 12)).foregroundColor(HUD.dim)
            Spacer()
            Text(bytes >= 1_073_741_824 ? gb(bytes) : String(format: "%.0f MB", bytes / 1_048_576))
                .font(HUD.mono(12)).foregroundColor(HUD.text)
            Text(mon.total > 0 ? String(format: "%4.1f%%", bytes / mon.total * 100) : "")
                .font(HUD.mono(10)).foregroundColor(HUD.dim).frame(width: 44, alignment: .trailing)
        }.help(hint)
    }

    var guardMeter: some View {
        let limit = (e.health?["mem_limit"] as? Double) ?? Double(e.g("mem_limit", 88)) / 100
        let used = mon.total > 0 ? mon.used / mon.total : 0
        let color = used > limit ? HUD.magenta : (used > limit - 0.05 ? HUD.amber : HUD.cyan)
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(L("RAM guard", "Сторож RAM")).font(.system(size: 12)).foregroundColor(HUD.dim)
                Spacer()
                Text("\(Int(used * 100))% / \(Int(limit * 100))%").font(HUD.mono(12, .medium)).foregroundColor(color)
            }
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.06))
                    Capsule().fill(color).frame(width: g.size.width * min(1, used)).shadow(color: color, radius: 4)
                    Rectangle().fill(HUD.magenta).frame(width: 1.5, height: 11).offset(x: g.size.width * limit - 0.75).shadow(color: HUD.magenta, radius: 3)
                }
            }.frame(height: 5)
        }.padding(.top, 2)
        .help(L("System memory in use vs. the limit. Above it the engine trims output, shrinks prefill chunks or stops the request instead of letting the Mac swap.",
                "Занятая память системы и предел. Выше предела движок урезает ответ, уменьшает чанки prefill или останавливает запрос, не давая Mac уйти в своп."))
    }

    @ViewBuilder var tokens: some View {
        if let st = e.stats {
            let s = st["session"] as? [String: Any] ?? [:]
            let a = st["all_time"] as? [String: Any] ?? [:]
            Card {
                Caption(text: L("Token ledger", "Учёт токенов"))
                HStack {
                    Spacer()
                    Text(L("SESSION", "СЕССИЯ")).frame(width: 66, alignment: .trailing)
                    Text(L("ALL TIME", "ВСЕГО")).frame(width: 66, alignment: .trailing)
                }.font(HUD.mono(8, .semibold)).tracking(1.2).foregroundColor(HUD.dim)
                ledger(L("Requests", "Запросы"), s["requests"], a["requests"])
                ledger(L("Prompt", "Промпт"), s["prompt_tokens"], a["prompt_tokens"])
                ledger(L("From cache", "Из кэша"), s["cached_tokens"], a["cached_tokens"],
                       hint: L("Prompt tokens reused from the prefix cache instead of recomputed", "Токены промпта, взятые из prefix-кэша без пересчёта"))
                ledger(L("Generated", "Сгенерировано"), s["completion_tokens"], a["completion_tokens"])
                ledger(L("Cache hit", "Попадание в кэш"), pct(s["cache_hit"]), pct(a["cache_hit"]))
                ledger(L("Decode, tok/s", "Генерация, ток/с"), s["avg_decode_tps"], a["avg_decode_tps"])
                ledger(L("Prefill, tok/s", "Prefill, ток/с"), s["avg_prefill_tps"], a["avg_prefill_tps"])
                if let hist = st["history"] as? [[String: Any]], hist.count > 1 {
                    let vals = hist.map { ($0["decode_tps"] as? Double) ?? 0 }
                    let top = max(1, vals.max() ?? 1)
                    Sparkline(series: [(vals.map { $0 / top }, HUD.phosphor)], height: 30)
                        .help(L("Decode speed of the last \(hist.count) requests (peak \(num(top)) tok/s)",
                                "Скорость генерации последних \(hist.count) запросов (пик \(num(top)) ток/с)"))
                }
            }
        }
    }

    func pct(_ v: Any?) -> Any? { (v as? Double).map { "\(Int($0 * 100))%" } }

    func ledger(_ label: String, _ s: Any?, _ a: Any?, hint: String = "") -> some View {
        HStack {
            Text(label).font(.system(size: 12)).foregroundColor(HUD.dim)
            Spacer()
            Text(num(s)).frame(width: 66, alignment: .trailing)
            Text(num(a)).frame(width: 66, alignment: .trailing).foregroundColor(HUD.text.opacity(0.7))
        }.font(HUD.mono(12)).foregroundColor(HUD.text).help(hint)
    }
}

// MARK: - Monitor tab

struct MonitorTab: View {
    @ObservedObject var mon: SystemMonitor

    var body: some View {
        VStack(spacing: 10) {
            Card {
                Caption(text: "CPU")
                load(L("E-cores", "E-ядра"), mon.eLoad, HUD.amber)
                load(L("P-cores", "P-ядра"), mon.pLoad, HUD.cyan)
                Sparkline(series: [(mon.eHist, HUD.amber), (mon.pHist, HUD.cyan)])
                    .help(L("Efficiency (amber) and performance (cyan) cores, last 2 minutes", "E-ядра (янтарный) и P-ядра (циан) за 2 минуты"))
                Row(label: L("Thermal state", "Тепловое состояние"), value: thermal.0, hint: L(
                    "macOS thermal pressure. Serious/critical lowers GPU clocks and generation speed.",
                    "Тепловое давление macOS. «Высокое» и «Критическое» снижают частоты GPU и скорость генерации."), color: thermal.1)
                Row(label: L("Load average", "Средняя нагрузка"), value: mon.load.map { String(format: "%.2f", $0) }.joined(separator: " · "))
                Row(label: L("Uptime", "Время работы"), value: uptime)
            }
            Card(glow: HUD.phosphor) {
                Caption(text: "GPU", color: HUD.phosphor)
                load("GPU", mon.gpu, HUD.phosphor)
                Row(label: L("GPU memory", "Память GPU"), value: gb(mon.gpuMem),
                    hint: L("Unified memory held by GPU clients — the model weights live here", "Unified-память GPU-клиентов — здесь лежат веса модели"))
                Sparkline(series: [(mon.gpuHist, HUD.phosphor), (mon.gpuMemHist, HUD.cyan)])
                    .help(L("Utilization (green) and GPU memory (cyan)", "Загрузка (зелёный) и память GPU (циан)"))
            }
            Card {
                Caption(text: L("Memory", "Память"))
                HStack(alignment: .firstTextBaseline) {
                    Text("\(gb(mon.used)) / \(gb(mon.total))").font(HUD.mono(16, .semibold)).foregroundColor(HUD.text)
                    Spacer()
                    Text("\(mon.total > 0 ? Int(mon.used / mon.total * 100) : 0)%").font(HUD.mono(13)).foregroundColor(HUD.dim)
                }
                GeometryReader { g in
                    HStack(spacing: 1.5) {
                        seg(HUD.cyan, mon.wired, g.size.width)
                        seg(HUD.magenta, mon.active, g.size.width)
                        seg(Color.purple, mon.compressed, g.size.width)
                        Capsule().fill(Color.white.opacity(0.06))
                    }
                }.frame(height: 7)
                legend(HUD.cyan, L("Wired", "Закреплённая"), mon.wired, L("Cannot be paged out — the model weights are pinned here", "Не выгружается — здесь закреплены веса модели"))
                legend(HUD.magenta, L("Active", "Активная"), mon.active, "")
                legend(.purple, L("Compressed", "Сжатая"), mon.compressed, L("Pages macOS compressed to save RAM; growth means memory pressure", "Страницы, сжатые macOS; рост означает нехватку памяти"))
                legend(lightTheme ? Color.black.opacity(0.18) : Color.white.opacity(0.25), L("Free", "Свободная"), max(0, mon.total - mon.used), "")
            }
        }
    }

    func seg(_ c: Color, _ v: Double, _ w: CGFloat) -> some View {
        Capsule().fill(c).frame(width: max(0, w * CGFloat(mon.total > 0 ? v / mon.total : 0))).shadow(color: c.opacity(0.7 * HUD.glow), radius: 3)
    }

    var thermal: (String, Color) {
        switch mon.thermal {
        case .nominal: return (L("Normal", "Нормальное"), HUD.phosphor)
        case .fair: return (L("Fair", "Среднее"), HUD.amber)
        case .serious: return (L("Serious", "Высокое"), HUD.magenta)
        case .critical: return (L("Critical", "Критическое"), HUD.magenta)
        @unknown default: return ("?", HUD.dim)
        }
    }

    var uptime: String {
        let t = Int(ProcessInfo.processInfo.systemUptime)
        return t >= 86400 ? "\(t / 86400)d \(t % 86400 / 3600)h" : (t >= 3600 ? "\(t / 3600)h \(t % 3600 / 60)m" : "\(t / 60)m")
    }

    func load(_ name: String, _ v: Double, _ c: Color) -> some View {
        VStack(spacing: 4) {
            HStack {
                Text(name).font(.system(size: 13, weight: .medium)).foregroundColor(HUD.text)
                Spacer()
                Text("\(Int(v * 100))%").font(HUD.mono(13, .medium)).foregroundColor(c)
            }
            Bar(value: v, color: c)
        }
    }

    func legend(_ c: Color, _ name: String, _ v: Double, _ hint: String) -> some View {
        HStack {
            Circle().fill(c).frame(width: 7, height: 7).shadow(color: c.opacity(HUD.glow), radius: 2)
            Text(name).font(.system(size: 12)).foregroundColor(HUD.dim)
            Spacer()
            Text(gb(v)).font(HUD.mono(12)).foregroundColor(HUD.text)
        }.help(hint)
    }
}

// MARK: - Settings tab

struct SettingsTab: View {
    @ObservedObject var e: EngineState
    @State private var advanced = false

    var model: ModelInfo? { e.models.first { $0.path == e.modelPath } }

    var body: some View {
        VStack(spacing: 10) {
            Card {
                Caption(text: L("Model", "Модель"))
                Picker("", selection: Binding(get: { e.modelPath }, set: { e.setG("model", $0) })) {
                    ForEach(e.models) { m in
                        Text("\(m.name) · \(String(format: "%.1f", m.size)) GB" + (m.supported ? "" : L(" — unsupported", " — не поддерживается"))).tag(m.path)
                    }
                }.labelsHidden()
                Hint(text: L("Found in ~/.cache/huggingface/hub", "Найдены в ~/.cache/huggingface/hub"))
                field(L("Context, tokens", "Контекст, токенов"), int({ e.m("context", 0) }, { e.setM("context", $0) }),
                      hint: L("0 = auto: from RAM, model size and KV cost (~64 KB/token)", "0 = авто: из RAM, размера модели и KV (~64 КБ/токен)"))
                field(L("Max output, tokens", "Макс. ответ, токенов"), int({ e.m("max_output", 8192) }, { e.setM("max_output", $0) }),
                      hint: L("Cap for one answer; larger requests are clamped", "Потолок одного ответа; большие запросы урезаются"))
                if model?.thinking ?? true {
                    toggle(L("Reasoning by default", "Рассуждения по умолчанию"), Binding(get: { e.m("thinking", true) }, set: { e.setM("thinking", $0) }),
                           hint: L("Clients can switch it per request", "Клиенты могут переключать в каждом запросе"))
                    if let ef = model?.efforts, !ef.isEmpty {
                        Picker("", selection: Binding(get: { e.m("effort", "medium") }, set: { e.setM("effort", $0) })) {
                            ForEach(ef, id: \.self) { Text($0).tag($0) }
                        }.pickerStyle(.segmented).labelsHidden()
                        Hint(text: L("Reasoning effort · xhigh thinks much longer", "Глубина рассуждений · xhigh думает намного дольше"))
                    }
                }
                if model?.vision ?? false {
                    toggle(L("Images", "Картинки"), Binding(get: { e.m("vision", true) }, set: { e.setM("vision", $0) }),
                           hint: L("Tower loads on demand; off saves ~0.5 GB", "Башня грузится по требованию; выкл. экономит ~0,5 ГБ"))
                }
                toggle(L("Speculative decoding", "Спекулятивный декод"), Binding(get: { e.g("speculative", true) }, set: { e.setG("speculative", $0) }),
                       hint: L("~2.5× on code edits, identical output", "~×2,5 на правках кода, ответ не меняется"))
            }
            Card(glow: HUD.magenta) {
                Caption(text: L("Memory guard", "Сторож памяти"), color: HUD.magenta)
                HStack {
                    Text(L("System RAM limit", "Предел RAM системы")).font(.system(size: 12)).foregroundColor(HUD.text)
                    Spacer()
                    Text("\(e.g("mem_limit", 88))%").font(HUD.mono(13, .semibold)).foregroundColor(HUD.magenta)
                }
                Slider(value: Binding(get: { Double(e.g("mem_limit", 88)) }, set: { e.setG("mem_limit", Int($0)) }), in: 70...95, step: 1)
                    .tint(HUD.magenta)
                Hint(text: L("On 16 GB keep ≤ 88%: above ~90% macOS swaps and freezes", "На 16 ГБ держите ≤ 88%: выше ~90% macOS уходит в своп и зависает"))
                field(L("Sleep after idle, min", "Сон после простоя, мин"), int({ e.g("idle_sleep", 15) }, { e.setG("idle_sleep", $0) }),
                      hint: L("Unloads the weights (frees ~7 GB); the next request wakes the model in ~2 s. 0 = never.",
                              "Выгружает веса (освобождает ~7 ГБ); следующий запрос будит модель за ~2 с. 0 = никогда."))
                field(L("SSD prefix cache, GB", "Кэш префиксов на SSD, ГБ"), int({ e.g("disk_cache_gb", 8) }, { e.setG("disk_cache_gb", $0) }),
                      hint: L("Long prompts (agent system prompts) are restored from disk instead of re-prefilled. 0 = off.",
                              "Длинные промпты (системные промпты агентов) восстанавливаются с диска без пересчёта. 0 = выкл."))
            }
            Card {
                Caption(text: L("Network & interface", "Сеть и интерфейс"))
                field(L("Port", "Порт"), int({ e.g("port", 8090) }, { e.setG("port", $0) }),
                      hint: L("OpenAI-compatible API on http://localhost:<port>/v1", "OpenAI-совместимый API на http://localhost:<порт>/v1"))
                toggle(L("LAN access (0.0.0.0)", "Доступ из сети (0.0.0.0)"), Binding(get: { e.g("lan", false) }, set: { e.setG("lan", $0) }),
                       hint: L("Other devices can use the model — set an API key", "Модель доступна другим устройствам — задайте API-ключ"))
                HStack {
                    Text(L("API key", "API-ключ")).font(.system(size: 12)).foregroundColor(HUD.text)
                    Spacer()
                    SecureField(L("not set — open access", "не задан — без ключа"), text: Binding(get: { e.g("api_key", "") }, set: { e.setG("api_key", $0) }))
                        .frame(width: 170).textFieldStyle(.roundedBorder)
                }
                Hint(text: L("Clients send it as Authorization: Bearer <key>", "Клиенты передают его как Authorization: Bearer <ключ>"))
                Divider().opacity(0.3).padding(.vertical, 2)
                labeledPicker(L("Language", "Язык"), Binding(get: { e.g("lang", "auto") }, set: { e.setG("lang", $0); e.loadConfig() }),
                              [("auto", L("System", "Системный")), ("en", "English"), ("ru", "Русский")])
                labeledPicker(L("Theme", "Тема"), Binding(get: { e.g("theme", "system") }, set: { e.setG("theme", $0); e.loadConfig(); e.onTheme?() }),
                              [("system", L("System", "Системная")), ("dark", L("Dark", "Тёмная")), ("light", L("Light", "Светлая"))])
            }
            Card(glow: HUD.amber) {
                DisclosureGroup(isExpanded: $advanced) {
                    VStack(alignment: .leading, spacing: 7) {
                        Hint(text: L("Fine tuning. Empty = model / engine default.", "Тонкая настройка. Пусто = значение модели или движка."))
                        adv("temperature", "Temperature", L("Randomness · presets 1.0 (reasoning) / 0.7 · 0 = greedy", "Случайность · пресеты 1.0 (рассуждения) / 0.7 · 0 = greedy"))
                        adv("top_p", "Top-p", L("Nucleus cutoff, 0–1", "Порог nucleus, 0–1"))
                        adv("top_k", "Top-k", L("Sample among the k most likely tokens", "Выбор среди k самых вероятных токенов"))
                        adv("min_p", "Min-p", L("Drop tokens below min_p × top probability", "Отсечь токены ниже min_p × максимум"))
                        adv("presence_penalty", "Presence", L("Penalty for repeated tokens", "Штраф за повторы"))
                        adv("draft_max", L("Draft length", "Длина черновика"), L("Longest speculative draft, 1–15 (15)", "Самый длинный черновик, 1–15 (15)"))
                        adv("ngram_min", L("N-gram min", "Мин. n-грамма"), L("Shortest match that starts a draft, 1–4 (2)", "Минимальное совпадение для черновика, 1–4 (2)"))
                        adv("cache_limit_mb", L("MLX cache, MB", "Кэш MLX, МБ"), L("Allocator cache between requests (128)", "Кэш аллокатора между запросами (128)"))
                        field(L("Prefill chunk", "Чанк prefill"), int({ e.g("prefill_step", 1024) }, { e.setG("prefill_step", $0) }),
                              hint: L("1024 is fastest; the guard shrinks it when RAM is tight", "1024 быстрее всего; сторож уменьшает при нехватке RAM"))
                        field(L("Image max pixels", "Макс. пикселей"), int({ e.m("max_pixels", 1_048_576) }, { e.setM("max_pixels", $0) }),
                              hint: L("Bigger images are downscaled; ~1 token per 32×32 px", "Большие картинки уменьшаются; ~1 токен на 32×32 px"))
                    }.padding(.top, 6)
                } label: {
                    Text(L("ADVANCED", "ДОПОЛНИТЕЛЬНО")).font(HUD.mono(10, .semibold)).tracking(2.2).foregroundColor(HUD.amber)
                }
            }
            Button(action: {
                e.saveConfig()
                e.command([e.running ? "restart" : "start"], label: e.running ? "restart" : "start")
            }) {
                Text(e.running ? L("APPLY & RESTART", "ПРИМЕНИТЬ И ПЕРЕЗАПУСТИТЬ") : L("START WITH THESE SETTINGS", "ЗАПУСТИТЬ С ЭТИМИ НАСТРОЙКАМИ"))
                    .font(HUD.mono(11, .semibold)).tracking(1.6).frame(maxWidth: .infinity).padding(.vertical, 7)
            }.buttonStyle(NeonButton(color: e.running ? HUD.cyan : HUD.phosphor)).disabled(e.busy != nil)
            Hint(text: L("Changes are saved as you edit; the engine applies them when it (re)starts",
                         "Изменения сохраняются сразу; движок применяет их при запуске или перезапуске"))
        }
    }

    func labeledPicker(_ label: String, _ b: Binding<String>, _ options: [(String, String)]) -> some View {
        HStack {
            Text(label).font(.system(size: 12)).foregroundColor(HUD.text)
            Spacer()
            Picker("", selection: b) { ForEach(options, id: \.0) { Text($0.1).tag($0.0) } }
                .pickerStyle(.segmented).labelsHidden().frame(width: 240)
        }
    }

    func int(_ get: @escaping () -> Int, _ set: @escaping (Int) -> Void) -> Binding<String> {
        Binding(get: { "\(get())" }, set: { if let v = Int($0.filter(\.isNumber)) { set(v) } })
    }

    func field(_ label: String, _ b: Binding<String>, hint: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.system(size: 12)).foregroundColor(HUD.text)
                Spacer()
                TextField("", text: b).font(HUD.mono(12)).frame(width: 90).multilineTextAlignment(.trailing).textFieldStyle(.roundedBorder)
            }
            Hint(text: hint)
        }.help(hint)
    }

    func toggle(_ label: String, _ b: Binding<Bool>, hint: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Toggle(isOn: b) { Text(label).font(.system(size: 12)).foregroundColor(HUD.text) }.toggleStyle(.switch).tint(HUD.cyan)
            Hint(text: hint)
        }.help(hint)
    }

    func adv(_ key: String, _ label: String, _ hint: String) -> some View {
        HStack {
            Text(label).font(.system(size: 12)).foregroundColor(HUD.text)
            Spacer()
            TextField(L("auto", "авто"), text: Binding(get: { e.adv(key) }, set: { e.setAdv(key, $0) }))
                .font(HUD.mono(12)).frame(width: 80).multilineTextAlignment(.trailing).textFieldStyle(.roundedBorder)
        }.help(hint)
    }
}


// MARK: - Models tab (installed · switch · hub search · download)

struct HubModel: Identifiable {
    let id: String
    let downloads: Int
    let size: Double?
    let compatible, local, fits: Bool
}

final class Downloader: ObservableObject {
    @Published var repo: String? = nil
    @Published var fraction = 0.0
    @Published var file = ""
    @Published var error: String? = nil
    private var proc: Process?

    func start(_ repo: String, done: @escaping () -> Void) {
        self.repo = repo; fraction = 0; file = ""; error = nil
        let p = Process()
        p.executableURL = URL(fileURLWithPath: Paths.shared.root + "/llm")
        p.arguments = ["pull", repo, "--progress"]
        p.currentDirectoryURL = URL(fileURLWithPath: Paths.shared.root)
        var env = ProcessInfo.processInfo.environment
        env["LLM_HOME"] = Paths.shared.home
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        var buffer = ""
        pipe.fileHandleForReading.readabilityHandler = { h in
            let chunk = String(decoding: h.availableData, as: UTF8.self)
            DispatchQueue.main.async {
                buffer += chunk
                while let nl = buffer.firstIndex(of: "\n") {
                    let line = String(buffer[..<nl]); buffer.removeSubrange(...nl)
                    let parts = line.split(separator: " ", maxSplits: 3).map(String.init)
                    if parts.first == "PROGRESS", parts.count >= 3, let d = Double(parts[1]), let t = Double(parts[2]), t > 0 {
                        self.fraction = d / t
                        self.file = parts.count > 3 ? parts[3] : ""
                    } else if line.contains("✗") || line.lowercased().contains("error") {
                        self.error = line
                    }
                }
            }
        }
        p.terminationHandler = { proc in
            DispatchQueue.main.async {
                pipe.fileHandleForReading.readabilityHandler = nil
                if proc.terminationStatus != 0 && self.error == nil { self.error = L("Download failed", "Загрузка не удалась") }
                self.repo = nil
                self.proc = nil
                done()
            }
        }
        proc = p
        try? p.run()
    }

    func cancel() { proc?.terminate() }
}

struct ModelsTab: View {
    @ObservedObject var e: EngineState
    @StateObject private var dl = Downloader()
    @State private var query = "qwen3.5"
    @State private var results: [HubModel] = []
    @State private var searching = false
    @State private var searchError: String? = nil

    var body: some View {
        VStack(spacing: 10) {
            Card {
                Caption(text: L("Installed", "Установленные"))
                if e.models.isEmpty { Hint(text: L("No models yet — search the hub below", "Моделей пока нет — найдите ниже")) }
                ForEach(e.models) { m in
                    HStack(spacing: 8) {
                        Circle().fill(m.path == e.modelPath ? HUD.phosphor : HUD.track).frame(width: 7, height: 7)
                            .shadow(color: HUD.phosphor.opacity(m.path == e.modelPath ? HUD.glow : 0), radius: 3)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(m.name).font(.system(size: 12, weight: .medium)).foregroundColor(HUD.text).lineLimit(1)
                            Text("\(String(format: "%.1f", m.size)) GB · \(m.format)" + (m.supported ? "" : L(" · unsupported", " · не поддерживается")))
                                .font(HUD.mono(9)).foregroundColor(m.supported ? HUD.dim : HUD.magenta).lineLimit(1)
                        }
                        Spacer()
                        if m.path == e.modelPath {
                            Text(L("ACTIVE", "АКТИВНА")).font(HUD.mono(9, .semibold)).tracking(1.2).foregroundColor(HUD.phosphor)
                        } else if m.supported {
                            Button(L("Use", "Включить")) {
                                e.setG("model", m.path)
                                if e.running { e.command(["restart"], label: "restart") }
                            }.buttonStyle(NeonButton(color: HUD.cyan)).font(HUD.mono(10, .semibold)).disabled(e.busy != nil)
                                .help(L("Make this the active model (restarts the engine if it runs)", "Сделать активной (движок перезапустится, если работает)"))
                        }
                    }
                }
            }
            Card(glow: HUD.phosphor) {
                Caption(text: L("Hugging Face hub", "Hugging Face"), color: HUD.phosphor)
                HStack {
                    TextField(L("Search MLX models", "Поиск MLX-моделей"), text: $query)
                        .textFieldStyle(.roundedBorder).font(HUD.mono(12)).onSubmit(search)
                    Button(action: search) { Image(systemName: searching ? "hourglass" : "magnifyingglass").frame(width: 24, height: 20) }
                        .buttonStyle(NeonButton(color: HUD.phosphor)).disabled(searching)
                }
                Hint(text: L("MLX-tagged repos, most downloaded first. Mirrors: set HF_ENDPOINT; private repos: HF_TOKEN.",
                             "Репозитории с тегом MLX, популярные сверху. Зеркало: HF_ENDPOINT; закрытые модели: HF_TOKEN."))
                if let err = searchError { Text(err).font(HUD.mono(10)).foregroundColor(HUD.magenta) }
                if let repo = dl.repo {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(L("Downloading ", "Загрузка ") + repo).font(.system(size: 11, weight: .medium)).foregroundColor(HUD.text).lineLimit(1)
                            Spacer()
                            Text("\(Int(dl.fraction * 100))%").font(HUD.mono(11)).foregroundColor(HUD.phosphor)
                            Button(action: dl.cancel) { Image(systemName: "xmark") }.buttonStyle(.plain).foregroundColor(HUD.dim)
                                .help(L("Cancel (resumes next time)", "Отменить (докачается в следующий раз)"))
                        }
                        Bar(value: dl.fraction, color: HUD.phosphor)
                        Text(dl.file).font(HUD.mono(9)).foregroundColor(HUD.dim).lineLimit(1)
                    }
                }
                if let err = dl.error { Text(err).font(HUD.mono(10)).foregroundColor(HUD.magenta).lineLimit(2) }
                ForEach(results) { r in
                    HStack(spacing: 8) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(r.id).font(.system(size: 11, weight: .medium)).foregroundColor(HUD.text).lineLimit(1)
                            HStack(spacing: 6) {
                                badge(r.size.map { String(format: "%.1f GB", $0) } ?? "? GB", HUD.dim)
                                badge("⬇ \(r.downloads >= 1000 ? "\(r.downloads / 1000)K" : "\(r.downloads)")", HUD.dim)
                                badge(r.compatible ? L("engine ✓", "движок ✓") : L("other engine", "другой движок"), r.compatible ? HUD.phosphor : HUD.dim)
                                if r.size != nil { badge(r.fits ? L("fits RAM", "влезет") : L("too big", "не влезет"), r.fits ? HUD.cyan : HUD.magenta) }
                            }
                        }
                        Spacer()
                        if r.local {
                            Text(L("HAVE", "ЕСТЬ")).font(HUD.mono(9, .semibold)).foregroundColor(HUD.phosphor)
                        } else {
                            Button(action: { dl.start(r.id) { e.loadModels(); search() } }) { Image(systemName: "arrow.down.circle") }
                                .buttonStyle(.plain).font(.system(size: 16)).foregroundColor(r.compatible ? HUD.phosphor : HUD.dim)
                                .disabled(dl.repo != nil)
                                .help(r.compatible ? L("Download (resumable)", "Скачать (с докачкой)")
                                                   : L("Downloads, but this engine cannot run it", "Скачается, но этот движок её не запустит"))
                        }
                    }.padding(.vertical, 2)
                }
            }
        }.onAppear { if results.isEmpty { search() } }
    }

    func badge(_ t: String, _ c: Color) -> some View {
        Text(t).font(HUD.mono(9)).foregroundColor(c)
            .padding(.horizontal, 5).padding(.vertical, 1)
            .overlay(Capsule().strokeBorder(c.opacity(0.4), lineWidth: 0.6))
    }

    func search() {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return }
        searching = true; searchError = nil
        CLI.run(["search", q, "--json"]) { code, out in
            searching = false
            guard code == 0, let d = out.data(using: .utf8),
                  let arr = try? JSONSerialization.jsonObject(with: d) as? [[String: Any]] else {
                searchError = L("Search failed — check the connection", "Поиск не удался — проверьте сеть"); return
            }
            results = arr.map {
                HubModel(id: $0["id"] as? String ?? "?", downloads: $0["downloads"] as? Int ?? 0, size: $0["size"] as? Double,
                         compatible: $0["compatible"] as? Bool ?? false, local: $0["local"] as? Bool ?? false, fits: $0["fits"] as? Bool ?? false)
            }
        }
    }
}

// MARK: - Tools tab

struct ToolsTab: View {
    @ObservedObject var e: EngineState
    var body: some View {
        VStack(spacing: 10) {
            Card {
                Caption(text: L("Interface", "Интерфейс"))
                tool("bubble.left.and.bubble.right", L("Chat", "Чат"), L("Streaming chat in Terminal · /image /think /effort", "Чат в Терминале · /image /think /effort")) {
                    CLI.terminal(e.running ? "./llm chat && ./llm" : "./llm up")
                }
                tool("cpu", L("Coding agent", "Агент для кода"), L("Reads, searches, edits files and runs commands — asks first", "Читает, ищет, правит файлы и запускает команды — с подтверждением")) {
                    CLI.terminal("LLM_CWD=$HOME ./llm agent")
                }
                Menu {
                    Button("Continue") { e.command(["connect", "continue"], label: "connect continue") }
                    Button("Kilo Code") { e.command(["connect", "kilo"], label: "connect kilo") }
                    Button("opencode") { e.command(["connect", "opencode"], label: "connect opencode") }
                } label: {
                    toolLabel("puzzlepiece.extension", L("Connect an editor", "Подключить редактор"), "Continue · Kilo Code · opencode")
                }.menuStyle(.borderlessButton).menuIndicator(.hidden)
                    .help(L("Writes URL and limits into the editor config (.bak kept)", "Записывает адрес и лимиты в конфиг редактора (копия .bak)"))
                tool("link", L("Copy API URL", "Скопировать URL API"), e.apiURL) {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(e.apiURL, forType: .string)
                }
            }
            Card(glow: HUD.phosphor) {
                Caption(text: L("Diagnostics", "Диагностика"), color: HUD.phosphor)
                tool("text.alignleft", L("Logs", "Логи"), L("Engine log, live", "Лог движка в реальном времени")) { CLI.terminal("./llm logs") }
                tool("checkmark.seal", L("Tests", "Тесты"), L("Unit tests (+ integration when stopped)", "Юнит-тесты (+ интеграционные при остановке)")) { CLI.terminal("./llm test") }
                tool("speedometer", L("Benchmarks", "Бенчмарки"), L("Compare with the baseline · stop the engine first", "Сравнение с базой · сначала остановите движок")) { CLI.terminal("./llm perf") }
                tool("terminal", L("Terminal menu", "Меню в Терминале"), "./llm") { CLI.terminal("./llm") }
            }
        }
    }

    func toolLabel(_ icon: String, _ title: String, _ sub: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon).font(.system(size: 14)).foregroundColor(HUD.cyan).frame(width: 22).shadow(color: HUD.cyan.opacity(0.7), radius: 3)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.system(size: 13, weight: .medium)).foregroundColor(HUD.text)
                Text(sub).font(.system(size: 10)).foregroundColor(HUD.dim).lineLimit(1)
            }
            Spacer()
            Image(systemName: "chevron.right").font(.system(size: 10)).foregroundColor(HUD.dim)
        }.contentShape(Rectangle())
    }

    func tool(_ icon: String, _ title: String, _ sub: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) { toolLabel(icon, title, sub) }.buttonStyle(.plain).help(sub)
    }
}

// MARK: - root

struct NeonButton: ButtonStyle {
    var color: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(configuration.isPressed ? HUD.ink : color)
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(configuration.isPressed ? color : color.opacity(0.10)))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(color.opacity(0.7), lineWidth: 0.8))
            .shadow(color: color.opacity(0.45), radius: 6)
    }
}

struct ContentView: View {
    @ObservedObject var e: EngineState
    @ObservedObject var mon: SystemMonitor
    @State private var tab = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Picker("", selection: $tab) {
                Text(L("Engine", "Движок")).tag(0)
                Text(L("Monitor", "Монитор")).tag(1)
                Text(L("Models", "Модели")).tag(2)
                Text(L("Settings", "Настройки")).tag(3)
                Text(L("Tools", "Тулзы")).tag(4)
            }.pickerStyle(.segmented).labelsHidden()
            ScrollView(showsIndicators: false) {
                Group {
                    switch tab {
                    case 0: EngineTab(e: e, mon: mon)
                    case 1: MonitorTab(mon: mon)
                    case 2: ModelsTab(e: e)
                    case 3: SettingsTab(e: e)
                    default: ToolsTab(e: e)
                    }
                }
            }.frame(height: ContentView.scrollHeight)
            if let msg = e.message {
                Text(msg).font(HUD.mono(10)).foregroundColor(HUD.magenta).lineLimit(3)
            }
            HStack {
                Text("AI-llm-ENGINE · SECTION 9 HUD").font(HUD.mono(8, .medium)).tracking(1.6).foregroundColor(HUD.dim.opacity(0.7))
                Spacer()
                Button(L("Quit", "Выйти")) { NSApp.terminate(nil) }.buttonStyle(.plain).font(HUD.mono(10)).foregroundColor(HUD.dim)
                    .help(L("Quit LLMBar; the model keeps running", "Закрыть LLMBar; модель продолжит работать"))
            }
        }
        .padding(16)
        .frame(width: 392)
        .background(background)
        .environment(\.colorScheme, lightTheme ? .light : .dark)
        .tint(HUD.cyan)
        .onAppear { e.loadModels() }
        .background(Button("") { NSApp.keyWindow?.orderOut(nil) }.keyboardShortcut(.cancelAction).opacity(0))
    }

    var background: some View {
        ZStack {
            VisualEffect()
            // light tint over the blur — the desktop should show through like frosted glass
            LinearGradient(colors: lightTheme
                           ? [Color(red: 0.97, green: 0.99, blue: 1.0).opacity(0.50), Color(red: 0.90, green: 0.94, blue: 0.96).opacity(0.42)]
                           : [HUD.ink.opacity(0.55), Color(red: 0.03, green: 0.07, blue: 0.10).opacity(0.48)],
                           startPoint: .top, endPoint: .bottom)
            // specular sheen across the top
            LinearGradient(colors: [Color.white.opacity(lightTheme ? 0.35 : 0.07), .clear], startPoint: .top, endPoint: .center)
            // scanlines
            Canvas { ctx, size in
                var y: CGFloat = 0
                while y < size.height {
                    ctx.fill(Path(CGRect(x: 0, y: y, width: size.width, height: 0.6)), with: .color(lightTheme ? .black.opacity(0.018) : .white.opacity(0.025)))
                    y += 3
                }
            }
            RadialGradient(colors: [HUD.cyan.opacity(0.10), .clear], center: .topTrailing, startRadius: 0, endRadius: 320)
            // glass rim: a thin light edge that catches the "light" at the top
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(LinearGradient(colors: [Color.white.opacity(lightTheme ? 0.7 : 0.22), HUD.cyan.opacity(0.10), HUD.cyan.opacity(0.06)],
                                             startPoint: .top, endPoint: .bottom), lineWidth: 1)
        }.ignoresSafeArea()
    }

    var header: some View {
        HStack(alignment: .center, spacing: 11) {
            ZStack {
                Circle().fill(statusColor.opacity(0.25)).frame(width: 18, height: 18).blur(radius: 3)
                Circle().fill(statusColor).frame(width: 8, height: 8).shadow(color: statusColor, radius: 5)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text((e.health?["model"] as? String) ?? e.models.first(where: { $0.path == e.modelPath })?.name ?? L("No model", "Нет модели"))
                    .font(.system(size: 13, weight: .semibold)).foregroundColor(HUD.text).lineLimit(1)
                Text(e.busy.map { L("EXEC llm \($0)…", "ВЫПОЛНЯЮ llm \($0)…") }
                     ?? (e.sleeping ? L("SLEEPING · wakes on request", "СПИТ · проснётся по запросу") : (e.running ? e.apiURL : L("OFFLINE", "НЕ В СЕТИ"))))
                    .font(HUD.mono(10)).tracking(0.6).foregroundColor(e.running ? HUD.cyan : HUD.dim)
            }
            Spacer()
            if e.running {
                iconButton("arrow.clockwise", L("Restart", "Перезапустить")) { e.command(["restart"], label: "restart") }
                iconButton("power", L("Stop", "Остановить")) { e.command(["stop"], label: "stop") }
            } else {
                Button(action: { e.command(["start"], label: "start") }) {
                    Text(L("START", "ЗАПУСК")).font(HUD.mono(11, .bold)).tracking(1.6).padding(.horizontal, 12).padding(.vertical, 5)
                }.buttonStyle(NeonButton(color: HUD.phosphor)).disabled(e.busy != nil)
                    .help(L("Loads the selected model (or the first one that fits)", "Загружает выбранную модель (или первую подходящую)"))
            }
        }
    }

    /// Fit under the menu bar on small screens (the panel window must stay on screen).
    static var scrollHeight: CGFloat {
        let visible = NSScreen.main?.visibleFrame.height ?? 800
        return max(260, min(520, visible - 210))
    }

    var statusColor: Color { e.busy != nil ? HUD.amber : (e.sleeping ? HUD.cyan.opacity(0.6) : (e.running ? HUD.phosphor : HUD.dim)) }

    func iconButton(_ icon: String, _ hint: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon).font(.system(size: 12, weight: .semibold)).frame(width: 26, height: 22)
        }.buttonStyle(NeonButton(color: HUD.cyan)).disabled(e.busy != nil).help(hint)
    }
}

struct VisualEffect: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = lightTheme ? .popover : .hudWindow
        v.blendingMode = .behindWindow
        v.state = .active
        return v
    }
    func updateNSView(_ v: NSVisualEffectView, context: Context) {}
}

// MARK: - app shell (AppKit: status item, panel, window)

/// Borderless windows refuse key status by default; the panel needs it for text fields.
final class PanelWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    /// Menu-bar mark: a tiny CRT bezel with "Ai" and a cursor — the app icon at 16 pt, as a template
    /// so it follows the menu bar's light/dark tint.
    static func trayIcon() -> NSImage {
        let img = NSImage(size: NSSize(width: 24, height: 16), flipped: false) { r in
            NSColor.black.set()
            let bezel = NSBezierPath(roundedRect: r.insetBy(dx: 1, dy: 1.2), xRadius: 4.2, yRadius: 4.2)
            bezel.lineWidth = 1.4
            bezel.stroke()
            let font = NSFont.monospacedSystemFont(ofSize: 8.5, weight: .heavy)
            let text = NSAttributedString(string: "Ai", attributes: [.font: font, .foregroundColor: NSColor.black])
            let size = text.size()
            text.draw(at: NSPoint(x: 4.2, y: (r.height - size.height) / 2 + 0.4))
            NSBezierPath(rect: NSRect(x: 4.2 + size.width + 1.2, y: 4.2, width: 4.4, height: 1.6)).fill()
            return true
        }
        img.isTemplate = true
        return img
    }

    let engine = EngineState()
    let monitor = SystemMonitor()
    let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    var window: NSWindow?
    var fastTimer: Timer?
    var hiddenAt = Date.distantPast

    func applicationDidFinishLaunching(_ n: Notification) {
        engine.onTheme = { [weak self] in self?.applyTheme() }
        applyTheme()
        item.autosaveName = "AI-llm-Engine"  // keeps a ⌘-dragged position across launches
        if let b = item.button {
            b.image = AppDelegate.trayIcon()
            b.imagePosition = .imageLeft
            b.action = #selector(toggle)
            b.target = self
        }
        engine.poll()
        monitor.sample()
        refreshIcon()
        // slow poll for the status icon; full sampling only while the UI is visible
        Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            guard let self else { return }
            if !self.uiVisible { self.engine.poll() }
            self.refreshIcon()
        }
    }

    func applyTheme() {
        let a = NSAppearance(named: lightTheme ? .aqua : .darkAqua)
        window?.appearance = a
    }

    var uiVisible: Bool { window?.isVisible ?? false }

    func refreshIcon() {
        let busy = (engine.health?["busy"] as? Bool) ?? false
        item.button?.title = engine.busy != nil ? "◌" : (engine.running ? (busy ? "◉" : "●") : "○")
        item.button?.toolTip = engine.running ? "AI-llm-Engine · \(engine.apiURL)" : L("AI-llm-Engine · offline", "AI-llm-Engine · не в сети")
    }

    func startFastSampling() {
        guard fastTimer == nil else { return }
        monitor.sample()
        engine.poll()
        fastTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            guard let self else { return }
            if !self.uiVisible { self.fastTimer?.invalidate(); self.fastTimer = nil; return }
            self.monitor.sample()
            self.engine.poll()
            self.refreshIcon()
        }
    }

    /// Status-icon click: toggle the panel window (no NSPopover — it cannot be kept on screen
    /// when the icon is hidden by the notch).
    @objc func toggle() {
        if let w = window, w.isVisible { w.orderOut(nil); return }
        // a click on the icon first takes focus from the panel (which hides it) — don't reopen
        if Date().timeIntervalSince(hiddenAt) < 0.3 { return }
        // the user just clicked the icon, so it is visible: open the panel right under it
        showWindow(anchorX: item.button?.window?.frame.midX)
    }

    func showWindow(anchorX: CGFloat? = nil) {
        if window == nil {
            // borderless panel: our own rounded corners on all four sides + the system shadow
            let w = PanelWindow(contentRect: NSRect(x: 0, y: 0, width: 392, height: 660),
                                styleMask: [.borderless, .fullSizeContentView], backing: .buffered, defer: false)
            w.title = "AI-llm-Engine"
            w.hasShadow = true
            w.isMovableByWindowBackground = true
            w.appearance = NSAppearance(named: lightTheme ? .aqua : .darkAqua)
            w.isReleasedWhenClosed = false
            w.isOpaque = false
            w.backgroundColor = .clear  // let the vibrancy blur the desktop behind the panel
            w.contentViewController = NSHostingController(rootView: ContentView(e: engine, mon: monitor)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous)).ignoresSafeArea())
            w.delegate = self  // closes on Esc or when focus leaves
            window = w
        }
        // under the icon if it is visible, else the top-right corner — always fully on screen
        if let w = window, !w.isVisible || anchorX != nil, let screen = NSScreen.main {
            let v = screen.visibleFrame
            // the frame is still empty before SwiftUI lays out — size from the content instead
            var size = w.contentViewController?.view.fittingSize ?? .zero
            if size.width < 100 || size.height < 100 { size = NSSize(width: 392, height: 680) }
            let h = min(size.height, v.height - 8)
            // Under the icon when opened by a click on it; otherwise (launch, double-click) the
            // top-right corner like Control Center — a status item can be covered by the notch or
            // the active app's menus while macOS still reports it visible.
            var x = anchorX.map { $0 - size.width / 2 } ?? (v.maxX - size.width - 12)
            x = min(max(x, v.minX + 8), v.maxX - size.width - 8)
            w.setFrame(NSRect(x: x, y: v.maxY - h - 6, width: size.width, height: h), display: true)
        }
        engine.loadConfig()
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
        startFastSampling()
    }

    /// First launch / icon hidden by the notch: show a window; otherwise open the panel.
    /// Double-click on the app while it runs: open the panel under its icon.
    func reveal() {
        showWindow(anchorX: item.button?.window?.frame.midX)
    }

    func windowDidResignKey(_ n: Notification) {
        // like Control Center: clicking anywhere else hides the panel (the model keeps running)
        window?.orderOut(nil)
        hiddenAt = Date()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        reveal()
        return false
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()

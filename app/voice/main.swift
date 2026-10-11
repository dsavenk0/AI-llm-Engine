// LLM Voice — a small, separate app: press the orb, speak, hear the local model answer.
// Not part of LLMEditor or LLMBar; it only runs while you use it, talks to the engine's OpenAI-compatible API on
// this Mac, and starts the model through the same `llm` command everything else uses.
import AppKit
import AVFoundation
import NaturalLanguage
import Speech
import SwiftUI

// MARK: - configuration (same files and arguments as the other apps)

enum Setup {
    static var llmRoot: String = {
        var root = Bundle.main.object(forInfoDictionaryKey: "LLMRoot") as? String ?? FileManager.default.currentDirectoryPath
        let a = CommandLine.arguments
        if let i = a.firstIndex(of: "--llm-root"), i + 1 < a.count { root = a[i + 1] }
        return root
    }()

    private static var userConfig: [String: Any] = {
        let path = NSHomeDirectory() + "/.llm/config.json"
        guard let d = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return [:] }
        return o
    }()

    static var port: Int = {
        let a = CommandLine.arguments
        if let i = a.firstIndex(of: "--engine-port"), i + 1 < a.count, let p = Int(a[i + 1]) { return p }
        return userConfig["port"] as? Int ?? 8090
    }()

    static var apiKey: String { userConfig["api_key"] as? String ?? "" }
    static var theme: String { userConfig["theme"] as? String ?? "system" }
    /// LLMBar's transparency slider (0.15 … 0.85): how much of the desktop shows through the panel.
    static var transparency: Double { min(0.85, max(0.15, userConfig["glass_transparency"] as? Double ?? 0.55)) }
    static var base: URL { URL(string: "http://127.0.0.1:\(port)")! }
}

// MARK: - LLMBar's look: graphite-blue glass, cyan / phosphor-mint accents, monospaced labels

/// Light or dark, from the same "theme" setting LLMBar uses (system / dark / light); `--theme` overrides it.
var lightTheme = false

func resolveTheme() {
    let a = CommandLine.arguments
    var setting = Setup.theme
    if let i = a.firstIndex(of: "--theme"), i + 1 < a.count { setting = a[i + 1] }
    switch setting {
    case "light": lightTheme = true
    case "dark": lightTheme = false
    default:
        lightTheme = UserDefaults.standard.string(forKey: "AppleInterfaceStyle") == nil   // unset = Light
    }
}

enum HUD {
    static var cyan: Color { lightTheme ? Color(red: 0.0, green: 0.45, blue: 0.62) : Color(red: 0.0, green: 0.90, blue: 1.0) }
    static var phosphor: Color { lightTheme ? Color(red: 0.0, green: 0.56, blue: 0.45) : Color(red: 0.30, green: 0.95, blue: 0.76) }
    static var magenta: Color { lightTheme ? Color(red: 0.82, green: 0.08, blue: 0.40) : Color(red: 1.0, green: 0.18, blue: 0.53) }
    static var yellow: Color { lightTheme ? Color(red: 0.70, green: 0.56, blue: 0.0) : Color(red: 1.0, green: 0.94, blue: 0.0) }
    static var ink: Color { lightTheme ? Color(red: 0.97, green: 0.98, blue: 0.99) : Color(red: 0.02, green: 0.04, blue: 0.06) }
    static var panel: Color { lightTheme ? Color.white : Color(red: 0.04, green: 0.07, blue: 0.11) }
    static var text: Color { lightTheme ? Color(red: 0.05, green: 0.09, blue: 0.13) : Color(red: 0.86, green: 0.95, blue: 0.98) }
    static var dim: Color { lightTheme ? Color(red: 0.36, green: 0.43, blue: 0.48) : Color(red: 0.45, green: 0.58, blue: 0.64) }
    /// Window tint: the backdrop never shows at full strength, so loose text stays legible (0.32 … 0.60).
    static var windowShade: Double { 0.32 + (0.85 - Setup.transparency) * (0.28 / 0.70) }
    /// Card backing, denser than the window (0.22 … 0.62).
    static var cardShade: Double { 0.22 + (0.85 - Setup.transparency) * (0.40 / 0.70) + 0.2 }
    static var blue: Color { lightTheme ? Color(red: 0.10, green: 0.30, blue: 0.85) : Color(red: 0.20, green: 0.55, blue: 1.0) }
    static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

// MARK: - pure helpers (kept free of UI so they stay easy to reason about)

enum Sentences {
    /// Splits `buffer` into complete sentences and the unfinished rest. A sentence ends at . ! ? … or a newline
    /// that is followed by a space / the end; digits like "3.14" do not end one.
    static func split(_ buffer: String) -> (done: [String], rest: String) {
        var done: [String] = [], current = "", chars = Array(buffer), i = 0
        while i < chars.count {
            let c = chars[i]
            current.append(c)
            let terminal = ".!?…。！？".contains(c) || c == "\n"
            if terminal {
                let next: Character? = i + 1 < chars.count ? chars[i + 1] : nil
                let digitDot = c == "." && i > 0 && chars[i - 1].isNumber && (next?.isNumber ?? false)
                if !digitDot, next == nil || next!.isWhitespace || next! == "\"" || next! == ")" {
                    let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
                    if t.count > 1 { done.append(t) }
                    current = ""
                }
            }
            i += 1
        }
        return (done, current)
    }

    /// Markdown and symbols a voice should not read out.
    static func spoken(_ s: String) -> String {
        var t = s
        for junk in ["**", "__", "`", "#", "*"] { t = t.replacingOccurrences(of: junk, with: "") }
        t = t.replacingOccurrences(of: "\u{2022}", with: "")
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

enum VoiceState: Equatable {
    case idle, starting, listening, thinking, speaking
    case failed(String)
}

// MARK: - the conversation

@MainActor
final class Voice: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published var state: VoiceState = .idle
    @Published var level: Double = 0          // 0…1, what the orb moves with
    @Published var heard = ""
    @Published var reply = ""
    @Published var speakReplies: Bool {
        didSet { UserDefaults.standard.set(speakReplies, forKey: "speakReplies") }
    }
    @Published var localeID: String {
        didSet { UserDefaults.standard.set(localeID, forKey: "voiceLocale") }
    }

    private var engine: AVAudioEngine?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var lastChange = Date()
    private var silenceTimer: Timer?
    private var ask: Task<Void, Never>?
    private let synth = AVSpeechSynthesizer()
    private var pendingUtterances = 0
    private var streamDone = true
    private var history: [[String: String]] = []
    private var levelTarget: Double = 0

    override init() {
        speakReplies = UserDefaults.standard.object(forKey: "speakReplies") as? Bool ?? true
        localeID = UserDefaults.standard.string(forKey: "voiceLocale") ?? ""
        super.init()
        synth.delegate = self
        Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
    }

    private var recognizerLocale: Locale {
        Locale(identifier: localeID.isEmpty ? (Locale.preferredLanguages.first ?? "en-US") : localeID)
    }

    /// Eases `level` toward its target so the orb never jumps.
    private func tick() {
        level += (levelTarget - level) * 0.28
        if state == .speaking || state == .thinking || state == .idle { levelTarget *= 0.9 }
    }

    // ---- the button

    func toggle() {
        switch state {
        case .listening: finishListening()
        case .speaking, .thinking:
            ask?.cancel(); synth.stopSpeaking(at: .immediate); pendingUtterances = 0
            begin()
        case .starting: break
        default: begin()
        }
    }

    func begin() {
        heard = ""; reply = ""
        Task {
            guard await ensureModel() else { return }
            await startListening()
        }
    }

    func stopEverything() {
        silenceTimer?.invalidate()
        ask?.cancel()
        synth.stopSpeaking(at: .immediate)
        stopCapture()
    }

    // ---- the model

    private func engineAnswers() async -> Bool {
        var r = URLRequest(url: Setup.base.appendingPathComponent("health"))
        r.timeoutInterval = 1.5
        guard let (_, resp) = try? await URLSession.shared.data(for: r),
              (resp as? HTTPURLResponse)?.statusCode == 200 else { return false }
        return true
    }

    /// Starts the model with the `llm` command when nothing answers (it picks the selected model, or the first
    /// one that fits), then waits for it.
    private func ensureModel() async -> Bool {
        if await engineAnswers() { return true }
        state = .starting
        let root = Setup.llmRoot
        await Task.detached {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: root + "/llm")
            p.arguments = ["start"]
            p.currentDirectoryURL = URL(fileURLWithPath: root)
            p.standardInput = FileHandle.nullDevice
            p.standardOutput = FileHandle.nullDevice
            p.standardError = FileHandle.nullDevice
            try? p.run()
            p.waitUntilExit()
        }.value
        for _ in 0..<5 {
            if await engineAnswers() { state = .idle; return true }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        state = .failed("The model did not start. Open LLMBar and press Start, then try again.")
        return false
    }

    // ---- listening

    private func startListening() async {
        let speechOK: Bool = await withCheckedContinuation { c in
            SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0 == .authorized) }
        }
        guard speechOK else { state = .failed("Speech recognition is not allowed (System Settings → Privacy & Security)"); return }
        let micOK: Bool = await withCheckedContinuation { c in
            AVCaptureDevice.requestAccess(for: .audio) { c.resume(returning: $0) }
        }
        guard micOK else { state = .failed("Microphone access is not allowed (System Settings → Privacy & Security)"); return }
        guard let rec = SFSpeechRecognizer(locale: recognizerLocale), rec.isAvailable else {
            state = .failed("Speech recognition is not available for \(recognizerLocale.identifier)"); return
        }
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        if rec.supportsOnDeviceRecognition { req.requiresOnDeviceRecognition = true }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buf, _ in
            req.append(buf)
            guard let ch = buf.floatChannelData?[0] else { return }
            let n = Int(buf.frameLength)
            var sum: Float = 0
            for i in 0..<n { sum += ch[i] * ch[i] }
            let rms = Double((sum / Float(max(n, 1))).squareRoot())
            let v = min(1.0, rms * 9)                                   // speech is ~0.02–0.1 RMS
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.levelTarget = v } }
        }
        do { try engine.start() } catch { state = .failed("Cannot use the microphone: \(error.localizedDescription)"); return }
        self.engine = engine
        request = req
        state = .listening
        lastChange = Date()
        task = rec.recognitionTask(with: req) { [weak self] result, error in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self, self.state == .listening else { return }
                    if let r = result {
                        let text = r.bestTranscription.formattedString
                        if text != self.heard { self.heard = text; self.lastChange = Date() }
                        if r.isFinal { self.finishListening() }
                    } else if error != nil { self.finishListening() }
                }
            }
        }
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.state == .listening else { return }
                let quiet = Date().timeIntervalSince(self.lastChange)
                if !self.heard.isEmpty, quiet > 1.4 { self.finishListening() }
                else if self.heard.isEmpty, quiet > 9 { self.stopCapture(); self.state = .idle }
            }
        }
    }

    private func stopCapture() {
        silenceTimer?.invalidate()
        engine?.stop()
        engine?.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        engine = nil; request = nil; task = nil
        levelTarget = 0
    }

    private func finishListening() {
        guard state == .listening else { return }
        stopCapture()
        let text = heard.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { state = .idle; return }
        state = .thinking
        ask = Task { await answer(text) }
    }

    // ---- the answer

    private func answer(_ question: String) async {
        history.append(["role": "user", "content": question])
        history = Array(history.suffix(10))
        let system = "You are a voice assistant. Answer in the language the user spoke, in one to three short "
            + "sentences, plain spoken text: no markdown, no lists, no emoji, no code blocks."
        let body: [String: Any] = [
            "model": "local", "stream": true, "max_tokens": 400, "temperature": 0.6,
            "chat_template_kwargs": ["enable_thinking": false],
            "messages": [["role": "system", "content": system]] + history,
        ]
        var req = URLRequest(url: Setup.base.appendingPathComponent("v1/chat/completions"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !Setup.apiKey.isEmpty { req.setValue("Bearer " + Setup.apiKey, forHTTPHeaderField: "Authorization") }
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        req.timeoutInterval = 600
        streamDone = false
        var buffer = "", full = ""
        do {
            let (bytes, resp) = try await URLSession.shared.bytes(for: req)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                state = .failed("The model answered with an error."); streamDone = true; return
            }
            for try await line in bytes.lines {
                if Task.isCancelled { break }
                guard line.hasPrefix("data:"), !line.contains("[DONE]"),
                      let data = line.dropFirst(5).trimmingCharacters(in: .whitespaces).data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let choice = (obj["choices"] as? [[String: Any]])?.first,
                      let piece = (choice["delta"] as? [String: Any])?["content"] as? String else { continue }
                full += piece; buffer += piece
                reply = full
                levelTarget = 0.35
                let (done, rest) = Sentences.split(buffer)
                buffer = rest
                for s in done { say(s) }
            }
            let tail = Sentences.spoken(buffer)
            if !tail.isEmpty { say(tail) }
        } catch {
            if !Task.isCancelled { state = .failed("Cannot reach the model: \(error.localizedDescription)") }
            streamDone = true
            return
        }
        streamDone = true
        if !full.isEmpty { history.append(["role": "assistant", "content": full]) }
        if pendingUtterances == 0, state != .speaking { state = .idle }
    }

    private func say(_ sentence: String) {
        let text = Sentences.spoken(sentence)
        guard speakReplies, !text.isEmpty else { return }
        let u = AVSpeechUtterance(string: text)
        let lang = NLLanguageRecognizer.dominantLanguage(for: text)?.rawValue
        if let lang, let voice = AVSpeechSynthesisVoice(language: lang) ?? AVSpeechSynthesisVoice(language: String(lang.prefix(2))) {
            u.voice = voice
        }
        pendingUtterances += 1
        if state != .speaking { state = .speaking }
        synth.speak(u)
    }

    // ---- AVSpeechSynthesizerDelegate: the orb breathes with the words

    nonisolated func speechSynthesizer(_ s: AVSpeechSynthesizer, willSpeakRangeOfSpeechString r: NSRange,
                                       utterance: AVSpeechUtterance) {
        Task { @MainActor in self.levelTarget = Double.random(in: 0.45...0.95) }
    }

    nonisolated func speechSynthesizer(_ s: AVSpeechSynthesizer, didFinish u: AVSpeechUtterance) {
        Task { @MainActor in
            self.pendingUtterances = max(0, self.pendingUtterances - 1)
            if self.pendingUtterances == 0, self.streamDone { self.state = .idle; self.levelTarget = 0 }
        }
    }
}

// MARK: - the orb: a burst of energy — a white-hot core, dozens of thin curling filaments, a haze behind them

struct Orb: View {
    @ObservedObject var voice: Voice

    /// (filament, accent, haze) colours for the state.
    private var colors: (Color, Color, Color) {
        switch voice.state {
        case .listening: return (HUD.cyan, HUD.phosphor, HUD.blue)
        case .thinking, .starting: return (HUD.blue, HUD.yellow, HUD.cyan)
        case .speaking: return (HUD.phosphor, HUD.magenta, HUD.cyan)
        case .failed: return (HUD.magenta, HUD.magenta.opacity(0.7), HUD.magenta)
        case .idle: return (HUD.cyan.opacity(0.85), HUD.phosphor.opacity(0.8), HUD.blue)
        }
    }

    /// Smooth pseudo-noise in -1…1, deterministic per (filament, step, time).
    private func wobble(_ i: Int, _ k: Int, _ t: Double, _ speed: Double) -> CGFloat {
        let a = sin(t * speed * 1.3 + Double(i) * 1.7 + Double(k) * 0.9)
        let b = sin(t * speed * 0.7 + Double(i) * 2.9 + Double(k) * 1.4)
        return CGFloat(a * 0.6 + b * 0.4)
    }

    /// 0…1, fixed per index: lengths, curls and thicknesses differ from filament to filament.
    private func rnd(_ i: Int, _ salt: Double) -> Double {
        let v = sin(Double(i) * 12.9898 + salt * 78.233) * 43758.5453
        return v - v.rounded(.down)
    }

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { ctx, size in
                let t = timeline.date.timeIntervalSinceReferenceDate
                let lv = voice.level
                let busy = voice.state == .thinking || voice.state == .starting
                let idle = voice.state == .idle
                let (fil, accent, haze) = colors
                let c = CGPoint(x: size.width / 2, y: size.height / 2)
                let R = min(size.width, size.height) * 0.38
                let speed = busy ? 2.2 : (idle ? 0.6 : 1.0 + lv * 2.0)
                let count = idle ? 34 : (busy ? 54 : 46 + Int(lv * 44))
                let steps = 12
                let r0 = R * ((lightTheme ? 0.07 : 0.05) + 0.06 * CGFloat(lv) + (busy ? 0.015 * CGFloat(sin(t * 3)) : 0))

                // the haze: a few big soft blue clouds turning slowly
                ctx.drawLayer { layer in
                    layer.addFilter(.blur(radius: R * 0.32))
                    layer.blendMode = lightTheme ? .normal : .screen
                    for j in 0..<4 {
                        let a = t * (0.12 + 0.05 * Double(j)) + Double(j) * 1.6
                        let off = R * CGFloat(0.22 + 0.12 * lv)
                        let x = c.x + CGFloat(cos(a)) * off, y = c.y + CGFloat(sin(a * 1.2)) * off
                        let r = R * CGFloat(0.50 + 0.08 * sin(a * 2))
                        layer.fill(Path(ellipseIn: CGRect(x: x - r, y: y - r, width: r * 2, height: r * 2)),
                                   with: .color((j % 2 == 0 ? haze : fil).opacity((lightTheme ? 0.10 : 0.16) + 0.22 * lv)))
                    }
                }

                // all filaments in one path for the soft glow, each one again sharp
                var all = Path()
                var sharp: [(Path, Double, CGFloat, Bool)] = []
                for i in 0..<count {
                    let angle = Double(i) / Double(count) * 2 * .pi + Double(rnd(i, 1)) * 0.5 + t * 0.05 * (rnd(i, 2) - 0.5)
                    let length = R * CGFloat(0.45 + 0.55 * rnd(i, 3)) * CGFloat(0.8 + 0.35 * lv)
                    let curl = CGFloat(rnd(i, 4) - 0.5) * R * 0.5 * CGFloat(0.7 + 0.8 * lv)
                    var line = Path()
                    for k in 0...steps {
                        let f = CGFloat(k) / CGFloat(steps)
                        let dist = r0 + (length - r0) * f
                        let lateral = curl * f * f + wobble(i, k, t, speed) * R * 0.045 * f * CGFloat(0.5 + lv)
                        let ca = CGFloat(cos(angle)), sa = CGFloat(sin(angle))
                        let pt = CGPoint(x: c.x + ca * dist - sa * lateral, y: c.y + sa * dist + ca * lateral)
                        if k == 0 { line.move(to: pt) } else { line.addLine(to: pt) }
                    }
                    all.addPath(line)
                    let flicker = 0.5 + 0.5 * Double((wobble(i, 77, t, speed * 1.5) + 1) / 2)
                    sharp.append((line, flicker * (0.45 + 0.55 * rnd(i, 5)), CGFloat(0.5 + 0.9 * rnd(i, 6)), rnd(i, 7) > 0.8))
                }
                ctx.drawLayer { layer in
                    layer.addFilter(.blur(radius: 3.5 + 3 * CGFloat(lv)))
                    layer.blendMode = lightTheme ? .normal : .screen
                    layer.stroke(all, with: .color(fil.opacity(lightTheme ? 0.30 : 0.55)), lineWidth: 3)
                }
                ctx.drawLayer { layer in
                    layer.blendMode = lightTheme ? .normal : .screen
                    for (line, alpha, width, tinted) in sharp {
                        layer.stroke(line, with: .color((tinted ? accent : fil).opacity(lightTheme ? min(1, alpha + 0.2) : alpha)),
                                     lineWidth: lightTheme ? width * 1.1 : width)
                    }
                }

                // the white-hot core
                ctx.drawLayer { layer in
                    layer.blendMode = lightTheme ? .normal : .screen
                    var glow = layer
                    glow.addFilter(.blur(radius: r0 * 1.8))
                    glow.fill(Path(ellipseIn: CGRect(x: c.x - r0 * 3.2, y: c.y - r0 * 3.2, width: r0 * 6.4, height: r0 * 6.4)),
                              with: .color(fil.opacity((lightTheme ? 0.45 : 0.65) + 0.3 * lv)))
                    // dark page: white-hot; light page: a saturated dot with a white heart (white would vanish)
                    layer.fill(Path(ellipseIn: CGRect(x: c.x - r0 * 1.4, y: c.y - r0 * 1.4, width: r0 * 2.8, height: r0 * 2.8)),
                               with: .radialGradient(Gradient(colors: lightTheme
                                    ? [Color.white, fil, fil.opacity(0)]
                                    : [Color.white, Color.white.opacity(0.85), fil.opacity(0)]),
                                                     center: c, startRadius: 0, endRadius: r0 * 1.4))
                }
            }
        }
    }
}

// MARK: - the panel

struct VoiceView: View {
    @ObservedObject var voice: Voice

    private var label: (String, Color) {
        switch voice.state {
        case .idle: return ("READY · CLICK THE SPHERE", HUD.dim)
        case .starting: return ("LOADING THE MODEL…", HUD.yellow)
        case .listening: return ("LISTENING", HUD.cyan)
        case .thinking: return ("THINKING", HUD.yellow)
        case .speaking: return ("SPEAKING", HUD.phosphor)
        case .failed: return ("PROBLEM", HUD.magenta)
        }
    }

    private func card(_ tag: String, _ text: String, _ color: Color, size: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                RoundedRectangle(cornerRadius: 1).fill(color).frame(width: 2.5, height: 9)
                Text(tag).font(HUD.mono(9, .semibold)).tracking(1.6).foregroundColor(color)
            }
            Text(text).font(.system(size: size)).foregroundColor(HUD.text)
                .frame(maxWidth: .infinity, alignment: .leading).fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(HUD.panel.opacity(HUD.cardShade))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(LinearGradient(colors: [color.opacity(0.45), color.opacity(0.08)],
                                             startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1)))
    }

    var body: some View {
        let (stateText, stateColor) = label
        VStack(spacing: 8) {
            HStack(spacing: 7) {
                Circle().fill(stateColor).frame(width: 7, height: 7).shadow(color: stateColor.opacity(0.7), radius: 3)
                Text("LLM VOICE").font(HUD.mono(10, .semibold)).tracking(1.6).foregroundColor(HUD.dim)
                Spacer()
                Menu {
                    Toggle("Speak replies", isOn: $voice.speakReplies)
                    Picker("Recognition language", selection: $voice.localeID) {
                        Text("System language").tag("")
                        Text("English").tag("en-US")
                        Text("Русский").tag("ru-RU")
                        Text("Deutsch").tag("de-DE")
                        Text("Français").tag("fr-FR")
                        Text("Español").tag("es-ES")
                        Text("中文").tag("zh-CN")
                        Text("日本語").tag("ja-JP")
                    }
                } label: { Image(systemName: "ellipsis").foregroundColor(HUD.dim) }
                    .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
            }.padding(.horizontal, 16).padding(.top, 12)

            Orb(voice: voice)
                .frame(height: 250)
                .contentShape(Rectangle())
                .onTapGesture { voice.toggle() }

            Text(stateText).font(HUD.mono(10, .semibold)).tracking(1.8).foregroundColor(stateColor)

            VStack(spacing: 8) {
                if case .failed(let message) = voice.state {
                    card("WHAT HAPPENED", message, HUD.magenta, size: 12.5)
                } else {
                    if !voice.heard.isEmpty { card("YOU", voice.heard, HUD.cyan, size: 12.5) }
                    if !voice.reply.isEmpty {
                        ScrollView { card("REPLY", voice.reply, HUD.phosphor, size: 13.5) }.frame(maxHeight: 130)
                    }
                }
            }.padding(.horizontal, 16)
            Spacer(minLength: 0)
        }
        .frame(width: 340, height: 500)
        .background((lightTheme ? Color.white : HUD.ink).opacity(HUD.windowShade))
        .clipShape(RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous)
            .strokeBorder(LinearGradient(colors: [HUD.cyan.opacity(0.5), HUD.cyan.opacity(0.08)],
                                         startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1))
        .preferredColorScheme(lightTheme ? .light : .dark)
    }
}

// MARK: - app

final class VoicePanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override func cancelOperation(_ sender: Any?) { close() }                // Esc
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var panel: VoicePanel?
    private let voice = Voice()

    func applicationDidFinishLaunching(_ n: Notification) {
        resolveTheme()
        NSApp.setActivationPolicy(.accessory)                               // no Dock icon, like Siri's panel
        let host = NSHostingView(rootView: VoiceView(voice: voice))
        let p = VoicePanel(contentRect: NSRect(x: 0, y: 0, width: 340, height: 500),
                           styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
                           backing: .buffered, defer: false)
        host.wantsLayer = true
        host.layer?.cornerRadius = 26
        host.layer?.cornerCurve = .continuous
        host.layer?.masksToBounds = true
        host.layer?.backgroundColor = NSColor.clear.cgColor
        p.appearance = NSAppearance(named: lightTheme ? .aqua : .darkAqua)
        if #available(macOS 26.0, *) {
            let glass = NSGlassEffectView()          // the system's real glass, as in LLMBar
            glass.cornerRadius = 26
            glass.style = .clear
            glass.contentView = host
            p.contentView = glass
            p.hasShadow = false                      // a glass panel draws its own edge
        } else {
            let fx = NSVisualEffectView()
            fx.material = .hudWindow
            fx.blendingMode = .behindWindow
            fx.state = .active
            fx.wantsLayer = true
            fx.layer?.cornerRadius = 26
            fx.layer?.cornerCurve = .continuous
            fx.layer?.masksToBounds = true
            host.frame = fx.bounds
            host.autoresizingMask = [.width, .height]
            fx.addSubview(host)
            p.contentView = fx
            p.hasShadow = true
        }
        p.isOpaque = false
        p.backgroundColor = .clear
        p.level = .floating
        p.isMovableByWindowBackground = true
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        if let screen = NSScreen.main {
            let f = screen.visibleFrame
            p.setFrameOrigin(NSPoint(x: f.midX - 170, y: f.minY + f.height * 0.22))
        }
        p.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        panel = p
        voice.begin()                                                       // opening the app means "I want to talk"
        NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: p, queue: .main) { _ in
            MainActor.assumeIsolated { self.voice.stopEverything(); NSApp.terminate(nil) }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
}

// `LLMVoice --snapshot out.png [--state listening|thinking|speaking|idle|failed] [--level 0.6]`: renders the
// panel for a look at the sphere without starting the model or touching the microphone.
@MainActor
func renderSnapshot(_ args: [String]) {
    func value(_ flag: String) -> String? { args.firstIndex(of: flag).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
    guard let out = value("--snapshot") else { return }
    resolveTheme()
    let voice = Voice()
    switch value("--state") ?? "idle" {
    case "listening": voice.state = .listening; voice.heard = "What's the weather like in Lisbon in May?"
    case "thinking": voice.state = .thinking; voice.heard = "What's the weather like in Lisbon in May?"
    case "speaking":
        voice.state = .speaking; voice.heard = "What's the weather like in Lisbon in May?"
        voice.reply = "Lisbon in May is warm and mostly sunny, around 22 degrees, with only a little rain."
    case "failed": voice.state = .failed("The model did not start. Open LLMBar and press Start, then try again.")
    default: break
    }
    voice.level = Double(value("--level") ?? "0.5") ?? 0.5
    let host = NSHostingView(rootView: VoiceView(voice: voice))
    host.frame = NSRect(x: 0, y: 0, width: 340, height: 500)
    let win = NSWindow(contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    win.isOpaque = false
    win.backgroundColor = .clear
    win.contentView = host
    win.orderFrontRegardless()
    RunLoop.current.run(until: Date().addingTimeInterval(0.6))
    guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { exit(3) }
    host.cacheDisplay(in: host.bounds, to: rep)
    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: out))
    print("SNAPSHOT \(out)")
    exit(0)
}

let app = NSApplication.shared
private let delegate = MainActor.assumeIsolated { AppDelegate() }
if CommandLine.arguments.contains("--snapshot") {
    MainActor.assumeIsolated { renderSnapshot(CommandLine.arguments) }
}
app.delegate = delegate
app.run()

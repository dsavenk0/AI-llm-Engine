// The optional neural voice of LLM Voice (Qwen3-TTS through `llm speech serve`, engine/tts.py). Off by default —
// the system voice reads replies — and used only when chosen in LLMBar's settings. Its helper process runs only
// while the voice window is open: started with it, stopped with it, so the model shares memory with the language
// model just for as long as it is needed. Russian stays with the system voice, as does any sentence the helper
// cannot do (not ready yet, an error): the reply is never left silent.
import AVFoundation
import Foundation

@MainActor
final class NeuralVoice: NSObject, AVAudioPlayerDelegate {
    static let engineKey = "speechEngine"          // "system" (default) | "qwen3tts"
    static let speakerKey = "speechSpeaker"        // one of the model's voices; "aiden" by default
    nonisolated static let model = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"   // = llm.py SPEECH_MODEL
    static let speakers = ["aiden", "ryan", "eric", "dylan", "serena", "vivian", "ono_anna", "sohee", "uncle_fu"]

    static var chosen: Bool { UserDefaults.standard.string(forKey: engineKey) == "qwen3tts" }
    static var speaker: String { UserDefaults.standard.string(forKey: speakerKey) ?? "aiden" }

    /// The model's language name for a two-letter code, or nil: that language stays with the system voice.
    static func language(_ code: String) -> String? {
        ["en": "english", "de": "german", "fr": "french", "es": "spanish", "it": "italian", "pt": "portuguese",
         "ja": "japanese", "ko": "korean", "zh": "chinese", "ru": "russian"][String(code.prefix(2)).lowercased()]
    }

    var onFinished: (() -> Void)?                  // one sentence has been played (or skipped)
    var onFallback: ((String) -> Void)?            // a sentence it could not voice: the system voice reads it
    var onLevel: ((Double) -> Void)?

    private(set) var ready = false
    private var proc: Process?
    private var input: FileHandle?
    private var buffer = ""
    private var waiting: [(text: String, lang: String)] = []   // sentences not yet synthesized
    private var busy = false                                   // one synthesis at a time
    private var playlist: [URL] = []
    private var player: AVAudioPlayer?
    private var meter: Timer?
    private var generation = 0                                 // bumps on stop: late results are dropped
    private var count = 0
    private let dir = FileManager.default.temporaryDirectory.appendingPathComponent("llmbar-voice", isDirectory: true)

    /// Start the helper if the neural voice is chosen and not running yet (it loads the model in a few seconds).
    func start(root: String) {
        guard NeuralVoice.chosen, proc == nil else { return }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: root + "/llm")
        p.arguments = ["speech", "serve"]
        p.currentDirectoryURL = URL(fileURLWithPath: root)
        let inPipe = Pipe(), outPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = FileHandle.nullDevice
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            if data.isEmpty { h.readabilityHandler = nil; return }       // EOF: the helper is gone, stop polling
            let chunk = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self?.received(chunk) }
        }
        p.terminationHandler = { [weak self] ended in
            Task { @MainActor in self?.ended(ended) }
        }
        do { try p.run() } catch { return }
        proc = p
        input = inPipe.fileHandleForWriting
    }

    /// Stop the helper (the voice window closed, or the system voice was chosen again).
    func shutdown() {
        stop()
        if let input { try? input.write(contentsOf: Data("{\"quit\": true}\n".utf8)) }
        proc?.terminate()
        proc = nil
        input = nil
        ready = false
    }

    /// Whether this sentence should go to the neural voice now.
    func takes(_ lang: String?) -> Bool {
        guard NeuralVoice.chosen, ready, let lang else { return false }
        // a reply in Russian is the system voice's (it reads Russian better); a Russian sentence inside an
        // English reply still goes to the model, with its own language
        return !lang.lowercased().hasPrefix("ru") && NeuralVoice.language(lang) != nil
    }

    func speak(_ text: String, lang: String) {
        waiting.append((text, NeuralVoice.language(lang) ?? "auto"))
        pump()
    }

    /// Silence: drop what is waiting and stop what is playing (the helper keeps the model loaded).
    func stop() {
        generation += 1
        waiting.removeAll()
        playlist.removeAll()
        player?.stop()
        player = nil
        meter?.invalidate()
        // a request already sent stays "busy" until its reply arrives (then dropped as stale): sending the next one
        // earlier would let the old sentence's audio come back as if it were the new one's
        pendingText = nil
    }

    // ── the helper's replies ──
    private func received(_ chunk: String) {
        buffer += chunk
        while let nl = buffer.firstIndex(of: "\n") {
            let line = String(buffer[..<nl]); buffer.removeSubrange(...nl)
            guard let obj = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else { continue }
            if obj["ready"] as? Bool == true { ready = true; continue }
            busy = false
            if pendingGen != generation {           // asked before a stop: nobody is waiting for it any more
                if let path = obj["ok"] as? String { try? FileManager.default.removeItem(atPath: path) }
            } else if let path = obj["ok"] as? String {
                playlist.append(URL(fileURLWithPath: path))
                playNext()
            } else if pendingText != nil {
                onFinished?()                      // one sentence the model could not voice is skipped, never
            }                                      // read by a second voice over the first one
            pendingText = nil
            pump()
        }
    }

    private var pendingText: String?
    private var pendingGen = 0

    private func ended(_ which: Process) {
        guard which === proc else { return }       // a helper shut down earlier (or replaced) ends quietly
        proc = nil
        input = nil
        ready = false
        busy = false
        // the helper died mid-reply: what it still owed goes to the system voice, in order
        var left = waiting.map { $0.text }
        if let text = pendingText { left.insert(text, at: 0) }
        pendingText = nil
        waiting.removeAll()
        for t in left { onFallback?(t) }
    }

    private func pump() {
        guard !busy, let input, !waiting.isEmpty else { return }
        let next = waiting.removeFirst()
        count += 1
        let out = dir.appendingPathComponent("reply-\(count).wav").path
        let req: [String: Any] = ["say": next.text, "out": out, "speaker": NeuralVoice.speaker, "lang": next.lang]
        guard let data = try? JSONSerialization.data(withJSONObject: req) else { return }
        busy = true
        pendingText = next.text
        pendingGen = generation
        do { try input.write(contentsOf: data + Data("\n".utf8)) } catch { busy = false; onFallback?(next.text) }
    }

    // ── playback, one sentence after another ──
    private func playNext() {
        guard player == nil, !playlist.isEmpty else { return }
        let url = playlist.removeFirst()
        guard let p = try? AVAudioPlayer(contentsOf: url) else { onFinished?(); playNext(); return }
        p.delegate = self
        p.isMeteringEnabled = true
        player = p
        p.play()
        meter?.invalidate()
        meter = Timer.scheduledTimer(withTimeInterval: 1.0 / 20, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let pl = self?.player else { return }
                pl.updateMeters()
                self?.onLevel?(max(0, min(1, Double(pl.averagePower(forChannel: 0) + 50) / 50)))
            }
        }
    }

    nonisolated func audioPlayerDidFinishPlaying(_ p: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            if let url = p.url { try? FileManager.default.removeItem(at: url) }
            self.player = nil
            self.meter?.invalidate()
            self.onFinished?()
            self.playNext()
        }
    }
}

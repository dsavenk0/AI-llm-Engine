// Talks to the local Python backend (`python -m ui.server`): models, a tiny HTTP client, the process.
import Foundation

// MARK: - wire models

struct TreeEntry: Decodable, Identifiable, Hashable {
    let name: String, path: String, type: String, size: Int
    var id: String { path }
    var isDir: Bool { type == "dir" }
}
struct TreeResponse: Decodable { let entries: [TreeEntry] }
struct FilesResponse: Decodable { let files: [String] }

struct Symbol: Decodable, Identifiable, Hashable {
    let kind: String, name: String, line: Int, end: Int
    let children: [Symbol]
    var id: String { "\(kind)|\(name)|\(line)" }
}

struct Span: Decodable {            // arrives as [start, end, "kind"]
    let start: Int, end: Int, kind: String
    init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        start = try c.decode(Int.self); end = try c.decode(Int.self); kind = try c.decode(String.self)
    }
}

struct FileResponse: Decodable {
    let path: String
    let text: String?, language: String?
    let mtime: Int64?
    let outline: [Symbol]?, spans: [Span]?
    let binary: Bool?, too_large: Bool?, not_utf8: Bool?
}
struct AnalyzeResponse: Decodable {
    let language: String
    let spans: [Span]
    let outline: [Symbol]
    let diagnostics: [Diagnostic]?
}
struct DiagnosticsResponse: Decodable { let language: String; let diagnostics: [Diagnostic]; let deep: Bool }
struct SaveResponse: Decodable { let mtime: Int64 }

struct GitFile: Decodable, Identifiable, Hashable {
    let path: String, status: String
    var id: String { path }
}
struct GitStatus: Decodable {
    let repo: Bool
    let branch: String?, upstream: String?
    let ahead: Int?, behind: Int?
    let files: [GitFile]?, remotes: [String]?
}
struct CommitResponse: Decodable { let commit: String; let git: GitStatus }
struct PushResponse: Decodable { let output: String; let git: GitStatus }
struct DiffResponse: Decodable { let diff: String }
struct SearchHit: Decodable, Identifiable, Hashable {
    let path: String, line: String, text: String
    var id: String { "\(path):\(line)" }
}
struct SearchResponse: Decodable { let hits: [SearchHit] }
struct EngineInfo: Decodable { let model: String?, state: String?, context: Int?, vision: Bool?, vision_capable: Bool? }
struct GitHubStatus: Decodable {
    let installed: Bool, authenticated: Bool
    let user: String?, repo: String?, remote: String?
}
struct PullRequest: Decodable, Identifiable, Hashable {
    let number: Int, title: String, headRefName: String, url: String
    var id: Int { number }
}
struct PullRequestsResponse: Decodable { let prs: [PullRequest] }
struct URLResponseBody: Decodable { let url: String }
struct CreateRepoResponse: Decodable { let url: String; let git: GitStatus }
struct BranchesResponse: Decodable {
    let repo: Bool
    let current: String?
    let local: [String]
    let remote: [String]
}
struct CheckoutResponse: Decodable { let branches: BranchesResponse; let git: GitStatus }
struct InfoResponse: Decodable {
    let root: String, name: String, engine: EngineInfo?, git: GitStatus
    let remote: Bool?, endpoint: String?
}

struct BackendError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// Runs a block at most once, from any thread.
final class Once: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func run(_ body: () -> Void) {
        lock.lock(); defer { lock.unlock() }
        if !done { done = true; body() }
    }
}

// MARK: - client

final class Backend: @unchecked Sendable {
    private(set) var base: URL?
    private(set) var token = ""
    private var process: Process?

    static func pythonPath() -> String? {
        ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// Starts `python -m ui.server` for `root` and waits for the URL it prints.
    func start(root: String, llmRoot: String, enginePort: Int) async throws {
        stop()
        guard let py = Backend.pythonPath() else { throw BackendError(message: "python3 not found") }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: py)
        p.arguments = ["-B", "-m", "ui.server", "--root", root, "--engine-port", String(enginePort),
                       "--exit-with-parent", String(ProcessInfo.processInfo.processIdentifier)]
        p.currentDirectoryURL = URL(fileURLWithPath: llmRoot)
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        try p.run()
        process = p
        let once = Once()
        let found: (Int, String)? = await withCheckedContinuation { cont in
            pipe.fileHandleForReading.readabilityHandler = { h in
                let chunk = String(decoding: h.availableData, as: UTF8.self)
                if let r = parseBackendURL(chunk) {
                    once.run { cont.resume(returning: (r.port, r.token)) }
                } else if chunk.isEmpty {
                    once.run { cont.resume(returning: nil) }
                }
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + 15) {
                once.run { cont.resume(returning: nil) }
            }
        }
        // keep reading (and discarding): a pipe nobody empties fills at 64 KB and blocks the backend's stderr
        pipe.fileHandleForReading.readabilityHandler = { h in
            if h.availableData.isEmpty { h.readabilityHandler = nil }
        }
        guard let (port, tok) = found else { stop(); throw BackendError(message: "the editor backend did not start") }
        base = URL(string: "http://127.0.0.1:\(port)")
        token = tok
    }

    func stop() {
        process?.terminate()
        process = nil
        base = nil
    }

    // MARK: requests

    private func make(_ method: String, _ path: String, body: [String: Any]?) throws -> URLRequest {
        guard let base, let url = URL(string: path, relativeTo: base) else { throw BackendError(message: "backend is not running") }
        var r = URLRequest(url: url)
        r.httpMethod = method
        r.setValue(token, forHTTPHeaderField: "X-UI-Token")
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let body { r.httpBody = try JSONSerialization.data(withJSONObject: body) }
        r.timeoutInterval = 120
        return r
    }

    func request<T: Decodable>(_ method: String = "GET", _ path: String, body: [String: Any]? = nil) async throws -> T {
        let (data, resp) = try await URLSession.shared.data(for: make(method, path, body: body))
        guard let http = resp as? HTTPURLResponse else { throw BackendError(message: "no response") }
        if http.statusCode >= 400 {
            let msg = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["error"] as? String
            throw BackendError(message: msg ?? "HTTP \(http.statusCode)")
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    func get<T: Decodable>(_ path: String) async throws -> T {
        try await request("GET", path)
    }

    func post(_ path: String, body: [String: Any]) async throws {
        struct Ack: Decodable {}
        let _: Ack = try await request("POST", path, body: body)
    }

    func encoded(_ s: String) -> String { s.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? s }

    /// Streams the agent's events for one user turn. Cancelling the consuming task closes the connection,
    /// which the backend notices (and abandons any pending approval).
    func agent(messages: [[String: String]], attachments: [String], inline: [[String: String]] = [],
               images: [[String: String]] = [], think: Bool, autoEdit: Bool) -> AsyncThrowingStream<AgentEvent, Error> {
        AsyncThrowingStream { cont in
            let task = Task {
                do {
                    var r = try make("POST", "/api/agent", body: ["messages": messages, "attachments": attachments,
                                                                  "inline": inline, "images": images,
                                                                  "think": think, "auto_edit": autoEdit])
                    r.timeoutInterval = 3600
                    let (bytes, resp) = try await URLSession.shared.bytes(for: r)
                    if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
                        var data = Data()
                        for try await b in bytes { data.append(b) }
                        let msg = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["error"] as? String
                        throw BackendError(message: msg ?? "HTTP \(http.statusCode)")
                    }
                    for try await line in bytes.lines {
                        if let ev = AgentEvent.parse(line: line) { cont.yield(ev) }
                    }
                    cont.finish()
                } catch { cont.finish(throwing: error) }
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }
}

// Model source: this Mac's engine (default) or any OpenAI-compatible server — another computer running this
// engine, OpenRouter, a company gateway. The key lives in the Keychain, never in a file.
import AppKit
import Security
import SwiftUI

struct EndpointInfo: Decodable {
    let remote: Bool
    let url: String?
    let model: String?
    let has_key: Bool?
}

enum Keychain {
    private static let service = "local.llm-engine.editor"

    static func get(_ account: String) -> String? {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecAttrAccount as String: account, kSecReturnData as String: true,
                                kSecMatchLimit as String: kSecMatchLimitOne]
        var out: AnyObject?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }

    static func set(_ account: String, _ value: String) {
        let base: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                   kSecAttrAccount as String: account]
        SecItemDelete(base as CFDictionary)
        guard !value.isEmpty else { return }
        var item = base
        item[kSecValueData as String] = Data(value.utf8)
        SecItemAdd(item as CFDictionary, nil)
    }
}

extension Workspace {
    /// The last remote server the person chose (address and model in the defaults, key in the Keychain).
    static var savedRemote: (url: String, model: String, key: String)? {
        guard UserDefaults.standard.bool(forKey: "useRemoteEndpoint"),
              let url = UserDefaults.standard.string(forKey: "remoteEndpointURL"), !url.isEmpty else { return nil }
        return (url, UserDefaults.standard.string(forKey: "remoteEndpointModel") ?? "", Keychain.get(url) ?? "")
    }

    /// Called once the backend is up: re-apply a remembered remote server (the backend forgets it on restart).
    func restoreEndpoint() async {
        guard let r = Workspace.savedRemote else { return }
        let _: EndpointInfo? = try? await backend.request("PUT", "/api/endpoint", body: [
            "url": r.url, "key": r.key, "model": r.model, "confirm": true])
        await refreshEngine()
    }

    func refreshEngine() async {
        if let info: InfoResponse = try? await backend.get("/api/info") {
            engine = info.engine
            git = info.git
            remoteURL = info.endpoint
        }
    }

    func useLocalEngine() {
        UserDefaults.standard.set(false, forKey: "useRemoteEndpoint")
        guarded("Model source") {
            let _: EndpointInfo = try await self.backend.request("PUT", "/api/endpoint", body: ["reset": true])
            await self.refreshEngine()
            self.say("Using the engine on this Mac")
        }
    }

    func useRemote(url: String, key: String, model: String) {
        guarded("Model source") {
            let _: EndpointInfo = try await self.backend.request("PUT", "/api/endpoint", body: [
                "url": url, "key": key, "model": model, "confirm": true])
            UserDefaults.standard.set(true, forKey: "useRemoteEndpoint")
            UserDefaults.standard.set(url, forKey: "remoteEndpointURL")
            UserDefaults.standard.set(model, forKey: "remoteEndpointModel")
            Keychain.set(url, key)
            await self.refreshEngine()
            self.say("Using \(url)")
        }
    }
}

struct RemoteSheet: View {
    @ObservedObject var ws: Workspace
    @Binding var shown: Bool
    @State private var url = Workspace.savedRemote?.url ?? ""
    @State private var model = Workspace.savedRemote?.model ?? ""
    @State private var key = Workspace.savedRemote?.key ?? ""

    private let presets: [(String, String, String)] = [
        ("OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
        ("Another computer with this engine", "https://your-host:8090/v1", "local"),
    ]

    private var valid: Bool {
        guard let u = URL(string: url), let s = u.scheme, ["http", "https"].contains(s), u.host != nil else { return false }
        return true
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Model source").font(.headline)
            Text("Point the assistant at any OpenAI-compatible server. While it is set, your messages, the files you attach and the code the assistant reads are sent to that server.")
                .font(.callout).foregroundColor(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack {
                ForEach(presets, id: \.0) { p in
                    Button(p.0) { url = p.1; model = p.2 }.buttonStyle(.bordered)
                }
            }
            TextField("Address, ending in /v1", text: $url).textFieldStyle(.roundedBorder)
            TextField("Model name (empty = the server's default)", text: $model).textFieldStyle(.roundedBorder)
            SecureField("API key", text: $key).textFieldStyle(.roundedBorder)
            Text("The key is kept in your Keychain. Over plain http it is only sent to this Mac.")
                .font(.caption).foregroundColor(.secondary)
            HStack {
                Button("Use this Mac's engine") { ws.useLocalEngine(); shown = false }
                Spacer()
                Button("Cancel") { shown = false }.keyboardShortcut(.cancelAction)
                Button("Use this server") {
                    ws.useRemote(url: url.trimmingCharacters(in: .whitespaces), key: key,
                                 model: model.trimmingCharacters(in: .whitespaces))
                    shown = false
                }.keyboardShortcut(.defaultAction).disabled(!valid)
            }
        }.padding(20).frame(width: 520)
    }
}

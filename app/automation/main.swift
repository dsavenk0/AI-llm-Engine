// LLMBar Automation — a deliberately small, local-only macOS Accessibility bridge.
// It accepts one JSON request per Unix-socket connection from assistant/service.py.  The browser extension
// never reaches this executable: it owns no socket path or secret and only talks to the authenticated service.

import AppKit
import ApplicationServices
import Foundation

let allowedBundles: Set<String> = [
    "com.apple.Safari", "com.google.Chrome", "com.google.Chrome.beta", "org.chromium.Chromium",
    "com.microsoft.edgemac", "com.brave.Browser", "company.thebrowser.Browser", "com.vivaldi.Vivaldi",
    "org.mozilla.firefox"
]
let chromiumBundles = allowedBundles.subtracting(["com.apple.Safari", "org.mozilla.firefox"])
let allowedKeys: Set<String> = ["Enter", "Tab", "Escape", "ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight", "PageDown", "PageUp", "Home", "End", "Backspace", "Space"]

struct Request: Decodable {
    let secret: String
    let action: String
    let bundle: String
    let x: Double?
    let y: Double?
    let value: String?
    let key: String?
}

func writeAll(_ fd: Int32, _ data: Data) {
    data.withUnsafeBytes { raw in
        guard let base = raw.baseAddress else { return }
        var offset = 0
        while offset < data.count {
            let n = Darwin.write(fd, base.advanced(by: offset), data.count - offset)
            if n <= 0 { return }
            offset += n
        }
    }
}

func reply(_ fd: Int32, _ body: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: body), data.count < 32_768 else { return }
    writeAll(fd, data + Data([0x0a]))
}

func axValue<T>(_ element: AXUIElement, _ attribute: CFString, _ type: T.Type) -> T? {
    var raw: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, attribute, &raw) == .success else { return nil }
    return raw as? T
}

func point(_ element: AXUIElement, _ attribute: CFString) -> CGPoint? {
    guard let value: AXValue = axValue(element, attribute, AXValue.self) else { return nil }
    var p = CGPoint.zero
    return AXValueGetValue(value, .cgPoint, &p) ? p : nil
}

func size(_ element: AXUIElement, _ attribute: CFString) -> CGSize? {
    guard let value: AXValue = axValue(element, attribute, AXValue.self) else { return nil }
    var s = CGSize.zero
    return AXValueGetValue(value, .cgSize, &s) ? s : nil
}

func webArea(in root: AXUIElement, depth: Int = 0) -> AXUIElement? {
    guard depth < 10 else { return nil }
    if (axValue(root, kAXRoleAttribute as CFString, String.self) ?? "") == "AXWebArea",
       let s = size(root, kAXSizeAttribute as CFString), s.width > 40, s.height > 40 { return root }
    guard let children: [AXUIElement] = axValue(root, kAXChildrenAttribute as CFString, [AXUIElement].self) else { return nil }
    var best: (AXUIElement, CGFloat)?
    for child in children {
        if let found = webArea(in: child, depth: depth + 1), let s = size(found, kAXSizeAttribute as CFString) {
            let area = s.width * s.height
            if best == nil || area > best!.1 { best = (found, area) }
        }
    }
    return best?.0
}

func matchesBrowser(_ requested: String, _ actual: String) -> Bool {
    requested == actual || (requested == "chromium" && chromiumBundles.contains(actual))
}

func focusedWebArea(bundle: String) -> (CGPoint, CGSize)? {
    guard let app = NSWorkspace.shared.frontmostApplication, let actual = app.bundleIdentifier,
          matchesBrowser(bundle, actual) else { return nil }
    let axApp = AXUIElementCreateApplication(app.processIdentifier)
    let focused: AXUIElement = axValue(axApp, kAXFocusedWindowAttribute as CFString, AXUIElement.self) ?? axApp
    guard let area = webArea(in: focused), let p = point(area, kAXPositionAttribute as CFString),
          let s = size(area, kAXSizeAttribute as CFString), s.width > 40, s.height > 40 else { return nil }
    return (p, s)
}

func postMouse(_ at: CGPoint) -> Bool {
    guard let source = CGEventSource(stateID: .hidSystemState),
          let move = CGEvent(mouseEventSource: source, mouseType: .mouseMoved, mouseCursorPosition: at, mouseButton: .left),
          let down = CGEvent(mouseEventSource: source, mouseType: .leftMouseDown, mouseCursorPosition: at, mouseButton: .left),
          let up = CGEvent(mouseEventSource: source, mouseType: .leftMouseUp, mouseCursorPosition: at, mouseButton: .left) else { return false }
    move.post(tap: .cghidEventTap); usleep(80_000); down.post(tap: .cghidEventTap); usleep(45_000); up.post(tap: .cghidEventTap)
    return true
}

func postText(_ text: String) -> Bool {
    guard text.count <= 2000, let source = CGEventSource(stateID: .hidSystemState),
          let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true),
          let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false) else { return false }
    down.keyboardSetUnicodeString(stringLength: text.utf16.count, unicodeString: Array(text.utf16))
    up.keyboardSetUnicodeString(stringLength: text.utf16.count, unicodeString: Array(text.utf16))
    down.post(tap: .cghidEventTap); up.post(tap: .cghidEventTap); return true
}

let keyCodes: [String: CGKeyCode] = ["Enter": 36, "Tab": 48, "Escape": 53, "Space": 49, "Backspace": 51,
    "ArrowLeft": 123, "ArrowRight": 124, "ArrowDown": 125, "ArrowUp": 126, "PageUp": 116, "PageDown": 121,
    "Home": 115, "End": 119]
func postKey(_ key: String) -> Bool {
    guard allowedKeys.contains(key), let code = keyCodes[key], let source = CGEventSource(stateID: .hidSystemState),
          let down = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: true),
          let up = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: false) else { return false }
    down.post(tap: .cghidEventTap); up.post(tap: .cghidEventTap); return true
}

func handle(_ req: Request, secret: String) -> [String: Any] {
    guard req.secret == secret else { return ["ok": false, "error": "unauthorized"] }
    guard allowedBundles.contains(req.bundle) || req.bundle == "chromium" else {
        return ["ok": false, "error": "unsupported_browser"]
    }
    let prompt = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
    guard AXIsProcessTrustedWithOptions(prompt) else { return ["ok": false, "error": "accessibility_required"] }
    guard let (origin, size) = focusedWebArea(bundle: req.bundle) else { return ["ok": false, "error": "browser_not_active"] }
    switch req.action {
    case "click":
        guard let x = req.x, let y = req.y, (0...1).contains(x), (0...1).contains(y) else { return ["ok": false, "error": "bad_coordinates"] }
        let target = CGPoint(x: origin.x + CGFloat(x) * size.width, y: origin.y + CGFloat(y) * size.height)
        return ["ok": postMouse(target), "route": "macos"]
    case "type":
        guard let value = req.value, !value.isEmpty else { return ["ok": false, "error": "bad_text"] }
        return ["ok": postText(value), "route": "macos"]
    case "press":
        guard let key = req.key else { return ["ok": false, "error": "bad_key"] }
        return ["ok": postKey(key), "route": "macos"]
    default: return ["ok": false, "error": "bad_action"]
    }
}

func serve(socketPath: String, secret: String) {
    unlink(socketPath)
    let server = socket(AF_UNIX, SOCK_STREAM, 0)
    guard server >= 0 else { exit(2) }
    var address = sockaddr_un(); address.sun_family = sa_family_t(AF_UNIX)
    let pathCapacity = MemoryLayout.size(ofValue: address.sun_path)
    guard socketPath.utf8.count + 1 <= pathCapacity else { exit(3) }
    socketPath.withCString { src in withUnsafeMutablePointer(to: &address.sun_path) { dest in
        UnsafeMutableRawPointer(dest).assumingMemoryBound(to: CChar.self).update(from: src, count: strlen(src) + 1)
    }}
    let length = socklen_t(MemoryLayout<sa_family_t>.size + socketPath.utf8.count + 1)
    guard withUnsafePointer(to: &address, { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(server, $0, length) } }) == 0 else { exit(3) }
    chmod(socketPath, 0o600); listen(server, 8)
    while true {
        let client = accept(server, nil, nil); if client < 0 { continue }
        // 2,000 Unicode characters may occupy more than 8 KiB once UTF-8 and JSON escaping are included.
        var bytes = [UInt8](), chunk = [UInt8](repeating: 0, count: 4096)
        while bytes.count <= 32_768 {
            let n = read(client, &chunk, chunk.count)
            if n <= 0 { break }
            bytes.append(contentsOf: chunk.prefix(n))
            if let newline = bytes.firstIndex(of: 0x0a) {
                bytes.removeSubrange(newline...)
                break
            }
        }
        if !bytes.isEmpty, bytes.count <= 32_768,
           let request = try? JSONDecoder().decode(Request.self, from: Data(bytes)) {
            reply(client, handle(request, secret: secret))
        }
        close(client)
    }
}

let args = CommandLine.arguments
if args.count == 2, args[1] == "--trusted" { exit(AXIsProcessTrusted() ? 0 : 1) }
guard args.count == 4, args[1] == "--serve" else { exit(64) }
serve(socketPath: args[2], secret: args[3])

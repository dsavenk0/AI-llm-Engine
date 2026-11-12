// The native half of the Safari extension that ships inside LLMBar.app (Contents/PlugIns/LLMBar Assistant.appex).
// Safari loads the web extension from this bundle's Resources; this handler answers the one native message the
// extension sends — "token" — with the pairing token of the local assistant service, so the extension connects
// on its own instead of asking the person to paste it, and with the version of the extension this copy of the app
// ships: an extension Safari still runs from before an update can then say so. It reads one file, nothing else.

import Foundation
import SafariServices

@objc(SafariWebExtensionHandler)
final class SafariWebExtensionHandler: NSObject, NSExtensionRequestHandling {
    /// The token file the service writes (assistant/service.py TOKEN_FILE). The sandbox lets this extension read
    /// only that folder (see its entitlements in build-app.sh).
    private static var tokenFile: String {
        let home = (Bundle.main.object(forInfoDictionaryKey: "LLMAssistantHome") as? String)
            ?? (NSHomeDirectory() as NSString).appendingPathComponent("Library/Application Support/AI-llm-Engine")
        return (home as NSString).appendingPathComponent("browser-assistant.json")
    }

    func beginRequest(with context: NSExtensionContext) {
        let item = context.inputItems.first as? NSExtensionItem
        let message = item?.userInfo?[SFExtensionMessageKey] as? [String: Any]
        // the version on disk now — after an LLMBar update, newer than the extension Safari may still be running
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? ""
        var reply: [String: Any] = ["ok": false, "version": version]
        if (message?["kind"] as? String) == "token" {
            if let data = FileManager.default.contents(atPath: Self.tokenFile),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let token = obj["token"] as? String, token.count >= 24 {
                reply = ["ok": true, "token": token, "version": version]
            } else {
                reply = ["ok": false, "version": version, "error": "the assistant service has not been set up yet"]
            }
        }
        let response = NSExtensionItem()
        response.userInfo = [SFExtensionMessageKey: reply]
        context.completeRequest(returningItems: [response], completionHandler: nil)
    }
}

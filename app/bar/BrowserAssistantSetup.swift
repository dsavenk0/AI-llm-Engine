// Tools ▸ Browser Assistant: install what an app may install, then hand the person straight to the one switch
// only they may flip. The Safari extension ships inside LLMBar.app (Contents/PlugIns/LLMBar Assistant.appex),
// so Safari lists it; Apple lets an app open Safari ▸ Settings ▸ Extensions on it, never tick it.

import AppKit
import SafariServices

enum BrowserAssistantSetup {
    static let extensionID = "local.llm-engine.llmbar.safari"

    static func run() {
        // the service (a LaunchAgent, so the extension works with LLMBar closed), the token, the files —
        // quietly, in the background; it prints steps for a terminal that nobody reads here
        CLI.run(["assistant", "setup"]) { code, out in
            DispatchQueue.main.async {
                guard code == 0 else {
                    alert(L("The browser assistant could not be set up", "Не удалось настроить браузерный ассистент"),
                          String(out.suffix(600)))
                    return
                }
                showInSafari()
            }
        }
    }

    private static func showInSafari() {
        SFSafariExtensionManager.getStateOfSafariExtension(withIdentifier: extensionID) { state, error in
            DispatchQueue.main.async {
                if let state, state.isEnabled {
                    openSafari()                         // already on: just bring Safari forward
                    return
                }
                if state != nil {
                    // Safari knows the extension: open its page in Safari's settings, where the person ticks it
                    SFSafariApplication.showPreferencesForExtension(withIdentifier: extensionID) { err in
                        if err != nil { DispatchQueue.main.async { unsigned() } }
                    }
                    return
                }
                unsigned()                               // Safari does not list it yet
            }
        }
    }

    /// Safari lists an extension that is not signed by a registered developer only while unsigned extensions are
    /// allowed — a switch that only the person can turn on, and that Safari turns off again when it quits.
    private static func unsigned() {
        openSafari()
        alert(L("One switch in Safari first", "Сначала один переключатель в Safari"),
              L("""
                This build of LLMBar is not signed by an Apple developer, so Safari shows its extension only after:
                1. Safari ▸ Settings ▸ Advanced: turn on “Show features for web developers”.
                2. Menu Develop ▸ “Allow Unsigned Extensions” (Safari asks for your password).
                Then press Browser Assistant again: Safari opens on the extension, tick “LLMBar Assistant”.
                Safari forgets step 2 when it quits. Signing the app with a free Apple Development certificate \
                (sign in to Xcode with your Apple ID) makes the extension stay enabled.
                """, """
                Эта сборка LLMBar не подписана разработчиком Apple, поэтому Safari покажет её расширение только после:
                1. Safari ▸ Настройки ▸ Дополнения: включите «Показывать функции для веб-разработчиков».
                2. Меню «Разработка» ▸ «Разрешить неподписанные расширения» (Safari спросит пароль).
                Затем снова нажмите «Браузерный ассистент»: Safari откроется на расширении — отметьте «LLMBar Assistant».
                Шаг 2 Safari забывает при выходе. Если подписать приложение бесплатным сертификатом Apple Development \
                (войти в Xcode своим Apple ID), расширение останется включённым.
                """))
    }

    private static func openSafari() {
        if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Safari") {
            NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration())
        }
    }

    private static func alert(_ title: String, _ text: String) {
        NSApp.activate(ignoringOtherApps: true)
        let a = NSAlert()
        a.messageText = title
        a.informativeText = text
        a.addButton(withTitle: "OK")
        a.runModal()
    }
}

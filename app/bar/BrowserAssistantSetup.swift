// Tools ▸ Browser Assistant: install what an app may install, then hand the person the remaining browser step.
// The Safari extension ships inside LLMBar.app; Chromium and Firefox load their prepared extension from
// Application Support.

import AppKit
import SafariServices

enum BrowserAssistantSetup {
    static let extensionID = "local.llm-engine.llmbar.safari"

    /// The build of the extension this copy of LLMBar ships ("0.1.0 (c51b127)"), for the Tools row.
    static var bundledVersion: String? {
        let url = Bundle.main.bundleURL.appendingPathComponent("Contents/PlugIns/LLMBar Assistant.appex")
        guard let info = Bundle(url: url)?.infoDictionary else { return nil }
        return (info["LLMExtensionBuild"] as? String) ?? (info["CFBundleVersion"] as? String)
    }

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
                Then press Browser Assistant · Safari again: Safari opens on the extension, tick “LLMBar Assistant”.
                Safari forgets step 2 when it quits. Signing the app with a free Apple Development certificate \
                (sign in to Xcode with your Apple ID) makes the extension stay enabled.
                """, """
                Эта сборка LLMBar не подписана разработчиком Apple, поэтому Safari покажет её расширение только после:
                1. Safari ▸ Настройки ▸ Дополнения: включите «Показывать функции для веб-разработчиков».
                2. Меню «Разработка» ▸ «Разрешить неподписанные расширения» (Safari спросит пароль).
                Затем снова нажмите «Браузерный ассистент · Safari»: Safari откроется на расширении — отметьте «LLMBar Assistant».
                Шаг 2 Safari забывает при выходе. Если подписать приложение бесплатным сертификатом Apple Development \
                (войти в Xcode своим Apple ID), расширение останется включённым.
                """))
    }

    // ───────────── Chrome and the other Chromium browsers ─────────────

    /// The first Chromium browser on this Mac, as (name, bundle id, its extensions page).
    static var chromium: (name: String, id: String, page: String)? {
        let known = [("Chrome", "com.google.Chrome", "chrome://extensions"),
                     ("Edge", "com.microsoft.edgemac", "edge://extensions"),
                     ("Brave", "com.brave.Browser", "brave://extensions"),
                     ("Arc", "company.thebrowser.Browser", "arc://extensions"),
                     ("Vivaldi", "com.vivaldi.Vivaldi", "vivaldi://extensions")]
        return known.first { NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0.1) != nil }
    }

    /// Firefox's temporary add-on flow is handled by about:debugging; Firefox does not accept an app-installed
    /// unpacked extension, so the person loads the prepared manifest once per browser session.
    static var firefox: (name: String, id: String, page: String)? {
        let id = "org.mozilla.firefox"
        guard NSWorkspace.shared.urlForApplication(withBundleIdentifier: id) != nil else { return nil }
        return ("Firefox", id, "about:debugging#/runtime/this-firefox")
    }

    /// The extension's version in the folder a Chromium browser loads (Application Support), if it is there.
    static var chromiumVersion: String? {
        let manifest = (NSHomeDirectory() as NSString)
            .appendingPathComponent("Library/Application Support/AI-llm-Engine/browser-assistant/chromium/manifest.json")
        guard let d = FileManager.default.contents(atPath: manifest),
              let m = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return nil }
        return (m["version_name"] as? String) ?? (m["version"] as? String)
    }

    static var firefoxVersion: String? {
        let manifest = (NSHomeDirectory() as NSString)
            .appendingPathComponent("Library/Application Support/AI-llm-Engine/browser-assistant/firefox/manifest.json")
        guard let d = FileManager.default.contents(atPath: manifest),
              let m = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return nil }
        return (m["version_name"] as? String) ?? (m["version"] as? String)
    }

    /// Chrome lets no app install an extension that is not from its store, so: everything around that one step —
    /// the files, the service, the pairing host — then the browser opened on its extensions page, the folder shown
    /// in Finder and its path on the clipboard for "Load unpacked".
    static func runChromium() {
        guard let browser = chromium else { return }
        CLI.run(["assistant", "setup"]) { code, out in
            DispatchQueue.main.async {
                guard code == 0 else {
                    alert(L("The browser assistant could not be set up", "Не удалось настроить браузерный ассистент"),
                          String(out.suffix(600)))
                    return
                }
                let folder = (NSHomeDirectory() as NSString)
                    .appendingPathComponent("Library/Application Support/AI-llm-Engine/browser-assistant/chromium")
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(folder, forType: .string)
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: folder)])
                // an app may not open a browser's internal pages by link; asking the browser itself usually works
                let script = "tell application id \"\(browser.id)\" to open location \"\(browser.page)\""
                var err: NSDictionary?
                NSAppleScript(source: script)?.executeAndReturnError(&err)
                if err != nil, let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: browser.id) {
                    NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration())
                }
                alert(L("Two clicks in \(browser.name)", "Два нажатия в \(browser.name)"),
                      L("""
                        \(browser.name) installs an extension that is not from its store only by your hand:
                        1. On \(browser.page): turn on “Developer mode” (top right).
                        2. “Load unpacked” → press ⌘⇧G, paste (the folder's path is on the clipboard), Open.
                        Then click the extension's toolbar button: the side panel opens and connects by itself.
                        Already loaded? Press its reload ↻ there to pick up this version.
                        """, """
                        \(browser.name) ставит расширение не из своего магазина только вашими руками:
                        1. На странице \(browser.page): включите «Режим разработчика» (справа вверху).
                        2. «Загрузить распакованное» → ⌘⇧G, вставьте (путь к папке уже в буфере обмена), «Открыть».
                        Затем нажмите кнопку расширения на панели: откроется боковая панель и подключится сама.
                        Уже загружено? Нажмите там ↻ у расширения, чтобы подхватить эту версию.
                        """))
            }
        }
    }

    /// Firefox requires the user to load a temporary add-on. LLMBar installs its native host and prepares the
    /// extension first, then opens Firefox's Add-ons Debugging page with the manifest path on the clipboard.
    static func runFirefox() {
        guard let browser = firefox else { return }
        CLI.run(["assistant", "setup", "--firefox"]) { code, out in
            DispatchQueue.main.async {
                guard code == 0 else {
                    alert(L("The browser assistant could not be set up", "Не удалось настроить браузерный ассистент"),
                          String(out.suffix(600)))
                    return
                }
                let manifest = (NSHomeDirectory() as NSString)
                    .appendingPathComponent("Library/Application Support/AI-llm-Engine/browser-assistant/firefox/manifest.json")
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(manifest, forType: .string)
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: manifest)])
                let script = "tell application id \"\(browser.id)\" to open location \"\(browser.page)\""
                var err: NSDictionary?
                NSAppleScript(source: script)?.executeAndReturnError(&err)
                if err != nil, let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: browser.id) {
                    NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration())
                }
                alert(L("Load the Firefox extension", "Загрузите расширение в Firefox"),
                      L("In about:debugging, choose “This Firefox” → “Load Temporary Add-on…”. Press ⌘⇧G and paste the manifest path from the clipboard. Firefox removes temporary add-ons when it quits; repeat this step after restarting Firefox.",
                        "На странице about:debugging выберите «Этот Firefox» → «Загрузить временное дополнение…». Нажмите ⌘⇧G и вставьте путь к манифесту из буфера обмена. После выхода Firefox удаляет временные дополнения — загрузите его снова после перезапуска."))
            }
        }
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

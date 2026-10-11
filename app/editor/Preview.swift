// Preview: a small browser pane for the page you are editing (an .html file, or a dev server on localhost).
// An optional module — the web view is created only when the tab is first shown.
import AppKit
import SwiftUI
import WebKit

@MainActor
final class PreviewModel: ObservableObject {
    @Published var address = ""
    @Published var title = ""
    @Published var canBack = false
    @Published var canForward = false
    @Published var loading = false
    weak var web: WKWebView?
    private(set) var projectRoot = ""

    func configure(root: String) { projectRoot = root }

    /// A project file is loaded from disk (with read access to the whole project, so relative CSS/JS/images
    /// resolve); anything else is treated as a web address.
    func load(_ text: String) {
        guard let web else { return }
        let t = text.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return }
        let local = t.hasPrefix("/") ? t : projectRoot + "/" + t
        if !t.contains("://"), FileManager.default.fileExists(atPath: local) {
            let url = URL(fileURLWithPath: local)
            web.loadFileURL(url, allowingReadAccessTo: URL(fileURLWithPath: projectRoot.isEmpty ? url.deletingLastPathComponent().path : projectRoot))
            address = url.path
            return
        }
        var s = t
        if !s.contains("://") { s = (s.hasPrefix("localhost") || s.hasPrefix("127.") ? "http://" : "https://") + s }
        guard let url = URL(string: s) else { return }
        web.load(URLRequest(url: url))
        address = s
    }

    func reload() { web?.reload() }
    func back() { web?.goBack() }
    func forward() { web?.goForward() }
    func openExternally() {
        guard let u = web?.url else { return }
        NSWorkspace.shared.open(u)
    }
}

struct WebSurface: NSViewRepresentable {
    @ObservedObject var model: PreviewModel

    func makeCoordinator() -> Coordinator { Coordinator(model) }

    func makeNSView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        cfg.websiteDataStore = .nonPersistent()               // a preview keeps no cookies or history of its own
        let web = WKWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = context.coordinator
        web.allowsBackForwardNavigationGestures = true
        model.web = web
        return web
    }

    func updateNSView(_ web: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKNavigationDelegate {
        let model: PreviewModel
        init(_ m: PreviewModel) { model = m }

        func webView(_ w: WKWebView, didStartProvisionalNavigation n: WKNavigation!) {
            Task { @MainActor in self.model.loading = true }
        }
        func webView(_ w: WKWebView, didFinish n: WKNavigation!) {
            Task { @MainActor in
                self.model.loading = false
                self.model.title = w.title ?? ""
                self.model.canBack = w.canGoBack
                self.model.canForward = w.canGoForward
                if let u = w.url { self.model.address = u.isFileURL ? u.path : u.absoluteString }
            }
        }
        func webView(_ w: WKWebView, didFail n: WKNavigation!, withError e: Error) {
            Task { @MainActor in self.model.loading = false }
        }
        func webView(_ w: WKWebView, didFailProvisionalNavigation n: WKNavigation!, withError e: Error) {
            Task { @MainActor in
                self.model.loading = false
                let page = "<html><body style=\"font:15px -apple-system;padding:2em;color:#888\"><h3>Cannot open the page</h3>"
                    + "<p>\(e.localizedDescription)</p><p>Is the dev server running?</p></body></html>"
                w.loadHTMLString(page, baseURL: nil)
            }
        }
    }
}

struct PreviewPanel: View {
    @ObservedObject var ws: Workspace
    @StateObject private var model = PreviewModel()
    @State private var field = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Button { model.back() } label: { Image(systemName: "chevron.left") }.disabled(!model.canBack).buttonStyle(.plain)
                Button { model.forward() } label: { Image(systemName: "chevron.right") }.disabled(!model.canForward).buttonStyle(.plain)
                Button { model.reload() } label: { Image(systemName: "arrow.clockwise") }.buttonStyle(.plain)
                TextField("File in the project, or http://localhost:3000", text: $field)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { model.load(field) }
                Button("Current file") {
                    if let p = ws.activePath { field = p; model.load(p) }
                }.disabled(ws.activePath == nil).help("Show the file that is open in the editor")
                Button { model.openExternally() } label: { Image(systemName: "safari") }.buttonStyle(.plain).help("Open in the browser")
                if model.loading { ProgressView().controlSize(.mini) }
            }.padding(.horizontal, 10).padding(.vertical, 5).background(Palette.panel)
            Divider()
            WebSurface(model: model)
        }
        .onAppear {
            model.configure(root: (ws.root as NSString).resolvingSymlinksInPath)
            if field.isEmpty, let p = ws.activePath, p.lowercased().hasSuffix(".html") || p.lowercased().hasSuffix(".htm") {
                field = p
                DispatchQueue.main.async { model.load(p) }
            }
        }
        // a saved file that is the one on show: reload (a dev server does its own hot reload)
        .onChange(of: ws.saveTick) { _ in
            let shown = model.address
            if !shown.isEmpty, shown.hasPrefix("/") || !shown.contains("://") { model.reload() }
        }
        .onChange(of: model.address) { a in if field != a { field = a } }
    }
}

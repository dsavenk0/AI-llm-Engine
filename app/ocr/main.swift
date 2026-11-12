// A tiny local OCR tool: Apple Vision (VNRecognizeText) over images and PDF pages. On-device, no network,
// no third-party dependency — the macOS frameworks only. The Python side (engine/documents.py) runs it for
// scanned PDFs and image files; without it those are reported as "needs OCR" instead.
//
//   ocr <file.pdf | image> [more files…] [--lang en,ru]   →  recognised text on stdout
//   ocr --describe <image> [more images…]                 →  one JSON line per image: {"text", "labels"} —
//        what a model that cannot see is told about a picture (the browser assistant, assistant/describe.py)
//   ocr --pages <file.pdf> [--ocr-missing] [--lang en,ru]  →  JSON lines for the knowledge index
//        (engine/documents.py extract_pages): first {"pages": N}, then one {"page", "text", "source"} per page,
//        streamed as each page is done. The text layer comes from PDFKit (Unicode, CID fonts); Vision runs only
//        on pages without usable text, and only with --ocr-missing. source: "text" | "ocr" | "none".
//        A PDF that cannot be opened or is locked prints {"error": "…"} and exits 3.
import Foundation
import Vision
import PDFKit
import AppKit

func recognize(_ cg: CGImage, _ langs: [String]) -> String {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    if !langs.isEmpty { req.recognitionLanguages = langs }
    do { try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req]) } catch { return "" }
    return (req.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
}

func imageFile(_ path: String) -> CGImage? {
    guard let img = NSImage(contentsOfFile: path) else { return nil }
    var rect = CGRect(x: 0, y: 0, width: img.size.width, height: img.size.height)
    return img.cgImage(forProposedRect: &rect, context: nil, hints: nil)
}

func renderPage(_ page: PDFPage, scale: CGFloat) -> CGImage? {
    let rect = page.bounds(for: .mediaBox)
    let w = Int(rect.width * scale), h = Int(rect.height * scale)
    guard w > 0, h > 0, let ctx = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
    ctx.setFillColor(CGColor(gray: 1, alpha: 1))            // white page behind the content
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    ctx.scaleBy(x: scale, y: scale)
    ctx.translateBy(x: -rect.origin.x, y: -rect.origin.y)
    page.draw(with: .mediaBox, to: ctx)
    return ctx.makeImage()
}

func ocr(_ path: String, _ langs: [String]) -> String {
    if path.lowercased().hasSuffix(".pdf"), let doc = PDFDocument(url: URL(fileURLWithPath: path)) {
        var out: [String] = []
        for i in 0..<doc.pageCount {
            if let page = doc.page(at: i), let cg = renderPage(page, scale: 2.0) {
                let t = recognize(cg, langs)
                if !t.isEmpty { out.append(t) }
            }
        }
        return out.joined(separator: "\n\n")
    }
    if let cg = imageFile(path) { return recognize(cg, langs) }
    return ""
}

/// What is in a picture, as words: Vision's scene labels above a confidence floor, most likely first.
func labels(_ cg: CGImage) -> [String] {
    let req = VNClassifyImageRequest()
    do { try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req]) } catch { return [] }
    return (req.results ?? []).filter { $0.confidence >= 0.3 }
        .sorted { $0.confidence > $1.confidence }.prefix(8)
        .map { $0.identifier.replacingOccurrences(of: "_", with: " ") }
}

func emit(_ obj: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: obj), let line = String(data: data, encoding: .utf8) {
        print(line)
        fflush(stdout)                                      // the reader stops (or is stopped) between pages
    }
}

/// A page with fewer visible characters than this is treated as having no text layer (a scan, or only a
/// page number on top of one).
let minPageChars = 16

func pages(_ path: String, ocrMissing: Bool, langs: [String]) -> Int32 {
    guard let doc = PDFDocument(url: URL(fileURLWithPath: path)) else {
        emit(["error": "cannot open the PDF (damaged or not a PDF)"]); return 3
    }
    if doc.isLocked { emit(["error": "the PDF is password-protected"]); return 3 }
    emit(["pages": doc.pageCount])
    for i in 0..<doc.pageCount {
        autoreleasepool {
            guard let page = doc.page(at: i) else { emit(["page": i + 1, "text": "", "source": "none"]); return }
            let native = page.string ?? ""
            let visible = native.unicodeScalars.filter { !CharacterSet.whitespacesAndNewlines.contains($0) }.count
            if visible >= minPageChars { emit(["page": i + 1, "text": native, "source": "text"]); return }
            if ocrMissing, let cg = renderPage(page, scale: 2.0) {
                let t = recognize(cg, langs)
                if !t.isEmpty { emit(["page": i + 1, "text": t, "source": "ocr"]); return }
            }
            emit(["page": i + 1, "text": native, "source": visible > 0 ? "text" : "none"])
        }
    }
    return 0
}

var args = Array(CommandLine.arguments.dropFirst())
if let i = args.firstIndex(of: "--pages") {
    args.remove(at: i)
    let ocrMissing = args.contains("--ocr-missing")
    args.removeAll { $0 == "--ocr-missing" }
    var pageLangs: [String] = []
    if let j = args.firstIndex(of: "--lang"), j + 1 < args.count {
        pageLangs = args[j + 1].split(separator: ",").map(String.init)
        args.removeSubrange(j...(j + 1))
    }
    guard args.count == 1 else {
        FileHandle.standardError.write(Data("usage: ocr --pages <file.pdf> [--ocr-missing] [--lang en,ru]\n".utf8))
        exit(2)
    }
    exit(pages(args[0], ocrMissing: ocrMissing, langs: pageLangs))
}
if let i = args.firstIndex(of: "--describe") {
    args.remove(at: i)
    for path in args {
        var out: [String: Any] = ["text": "", "labels": [String]()]
        if let cg = imageFile(path) {
            out["text"] = recognize(cg, [])
            out["labels"] = labels(cg)
        }
        if let data = try? JSONSerialization.data(withJSONObject: out), let line = String(data: data, encoding: .utf8) {
            print(line)
        }
    }
    exit(0)
}
var langs: [String] = []
if let i = args.firstIndex(of: "--lang"), i + 1 < args.count {
    langs = args[i + 1].split(separator: ",").map(String.init)
    args.removeSubrange(i...(i + 1))
}
if args.isEmpty {
    FileHandle.standardError.write(Data("usage: ocr <file.pdf|image> [--lang en,ru]\n".utf8))
    exit(2)
}
print(args.map { ocr($0, langs) }.joined(separator: "\n\n"))

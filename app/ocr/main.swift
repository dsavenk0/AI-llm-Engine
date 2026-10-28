// A tiny local OCR tool: Apple Vision (VNRecognizeText) over images and PDF pages. On-device, no network,
// no third-party dependency — the macOS frameworks only. The Python side (engine/documents.py) runs it for
// scanned PDFs and image files; without it those are reported as "needs OCR" instead.
//
//   ocr <file.pdf | image> [more files…] [--lang en,ru]   →  recognised text on stdout
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

var args = Array(CommandLine.arguments.dropFirst())
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

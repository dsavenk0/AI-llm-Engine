import SwiftUI
import AppKit

// MARK: - theme tokens from LLMBar.swift
var lightTheme = false

enum HUD {
    static var cyan: Color { lightTheme ? Color(red: 0.0, green: 0.45, blue: 0.62) : Color(red: 0.0, green: 0.90, blue: 1.0) }
    static var phosphor: Color { lightTheme ? Color(red: 0.0, green: 0.56, blue: 0.45) : Color(red: 0.30, green: 0.95, blue: 0.76) }
    static var magenta: Color { lightTheme ? Color(red: 0.82, green: 0.08, blue: 0.40) : Color(red: 1.0, green: 0.18, blue: 0.53) }
    static var amber: Color { lightTheme ? Color(red: 0.78, green: 0.42, blue: 0.0) : Color(red: 1.0, green: 0.64, blue: 0.16) }
    static var ink: Color { lightTheme ? Color(red: 0.97, green: 0.98, blue: 0.99) : Color(red: 0.02, green: 0.04, blue: 0.06) }
    static var text: Color { lightTheme ? Color(red: 0.05, green: 0.09, blue: 0.13) : Color(red: 0.86, green: 0.95, blue: 0.98) }
    static var dim: Color { lightTheme ? Color(red: 0.36, green: 0.43, blue: 0.48) : Color(red: 0.45, green: 0.58, blue: 0.64) }
    static var panel: Color { lightTheme ? Color.white.opacity(0.42) : Color.black.opacity(0.20) }
    static var track: Color { lightTheme ? Color.black.opacity(0.07) : Color.white.opacity(0.06) }

    static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

struct Card<Content: View>: View {
    var glow: Color = HUD.cyan
    @ViewBuilder let content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 7) { content }
            .padding(11)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(HUD.panel)
                    .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(LinearGradient(colors: [glow.opacity(0.45), glow.opacity(0.08)],
                                                     startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 0.8))
            )
    }
}

struct Caption: View {
    let text: String
    var color: Color = HUD.cyan
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 5, height: 5)
            Text(text.uppercased()).font(HUD.mono(10, .semibold)).tracking(2.2).foregroundColor(color)
        }
    }
}

struct MetricBadge: View {
    let val: String
    let unit: String
    let label: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(val).font(HUD.mono(22, .bold)).foregroundColor(color)
                Text(unit).font(HUD.mono(11)).foregroundColor(HUD.dim)
            }
            Text(label.uppercased()).font(HUD.mono(8, .medium)).tracking(1.2).foregroundColor(HUD.dim)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// Window Chrome Mockup
struct AppMockupWindow<Content: View>: View {
    let title: String
    let subtitle: String
    let activeTab: Int
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // macOS traffic lights & title
            HStack {
                HStack(spacing: 6) {
                    Circle().fill(Color(red: 1.0, green: 0.37, blue: 0.34)).frame(width: 10, height: 10)
                    Circle().fill(Color(red: 1.0, green: 0.74, blue: 0.18)).frame(width: 10, height: 10)
                    Circle().fill(Color(red: 0.15, green: 0.79, blue: 0.25)).frame(width: 10, height: 10)
                }
                Spacer()
                VStack(spacing: 2) {
                    Text(title).font(HUD.mono(12, .semibold)).foregroundColor(HUD.text)
                    Text(subtitle).font(HUD.mono(9)).foregroundColor(HUD.dim)
                }
                Spacer()
                Circle().fill(HUD.phosphor).frame(width: 6, height: 6)
            }
            .padding(.bottom, 2)

            // Segmented Picker
            HStack(spacing: 2) {
                ForEach(["Engine", "Monitor", "Models", "Settings", "Tools"].indices, id: \.self) { idx in
                    let names = ["Engine", "Monitor", "Models", "Settings", "Tools"]
                    Text(names[idx])
                        .font(HUD.mono(10, idx == activeTab ? .semibold : .regular))
                        .foregroundColor(idx == activeTab ? HUD.text : HUD.dim)
                        .padding(.vertical, 5)
                        .frame(maxWidth: .infinity)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(idx == activeTab ? Color.white.opacity(0.12) : Color.clear)
                        )
                }
            }
            .padding(3)
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Color.black.opacity(0.35)))

            content

            HStack {
                Text("AI-llm-ENGINE · SECTION 9 HUD").font(HUD.mono(8, .medium)).tracking(1.6).foregroundColor(HUD.dim.opacity(0.7))
                Spacer()
                Text("v1.0 · M-Series").font(HUD.mono(8)).foregroundColor(HUD.dim)
            }
            .padding(.top, 4)
        }
        .padding(16)
        .frame(width: 410)
        .background(
            ZStack {
                Color(red: 0.04, green: 0.06, blue: 0.08)
                LinearGradient(colors: [HUD.cyan.opacity(0.08), .clear], startPoint: .topLeading, endPoint: .bottomTrailing)
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(LinearGradient(colors: [HUD.cyan.opacity(0.35), HUD.cyan.opacity(0.08)], startPoint: .top, endPoint: .bottom), lineWidth: 1)
            }
        )
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .shadow(color: Color.black.opacity(0.6), radius: 25, x: 0, y: 15)
        .padding(20)
    }
}

// 1. Engine Screenshot
struct EngineMockView: View {
    var body: some View {
        AppMockupWindow(title: "LLMBar", subtitle: "Bonsai-2-27B-CRACK-Ternary-JANG", activeTab: 0) {
            VStack(spacing: 10) {
                Card {
                    Caption(text: "Core Status")
                    HStack {
                        Text("Context · output").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("45,056 · 32,000").font(HUD.mono(11, .medium)).foregroundColor(HUD.text)
                    }
                    HStack {
                        Text("Hadamard").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("Fused Matmul (JANG) ✓").font(HUD.mono(11, .medium)).foregroundColor(HUD.phosphor)
                    }
                    HStack {
                        Text("RAM guard").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("55% / 88%").font(HUD.mono(11, .medium)).foregroundColor(HUD.cyan)
                    }
                }

                Card(glow: HUD.phosphor) {
                    Caption(text: "Live Performance", color: HUD.phosphor)
                    HStack(spacing: 0) {
                        MetricBadge(val: "34.8", unit: "tok/s", label: "Speculative gen", color: HUD.phosphor)
                        MetricBadge(val: "119.5", unit: "tok/s", label: "Prefill speed", color: HUD.cyan)
                        MetricBadge(val: "4.0", unit: "sec", label: "Cold start", color: HUD.text)
                    }
                }

                Card {
                    Caption(text: "Unified Memory Breakdown")
                    HStack {
                        Text("Model weights (pinned)").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("7.61 GB").font(HUD.mono(11)).foregroundColor(HUD.cyan)
                    }
                    HStack {
                        Text("GatedDeltaNet SSM state").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("0.41 GB").font(HUD.mono(11)).foregroundColor(HUD.text)
                    }
                    HStack {
                        Text("SSD prefix cache reused").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("310 / 340 tok").font(HUD.mono(11)).foregroundColor(HUD.phosphor)
                    }
                }
            }
        }
    }
}

// 2. Monitor Screenshot
struct MonitorMockView: View {
    var body: some View {
        AppMockupWindow(title: "LLMBar", subtitle: "System Monitor · Activity", activeTab: 1) {
            VStack(spacing: 10) {
                Card {
                    Caption(text: "System Memory")
                    HStack {
                        Text("Used / Total").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("8.8 GB / 16.0 GB (55%)").font(HUD.mono(12, .semibold)).foregroundColor(HUD.text)
                    }
                    // Bar
                    GeometryReader { g in
                        HStack(spacing: 2) {
                            RoundedRectangle(cornerRadius: 3).fill(HUD.cyan).frame(width: g.size.width * 0.48)
                            RoundedRectangle(cornerRadius: 3).fill(Color.purple).frame(width: g.size.width * 0.08)
                            RoundedRectangle(cornerRadius: 3).fill(HUD.phosphor).frame(width: g.size.width * 0.22)
                            RoundedRectangle(cornerRadius: 3).fill(Color.white.opacity(0.12)).frame(width: g.size.width * 0.20)
                        }
                    }.frame(height: 10)

                    HStack {
                        Text("Thermal State").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("Nominal ⚡").font(HUD.mono(11, .medium)).foregroundColor(HUD.phosphor)
                    }
                }

                Card(glow: HUD.amber) {
                    Caption(text: "Hardware Protection", color: HUD.amber)
                    HStack {
                        Text("Disk Swapping").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("0 bytes (Zero Swap)").font(HUD.mono(11, .medium)).foregroundColor(HUD.phosphor)
                    }
                    HStack {
                        Text("Wired Memory Limit").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("7.6 GB Pinned").font(HUD.mono(11, .medium)).foregroundColor(HUD.amber)
                    }
                    HStack {
                        Text("Auto-Sleep (Idle)").font(.system(size: 11)).foregroundColor(HUD.dim)
                        Spacer()
                        Text("15 min → Frees 7.6 GB").font(HUD.mono(11)).foregroundColor(HUD.text)
                    }
                }
            }
        }
    }
}

// 3. Models Screenshot
struct ModelsMockView: View {
    var body: some View {
        AppMockupWindow(title: "LLMBar", subtitle: "Model Hub · Hugging Face", activeTab: 2) {
            VStack(spacing: 8) {
                Card(glow: HUD.phosphor) {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Bonsai-2-27B-CRACK-Ternary-JANG").font(HUD.mono(11, .semibold)).foregroundColor(HUD.phosphor)
                            Text("27B · ~7.6 GB · Hadamard Ternary · Fits 16GB").font(.system(size: 10)).foregroundColor(HUD.dim)
                        }
                        Spacer()
                        Text("ACTIVE ✓").font(HUD.mono(9, .bold)).foregroundColor(HUD.phosphor)
                            .padding(.horizontal, 6).padding(.vertical, 3)
                            .background(RoundedRectangle(cornerRadius: 4).fill(HUD.phosphor.opacity(0.15)))
                    }
                }

                Card {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Qwen3.5-35B-A3B-Ternary-JANG").font(HUD.mono(11, .semibold)).foregroundColor(HUD.text)
                            Text("35B MoE · 12.2 GB · Needs 24GB+ RAM").font(.system(size: 10)).foregroundColor(HUD.dim)
                        }
                        Spacer()
                        Text("Switch").font(HUD.mono(9)).foregroundColor(HUD.cyan)
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(RoundedRectangle(cornerRadius: 4).fill(HUD.cyan.opacity(0.15)))
                    }
                }

                Card {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Qwen3.5-4B-MLX-4bit").font(HUD.mono(11, .semibold)).foregroundColor(HUD.text)
                            Text("4B · 2.4 GB · Ultra-light · 55+ tok/s").font(.system(size: 10)).foregroundColor(HUD.dim)
                        }
                        Spacer()
                        Text("Pull").font(HUD.mono(9)).foregroundColor(HUD.dim)
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(RoundedRectangle(cornerRadius: 4).fill(Color.white.opacity(0.08)))
                    }
                }
            }
        }
    }
}

// Render helper
func renderView<V: View>(_ view: V, to path: String, width: CGFloat = 450, height: CGFloat = 460) {
    let hosting = NSHostingView(rootView: view)
    hosting.frame = NSRect(x: 0, y: 0, width: width, height: height)
    hosting.layoutSubtreeIfNeeded()

    guard let rep = hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds) else { return }
    hosting.cacheDisplay(in: hosting.bounds, to: rep)
    let png = rep.representation(using: .png, properties: [:])
    try? png?.write(to: URL(fileURLWithPath: path))
    print("Rendered: \(path)")
}

renderView(EngineMockView(), to: "docs/assets/screenshot-engine.png")
renderView(MonitorMockView(), to: "docs/assets/screenshot-monitor.png")
renderView(ModelsMockView(), to: "docs/assets/screenshot-models.png")

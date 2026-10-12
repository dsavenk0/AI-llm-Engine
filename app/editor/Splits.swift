// Panel sizes the person sets are kept: a thin draggable divider that remembers its pane's width or height in the
// user defaults (the system split views of SwiftUI forget them every launch).
import AppKit
import SwiftUI

enum SplitPrefs {
    static func get(_ key: String, _ fallback: CGFloat) -> CGFloat {
        let v = UserDefaults.standard.double(forKey: key)
        return v > 0 ? CGFloat(v) : fallback
    }
    static func set(_ key: String, _ value: CGFloat) { UserDefaults.standard.set(Double(value), forKey: key) }
    static func clamp(_ v: CGFloat, _ lo: CGFloat, _ hi: CGFloat) -> CGFloat { max(lo, min(max(lo, hi), v)) }
}

/// A divider between two panes. `size` is the size of the pane it resizes; dragging changes it within `range` and
/// the result is saved under `key`. `growsWithDrag` is false for a pane on the far side (dragging left widens it).
struct SplitHandle: View {
    enum Axis { case columns, rows }
    let axis: Axis
    @Binding var size: CGFloat
    let range: ClosedRange<CGFloat>
    var growsWithDrag = true
    let key: String
    @State private var start: CGFloat?

    var body: some View {
        let vertical = axis == .columns
        Rectangle().fill(Color.clear)
            .frame(width: vertical ? 7 : nil, height: vertical ? nil : 7)
            .overlay(Rectangle().fill(Color.primary.opacity(0.14))
                        .frame(width: vertical ? 1 : nil, height: vertical ? nil : 1))
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { (vertical ? NSCursor.resizeLeftRight : NSCursor.resizeUpDown).push() } else { NSCursor.pop() }
            }
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { g in
                    if start == nil { start = size }
                    let d = vertical ? g.translation.width : g.translation.height
                    size = SplitPrefs.clamp((start ?? size) + (growsWithDrag ? d : -d), range.lowerBound, range.upperBound)
                }
                .onEnded { _ in
                    start = nil
                    SplitPrefs.set(key, size)
                })
    }
}

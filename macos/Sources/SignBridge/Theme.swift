import AppKit
import SwiftUI

/// SignBridge's look: a violet-to-coral gradient (the app icon), used sparingly for emphasis.
enum Brand {
    static let violet = Color(red: 0.36, green: 0.29, blue: 0.96)
    static let magenta = Color(red: 0.71, green: 0.29, blue: 0.85)
    static let coral = Color(red: 1.0, green: 0.48, blue: 0.35)
    static let gradient = LinearGradient(colors: [violet, magenta, coral], startPoint: .topLeading,
                                         endPoint: .bottomTrailing)

    /// Menu bar glyph from the bundle (a template image, so it follows light and dark menu bars).
    static func menuBarImage(listening: Bool) -> NSImage? {
        let name = listening ? "menubar-on" : "menubar"
        guard let url = Bundle.main.url(forResource: name, withExtension: "svg"), let image = NSImage(contentsOf: url)
        else { return NSImage(systemSymbolName: listening ? "hand.raised.fill" : "hand.raised",
                              accessibilityDescription: "SignBridge") }
        image.size = NSSize(width: 18, height: 18)
        image.isTemplate = true
        image.accessibilityDescription = "SignBridge"
        return image
    }
}

/// The app icon, or a gradient tile when running unbundled (swift run).
struct Logo: View {
    var size: CGFloat = 64
    var body: some View {
        if Bundle.main.url(forResource: "AppIcon", withExtension: "icns") != nil {
            Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: size, height: size)
        } else {
            RoundedRectangle(cornerRadius: size * 0.225).fill(Brand.gradient).frame(width: size * 0.8, height: size * 0.8)
                .overlay(Image(systemName: "hand.raised").font(.system(size: size * 0.4)).foregroundStyle(.white))
                .frame(width: size, height: size)
        }
    }
}

/// Text filled with the brand gradient.
struct GradientText: View {
    let text: String, font: Font
    init(_ text: String, font: Font) { self.text = text; self.font = font }
    var body: some View {
        Text(text).font(font).foregroundStyle(Brand.gradient)
    }
}

/// A small rounded tag, used for recognized signs.
struct Chip: View {
    let text: String
    var body: some View {
        Text(text).font(.system(.title3, design: .rounded).weight(.semibold))
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(Brand.violet.opacity(0.18), in: Capsule())
            .overlay(Capsule().strokeBorder(Brand.violet.opacity(0.45)))
    }
}

/// A dot that pulses while something is live.
struct PulseDot: View {
    let color: Color
    var active = true
    @State private var on = false
    var body: some View {
        Circle().fill(color).frame(width: 9, height: 9)
            .overlay(Circle().stroke(color, lineWidth: 2).scaleEffect(on ? 2.2 : 1).opacity(on ? 0 : 0.8))
            .onAppear { on = active }
            .animation(active ? .easeOut(duration: 1.1).repeatForever(autoreverses: false) : .default, value: on)
            .onChange(of: active) { _, a in on = a }
    }
}

extension NSAttributedString {
    func appending(_ other: NSAttributedString) -> NSAttributedString {
        let s = NSMutableAttributedString(attributedString: self)
        s.append(other)
        return s
    }
}

/// The native translucent sidebar background.
struct SidebarBackground: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .sidebar
        v.blendingMode = .behindWindow
        v.state = .followsWindowActiveState
        return v
    }
    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}

/// How each output route is shown.
enum RouteInfo {
    static func icon(_ route: String?) -> String {
        ["siri": "mic.circle.fill", "text": "keyboard", "voice": "speaker.wave.2.fill"][route ?? ""] ?? "arrow.right"
    }
    static func label(_ route: String?, target: String?) -> String {
        switch route {
        case "siri": "Ask Siri"
        case "text": "Type into \(target ?? "the active app")"
        case "voice": "Say it in the call"
        default: ""
        }
    }
}

/// How each control gesture is shown.
enum GestureInfo {
    static func name(_ g: String) -> String {
        ["wake": "Start", "confirm": "Send", "cancel": "Cancel"][g] ?? g.capitalized
    }
    static func icon(_ g: String) -> String {
        ["wake": "hand.raised", "confirm": "hand.thumbsup", "cancel": "hand.thumbsdown"][g] ?? "hand.raised"
    }
}

import SwiftUI

/// Live view for the user and for demos: camera with tracking, what was signed, Claude's sentence and route.
struct PreviewView: View {
    @ObservedObject var engine: EngineClient
    var showGuide: () -> Void = {}
    private let fakePhrases: [[String]] = [["RAIN", "TOMORROW"], ["HUNGRY", "PIZZA"], ["HELLO"], ["WAIT"],
                                           ["CALLONPHONE", "MOM"], ["TIME", "NOW"]]

    private var live: Bool { engine.state == "listening" }

    var body: some View {
        VStack(spacing: 14) {
            topBar
            ZStack(alignment: .bottom) {
                LivePreview(engine: engine).frame(maxHeight: .infinity)
                    .overlay(RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(Brand.gradient, lineWidth: live ? 3 : 0)
                        .shadow(color: Brand.magenta.opacity(live ? 0.7 : 0), radius: 12))
                    .animation(.easeInOut(duration: 0.3), value: live)
                if let h = engine.holding {
                    HStack(spacing: 10) {
                        Image(systemName: gestureIcon(h.0))
                        Text(gestureName(h.0)).bold()
                        ProgressView(value: h.1).frame(width: 110).tint(.white)
                    }
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule()).padding(12)
                    .transition(.opacity)
                }
            }
            transcript
            controls
        }
        .padding([.horizontal, .bottom], 18).padding(.top, 6)
        .frame(minWidth: 600, minHeight: 600)
        .background(Color(white: 0.07))
        .environment(\.colorScheme, .dark)
        .tint(Brand.violet)
        .ignoresSafeArea(.container, edges: .top)
    }

    // MARK: pieces

    private var topBar: some View {
        HStack(spacing: 12) {
            Spacer().frame(width: 58)  // room for the window buttons
            Logo(size: 28)
            Text("SignBridge").font(.headline)
            StatePill(state: engine.state)
            Spacer()
            if let target = engine.target, engine.state != "idle" {
                Label(target, systemImage: "macwindow").foregroundStyle(.secondary)
            }
            if engine.fps > 0 {
                Text("\(Int(engine.fps)) fps").font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
            }
        }
    }

    private var transcript: some View {
        VStack(alignment: .leading, spacing: 12) {
            switch engine.state {
            case "listening", "thinking":
                if engine.glosses.isEmpty {
                    Text("Sign, then rest your hands.").font(.title2).foregroundStyle(.secondary)
                } else {
                    HStack(spacing: 8) {
                        ForEach(Array(engine.glosses.enumerated()), id: \.offset) { _, g in Chip(text: g) }
                        if engine.state == "thinking" {
                            Image(systemName: "arrow.right").foregroundStyle(.secondary)
                            ProgressView().controlSize(.small)
                            Text("Claude is writing…").foregroundStyle(.secondary)
                        }
                    }
                }
            case "confirming", "sending":
                HStack(spacing: 6) {
                    ForEach(Array(engine.glosses.enumerated()), id: \.offset) { _, g in
                        Text(g).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    }
                }
                Text(engine.sentence ?? "").font(.system(size: 30, weight: .semibold, design: .rounded))
                    .fixedSize(horizontal: false, vertical: true)
                HStack {
                    Label(routeText, systemImage: routeIcon).foregroundStyle(Brand.gradient).font(.headline)
                    Spacer()
                    if engine.state == "confirming" {
                        Text("👍 send   👎 cancel").foregroundStyle(.secondary)
                    } else {
                        ProgressView().controlSize(.small)
                    }
                }
            default:
                HStack(spacing: 10) {
                    Image(systemName: "hand.raised").font(.title2).foregroundStyle(Brand.gradient)
                    Text("Show both open palms to start.").font(.title2).foregroundStyle(.secondary)
                }
            }
            if engine.state == "idle", let last = engine.log.last {  // e.g. what was just sent, or an error
                Text(last).font(.caption).foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 110, alignment: .leading)
        .padding(16)
        .background(Color(white: 0.12), in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Color.white.opacity(0.08)))
        .animation(.easeInOut(duration: 0.2), value: engine.state)
    }

    private var controls: some View {
        HStack(spacing: 10) {
            switch engine.state {
            case "idle":
                Button { engine.gesture("wake") } label: { Label("Start", systemImage: "hand.raised") }
                    .buttonStyle(.borderedProminent)
            case "confirming":
                Button { engine.gesture("confirm") } label: { Label("Send", systemImage: "hand.thumbsup") }
                    .buttonStyle(.borderedProminent).tint(.green)
                Button { engine.gesture("cancel") } label: { Label("Cancel", systemImage: "hand.thumbsdown") }
            default:
                Button { engine.gesture("cancel") } label: { Label("Cancel", systemImage: "hand.thumbsdown") }
            }
            Spacer()
            Button(action: showGuide) { Label("Signs", systemImage: "book") }.buttonStyle(.borderless)
            Menu {
                ForEach(fakePhrases, id: \.self) { p in
                    Button(p.joined(separator: " ")) { engine.fake(p) }
                }
            } label: {
                Label("Test phrases", systemImage: "text.badge.plus")
            }
            .menuStyle(.borderlessButton).fixedSize()
        }
        .controlSize(.large)
    }

    private var routeText: String {
        switch engine.route {
        case "siri": "Ask Siri"
        case "text": "Type into \(engine.target ?? "the active app")"
        case "voice": "Say it in the call"
        default: ""
        }
    }

    private var routeIcon: String {
        ["siri": "mic.circle.fill", "text": "keyboard", "voice": "speaker.wave.2.fill"][engine.route ?? ""] ?? "arrow.right"
    }

    private func gestureName(_ g: String) -> String {
        ["wake": "Start", "confirm": "Send", "cancel": "Cancel"][g] ?? g.capitalized
    }

    private func gestureIcon(_ g: String) -> String {
        ["wake": "hand.raised", "confirm": "hand.thumbsup", "cancel": "hand.thumbsdown"][g] ?? "hand.raised"
    }
}

struct StatePill: View {
    let state: String
    var body: some View {
        HStack(spacing: 7) {
            PulseDot(color: color, active: state == "listening" || state == "thinking")
            Text(label).font(.subheadline.weight(.semibold))
        }
        .padding(.horizontal, 11).padding(.vertical, 5)
        .background(color.opacity(0.16), in: Capsule())
        .foregroundStyle(color)
    }
    private var label: String {
        ["idle": "Waiting", "listening": "Listening", "thinking": "Thinking", "confirming": "Confirm",
         "sending": "Sending"][state] ?? state
    }
    private var color: Color {
        ["idle": .gray, "listening": Brand.coral, "thinking": Brand.magenta, "confirming": .yellow,
         "sending": .green][state] ?? .gray
    }
}

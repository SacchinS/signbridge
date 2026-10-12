import SwiftUI

/// Live view for the user and for demos: camera with tracking, what was signed, Claude's sentence and route.
struct PreviewView: View {
    @ObservedObject var engine: EngineClient
    private let fakePhrases: [[String]] = [["RAIN", "TOMORROW"], ["HUNGRY", "PIZZA"], ["HELLO"], ["WAIT"],
                                           ["CALLONPHONE", "MOM"], ["TIME", "NOW"]]

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                StatePill(state: engine.state)
                if let target = engine.target, engine.state != "idle" {
                    Text("→ \(target)").foregroundStyle(.secondary)
                }
                Spacer()
                if engine.fps > 0 { Text("\(Int(engine.fps)) fps").font(.caption).foregroundStyle(.secondary) }
            }
            ZStack(alignment: .bottom) {
                LivePreview(engine: engine).frame(maxHeight: .infinity)
                if let h = engine.holding {
                    HStack {
                        Text(h.0.capitalized).bold()
                        ProgressView(value: h.1).frame(width: 120)
                    }
                    .padding(8).background(.regularMaterial, in: Capsule()).padding(10)
                }
            }
            caption
            HStack {
                Button { engine.gesture("wake") } label: { Label("Start", systemImage: "hand.raised") }
                Button { engine.gesture("confirm") } label: { Label("Send", systemImage: "hand.thumbsup") }
                    .disabled(engine.state != "confirming")
                Button { engine.gesture("cancel") } label: { Label("Cancel", systemImage: "hand.thumbsdown") }
                Spacer()
                Menu("Test phrases") {
                    ForEach(fakePhrases, id: \.self) { p in
                        Button(p.joined(separator: " ")) { engine.fake(p) }
                    }
                }
                .frame(width: 140)
            }
        }
        .padding(16)
        .frame(minWidth: 560, minHeight: 520)
    }

    @ViewBuilder private var caption: some View {
        VStack(alignment: .leading, spacing: 6) {
            switch engine.state {
            case "listening":
                Text(engine.glosses.isEmpty ? "Sign, then rest your hands." : engine.glosses.joined(separator: " "))
                    .font(.system(size: 30, weight: .semibold, design: .rounded))
            case "thinking":
                Text(engine.glosses.joined(separator: " ")).font(.system(size: 30, weight: .semibold, design: .rounded))
                ProgressView().controlSize(.small)
            case "confirming", "sending":
                Text("“\(engine.sentence ?? "")”").font(.system(size: 28, weight: .semibold, design: .rounded))
                Text(routeText).foregroundStyle(.secondary)
                if engine.state == "confirming" {
                    Text("👍 Thumbs up to send   ·   👎 Thumbs down to cancel").foregroundStyle(.secondary)
                }
            default:
                Text("Show both open palms to start.").font(.title2).foregroundStyle(.secondary)
            }
            if let last = engine.log.last { Text(last).font(.caption).foregroundStyle(.tertiary) }
        }
        .frame(maxWidth: .infinity, minHeight: 90, alignment: .leading)
    }

    private var routeText: String {
        switch engine.route {
        case "siri": "→ Ask Siri"
        case "text": "→ Type into \(engine.target ?? "the active app")"
        case "voice": "→ Say it out loud in the call"
        default: ""
        }
    }
}

struct StatePill: View {
    let state: String
    var body: some View {
        Text(label).font(.headline).padding(.horizontal, 12).padding(.vertical, 5)
            .background(color.opacity(0.2), in: Capsule()).foregroundStyle(color)
    }
    private var label: String {
        ["idle": "Waiting", "listening": "Listening", "thinking": "Thinking", "confirming": "Check and confirm",
         "sending": "Sending"][state] ?? state
    }
    private var color: Color {
        ["idle": .gray, "listening": .red, "thinking": .orange, "confirming": .yellow, "sending": .green][state] ?? .gray
    }
}

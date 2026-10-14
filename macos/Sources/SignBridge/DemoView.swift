import SwiftUI

/// Full-screen stage for showing SignBridge to an audience: big camera, big text, and what happened so far.
struct DemoView: View {
    @ObservedObject var engine: EngineClient
    let exit: () -> Void

    private var live: Bool { engine.state == "listening" }

    var body: some View {
        HStack(spacing: 28) {
            VStack(alignment: .leading, spacing: 18) {
                HStack(spacing: 14) {
                    Logo(size: 44)
                    GradientText("SignBridge", font: .system(size: 30, weight: .bold, design: .rounded))
                    StatePill(state: engine.state).scaleEffect(1.25).padding(.leading, 10)
                    Spacer()
                    if let target = engine.target, engine.state != "idle" {
                        Label(target, systemImage: "macwindow").font(.title3).foregroundStyle(.secondary)
                    }
                }
                ZStack(alignment: .bottom) {
                    LivePreview(engine: engine, maxHeight: .infinity)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .overlay(RoundedRectangle(cornerRadius: 10)
                            .strokeBorder(Brand.gradient, lineWidth: live ? 4 : 0)
                            .shadow(color: Brand.magenta.opacity(live ? 0.8 : 0), radius: 18))
                        .animation(.easeInOut(duration: 0.3), value: live)
                    if let h = engine.holding {
                        HStack(spacing: 12) {
                            Image(systemName: GestureInfo.icon(h.0))
                            Text(GestureInfo.name(h.0)).bold()
                            ProgressView(value: h.1).frame(width: 160).tint(.white)
                        }
                        .font(.title2)
                        .padding(.horizontal, 20).padding(.vertical, 12)
                        .background(.ultraThinMaterial, in: Capsule()).padding(18)
                    }
                }
                stage.frame(minHeight: 190)
            }
            history.frame(width: 340)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(white: 0.05))
        .environment(\.colorScheme, .dark)
        .tint(Brand.violet)
        .background { shortcuts }
        .overlay(alignment: .topTrailing) {
            Button("Exit Demo", action: exit).keyboardShortcut(.cancelAction).buttonStyle(.borderless)
                .foregroundStyle(.tertiary).padding(10)
        }
    }

    /// Backup keys for a live demo: W start, C send, X cancel, 1-6 test phrases (as in dev_app.py).
    private var shortcuts: some View {
        let phrases = [["RAIN", "TOMORROW"], ["HUNGRY", "PIZZA"], ["HELLO"], ["WAIT"], ["CALLONPHONE", "MOM"], ["TIME", "NOW"]]
        return ZStack {
            Button("") { engine.gesture("wake") }.keyboardShortcut("w", modifiers: [])
            Button("") { engine.gesture("confirm") }.keyboardShortcut("c", modifiers: [])
            Button("") { engine.gesture("cancel") }.keyboardShortcut("x", modifiers: [])
            ForEach(Array(phrases.enumerated()), id: \.offset) { i, p in
                Button("") { engine.fake(p) }.keyboardShortcut(KeyEquivalent(Character("\(i + 1)")), modifiers: [])
            }
        }
        .opacity(0).allowsHitTesting(false)
    }

    /// What's happening now, sized for the back of the room.
    private var stage: some View {
        VStack(alignment: .leading, spacing: 16) {
            switch engine.state {
            case "listening", "thinking":
                if engine.glosses.isEmpty {
                    hint("hand.raised", "Signing… rest your hands to finish")
                } else {
                    HStack(spacing: 12) {
                        ForEach(Array(engine.glosses.enumerated()), id: \.offset) { _, g in
                            Text(g).font(.system(size: 34, weight: .bold, design: .rounded))
                                .padding(.horizontal, 20).padding(.vertical, 10)
                                .background(Brand.violet.opacity(0.2), in: Capsule())
                                .overlay(Capsule().strokeBorder(Brand.violet.opacity(0.5), lineWidth: 1.5))
                                .transition(.scale.combined(with: .opacity))
                        }
                        if engine.state == "thinking" {
                            Image(systemName: "arrow.right").font(.title).foregroundStyle(.secondary)
                            ProgressView()
                            Text("Claude is writing…").font(.title2).foregroundStyle(.secondary)
                        }
                    }
                    .animation(.spring(duration: 0.35), value: engine.glosses)
                }
            case "confirming", "sending":
                Text(engine.glosses.joined(separator: "  ·  ")).font(.title3.weight(.semibold)).foregroundStyle(.secondary)
                Text(engine.sentence ?? "").font(.system(size: 52, weight: .bold, design: .rounded))
                    .fixedSize(horizontal: false, vertical: true)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                HStack {
                    Label(RouteInfo.label(engine.route, target: engine.target), systemImage: RouteInfo.icon(engine.route))
                        .font(.title2.weight(.semibold)).foregroundStyle(Brand.gradient)
                    Spacer()
                    if engine.state == "confirming" {
                        Text("👍 send    👎 cancel").font(.title2).foregroundStyle(.secondary)
                    } else {
                        ProgressView()
                        Text("Sending…").font(.title2).foregroundStyle(.secondary)
                    }
                }
            default:
                hint("hand.raised", "Show both open palms to start")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(24)
        .background(Color(white: 0.1), in: RoundedRectangle(cornerRadius: 20))
        .overlay(RoundedRectangle(cornerRadius: 20).strokeBorder(Color.white.opacity(0.08)))
        .animation(.easeInOut(duration: 0.25), value: engine.state)
    }

    private func hint(_ icon: String, _ text: String) -> some View {
        HStack(spacing: 14) {
            Image(systemName: icon).foregroundStyle(Brand.gradient)
            Text(text).foregroundStyle(.secondary)
        }
        .font(.system(size: 34, weight: .semibold, design: .rounded))
    }

    /// Recent phrases: signs, sentence, where it went, and whether it was sent.
    private var history: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent").font(.title3.weight(.semibold)).foregroundStyle(.secondary).padding(.top, 60)
            if engine.history.isEmpty {
                Text("Phrases you send or cancel show up here.").foregroundStyle(.tertiary)
            }
            ScrollView {
                VStack(spacing: 10) {
                    ForEach(engine.history) { item in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(item.glosses.joined(separator: " · ")).font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(item.date, style: .time).font(.caption).foregroundStyle(.tertiary)
                            }
                            Text(item.sentence).font(.title3.weight(.semibold)).fixedSize(horizontal: false, vertical: true)
                            HStack(spacing: 6) {
                                Image(systemName: RouteInfo.icon(item.route))
                                Text(item.sent ? RouteInfo.label(item.route, target: nil) : "Cancelled")
                                Spacer()
                                Image(systemName: item.sent ? "checkmark.circle.fill" : "xmark.circle.fill")
                                    .foregroundStyle(item.sent ? .green : .secondary)
                            }
                            .font(.callout).foregroundStyle(item.sent ? .primary : .secondary)
                        }
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(white: 0.1), in: RoundedRectangle(cornerRadius: 14))
                        .opacity(item.sent ? 1 : 0.6)
                        .transition(.move(edge: .top).combined(with: .opacity))
                    }
                }
                .animation(.spring(duration: 0.4), value: engine.history.map(\.id))
            }
        }
    }
}

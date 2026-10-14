import SwiftUI

/// "How it works" on the welcome page: four steps that light up in turn, each with a small animation.
struct Storyboard: View {
    @State private var active = 0
    @State private var since = Date()
    private let timer = Timer.publish(every: 2.6, on: .main, in: .common).autoconnect()
    private let steps: [(String, String)] = [
        ("Open palms", "Both hands up, held still, to start."),
        ("Sign", "Make your signs, then rest your hands."),
        ("Claude writes it", "Your signs become a sentence."),
        ("Thumbs up", "Sends it. Thumbs down cancels."),
    ]

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            ForEach(steps.indices, id: \.self) { i in
                let on = i == active
                VStack(alignment: .leading, spacing: 8) {
                    visual(i, on: on)
                        .frame(maxWidth: .infinity).frame(height: 118)
                        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 10))
                    HStack(spacing: 6) {
                        Text("\(i + 1)").font(.caption.weight(.bold)).foregroundStyle(.white)
                            .frame(width: 18, height: 18)
                            .background(on ? AnyShapeStyle(Brand.gradient) : AnyShapeStyle(Color.secondary.opacity(0.4)),
                                        in: Circle())
                        Text(steps[i].0).font(.headline)
                    }
                    Text(steps[i].1).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                .padding(10)
                .background(.quaternary.opacity(on ? 0.7 : 0.35), in: RoundedRectangle(cornerRadius: 14))
                .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Brand.gradient, lineWidth: on ? 2 : 0))
                .opacity(on ? 1 : 0.55)
                .scaleEffect(on ? 1.0 : 0.97)
                .onTapGesture { active = i }
            }
        }
        .animation(.easeInOut(duration: 0.35), value: active)
        .onReceive(timer) { _ in active = (active + 1) % steps.count }
        .onChange(of: active) { _, _ in since = Date() }
    }

    @ViewBuilder private func visual(_ i: Int, on: Bool) -> some View {
        switch i {
        case 0:
            HStack(spacing: 14) {
                Image(systemName: "hand.raised.fill").scaleEffect(x: -1)
                Image(systemName: "hand.raised.fill")
            }
            .font(.system(size: 40))
            .foregroundStyle(on ? AnyShapeStyle(Brand.gradient) : AnyShapeStyle(Color.secondary))
            .symbolEffect(.bounce, value: on)
        case 1:
            if let hello = GuideSign.all.first(where: { $0.name == "HELLO" }) {
                SignDiagram(sign: hello, animate: on, start: since).padding(4)
            } else {
                Image(systemName: "hand.wave").font(.system(size: 40)).symbolEffect(.bounce, value: on)
            }
        case 2:
            VStack(spacing: 8) {
                Chip(text: "HELLO").scaleEffect(0.8)
                Image(systemName: "arrow.down").foregroundStyle(.secondary).font(.caption)
                Text("“Hello!”").font(.system(.title3, design: .rounded).weight(.semibold))
                    .opacity(on ? 1 : 0.25)
                    .animation(.easeIn(duration: 0.5).delay(on ? 0.6 : 0), value: on)
            }
        default:
            VStack(spacing: 6) {
                Image(systemName: "hand.thumbsup.fill").font(.system(size: 40))
                    .foregroundStyle(on ? AnyShapeStyle(Color.green) : AnyShapeStyle(Color.secondary))
                    .symbolEffect(.bounce, value: on)
                Label("Sent", systemImage: "checkmark.circle.fill").font(.caption.weight(.semibold))
                    .foregroundStyle(.green).opacity(on ? 1 : 0)
                    .animation(.easeIn(duration: 0.3).delay(on ? 0.7 : 0), value: on)
            }
        }
    }
}

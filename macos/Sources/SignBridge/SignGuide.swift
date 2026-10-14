import SwiftUI

/// Sign guide data from export_sign_guide.py: one recorded take per sign, normalized to the shoulders
/// (midpoint 0,0, width 1, y down) and mirrored like the live preview.
struct GuideSign: Decodable, Identifiable {
    struct Frame: Decodable {
        let body: [[Double]]  // nose, eyes (2), mouth corners (2), shoulders L/R, elbows L/R, wrists L/R
        let hands: [[[Double]]?]  // left, right: 21 points each, or nil when the hand is down
    }
    let name: String, title: String, how: String
    let fps: Double, mover: Int
    let frames: [Frame]
    var id: String { name }

    static let all: [GuideSign] = {
        let bundled = Bundle.main.url(forResource: "signs", withExtension: "json")
        let dev = URL(fileURLWithPath: NSHomeDirectory() + "/code/signbridge/macos/Assets/signs.json")
        guard let data = try? Data(contentsOf: bundled ?? dev),
              let guide = try? JSONDecoder().decode([String: [GuideSign]].self, from: data) else { return [] }
        return guide["signs"] ?? []
    }()

    /// Center of the moving hand in each frame it shows: the motion arrow.
    var path: [CGPoint] {
        frames.compactMap { f in
            guard let h = f.hands[mover] else { return nil }
            return CGPoint(x: h.map { $0[0] }.reduce(0, +) / 21, y: h.map { $0[1] }.reduce(0, +) / 21)
        }
    }
}

private let handBones: [(Int, Int)] = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                                       (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                                       (18, 19), (19, 20), (0, 17)]

/// A looping skeleton replay of one sign: faint start pose, motion arrow, and the moving figure.
struct SignDiagram: View {
    let sign: GuideSign
    var animate = true
    private let hold = 0.9  // seconds to rest on the last frame before looping

    var body: some View {
        TimelineView(.animation(paused: !animate)) { context in
            Canvas { ctx, size in
                draw(&ctx, size, frame: frameIndex(context.date))
            }
        }
        .aspectRatio(1.1, contentMode: .fit)
    }

    private func frameIndex(_ date: Date) -> Int {
        guard animate else { return sign.frames.count - 1 }
        let length = Double(sign.frames.count) / sign.fps + hold
        let t = date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: length)
        return min(sign.frames.count - 1, Int(t * sign.fps))
    }

    private func draw(_ ctx: inout GraphicsContext, _ size: CGSize, frame index: Int) {
        // view box in shoulder widths: wide enough for hands at the forehead and out to the side
        let box = CGRect(x: -1.45, y: -1.55, width: 2.9, height: 2.65)
        let scale = min(size.width / box.width, size.height / box.height)
        let origin = CGPoint(x: (size.width - box.width * scale) / 2 - box.minX * scale,
                             y: (size.height - box.height * scale) / 2 - box.minY * scale)
        func p(_ v: [Double]) -> CGPoint { CGPoint(x: origin.x + v[0] * scale, y: origin.y + v[1] * scale) }
        func p(_ v: CGPoint) -> CGPoint { CGPoint(x: origin.x + v.x * scale, y: origin.y + v.y * scale) }

        let frame = sign.frames[index], first = sign.frames[0]

        // body as one flat shape (one layer, so overlapping parts don't darken)
        let b = frame.body
        var figure = ctx
        figure.opacity = 0.17
        figure.drawLayer { layer in
            let (ls, rs) = (p(b[5]), p(b[6]))
            var torso = Path()
            torso.move(to: ls)
            torso.addLine(to: rs)
            torso.addLine(to: CGPoint(x: rs.x + 0.1 * scale, y: size.height + 20))
            torso.addLine(to: CGPoint(x: ls.x - 0.1 * scale, y: size.height + 20))
            torso.closeSubpath()
            layer.fill(torso, with: .color(.primary))
            let head = p([(b[0][0] + b[1][0] + b[2][0]) / 3, (b[0][1] + b[1][1] + b[2][1]) / 3 + 0.04])
            let r = 0.31 * scale
            layer.fill(Path(ellipseIn: CGRect(x: head.x - r, y: head.y - r * 1.15, width: 2 * r, height: 2.3 * r)),
                       with: .color(.primary))
            var neck = Path()
            neck.move(to: head)
            neck.addLine(to: CGPoint(x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2))
            layer.stroke(neck, with: .color(.primary), lineWidth: 0.2 * scale)
            for side in 0..<2 {
                var arm = Path()
                arm.move(to: p(b[5 + side]))
                arm.addLine(to: p(b[7 + side]))
                arm.addLine(to: frame.hands[side].map { p($0[0]) } ?? p(b[9 + side]))
                layer.stroke(arm, with: .color(.primary), style: StrokeStyle(lineWidth: 0.22 * scale, lineCap: .round,
                                                                             lineJoin: .round))
            }
        }
        var mouth = Path()
        mouth.move(to: p(b[3]))
        mouth.addLine(to: p(b[4]))
        ctx.stroke(mouth, with: .color(Color.primary.opacity(0.4)), style: StrokeStyle(lineWidth: 2, lineCap: .round))

        // ghost of the moving hand where the sign starts
        if let start = first.hands[sign.mover] {
            stroke(hand: start, &ctx, p, color: .color(Color.primary.opacity(0.22)), width: 1.5, dash: [3, 3])
        }

        // the hands now: the signing hand in the brand gradient
        for side in 0..<2 {
            guard let hand = frame.hands[side] else { continue }
            let shading: GraphicsContext.Shading = side == sign.mover
                ? .linearGradient(Gradient(colors: [Brand.violet, Brand.coral]), startPoint: p(hand[0]), endPoint: p(hand[12]))
                : .color(Color.primary.opacity(0.55))
            stroke(hand: hand, &ctx, p, color: shading, width: 2.4)
            for tip in [4, 8, 12, 16, 20] {
                let c = p(hand[tip])
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 2.2, y: c.y - 2.2, width: 4.4, height: 4.4)), with: shading)
            }
        }

        // motion arrow along the hand's path
        let path = sign.path.map(p)
        if path.count > 1 {
            var trail = Path()
            trail.addLines(path)
            let gradient = Gradient(colors: [Color.primary.opacity(0.2), Color.primary.opacity(0.9)])
            var arrow = ctx
            arrow.addFilter(.shadow(color: .black.opacity(0.6), radius: 2))
            arrow.stroke(trail, with: .linearGradient(gradient, startPoint: path.first!, endPoint: path.last!),
                         style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round, dash: [1, 6]))
            arrowhead(&arrow, path, color: Color.primary.opacity(0.95))
        }
    }

    private func stroke(hand: [[Double]], _ ctx: inout GraphicsContext, _ p: ([Double]) -> CGPoint,
                        color: GraphicsContext.Shading, width: CGFloat, dash: [CGFloat] = []) {
        var bones = Path()
        for (a, b) in handBones {
            bones.move(to: p(hand[a]))
            bones.addLine(to: p(hand[b]))
        }
        ctx.stroke(bones, with: color, style: StrokeStyle(lineWidth: width, lineCap: .round, dash: dash))
    }

    private func arrowhead(_ ctx: inout GraphicsContext, _ path: [CGPoint], color: Color) {
        // direction from a few frames back, so jitter at the very end doesn't twist the arrow
        let tip = path.last!, from = path[max(0, path.count - 5)]
        let angle = atan2(tip.y - from.y, tip.x - from.x)
        let size: CGFloat = 11
        var head = Path()
        head.move(to: CGPoint(x: tip.x + cos(angle) * size * 0.6, y: tip.y + sin(angle) * size * 0.6))
        head.addLine(to: CGPoint(x: tip.x + cos(angle + 2.5) * size, y: tip.y + sin(angle + 2.5) * size))
        head.addLine(to: CGPoint(x: tip.x + cos(angle - 2.5) * size, y: tip.y + sin(angle - 2.5) * size))
        head.closeSubpath()
        ctx.fill(head, with: .color(color))
    }
}

/// One card per sign: the diagram, its name, and how to make it.
struct SignCard: View {
    let sign: GuideSign
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SignDiagram(sign: sign)
                .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
            Text(sign.name).font(.system(.title3, design: .rounded).weight(.bold))
            Text(sign.title).font(.subheadline.weight(.semibold)).foregroundStyle(Brand.gradient)
            Text(sign.how).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .padding(12)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 14))
    }
}

/// All signs SignBridge knows, as a grid of animated cards.
struct SignGuideView: View {
    var columns = 2
    var body: some View {
        if GuideSign.all.isEmpty {
            Text("The sign guide isn't in this build.").foregroundStyle(.secondary)
        } else {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 14, alignment: .top), count: columns),
                      spacing: 14) {
                ForEach(GuideSign.all) { SignCard(sign: $0) }
            }
        }
    }
}

/// The guide in its own window, opened from the menu bar or the preview.
struct SignGuideWindow: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 12) {
                    Logo(size: 40)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Signs SignBridge knows").font(.title2.bold())
                        Text("Shown mirrored, like the preview. Copy the colored hand; the arrow shows where it moves.")
                            .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(.bottom, 12)
                SignGuideView(columns: 2)
            }
            .padding(22)
        }
        .tint(Brand.violet)
    }
}

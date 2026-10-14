import AppKit
import ApplicationServices
import AVFoundation
import SwiftUI

enum SetupStep: Int, CaseIterable, Identifiable {
    case welcome, camera, accessibility, claude, siri, voice, framing, gestures, signs, done
    var id: Int { rawValue }
    var title: String {
        switch self {
        case .welcome: "Welcome"
        case .camera: "Camera"
        case .accessibility: "Accessibility"
        case .claude: "Claude"
        case .siri: "Siri"
        case .voice: "Voice"
        case .framing: "Position"
        case .gestures: "Gestures"
        case .signs: "Signs"
        case .done: "Done"
        }
    }
    var icon: String {
        switch self {
        case .welcome: "hand.wave"
        case .camera: "camera"
        case .accessibility: "accessibility"
        case .claude: "sparkles"
        case .siri: "mic.circle"
        case .voice: "speaker.wave.2"
        case .framing: "person.crop.rectangle"
        case .gestures: "hand.thumbsup"
        case .signs: "book"
        case .done: "checkmark.seal"
        }
    }
}

func openSettingsPane(_ anchor: String) {
    if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)") {
        NSWorkspace.shared.open(url)
    }
}

/// First-run setup wizard. Each step checks something and explains how to fix it.
struct SetupView: View {
    @ObservedObject var engine: EngineClient
    let finish: () -> Void
    @State private var step: SetupStep = .welcome
    @State private var completed: Set<SetupStep> = []

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 10) {
                    Logo(size: 38)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("SignBridge").font(.headline)
                        Text("Setup").font(.subheadline).foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 16).padding(.top, 40).padding(.bottom, 6)
                List(SetupStep.allCases, selection: Binding(get: { step }, set: { if let s = $0 { go(s) } })) { s in
                    Label {
                        Text(s.title)
                    } icon: {
                        Image(systemName: completed.contains(s) ? "checkmark.circle.fill" : s.icon)
                            .foregroundStyle(completed.contains(s) ? .green : .secondary)
                    }
                    .tag(s)
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
                VStack(alignment: .leading, spacing: 6) {
                    let total = SetupStep.allCases.count - 2
                    let finished = completed.subtracting([.welcome, .done]).count
                    Text("\(finished) of \(total) done").font(.caption).foregroundStyle(.secondary)
                    ProgressView(value: Double(finished), total: Double(total)).tint(Brand.magenta)
                }
                .padding(16)
            }
            .frame(width: 200)
            .background(SidebarBackground())

            VStack(alignment: .leading, spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) { page }
                        .padding(28).padding(.top, 16).frame(maxWidth: .infinity, alignment: .leading)
                }
                Divider()
                HStack {
                    if step != .welcome { Button("Back") { go(SetupStep(rawValue: step.rawValue - 1)!) } }
                    Spacer()
                    if step == .done {
                        Button("Start SignBridge") {
                            engine.saveConfig(["onboarded": true])
                            finish()
                        }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .keyboardShortcut(.defaultAction)
                    } else {
                        Button(completed.contains(step) || step == .welcome ? "Continue" : "Skip for now") {
                            if step == .welcome { completed.insert(.welcome) }
                            go(SetupStep(rawValue: step.rawValue + 1)!)
                        }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .keyboardShortcut(.defaultAction)
                    }
                }
                .padding(16)
            }
        }
        .tint(Brand.violet)
        .ignoresSafeArea(.container, edges: .top)
        .onAppear { go(.welcome) }
    }

    private func go(_ s: SetupStep) {
        step = s
        // Camera steps need the engine's camera and preview; others don't.
        let live = s == .framing || s == .gestures
        engine.setMode(s == .framing ? "framing" : s == .gestures ? "gestures" : "off")
        engine.setPreview(live)
    }

    private func done(_ s: SetupStep, _ ok: Bool) {
        if ok { completed.insert(s) } else { completed.remove(s) }
    }

    @ViewBuilder private var page: some View {
        switch step {
        case .welcome: WelcomePage()
        case .camera: CameraPage(done: { done(.camera, $0) })
        case .accessibility: AccessibilityPage(done: { done(.accessibility, $0) })
        case .claude: ClaudePage(engine: engine, done: { done(.claude, $0) })
        case .siri: SiriPage(engine: engine, done: { done(.siri, $0) })
        case .voice: VoicePage(engine: engine, done: { done(.voice, $0) })
        case .framing: FramingPage(engine: engine, done: { done(.framing, $0) })
        case .gestures: GesturesPage(engine: engine, done: { done(.gestures, $0) })
        case .signs: SignsPage(done: { done(.signs, true) })
        case .done: DonePage(skipped: SetupStep.allCases.filter { ![.welcome, .done].contains($0) && !completed.contains($0) })
        }
    }
}

// MARK: pages

struct PageHeader: View {
    let title: String, subtitle: String
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.largeTitle.bold())
            Text(subtitle).font(.title3).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .padding(.bottom, 16)
    }
}

struct StatusLine: View {
    let ok: Bool?, text: String
    var body: some View {
        HStack(alignment: .top) {
            Image(systemName: ok == nil ? "circle.dotted" : ok! ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(ok == nil ? Color.secondary : ok! ? Color.green : Color.orange)
            Text(text).fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 4)
    }
}

struct WelcomePage: View {
    var body: some View {
        HStack(spacing: 18) {
            Logo(size: 88)
            VStack(alignment: .leading, spacing: 4) {
                GradientText("SignBridge", font: .system(size: 40, weight: .bold, design: .rounded))
                Text("Sign to your Mac. Ask Siri, type messages, speak in calls.")
                    .font(.title3).foregroundStyle(.secondary)
            }
        }
        .padding(.bottom, 18)
        Storyboard()
        Label("Video never leaves this Mac. Only the recognized words go to Claude.", systemImage: "lock.fill")
            .foregroundStyle(.secondary).padding(.top, 10)
        Text("Setup takes about 3 minutes.").foregroundStyle(.secondary).padding(.top, 14)
    }
}

struct CameraPage: View {
    let done: (Bool) -> Void
    @State private var status = AVCaptureDevice.authorizationStatus(for: .video)
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    var body: some View {
        PageHeader(title: "Camera", subtitle: "SignBridge watches your hands. Video stays on your Mac.")
        switch status {
        case .authorized:
            StatusLine(ok: true, text: "Camera access is on.")
        case .notDetermined:
            Button("Allow Camera Access") {
                AVCaptureDevice.requestAccess(for: .video) { _ in
                    DispatchQueue.main.async { refresh() }
                }
            }
            .controlSize(.large)
        default:
            StatusLine(ok: false, text: "Camera access is off. Turn on SignBridge in Privacy & Security > Camera.")
            Button("Open Camera Settings") { openSettingsPane("Privacy_Camera") }
        }
        Color.clear.frame(height: 1).onAppear { refresh() }.onReceive(timer) { _ in refresh() }
    }
    private func refresh() {
        let now = AVCaptureDevice.authorizationStatus(for: .video)
        if now != status { status = now }
        done(status == .authorized)
    }
}

struct AccessibilityPage: View {
    let done: (Bool) -> Void
    @State private var trusted = AXIsProcessTrusted()
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    var body: some View {
        PageHeader(title: "Accessibility",
                   subtitle: "Lets SignBridge type your message into Siri and text fields, like a keyboard would.")
        if trusted {
            StatusLine(ok: true, text: "Accessibility access is on.")
        } else {
            StatusLine(ok: false, text: "Turn on SignBridge in Privacy & Security > Accessibility. This page updates by itself.")
            Button("Open Accessibility Settings") {
                let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
                _ = AXIsProcessTrustedWithOptions(options)
                openSettingsPane("Privacy_Accessibility")
            }
            .controlSize(.large)
        }
        Color.clear.frame(height: 1)
            .onAppear { done(trusted) }
            .onReceive(timer) { _ in
                trusted = AXIsProcessTrusted()
                done(trusted)
            }
    }
}

struct ClaudePage: View {
    @ObservedObject var engine: EngineClient
    let done: (Bool) -> Void
    var body: some View {
        PageHeader(title: "Claude",
                   subtitle: "Claude turns the signs you make into a natural sentence and picks where it goes.")
        Text("SignBridge uses your Claude Code sign-in. If the check fails, open Terminal, run `claude`, and sign in with /login.")
            .fixedSize(horizontal: false, vertical: true)
        Button("Check Connection") { engine.check("claude") }.controlSize(.large).padding(.vertical, 8)
        if let r = engine.checks["claude"] {
            StatusLine(ok: r.ok, text: r.detail)
        } else if engine.running {
            Text("Press the button to test (takes a few seconds).").foregroundStyle(.secondary)
        }
        Color.clear.frame(height: 1).onChange(of: engine.checks["claude"]?.ok) { _, ok in done(ok ?? false) }
    }
}

struct SiriPage: View {
    @ObservedObject var engine: EngineClient
    let done: (Bool) -> Void
    @State private var sent = false
    @State private var trusted = AXIsProcessTrusted()
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    var body: some View {
        PageHeader(title: "Siri", subtitle: "SignBridge asks Siri your questions by typing them for you.")
        VStack(alignment: .leading, spacing: 8) {
            Text("1. In Apple Intelligence & Siri settings, set Keyboard Shortcut to **Hold Command Space**.")
            Text("2. In Accessibility > Siri, turn on **Type to Siri**.")
            Button("Open Siri Settings") {
                NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.Siri-Settings.extension")!)
            }
        }
        Divider().padding(.vertical, 12)
        if !trusted {
            StatusLine(ok: false, text: "Turn on Accessibility for SignBridge first (the Accessibility step).")
        }
        Button("Ask Siri \"What time is it?\"") {
            sent = true
            engine.testRoute("siri", text: "What time is it?")
        }
        .controlSize(.large)
        .disabled(!trusted)
        .onReceive(timer) { _ in trusted = AXIsProcessTrusted() }
        RouteResult(engine: engine, route: "siri", okText: "Sent to Siri.")
        if sent, engine.lastRouteTest?.ok == true {
            Text("Did Siri answer?").padding(.top, 8)
            HStack {
                Button("Yes, it answered") { done(true) }
                Button("No") { done(false) }
            }
        }
    }
}

struct VoicePage: View {
    @ObservedObject var engine: EngineClient
    let done: (Bool) -> Void
    var body: some View {
        PageHeader(title: "Voice in calls (optional)",
                   subtitle: "SignBridge can speak for you in FaceTime, Zoom or Discord through a virtual microphone.")
        if let r = engine.checks["voice"] {
            StatusLine(ok: r.ok, text: r.detail)
            if r.ok {
                Text("In your call app, choose **BlackHole 2ch** as the microphone.").padding(.top, 6)
                Button("Play a test phrase into it") { engine.testRoute("voice", text: "SignBridge voice test.") }
                RouteResult(engine: engine, route: "voice",
                            okText: "Played. You won't hear it here; someone in your call would.")
            } else {
                Text("Install the free BlackHole driver: open Terminal and run").padding(.top, 6)
                Text("brew install blackhole-2ch").font(.body.monospaced()).textSelection(.enabled)
                    .padding(8).background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
                Button("Check Again") { engine.check("voice") }
            }
        } else {
            ProgressView().onAppear { engine.check("voice") }
        }
        Color.clear.frame(height: 1).onChange(of: engine.checks["voice"]?.ok) { _, ok in done(ok ?? false) }
    }
}

/// The result of the last test send for one route.
struct RouteResult: View {
    @ObservedObject var engine: EngineClient
    let route: String, okText: String
    var body: some View {
        if let r = engine.lastRouteTest, r.route == route {
            StatusLine(ok: r.ok, text: r.ok ? okText : (r.detail.isEmpty ? "That didn't work." : r.detail))
        }
    }
}

struct LivePreview: View {
    @ObservedObject var engine: EngineClient
    var maxHeight: CGFloat = 340
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10).fill(.black)
            if let image = engine.frame {
                Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            } else {
                if let error = engine.engineError {
                    VStack(spacing: 10) {
                        Image(systemName: "video.slash").font(.system(size: 34)).foregroundStyle(Brand.gradient)
                        Text(error).foregroundStyle(.white).multilineTextAlignment(.center)
                            .fixedSize(horizontal: false, vertical: true)
                        if error.localizedCaseInsensitiveContains("camera") {
                            Button("Open Camera Settings") { openSettingsPane("Privacy_Camera") }
                        }
                    }
                    .padding(24)
                } else {
                    VStack {
                        ProgressView()
                        Text("Starting the camera…").foregroundStyle(.white).padding(.top, 6)
                    }
                }
            }
        }
        .aspectRatio(4 / 3, contentMode: .fit)
        .frame(maxHeight: maxHeight)
    }
}

struct FramingPage: View {
    @ObservedObject var engine: EngineClient
    let done: (Bool) -> Void
    var body: some View {
        PageHeader(title: "Your position", subtitle: "Sit where you'll sign. Your head, shoulders and chest should show.")
        LivePreview(engine: engine)
        StatusLine(ok: engine.framingHint.isEmpty ? nil : engine.framingOK, text: engine.framingHint.isEmpty
                   ? "Looking for you…" : engine.framingHint)
        HStack {
            Button("Save This Position") { engine.calibrate() }.controlSize(.large).disabled(!engine.framingOK)
            if let msg = engine.lastCalibration { Text(msg).foregroundStyle(.secondary) }
        }
        Text("Rest your hands below the orange line (or out of view) to finish a phrase.")
            .foregroundStyle(.secondary).padding(.top, 4)
        Color.clear.frame(height: 1).onChange(of: engine.lastCalibration) { _, msg in done(msg?.hasPrefix("Saved") ?? false) }
    }
}

struct GesturesPage: View {
    @ObservedObject var engine: EngineClient
    let done: (Bool) -> Void
    private let info: [(String, String, String)] = [
        ("wake", "Start", "Both open palms facing the camera, apart, held still"),
        ("confirm", "Send", "Thumbs up, held still, other hand down"),
        ("cancel", "Cancel", "Thumbs down, held still"),
    ]
    var body: some View {
        PageHeader(title: "Control gestures", subtitle: "Try each one. Hold it still for about a second.")
        HStack(alignment: .top, spacing: 20) {
            LivePreview(engine: engine)
            VStack(alignment: .leading, spacing: 14) {
                ForEach(info, id: \.0) { name, title, how in
                    let isDone = engine.practiceDone.contains(name)
                    let isNext = engine.practiceTodo.first == name
                    HStack(alignment: .top) {
                        Image(systemName: isDone ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(isDone ? .green : .secondary).font(.title2)
                        VStack(alignment: .leading) {
                            Text(title).font(.headline)
                            Text(how).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                            if isNext, let h = engine.holding, h.0 == name {
                                ProgressView(value: h.1).frame(width: 140)
                            }
                        }
                    }
                    .opacity(isDone || isNext ? 1 : 0.5)
                }
            }
            .frame(width: 230)
        }
        Color.clear.frame(height: 1).onChange(of: engine.practiceTodo.isEmpty && !engine.practiceDone.isEmpty) { _, all in
            if all { done(true) }
        }
    }
}

struct SignsPage: View {
    let done: () -> Void
    var body: some View {
        PageHeader(title: "Signs to try",
                   subtitle: "SignBridge knows these signs so far. Copy the colored hand; the arrow shows its path.")
        SignGuideView(columns: 2).onAppear(perform: done)
        Text("Open this guide any time from the menu bar: Sign Guide.").foregroundStyle(.secondary).padding(.top, 8)
    }
}

struct DonePage: View {
    let skipped: [SetupStep]
    var body: some View {
        PageHeader(title: "You're set", subtitle: "SignBridge lives in the menu bar (the hand icon).")
        VStack(alignment: .leading, spacing: 10) {
            Label("Click into the app you want to use (Messages, Notes, a call), then show both palms.", systemImage: "1.circle")
            Label("Sign, then rest your hands.", systemImage: "2.circle")
            Label("Check the sentence, then thumbs up to send or thumbs down to cancel.", systemImage: "3.circle")
        }
        .font(.title3)
        if !skipped.isEmpty {
            Text("Not finished: \(skipped.map(\.title).joined(separator: ", ")). You can run setup again from the menu bar.")
                .foregroundStyle(.orange).padding(.top, 16)
        }
    }
}

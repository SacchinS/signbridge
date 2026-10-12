import AppKit
import ApplicationServices
import AVFoundation
import SwiftUI

enum SetupStep: Int, CaseIterable, Identifiable {
    case welcome, camera, accessibility, claude, siri, voice, framing, gestures, done
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
            .frame(width: 190)

            VStack(alignment: .leading, spacing: 0) {
                ScrollView {
                    page.padding(28).frame(maxWidth: .infinity, alignment: .leading)
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
                        .keyboardShortcut(.defaultAction)
                    } else {
                        Button(completed.contains(step) || step == .welcome ? "Continue" : "Skip for now") {
                            if step == .welcome { completed.insert(.welcome) }
                            go(SetupStep(rawValue: step.rawValue + 1)!)
                        }
                        .keyboardShortcut(.defaultAction)
                    }
                }
                .padding(16)
            }
        }
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
        PageHeader(title: "Welcome to SignBridge",
                   subtitle: "Sign ASL to your webcam to ask Siri, type messages, and speak in calls.")
        VStack(alignment: .leading, spacing: 14) {
            Label("Show both open palms to start. Sign, then rest your hands.", systemImage: "hand.raised")
            Label("SignBridge writes what you signed as a sentence and shows it to you.", systemImage: "text.bubble")
            Label("Thumbs up sends it. Thumbs down cancels. Nothing is sent without your thumbs up.",
                  systemImage: "hand.thumbsup")
            Label("Your camera video never leaves this Mac. Only the recognized words are sent to Claude.",
                  systemImage: "lock")
        }
        .font(.title3)
        Text("Setup takes about 3 minutes.").foregroundStyle(.secondary).padding(.top, 20)
    }
}

struct CameraPage: View {
    let done: (Bool) -> Void
    @State private var status = AVCaptureDevice.authorizationStatus(for: .video)
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
        Color.clear.frame(height: 1).onAppear { refresh() }
    }
    private func refresh() {
        status = AVCaptureDevice.authorizationStatus(for: .video)
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
        Button("Ask Siri \"What time is it?\"") {
            sent = true
            engine.testRoute("siri", text: "What time is it?")
        }
        .controlSize(.large)
        if sent {
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

struct LivePreview: View {
    @ObservedObject var engine: EngineClient
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10).fill(.black)
            if let image = engine.frame {
                Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            } else {
                VStack {
                    ProgressView()
                    Text(engine.engineError ?? "Starting the camera…").foregroundStyle(.white).padding(.top, 6)
                }
            }
        }
        .aspectRatio(4 / 3, contentMode: .fit)
        .frame(maxHeight: 340)
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

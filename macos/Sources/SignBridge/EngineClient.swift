import AppKit
import Foundation

/// One finished phrase, for the demo's history.
struct HistoryItem: Identifiable {
    let id = UUID()
    let glosses: [String], sentence: String, route: String, sent: Bool, date: Date
}

/// Launches the Python engine (signbridge.server) and talks JSON lines with it.
/// Inside the app bundle it uses the bundled Python; in development it uses the repo's .venv.
@MainActor
final class EngineClient: ObservableObject {
    // Status mirrored from the engine
    @Published var running = false
    @Published var state = "idle"
    @Published var mode = "off"
    @Published var glosses: [String] = []
    @Published var sentence: String?
    @Published var route: String?
    @Published var log: [String] = []
    @Published var holding: (String, Double)? = nil
    @Published var target: String?
    @Published var fps = 0.0
    @Published var engineError: String?
    @Published var framingHint = ""
    @Published var framingOK = false
    @Published var practiceDone: [String] = []
    @Published var practiceTodo: [String] = []
    @Published var frame: NSImage?
    @Published var config: [String: Any] = [:]
    @Published var checks: [String: (ok: Bool, detail: String)] = [:]
    @Published var lastCalibration: String?
    @Published var lastRouteTest: (route: String, ok: Bool, detail: String)?
    @Published var history: [HistoryItem] = []  // newest first
    private var pending: (glosses: [String], sentence: String, route: String)?

    private var process: Process?
    private var input: FileHandle?
    private var buffer = Data()
    private var stopping = false

    var onboarded: Bool { (config["onboarded"] as? Bool) ?? false }

    // MARK: launching

    static func locate() -> (python: URL, resources: URL, pythonPath: String)? {
        let fm = FileManager.default
        if let res = Bundle.main.resourceURL {
            let py = res.appendingPathComponent("python/bin/python3")
            if fm.isExecutableFile(atPath: py.path) {
                let path = [res.appendingPathComponent("engine").path, res.appendingPathComponent("site-packages").path]
                return (py, res, path.joined(separator: ":"))
            }
        }
        // Development: SIGNBRIDGE_DEV_ROOT or ~/code/signbridge
        let root = ProcessInfo.processInfo.environment["SIGNBRIDGE_DEV_ROOT"]
            .map { URL(fileURLWithPath: $0) }
            ?? fm.homeDirectoryForCurrentUser.appendingPathComponent("code/signbridge")
        let py = root.appendingPathComponent(".venv/bin/python")
        guard fm.isExecutableFile(atPath: py.path) else { return nil }
        return (py, root, root.path)
    }

    func start() {
        guard process == nil else { return }
        guard let loc = Self.locate() else {
            engineError = "Couldn't find the SignBridge engine. Reinstall the app."
            return
        }
        let p = Process()
        p.executableURL = loc.python
        p.arguments = ["-m", "signbridge.server"]
        p.currentDirectoryURL = loc.resources
        var env = ProcessInfo.processInfo.environment
        env["SIGNBRIDGE_RESOURCES"] = loc.resources.path
        env["PYTHONPATH"] = loc.pythonPath
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = (env["PATH"] ?? "/usr/bin:/bin") + ":/opt/homebrew/bin:/usr/local/bin:" +
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".local/bin").path
        p.environment = env
        let inPipe = Pipe(), outPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = FileHandle(forWritingAtPath: Self.logPath()) ?? FileHandle.standardError
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            Task { @MainActor in self?.receive(data) }
        }
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in self?.ended(code: proc.terminationStatus) }
        }
        do {
            try p.run()
        } catch {
            engineError = "Couldn't start the engine: \(error.localizedDescription)"
            return
        }
        process = p
        input = inPipe.fileHandleForWriting
        running = true
        engineError = nil
    }

    static func logPath() -> String {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/SignBridge")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("engine.log")
        FileManager.default.createFile(atPath: file.path, contents: nil)
        return file.path
    }

    func stop() {
        stopping = true
        send(["cmd": "quit"])
        process?.waitUntilExit()
    }

    private func ended(code: Int32) {
        process = nil
        input = nil
        running = false
        if !stopping {
            engineError = "The engine stopped (code \(code)). Restarting… Details: ~/Library/Logs/SignBridge/engine.log"
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in self?.start() }
        }
    }

    // MARK: commands

    func send(_ command: [String: Any]) {
        guard let input, let data = try? JSONSerialization.data(withJSONObject: command) else { return }
        input.write(data + Data([0x0A]))
    }

    func setMode(_ mode: String) { send(["cmd": "mode", "mode": mode]) }
    func setPreview(_ on: Bool) {
        send(["cmd": "preview", "on": on])
        if !on { frame = nil }
    }
    func gesture(_ name: String) { send(["cmd": "gesture", "name": name]) }
    func fake(_ glosses: [String]) { send(["cmd": "fake", "glosses": glosses]) }
    func check(_ name: String) {
        checks[name] = nil
        send(["cmd": "check", "name": name])
    }
    func testRoute(_ route: String, text: String) {
        lastRouteTest = nil
        send(["cmd": "test_route", "route": route, "text": text])
    }
    func calibrate() { send(["cmd": "calibrate"]) }
    func saveConfig(_ values: [String: Any]) { send(["cmd": "config", "values": values]) }

    // MARK: events

    private func receive(_ data: Data) {
        if data.isEmpty { return }
        buffer.append(data)
        while let nl = buffer.firstIndex(of: 0x0A) {
            let line = buffer.subdata(in: buffer.startIndex..<nl)
            buffer.removeSubrange(buffer.startIndex...nl)
            if let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any] {
                handle(obj)
            }
        }
    }

    private func handle(_ e: [String: Any]) {
        switch e["type"] as? String {
        case "ready":
            config = e["config"] as? [String: Any] ?? [:]
        case "config":
            config = e["config"] as? [String: Any] ?? config
        case "status":
            let newState = e["state"] as? String ?? "idle"
            if newState == "confirming", let intent = e["intent"] as? [String: Any],
               let text = intent["sentence"] as? String {
                pending = (e["glosses"] as? [String] ?? [], text, intent["route"] as? String ?? "")
            }
            if newState == "idle", state == "sending" || state == "confirming", let p = pending {
                history.insert(HistoryItem(glosses: p.glosses, sentence: p.sentence, route: p.route,
                                           sent: state == "sending", date: Date()), at: 0)
                history = Array(history.prefix(20))
                pending = nil
            }
            state = newState
            mode = e["mode"] as? String ?? "off"
            glosses = e["glosses"] as? [String] ?? []
            let intent = e["intent"] as? [String: Any]
            sentence = intent?["sentence"] as? String
            route = intent?["route"] as? String
            log = e["log"] as? [String] ?? []
            target = e["target"] as? String
            fps = e["fps"] as? Double ?? 0
            engineError = e["error"] as? String
            if let h = e["holding"] as? [Any], let name = h.first as? String, let prog = h.last as? Double {
                holding = (name, prog)
            } else {
                holding = nil
            }
            let framing = e["framing"] as? [String: Any]
            framingHint = framing?["hint"] as? String ?? ""
            framingOK = framing?["ok"] as? Bool ?? false
            let practice = e["practice"] as? [String: Any]
            practiceDone = practice?["done"] as? [String] ?? []
            practiceTodo = practice?["todo"] as? [String] ?? []
        case "frame":
            if let b64 = e["jpeg"] as? String, let data = Data(base64Encoded: b64) {
                frame = NSImage(data: data)
            }
        case "check":
            if let name = e["name"] as? String {
                checks[name] = (e["ok"] as? Bool ?? false, e["detail"] as? String ?? "")
            }
        case "calibrated":
            lastCalibration = e["message"] as? String
        case "route_tested":
            lastRouteTest = (e["route"] as? String ?? "", e["ok"] as? Bool ?? false, e["detail"] as? String ?? "")
        case "error":
            engineError = e["message"] as? String
        default:
            break
        }
    }
}

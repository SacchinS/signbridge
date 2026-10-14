import AppKit
import AVFoundation
import SwiftUI

/// Menu-bar app: a status item with a menu, plus windows for setup, the live preview and settings.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let engine = EngineClient()
    private var statusItem: NSStatusItem!
    private var windows: [String: NSWindow] = [:]
    private var listening = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.image = Brand.menuBarImage(listening: false)
        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu
        engine.start()
        // First launch: walk through setup once the engine has reported its config.
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) { [weak self] in
            guard let self else { return }
            if !self.engine.onboarded { self.showSetup() }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        engine.stop()
    }

    // Rebuild the menu each time it opens so it shows the current state.
    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        let states = ["idle": "Watching for open palms", "listening": "Listening to your signs",
                      "thinking": "Writing your sentence", "confirming": "Waiting for your thumbs up",
                      "sending": "Sending"]
        let state = engine.running ? (listening ? states[engine.state] ?? engine.state : "Paused") : "Starting…"
        let header = NSMenuItem(title: "", action: nil, keyEquivalent: "")
        header.attributedTitle = NSAttributedString(
            string: "SignBridge\n", attributes: [.font: NSFont.boldSystemFont(ofSize: 13)])
            .appending(NSAttributedString(string: state, attributes: [
                .font: NSFont.systemFont(ofSize: 12), .foregroundColor: NSColor.secondaryLabelColor]))
        menu.addItem(header)
        if let error = engine.engineError {
            menu.addItem(NSMenuItem(title: "⚠︎ \(error.prefix(60))", action: nil, keyEquivalent: ""))
        }
        menu.addItem(.separator())
        menu.addItem(item(listening ? "Pause SignBridge" : "Start SignBridge", #selector(toggleListening), "s"))
        menu.addItem(item("Show Preview", #selector(showPreview), "p"))
        menu.addItem(.separator())
        menu.addItem(item("Run Setup…", #selector(showSetup), ""))
        menu.addItem(item("Settings…", #selector(showSettings), ","))
        menu.addItem(.separator())
        menu.addItem(item("Quit SignBridge", #selector(quit), "q"))
    }

    private func item(_ title: String, _ action: Selector, _ key: String) -> NSMenuItem {
        let i = NSMenuItem(title: title, action: action, keyEquivalent: key)
        i.target = self
        return i
    }

    @objc func toggleListening() {
        listening.toggle()
        engine.setMode(listening ? "run" : "off")
        statusItem.button?.image = Brand.menuBarImage(listening: listening)
    }

    func startListening() {
        if !listening { toggleListening() }
    }

    @objc func showPreview() {
        if !listening { startListening() }
        engine.setPreview(true)
        open("preview", title: "SignBridge", size: NSSize(width: 720, height: 600),
             view: PreviewView(engine: engine), seamless: true) { [weak self] in self?.engine.setPreview(false) }
        windows["preview"]?.appearance = NSAppearance(named: .darkAqua)
    }

    @objc func showSetup() {
        if listening { toggleListening() }
        open("setup", title: "Set up SignBridge", size: NSSize(width: 820, height: 620),
             view: SetupView(engine: engine, finish: { [weak self] in
                 self?.windows["setup"]?.close()
                 self?.startListening()
                 self?.showPreview()
             }), seamless: true) { [weak self] in
            self?.engine.setPreview(false)
            if self?.listening == false { self?.engine.setMode("off") }
        }
    }

    @objc func showSettings() {
        open("settings", title: "SignBridge Settings", size: NSSize(width: 480, height: 470),
             view: SettingsView(engine: engine, runSetup: { [weak self] in self?.showSetup() }))
    }

    @objc func quit() { NSApp.terminate(nil) }

    private func open<V: View>(_ key: String, title: String, size: NSSize, view: V, seamless: Bool = false,
                               onClose: (() -> Void)? = nil) {
        if let w = windows[key] {
            w.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        let w = NSWindow(contentRect: NSRect(origin: .zero, size: size),
                         styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        w.title = title
        if seamless {  // content runs under a transparent title bar
            w.styleMask.insert(.fullSizeContentView)
            w.titlebarAppearsTransparent = true
            w.titleVisibility = .hidden
        }
        w.contentView = NSHostingView(rootView: view)
        w.isReleasedWhenClosed = false
        w.center()
        let observer = WindowCloseObserver { [weak self] in
            self?.windows[key] = nil
            onClose?()
        }
        w.delegate = observer
        objc_setAssociatedObject(w, "closeObserver", observer, .OBJC_ASSOCIATION_RETAIN)
        windows[key] = w
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

final class WindowCloseObserver: NSObject, NSWindowDelegate {
    let onClose: () -> Void
    init(_ onClose: @escaping () -> Void) { self.onClose = onClose }
    func windowWillClose(_ notification: Notification) { onClose() }
}

MainActor.assumeIsolated {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    app.run()
}

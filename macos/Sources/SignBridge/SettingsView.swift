import SwiftUI

struct SettingsView: View {
    @ObservedObject var engine: EngineClient
    let runSetup: () -> Void
    @State private var still = 0.5
    @State private var phraseRest = 1.5
    @State private var minConf = 0.6
    @State private var saved = false

    var body: some View {
        Form {
            Section {
                HStack(spacing: 12) {
                    Logo(size: 44)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("SignBridge").font(.headline)
                        Text("Version \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev")")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Run Setup Again", action: runSetup)
                }
            }
            Section {
                setting("Pause between signs", "Hold still this long to end one sign.",
                        value: $still, range: 0.3...1.2, text: String(format: "%.1f s", still))
                setting("Rest to finish", "Hands down this long sends the phrase to Claude.",
                        value: $phraseRest, range: 0.8...3, text: String(format: "%.1f s", phraseRest))
                setting("Minimum confidence", "Signs the model is less sure of are ignored.",
                        value: $minConf, range: 0.3...0.95, text: "\(Int((minConf * 100).rounded()))%")
            } header: {
                Text("Recognition")
            } footer: {
                HStack {
                    if saved { Label("Saved", systemImage: "checkmark.circle.fill").foregroundStyle(.green) }
                    Spacer()
                    Button("Save") {
                        engine.saveConfig(["still": (still * 10).rounded() / 10, "phrase_rest": (phraseRest * 10).rounded() / 10,
                                           "min_conf": (minConf * 20).rounded() / 20])
                        withAnimation { saved = true }
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                }
                .padding(.top, 8)
            }
        }
        .formStyle(.grouped)
        .tint(Brand.violet)
        .onChange(of: [still, phraseRest, minConf]) { _, _ in saved = false }
        .onAppear {
            still = engine.config["still"] as? Double ?? 0.5
            phraseRest = engine.config["phrase_rest"] as? Double ?? 1.5
            minConf = engine.config["min_conf"] as? Double ?? 0.6
        }
    }

    private func setting(_ title: String, _ help: String, value: Binding<Double>, range: ClosedRange<Double>,
                         text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text(text).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: value, in: range).labelsHidden()
            Text(help).font(.caption).foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

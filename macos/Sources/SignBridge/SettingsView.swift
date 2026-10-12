import SwiftUI

struct SettingsView: View {
    @ObservedObject var engine: EngineClient
    let runSetup: () -> Void
    @State private var still = 0.5
    @State private var phraseRest = 1.5
    @State private var minConf = 0.6

    var body: some View {
        Form {
            Section("Recognition") {
                LabeledContent("Pause between signs") {
                    Slider(value: $still, in: 0.3...1.2, step: 0.1) { Text("") }
                    Text(String(format: "%.1f s", still)).monospacedDigit().frame(width: 44)
                }
                LabeledContent("Rest to finish a phrase") {
                    Slider(value: $phraseRest, in: 0.8...3, step: 0.1) { Text("") }
                    Text(String(format: "%.1f s", phraseRest)).monospacedDigit().frame(width: 44)
                }
                LabeledContent("Ignore signs below") {
                    Slider(value: $minConf, in: 0.3...0.95, step: 0.05) { Text("") }
                    Text("\(Int(minConf * 100))%").monospacedDigit().frame(width: 44)
                }
            }
            HStack {
                Button("Run Setup Again", action: runSetup)
                Spacer()
                Button("Save") {
                    engine.saveConfig(["still": still, "phrase_rest": phraseRest, "min_conf": minConf])
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .formStyle(.grouped)
        .padding()
        .onAppear {
            still = engine.config["still"] as? Double ?? 0.5
            phraseRest = engine.config["phrase_rest"] as? Double ?? 1.5
            minConf = engine.config["min_conf"] as? Double ?? 0.6
        }
    }
}

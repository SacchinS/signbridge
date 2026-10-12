// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "SignBridge",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "SignBridge", path: "Sources/SignBridge")
    ]
)

// swift render.swift icon.svg Out.iconset — rasterize the icon SVG into every size an .iconset needs
import AppKit

let args = CommandLine.arguments
guard let svg = NSImage(contentsOfFile: args[1]) else { fatalError("can't read \(args[1])") }

func png(_ pixels: Int, to path: String) {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                               bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    svg.draw(in: NSRect(x: 0, y: 0, width: pixels, height: pixels))
    NSGraphicsContext.restoreGraphicsState()
    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: path))
}

try! FileManager.default.createDirectory(atPath: args[2], withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    png(size, to: "\(args[2])/icon_\(size)x\(size).png")
    png(size * 2, to: "\(args[2])/icon_\(size)x\(size)@2x.png")
}

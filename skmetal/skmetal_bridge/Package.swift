// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "SkMetalBridge",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "SkMetalBridge", type: .dynamic, targets: ["SkMetalBridge"]),
        .library(name: "SkMetalBridgeC", type: .dynamic, targets: ["SkMetalBridgeC"]),
    ],
    targets: [
        .target(
            name: "SkMetalBridge",
            dependencies: [],
            resources: [.process("Kernels")],
            swiftSettings: [],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Accelerate"),
            ]
        ),
        .target(
            name: "SkMetalBridgeC",
            dependencies: ["SkMetalBridge"],
            path: "Sources/SkMetalBridgeC",
            publicHeadersPath: "include"
        ),
        .testTarget(name: "SkMetalBridgeTests", dependencies: ["SkMetalBridge"]),
    ]
)
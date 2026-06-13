import Metal
import MetalPerformanceShaders

final class MetalContext: @unchecked Sendable {
    nonisolated(unsafe) private static var _shared: MetalContext?

    nonisolated static var shared: MetalContext {
        if let instance = _shared { return instance }
        let instance = MetalContext()
        _shared = instance
        return instance
    }

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let bufferPool: BufferPool
    let library: MTLLibrary
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    private let cacheLock = NSLock()

    private init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("Metal not supported on this device")
        }
        self.device = device
        self.commandQueue = queue
        self.bufferPool = BufferPool(device: device)
        self.library = MetalContext.compileLibrary(device: device)
    }

    private static func compileLibrary(device: MTLDevice) -> MTLLibrary {
        // Try precompiled .metallib first
        let metallibName = "SkMetalBridge"
        let searchURLs: [URL?] = [
            Bundle.module.url(forResource: metallibName, withExtension: "metallib"),
            Bundle.module.url(forResource: metallibName, withExtension: "metallib", subdirectory: "Kernels"),
        ]
        for case let url? in searchURLs {
            do {
                return try device.makeLibrary(URL: url)
            } catch {}
        }

        fatalError("""
            SkMetalBridge.metallib not found. Run: cd skmetal/skmetal_bridge && ./compile_metal.sh
            Searched Bundle.module and Kernels/ subdirectory.
            """)
    }

    func getPipeline(name: String, functionName: String) -> MTLComputePipelineState? {
        cacheLock.lock()
        defer { cacheLock.unlock() }

        if let cached = pipelineCache[name] { return cached }

        guard let function = library.makeFunction(name: functionName) else {
            return nil
        }
        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelineCache[name] = pipeline
            return pipeline
        } catch {
            return nil
        }
    }

}

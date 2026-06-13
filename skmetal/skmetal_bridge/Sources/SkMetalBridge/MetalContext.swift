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
        // Runtime compilation — most portable, avoids metallib path issues
        do {
            let lib = try device.makeLibrary(source: MetalSource.all, options: nil)
            return lib
        } catch {
            fatalError("Failed to compile Metal library: \(error)")
        }
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

    func getMPSGemm(M: Int, N: Int, K: Int, transA: Bool, transB: Bool) -> MPSMatrixMultiplication? {
        return MPSMatrixMultiplication(
            device: device,
            transposeLeft: transA,
            transposeRight: transB,
            resultRows: M,
            resultColumns: N,
            interiorColumns: K,
            alpha: 1.0,
            beta: 0.0
        )
    }
}

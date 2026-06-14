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
    let library: MTLLibrary
    private var queueMap: [Int: MTLCommandQueue] = [:]
    private let queueLock = NSLock()
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    private let cacheLock = NSLock()
    private var bufferCache: [Int: [MTLBuffer]] = [:]
    private let bufferLock = NSLock()
    private var mpsGemmCache: [String: MPSMatrixMultiplication] = [:]
    private let mpsLock = NSLock()

    var commandQueue: MTLCommandQueue {
        let tid = Thread.current.hashValue
        queueLock.lock()
        if let q = queueMap[tid] {
            queueLock.unlock()
            return q
        }
        guard let q = device.makeCommandQueue() else {
            queueLock.unlock()
            fatalError("Failed to create MTLCommandQueue")
        }
        queueMap[tid] = q
        queueLock.unlock()
        return q
    }

    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not supported on this device")
        }
        self.device = device
        self.library = MetalContext.compileLibrary(device: device)
    }

    private static func compileLibrary(device: MTLDevice) -> MTLLibrary {
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

    func reusableBuffer(length: Int) -> MTLBuffer? {
        bufferLock.lock()
        if var pool = bufferCache[length], !pool.isEmpty {
            let buf = pool.removeLast()
            bufferLock.unlock()
            return buf
        }
        bufferLock.unlock()
        return device.makeBuffer(length: length, options: .storageModeShared)
    }

    func recycleBuffer(_ buffer: MTLBuffer) {
        bufferLock.lock()
        let key = buffer.length
        if bufferCache[key] == nil {
            bufferCache[key] = []
        }
        bufferCache[key]!.append(buffer)
        bufferLock.unlock()
    }

    func getMPSGemm(transposeLeft: Bool, transposeRight: Bool,
                     resultRows: Int, resultColumns: Int, interiorColumns: Int,
                     alpha: Double, beta: Double) -> MPSMatrixMultiplication {
        let key = "\(transposeLeft)_\(transposeRight)_\(resultRows)_\(resultColumns)_\(interiorColumns)"
        mpsLock.lock()
        defer { mpsLock.unlock() }
        if let cached = mpsGemmCache[key] { return cached }
        let gemm = MPSMatrixMultiplication(
            device: device,
            transposeLeft: transposeLeft,
            transposeRight: transposeRight,
            resultRows: resultRows,
            resultColumns: resultColumns,
            interiorColumns: interiorColumns,
            alpha: alpha,
            beta: beta)
        mpsGemmCache[key] = gemm
        return gemm
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

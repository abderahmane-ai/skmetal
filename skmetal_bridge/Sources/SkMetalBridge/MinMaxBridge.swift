import Foundation
import Metal

// MARK: - MinMax GPU Transform (zero-copy)

@_cdecl("skmetal_minmax_transform")
public func skmetal_minmax_transform(
    X: UnsafeRawPointer,
    X_out: UnsafeMutableRawPointer,
    min_vals: UnsafeRawPointer,
    max_vals: UnsafeRawPointer,
    n: Int,
    d: Int,
    feature_min: Float,
    feature_max: Float
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let xSize = n * d * fs
    let statSize = d * fs

    guard let inputBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let outputBuffer = wrapOutput(X_out, length: xSize, device: ctx.device),
          let minBuffer = wrapInput(min_vals, length: statSize, device: ctx.device),
          let maxBuffer = wrapInput(max_vals, length: statSize, device: ctx.device),
          let pipeline = ctx.getPipeline(name: "minmax_transform", functionName: "minmax_transform") else {
        return 1
    }

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    guard let enc = cb.makeComputeCommandEncoder() else { return 1 }

    enc.setComputePipelineState(pipeline)
    enc.setBuffer(inputBuffer, offset: 0, index: 0)
    enc.setBuffer(outputBuffer, offset: 0, index: 1)
    enc.setBuffer(minBuffer, offset: 0, index: 2)
    enc.setBuffer(maxBuffer, offset: 0, index: 3)
    var nU = UInt32(n); var dU = UInt32(d)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 5)
    var fMin = feature_min; var fMax = feature_max
    enc.setBytes(&fMin, length: MemoryLayout<Float>.stride, index: 6)
    enc.setBytes(&fMax, length: MemoryLayout<Float>.stride, index: 7)

    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let tgCount = MTLSize(width: (d + 15) / 16, height: (n + 15) / 16, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()

    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

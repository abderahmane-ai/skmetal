import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - Pairwise Distance (via expanded formula: norm² + norm² - 2*cross)

@_cdecl("skmetal_pairwise_distance")
public func skmetal_pairwise_distance(
    X: UnsafeRawPointer,
    D: UnsafeMutableRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let byteSize = n * d * fs
    let nSize = n * fs
    let outputSize = n * n * fs

    guard let inputBuffer = wrapInput(X, length: byteSize, device: ctx.device),
          let outputBuffer = wrapOutput(D, length: outputSize, device: ctx.device) else {
        return 1
    }

    guard let normBuf = ctx.reusableBuffer(length: nSize),
          let crossBuf = ctx.reusableBuffer(length: outputSize) else {
        return 1
    }

    guard let normPipeline = ctx.getPipeline(name: "row_norm_sq", functionName: "row_norm_sq"),
          let combinePipeline = ctx.getPipeline(name: "pairwise_from_cross", functionName: "pairwise_from_cross") else {
        ctx.recycleBuffer(normBuf); ctx.recycleBuffer(crossBuf)
        return 1
    }

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }

    // 1) Row norms: one thread per row, simd_sum for reduction
    guard let encNorm = cb.makeComputeCommandEncoder() else { return 1 }
    encNorm.setComputePipelineState(normPipeline)
    encNorm.setBuffer(inputBuffer, offset: 0, index: 0)
    encNorm.setBuffer(normBuf, offset: 0, index: 1)
    var nU = UInt32(n); var dU = UInt32(d)
    encNorm.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    encNorm.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 3)
    encNorm.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encNorm.endEncoding()

    // 2) Cross product: X @ X^T via MPS GEMM with transpose
    let rowBytesX = d * fs
    let rowBytesN = n * fs
    let descX = MPSMatrixDescriptor(dimensions: n, columns: d, rowBytes: rowBytesX, dataType: .float32)
    let descC = MPSMatrixDescriptor(dimensions: n, columns: n, rowBytes: rowBytesN, dataType: .float32)
    let matrixX = MPSMatrix(buffer: inputBuffer, descriptor: descX)
    let matrixC = MPSMatrix(buffer: crossBuf, descriptor: descC)

    let gemm = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: true,
        resultRows: n, resultColumns: n, interiorColumns: d,
        alpha: 1.0, beta: 0.0)
    gemm.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixX, resultMatrix: matrixC)

    // 3) Combine: D[i][j] = norm[i] + norm[j] - 2*cross[i][j]
    guard let encComb = cb.makeComputeCommandEncoder() else { return 1 }
    encComb.setComputePipelineState(combinePipeline)
    encComb.setBuffer(normBuf, offset: 0, index: 0)
    encComb.setBuffer(crossBuf, offset: 0, index: 1)
    encComb.setBuffer(outputBuffer, offset: 0, index: 2)
    var nU2 = UInt32(n)
    encComb.setBytes(&nU2, length: MemoryLayout<UInt32>.stride, index: 3)
    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let tgCount = MTLSize(width: (n + 15) / 16, height: (n + 15) / 16, depth: 1)
    encComb.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    encComb.endEncoding()

    cb.commit()
    cb.waitUntilCompleted()

    ctx.recycleBuffer(normBuf)
    ctx.recycleBuffer(crossBuf)

    return 0
}

// MARK: - Compute per-point min distances (for k-means++ init, avoids CPU diff)

@_cdecl("skmetal_compute_mindists")
public func skmetal_compute_mindists(
    X: UnsafeRawPointer,
    centroids: UnsafeRawPointer,
    assignments: UnsafeRawPointer,
    dists: UnsafeMutableRawPointer,
    n: Int,
    d: Int,
    k: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let cSize = k * d * MemoryLayout<Float>.stride
    let aSize = n * MemoryLayout<UInt32>.stride
    let dSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "compute_mindists", functionName: "compute_mindists"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let cBuffer = wrapInput(centroids, length: cSize, device: ctx.device),
          let aBuffer = wrapInput(assignments, length: aSize, device: ctx.device),
          let dBuffer = wrapOutput(dists, length: dSize, device: ctx.device) else {
        return 1
    }

    guard let commandBuffer = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return 1 }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(cBuffer, offset: 0, index: 1)
    encoder.setBuffer(aBuffer, offset: 0, index: 2)
    encoder.setBuffer(dBuffer, offset: 0, index: 3)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 5)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KMeans assign

@_cdecl("skmetal_kmeans_assign")
public func skmetal_kmeans_assign(
    X: UnsafeRawPointer,
    centroids: UnsafeRawPointer,
    assignments: UnsafeMutableRawPointer,
    n: Int,
    d: Int,
    k: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let cSize = k * d * MemoryLayout<Float>.stride
    let aSize = n * MemoryLayout<UInt32>.stride

    guard let pipeline = ctx.getPipeline(name: "kmeans_assign", functionName: "kmeans_assign"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let cBuffer = wrapInput(centroids, length: cSize, device: ctx.device),
          let aBuffer = wrapOutput(assignments, length: aSize, device: ctx.device) else {
        return 1
    }

    guard let commandBuffer = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return 1 }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(cBuffer, offset: 0, index: 1)
    encoder.setBuffer(aBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    var kUint = UInt32(k)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 5)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KMeans batched: assign + accumulate (CAS-free) + combine_normalize
// Single code path for all problem sizes — no split into fast/slow.
// 3 dispatches per iteration (vs 2-4+N in the old code).

@_cdecl("skmetal_kmeans_batch_fused")
public func skmetal_kmeans_batch_fused(
    X: UnsafeRawPointer,
    centroids: UnsafeMutableRawPointer,
    assignments: UnsafeMutableRawPointer,
    n: Int,
    d: Int,
    k: Int,
    numGroups: Int,
    maxIter: Int,
    tol: Float,
    n_iter_out: UnsafeMutablePointer<Int32>?
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let xSize = n * d * fs
    let cSize = k * d * fs
    let aSize = n * MemoryLayout<UInt32>.stride

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let cBuffer = wrapOutput(centroids, length: cSize, device: ctx.device),
          let aBuffer = wrapOutput(assignments, length: aSize, device: ctx.device) else {
        return 1
    }

    guard let assignPipeline = ctx.getPipeline(name: "kmeans_assign", functionName: "kmeans_assign"),
          let accumulatePipeline = ctx.getPipeline(name: "kmeans_accumulate", functionName: "kmeans_accumulate"),
          let combineNormPipeline = ctx.getPipeline(name: "kmeans_combine_normalize", functionName: "kmeans_combine_normalize"),
          let shiftPipeline = ctx.getPipeline(name: "kmeans_shift", functionName: "kmeans_shift") else {
        return 1
    }

    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let assignGrid = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    let accumulateGrid = MTLSize(width: k, height: numGroups, depth: 1)

    let pcSize = numGroups * k * d * fs
    let pnSize = numGroups * k * MemoryLayout<UInt32>.stride

    let buf1 = ctx.reusableBuffer(length: pcSize)
    let buf2 = ctx.reusableBuffer(length: pnSize)
    guard let buf1, let buf2 else {
        if let b = buf1 { ctx.recycleBuffer(b) }
        if let b = buf2 { ctx.recycleBuffer(b) }
        return 1
    }
    let pcBuffer = buf1
    let pnBuffer = buf2

    guard let oldCentroids = ctx.device.makeBuffer(length: cSize, options: .storageModeShared),
          let partialShiftBuf = ctx.device.makeBuffer(
            length: max(1, (k + 255) / 256) * fs, options: .storageModeShared) else {
        ctx.recycleBuffer(pcBuffer)
        ctx.recycleBuffer(pnBuffer)
        return 1
    }

    memcpy(oldCentroids.contents(), cBuffer.contents(), cSize)

    var nU = UInt32(n), dU = UInt32(d), kU = UInt32(k), ngU = UInt32(numGroups)
    var nIter: Int32 = 0

    for it in 0..<maxIter {
        nIter = Int32(it + 1)
        guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }

        guard let enc1 = cb.makeComputeCommandEncoder() else { return 1 }
        enc1.setComputePipelineState(assignPipeline)
        enc1.setBuffer(xBuffer, offset: 0, index: 0)
        enc1.setBuffer(cBuffer, offset: 0, index: 1)
        enc1.setBuffer(aBuffer, offset: 0, index: 2)
        enc1.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc1.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc1.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc1.dispatchThreadgroups(assignGrid, threadsPerThreadgroup: tgSize)
        enc1.endEncoding()

        guard let enc2 = cb.makeComputeCommandEncoder() else { return 1 }
        enc2.setComputePipelineState(accumulatePipeline)
        enc2.setBuffer(xBuffer, offset: 0, index: 0)
        enc2.setBuffer(aBuffer, offset: 0, index: 1)
        enc2.setBuffer(pcBuffer, offset: 0, index: 2)
        enc2.setBuffer(pnBuffer, offset: 0, index: 3)
        enc2.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc2.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc2.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 6)
        enc2.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 7)
        enc2.dispatchThreadgroups(accumulateGrid,
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc2.endEncoding()

        guard let enc3 = cb.makeComputeCommandEncoder() else { return 1 }
        enc3.setComputePipelineState(combineNormPipeline)
        enc3.setBuffer(pcBuffer, offset: 0, index: 0)
        enc3.setBuffer(pnBuffer, offset: 0, index: 1)
        enc3.setBuffer(cBuffer, offset: 0, index: 2)
        enc3.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc3.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc3.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc3.dispatchThreadgroups(MTLSize(width: d, height: k, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc3.endEncoding()

        // Centroid shift: max row-wise euclidean distance between old and new
        let shiftNumGroups = max(1, (k + 255) / 256)
        guard let enc4 = cb.makeComputeCommandEncoder() else { return 1 }
        enc4.setComputePipelineState(shiftPipeline)
        enc4.setBuffer(cBuffer, offset: 0, index: 0)
        enc4.setBuffer(oldCentroids, offset: 0, index: 1)
        enc4.setBuffer(partialShiftBuf, offset: 0, index: 2)
        enc4.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc4.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc4.dispatchThreads(MTLSize(width: k, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc4.endEncoding()

        cb.commit()
        cb.waitUntilCompleted()

        let partialPtr = partialShiftBuf.contents().assumingMemoryBound(to: Float.self)
        var maxSq: Float = 0.0
        for i in 0..<shiftNumGroups { maxSq = max(maxSq, partialPtr[i]) }
        let shift = sqrt(maxSq)
        if shift < tol { break }

        memcpy(oldCentroids.contents(), cBuffer.contents(), cSize)
    }

    ctx.recycleBuffer(pcBuffer)
    ctx.recycleBuffer(pnBuffer)

    n_iter_out?.pointee = nIter
    return 0
}

// MARK: - KMeans inertia (total squared distance)

@_cdecl("skmetal_kmeans_inertia")
public func skmetal_kmeans_inertia(
    X: UnsafeRawPointer,
    centroids: UnsafeRawPointer,
    assignments: UnsafeRawPointer,
    n: Int,
    d: Int,
    k: Int
) -> Float {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let xSize = n * d * fs
    let cSize = k * d * fs
    let aSize = n * MemoryLayout<UInt32>.stride

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let cBuffer = wrapInput(centroids, length: cSize, device: ctx.device),
          let aBuffer = wrapInput(assignments, length: aSize, device: ctx.device) else {
        return -1.0
    }

    let numGroups = max(1, (n + 255) / 256)
    let partialSize = numGroups * fs
    guard let partialBuffer = ctx.device.makeBuffer(length: partialSize, options: .storageModeShared) else {
        return -1.0
    }

    guard let commandBuffer = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    if let pipeline = ctx.getPipeline(name: "kmeans_inertia", functionName: "kmeans_inertia") {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xBuffer, offset: 0, index: 0)
        enc.setBuffer(cBuffer, offset: 0, index: 1)
        enc.setBuffer(aBuffer, offset: 0, index: 2)
        enc.setBuffer(partialBuffer, offset: 0, index: 3)
        var nU = UInt32(n); var dU = UInt32(d)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let ptr = partialBuffer.contents().assumingMemoryBound(to: Float.self)
    var total: Float = 0.0
    for i in 0..<numGroups { total += ptr[i] }
    return total
}

// MARK: - KMeans shift (max centroid movement)

@_cdecl("skmetal_kmeans_shift")
public func skmetal_kmeans_shift(
    new_centroids: UnsafeRawPointer,
    old_centroids: UnsafeRawPointer,
    k: Int,
    d: Int
) -> Float {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let cSize = k * d * fs

    guard let newBuffer = wrapInput(new_centroids, length: cSize, device: ctx.device),
          let oldBuffer = wrapInput(old_centroids, length: cSize, device: ctx.device) else {
        return -1.0
    }

    let numGroups = max(1, (k + 255) / 256)
    let partialSize = numGroups * fs
    guard let partialBuffer = ctx.device.makeBuffer(length: partialSize, options: .storageModeShared) else {
        return -1.0
    }

    guard let commandBuffer = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    if let pipeline = ctx.getPipeline(name: "kmeans_shift", functionName: "kmeans_shift") {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(newBuffer, offset: 0, index: 0)
        enc.setBuffer(oldBuffer, offset: 0, index: 1)
        enc.setBuffer(partialBuffer, offset: 0, index: 2)
        var kU = UInt32(k); var dU = UInt32(d)
        enc.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.dispatchThreads(MTLSize(width: k, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let ptr = partialBuffer.contents().assumingMemoryBound(to: Float.self)
    var maxSq: Float = 0.0
    for i in 0..<numGroups { maxSq = max(maxSq, ptr[i]) }
    return sqrt(maxSq)
}

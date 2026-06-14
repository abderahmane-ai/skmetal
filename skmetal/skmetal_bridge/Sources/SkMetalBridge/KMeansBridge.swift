import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - Pairwise Distance (zero-copy)

@_cdecl("skmetal_pairwise_distance")
public func skmetal_pairwise_distance(
    X: UnsafeRawPointer,
    D: UnsafeMutableRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * d * MemoryLayout<Float>.stride
    let outputSize = n * n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "pairwise_distance_direct", functionName: "pairwise_distance_direct"),
          let inputBuffer = wrapInput(X, length: byteSize, device: ctx.device),
          let outputBuffer = wrapOutput(D, length: outputSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
    let gridSize = MTLSize(width: n, height: n, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - Row-wise norm (for KMeans GEMM path)

@_cdecl("skmetal_row_norm_sq")
public func skmetal_row_norm_sq(
    X: UnsafeRawPointer,
    norms: UnsafeMutableRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let outSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "row_norm_sq", functionName: "row_norm_sq"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let outBuffer = wrapOutput(norms, length: outSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(outBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - Distance correction (expansion trick: D = X_norm² + C_norm² - 2*raw_D)

@_cdecl("skmetal_distance_correct")
public func skmetal_distance_correct(
    D: UnsafeMutableRawPointer,
    X_norm: UnsafeRawPointer,
    C_norm: UnsafeRawPointer,
    n: Int,
    k: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let dSize = n * k * MemoryLayout<Float>.stride
    let normSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "distance_correct", functionName: "distance_correct"),
          let dBuffer = wrapOutput(D, length: dSize, device: ctx.device),
          let xnBuffer = wrapInput(X_norm, length: normSize, device: ctx.device),
          let cnBuffer = wrapInput(C_norm, length: k * MemoryLayout<Float>.stride, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(dBuffer, offset: 0, index: 0)
    encoder.setBuffer(xnBuffer, offset: 0, index: 1)
    encoder.setBuffer(cnBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    var kUint = UInt32(k)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 4)

    let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
    let gridSize = MTLSize(width: k, height: n, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - Argmin per row (for KMeans GEMM path)

@_cdecl("skmetal_argmin_rows")
public func skmetal_argmin_rows(
    matrix: UnsafeRawPointer,
    indices: UnsafeMutableRawPointer,
    n: Int,
    k: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * k * MemoryLayout<Float>.stride
    let idxSize = n * MemoryLayout<UInt32>.stride

    guard let pipeline = ctx.getPipeline(name: "argmin_rows", functionName: "argmin_rows"),
          let matBuffer = wrapInput(matrix, length: matSize, device: ctx.device),
          let idxBuffer = wrapOutput(indices, length: idxSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(matBuffer, offset: 0, index: 0)
    encoder.setBuffer(idxBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    var kUint = UInt32(k)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: n, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
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

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

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

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

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
    let gridSize = MTLSize(width: n, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KMeans fused combine + normalize

@_cdecl("skmetal_kmeans_combine_normalize")
public func skmetal_kmeans_combine_normalize(
    partialCentroids: UnsafeRawPointer,
    partialCounts: UnsafeRawPointer,
    centroids: UnsafeMutableRawPointer,
    k: Int,
    d: Int,
    numGroups: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let pcSize = numGroups * k * d * MemoryLayout<Float>.stride
    let pcountSize = numGroups * k * MemoryLayout<UInt32>.stride
    let cSize = k * d * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "kmeans_combine_normalize", functionName: "kmeans_combine_normalize"),
          let pcBuffer = wrapInput(partialCentroids, length: pcSize, device: ctx.device),
          let pcountBuffer = wrapInput(partialCounts, length: pcountSize, device: ctx.device),
          let cBuffer = wrapOutput(centroids, length: cSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(pcBuffer, offset: 0, index: 0)
    encoder.setBuffer(pcountBuffer, offset: 0, index: 1)
    encoder.setBuffer(cBuffer, offset: 0, index: 2)
    var kUint = UInt32(k)
    var dUint = UInt32(d)
    var ngUint = UInt32(numGroups)
    encoder.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&ngUint, length: MemoryLayout<UInt32>.stride, index: 5)

    let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
    let gridSize = MTLSize(width: d, height: k, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KMeans batched: assign + partial_sum (cluster‑batched) + combine_normalize

@_cdecl("skmetal_kmeans_batch_fused")
public func skmetal_kmeans_batch_fused(
    X: UnsafeRawPointer,
    centroids: UnsafeMutableRawPointer,
    assignments: UnsafeMutableRawPointer,
    n: Int,
    d: Int,
    k: Int,
    numGroups: Int,
    maxIter: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let cSize = k * d * MemoryLayout<Float>.stride
    let aSize = n * MemoryLayout<UInt32>.stride

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let cBuffer = wrapOutput(centroids, length: cSize, device: ctx.device),
          let aBuffer = wrapOutput(assignments, length: aSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: numGroups, height: 1, depth: 1)

    var nU = UInt32(n), dU = UInt32(d), kU = UInt32(k)

    let pcSize = numGroups * k * d * MemoryLayout<Float>.stride
    let pnSize = numGroups * k * MemoryLayout<UInt32>.stride

    var pcBuffer: MTLBuffer?
    var pnBuffer: MTLBuffer?

    if k * (d + 1) <= 7168 && k <= 256 {
        guard let assignPartialPipeline = ctx.getPipeline(
                name: "kmeans_assign_partial", functionName: "kmeans_assign_partial"),
              let combineNormPipeline = ctx.getPipeline(
                name: "kmeans_combine_normalize", functionName: "kmeans_combine_normalize") else {
            return 1
        }
        let buf1 = ctx.reusableBuffer(length: pcSize)
        let buf2 = ctx.reusableBuffer(length: pnSize)
        guard let buf1, let buf2 else {
            if let b = buf1 { ctx.recycleBuffer(b) }
            if let b = buf2 { ctx.recycleBuffer(b) }
            return 1
        }
        pcBuffer = buf1
        pnBuffer = buf2

        var ngU = UInt32(numGroups)

        for _ in 0..<maxIter {
            let enc1 = commandBuffer.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(assignPartialPipeline)
            enc1.setBuffer(xBuffer, offset: 0, index: 0)
            enc1.setBuffer(cBuffer, offset: 0, index: 1)
            enc1.setBuffer(pcBuffer!, offset: 0, index: 2)
            enc1.setBuffer(pnBuffer!, offset: 0, index: 3)
            enc1.setBuffer(aBuffer, offset: 0, index: 4)
            enc1.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc1.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 6)
            enc1.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
            enc1.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 8)
            enc1.dispatchThreadgroups(gridSize, threadsPerThreadgroup: tgSize)
            enc1.endEncoding()

            let enc2 = commandBuffer.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(combineNormPipeline)
            enc2.setBuffer(pcBuffer!, offset: 0, index: 0)
            enc2.setBuffer(pnBuffer!, offset: 0, index: 1)
            enc2.setBuffer(cBuffer, offset: 0, index: 2)
            enc2.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc2.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc2.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc2.dispatchThreadgroups(MTLSize(width: d, height: k, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc2.endEncoding()
        }
    } else {
        guard let assignPipeline = ctx.getPipeline(
                name: "kmeans_assign", functionName: "kmeans_assign"),
              let partialPipeline = ctx.getPipeline(
                name: "kmeans_partial_sum", functionName: "kmeans_partial_sum"),
              let combineNormPipeline = ctx.getPipeline(
                name: "kmeans_combine_normalize", functionName: "kmeans_combine_normalize") else {
            return 1
        }
        let buf1 = ctx.reusableBuffer(length: pcSize)
        let buf2 = ctx.reusableBuffer(length: pnSize)
        guard let buf1, let buf2 else {
            if let b = buf1 { ctx.recycleBuffer(b) }
            if let b = buf2 { ctx.recycleBuffer(b) }
            return 1
        }
        pcBuffer = buf1
        pnBuffer = buf2

        var ngU = UInt32(numGroups)

        let maxBatchClusters = min(256, max(1, 7168 / (d + 1)))

        for _ in 0..<maxIter {
            let clearEnc = commandBuffer.makeBlitCommandEncoder()!
            clearEnc.fill(buffer: pcBuffer!, range: 0..<pcSize, value: 0)
            clearEnc.fill(buffer: pnBuffer!, range: 0..<pnSize, value: 0)
            clearEnc.endEncoding()

            let enc1 = commandBuffer.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(assignPipeline)
            enc1.setBuffer(xBuffer, offset: 0, index: 0)
            enc1.setBuffer(cBuffer, offset: 0, index: 1)
            enc1.setBuffer(aBuffer, offset: 0, index: 2)
            enc1.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc1.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc1.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc1.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
            enc1.endEncoding()

            var clusterStart = 0
            while clusterStart < k {
                let batchK = min(k - clusterStart, maxBatchClusters)
                var csU = UInt32(clusterStart)
                var bkU = UInt32(batchK)

                let enc2 = commandBuffer.makeComputeCommandEncoder()!
                enc2.setComputePipelineState(partialPipeline)
                enc2.setBuffer(xBuffer, offset: 0, index: 0)
                enc2.setBuffer(aBuffer, offset: 0, index: 1)
                enc2.setBuffer(pcBuffer!, offset: 0, index: 2)
                enc2.setBuffer(pnBuffer!, offset: 0, index: 3)
                enc2.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
                enc2.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 5)
                enc2.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 6)
                enc2.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
                enc2.setBytes(&csU, length: MemoryLayout<UInt32>.stride, index: 8)
                enc2.setBytes(&bkU, length: MemoryLayout<UInt32>.stride, index: 9)
                enc2.dispatchThreadgroups(gridSize, threadsPerThreadgroup: tgSize)
                enc2.endEncoding()

                clusterStart += batchK
            }

            let enc3 = commandBuffer.makeComputeCommandEncoder()!
            enc3.setComputePipelineState(combineNormPipeline)
            enc3.setBuffer(pcBuffer!, offset: 0, index: 0)
            enc3.setBuffer(pnBuffer!, offset: 0, index: 1)
            enc3.setBuffer(cBuffer, offset: 0, index: 2)
            enc3.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc3.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc3.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc3.dispatchThreadgroups(MTLSize(width: d, height: k, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc3.endEncoding()
        }
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let b1 = pcBuffer { ctx.recycleBuffer(b1) }
    if let b2 = pnBuffer { ctx.recycleBuffer(b2) }

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

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    if let pipeline = ctx.getPipeline(name: "kmeans_inertia", functionName: "kmeans_inertia") {
        let enc = commandBuffer.makeComputeCommandEncoder()!
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

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    if let pipeline = ctx.getPipeline(name: "kmeans_shift", functionName: "kmeans_shift") {
        let enc = commandBuffer.makeComputeCommandEncoder()!
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

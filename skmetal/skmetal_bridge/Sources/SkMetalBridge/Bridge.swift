import Foundation
import Metal
import MetalPerformanceShaders

// MARK: - Zero-copy buffer helpers

private func wrapInput(_ ptr: UnsafeRawPointer, length: Int, device: MTLDevice) -> MTLBuffer? {
    let mut = UnsafeMutableRawPointer(mutating: ptr)
    return device.makeBuffer(bytesNoCopy: mut, length: length,
                             options: .storageModeShared, deallocator: nil)
}

private func wrapOutput(_ ptr: UnsafeMutableRawPointer, length: Int, device: MTLDevice) -> MTLBuffer? {
    return device.makeBuffer(bytesNoCopy: ptr, length: length,
                             options: .storageModeShared, deallocator: nil)
}

@_cdecl("skmetal_init")
public func skmetal_init() -> Int32 {
    _ = MetalContext.shared
    return 0
}

@_cdecl("skmetal_device_info")
public func skmetal_device_info(
    name: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    maxThreads: UnsafeMutablePointer<Int>?
) -> Int32 {
    let ctx = MetalContext.shared
    let nameStr = ctx.device.name
    name.pointee = strdup(nameStr)
    maxThreads?.pointee = ctx.device.maxThreadsPerThreadgroup.width
    return 0
}

// MARK: - GEMM (MPS path, zero-copy)

@_cdecl("skmetal_gemm")
public func skmetal_gemm(
    A: UnsafeRawPointer,
    B: UnsafeRawPointer,
    C: UnsafeMutableRawPointer,
    M: Int,
    N: Int,
    K: Int,
    alpha: Float,
    beta: Float,
    transA: Int32,
    transB: Int32
) -> Int32 {
    let ctx = MetalContext.shared
    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!

    let transA_ = transA != 0
    let transB_ = transB != 0

    let aRows = transA_ ? K : M
    let aCols = transA_ ? M : K
    let bRows = transB_ ? N : K
    let bCols = transB_ ? K : N

    let rowBytesA = aCols * MemoryLayout<Float>.stride
    let rowBytesB = bCols * MemoryLayout<Float>.stride
    let rowBytesC = N * MemoryLayout<Float>.stride

    let byteSizeA = aRows * rowBytesA
    let byteSizeB = bRows * rowBytesB
    let byteSizeC = M * rowBytesC

    guard let bufferA = wrapInput(A, length: byteSizeA, device: ctx.device),
          let bufferB = wrapInput(B, length: byteSizeB, device: ctx.device),
          let bufferC = wrapOutput(C, length: byteSizeC, device: ctx.device) else {
        return 1
    }

    let descA = MPSMatrixDescriptor(dimensions: aRows, columns: aCols, rowBytes: rowBytesA, dataType: .float32)
    let descB = MPSMatrixDescriptor(dimensions: bRows, columns: bCols, rowBytes: rowBytesB, dataType: .float32)
    let descC = MPSMatrixDescriptor(dimensions: M, columns: N, rowBytes: rowBytesC, dataType: .float32)

    let matrixA = MPSMatrix(buffer: bufferA, descriptor: descA)
    let matrixB = MPSMatrix(buffer: bufferB, descriptor: descB)
    let matrixC = MPSMatrix(buffer: bufferC, descriptor: descC)

    let gemm = MPSMatrixMultiplication(
        device: ctx.device,
        transposeLeft: transA_,
        transposeRight: transB_,
        resultRows: M,
        resultColumns: N,
        interiorColumns: K,
        alpha: Double(alpha),
        beta: Double(beta)
    )

    gemm.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return 0
}

// MARK: - Reductions (custom Metal kernels, zero-copy input/output)

@_cdecl("skmetal_reduce_sum")
public func skmetal_reduce_sum(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride
    let numGroups = max(1, (n + 255) / 256)
    let partialSize = numGroups * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum"),
          let inputBuffer = wrapInput(input, length: byteSize, device: ctx.device),
          let partialBuffer = ctx.device.makeBuffer(length: partialSize, options: .storageModeShared) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(partialBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)
    var ngUint = UInt32(numGroups)
    encoder.setBytes(&ngUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: numGroups, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let partials = partialBuffer.contents().assumingMemoryBound(to: Float.self)
    var total: Float = 0.0
    for i in 0..<numGroups {
        total += partials[i]
    }
    output.storeBytes(of: total, as: Float.self)
    return 0
}

@_cdecl("skmetal_reduce_mean_var")
public func skmetal_reduce_mean_var(
    input: UnsafeRawPointer,
    meanOut: UnsafeMutableRawPointer,
    varOut: UnsafeMutableRawPointer,
    n: Int,
    eps: Float
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride
    let numGroups = max(1, (n + 255) / 256)
    let partialSize = numGroups * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "reduce_mean_var", functionName: "reduce_mean_var"),
          let inputBuffer = wrapInput(input, length: byteSize, device: ctx.device),
          let meanBuffer = ctx.device.makeBuffer(length: partialSize, options: .storageModeShared),
          let m2Buffer = ctx.device.makeBuffer(length: partialSize, options: .storageModeShared),
          let countBuffer = ctx.device.makeBuffer(length: numGroups * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(meanBuffer, offset: 0, index: 1)
    encoder.setBuffer(m2Buffer, offset: 0, index: 2)
    encoder.setBuffer(countBuffer, offset: 0, index: 3)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 4)
    var ngUint = UInt32(numGroups)
    encoder.setBytes(&ngUint, length: MemoryLayout<UInt32>.stride, index: 5)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: numGroups, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let means = meanBuffer.contents().assumingMemoryBound(to: Float.self)
    let m2s = m2Buffer.contents().assumingMemoryBound(to: Float.self)
    let counts = countBuffer.contents().assumingMemoryBound(to: UInt32.self)

    var combinedMean: Float = 0.0
    var combinedM2: Float = 0.0
    var combinedCount: UInt32 = 0

    for i in 0..<numGroups {
        let count_i = counts[i]
        if count_i == 0 { continue }
        if combinedCount == 0 {
            combinedMean = means[i]
            combinedM2 = m2s[i]
            combinedCount = count_i
        } else {
            let delta = means[i] - combinedMean
            let newCount = combinedCount + count_i
            combinedMean = combinedMean + delta * (Float(count_i) / Float(newCount))
            combinedM2 = combinedM2 + m2s[i] + delta * delta * Float(combinedCount) * Float(count_i) / Float(newCount)
            combinedCount = newCount
        }
    }

    if combinedCount > 0 {
        meanOut.storeBytes(of: combinedMean, as: Float.self)
        varOut.storeBytes(of: combinedM2 / Float(combinedCount), as: Float.self)
    } else {
        meanOut.storeBytes(of: 0.0, as: Float.self)
        varOut.storeBytes(of: 0.0, as: Float.self)
    }
    return 0
}

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

    let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
    let gridSize = MTLSize(width: n, height: 1, depth: 1)
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

// MARK: - Fused StandardScaler (all columns in one dispatch)

@_cdecl("skmetal_scaler_fit")
public func skmetal_scaler_fit(
    X: UnsafeRawPointer,
    meanOut: UnsafeMutableRawPointer,
    varOut: UnsafeMutableRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let statSize = d * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "scaler_fit", functionName: "scaler_fit"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let meanBuffer = wrapOutput(meanOut, length: statSize, device: ctx.device),
          let varBuffer = wrapOutput(varOut, length: statSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(meanBuffer, offset: 0, index: 1)
    encoder.setBuffer(varBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: d, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - MinMaxScaler: per-column min and max in one dispatch

@_cdecl("skmetal_column_minmax")
public func skmetal_column_minmax(
    X: UnsafeRawPointer,
    minOut: UnsafeMutableRawPointer,
    maxOut: UnsafeMutableRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let statSize = d * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "column_minmax", functionName: "column_minmax"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let minBuffer = wrapOutput(minOut, length: statSize, device: ctx.device),
          let maxBuffer = wrapOutput(maxOut, length: statSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(minBuffer, offset: 0, index: 1)
    encoder.setBuffer(maxBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: d, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - IRLS helper (for Logistic Regression)

@_cdecl("skmetal_irls_weight")
public func skmetal_irls_weight(
    p: UnsafeRawPointer,
    weights: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "irls_weight", functionName: "irls_weight"),
          let pBuffer = wrapInput(p, length: byteSize, device: ctx.device),
          let wBuffer = wrapOutput(weights, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(pBuffer, offset: 0, index: 0)
    encoder.setBuffer(wBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_scale_rows")
public func skmetal_scale_rows(
    X: UnsafeRawPointer,
    weights: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * d * MemoryLayout<Float>.stride
    let wSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "scale_rows", functionName: "scale_rows"),
          let xBuffer = wrapInput(X, length: matSize, device: ctx.device),
          let wBuffer = wrapInput(weights, length: wSize, device: ctx.device),
          let oBuffer = wrapOutput(output, length: matSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(wBuffer, offset: 0, index: 1)
    encoder.setBuffer(oBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n * d + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KMeans

@_cdecl("skmetal_kmeans_update")
public func skmetal_kmeans_update(
    X: UnsafeRawPointer,
    assignments: UnsafeRawPointer,
    centroids: UnsafeMutableRawPointer,
    counts: UnsafeMutableRawPointer,
    n: Int,
    d: Int,
    k: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let aSize = n * MemoryLayout<UInt32>.stride
    let cSize = k * d * MemoryLayout<Float>.stride
    let countSize = k * MemoryLayout<UInt32>.stride

    guard let pipeline = ctx.getPipeline(name: "kmeans_update", functionName: "kmeans_update"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let aBuffer = wrapInput(assignments, length: aSize, device: ctx.device),
          let cBuffer = wrapOutput(centroids, length: cSize, device: ctx.device),
          let countBuffer = wrapOutput(counts, length: countSize, device: ctx.device) else {
        return 1
    }

    memset(cBuffer.contents(), 0, cSize)
    memset(countBuffer.contents(), 0, countSize)

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(aBuffer, offset: 0, index: 1)
    encoder.setBuffer(cBuffer, offset: 0, index: 2)
    encoder.setBuffer(countBuffer, offset: 0, index: 3)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    var kUint = UInt32(k)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 5)
    encoder.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 6)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: n, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KMeans (threadgroup-local update, zero-copy)

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

@_cdecl("skmetal_kmeans_partial_update")
public func skmetal_kmeans_partial_update(
    X: UnsafeRawPointer,
    assignments: UnsafeRawPointer,
    partialCentroids: UnsafeMutableRawPointer,
    partialCounts: UnsafeMutableRawPointer,
    n: Int,
    d: Int,
    k: Int,
    numGroups: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let aSize = n * MemoryLayout<UInt32>.stride
    let pcSize = numGroups * k * d * MemoryLayout<Float>.stride
    let pcountSize = numGroups * k * MemoryLayout<UInt32>.stride

    // Fallback to device-level atomics when k*d doesn't fit in threadgroup memory
    if k * d > 7168 {
        return skmetal_kmeans_update(X: X, assignments: assignments,
                                     centroids: partialCentroids, counts: partialCounts,
                                     n: n, d: d, k: k)
    }

    guard let pipeline = ctx.getPipeline(name: "kmeans_partial_update", functionName: "kmeans_partial_update"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let aBuffer = wrapInput(assignments, length: aSize, device: ctx.device),
          let pcBuffer = wrapOutput(partialCentroids, length: pcSize, device: ctx.device),
          let pcountBuffer = wrapOutput(partialCounts, length: pcountSize, device: ctx.device) else {
        return 1
    }

    // Zero out output buffers
    memset(pcBuffer.contents(), 0, pcSize)
    memset(pcountBuffer.contents(), 0, pcountSize)

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(aBuffer, offset: 0, index: 1)
    encoder.setBuffer(pcBuffer, offset: 0, index: 2)
    encoder.setBuffer(pcountBuffer, offset: 0, index: 3)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    var kUint = UInt32(k)
    var ngUint = UInt32(numGroups)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 5)
    encoder.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 6)
    encoder.setBytes(&ngUint, length: MemoryLayout<UInt32>.stride, index: 7)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: numGroups, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_kmeans_combine")
public func skmetal_kmeans_combine(
    partialCentroids: UnsafeRawPointer,
    partialCounts: UnsafeRawPointer,
    centroids: UnsafeMutableRawPointer,
    counts: UnsafeMutableRawPointer,
    k: Int,
    d: Int,
    numGroups: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let pcSize = numGroups * k * d * MemoryLayout<Float>.stride
    let pcountSize = numGroups * k * MemoryLayout<UInt32>.stride
    let cSize = k * d * MemoryLayout<Float>.stride
    let countSize = k * MemoryLayout<UInt32>.stride

    guard let pipeline = ctx.getPipeline(name: "kmeans_combine", functionName: "kmeans_combine"),
          let pcBuffer = wrapInput(partialCentroids, length: pcSize, device: ctx.device),
          let pcountBuffer = wrapInput(partialCounts, length: pcountSize, device: ctx.device),
          let cBuffer = wrapOutput(centroids, length: cSize, device: ctx.device),
          let countBuffer = wrapOutput(counts, length: countSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(pcBuffer, offset: 0, index: 0)
    encoder.setBuffer(pcountBuffer, offset: 0, index: 1)
    encoder.setBuffer(cBuffer, offset: 0, index: 2)
    encoder.setBuffer(countBuffer, offset: 0, index: 3)
    var kUint = UInt32(k)
    var dUint = UInt32(d)
    var ngUint = UInt32(numGroups)
    encoder.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 5)
    encoder.setBytes(&ngUint, length: MemoryLayout<UInt32>.stride, index: 6)

    let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
    let gridSize = MTLSize(width: k, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_kmeans_normalize")
public func skmetal_kmeans_normalize(
    centroids: UnsafeMutableRawPointer,
    counts: UnsafeRawPointer,
    k: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let cSize = k * d * MemoryLayout<Float>.stride
    let countSize = k * MemoryLayout<UInt32>.stride

    guard let pipeline = ctx.getPipeline(name: "kmeans_normalize", functionName: "kmeans_normalize"),
          let cBuffer = wrapOutput(centroids, length: cSize, device: ctx.device),
          let countBuffer = wrapInput(counts, length: countSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(cBuffer, offset: 0, index: 0)
    encoder.setBuffer(countBuffer, offset: 0, index: 1)
    var kUint = UInt32(k)
    var dUint = UInt32(d)
    encoder.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: k * d, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - SVD (MPS path, zero-copy)

@_cdecl("skmetal_svd")
public func skmetal_svd(
    A: UnsafeRawPointer,
    U: UnsafeMutableRawPointer,
    S: UnsafeMutableRawPointer,
    Vt: UnsafeMutableRawPointer,
    m: Int,
    n: Int,
    k: Int
) -> Int32 {
    let aPtr = UnsafeMutablePointer(mutating: A.assumingMemoryBound(to: Float.self))
    let uPtr = U.assumingMemoryBound(to: Float.self)
    let sPtr = S.assumingMemoryBound(to: Float.self)
    let vtPtr = Vt.assumingMemoryBound(to: Float.self)

    return MPSSVD.compute(A: aPtr, m: m, n: n, k: k, U: uPtr, S: sPtr, Vt: vtPtr) ? 0 : 1
}

// MARK: - Element-Wise Ops (for Logistic Regression)

@_cdecl("skmetal_sigmoid")
public func skmetal_sigmoid(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "sigmoid", functionName: "sigmoid"),
          let inputBuffer = wrapInput(input, length: byteSize, device: ctx.device),
          let outputBuffer = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_subtract")
public func skmetal_subtract(
    a: UnsafeRawPointer,
    b: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "subtract", functionName: "subtract"),
          let aBuffer = wrapInput(a, length: byteSize, device: ctx.device),
          let bBuffer = wrapInput(b, length: byteSize, device: ctx.device),
          let outputBuffer = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(outputBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_axpy")
public func skmetal_axpy(
    a: UnsafeMutableRawPointer,
    b: UnsafeRawPointer,
    alpha: Float,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "axpy", functionName: "axpy"),
          let aBuffer = wrapOutput(a, length: byteSize, device: ctx.device),
          let bBuffer = wrapInput(b, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    var alphaF = alpha
    encoder.setBytes(&alphaF, length: MemoryLayout<Float>.stride, index: 2)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_norm_sq")
public func skmetal_norm_sq(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "norm_sq", functionName: "norm_sq"),
          let inputBuffer = wrapInput(input, length: byteSize, device: ctx.device),
          let outputBuffer = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - Fused Ridge: center X in-place + X^T X + X^T y in one command buffer
// NOTE: X is modified in-place (centered). Callers that need to preserve X
// should pass a copy.

@_cdecl("skmetal_ridge_fit")
public func skmetal_ridge_fit(
    X: UnsafeMutableRawPointer,
    y: UnsafeRawPointer,
    XTX: UnsafeMutableRawPointer,
    XTy: UnsafeMutableRawPointer,
    X_mean_out: UnsafeMutableRawPointer,
    n: Int,
    p: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * p * MemoryLayout<Float>.stride
    let ySize = n * MemoryLayout<Float>.stride
    let xtxSize = p * p * MemoryLayout<Float>.stride
    let xtySize = p * MemoryLayout<Float>.stride
    let meanSize = p * MemoryLayout<Float>.stride

    guard let xBuffer = wrapOutput(X, length: xSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: ySize, device: ctx.device),
          let xtxBuffer = wrapOutput(XTX, length: xtxSize, device: ctx.device),
          let xtyBuffer = wrapOutput(XTy, length: xtySize, device: ctx.device),
          let meanBuffer = wrapOutput(X_mean_out, length: meanSize, device: ctx.device) else {
        return 1
    }

    let rowBytesX = p * MemoryLayout<Float>.stride
    let rowBytesXTX = p * MemoryLayout<Float>.stride

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descXTX = MPSMatrixDescriptor(dimensions: p, columns: p, rowBytes: rowBytesXTX, dataType: .float32)
    let descY = MPSMatrixDescriptor(dimensions: n, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
    let descXTy = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixXTX = MPSMatrix(buffer: xtxBuffer, descriptor: descXTX)
    let matrixY = MPSMatrix(buffer: yBuffer, descriptor: descY)
    let matrixXTy = MPSMatrix(buffer: xtyBuffer, descriptor: descXTy)

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!

    // Step 1: column means of X (one threadgroup per column, 256 threads each)
    var nU: UInt32 = UInt32(n)
    var pU: UInt32 = UInt32(p)
    let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
    if let pipeline = ctx.getPipeline(name: "column_means", functionName: "column_means") {
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(meanBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 3)
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(MTLSize(width: p, height: 1, depth: 1),
                                            threadsPerThreadgroup: tgSize)
    }
    computeEncoder.endEncoding()

    // Step 2: center X in-place
    let centerEncoder = commandBuffer.makeComputeCommandEncoder()!
    if let pipeline = ctx.getPipeline(name: "center_columns", functionName: "center_columns") {
        centerEncoder.setComputePipelineState(pipeline)
        centerEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        centerEncoder.setBuffer(meanBuffer, offset: 0, index: 1)
        centerEncoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        centerEncoder.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 3)
        centerEncoder.dispatchThreads(MTLSize(width: n * p, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    centerEncoder.endEncoding()

    // Step 3: X^T @ X (from centered X)
    let gemmXTX = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: p, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmXTX.encode(commandBuffer: commandBuffer, leftMatrix: matrixX, rightMatrix: matrixX, resultMatrix: matrixXTX)

    // Step 4: X^T @ y (from centered X, original y — Xc^T y = Xc^T yc)
    let gemmXTy = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: 1, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmXTy.encode(commandBuffer: commandBuffer, leftMatrix: matrixX, rightMatrix: matrixY, resultMatrix: matrixXTy)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - Fused IRLS iteration for LogisticRegression

@_cdecl("skmetal_logreg_irls_iter")
public func skmetal_logreg_irls_iter(
    X: UnsafeRawPointer,
    y: UnsafeRawPointer,
    w: UnsafeRawPointer,
    b: Float,
    linear: UnsafeMutableRawPointer,
    weight: UnsafeMutableRawPointer,
    X_scaled: UnsafeMutableRawPointer,
    Hessian: UnsafeMutableRawPointer,
    gradient: UnsafeMutableRawPointer,
    n: Int,
    p: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let nSize = n * MemoryLayout<Float>.stride
    let pSize = p * MemoryLayout<Float>.stride
    let xSize = n * p * MemoryLayout<Float>.stride
    let hessianSize = p * p * MemoryLayout<Float>.stride

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: nSize, device: ctx.device),
          let wBuffer = wrapInput(w, length: pSize, device: ctx.device),
          let linearBuffer = wrapOutput(linear, length: nSize, device: ctx.device),
          let weightBuffer = wrapOutput(weight, length: nSize, device: ctx.device),
          let xsBuffer = wrapOutput(X_scaled, length: xSize, device: ctx.device),
          let hBuffer = wrapOutput(Hessian, length: hessianSize, device: ctx.device),
          let gBuffer = wrapOutput(gradient, length: pSize, device: ctx.device) else {
        return 1
    }

    let rowBytesX = p * MemoryLayout<Float>.stride
    let rowBytesH = p * MemoryLayout<Float>.stride

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
    let descLin = MPSMatrixDescriptor(dimensions: n, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
    let descXS = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descH = MPSMatrixDescriptor(dimensions: p, columns: p, rowBytes: rowBytesH, dataType: .float32)
    let descG = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
    let descY = MPSMatrixDescriptor(dimensions: n, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixLin = MPSMatrix(buffer: linearBuffer, descriptor: descLin)
    let matrixXS = MPSMatrix(buffer: xsBuffer, descriptor: descXS)
    let matrixH = MPSMatrix(buffer: hBuffer, descriptor: descH)
    let matrixG = MPSMatrix(buffer: gBuffer, descriptor: descG)
    let matrixY = MPSMatrix(buffer: yBuffer, descriptor: descY)

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!

    // Step 1: X @ w → linear (n×1)
    let gemmXW = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: 1, interiorColumns: p,
        alpha: 1.0, beta: 0.0)
    gemmXW.encode(commandBuffer: commandBuffer, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixLin)

    // Step 2: linear += b (bias)
    if let pipeline = ctx.getPipeline(name: "add_scalar", functionName: "add_scalar") {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(linearBuffer, offset: 0, index: 0)
        var bScalar = b
        encoder.setBytes(&bScalar, length: MemoryLayout<Float>.stride, index: 1)
        var nU = UInt32(n)
        encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // Step 3: sigmoid(linear) → linear (in-place, now prob)
    if let pipeline = ctx.getPipeline(name: "sigmoid", functionName: "sigmoid") {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(linearBuffer, offset: 0, index: 0)
        encoder.setBuffer(linearBuffer, offset: 0, index: 1)
        var nU = UInt32(n)
        encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // Step 4: irls_weight(prob, weight) → weight
    if let pipeline = ctx.getPipeline(name: "irls_weight", functionName: "irls_weight") {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(linearBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        var nU = UInt32(n)
        encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // Step 5: subtract(prob, y, linear) → linear (now error)
    if let pipeline = ctx.getPipeline(name: "subtract", functionName: "subtract") {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(linearBuffer, offset: 0, index: 0)
        encoder.setBuffer(yBuffer, offset: 0, index: 1)
        encoder.setBuffer(linearBuffer, offset: 0, index: 2)
        var nU = UInt32(n)
        encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // Step 6: scale_rows(X, weight, X_scaled) → X_scaled
    if let pipeline = ctx.getPipeline(name: "scale_rows", functionName: "scale_rows") {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(xBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(xsBuffer, offset: 0, index: 2)
        var nU = UInt32(n)
        var pU = UInt32(p)
        encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreads(MTLSize(width: n * p, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // Step 7: X_scaled^T @ X_scaled → Hessian (p×p)
    let gemmHH = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: p, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmHH.encode(commandBuffer: commandBuffer, leftMatrix: matrixXS, rightMatrix: matrixXS, resultMatrix: matrixH)

    // Step 8: X^T @ error → gradient (p×1, error = linear buffer)
    let gemmGrad = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: 1, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmGrad.encode(commandBuffer: commandBuffer, leftMatrix: matrixX, rightMatrix: matrixLin, resultMatrix: matrixG)

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

// MARK: - GPU mean centering (for Ridge / PCA)

@_cdecl("skmetal_center_columns")
public func skmetal_center_columns(
    X: UnsafeMutableRawPointer,
    mean: UnsafeRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let meanSize = d * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "center_columns", functionName: "center_columns"),
          let xBuffer = wrapOutput(X, length: xSize, device: ctx.device),
          let meanBuffer = wrapInput(mean, length: meanSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(meanBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n * d + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KMeans fused iteration (assign + partial_update + combine_normalize in one command buffer)

@_cdecl("skmetal_kmeans_iter")
public func skmetal_kmeans_iter(
    X: UnsafeRawPointer,
    centroidsIn: UnsafeRawPointer,
    assignments: UnsafeMutableRawPointer,
    partialCentroids: UnsafeMutableRawPointer,
    partialCounts: UnsafeMutableRawPointer,
    centroidsOut: UnsafeMutableRawPointer,
    n: Int,
    d: Int,
    k: Int,
    numGroups: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let cSize = k * d * MemoryLayout<Float>.stride
    let aSize = n * MemoryLayout<UInt32>.stride
    let pcSize = numGroups * k * d * MemoryLayout<Float>.stride
    let pcountSize = numGroups * k * MemoryLayout<UInt32>.stride

    guard let assignPipeline = ctx.getPipeline(name: "kmeans_assign", functionName: "kmeans_assign"),
          let partialPipeline = ctx.getPipeline(name: "kmeans_partial_update", functionName: "kmeans_partial_update"),
          let combinePipeline = ctx.getPipeline(name: "kmeans_combine_normalize", functionName: "kmeans_combine_normalize"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let cBuffer = wrapInput(centroidsIn, length: cSize, device: ctx.device),
          let aBuffer = wrapOutput(assignments, length: aSize, device: ctx.device),
          let pcBuffer = wrapOutput(partialCentroids, length: pcSize, device: ctx.device),
          let pcountBuffer = wrapOutput(partialCounts, length: pcountSize, device: ctx.device),
          let coBuffer = wrapOutput(centroidsOut, length: cSize, device: ctx.device) else {
        return 1
    }

    memset(pcBuffer.contents(), 0, pcSize)
    memset(pcountBuffer.contents(), 0, pcountSize)

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!

    // 1. Assign
    let enc1 = commandBuffer.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(assignPipeline)
    enc1.setBuffer(xBuffer, offset: 0, index: 0)
    enc1.setBuffer(cBuffer, offset: 0, index: 1)
    enc1.setBuffer(aBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    var kUint = UInt32(k)
    enc1.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)
    enc1.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)
    enc1.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 5)
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let gSize = MTLSize(width: n, height: 1, depth: 1)
    enc1.dispatchThreadgroups(gSize, threadsPerThreadgroup: tgSize)
    enc1.endEncoding()

    // 2. Partial update
    let enc2 = commandBuffer.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(partialPipeline)
    enc2.setBuffer(xBuffer, offset: 0, index: 0)
    enc2.setBuffer(aBuffer, offset: 0, index: 1)
    enc2.setBuffer(pcBuffer, offset: 0, index: 2)
    enc2.setBuffer(pcountBuffer, offset: 0, index: 3)
    var ngUint = UInt32(numGroups)
    enc2.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 4)
    enc2.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 5)
    enc2.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 6)
    enc2.setBytes(&ngUint, length: MemoryLayout<UInt32>.stride, index: 7)
    let pgSize = MTLSize(width: numGroups, height: 1, depth: 1)
    enc2.dispatchThreadgroups(pgSize, threadsPerThreadgroup: tgSize)
    enc2.endEncoding()

    // 3. Combine + normalize
    let enc3 = commandBuffer.makeComputeCommandEncoder()!
    enc3.setComputePipelineState(combinePipeline)
    enc3.setBuffer(pcBuffer, offset: 0, index: 0)
    enc3.setBuffer(pcountBuffer, offset: 0, index: 1)
    enc3.setBuffer(coBuffer, offset: 0, index: 2)
    enc3.setBytes(&kUint, length: MemoryLayout<UInt32>.stride, index: 3)
    enc3.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)
    enc3.setBytes(&ngUint, length: MemoryLayout<UInt32>.stride, index: 5)
    let cnSize = MTLSize(width: d, height: k, depth: 1)
    enc3.dispatchThreadgroups(cnSize, threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc3.endEncoding()

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

// MARK: - Batched KMeans: all iterations in one command buffer

@_cdecl("skmetal_kmeans_batch")
public func skmetal_kmeans_batch(
    X: UnsafeRawPointer,
    centroids: UnsafeMutableRawPointer,
    assignments: UnsafeMutableRawPointer,
    partialCentroids: UnsafeMutableRawPointer,
    partialCounts: UnsafeMutableRawPointer,
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
    let pcSize = numGroups * k * d * MemoryLayout<Float>.stride
    let pcountSize = numGroups * k * MemoryLayout<UInt32>.stride

    guard let assignPipeline = ctx.getPipeline(name: "kmeans_assign", functionName: "kmeans_assign"),
          let partialPipeline = ctx.getPipeline(name: "kmeans_partial_update", functionName: "kmeans_partial_update"),
          let combinePipeline = ctx.getPipeline(name: "kmeans_combine_normalize", functionName: "kmeans_combine_normalize"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let cBuffer = wrapOutput(centroids, length: cSize, device: ctx.device),
          let aBuffer = wrapOutput(assignments, length: aSize, device: ctx.device),
          let pcBuffer = wrapOutput(partialCentroids, length: pcSize, device: ctx.device),
          let pcountBuffer = wrapOutput(partialCounts, length: pcountSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)

    for _ in 0..<maxIter {
        // Zero partial buffers
        let clearEncoder = commandBuffer.makeBlitCommandEncoder()!
        clearEncoder.fill(buffer: pcBuffer, range: 0..<pcSize, value: 0)
        clearEncoder.fill(buffer: pcountBuffer, range: 0..<pcountSize, value: 0)
        clearEncoder.endEncoding()

        // 1. Assign
        let enc1 = commandBuffer.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(assignPipeline)
        enc1.setBuffer(xBuffer, offset: 0, index: 0)
        enc1.setBuffer(cBuffer, offset: 0, index: 1)
        enc1.setBuffer(aBuffer, offset: 0, index: 2)
        var nU = UInt32(n); var dU = UInt32(d); var kU = UInt32(k)
        enc1.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc1.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc1.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc1.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
        enc1.endEncoding()

        // 2. Partial update
        let enc2 = commandBuffer.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(partialPipeline)
        enc2.setBuffer(xBuffer, offset: 0, index: 0)
        enc2.setBuffer(aBuffer, offset: 0, index: 1)
        enc2.setBuffer(pcBuffer, offset: 0, index: 2)
        enc2.setBuffer(pcountBuffer, offset: 0, index: 3)
        var ngU = UInt32(numGroups)
        enc2.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc2.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc2.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 6)
        enc2.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 7)
        enc2.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
        enc2.endEncoding()

        // 3. Combine + normalize (writes directly to cBuffer for next iteration)
        let enc3 = commandBuffer.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(combinePipeline)
        enc3.setBuffer(pcBuffer, offset: 0, index: 0)
        enc3.setBuffer(pcountBuffer, offset: 0, index: 1)
        enc3.setBuffer(cBuffer, offset: 0, index: 2)
        enc3.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc3.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc3.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc3.dispatchThreadgroups(MTLSize(width: d, height: k, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc3.endEncoding()
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KMeans fused batch (single dispatch per iteration, double-buffered)

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

    var nU = UInt32(n), dU = UInt32(d), kU = UInt32(k), ngU = UInt32(numGroups)

    guard let assignPartialPipeline = ctx.getPipeline(name: "kmeans_assign_partial", functionName: "kmeans_assign_partial"),
          let combineNormPipeline = ctx.getPipeline(name: "kmeans_combine_normalize", functionName: "kmeans_combine_normalize") else {
        return 1
    }

    let pcSize = numGroups * k * d * MemoryLayout<Float>.stride
    let pnSize = numGroups * k * MemoryLayout<UInt32>.stride
    guard let pcBuffer = ctx.device.makeBuffer(length: pcSize, options: .storageModeShared),
          let pnBuffer = ctx.device.makeBuffer(length: pnSize, options: .storageModeShared) else {
        return 1
    }

    // Zero partial buffers once (subsequent iterations re-zero by the kernel)
    let clearEnc = commandBuffer.makeBlitCommandEncoder()!
    clearEnc.fill(buffer: pcBuffer, range: 0..<pcSize, value: 0)
    clearEnc.fill(buffer: pnBuffer, range: 0..<pnSize, value: 0)
    clearEnc.endEncoding()

    for _ in 0..<maxIter {
        // 1. Assign + partial accumulate (fused, 1 dispatch)
        let enc1 = commandBuffer.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(assignPartialPipeline)
        enc1.setBuffer(xBuffer, offset: 0, index: 0)
        enc1.setBuffer(cBuffer, offset: 0, index: 1)
        enc1.setBuffer(pcBuffer, offset: 0, index: 2)
        enc1.setBuffer(pnBuffer, offset: 0, index: 3)
        enc1.setBuffer(aBuffer, offset: 0, index: 4)
        enc1.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc1.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 6)
        enc1.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
        enc1.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 8)
        enc1.dispatchThreadgroups(gridSize, threadsPerThreadgroup: tgSize)
        enc1.endEncoding()

        // 2. Combine + normalize (1 dispatch, overwrites cBuffer with new centroids)
        let enc2 = commandBuffer.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(combineNormPipeline)
        enc2.setBuffer(pcBuffer, offset: 0, index: 0)
        enc2.setBuffer(pnBuffer, offset: 0, index: 1)
        enc2.setBuffer(cBuffer, offset: 0, index: 2)
        enc2.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc2.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc2.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc2.dispatchThreadgroups(MTLSize(width: d, height: k, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc2.endEncoding()
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return 0
}

// MARK: - Pipeline warmup

@_cdecl("skmetal_warmup")
public func skmetal_warmup() -> Int32 {
    let ctx = MetalContext.shared
    let pipelineNames: [(name: String, function: String)] = [
        ("reduce_sum", "reduce_sum"),
        ("reduce_mean_var", "reduce_mean_var"),
        ("pairwise_distance_direct", "pairwise_distance_direct"),
        ("row_norm_sq", "row_norm_sq"),
        ("distance_correct", "distance_correct"),
        ("argmin_rows", "argmin_rows"),
        ("scaler_fit", "scaler_fit"),
        ("column_minmax", "column_minmax"),
        ("irls_weight", "irls_weight"),
        ("scale_rows", "scale_rows"),
        ("sigmoid", "sigmoid"),
        ("subtract", "subtract"),
        ("axpy", "axpy"),
        ("norm_sq", "norm_sq"),
        ("add_scalar", "add_scalar"),
        ("column_means", "column_means"),
        ("center_columns", "center_columns"),
        ("compute_mindists", "compute_mindists"),
        ("kmeans_assign", "kmeans_assign"),
        ("kmeans_partial_update", "kmeans_partial_update"),
        ("kmeans_combine", "kmeans_combine"),
        ("kmeans_normalize", "kmeans_normalize"),
        ("kmeans_combine_normalize", "kmeans_combine_normalize"),
        ("kmeans_assign_partial", "kmeans_assign_partial"),
        ("kmeans_update", "kmeans_update"),
    ]

    let cb = ctx.commandQueue.makeCommandBuffer()!

    for (name, funcName) in pipelineNames {
        guard let pipeline = ctx.getPipeline(name: name, functionName: funcName) else {
            continue
        }
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
    }

    cb.commit()
    cb.waitUntilCompleted()

    // Also trigger MPS pipeline compilation via a trivial GEMM
    let aSize = 4 * MemoryLayout<Float>.stride
    let bSize = 4 * MemoryLayout<Float>.stride
    let cSize = 4 * MemoryLayout<Float>.stride
    if let aBuf = ctx.device.makeBuffer(length: aSize, options: .storageModeShared),
       let bBuf = ctx.device.makeBuffer(length: bSize, options: .storageModeShared),
       let cBuf = ctx.device.makeBuffer(length: cSize, options: .storageModeShared) {
        let desc2x2 = MPSMatrixDescriptor(dimensions: 2, columns: 2, rowBytes: 8, dataType: .float32)
        let mA = MPSMatrix(buffer: aBuf, descriptor: desc2x2)
        let mB = MPSMatrix(buffer: bBuf, descriptor: desc2x2)
        let mC = MPSMatrix(buffer: cBuf, descriptor: desc2x2)
        let gemm = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: false, transposeRight: false,
            resultRows: 2, resultColumns: 2, interiorColumns: 2,
            alpha: 1.0, beta: 0.0)
        let cb2 = ctx.commandQueue.makeCommandBuffer()!
        gemm.encode(commandBuffer: cb2, leftMatrix: mA, rightMatrix: mB, resultMatrix: mC)
        cb2.commit()
        cb2.waitUntilCompleted()
    }

    return 0
}

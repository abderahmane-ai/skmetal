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

    guard let assignPipeline = ctx.getPipeline(name: "kmeans_assign", functionName: "kmeans_assign"),
          let partialPipeline = ctx.getPipeline(name: "kmeans_partial_sum", functionName: "kmeans_partial_sum"),
          let combineNormPipeline = ctx.getPipeline(name: "kmeans_combine_normalize", functionName: "kmeans_combine_normalize"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let cBuffer = wrapOutput(centroids, length: cSize, device: ctx.device),
          let aBuffer = wrapOutput(assignments, length: aSize, device: ctx.device) else {
        return 1
    }

    let pcSize = numGroups * k * d * MemoryLayout<Float>.stride
    let pnSize = numGroups * k * MemoryLayout<UInt32>.stride
    guard let pcBuffer = ctx.device.makeBuffer(length: pcSize, options: .storageModeShared),
          let pnBuffer = ctx.device.makeBuffer(length: pnSize, options: .storageModeShared) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let assignGrid = MTLSize(width: n, height: 1, depth: 1)
    let partialGrid = MTLSize(width: numGroups, height: 1, depth: 1)

    var nU = UInt32(n), dU = UInt32(d), kU = UInt32(k), ngU = UInt32(numGroups)

    // Max clusters that fit in 28 KB threadgroup memory for centroids + 256 uint for counts
    let maxBatchClusters = min(256, max(1, 7168 / d))

    for _ in 0..<maxIter {
        // Zero partial buffers
        let clearEnc = commandBuffer.makeBlitCommandEncoder()!
        clearEnc.fill(buffer: pcBuffer, range: 0..<pcSize, value: 0)
        clearEnc.fill(buffer: pnBuffer, range: 0..<pnSize, value: 0)
        clearEnc.endEncoding()

        // 1. Assign: compute nearest centroid for each point
        let enc1 = commandBuffer.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(assignPipeline)
        enc1.setBuffer(xBuffer, offset: 0, index: 0)
        enc1.setBuffer(cBuffer, offset: 0, index: 1)
        enc1.setBuffer(aBuffer, offset: 0, index: 2)
        enc1.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc1.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc1.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc1.dispatchThreadgroups(assignGrid, threadsPerThreadgroup: tgSize)
        enc1.endEncoding()

        // 2. Partial sum: per-threadgroup accumulation, dispatched per cluster batch
        var clusterStart = 0
        while clusterStart < k {
            let batchK = min(k - clusterStart, maxBatchClusters)
            var csU = UInt32(clusterStart)
            var bkU = UInt32(batchK)

            let enc2 = commandBuffer.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(partialPipeline)
            enc2.setBuffer(xBuffer, offset: 0, index: 0)
            enc2.setBuffer(aBuffer, offset: 0, index: 1)
            enc2.setBuffer(pcBuffer, offset: 0, index: 2)
            enc2.setBuffer(pnBuffer, offset: 0, index: 3)
            enc2.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc2.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc2.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 6)
            enc2.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
            enc2.setBytes(&csU, length: MemoryLayout<UInt32>.stride, index: 8)
            enc2.setBytes(&bkU, length: MemoryLayout<UInt32>.stride, index: 9)
            enc2.dispatchThreadgroups(partialGrid, threadsPerThreadgroup: tgSize)
            enc2.endEncoding()

            clusterStart += batchK
        }

        // 3. Combine + normalize: reduce partials to new centroids
        let enc3 = commandBuffer.makeComputeCommandEncoder()!
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
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KNN vote classification

@_cdecl("skmetal_knn_vote_classify")
public func skmetal_knn_vote_classify(
    indices: UnsafeRawPointer,
    trainLabels: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    N: Int,
    k: Int,
    nTrain: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let idxSize = N * k * MemoryLayout<Int32>.stride
    let predSize = N * MemoryLayout<Float>.stride
    let labelSize = nTrain * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "knn_vote_classify", functionName: "knn_vote_classify"),
          let idxBuffer = wrapInput(indices, length: idxSize, device: ctx.device),
          let tlBuffer = wrapInput(trainLabels, length: labelSize, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(idxBuffer, offset: 0, index: 0)
    encoder.setBuffer(tlBuffer, offset: 0, index: 1)
    encoder.setBuffer(predBuffer, offset: 0, index: 2)
    var nU = UInt32(N); var kU = UInt32(k)
    encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 4)

    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(MTLSize(width: (N + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KNN vote regression

@_cdecl("skmetal_knn_vote_regress")
public func skmetal_knn_vote_regress(
    indices: UnsafeRawPointer,
    trainTargets: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    N: Int,
    k: Int,
    nTrain: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let idxSize = N * k * MemoryLayout<Int32>.stride
    let predSize = N * MemoryLayout<Float>.stride
    let trainSize = nTrain * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "knn_vote_regress", functionName: "knn_vote_regress"),
          let idxBuffer = wrapInput(indices, length: idxSize, device: ctx.device),
          let ttBuffer = wrapInput(trainTargets, length: trainSize, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(idxBuffer, offset: 0, index: 0)
    encoder.setBuffer(ttBuffer, offset: 0, index: 1)
    encoder.setBuffer(predBuffer, offset: 0, index: 2)
    var nU = UInt32(N); var kU = UInt32(k)
    encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 4)

    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(MTLSize(width: (N + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_knn_vote_classify_weighted")
public func skmetal_knn_vote_classify_weighted(
    indices: UnsafeRawPointer,
    distances: UnsafeRawPointer,
    trainLabels: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    N: Int,
    k: Int,
    nTrain: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let idxSize = N * k * MemoryLayout<Int32>.stride
    let distSize = N * k * MemoryLayout<Float>.stride
    let predSize = N * MemoryLayout<Float>.stride
    let trainSize = nTrain * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "knn_vote_classify_weighted",
                                          functionName: "knn_vote_classify_weighted"),
          let idxBuffer = wrapInput(indices, length: idxSize, device: ctx.device),
          let distBuffer = wrapInput(distances, length: distSize, device: ctx.device),
          let tlBuffer = wrapInput(trainLabels, length: trainSize, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(idxBuffer, offset: 0, index: 0)
    enc.setBuffer(distBuffer, offset: 0, index: 1)
    enc.setBuffer(tlBuffer, offset: 0, index: 2)
    enc.setBuffer(predBuffer, offset: 0, index: 3)
    var nU = UInt32(N); var kU = UInt32(k)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(MTLSize(width: (N + 255) / 256, height: 1, depth: 1),
                              threadsPerThreadgroup: tg)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_knn_vote_regress_weighted")
public func skmetal_knn_vote_regress_weighted(
    indices: UnsafeRawPointer,
    distances: UnsafeRawPointer,
    trainTargets: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    N: Int,
    k: Int,
    nTrain: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let idxSize = N * k * MemoryLayout<Int32>.stride
    let distSize = N * k * MemoryLayout<Float>.stride
    let predSize = N * MemoryLayout<Float>.stride
    let trainSize = nTrain * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "knn_vote_regress_weighted",
                                          functionName: "knn_vote_regress_weighted"),
          let idxBuffer = wrapInput(indices, length: idxSize, device: ctx.device),
          let distBuffer = wrapInput(distances, length: distSize, device: ctx.device),
          let ttBuffer = wrapInput(trainTargets, length: trainSize, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(idxBuffer, offset: 0, index: 0)
    enc.setBuffer(distBuffer, offset: 0, index: 1)
    enc.setBuffer(ttBuffer, offset: 0, index: 2)
    enc.setBuffer(predBuffer, offset: 0, index: 3)
    var nU = UInt32(N); var kU = UInt32(k)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(MTLSize(width: (N + 255) / 256, height: 1, depth: 1),
                              threadsPerThreadgroup: tg)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Tiled KNN: process training data in tiles to avoid materializing full N×M distance matrix

@_cdecl("skmetal_knn_tiled_kneighbors")
public func skmetal_knn_tiled_kneighbors(
    XQuery: UnsafeRawPointer,
    XTrain: UnsafeRawPointer,
    outIndices: UnsafeMutableRawPointer,
    outValues: UnsafeMutableRawPointer,
    nQ: Int,
    nT: Int,
    d: Int,
    k: Int,
    tileSize: Int,
    metric: Int32
) -> Int32 {
    let ctx = MetalContext.shared
    guard k > 0 else { return 1 }
    let fs = MemoryLayout<Float>.stride
    let isize = MemoryLayout<Int32>.stride
    let querySize = nQ * d * fs
    let trainSize = nT * d * fs
    let kSize = nQ * k * fs
    let kIdxSize = nQ * k * isize

    guard let xQueryBuffer = wrapInput(XQuery, length: querySize, device: ctx.device),
          let xTrainBuffer = wrapInput(XTrain, length: trainSize, device: ctx.device),
          let outValsBuffer = wrapOutput(outValues, length: kSize, device: ctx.device),
          let outIdxsBuffer = wrapOutput(outIndices, length: kIdxSize, device: ctx.device) else {
        return 1
    }

    let isCosine = metric == 2
    let isManhattan = metric == 1

    // Allocate norm buffers (not needed for Manhattan)
    let rqSize = nQ * fs
    let rtSize = nT * fs
    let rqBuffer = (!isManhattan) ? ctx.device.makeBuffer(length: rqSize, options: .storageModeShared) : nil
    let rtBuffer = (!isManhattan) ? ctx.device.makeBuffer(length: rtSize, options: .storageModeShared) : nil

    let cb = ctx.commandQueue.makeCommandBuffer()!

    // Row norms (Euclidean and Cosine only)
    if !isManhattan, let normPpl = ctx.getPipeline(name: "row_norm_sq", functionName: "row_norm_sq"),
       let rq = rqBuffer, let rt = rtBuffer {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(normPpl)
        enc.setBuffer(xQueryBuffer, offset: 0, index: 0)
        enc.setBuffer(rq, offset: 0, index: 1)
        var nqU = UInt32(nQ); var dU = UInt32(d)
        enc.setBytes(&nqU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: nQ, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()

        let enc2 = cb.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(normPpl)
        enc2.setBuffer(xTrainBuffer, offset: 0, index: 0)
        enc2.setBuffer(rt, offset: 0, index: 1)
        var ntU = UInt32(nT)
        enc2.setBytes(&ntU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc2.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc2.dispatchThreadgroups(MTLSize(width: nT, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc2.endEncoding()
    }

    // Allocate global top-k buffer and initialize to INFINITY
    guard let gValsBuffer = ctx.device.makeBuffer(length: kSize, options: .storageModeShared),
          let gIdxsBuffer = ctx.device.makeBuffer(length: kIdxSize, options: .storageModeShared) else {
        return 1
    }
    let gValsPtr = gValsBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(nQ * k) { gValsPtr[i] = .infinity }
    memset(gIdxsBuffer.contents(), 0, kIdxSize)

    // Select pipeline by metric
    let selectName: String
    let selectFunc: String
    if isManhattan {
        selectName = "knn_select_tile_topk_manhattan"
        selectFunc = "knn_select_tile_topk_manhattan"
    } else if isCosine {
        selectName = "knn_select_tile_topk_cosine"
        selectFunc = "knn_select_tile_topk_cosine"
    } else {
        selectName = "knn_select_tile_topk"
        selectFunc = "knn_select_tile_topk"
    }
    guard let selectPipeline = ctx.getPipeline(name: selectName, functionName: selectFunc),
          let mergePipeline = ctx.getPipeline(name: "knn_merge_topk", functionName: "knn_merge_topk") else {
        return 1
    }

    // Reusable tile scratch buffers
    let maxTileN = min(tileSize, nT)
    let dotSize = nQ * maxTileN * fs
    let tileValsSize = nQ * k * fs
    let tileIdxsSize = nQ * k * isize

    // Dot buffer only needed for Euclidean/Cosine
    let dotBuffer = (!isManhattan) ? ctx.device.makeBuffer(length: dotSize, options: .storageModeShared) : nil
    guard let tValsBuffer = ctx.device.makeBuffer(length: tileValsSize, options: .storageModeShared),
          let tIdxsBuffer = ctx.device.makeBuffer(length: tileIdxsSize, options: .storageModeShared) else {
        return 1
    }

    let tempValsBuffer = ctx.device.makeBuffer(length: tileValsSize, options: .storageModeShared)
    let tempIdxsBuffer = ctx.device.makeBuffer(length: tileIdxsSize, options: .storageModeShared)

    // MPS query matrix (Euclidean/Cosine)
    let descQ = MPSMatrixDescriptor(dimensions: nQ, columns: d,
                                     rowBytes: d * fs, dataType: .float32)
    let matrixQ = MPSMatrix(buffer: xQueryBuffer, descriptor: descQ)

    let tg1 = MTLSize(width: 1, height: 1, depth: 1)

    var tileStart = 0
    while tileStart < nT {
        let tileEnd = min(tileStart + tileSize, nT)
        let tileN = tileEnd - tileStart

        // X_train tile slice (all metrics)
        let trainSlicePtr = xTrainBuffer.contents().advanced(by: tileStart * d * fs)
        guard let trainSliceBuffer = ctx.device.makeBuffer(
            bytesNoCopy: trainSlicePtr,
            length: tileN * d * fs,
            options: .storageModeShared,
            deallocator: nil) else { return 1 }

        if isManhattan {
            // Manhattan select: direct L1, no GEMM or norms
            let encSel = cb.makeComputeCommandEncoder()!
            encSel.setComputePipelineState(selectPipeline)
            encSel.setBuffer(xQueryBuffer, offset: 0, index: 0)
            encSel.setBuffer(trainSliceBuffer, offset: 0, index: 1)
            encSel.setBuffer(tValsBuffer, offset: 0, index: 2)
            encSel.setBuffer(tIdxsBuffer, offset: 0, index: 3)
            var nqU = UInt32(nQ); var tnU = UInt32(tileN); var dU = UInt32(d); var kU = UInt32(k)
            encSel.setBytes(&nqU, length: MemoryLayout<UInt32>.stride, index: 4)
            encSel.setBytes(&tnU, length: MemoryLayout<UInt32>.stride, index: 5)
            encSel.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 6)
            encSel.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
            encSel.dispatchThreadgroups(MTLSize(width: nQ, height: 1, depth: 1),
                                        threadsPerThreadgroup: tg1)
            encSel.endEncoding()
        } else {
            guard let rq = rqBuffer, let rt = rtBuffer, let db = dotBuffer else { return 1 }

            // rt tile slice
            let rtSlicePtr = rt.contents().advanced(by: tileStart * fs)
            guard let rtSliceBuffer = ctx.device.makeBuffer(
                bytesNoCopy: rtSlicePtr,
                length: tileN * fs,
                options: .storageModeShared,
                deallocator: nil) else { return 1 }

            // GEMM — X_query @ X_train_tile^T
            let descTSlice = MPSMatrixDescriptor(dimensions: tileN, columns: d,
                                                 rowBytes: d * fs, dataType: .float32)
            let descDot = MPSMatrixDescriptor(dimensions: nQ, columns: tileN,
                                              rowBytes: tileN * fs, dataType: .float32)
            let matrixTSlice = MPSMatrix(buffer: trainSliceBuffer, descriptor: descTSlice)
            let matrixDot = MPSMatrix(buffer: db, descriptor: descDot)

            let gemm = MPSMatrixMultiplication(
                device: ctx.device, transposeLeft: false, transposeRight: true,
                resultRows: nQ, resultColumns: tileN, interiorColumns: d,
                alpha: 1.0, beta: 0.0)
            gemm.encode(commandBuffer: cb, leftMatrix: matrixQ, rightMatrix: matrixTSlice,
                        resultMatrix: matrixDot)

            // k-select (Euclidean or Cosine)
            let encSel = cb.makeComputeCommandEncoder()!
            encSel.setComputePipelineState(selectPipeline)
            encSel.setBuffer(db, offset: 0, index: 0)
            encSel.setBuffer(rq, offset: 0, index: 1)
            encSel.setBuffer(rtSliceBuffer, offset: 0, index: 2)
            encSel.setBuffer(tValsBuffer, offset: 0, index: 3)
            encSel.setBuffer(tIdxsBuffer, offset: 0, index: 4)
            var nqU = UInt32(nQ); var tnU = UInt32(tileN); var kU = UInt32(k)
            encSel.setBytes(&nqU, length: MemoryLayout<UInt32>.stride, index: 5)
            encSel.setBytes(&tnU, length: MemoryLayout<UInt32>.stride, index: 6)
            encSel.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
            encSel.dispatchThreadgroups(MTLSize(width: nQ, height: 1, depth: 1),
                                        threadsPerThreadgroup: tg1)
            encSel.endEncoding()
        }

        // Merge tile-local top-k into global top-k
        let encMerge = cb.makeComputeCommandEncoder()!
        encMerge.setComputePipelineState(mergePipeline)
        encMerge.setBuffer(tValsBuffer, offset: 0, index: 0)
        encMerge.setBuffer(tIdxsBuffer, offset: 0, index: 1)
        encMerge.setBuffer(gValsBuffer, offset: 0, index: 2)
        encMerge.setBuffer(gIdxsBuffer, offset: 0, index: 3)
        if let tvb = tempValsBuffer { encMerge.setBuffer(tvb, offset: 0, index: 4) }
        if let tib = tempIdxsBuffer { encMerge.setBuffer(tib, offset: 0, index: 5) }
        var nqU = UInt32(nQ); var kU = UInt32(k)
        encMerge.setBytes(&nqU, length: MemoryLayout<UInt32>.stride, index: 6)
        encMerge.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
        var tsU = UInt32(tileStart)
        encMerge.setBytes(&tsU, length: MemoryLayout<UInt32>.stride, index: 8)
        encMerge.dispatchThreadgroups(MTLSize(width: nQ, height: 1, depth: 1),
                                      threadsPerThreadgroup: tg1)
        encMerge.endEncoding()

        tileStart += tileSize
    }

    cb.commit()
    cb.waitUntilCompleted()

    memcpy(outValsBuffer.contents(), gValsBuffer.contents(), kSize)
    memcpy(outIdxsBuffer.contents(), gIdxsBuffer.contents(), kIdxSize)
    return 0
}

// MARK: - Soft threshold (Lasso FISTA)

@_cdecl("skmetal_soft_threshold")
public func skmetal_soft_threshold(
    w: UnsafeMutableRawPointer,
    wTemp: UnsafeRawPointer,
    threshold: Float,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "soft_threshold", functionName: "soft_threshold"),
          let wBuffer = wrapOutput(w, length: byteSize, device: ctx.device),
          let wtBuffer = wrapInput(wTemp, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(wBuffer, offset: 0, index: 0)
    encoder.setBuffer(wtBuffer, offset: 0, index: 1)
    var thresh = threshold
    encoder.setBytes(&thresh, length: MemoryLayout<Float>.stride, index: 2)
    var nU = UInt32(n)
    encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)

    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - GPU-resident FISTA (Lasso/ElasticNet)

@_cdecl("skmetal_fista_fit")
public func skmetal_fista_fit(
    X: UnsafeRawPointer,
    y: UnsafeRawPointer,
    coef_out: UnsafeMutableRawPointer,
    n: Int,
    p: Int,
    alpha: Float,
    l1_ratio: Float,
    tol: Float,
    max_iter: Int32,
    n_iter_out: UnsafeMutablePointer<Int32>?
) -> Int32 {
    let ctx = MetalContext.shared
    guard p > 0 else { return 1 }
    guard max_iter > 0 else {
        memset(coef_out, 0, p * MemoryLayout<Float>.stride)
        n_iter_out?.pointee = 0
        return 0
    }

    let fs = MemoryLayout<Float>.stride
    let xBufSize = n * p * fs
    let yBufSize = n * fs
    let pBufSize = p * fs
    let ppBufSize = p * p * fs

    guard let xBuf = wrapInput(X, length: xBufSize, device: ctx.device),
          let yBuf = wrapInput(y, length: yBufSize, device: ctx.device),
          let coefBuf = wrapOutput(coef_out, length: pBufSize, device: ctx.device) else {
        return 1
    }

    // GPU-resident buffers (shared = HBM on Apple Silicon)
    guard let xtxBuf = ctx.device.makeBuffer(length: ppBufSize, options: .storageModeShared),
          let xtyBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let xBuf_g = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let zBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let xPrevBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let xTempBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let gradBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared) else {
        return 1
    }

    // Initialize x, z to zero
    memset(xBuf_g.contents(), 0, pBufSize)
    memset(zBuf.contents(), 0, pBufSize)

    // Pipelines
    guard let axpyPpl = ctx.getPipeline(name: "axpy", functionName: "axpy"),
          let subPpl = ctx.getPipeline(name: "subtract", functionName: "subtract"),
          let stPpl = ctx.getPipeline(name: "soft_threshold", functionName: "soft_threshold"),
          let scalePpl = ctx.getPipeline(name: "scale_f32", functionName: "scale_f32"),
          let rsPpl = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum"),
          let nsPpl = ctx.getPipeline(name: "norm_sq", functionName: "norm_sq") else {
        return 1
    }

    // MPS matrix descriptors
    let xtxDesc = MPSMatrixDescriptor(dimensions: p, columns: p,
                                       rowBytes: p * fs, dataType: .float32)
    let colDesc = MPSMatrixDescriptor(dimensions: p, columns: 1,
                                       rowBytes: fs, dataType: .float32)
    let xDesc = MPSMatrixDescriptor(dimensions: n, columns: p,
                                     rowBytes: p * fs, dataType: .float32)
    let yDesc = MPSMatrixDescriptor(dimensions: n, columns: 1,
                                     rowBytes: fs, dataType: .float32)

    let mXTX = MPSMatrix(buffer: xtxBuf, descriptor: xtxDesc)
    let mXTy = MPSMatrix(buffer: xtyBuf, descriptor: colDesc)
    let mX = MPSMatrix(buffer: xBuf, descriptor: xDesc)
    let mY = MPSMatrix(buffer: yBuf, descriptor: yDesc)
    let mZ = MPSMatrix(buffer: zBuf, descriptor: colDesc)
    let mGrad = MPSMatrix(buffer: gradBuf, descriptor: colDesc)

    // Step 1: XTX = X^T @ X
    do {
        let cb = ctx.commandQueue.makeCommandBuffer()!
        let gemm = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: p, interiorColumns: n,
            alpha: 1.0, beta: 0.0)
        gemm.encode(commandBuffer: cb, leftMatrix: mX, rightMatrix: mX, resultMatrix: mXTX)
        cb.commit()
        cb.waitUntilCompleted()
    }

    // Step 2: XTy = X^T @ y
    do {
        let cb = ctx.commandQueue.makeCommandBuffer()!
        let gemm = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: 1, interiorColumns: n,
            alpha: 1.0, beta: 0.0)
        gemm.encode(commandBuffer: cb, leftMatrix: mX, rightMatrix: mY, resultMatrix: mXTy)
        cb.commit()
        cb.waitUntilCompleted()
    }

    // Step 3: L = ||XTX||_2 via power iteration
    let L: Float = {
        if p == 1 {
            return abs(xtxBuf.contents().load(as: Float.self))
        }
        guard let vBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
              let uBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
              let sumBuf = ctx.device.makeBuffer(length: fs, options: .storageModeShared) else {
            return 1
        }
        // Init v with first column of XTX
        let vPtr = vBuf.contents().assumingMemoryBound(to: Float.self)
        let xtxPtr = xtxBuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<p { vPtr[i] = xtxPtr[i * p] }

        let mV = MPSMatrix(buffer: vBuf, descriptor: colDesc)
        let mU = MPSMatrix(buffer: uBuf, descriptor: colDesc)

        let powerIters = 20
        for _ in 0..<powerIters {
            let cb = ctx.commandQueue.makeCommandBuffer()!

            let gemm = MPSMatrixMultiplication(
                device: ctx.device, transposeLeft: false, transposeRight: false,
                resultRows: p, resultColumns: 1, interiorColumns: p,
                alpha: 1.0, beta: 0.0)
            gemm.encode(commandBuffer: cb, leftMatrix: mXTX, rightMatrix: mV, resultMatrix: mU)

            // norm_sq: vBuf = uBuf^2
            let encNS = cb.makeComputeCommandEncoder()!
            encNS.setComputePipelineState(nsPpl)
            encNS.setBuffer(uBuf, offset: 0, index: 0)
            encNS.setBuffer(vBuf, offset: 0, index: 1)
            var nU32 = UInt32(p)
            encNS.setBytes(&nU32, length: MemoryLayout<UInt32>.stride, index: 2)
            encNS.dispatchThreadgroups(MTLSize(width: (p + 255) / 256, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encNS.endEncoding()

            // reduce_sum: sum vBuf into sumBuf
            let ng = max(1, (p + 255) / 256)
            let encRS = cb.makeComputeCommandEncoder()!
            encRS.setComputePipelineState(rsPpl)
            encRS.setBuffer(vBuf, offset: 0, index: 0)
            encRS.setBuffer(sumBuf, offset: 0, index: 1)
            nU32 = UInt32(p)
            encRS.setBytes(&nU32, length: MemoryLayout<UInt32>.stride, index: 2)
            var ngU32 = UInt32(ng)
            encRS.setBytes(&ngU32, length: MemoryLayout<UInt32>.stride, index: 3)
            encRS.dispatchThreadgroups(MTLSize(width: ng, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encRS.endEncoding()

            cb.commit()
            cb.waitUntilCompleted()

            let norm = sqrt(sumBuf.contents().load(as: Float.self))
            if norm < 1e-10 { break }

            // Normalize on CPU: v = u / norm
            let uPtr = uBuf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<p { vPtr[i] = uPtr[i] / norm }
        }

        // Rayleigh quotient: L = v^T @ (XTX @ v) / ||v||^2
        // v is normalized so ||v||^2 = 1, and u = XTX @ v, so L = v^T u
        let uPtr = uBuf.contents().assumingMemoryBound(to: Float.self)
        var Lval: Float = 0
        for i in 0..<p { Lval += vPtr[i] * uPtr[i] }
        return abs(Lval)
    }()

    guard L > 1e-10 else { return 1 }
    let step = 1.0 / L
    let thresh = step * alpha * l1_ratio * Float(n)
    let enDenom: Float = (l1_ratio < 1.0) ? (1.0 + step * alpha * (1.0 - l1_ratio) * Float(n)) : 1.0
    let enScale: Float = (enDenom != 1.0) ? (1.0 / enDenom) : 1.0

    // Step 4: FISTA loop
    let checkEvery = 10
    let tg256 = MTLSize(width: 256, height: 1, depth: 1)
    let grd256 = MTLSize(width: (p + 255) / 256, height: 1, depth: 1)
    var t: Float = 1.0
    var it: Int32 = 0

    for itCount in 0..<Int(max_iter) {
        it = Int32(itCount + 1)
        let cb = ctx.commandQueue.makeCommandBuffer()!

        // 1. x_prev = x (blit copy)
        let blit1 = cb.makeBlitCommandEncoder()!
        blit1.copy(from: xBuf_g, sourceOffset: 0, to: xPrevBuf, destinationOffset: 0, size: pBufSize)
        blit1.endEncoding()

        // 2. grad = XTX @ z (MPS GEMM)
        let gemm = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: false, transposeRight: false,
            resultRows: p, resultColumns: 1, interiorColumns: p,
            alpha: 1.0, beta: 0.0)
        gemm.encode(commandBuffer: cb, leftMatrix: mXTX, rightMatrix: mZ, resultMatrix: mGrad)

        // 3. grad -= XTy
        let encGradSub = cb.makeComputeCommandEncoder()!
        encGradSub.setComputePipelineState(axpyPpl)
        encGradSub.setBuffer(gradBuf, offset: 0, index: 0)
        encGradSub.setBuffer(xtyBuf, offset: 0, index: 1)
        var m1: Float = -1.0
        encGradSub.setBytes(&m1, length: fs, index: 2)
        var nU = UInt32(p)
        encGradSub.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        encGradSub.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
        encGradSub.endEncoding()

        // 4. x_temp = z (blit copy)
        let blit2 = cb.makeBlitCommandEncoder()!
        blit2.copy(from: zBuf, sourceOffset: 0, to: xTempBuf, destinationOffset: 0, size: pBufSize)
        blit2.endEncoding()

        // 5. x_temp += -step * grad  → x_temp = z - step*grad
        let encStep = cb.makeComputeCommandEncoder()!
        encStep.setComputePipelineState(axpyPpl)
        encStep.setBuffer(xTempBuf, offset: 0, index: 0)
        encStep.setBuffer(gradBuf, offset: 0, index: 1)
        var negStep: Float = -step
        encStep.setBytes(&negStep, length: fs, index: 2)
        nU = UInt32(p)
        encStep.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        encStep.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
        encStep.endEncoding()

        // 6. x = soft_threshold(x_temp, threshold)
        let encST = cb.makeComputeCommandEncoder()!
        encST.setComputePipelineState(stPpl)
        encST.setBuffer(xBuf_g, offset: 0, index: 0)
        encST.setBuffer(xTempBuf, offset: 0, index: 1)
        var thr = thresh
        encST.setBytes(&thr, length: fs, index: 2)
        nU = UInt32(p)
        encST.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        encST.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
        encST.endEncoding()

        // 7. ElasticNet: x *= enScale  (skip for pure Lasso)
        if enScale != 1.0 {
            let encScale = cb.makeComputeCommandEncoder()!
            encScale.setComputePipelineState(scalePpl)
            encScale.setBuffer(xBuf_g, offset: 0, index: 0)
            var sc = enScale
            encScale.setBytes(&sc, length: fs, index: 1)
            nU = UInt32(p)
            encScale.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
            encScale.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
            encScale.endEncoding()
        }

        // Nesterov momentum
        let tPrev = t
        t = (1.0 + sqrt(1.0 + 4.0 * tPrev * tPrev)) / 2.0
        let factor = (tPrev - 1.0) / t

        // 8. x_temp = x - x_prev
        let encSub = cb.makeComputeCommandEncoder()!
        encSub.setComputePipelineState(subPpl)
        encSub.setBuffer(xBuf_g, offset: 0, index: 0)
        encSub.setBuffer(xPrevBuf, offset: 0, index: 1)
        encSub.setBuffer(xTempBuf, offset: 0, index: 2)
        nU = UInt32(p)
        encSub.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        encSub.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
        encSub.endEncoding()

        // 9. z = x (blit copy)
        let blit3 = cb.makeBlitCommandEncoder()!
        blit3.copy(from: xBuf_g, sourceOffset: 0, to: zBuf, destinationOffset: 0, size: pBufSize)
        blit3.endEncoding()

        // 10. z += factor * x_temp  → z = x + factor*(x - x_prev)
        let encZUp = cb.makeComputeCommandEncoder()!
        encZUp.setComputePipelineState(axpyPpl)
        encZUp.setBuffer(zBuf, offset: 0, index: 0)
        encZUp.setBuffer(xTempBuf, offset: 0, index: 1)
        var fac = factor
        encZUp.setBytes(&fac, length: fs, index: 2)
        nU = UInt32(p)
        encZUp.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        encZUp.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
        encZUp.endEncoding()

        cb.commit()
        cb.waitUntilCompleted()

        // Convergence check
        if itCount % checkEvery == 0 || itCount == Int(max_iter) - 1 {
            let xP = xBuf_g.contents().assumingMemoryBound(to: Float.self)
            let xPrevP = xPrevBuf.contents().assumingMemoryBound(to: Float.self)
            var diff: Float = 0
            for i in 0..<p {
                let d = abs(xP[i] - xPrevP[i])
                if d > diff { diff = d }
            }
            if diff < tol { break }
        }
    }

    memcpy(coefBuf.contents(), xBuf_g.contents(), pBufSize)
    n_iter_out?.pointee = it
    return 0
}

// MARK: - Column transform (RobustScaler / StandardScaler)

@_cdecl("skmetal_column_transform")
public func skmetal_column_transform(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    center: UnsafeRawPointer,
    scale: UnsafeRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * d * MemoryLayout<Float>.stride
    let statSize = d * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "column_transform", functionName: "column_transform"),
          let inBuffer = wrapInput(input, length: matSize, device: ctx.device),
          let outBuffer = wrapOutput(output, length: matSize, device: ctx.device),
          let centerBuffer = wrapInput(center, length: statSize, device: ctx.device),
          let scaleBuffer = wrapInput(scale, length: statSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inBuffer, offset: 0, index: 0)
    encoder.setBuffer(outBuffer, offset: 0, index: 1)
    encoder.setBuffer(centerBuffer, offset: 0, index: 2)
    encoder.setBuffer(scaleBuffer, offset: 0, index: 3)
    var nU = UInt32(n); var dU = UInt32(d)
    encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 5)

    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(MTLSize(width: (n * d + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - GPU Alignment: transpose f32

@_cdecl("skmetal_transpose_f32")
public func skmetal_transpose_f32(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    rows: Int,
    cols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = rows * cols * MemoryLayout<Float>.stride
    guard let pipeline = ctx.getPipeline(name: "transpose_f32", functionName: "transpose_f32"),
          let inBuffer = wrapInput(input, length: byteSize, device: ctx.device),
          let outBuffer = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(inBuffer, offset: 0, index: 0)
    enc.setBuffer(outBuffer, offset: 0, index: 1)
    var rowsU = UInt32(rows); var colsU = UInt32(cols)
    enc.setBytes(&rowsU, length: MemoryLayout<UInt32>.stride, index: 2)
    enc.setBytes(&colsU, length: MemoryLayout<UInt32>.stride, index: 3)
    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let tgCount = MTLSize(width: (cols + 15) / 16, height: (rows + 15) / 16, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Shiloach-Vishkin: init parent array

@_cdecl("skmetal_sv_init")
public func skmetal_sv_init(
    parent: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Int32>.stride
    guard let pipeline = ctx.getPipeline(name: "sv_init", functionName: "sv_init"),
          let parentBuffer = wrapOutput(parent, length: byteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(parentBuffer, offset: 0, index: 0)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 1)
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let tgCount = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Shiloach-Vishkin: hook phase

@_cdecl("skmetal_sv_hook")
public func skmetal_sv_hook(
    edges: UnsafeRawPointer,
    parent: UnsafeMutableRawPointer,
    edgeCount: Int,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let edgeByteSize = edgeCount * 2 * MemoryLayout<Int32>.stride
    let parentByteSize = n * MemoryLayout<Int32>.stride
    guard let pipeline = ctx.getPipeline(name: "sv_hook", functionName: "sv_hook"),
          let edgesBuffer = wrapInput(edges, length: edgeByteSize, device: ctx.device),
          let parentBuffer = wrapOutput(parent, length: parentByteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(edgesBuffer, offset: 0, index: 0)
    enc.setBuffer(parentBuffer, offset: 0, index: 1)
    var ecU = UInt32(edgeCount)
    enc.setBytes(&ecU, length: MemoryLayout<UInt32>.stride, index: 2)
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let tgCount = MTLSize(width: (Int(edgeCount) + 255) / 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Shiloach-Vishkin: shortcut phase

@_cdecl("skmetal_sv_shortcut")
public func skmetal_sv_shortcut(
    parent: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Int32>.stride
    guard let pipeline = ctx.getPipeline(name: "sv_shortcut", functionName: "sv_shortcut"),
          let parentBuffer = wrapOutput(parent, length: byteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(parentBuffer, offset: 0, index: 0)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 1)
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let tgCount = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Tree Predict (GPU)

@_cdecl("skmetal_tree_predict_all")
public func skmetal_tree_predict_all(
    X: UnsafeRawPointer,
    allTreeValues: UnsafeRawPointer,
    allTreeFeature: UnsafeRawPointer,
    allTreeThreshold: UnsafeRawPointer,
    allTreeLeft: UnsafeRawPointer,
    allTreeRight: UnsafeRawPointer,
    allTreeIsLeaf: UnsafeRawPointer,
    treeOffsets: UnsafeRawPointer,
    treeNNodes: UnsafeRawPointer,
    baseline: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    n: Int,
    nFeatures: Int,
    nTrees: Int
) -> Int32 {
    let ctx = MetalContext.shared
    // Compute total nodes sum for flat array sizes
    let totalNodesPtr = treeNNodes.assumingMemoryBound(to: UInt32.self)
    var totalNodes: UInt32 = 0
    for i in 0..<nTrees { totalNodes += totalNodesPtr[i] }
    let tn = Int(totalNodes)

    let xSize = n * nFeatures * MemoryLayout<Float>.stride
    let arrSize = tn * MemoryLayout<Float>.stride
    let intSize = tn * MemoryLayout<Int32>.stride
    let offSize = nTrees * MemoryLayout<UInt32>.stride
    let predSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "tree_predict_all", functionName: "tree_predict_all"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let tvBuffer = wrapInput(allTreeValues, length: arrSize, device: ctx.device),
          let tfBuffer = wrapInput(allTreeFeature, length: intSize, device: ctx.device),
          let ttBuffer = wrapInput(allTreeThreshold, length: arrSize, device: ctx.device),
          let tlBuffer = wrapInput(allTreeLeft, length: intSize, device: ctx.device),
          let trBuffer = wrapInput(allTreeRight, length: intSize, device: ctx.device),
          let tleafBuffer = wrapInput(allTreeIsLeaf, length: tn * MemoryLayout<UInt8>.stride, device: ctx.device),
          let offBuffer = wrapInput(treeOffsets, length: offSize, device: ctx.device),
          let nnBuffer = wrapInput(treeNNodes, length: offSize, device: ctx.device),
          let blBuffer = wrapInput(baseline, length: MemoryLayout<Float>.stride, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(xBuffer, offset: 0, index: 0)
    enc.setBuffer(tvBuffer, offset: 0, index: 1)
    enc.setBuffer(tfBuffer, offset: 0, index: 2)
    enc.setBuffer(ttBuffer, offset: 0, index: 3)
    enc.setBuffer(tlBuffer, offset: 0, index: 4)
    enc.setBuffer(trBuffer, offset: 0, index: 5)
    enc.setBuffer(tleafBuffer, offset: 0, index: 6)
    enc.setBuffer(offBuffer, offset: 0, index: 7)
    enc.setBuffer(nnBuffer, offset: 0, index: 8)
    enc.setBuffer(blBuffer, offset: 0, index: 9)
    enc.setBuffer(predBuffer, offset: 0, index: 10)
    var nU = UInt32(n); var nfU = UInt32(nFeatures); var ntU = UInt32(nTrees)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 11)
    enc.setBytes(&nfU, length: MemoryLayout<UInt32>.stride, index: 12)
    enc.setBytes(&ntU, length: MemoryLayout<UInt32>.stride, index: 13)
    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_tree_predict")
public func skmetal_tree_predict(
    X: UnsafeRawPointer,
    treeValues: UnsafeRawPointer,
    treeFeature: UnsafeRawPointer,
    treeThreshold: UnsafeRawPointer,
    treeLeft: UnsafeRawPointer,
    treeRight: UnsafeRawPointer,
    treeIsLeaf: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    n: Int,
    nFeatures: Int,
    nNodes: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * nFeatures * MemoryLayout<Float>.stride
    let nodeSize = nNodes * MemoryLayout<Float>.stride
    let intSize = nNodes * MemoryLayout<Int32>.stride
    let predSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "tree_predict", functionName: "tree_predict"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let tvBuffer = wrapInput(treeValues, length: nodeSize, device: ctx.device),
          let tfBuffer = wrapInput(treeFeature, length: intSize, device: ctx.device),
          let ttBuffer = wrapInput(treeThreshold, length: nodeSize, device: ctx.device),
          let tlBuffer = wrapInput(treeLeft, length: intSize, device: ctx.device),
          let trBuffer = wrapInput(treeRight, length: intSize, device: ctx.device),
          let tleafBuffer = wrapInput(treeIsLeaf, length: nNodes * MemoryLayout<UInt8>.stride, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(xBuffer, offset: 0, index: 0)
    enc.setBuffer(tvBuffer, offset: 0, index: 1)
    enc.setBuffer(tfBuffer, offset: 0, index: 2)
    enc.setBuffer(ttBuffer, offset: 0, index: 3)
    enc.setBuffer(tlBuffer, offset: 0, index: 4)
    enc.setBuffer(trBuffer, offset: 0, index: 5)
    enc.setBuffer(tleafBuffer, offset: 0, index: 6)
    enc.setBuffer(predBuffer, offset: 0, index: 7)
    var nU = UInt32(n); var nfU = UInt32(nFeatures); var nnU = UInt32(nNodes)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 8)
    enc.setBytes(&nfU, length: MemoryLayout<UInt32>.stride, index: 9)
    enc.setBytes(&nnU, length: MemoryLayout<UInt32>.stride, index: 10)
    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Row max (for softmax)

@_cdecl("skmetal_row_max")
public func skmetal_row_max(
    matrix: UnsafeRawPointer,
    maxVals: UnsafeMutableRawPointer,
    n: Int,
    nCols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * nCols * MemoryLayout<Float>.stride
    let maxSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "row_max", functionName: "row_max"),
          let matBuffer = wrapInput(matrix, length: matSize, device: ctx.device),
          let maxBuffer = wrapOutput(maxVals, length: maxSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(matBuffer, offset: 0, index: 0)
    enc.setBuffer(maxBuffer, offset: 0, index: 1)
    var nU = UInt32(n); var ncU = UInt32(nCols)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    enc.setBytes(&ncU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Row sum (for softmax)

@_cdecl("skmetal_row_sum")
public func skmetal_row_sum(
    matrix: UnsafeRawPointer,
    sums: UnsafeMutableRawPointer,
    n: Int,
    nCols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * nCols * MemoryLayout<Float>.stride
    let sumSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "row_sum", functionName: "row_sum"),
          let matBuffer = wrapInput(matrix, length: matSize, device: ctx.device),
          let sumBuffer = wrapOutput(sums, length: sumSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(matBuffer, offset: 0, index: 0)
    enc.setBuffer(sumBuffer, offset: 0, index: 1)
    var nU = UInt32(n); var ncU = UInt32(nCols)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    enc.setBytes(&ncU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Softmax exp (numerically stable)

@_cdecl("skmetal_softmax_exp")
public func skmetal_softmax_exp(
    matrix: UnsafeRawPointer,
    maxVals: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int,
    nCols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * nCols * MemoryLayout<Float>.stride
    let maxSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "softmax_exp", functionName: "softmax_exp"),
          let matBuffer = wrapInput(matrix, length: matSize, device: ctx.device),
          let maxBuffer = wrapInput(maxVals, length: maxSize, device: ctx.device),
          let outBuffer = wrapOutput(output, length: matSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(matBuffer, offset: 0, index: 0)
    enc.setBuffer(maxBuffer, offset: 0, index: 1)
    enc.setBuffer(outBuffer, offset: 0, index: 2)
    var nU = UInt32(n); var ncU = UInt32(nCols)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.setBytes(&ncU, length: MemoryLayout<UInt32>.stride, index: 4)
    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let tgCount = MTLSize(width: (nCols + 15) / 16, height: (n + 15) / 16, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Softmax normalize + residual

@_cdecl("skmetal_softmax_normalize_residual")
public func skmetal_softmax_normalize_residual(
    prob: UnsafeMutableRawPointer,
    rowSums: UnsafeRawPointer,
    y: UnsafeRawPointer,
    residual: UnsafeMutableRawPointer,
    n: Int,
    nCols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * nCols * MemoryLayout<Float>.stride
    let sumSize = n * MemoryLayout<Float>.stride
    let ySize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "softmax_normalize_residual", functionName: "softmax_normalize_residual"),
          let probBuffer = wrapOutput(prob, length: matSize, device: ctx.device),
          let sumBuffer = wrapInput(rowSums, length: sumSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: ySize, device: ctx.device),
          let resBuffer = wrapOutput(residual, length: matSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(probBuffer, offset: 0, index: 0)
    enc.setBuffer(sumBuffer, offset: 0, index: 1)
    enc.setBuffer(yBuffer, offset: 0, index: 2)
    enc.setBuffer(resBuffer, offset: 0, index: 3)
    var nU = UInt32(n); var ncU = UInt32(nCols)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&ncU, length: MemoryLayout<UInt32>.stride, index: 5)
    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let tgCount = MTLSize(width: (nCols + 15) / 16, height: (n + 15) / 16, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Negate (element-wise)

@_cdecl("skmetal_negate")
public func skmetal_negate(
    a: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride
    guard let pipeline = ctx.getPipeline(name: "negate", functionName: "negate"),
          let aBuffer = wrapInput(a, length: byteSize, device: ctx.device),
          let outBuffer = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(aBuffer, offset: 0, index: 0)
    enc.setBuffer(outBuffer, offset: 0, index: 1)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Multinomial IRLS iteration

@_cdecl("skmetal_multinomial_irls_iter")
public func skmetal_multinomial_irls_iter(
    X: UnsafeRawPointer,
    W: UnsafeRawPointer,
    y: UnsafeRawPointer,
    scores: UnsafeMutableRawPointer,
    prob: UnsafeMutableRawPointer,
    maxVals: UnsafeMutableRawPointer,
    sumExp: UnsafeMutableRawPointer,
    residual: UnsafeMutableRawPointer,
    gradient: UnsafeMutableRawPointer,
    hessians: UnsafeMutableRawPointer,
    n: Int,
    p: Int,
    C: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let nFloat = n * MemoryLayout<Float>.stride
    let pFloat = p * MemoryLayout<Float>.stride
    let xSize = n * p * MemoryLayout<Float>.stride
    let wSize = p * C * MemoryLayout<Float>.stride
    let scoresSize = n * C * MemoryLayout<Float>.stride

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let wBuffer = wrapInput(W, length: wSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: nFloat, device: ctx.device),
          let scoresBuffer = wrapOutput(scores, length: scoresSize, device: ctx.device),
          let probBuffer = wrapOutput(prob, length: scoresSize, device: ctx.device),
          let maxBuffer = wrapOutput(maxVals, length: nFloat, device: ctx.device),
          let sumBuffer = wrapOutput(sumExp, length: nFloat, device: ctx.device),
          let resBuffer = wrapOutput(residual, length: scoresSize, device: ctx.device),
          let gBuffer = wrapOutput(gradient, length: p * C * MemoryLayout<Float>.stride, device: ctx.device),
          let hBuffer = wrapOutput(hessians, length: C * p * p * MemoryLayout<Float>.stride, device: ctx.device) else {
        return 1
    }

    let rowBytesX = p * MemoryLayout<Float>.stride
    let rowBytesC = C * MemoryLayout<Float>.stride
    let rowBytesP = p * MemoryLayout<Float>.stride

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: C, rowBytes: rowBytesC, dataType: .float32)
    let descScores = MPSMatrixDescriptor(dimensions: n, columns: C, rowBytes: rowBytesC, dataType: .float32)
    let descRes = MPSMatrixDescriptor(dimensions: n, columns: C, rowBytes: rowBytesC, dataType: .float32)
    let descG = MPSMatrixDescriptor(dimensions: p, columns: C, rowBytes: rowBytesC, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixScores = MPSMatrix(buffer: scoresBuffer, descriptor: descScores)
    let matrixRes = MPSMatrix(buffer: resBuffer, descriptor: descRes)
    let matrixG = MPSMatrix(buffer: gBuffer, descriptor: descG)

    let cb = ctx.commandQueue.makeCommandBuffer()!

    // Step 1: X @ W → scores (n×C)
    let gemmXW = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: C, interiorColumns: p,
        alpha: 1.0, beta: 0.0)
    gemmXW.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixScores)

    // Step 2: row_max(scores, maxVals)
    if let pipeline = ctx.getPipeline(name: "row_max", functionName: "row_max") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(scoresBuffer, offset: 0, index: 0)
        enc.setBuffer(maxBuffer, offset: 0, index: 1)
        var nU = UInt32(n); var cU = UInt32(C)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
    }

    // Step 3: softmax_exp(scores, maxVals, prob) → prob = exp(scores - max)
    if let pipeline = ctx.getPipeline(name: "softmax_exp", functionName: "softmax_exp") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(scoresBuffer, offset: 0, index: 0)
        enc.setBuffer(maxBuffer, offset: 0, index: 1)
        enc.setBuffer(probBuffer, offset: 0, index: 2)
        var nU = UInt32(n); var cU = UInt32(C)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 4)
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (C + 15) / 16, height: (n + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }

    // Step 4: row_sum(prob, sumExp)
    if let pipeline = ctx.getPipeline(name: "row_sum", functionName: "row_sum") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(probBuffer, offset: 0, index: 0)
        enc.setBuffer(sumBuffer, offset: 0, index: 1)
        var nU = UInt32(n); var cU = UInt32(C)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
    }

    // Step 5: softmax_normalize_residual(prob, sumExp, y, residual)
    if let pipeline = ctx.getPipeline(name: "softmax_normalize_residual", functionName: "softmax_normalize_residual") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(probBuffer, offset: 0, index: 0)
        enc.setBuffer(sumBuffer, offset: 0, index: 1)
        enc.setBuffer(yBuffer, offset: 0, index: 2)
        enc.setBuffer(resBuffer, offset: 0, index: 3)
        var nU = UInt32(n); var cU = UInt32(C)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 5)
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (C + 15) / 16, height: (n + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }

    // Step 6: X^T @ residual → gradient (p×C)
    let gemmGrad = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: C, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixRes, resultMatrix: matrixG)

    // Step 7: multinomial_hessians(X, prob, sumExp, hessians)
    if let pipeline = ctx.getPipeline(name: "multinomial_hessians", functionName: "multinomial_hessians") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xBuffer, offset: 0, index: 0)
        enc.setBuffer(probBuffer, offset: 0, index: 1)
        enc.setBuffer(sumBuffer, offset: 0, index: 2)
        enc.setBuffer(hBuffer, offset: 0, index: 3)
        var nU = UInt32(n); var pU = UInt32(p); var cU = UInt32(C)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 5)
        enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 6)
        let tgSize = MTLSize(width: 8, height: 8, depth: 1)
        let tgCount = MTLSize(width: (p + 7) / 8, height: (p + 7) / 8, depth: C)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }

    cb.commit()
    cb.waitUntilCompleted()
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
        ("kmeans_partial_sum", "kmeans_partial_sum"),
        ("kmeans_combine_normalize", "kmeans_combine_normalize"),
        ("knn_merge_topk", "knn_merge_topk"),
        ("knn_vote_classify", "knn_vote_classify"),
        ("knn_vote_regress", "knn_vote_regress"),
        ("soft_threshold", "soft_threshold"),
        ("column_transform", "column_transform"),
        ("transpose_f32", "transpose_f32"),
        ("sv_init", "sv_init"),
        ("sv_hook", "sv_hook"),
        ("sv_shortcut", "sv_shortcut"),
        ("tree_predict", "tree_predict"),
        ("tree_predict_all", "tree_predict_all"),
        ("row_max", "row_max"),
        ("row_sum", "row_sum"),
        ("softmax_exp", "softmax_exp"),
        ("softmax_normalize_residual", "softmax_normalize_residual"),
        ("negate", "negate"),
        ("multinomial_hessians", "multinomial_hessians"),
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

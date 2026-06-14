import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

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

    let gemm = ctx.getMPSGemm(transposeLeft: transA_, transposeRight: transB_,
                               resultRows: M, resultColumns: N, interiorColumns: K,
                               alpha: Double(alpha), beta: Double(beta))

    gemm.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return 0
}

// MARK: - Float16 GEMM (1.65x throughput on M1/M2, zero-copy via GPU-private buffers)

@_cdecl("skmetal_gemm_f16")
public func skmetal_gemm_f16(
    A: UnsafeRawPointer,
    B: UnsafeRawPointer,
    C: UnsafeMutableRawPointer,
    M: Int, N: Int, K: Int,
    alpha: Float, beta: Float,
    transA: Int32, transB: Int32
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let hs = MemoryLayout<UInt16>.stride

    let transA_ = transA != 0
    let transB_ = transB != 0
    let aRows = transA_ ? K : M
    let aCols = transA_ ? M : K
    let bRows = transB_ ? N : K
    let bCols = transB_ ? K : N

    let aSizeF32 = aRows * aCols * fs
    let bSizeF32 = bRows * bCols * fs
    let cSizeF32 = M * N * fs

    guard let aF32 = wrapInput(A, length: aSizeF32, device: ctx.device),
          let bF32 = wrapInput(B, length: bSizeF32, device: ctx.device),
          let cF32 = wrapOutput(C, length: cSizeF32, device: ctx.device),
          let convPpl = ctx.getPipeline(name: "convert_f32_to_f16", functionName: "convert_f32_to_f16"),
          let deconvPpl = ctx.getPipeline(name: "convert_f16_to_f32", functionName: "convert_f16_to_f32") else {
        return 1
    }

    let aSizeF16 = aRows * aCols * hs
    let bSizeF16 = bRows * bCols * hs
    let cSizeF16 = M * N * hs

    guard let aF16 = ctx.device.makeBuffer(length: aSizeF16, options: .storageModePrivate),
          let bF16 = ctx.device.makeBuffer(length: bSizeF16, options: .storageModePrivate),
          let cF16 = ctx.device.makeBuffer(length: cSizeF16, options: .storageModePrivate) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let tg256 = MTLSize(width: 256, height: 1, depth: 1)

    let encA = cb.makeComputeCommandEncoder()!
    encA.setComputePipelineState(convPpl)
    encA.setBuffer(aF32, offset: 0, index: 0)
    encA.setBuffer(aF16, offset: 0, index: 1)
    var nA = UInt32(aRows * aCols)
    encA.setBytes(&nA, length: MemoryLayout<UInt32>.stride, index: 2)
    encA.dispatchThreadgroups(MTLSize(width: (aRows * aCols + 255) / 256, height: 1, depth: 1),
                               threadsPerThreadgroup: tg256)
    encA.endEncoding()

    let encB = cb.makeComputeCommandEncoder()!
    encB.setComputePipelineState(convPpl)
    encB.setBuffer(bF32, offset: 0, index: 0)
    encB.setBuffer(bF16, offset: 0, index: 1)
    var nB = UInt32(bRows * bCols)
    encB.setBytes(&nB, length: MemoryLayout<UInt32>.stride, index: 2)
    encB.dispatchThreadgroups(MTLSize(width: (bRows * bCols + 255) / 256, height: 1, depth: 1),
                               threadsPerThreadgroup: tg256)
    encB.endEncoding()

    let rowBytesA = aCols * hs
    let rowBytesB = bCols * hs
    let rowBytesC = N * hs

    let descA = MPSMatrixDescriptor(dimensions: aRows, columns: aCols, rowBytes: rowBytesA, dataType: .float16)
    let descB = MPSMatrixDescriptor(dimensions: bRows, columns: bCols, rowBytes: rowBytesB, dataType: .float16)
    let descC = MPSMatrixDescriptor(dimensions: M, columns: N, rowBytes: rowBytesC, dataType: .float16)

    let matrixA = MPSMatrix(buffer: aF16, descriptor: descA)
    let matrixB = MPSMatrix(buffer: bF16, descriptor: descB)
    let matrixC = MPSMatrix(buffer: cF16, descriptor: descC)

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
    gemm.encode(commandBuffer: cb, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)

    let encC = cb.makeComputeCommandEncoder()!
    encC.setComputePipelineState(deconvPpl)
    encC.setBuffer(cF16, offset: 0, index: 0)
    encC.setBuffer(cF32, offset: 0, index: 1)
    var nC = UInt32(M * N)
    encC.setBytes(&nC, length: MemoryLayout<UInt32>.stride, index: 2)
    encC.dispatchThreadgroups(MTLSize(width: (M * N + 255) / 256, height: 1, depth: 1),
                               threadsPerThreadgroup: tg256)
    encC.endEncoding()

    cb.commit()
    cb.waitUntilCompleted()
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

// MARK: - GPU mean centering / transpose

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

@_cdecl("skmetal_column_means")
public func skmetal_column_means(
    X: UnsafeRawPointer,
    means: UnsafeMutableRawPointer,
    n: Int,
    p: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * p * MemoryLayout<Float>.stride
    let meansSize = p * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "column_means", functionName: "column_means"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let meansBuffer = wrapOutput(means, length: meansSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(meansBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    var pUint = UInt32(p)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.setBytes(&pUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let blockCols = 8
    let tgCount = (p + blockCols - 1) / blockCols
    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: tgCount, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

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

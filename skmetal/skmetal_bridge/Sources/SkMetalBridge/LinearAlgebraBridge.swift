import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - Float16 conversion (zero-copy GPU)

@_cdecl("skmetal_convert_f32_to_f16")
public func skmetal_convert_f32_to_f16(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride
    let halfSize = n * MemoryLayout<UInt16>.stride

    guard let pipeline = ctx.getPipeline(name: "convert_f32_to_f16", functionName: "convert_f32_to_f16"),
          let inBuf = wrapInput(input, length: byteSize, device: ctx.device),
          let outBuf = wrapOutput(output, length: halfSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(inBuf, offset: 0, index: 0)
    enc.setBuffer(outBuf, offset: 0, index: 1)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    let grid = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_convert_f16_to_f32")
public func skmetal_convert_f16_to_f32(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let halfSize = n * MemoryLayout<UInt16>.stride
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "convert_f16_to_f32", functionName: "convert_f16_to_f32"),
          let inBuf = wrapInput(input, length: halfSize, device: ctx.device),
          let outBuf = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(inBuf, offset: 0, index: 0)
    enc.setBuffer(outBuf, offset: 0, index: 1)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    let grid = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Float16 simdgroup GEMM (half-precision inputs/outputs)

@_cdecl("skmetal_gemm_f16")
public func skmetal_gemm_f16(
    A: UnsafeRawPointer,
    B: UnsafeRawPointer,
    C: UnsafeMutableRawPointer,
    M: Int,
    N: Int,
    K: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let TS = 8
    guard M % TS == 0, N % TS == 0, K % TS == 0,
          M <= 256, N <= 256, K <= 256 else {
        return 1  // unsupported, caller should fall back
    }

    guard let pipeline = ctx.getPipeline(name: "simdgroup_gemm_f16", functionName: "simdgroup_gemm_f16") else {
        return 1
    }

    let halfSizeA = M * K * MemoryLayout<UInt16>.stride
    let halfSizeB = K * N * MemoryLayout<UInt16>.stride
    let halfSizeC = M * N * MemoryLayout<UInt16>.stride

    guard let bufA = wrapInput(A, length: halfSizeA, device: ctx.device),
          let bufB = wrapInput(B, length: halfSizeB, device: ctx.device),
          let bufC = wrapOutput(C, length: halfSizeC, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)
    var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K)
    enc.setBytes(&mU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)

    let grid = MTLSize(width: (N + 7) / 8, height: (M + 7) / 8, depth: 1)
    let tg = MTLSize(width: 32, height: 1, depth: 1)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - GEMM (simdgroup path for small aligned matrices, MPS fallback)

// Try simdgroup_gemm_f32 kernel for small aligned matrices (M,N,K ≤ 256, all %8==0, no transpose).
// Returns true if handled, false to fall back to MPS.
private func trySimdgroupGemm(
    bufferA: MTLBuffer, bufferB: MTLBuffer, bufferC: MTLBuffer,
    M: Int, N: Int, K: Int,
    alpha: Float, beta: Float,
    commandBuffer: MTLCommandBuffer, ctx: MetalContext
) -> Bool {
    let maxDim = 256
    guard M <= maxDim, N <= maxDim, K <= maxDim,
          M % 8 == 0, N % 8 == 0, K % 8 == 0,
          alpha == 1.0, beta == 0.0 else {
        return false
    }

    guard let pipeline = ctx.getPipeline(name: "simdgroup_gemm_f32", functionName: "simdgroup_gemm_f32") else {
        return false
    }

    let enc = commandBuffer.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufferA, offset: 0, index: 0)
    enc.setBuffer(bufferB, offset: 0, index: 1)
    enc.setBuffer(bufferC, offset: 0, index: 2)
    var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K)
    enc.setBytes(&mU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)

    let grid = MTLSize(width: (N + 7) / 8, height: (M + 7) / 8, depth: 1)
    let tg = MTLSize(width: 32, height: 1, depth: 1)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    enc.endEncoding()

    return true
}

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

    // Try simdgroup path for small aligned matrices (no transpose, alpha=1, beta=0)
    if !transA_, !transB_,
       trySimdgroupGemm(bufferA: bufferA, bufferB: bufferB, bufferC: bufferC,
                        M: M, N: N, K: K,
                        alpha: alpha, beta: beta,
                        commandBuffer: commandBuffer, ctx: ctx) {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return 0
    }

    // Fall back to MPS
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



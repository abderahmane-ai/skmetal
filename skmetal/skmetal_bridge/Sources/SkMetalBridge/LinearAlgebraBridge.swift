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



import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - RBF Kernel (GPU-accelerated Gram matrix)

@_cdecl("skmetal_rbf_kernel_square")
public func skmetal_rbf_kernel_square(
    X: UnsafeRawPointer,
    K_out: UnsafeMutableRawPointer,
    gamma: Float,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let xSize = n * d * fs
    let normSize = n * fs
    let kSize = n * n * fs

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let kBuffer = wrapOutput(K_out, length: kSize, device: ctx.device) else {
        return 1
    }

    guard let normBuffer = ctx.device.makeBuffer(length: normSize, options: .storageModePrivate) else {
        return 1
    }

    let rowBytesX = d * fs
    let rowBytesK = n * fs

    let descX = MPSMatrixDescriptor(dimensions: n, columns: d, rowBytes: rowBytesX, dataType: .float32)
    let descK = MPSMatrixDescriptor(dimensions: n, columns: n, rowBytes: rowBytesK, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixK = MPSMatrix(buffer: kBuffer, descriptor: descK)

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }

    if let pipeline = ctx.getPipeline(name: "row_norm_sq", functionName: "row_norm_sq") {
        guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xBuffer, offset: 0, index: 0)
        enc.setBuffer(normBuffer, offset: 0, index: 1)
        var nU = UInt32(n); var dU = UInt32(d)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    let gemm = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: true,
        resultRows: n, resultColumns: n, interiorColumns: d,
        alpha: 1.0, beta: 0.0)
    gemm.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixX, resultMatrix: matrixK)

    if let pipeline = ctx.getPipeline(name: "distance_correct", functionName: "distance_correct") {
        guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(kBuffer, offset: 0, index: 0)
        enc.setBuffer(normBuffer, offset: 0, index: 1)
        enc.setBuffer(normBuffer, offset: 0, index: 2)
        var nU = UInt32(n); var dummyU = UInt32(n)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.setBytes(&dummyU, length: MemoryLayout<UInt32>.stride, index: 4)
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (n + 15) / 16, height: (n + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }

    if let pipeline = ctx.getPipeline(name: "rbf_apply", functionName: "rbf_apply") {
        guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(kBuffer, offset: 0, index: 0)
        var g = gamma
        enc.setBytes(&g, length: fs, index: 1)
        var nU = UInt32(n); var mU = UInt32(n)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&mU, length: MemoryLayout<UInt32>.stride, index: 3)
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (n + 15) / 16, height: (n + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }

    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_rbf_kernel_cross")
public func skmetal_rbf_kernel_cross(
    X1: UnsafeRawPointer,
    X2: UnsafeRawPointer,
    K_out: UnsafeMutableRawPointer,
    gamma: Float,
    n1: Int,
    n2: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let x1Size = n1 * d * fs
    let x2Size = n2 * d * fs
    let n1Size = n1 * fs
    let n2Size = n2 * fs
    let kSize = n1 * n2 * fs

    guard let x1Buffer = wrapInput(X1, length: x1Size, device: ctx.device),
          let x2Buffer = wrapInput(X2, length: x2Size, device: ctx.device),
          let kBuffer = wrapOutput(K_out, length: kSize, device: ctx.device) else {
        return 1
    }

    guard let n1Buffer = ctx.device.makeBuffer(length: n1Size, options: .storageModePrivate),
          let n2Buffer = ctx.device.makeBuffer(length: n2Size, options: .storageModePrivate) else {
        return 1
    }

    let rowBytesX1 = d * fs
    let rowBytesX2 = d * fs
    let rowBytesK = n2 * fs

    let descX1 = MPSMatrixDescriptor(dimensions: n1, columns: d, rowBytes: rowBytesX1, dataType: .float32)
    let descX2 = MPSMatrixDescriptor(dimensions: n2, columns: d, rowBytes: rowBytesX2, dataType: .float32)
    let descK = MPSMatrixDescriptor(dimensions: n1, columns: n2, rowBytes: rowBytesK, dataType: .float32)

    let matrixX1 = MPSMatrix(buffer: x1Buffer, descriptor: descX1)
    let matrixX2 = MPSMatrix(buffer: x2Buffer, descriptor: descX2)
    let matrixK = MPSMatrix(buffer: kBuffer, descriptor: descK)

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }

    if let pipeline = ctx.getPipeline(name: "row_norm_sq", functionName: "row_norm_sq") {
        guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x1Buffer, offset: 0, index: 0)
        enc.setBuffer(n1Buffer, offset: 0, index: 1)
        var nU = UInt32(n1); var dU = UInt32(d)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (n1 + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    if let pipeline = ctx.getPipeline(name: "row_norm_sq", functionName: "row_norm_sq") {
        guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x2Buffer, offset: 0, index: 0)
        enc.setBuffer(n2Buffer, offset: 0, index: 1)
        var nU = UInt32(n2); var dU = UInt32(d)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (n2 + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    let gemm = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: true,
        resultRows: n1, resultColumns: n2, interiorColumns: d,
        alpha: 1.0, beta: 0.0)
    gemm.encode(commandBuffer: cb, leftMatrix: matrixX1, rightMatrix: matrixX2, resultMatrix: matrixK)

    if let pipeline = ctx.getPipeline(name: "distance_correct", functionName: "distance_correct") {
        guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(kBuffer, offset: 0, index: 0)
        enc.setBuffer(n1Buffer, offset: 0, index: 1)
        enc.setBuffer(n2Buffer, offset: 0, index: 2)
        var n1U = UInt32(n1); var n2U = UInt32(n2)
        enc.setBytes(&n1U, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.setBytes(&n2U, length: MemoryLayout<UInt32>.stride, index: 4)
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (n2 + 15) / 16, height: (n1 + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }

    if let pipeline = ctx.getPipeline(name: "rbf_apply", functionName: "rbf_apply") {
        guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(kBuffer, offset: 0, index: 0)
        var g = gamma
        enc.setBytes(&g, length: fs, index: 1)
        var n1U = UInt32(n1); var n2U = UInt32(n2)
        enc.setBytes(&n1U, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&n2U, length: MemoryLayout<UInt32>.stride, index: 3)
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (n2 + 15) / 16, height: (n1 + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }

    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Matrix-free SVC predict

@_cdecl("skmetal_svc_predict_binary")
public func skmetal_svc_predict_binary(
    X_test: UnsafeRawPointer,
    X_sv: UnsafeRawPointer,
    dual_coef: UnsafeRawPointer,
    intercept: UnsafeRawPointer,
    decisions: UnsafeMutableRawPointer,
    n_test: Int,
    n_sv: Int,
    d: Int,
    gamma: Float
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let testSize = n_test * d * fs
    let svSize = n_sv * d * fs
    let dcSize = n_sv * fs
    let decSize = n_test * fs

    guard let pipeline = ctx.getPipeline(name: "svc_predict_binary", functionName: "svc_predict_binary"),
          let testBuffer = wrapInput(X_test, length: testSize, device: ctx.device),
          let svBuffer = wrapInput(X_sv, length: svSize, device: ctx.device),
          let dcBuffer = wrapInput(dual_coef, length: dcSize, device: ctx.device),
          let icBuffer = wrapInput(intercept, length: fs, device: ctx.device),
          let decBuffer = wrapOutput(decisions, length: decSize, device: ctx.device) else {
        return 1
    }

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(testBuffer, offset: 0, index: 0)
    enc.setBuffer(svBuffer, offset: 0, index: 1)
    enc.setBuffer(dcBuffer, offset: 0, index: 2)
    enc.setBuffer(icBuffer, offset: 0, index: 3)
    enc.setBuffer(decBuffer, offset: 0, index: 4)
    var nTestU = UInt32(n_test); var nSvU = UInt32(n_sv); var dU = UInt32(d)
    enc.setBytes(&nTestU, length: MemoryLayout<UInt32>.stride, index: 5)
    enc.setBytes(&nSvU, length: MemoryLayout<UInt32>.stride, index: 6)
    enc.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 7)
    var g = gamma
    enc.setBytes(&g, length: fs, index: 8)
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let tgCount = MTLSize(width: (n_test + 255) / 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

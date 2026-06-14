import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

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
    let colBytes = MemoryLayout<Float>.stride

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: colBytes, dataType: .float32)
    let descLin = MPSMatrixDescriptor(dimensions: n, columns: 1, rowBytes: colBytes, dataType: .float32)
    let descXS = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descH = MPSMatrixDescriptor(dimensions: p, columns: p, rowBytes: rowBytesH, dataType: .float32)
    let descG = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: colBytes, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixLin = MPSMatrix(buffer: linearBuffer, descriptor: descLin)
    let matrixXS = MPSMatrix(buffer: xsBuffer, descriptor: descXS)
    let matrixH = MPSMatrix(buffer: hBuffer, descriptor: descH)
    let matrixG = MPSMatrix(buffer: gBuffer, descriptor: descG)

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!

    let gemmXW = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: 1, interiorColumns: p,
        alpha: 1.0, beta: 0.0)
    gemmXW.encode(commandBuffer: commandBuffer, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixLin)

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

    let gemmHH = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: p, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmHH.encode(commandBuffer: commandBuffer, leftMatrix: matrixXS, rightMatrix: matrixXS, resultMatrix: matrixH)

    let gemmGrad = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: 1, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmGrad.encode(commandBuffer: commandBuffer, leftMatrix: matrixX, rightMatrix: matrixLin, resultMatrix: matrixG)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - Fused IRLS iteration with GPU Cholesky solve (BINARY)

@_cdecl("skmetal_logreg_irls_fused_solve")
public func skmetal_logreg_irls_fused_solve(
    X: UnsafeRawPointer,
    y: UnsafeRawPointer,
    w: UnsafeRawPointer,
    b: Float,
    linear: UnsafeMutableRawPointer,
    weight: UnsafeMutableRawPointer,
    X_scaled: UnsafeMutableRawPointer,
    Hessian: UnsafeMutableRawPointer,
    gradient: UnsafeMutableRawPointer,
    delta: UnsafeMutableRawPointer,
    alpha: Float,
    n: Int,
    p: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let nSize = n * fs
    let pSize = p * fs
    let xSize = n * p * fs
    let hessianSize = p * p * fs

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: nSize, device: ctx.device),
          let wBuffer = wrapInput(w, length: pSize, device: ctx.device),
          let linearBuffer = wrapOutput(linear, length: nSize, device: ctx.device),
          let weightBuffer = wrapOutput(weight, length: nSize, device: ctx.device),
          let xsBuffer = wrapOutput(X_scaled, length: xSize, device: ctx.device),
          let hBuffer = wrapOutput(Hessian, length: hessianSize, device: ctx.device),
          let gBuffer = wrapOutput(gradient, length: pSize, device: ctx.device),
          let dBuffer = wrapOutput(delta, length: pSize, device: ctx.device) else {
        return 1
    }

    let rowBytesX = p * fs
    let rowBytesH = p * fs

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)
    let descLin = MPSMatrixDescriptor(dimensions: n, columns: 1, rowBytes: fs, dataType: .float32)
    let descXS = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descH = MPSMatrixDescriptor(dimensions: p, columns: p, rowBytes: rowBytesH, dataType: .float32)
    let descG = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixLin = MPSMatrix(buffer: linearBuffer, descriptor: descLin)
    let matrixXS = MPSMatrix(buffer: xsBuffer, descriptor: descXS)
    let matrixH = MPSMatrix(buffer: hBuffer, descriptor: descH)
    let matrixG = MPSMatrix(buffer: gBuffer, descriptor: descG)

    let cb = ctx.commandQueue.makeCommandBuffer()!

    let gemmXW = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: 1, interiorColumns: p,
        alpha: 1.0, beta: 0.0)
    gemmXW.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixLin)

    if let pipeline = ctx.getPipeline(name: "compute_linear_irls", functionName: "compute_linear_irls") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(linearBuffer, offset: 0, index: 0)
        enc.setBuffer(weightBuffer, offset: 0, index: 1)
        var bScalar = b
        enc.setBytes(&bScalar, length: fs, index: 2)
        var nU = UInt32(n)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    if let pipeline = ctx.getPipeline(name: "compute_error_scale", functionName: "compute_error_scale") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(linearBuffer, offset: 0, index: 0)
        enc.setBuffer(yBuffer, offset: 0, index: 1)
        enc.setBuffer(xBuffer, offset: 0, index: 2)
        enc.setBuffer(weightBuffer, offset: 0, index: 3)
        enc.setBuffer(linearBuffer, offset: 0, index: 4)
        enc.setBuffer(xsBuffer, offset: 0, index: 5)
        var nU = UInt32(n); var pU = UInt32(p)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 6)
        enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 7)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    let gemmHH = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: p, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmHH.encode(commandBuffer: cb, leftMatrix: matrixXS, rightMatrix: matrixXS, resultMatrix: matrixH)

    let gemmGrad = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: 1, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixLin, resultMatrix: matrixG)

    if alpha != 0 {
        if let pipeline = ctx.getPipeline(name: "l2_reg_irls", functionName: "l2_reg_irls") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(hBuffer, offset: 0, index: 0)
            enc.setBuffer(gBuffer, offset: 0, index: 1)
            enc.setBuffer(wBuffer, offset: 0, index: 2)
            var a = alpha
            enc.setBytes(&a, length: fs, index: 3)
            var pU = UInt32(p)
            enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.dispatchThreads(MTLSize(width: p, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
    }

    cb.commit()
    cb.waitUntilCompleted()

    let p32 = Int32(p)
    let one32 = Int32(1)
    let hPtr = hBuffer.contents().assumingMemoryBound(to: Float.self)
    let gPtr = gBuffer.contents().assumingMemoryBound(to: Float.self)
    let dPtr = dBuffer.contents().assumingMemoryBound(to: Float.self)

    var uplo: CChar = 76
    var n_ = p32, nrhs_ = one32, lda_ = p32, ldb_ = p32
    var info: Int32 = 0
    spotrf_(&uplo, &n_, hPtr, &lda_, &info)
    guard info == 0 else { return 1 }
    spotrs_(&uplo, &n_, &nrhs_, hPtr, &lda_, gPtr, &ldb_, &info)
    guard info == 0 else { return 1 }

    memcpy(dPtr, gPtr, p * fs)

    return 0
}

// MARK: - Fused IRLS iteration (no solve — 5 dispatches, was 8)

@_cdecl("skmetal_logreg_irls_fused")
public func skmetal_logreg_irls_fused(
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
    let fs = MemoryLayout<Float>.stride
    let nSize = n * fs
    let pSize = p * fs
    let xSize = n * p * fs
    let hessianSize = p * p * fs

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

    let rowBytesX = p * fs
    let rowBytesH = p * fs

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)
    let descLin = MPSMatrixDescriptor(dimensions: n, columns: 1, rowBytes: fs, dataType: .float32)
    let descXS = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descH = MPSMatrixDescriptor(dimensions: p, columns: p, rowBytes: rowBytesH, dataType: .float32)
    let descG = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixLin = MPSMatrix(buffer: linearBuffer, descriptor: descLin)
    let matrixXS = MPSMatrix(buffer: xsBuffer, descriptor: descXS)
    let matrixH = MPSMatrix(buffer: hBuffer, descriptor: descH)
    let matrixG = MPSMatrix(buffer: gBuffer, descriptor: descG)

    let cb = ctx.commandQueue.makeCommandBuffer()!

    let gemmXW = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: 1, interiorColumns: p,
        alpha: 1.0, beta: 0.0)
    gemmXW.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixLin)

    if let pipeline = ctx.getPipeline(name: "compute_linear_irls", functionName: "compute_linear_irls") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(linearBuffer, offset: 0, index: 0)
        enc.setBuffer(weightBuffer, offset: 0, index: 1)
        var bScalar = b
        enc.setBytes(&bScalar, length: fs, index: 2)
        var nU = UInt32(n)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    if let pipeline = ctx.getPipeline(name: "compute_error_scale", functionName: "compute_error_scale") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(linearBuffer, offset: 0, index: 0)
        enc.setBuffer(yBuffer, offset: 0, index: 1)
        enc.setBuffer(xBuffer, offset: 0, index: 2)
        enc.setBuffer(weightBuffer, offset: 0, index: 3)
        enc.setBuffer(linearBuffer, offset: 0, index: 4)
        enc.setBuffer(xsBuffer, offset: 0, index: 5)
        var nU = UInt32(n); var pU = UInt32(p)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 6)
        enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 7)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    let gemmHH = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: p, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmHH.encode(commandBuffer: cb, leftMatrix: matrixXS, rightMatrix: matrixXS, resultMatrix: matrixH)

    let gemmGrad = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: 1, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixLin, resultMatrix: matrixG)

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
    let fs = MemoryLayout<Float>.stride
    let xSize = n * p * fs
    let wSize = p * C * fs
    let scoresSize = n * C * fs

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let wBuffer = wrapInput(W, length: wSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: n * fs, device: ctx.device),
          let scoresBuffer = wrapOutput(scores, length: scoresSize, device: ctx.device),
          let probBuffer = wrapOutput(prob, length: scoresSize, device: ctx.device),
          let maxBuffer = wrapOutput(maxVals, length: n * fs, device: ctx.device),
          let sumBuffer = wrapOutput(sumExp, length: n * fs, device: ctx.device),
          let resBuffer = wrapOutput(residual, length: scoresSize, device: ctx.device),
          let gBuffer = wrapOutput(gradient, length: wSize, device: ctx.device),
          let hBuffer = wrapOutput(hessians, length: C * p * p * fs, device: ctx.device) else {
        return 1
    }

    let rowBytesX = p * MemoryLayout<Float>.stride
    let rowBytesC = C * MemoryLayout<Float>.stride

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

    let gemmXW = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: C, interiorColumns: p,
        alpha: 1.0, beta: 0.0)
    gemmXW.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixScores)

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

    let gemmGrad = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: C, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixRes, resultMatrix: matrixG)

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

// MARK: - Multinomial IRLS fused solve (GPU batched Cholesky)

@_cdecl("skmetal_multinomial_irls_fused_solve")
public func skmetal_multinomial_irls_fused_solve(
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
    delta_W: UnsafeMutableRawPointer,
    alpha: Float,
    n: Int,
    p: Int,
    C: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let xSize = n * p * fs
    let wSize = p * C * fs
    let scoresSize = n * C * fs
    let hessiansSize = C * p * p * fs

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let wBuffer = wrapInput(W, length: wSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: n * fs, device: ctx.device),
          let scoresBuffer = wrapOutput(scores, length: scoresSize, device: ctx.device),
          let probBuffer = wrapOutput(prob, length: scoresSize, device: ctx.device),
          let sumBuffer = wrapOutput(sumExp, length: n * fs, device: ctx.device),
          let resBuffer = wrapOutput(residual, length: scoresSize, device: ctx.device),
          let gBuffer = wrapOutput(gradient, length: wSize, device: ctx.device),
          let hBuffer = wrapOutput(hessians, length: hessiansSize, device: ctx.device),
          let dBuffer = wrapOutput(delta_W, length: wSize, device: ctx.device) else {
        return 1
    }

    let gradBatchSize = C * p * fs
    guard let gradBatchBuffer = ctx.device.makeBuffer(length: gradBatchSize, options: .storageModeShared) else {
        return 1
    }

    let rowBytesX = p * fs
    let rowBytesC = C * fs

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: C, rowBytes: rowBytesC, dataType: .float32)
    let descScores = MPSMatrixDescriptor(dimensions: n, columns: C, rowBytes: rowBytesC, dataType: .float32)
    let descRes = MPSMatrixDescriptor(dimensions: n, columns: C, rowBytes: rowBytesC, dataType: .float32)
    let descG = MPSMatrixDescriptor(dimensions: p, columns: C, rowBytes: rowBytesC, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixScores = MPSMatrix(buffer: scoresBuffer, descriptor: descScores)
    let matrixProb = MPSMatrix(buffer: probBuffer, descriptor: descScores)
    let matrixRes = MPSMatrix(buffer: resBuffer, descriptor: descRes)
    let matrixG = MPSMatrix(buffer: gBuffer, descriptor: descG)

    let cb = ctx.commandQueue.makeCommandBuffer()!

    let gemmXW = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: C, interiorColumns: p,
        alpha: 1.0, beta: 0.0)
    gemmXW.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixScores)

    let softmax = MPSMatrixSoftMax(device: ctx.device)
    softmax.sourceRows = n
    softmax.sourceColumns = C
    softmax.encode(commandBuffer: cb, inputMatrix: matrixScores, resultMatrix: matrixProb)

    if let pipeline = ctx.getPipeline(name: "fill_f32", functionName: "fill_f32") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(sumBuffer, offset: 0, index: 0)
        var one: Float = 1.0
        enc.setBytes(&one, length: fs, index: 1)
        var nU = UInt32(n)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    if let pipeline = ctx.getPipeline(name: "softmax_residual", functionName: "softmax_residual") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(probBuffer, offset: 0, index: 0)
        enc.setBuffer(yBuffer, offset: 0, index: 1)
        enc.setBuffer(resBuffer, offset: 0, index: 2)
        var nU = UInt32(n); var cU = UInt32(C)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 4)
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (C + 15) / 16, height: (n + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }

    let gemmGrad = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: C, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixRes, resultMatrix: matrixG)

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

    if let pipeline = ctx.getPipeline(name: "transpose_f32", functionName: "transpose_f32") {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(gBuffer, offset: 0, index: 0)
        enc.setBuffer(gradBatchBuffer, offset: 0, index: 1)
        var rowsU = UInt32(p); var colsU = UInt32(C)
        enc.setBytes(&rowsU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&colsU, length: MemoryLayout<UInt32>.stride, index: 3)
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (C + 15) / 16, height: (p + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }

    if alpha != 0 {
        if let pipeline = ctx.getPipeline(name: "multinomial_l2_reg", functionName: "multinomial_l2_reg") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(hBuffer, offset: 0, index: 0)
            enc.setBuffer(gradBatchBuffer, offset: 0, index: 1)
            enc.setBuffer(wBuffer, offset: 0, index: 2)
            var a = alpha
            enc.setBytes(&a, length: fs, index: 3)
            var pU = UInt32(p); var cU = UInt32(C)
            enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 5)
            let tgSize = MTLSize(width: 16, height: 16, depth: 1)
            let tgCount = MTLSize(width: (C + 15) / 16, height: (p + 15) / 16, depth: 1)
            enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
        }
    }

    cb.commit()
    cb.waitUntilCompleted()

    let p32 = Int32(p)
    let one32 = Int32(1)
    let hBase = hBuffer.contents().assumingMemoryBound(to: Float.self)
    let gBase = gradBatchBuffer.contents().assumingMemoryBound(to: Float.self)
    let dBase = dBuffer.contents().assumingMemoryBound(to: Float.self)

    var uplo: CChar = 76
    var n_ = p32, nrhs_ = one32, lda_ = p32, ldb_ = p32
    var info: Int32 = 0
    for c in 0..<C {
        let hOff = c * p * p
        let gOff = c * p
        let hPtr = hBase.advanced(by: hOff)
        let gPtr = gBase.advanced(by: gOff)
        spotrf_(&uplo, &n_, hPtr, &lda_, &info)
        guard info == 0 else { return 1 }
        spotrs_(&uplo, &n_, &nrhs_, hPtr, &lda_, gPtr, &ldb_, &info)
        guard info == 0 else { return 1 }
    }

    for c in 0..<C {
        for i in 0..<p {
            dBase[i * C + c] = gBase[c * p + i]
        }
    }

    return 0
}

// MARK: - IRLS fit: full binary LogisticRegression loop in Swift

@_cdecl("skmetal_logreg_irls_fit")
public func skmetal_logreg_irls_fit(
    X: UnsafeRawPointer,
    y: UnsafeRawPointer,
    coef_out: UnsafeMutableRawPointer,
    C: Float,
    tol: Float,
    max_iter: Int32,
    fit_intercept: Int32,
    n: Int,
    p: Int,
    n_iter_out: UnsafeMutablePointer<Int32>?
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let nSize = n * fs
    let pSize = p * fs
    let xSize = n * p * fs
    let hSize = p * p * fs

    let alpha: Float = 1.0 / C

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: nSize, device: ctx.device) else {
        return 1
    }

    guard let wBuffer = ctx.device.makeBuffer(length: pSize, options: .storageModeShared),
          let linearBuffer = ctx.device.makeBuffer(length: nSize, options: .storageModeShared),
          let weightBuffer = ctx.device.makeBuffer(length: nSize, options: .storageModeShared),
          let xsBuffer = ctx.device.makeBuffer(length: xSize, options: .storageModeShared),
          let hBuffer = ctx.device.makeBuffer(length: hSize, options: .storageModeShared),
          let gBuffer = ctx.device.makeBuffer(length: pSize, options: .storageModeShared),
          let dBuffer = ctx.device.makeBuffer(length: pSize, options: .storageModeShared) else {
        return 1
    }

    memset(wBuffer.contents(), 0, pSize)

    let rowBytesX = p * fs
    let rowBytesH = p * fs

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)
    let descLin = MPSMatrixDescriptor(dimensions: n, columns: 1, rowBytes: fs, dataType: .float32)
    let descXS = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descH = MPSMatrixDescriptor(dimensions: p, columns: p, rowBytes: rowBytesH, dataType: .float32)
    let descG = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixLin = MPSMatrix(buffer: linearBuffer, descriptor: descLin)
    let matrixXS = MPSMatrix(buffer: xsBuffer, descriptor: descXS)
    let matrixH = MPSMatrix(buffer: hBuffer, descriptor: descH)
    let matrixG = MPSMatrix(buffer: gBuffer, descriptor: descG)

    let maxIter = Int(max_iter)
    var nIter: Int32 = 0

    for it in 0..<maxIter {
        nIter = Int32(it + 1)
        let cb = ctx.commandQueue.makeCommandBuffer()!

        let gemmXW = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: false, transposeRight: false,
            resultRows: n, resultColumns: 1, interiorColumns: p,
            alpha: 1.0, beta: 0.0)
        gemmXW.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixLin)

        if let pipeline = ctx.getPipeline(name: "compute_linear_irls", functionName: "compute_linear_irls") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(linearBuffer, offset: 0, index: 0)
            enc.setBuffer(weightBuffer, offset: 0, index: 1)
            var b: Float = 0.0
            enc.setBytes(&b, length: fs, index: 2)
            var nU = UInt32(n)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        if let pipeline = ctx.getPipeline(name: "compute_error_scale", functionName: "compute_error_scale") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(linearBuffer, offset: 0, index: 0)
            enc.setBuffer(yBuffer, offset: 0, index: 1)
            enc.setBuffer(xBuffer, offset: 0, index: 2)
            enc.setBuffer(weightBuffer, offset: 0, index: 3)
            enc.setBuffer(linearBuffer, offset: 0, index: 4)
            enc.setBuffer(xsBuffer, offset: 0, index: 5)
            var nU = UInt32(n); var pU = UInt32(p)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 6)
            enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 7)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        let gemmHH = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: p, interiorColumns: n,
            alpha: 1.0, beta: 0.0)
        gemmHH.encode(commandBuffer: cb, leftMatrix: matrixXS, rightMatrix: matrixXS, resultMatrix: matrixH)

        let gemmGrad = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: 1, interiorColumns: n,
            alpha: 1.0, beta: 0.0)
        gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixLin, resultMatrix: matrixG)

        cb.commit()
        cb.waitUntilCompleted()

        if alpha != 0 {
            let hPtr = hBuffer.contents().assumingMemoryBound(to: Float.self)
            let gPtr = gBuffer.contents().assumingMemoryBound(to: Float.self)
            let wPtr = wBuffer.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<p {
                hPtr[i * p + i] += alpha
                gPtr[i] += alpha * wPtr[i]
            }
        }

        let hPtr = hBuffer.contents().assumingMemoryBound(to: Float.self)
        let gPtr = gBuffer.contents().assumingMemoryBound(to: Float.self)
        let dPtr = dBuffer.contents().assumingMemoryBound(to: Float.self)
        let p32 = Int32(p)
        let one32 = Int32(1)
        var uplo: CChar = 76
        var n_ = p32, nrhs_ = one32, lda_ = p32, ldb_ = p32
        var info: Int32 = 0
        spotrf_(&uplo, &n_, hPtr, &lda_, &info)
        guard info == 0 else { return 1 }
        spotrs_(&uplo, &n_, &nrhs_, hPtr, &lda_, gPtr, &ldb_, &info)
        guard info == 0 else { return 1 }

        memcpy(dPtr, gPtr, pSize)

        let gNorm = cblas_snrm2(p32, gPtr, 1)
        let wPtr = wBuffer.contents().assumingMemoryBound(to: Float.self)
        let wNorm = cblas_snrm2(p32, wPtr, 1)
        if gNorm < tol * max(1.0, wNorm) { break }

        cblas_saxpy(p32, -1.0, dPtr, 1, wPtr, 1)
    }

    let coefOut = coef_out.assumingMemoryBound(to: Float.self)
    let wFinal = wBuffer.contents().assumingMemoryBound(to: Float.self)
    memcpy(coefOut, wFinal, pSize)
    n_iter_out?.pointee = nIter
    return 0
}

// MARK: - IRLS fit: full multinomial LogisticRegression loop in Swift

@_cdecl("skmetal_multinomial_irls_fit")
public func skmetal_multinomial_irls_fit(
    X: UnsafeRawPointer,
    y: UnsafeRawPointer,
    W_out: UnsafeMutableRawPointer,
    C: Float,
    tol: Float,
    max_iter: Int32,
    n: Int,
    p: Int,
    n_classes: Int,
    n_iter_out: UnsafeMutablePointer<Int32>?
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let xSize = n * p * fs
    let wSize = p * n_classes * fs
    let scoresSize = n * n_classes * fs
    let hessiansSize = n_classes * p * p * fs

    let alpha: Float = 1.0 / C

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: n * fs, device: ctx.device) else {
        return 1
    }

    guard let wBuffer = ctx.device.makeBuffer(length: wSize, options: .storageModeShared),
          let scoresBuffer = ctx.device.makeBuffer(length: scoresSize, options: .storageModeShared),
          let probBuffer = ctx.device.makeBuffer(length: scoresSize, options: .storageModeShared),
          let sumBuffer = ctx.device.makeBuffer(length: n * fs, options: .storageModeShared),
          let resBuffer = ctx.device.makeBuffer(length: scoresSize, options: .storageModeShared),
          let gBuffer = ctx.device.makeBuffer(length: wSize, options: .storageModeShared),
          let hBuffer = ctx.device.makeBuffer(length: hessiansSize, options: .storageModeShared),
          let dBuffer = ctx.device.makeBuffer(length: wSize, options: .storageModeShared),
          let gradBatchBuffer = ctx.device.makeBuffer(length: n_classes * p * fs, options: .storageModeShared) else {
        return 1
    }

    memset(wBuffer.contents(), 0, wSize)

    let rowBytesX = p * fs
    let rowBytesC = n_classes * fs

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: n_classes, rowBytes: rowBytesC, dataType: .float32)
    let descScores = MPSMatrixDescriptor(dimensions: n, columns: n_classes, rowBytes: rowBytesC, dataType: .float32)
    let descRes = MPSMatrixDescriptor(dimensions: n, columns: n_classes, rowBytes: rowBytesC, dataType: .float32)
    let descG = MPSMatrixDescriptor(dimensions: p, columns: n_classes, rowBytes: rowBytesC, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixScores = MPSMatrix(buffer: scoresBuffer, descriptor: descScores)
    let matrixProb = MPSMatrix(buffer: probBuffer, descriptor: descScores)
    let matrixRes = MPSMatrix(buffer: resBuffer, descriptor: descRes)
    let matrixG = MPSMatrix(buffer: gBuffer, descriptor: descG)

    let maxIter = Int(max_iter)
    var nIter: Int32 = 0
    let p32 = Int32(p)
    let one32 = Int32(1)

    for it in 0..<maxIter {
        nIter = Int32(it + 1)
        let cb = ctx.commandQueue.makeCommandBuffer()!

        let gemmXW = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: false, transposeRight: false,
            resultRows: n, resultColumns: n_classes, interiorColumns: p,
            alpha: 1.0, beta: 0.0)
        gemmXW.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixScores)

        let softmax = MPSMatrixSoftMax(device: ctx.device)
        softmax.sourceRows = n
        softmax.sourceColumns = n_classes
        softmax.encode(commandBuffer: cb, inputMatrix: matrixScores, resultMatrix: matrixProb)

        if let pipeline = ctx.getPipeline(name: "fill_f32", functionName: "fill_f32") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(sumBuffer, offset: 0, index: 0)
            var one: Float = 1.0
            enc.setBytes(&one, length: fs, index: 1)
            var nU = UInt32(n)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        if let pipeline = ctx.getPipeline(name: "softmax_residual", functionName: "softmax_residual") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(probBuffer, offset: 0, index: 0)
            enc.setBuffer(yBuffer, offset: 0, index: 1)
            enc.setBuffer(resBuffer, offset: 0, index: 2)
            var nU = UInt32(n); var cU = UInt32(n_classes)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 4)
            let tgSize = MTLSize(width: 16, height: 16, depth: 1)
            let tgCount = MTLSize(width: (n_classes + 15) / 16, height: (n + 15) / 16, depth: 1)
            enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
        }

        let gemmGrad = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: n_classes, interiorColumns: n,
            alpha: 1.0, beta: 0.0)
        gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixRes, resultMatrix: matrixG)

        if let pipeline = ctx.getPipeline(name: "multinomial_hessians", functionName: "multinomial_hessians") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(xBuffer, offset: 0, index: 0)
            enc.setBuffer(probBuffer, offset: 0, index: 1)
            enc.setBuffer(sumBuffer, offset: 0, index: 2)
            enc.setBuffer(hBuffer, offset: 0, index: 3)
            var nU = UInt32(n); var pU = UInt32(p); var cU = UInt32(n_classes)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 6)
            let tgSize = MTLSize(width: 8, height: 8, depth: 1)
            let tgCount = MTLSize(width: (p + 7) / 8, height: (p + 7) / 8, depth: n_classes)
            enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
        }

        if let pipeline = ctx.getPipeline(name: "transpose_f32", functionName: "transpose_f32") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(gBuffer, offset: 0, index: 0)
            enc.setBuffer(gradBatchBuffer, offset: 0, index: 1)
            var rowsU = UInt32(p); var colsU = UInt32(n_classes)
            enc.setBytes(&rowsU, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&colsU, length: MemoryLayout<UInt32>.stride, index: 3)
            let tgSize = MTLSize(width: 16, height: 16, depth: 1)
            let tgCount = MTLSize(width: (n_classes + 15) / 16, height: (p + 15) / 16, depth: 1)
            enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
        }

        if alpha != 0 {
            if let pipeline = ctx.getPipeline(name: "multinomial_l2_reg", functionName: "multinomial_l2_reg") {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(hBuffer, offset: 0, index: 0)
                enc.setBuffer(gradBatchBuffer, offset: 0, index: 1)
                enc.setBuffer(wBuffer, offset: 0, index: 2)
                var a = alpha
                enc.setBytes(&a, length: fs, index: 3)
                var pU = UInt32(p); var cU = UInt32(n_classes)
                enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 4)
                enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 5)
                let tgSize = MTLSize(width: 16, height: 16, depth: 1)
                let tgCount = MTLSize(width: (n_classes + 15) / 16, height: (p + 15) / 16, depth: 1)
                enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
                enc.endEncoding()
            }
        }

        cb.commit()
        cb.waitUntilCompleted()

        let hBase = hBuffer.contents().assumingMemoryBound(to: Float.self)
        let gBase = gradBatchBuffer.contents().assumingMemoryBound(to: Float.self)
        let dBase = dBuffer.contents().assumingMemoryBound(to: Float.self)

        var uplo: CChar = 76
        var n_ = p32, nrhs_ = one32, lda_ = p32, ldb_ = p32
        var info: Int32 = 0
        for c in 0..<n_classes {
            let hOff = c * p * p
            let gOff = c * p
            let hPtr = hBase.advanced(by: hOff)
            let gPtr = gBase.advanced(by: gOff)
            spotrf_(&uplo, &n_, hPtr, &lda_, &info)
            guard info == 0 else { return 1 }
            spotrs_(&uplo, &n_, &nrhs_, hPtr, &lda_, gPtr, &ldb_, &info)
            guard info == 0 else { return 1 }
        }

        for c in 0..<n_classes {
            for i in 0..<p {
                dBase[i * n_classes + c] = gBase[c * p + i]
            }
        }

        let gNorm = cblas_snrm2(p32 * Int32(n_classes), gBase, 1)
        let wPtr = wBuffer.contents().assumingMemoryBound(to: Float.self)
        let wNorm = cblas_snrm2(p32 * Int32(n_classes), wPtr, 1)
        if gNorm < tol * max(1.0, wNorm) { break }

        cblas_saxpy(p32 * Int32(n_classes), -1.0, dBase, 1, wPtr, 1)
    }

    let wOut = W_out.assumingMemoryBound(to: Float.self)
    let wFinal = wBuffer.contents().assumingMemoryBound(to: Float.self)
    memcpy(wOut, wFinal, wSize)
    n_iter_out?.pointee = nIter
    return 0
}

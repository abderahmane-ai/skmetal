import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate



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
    let matrixD = MPSMatrix(buffer: dBuffer, descriptor: descG)

    guard let statusBuffer = ctx.device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared) else {
        return 1
    }

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

        let cholesky = MPSMatrixDecompositionCholesky(device: ctx.device, lower: true, order: p)
        cholesky.encode(commandBuffer: cb, sourceMatrix: matrixH, resultMatrix: matrixH, status: statusBuffer)

        let solve = MPSMatrixSolveCholesky(device: ctx.device, upper: false, order: p, numberOfRightHandSides: 1)
        solve.encode(commandBuffer: cb, sourceMatrix: matrixH, rightHandSideMatrix: matrixG, solutionMatrix: matrixD)

        cb.commit()
        cb.waitUntilCompleted()

        let status = statusBuffer.contents().assumingMemoryBound(to: Int32.self)[0]
        guard status == 0 else { return 1 }

        let dPtr = dBuffer.contents().assumingMemoryBound(to: Float.self)
        let wPtr = wBuffer.contents().assumingMemoryBound(to: Float.self)
        let stepNorm = cblas_snrm2(Int32(p), dPtr, 1)
        let wNorm = cblas_snrm2(Int32(p), wPtr, 1)
        if stepNorm < tol * max(1.0, wNorm) { break }

        cblas_saxpy(Int32(p), -1.0, dPtr, 1, wPtr, 1)
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
    let pPacked = Int(p) * (Int(p) + 1) / 2
    let hessiansSize = pPacked * n_classes * fs

    let alpha: Float = 1.0 / C

    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: n * fs, device: ctx.device) else {
        return 1
    }

    guard let wBuffer = ctx.device.makeBuffer(length: wSize, options: .storageModeShared),
          let scoresBuffer = ctx.device.makeBuffer(length: scoresSize, options: .storageModeShared),
          let probBuffer = ctx.device.makeBuffer(length: scoresSize, options: .storageModeShared),
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
            enc.setBuffer(hBuffer, offset: 0, index: 2)
            var a = alpha
            enc.setBytes(&a, length: fs, index: 3)
            var nU = UInt32(n); var pU = UInt32(p); var cU = UInt32(n_classes)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 6)
            let pPackedSize = Int(p) * (Int(p) + 1) / 2
            let tgSize = MTLSize(width: 256, height: 1, depth: 1)
            let tgCount = MTLSize(width: (pPackedSize + 255) / 256, height: n_classes, depth: 1)
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
            if let pipeline = ctx.getPipeline(name: "multinomial_grad_l2", functionName: "multinomial_grad_l2") {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(gradBatchBuffer, offset: 0, index: 0)
                enc.setBuffer(wBuffer, offset: 0, index: 1)
                var a = alpha
                enc.setBytes(&a, length: fs, index: 2)
                var pU = UInt32(p); var cU = UInt32(n_classes)
                enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 3)
                enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 4)
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
        var info: Int32 = 0
        var nrhs_ = Int32(1), ldb_ = p32
        for c in 0..<n_classes {
            let hOff = c * pPacked
            let gOff = c * p
            let hPtr = hBase.advanced(by: hOff)
            let gPtr = gBase.advanced(by: gOff)
            var n_ = p32
            spptrf_(&uplo, &n_, hPtr, &info)
            guard info == 0 else { return 1 }
            spptrs_(&uplo, &n_, &nrhs_, hPtr, gPtr, &ldb_, &info)
            if info != 0 {
                let msg = String(format: "spotrf failed: class=%d info=%d", c, info)
                msg.withCString { fputs($0, stderr); fputs("\n", stderr) }
                return 1
            }
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

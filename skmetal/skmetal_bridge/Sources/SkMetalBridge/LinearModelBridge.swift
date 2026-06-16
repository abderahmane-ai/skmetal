import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - Fused Ridge: center + XTX + XTy + L2 + Cholesky solve (one command buffer)

@_cdecl("skmetal_ridge_fit_solve")
public func skmetal_ridge_fit_solve(
    X: UnsafeMutableRawPointer,
    y: UnsafeRawPointer,
    X_mean_out: UnsafeMutableRawPointer,
    coef_out: UnsafeMutableRawPointer,
    alpha: Float,
    n: Int,
    p: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let xSize = n * p * fs
    let ySize = n * fs
    let xtxSize = p * p * fs
    let xtySize = p * fs
    let meanSize = p * fs

    guard let xBuffer = wrapOutput(X, length: xSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: ySize, device: ctx.device),
          let meanBuffer = wrapOutput(X_mean_out, length: meanSize, device: ctx.device),
          let coefBuffer = wrapOutput(coef_out, length: xtySize, device: ctx.device) else {
        return 1
    }

    let xtxB = ctx.reusableBuffer(length: xtxSize)
    let xtyB = ctx.reusableBuffer(length: xtySize)
    guard let xtxBuffer = xtxB, let xtyBuffer = xtyB,
          let statusBuffer = ctx.device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared),
          let addDiagPpl = ctx.getPipeline(name: "add_diagonal", functionName: "add_diagonal") else {
        if let b = xtxB { ctx.recycleBuffer(b) }
        if let b = xtyB { ctx.recycleBuffer(b) }
        return 1
    }

    let rowBytesX = p * fs
    let rowBytesXTX = p * fs

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descXTX = MPSMatrixDescriptor(dimensions: p, columns: p, rowBytes: rowBytesXTX, dataType: .float32)
    let descY = MPSMatrixDescriptor(dimensions: n, columns: 1, rowBytes: fs, dataType: .float32)
    let descVec = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)

    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixXTX = MPSMatrix(buffer: xtxBuffer, descriptor: descXTX)
    let matrixY = MPSMatrix(buffer: yBuffer, descriptor: descY)
    let matrixXTy = MPSMatrix(buffer: xtyBuffer, descriptor: descVec)
    let matrixCoef = MPSMatrix(buffer: coefBuffer, descriptor: descVec)

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    var nU = UInt32(n), pU = UInt32(p)

    guard let computeEncoder = cb.makeComputeCommandEncoder() else { return 1 }
    if let pipeline = ctx.getPipeline(name: "column_means", functionName: "column_means") {
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(meanBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 3)
        let blockCols = 8
        let tgCount = (p + blockCols - 1) / blockCols
        computeEncoder.dispatchThreadgroups(MTLSize(width: tgCount, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    computeEncoder.endEncoding()

    guard let centerEncoder = cb.makeComputeCommandEncoder() else { return 1 }
    if let pipeline = ctx.getPipeline(name: "center_columns", functionName: "center_columns") {
        centerEncoder.setComputePipelineState(pipeline)
        centerEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        centerEncoder.setBuffer(meanBuffer, offset: 0, index: 1)
        centerEncoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        centerEncoder.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 3)
        centerEncoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    centerEncoder.endEncoding()

    let gemmXTX = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: p, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmXTX.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixX, resultMatrix: matrixXTX)

    let gemmXTy = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: 1, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmXTy.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixY, resultMatrix: matrixXTy)

    guard let encDiag = cb.makeComputeCommandEncoder() else { return 1 }
    encDiag.setComputePipelineState(addDiagPpl)
    encDiag.setBuffer(xtxBuffer, offset: 0, index: 0)
    var a = alpha
    encDiag.setBytes(&a, length: fs, index: 1)
    encDiag.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 2)
    encDiag.dispatchThreads(MTLSize(width: p, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encDiag.endEncoding()

    let cholesky = MPSMatrixDecompositionCholesky(device: ctx.device, lower: true, order: p)
    cholesky.encode(commandBuffer: cb, sourceMatrix: matrixXTX, resultMatrix: matrixXTX, status: statusBuffer)

    let solve = MPSMatrixSolveCholesky(device: ctx.device, upper: false, order: p, numberOfRightHandSides: 1)
    solve.encode(commandBuffer: cb, sourceMatrix: matrixXTX, rightHandSideMatrix: matrixXTy, solutionMatrix: matrixCoef)

    cb.commit()
    cb.waitUntilCompleted()

    let status = statusBuffer.contents().assumingMemoryBound(to: Int32.self)[0]
    ctx.recycleBuffer(xtxBuffer)
    ctx.recycleBuffer(xtyBuffer)
    guard status == 0 else { return 1 }

    return 0
}

// MARK: - Linear solve: unregularized Cholesky solve on GPU (LinearRegression)

// Internal Cholesky solver (not exported to Python — ridge_fit_solve handles both cases)
func skmetal_linear_solve(
    XTX: UnsafeMutableRawPointer,
    XTy: UnsafeMutableRawPointer,
    coef_out: UnsafeMutableRawPointer,
    p: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let xtxSize = p * p * fs
    let xtySize = p * fs

    guard let xtxBuffer = wrapOutput(XTX, length: xtxSize, device: ctx.device),
          let xtyBuffer = wrapOutput(XTy, length: xtySize, device: ctx.device),
          let coefBuffer = wrapOutput(coef_out, length: xtySize, device: ctx.device) else {
        return 1
    }

    guard let statusBuffer = ctx.device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared) else {
        return 1
    }

    let rowBytes = p * fs
    let descXTX = MPSMatrixDescriptor(dimensions: p, columns: p, rowBytes: rowBytes, dataType: .float32)
    let descVec = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)

    let matrixXTX = MPSMatrix(buffer: xtxBuffer, descriptor: descXTX)
    let matrixXTy = MPSMatrix(buffer: xtyBuffer, descriptor: descVec)
    let matrixCoef = MPSMatrix(buffer: coefBuffer, descriptor: descVec)

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }

    let cholesky = MPSMatrixDecompositionCholesky(device: ctx.device, lower: true, order: p)
    cholesky.encode(commandBuffer: cb, sourceMatrix: matrixXTX, resultMatrix: matrixXTX, status: statusBuffer)

    let solve = MPSMatrixSolveCholesky(device: ctx.device, upper: false, order: p, numberOfRightHandSides: 1)
    solve.encode(commandBuffer: cb, sourceMatrix: matrixXTX, rightHandSideMatrix: matrixXTy, solutionMatrix: matrixCoef)

    cb.commit()
    cb.waitUntilCompleted()

    let status = statusBuffer.contents().assumingMemoryBound(to: Int32.self)[0]
    guard status == 0 else { return 1 }

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

    guard let xtxBuf = ctx.device.makeBuffer(length: ppBufSize, options: .storageModeShared),
          let xtyBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let xBuf_g = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let zBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let xPrevBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let xTempBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
          let gradBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared) else {
        return 1
    }

    memset(xBuf_g.contents(), 0, pBufSize)
    memset(zBuf.contents(), 0, pBufSize)

    guard let axpyPpl = ctx.getPipeline(name: "axpy", functionName: "axpy"),
          let subPpl = ctx.getPipeline(name: "subtract", functionName: "subtract"),
          let stPpl = ctx.getPipeline(name: "soft_threshold", functionName: "soft_threshold"),
          let scalePpl = ctx.getPipeline(name: "scale_f32", functionName: "scale_f32"),
          let _ = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum") else {
        return 1
    }

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

    do {
        guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }
        let gemm = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: p, interiorColumns: n,
            alpha: 1.0, beta: 0.0)
        gemm.encode(commandBuffer: cb, leftMatrix: mX, rightMatrix: mX, resultMatrix: mXTX)
        cb.commit()
        cb.waitUntilCompleted()
    }

    do {
        guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }
        let gemm = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: 1, interiorColumns: n,
            alpha: 1.0, beta: 0.0)
        gemm.encode(commandBuffer: cb, leftMatrix: mX, rightMatrix: mY, resultMatrix: mXTy)
        cb.commit()
        cb.waitUntilCompleted()
    }

    let L: Float = {
        // Use Accelerate BLAS on CPU: XTX is only p×p (tiny for p ≤ 100).
        // Much simpler than GPU power iteration — no batching bugs, no NaN from float32 underflow.
        let xtxPtr = xtxBuf.contents().assumingMemoryBound(to: Float.self)
        var v = [Float](repeating: 0, count: p)
        var u = [Float](repeating: 0, count: p)
        for i in 0..<p { v[i] = xtxPtr[i * p] }  // diagonal initial vector
        for _ in 0..<20 {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(p), Int32(p),
                        1.0, xtxPtr, Int32(p), v, 1, 0.0, &u, 1)
            let norm = cblas_snrm2(Int32(p), &u, 1)
            guard norm >= 1e-10, !norm.isNaN else { return 0.0 }  // fallback: safe but slow
            for i in 0..<p { v[i] = u[i] / norm }
        }
        var Lv: Float = 0
        for i in 0..<p { Lv += v[i] * u[i] }
      return abs(Lv)
    }()

    if L <= 1e-10 { return 1 }
    let step = 1.0 / L
    let thresh = step * alpha * l1_ratio * Float(n)
    let enDenom: Float = (l1_ratio < 1.0) ? (1.0 + step * alpha * (1.0 - l1_ratio) * Float(n)) : 1.0
    let enScale: Float = (enDenom != 1.0) ? (1.0 / enDenom) : 1.0

    let tg256 = MTLSize(width: 256, height: 1, depth: 1)
    let grd256 = MTLSize(width: (p + 255) / 256, height: 1, depth: 1)
    // GPU-side convergence: encode iterations in batches of 50 per command buffer.
    // Avoids encoding 1000s of encoders into one enormous CB (Metal CB overhead ~O(N)).
    let batchSize = Int(max_iter)
    let cbBatch = 50
    let partialsPerIter = max(1, (p + 255) / 256)
    let convBufSize = batchSize * partialsPerIter * fs
    let snapBufSize = batchSize * pBufSize
    guard let convBuf = ctx.reusableBuffer(length: convBufSize),
          let snapBuf = ctx.reusableBuffer(length: snapBufSize),
          let maxDiffPpl = ctx.getPipeline(name: "max_abs_diff", functionName: "max_abs_diff") else {
        return 1
    }

    var t: Float = 1.0
    var globalIt: Int = 0
    var convergedInBatch = false

    while globalIt < batchSize && !convergedInBatch {
        let batchEnd = min(globalIt + cbBatch, batchSize)
        guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }

        for batchIt in globalIt..<batchEnd {

            let blit1 = cb.makeBlitCommandEncoder()!
            blit1.copy(from: xBuf_g, sourceOffset: 0, to: xPrevBuf, destinationOffset: 0, size: pBufSize)
            blit1.endEncoding()

            let gemm = MPSMatrixMultiplication(
                device: ctx.device, transposeLeft: false, transposeRight: false,
                resultRows: p, resultColumns: 1, interiorColumns: p,
                alpha: 1.0, beta: 0.0)
            gemm.encode(commandBuffer: cb, leftMatrix: mXTX, rightMatrix: mZ, resultMatrix: mGrad)

            guard let encGradSub = cb.makeComputeCommandEncoder() else { return 1 }
            encGradSub.setComputePipelineState(axpyPpl)
            encGradSub.setBuffer(gradBuf, offset: 0, index: 0)
            encGradSub.setBuffer(xtyBuf, offset: 0, index: 1)
            var m1: Float = -1.0
            encGradSub.setBytes(&m1, length: fs, index: 2)
            var nU = UInt32(p)
            encGradSub.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            encGradSub.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
            encGradSub.endEncoding()

            let blit2 = cb.makeBlitCommandEncoder()!
            blit2.copy(from: zBuf, sourceOffset: 0, to: xTempBuf, destinationOffset: 0, size: pBufSize)
            blit2.endEncoding()

            guard let encStep = cb.makeComputeCommandEncoder() else { return 1 }
            encStep.setComputePipelineState(axpyPpl)
            encStep.setBuffer(xTempBuf, offset: 0, index: 0)
            encStep.setBuffer(gradBuf, offset: 0, index: 1)
            var negStep: Float = -step
            encStep.setBytes(&negStep, length: fs, index: 2)
            nU = UInt32(p)
            encStep.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            encStep.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
            encStep.endEncoding()

            guard let encST = cb.makeComputeCommandEncoder() else { return 1 }
            encST.setComputePipelineState(stPpl)
            encST.setBuffer(xBuf_g, offset: 0, index: 0)
            encST.setBuffer(xTempBuf, offset: 0, index: 1)
            var thr = thresh
            encST.setBytes(&thr, length: fs, index: 2)
            nU = UInt32(p)
            encST.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            encST.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
            encST.endEncoding()

            if enScale != 1.0 {
                guard let encScale = cb.makeComputeCommandEncoder() else { return 1 }
                encScale.setComputePipelineState(scalePpl)
                encScale.setBuffer(xBuf_g, offset: 0, index: 0)
                var sc = enScale
                encScale.setBytes(&sc, length: fs, index: 1)
                nU = UInt32(p)
                encScale.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
                encScale.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
                encScale.endEncoding()
            }

            let tPrev = t
            t = (1.0 + sqrt(1.0 + 4.0 * tPrev * tPrev)) / 2.0
            let factor = (tPrev - 1.0) / t

            guard let encSub = cb.makeComputeCommandEncoder() else { return 1 }
            encSub.setComputePipelineState(subPpl)
            encSub.setBuffer(xBuf_g, offset: 0, index: 0)
            encSub.setBuffer(xPrevBuf, offset: 0, index: 1)
            encSub.setBuffer(xTempBuf, offset: 0, index: 2)
            nU = UInt32(p)
            encSub.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            encSub.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
            encSub.endEncoding()

            let blit3 = cb.makeBlitCommandEncoder()!
            blit3.copy(from: xBuf_g, sourceOffset: 0, to: zBuf, destinationOffset: 0, size: pBufSize)
            blit3.endEncoding()

            guard let encZUp = cb.makeComputeCommandEncoder() else { return 1 }
            encZUp.setComputePipelineState(axpyPpl)
            encZUp.setBuffer(zBuf, offset: 0, index: 0)
            encZUp.setBuffer(xTempBuf, offset: 0, index: 1)
            var fac = factor
            encZUp.setBytes(&fac, length: fs, index: 2)
            nU = UInt32(p)
            encZUp.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            encZUp.dispatchThreadgroups(grd256, threadsPerThreadgroup: tg256)
            encZUp.endEncoding()

            let snapBlit = cb.makeBlitCommandEncoder()!
            snapBlit.copy(from: xBuf_g, sourceOffset: 0,
                          to: snapBuf, destinationOffset: batchIt * pBufSize,
                          size: pBufSize)
            snapBlit.endEncoding()

            guard let encConv = cb.makeComputeCommandEncoder() else { return 1 }
            encConv.setComputePipelineState(maxDiffPpl)
            encConv.setBuffer(xBuf_g, offset: 0, index: 0)
            encConv.setBuffer(xPrevBuf, offset: 0, index: 1)
            encConv.setBuffer(convBuf, offset: 0, index: 2)
            var pU = UInt32(p)
            var ngU = UInt32(partialsPerIter)
            var wOff = UInt32(batchIt * partialsPerIter)
            encConv.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 3)
            encConv.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 4)
            encConv.setBytes(&wOff, length: MemoryLayout<UInt32>.stride, index: 5)
            encConv.dispatchThreadgroups(MTLSize(width: partialsPerIter, height: 1, depth: 1),
                                         threadsPerThreadgroup: tg256)
            encConv.endEncoding()
        }

        cb.commit()
        cb.waitUntilCompleted()

        // Early convergence check: if converged inside this batch, stop encoding more CBs
        let convPtr = convBuf.contents().assumingMemoryBound(to: Float.self)
        for iteration in globalIt..<batchEnd {
            let base = iteration * partialsPerIter
            var maxVal: Float = 0
            for j in 0..<partialsPerIter {
                if convPtr[base + j] > maxVal { maxVal = convPtr[base + j] }
            }
            if maxVal < tol {
                convergedInBatch = true
                break
            }
        }

        globalIt = batchEnd
    }

    // Post-process: find first converged iteration from convBuf
    let convPtr = convBuf.contents().assumingMemoryBound(to: Float.self)
    var convergedAt = batchSize - 1
    for iteration in 0..<batchSize {
        let base = iteration * partialsPerIter
        var maxVal: Float = 0
        for j in 0..<partialsPerIter {
            if convPtr[base + j] > maxVal { maxVal = convPtr[base + j] }
        }
        if maxVal < tol {
            convergedAt = iteration
            break
        }
    }

    let snapBase = snapBuf.contents().assumingMemoryBound(to: Float.self)
    memcpy(coefBuf.contents(), snapBase + convergedAt * p, pBufSize)
    n_iter_out?.pointee = Int32(convergedAt + 1)
    ctx.recycleBuffer(convBuf)
    ctx.recycleBuffer(snapBuf)
    return 0
}

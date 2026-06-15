import Foundation
import Metal
import MetalPerformanceShaders


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

    // Fused IRLS: encode iterations in batches of 20 per command buffer.
    // Avoids encoding 100s of encoders into one enormous CB.
    let maxIter = Int(max_iter)
    let cbBatch = 20
    let ng = max(1, (p + 255) / 256)
    let convEntries = ng * 2  // step_norm_sq + w_norm_sq per iteration
    let convBufSize = maxIter * convEntries * fs
    let snapBufSize = maxIter * pSize
    let statusSnapSize = maxIter * MemoryLayout<Int32>.stride

    guard let convBuf = ctx.reusableBuffer(length: convBufSize),
          let snapBuf = ctx.reusableBuffer(length: snapBufSize),
          let statusSnapBuf = ctx.reusableBuffer(length: statusSnapSize),
          let norm2Ppl = ctx.getPipeline(name: "norm2", functionName: "norm2"),
          let axpyPpl = ctx.getPipeline(name: "axpy", functionName: "axpy") else {
        return 1
    }

    // Reusable MPS objects (same dimensions every iteration)
    let gemmXW = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: 1, interiorColumns: p,
        alpha: 1.0, beta: 0.0)
    let gemmHH = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: p, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    let gemmGrad = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: 1, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    let cholesky = MPSMatrixDecompositionCholesky(device: ctx.device, lower: true, order: p)
    let solve = MPSMatrixSolveCholesky(device: ctx.device, upper: false, order: p, numberOfRightHandSides: 1)

    let tg256 = MTLSize(width: 256, height: 1, depth: 1)
    let ngSize = MTLSize(width: ng, height: 1, depth: 1)

    var globalIt = 0
    while globalIt < maxIter {
        let batchEnd = min(globalIt + cbBatch, maxIter)
        let cb = ctx.commandQueue.makeCommandBuffer()!

        for it in globalIt..<batchEnd {
            // X @ w → linear
            gemmXW.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixLin)

            // Sigmoid → weight
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
                                    threadsPerThreadgroup: tg256)
                enc.endEncoding()
            }

            // X_scaled = X * sqrt(weight[i]), error = sigmoid - y
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
                                    threadsPerThreadgroup: tg256)
                enc.endEncoding()
            }

            // Hessian = X_scaledᵀ @ X_scaled
            gemmHH.encode(commandBuffer: cb, leftMatrix: matrixXS, rightMatrix: matrixXS, resultMatrix: matrixH)

            // Gradient = Xᵀ @ error
            gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixLin, resultMatrix: matrixG)

            // L2: Hessian += α·I, gradient += α·w
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
                                        threadsPerThreadgroup: tg256)
                    enc.endEncoding()
                }
            }

            // Cholesky solve: H \ g → d
            cholesky.encode(commandBuffer: cb, sourceMatrix: matrixH, resultMatrix: matrixH, status: statusBuffer)
            solve.encode(commandBuffer: cb, sourceMatrix: matrixH, rightHandSideMatrix: matrixG, solutionMatrix: matrixD)

            // Snapshot current w (pre-update) for convergence retrieval
            let snapBlit = cb.makeBlitCommandEncoder()!
            snapBlit.copy(from: wBuffer, sourceOffset: 0,
                          to: snapBuf, destinationOffset: it * pSize,
                          size: pSize)
            snapBlit.endEncoding()

            // Snapshot Cholesky status
            let statusBlit = cb.makeBlitCommandEncoder()!
            statusBlit.copy(from: statusBuffer, sourceOffset: 0,
                            to: statusSnapBuf, destinationOffset: it * MemoryLayout<Int32>.stride,
                            size: MemoryLayout<Int32>.stride)
            statusBlit.endEncoding()

            // ||d||² → convBuf
            let encStepNorm = cb.makeComputeCommandEncoder()!
            encStepNorm.setComputePipelineState(norm2Ppl)
            encStepNorm.setBuffer(dBuffer, offset: 0, index: 0)
            encStepNorm.setBuffer(convBuf, offset: 0, index: 1)
            var pU32 = UInt32(p)
            var ngU32 = UInt32(ng)
            var stepOff = UInt32(it * convEntries)
            encStepNorm.setBytes(&pU32, length: MemoryLayout<UInt32>.stride, index: 2)
            encStepNorm.setBytes(&ngU32, length: MemoryLayout<UInt32>.stride, index: 3)
            encStepNorm.setBytes(&stepOff, length: MemoryLayout<UInt32>.stride, index: 4)
            encStepNorm.dispatchThreadgroups(ngSize, threadsPerThreadgroup: tg256)
            encStepNorm.endEncoding()

            // ||w||² → convBuf
            let encWNorm = cb.makeComputeCommandEncoder()!
            encWNorm.setComputePipelineState(norm2Ppl)
            encWNorm.setBuffer(wBuffer, offset: 0, index: 0)
            encWNorm.setBuffer(convBuf, offset: 0, index: 1)
            var wOff = UInt32(it * convEntries + ng)
            encWNorm.setBytes(&pU32, length: MemoryLayout<UInt32>.stride, index: 2)
            encWNorm.setBytes(&ngU32, length: MemoryLayout<UInt32>.stride, index: 3)
            encWNorm.setBytes(&wOff, length: MemoryLayout<UInt32>.stride, index: 4)
            encWNorm.dispatchThreadgroups(ngSize, threadsPerThreadgroup: tg256)
            encWNorm.endEncoding()

            // w -= d (GPU axpy)
            let encAxpy = cb.makeComputeCommandEncoder()!
            encAxpy.setComputePipelineState(axpyPpl)
            encAxpy.setBuffer(wBuffer, offset: 0, index: 0)
            encAxpy.setBuffer(dBuffer, offset: 0, index: 1)
            var negOne: Float = -1.0
            encAxpy.setBytes(&negOne, length: fs, index: 2)
            encAxpy.setBytes(&pU32, length: MemoryLayout<UInt32>.stride, index: 3)
            encAxpy.dispatchThreadgroups(ngSize, threadsPerThreadgroup: tg256)
            encAxpy.endEncoding()
        }

        cb.commit()
        cb.waitUntilCompleted()
        globalIt = batchEnd
    }

    // Post-process: find converged iteration from buffers
    let convBase = convBuf.contents().assumingMemoryBound(to: Float.self)
    let statusBase = statusSnapBuf.contents().assumingMemoryBound(to: Int32.self)
    let snapBase = snapBuf.contents().assumingMemoryBound(to: Float.self)
    var nIter: Int32 = 0
    var convergedAt = -1

    for it in 0..<maxIter {
        guard statusBase[it] == 0 else { break }
        nIter = Int32(it + 1)

        let base = it * convEntries
        var stepSq: Float = 0
        var wSq: Float = 0
        for j in 0..<ng {
            stepSq += convBase[base + j]
            wSq += convBase[base + ng + j]
        }
        let stepNorm = sqrt(stepSq)
        let wNorm = sqrt(wSq)

        if stepNorm < tol * max(1.0, wNorm) {
            convergedAt = it
            break
        }
    }

    let coefOut = coef_out.assumingMemoryBound(to: Float.self)
    if convergedAt >= 0 {
        memcpy(coefOut, snapBase + convergedAt * p, pSize)
    } else if nIter > 0 {
        memcpy(coefOut, snapBase + (Int(nIter) - 1) * p, pSize)
    }
    n_iter_out?.pointee = nIter
    ctx.recycleBuffer(convBuf)
    ctx.recycleBuffer(snapBuf)
    ctx.recycleBuffer(statusSnapBuf)
    return 0
}

// MARK: - L-BFGS full loop for binary LogisticRegression

@_cdecl("skmetal_logreg_lbfgs_fit")
public func skmetal_logreg_lbfgs_fit(
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

    let alpha: Float = 1.0 / C   // L2 regularization = alpha/2 * ||w||²
    let alphaN: Float = alpha / Float(n)  // scaled by n to match sklearn: C*sum(loss) + 0.5*||w||²
    let maxNg = max(1, (n + 255) / 256)

    // Allocate all buffers upfront
    guard let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: nSize, device: ctx.device),
          let wBuffer = ctx.device.makeBuffer(length: pSize, options: .storageModeShared),
          let gBuffer = ctx.device.makeBuffer(length: pSize, options: .storageModeShared),
          let linBuffer = ctx.device.makeBuffer(length: nSize, options: .storageModeShared),
          let lossBuf = ctx.device.makeBuffer(length: nSize, options: .storageModeShared),
          let sumBuf = ctx.device.makeBuffer(length: maxNg * fs, options: .storageModeShared) else {
        return 1
    }
    memset(wBuffer.contents(), 0, pSize)

    // Descriptors
    let rowBytesX = p * fs
    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: rowBytesX, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)
    let descLin = MPSMatrixDescriptor(dimensions: n, columns: 1, rowBytes: fs, dataType: .float32)
    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixLin = MPSMatrix(buffer: linBuffer, descriptor: descLin)

    // L-BFGS memory (m = 10)
    let m = 10
    let mSize = m * p * fs
    guard let sBuf = ctx.device.makeBuffer(length: mSize, options: .storageModeShared),
          let yBuf = ctx.device.makeBuffer(length: mSize, options: .storageModeShared),
          let rhoBuf = ctx.device.makeBuffer(length: m * fs, options: .storageModeShared) else {
        return 1
    }
    let sBase = sBuf.contents().assumingMemoryBound(to: Float.self)
    let yBase = yBuf.contents().assumingMemoryBound(to: Float.self)
    let rhoBase = rhoBuf.contents().assumingMemoryBound(to: Float.self)
    let wPtr = wBuffer.contents().assumingMemoryBound(to: Float.self)
    let gPtr = gBuffer.contents().assumingMemoryBound(to: Float.self)
    let sumPtr = sumBuf.contents().assumingMemoryBound(to: Float.self)

    let maxIter = Int(max_iter)
    var nIter: Int32 = 0

    // Buffers for line search / two-loop recursion (on CPU)
    var d = [Float](repeating: 0, count: p)       // search direction
    var wCpu = [Float](repeating: 0, count: p)    // CPU copy of w for sᵀy
    var gCpu = [Float](repeating: 0, count: p)    // CPU copy of gradient for sᵀy

    // Helper: compute wᵀ Xᵀ X w = ||Xw||² (L2 regularization term)
    func computeLoss() -> Float {
        // lin = X @ w is already computed
        // reduce sum of loss_i buffer
        let ng = max(1, (n + 255) / 256)
        let cb = ctx.commandQueue.makeCommandBuffer()!
        if let rsPpl = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(rsPpl)
            enc.setBuffer(lossBuf, offset: 0, index: 0)
            enc.setBuffer(sumBuf, offset: 0, index: 1)
            var nU = UInt32(n); var ngU = UInt32(ng)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: ng, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        var loss = Float(0)
        for i in 0..<ng { loss += sumPtr[i] }
        loss /= Float(n)
        if alpha != 0 {
            var wNorm2: Float = 0
            for i in 0..<p { wNorm2 += wPtr[i] * wPtr[i] }
            loss += Float(0.5) * alphaN * wNorm2
        }
        return loss
    }

    // Helper: compute wᵀ Xᵀ X w = ||Xw||² (L2 term)
    // Not needed separately — computeLoss() includes L2.

    // Helper: gradient = Xᵀ(σ(Xw) - y)/n + α·w  (overwrites wPtr with gradient)
    // Also returns loss (via computeLoss which reads lossBuf)
    func computeGradientLoss() -> Float {
        let cb = ctx.commandQueue.makeCommandBuffer()!

        // 1. lin = X @ w (MPS GEMM)
        let gemmLin = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: false, transposeRight: false,
            resultRows: n, resultColumns: 1, interiorColumns: p,
            alpha: 1.0, beta: 0.0)
        gemmLin.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixLin)

        // 2. Sigmoid + residual + log loss (fused kernel)
        if let pipeline = ctx.getPipeline(name: "sigmoid_grad_loss_binary",
                                          functionName: "sigmoid_grad_loss_binary") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(linBuffer, offset: 0, index: 0)
            enc.setBuffer(yBuffer, offset: 0, index: 1)
            enc.setBuffer(linBuffer, offset: 0, index: 2)   // reuse lin for prob
            enc.setBuffer(linBuffer, offset: 0, index: 3)   // reuse lin for residual
            enc.setBuffer(lossBuf, offset: 0, index: 4)
            var nU = UInt32(n)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 3. Gradient: grad = Xᵀ @ residual / n
        //   linBuffer now holds residual (overwritten by step 2)
        let gradDesc = MPSMatrixDescriptor(dimensions: p, columns: 1, rowBytes: fs, dataType: .float32)
        let matrixGrad = MPSMatrix(buffer: gBuffer, descriptor: gradDesc)
        let matrixRes = MPSMatrix(buffer: linBuffer, descriptor: descLin)
        let gemmGrad = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: 1, interiorColumns: n,
            alpha: 1.0 / Double(n), beta: 0.0)
        gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixRes, resultMatrix: matrixGrad)

        // 4. L2: grad[i] += alpha * w[i]
        if alpha != 0 {
            if let pipeline = ctx.getPipeline(name: "axpy", functionName: "axpy") {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(gBuffer, offset: 0, index: 0)  // output: g += a*x
                enc.setBuffer(wBuffer, offset: 0, index: 1)  // x = original w
                var a = alphaN
                enc.setBytes(&a, length: fs, index: 2)
                var nU32 = UInt32(p)
                enc.setBytes(&nU32, length: MemoryLayout<UInt32>.stride, index: 3)
                enc.dispatchThreads(MTLSize(width: p, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
        }

        // 5. Reduce sum of loss elements
        let ng = max(1, (n + 255) / 256)
        if let rsPpl = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(rsPpl)
            enc.setBuffer(lossBuf, offset: 0, index: 0)
            enc.setBuffer(sumBuf, offset: 0, index: 1)
            var nU = UInt32(n); var ngU = UInt32(ng)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: ng, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        cb.commit()
        cb.waitUntilCompleted()

        var loss = Float(0)
        for i in 0..<ng { loss += sumPtr[i] }
        loss /= Float(n)
        if alpha != 0 {
            var wNorm2: Float = 0
            for i in 0..<p { wNorm2 += wPtr[i] * wPtr[i] }
            loss += Float(0.5) * alphaN * wNorm2
        }
        return loss
    }

    // Line search buffers (reused across main iterations)
    let maxTrials = 20
    let trialBufSize = maxTrials * pSize
    let lsNg = max(1, (n + 255) / 256)
    let lsConvBufSize = maxTrials * lsNg * fs
    guard let trialWbuf = ctx.device.makeBuffer(length: trialBufSize, options: .storageModeShared),
          let lsConvBuf = ctx.device.makeBuffer(length: lsConvBufSize, options: .storageModeShared),
          let logLossPpl = ctx.getPipeline(name: "log_loss_binary", functionName: "log_loss_binary"),
          let rsPpl = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum") else {
        return 1
    }
    let lsTg256 = MTLSize(width: 256, height: 1, depth: 1)
    let lsNgSize = MTLSize(width: lsNg, height: 1, depth: 1)
    let gemmLinLS = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: false, transposeRight: false,
        resultRows: n, resultColumns: 1, interiorColumns: p,
        alpha: 1.0, beta: 0.0)

    // ------ Main L-BFGS loop ------
    var loss = computeGradientLoss()  // initial gradient + loss
    memcpy(&wCpu, wPtr, pSize)
    memcpy(&gCpu, gPtr, pSize)

    for it in 0..<maxIter {
        nIter = Int32(it + 1)

        // Convergence check: ||g||_∞ < tol * max(1, ||w||₂)
        var gNorm: Float = 0
        var wNorm2: Float = 0
        for i in 0..<p {
            gNorm = max(gNorm, abs(gCpu[i]))
            wNorm2 += wCpu[i] * wCpu[i]
        }
        let wNorm = sqrt(wNorm2)
        if gNorm < tol * max(1.0, wNorm) { break }

        // Two-loop recursion (L-BFGS)
        let k = min(it, m)
        // First loop
        var alphaArr = [Float](repeating: 0, count: k)
        var q = gCpu  // copy
        for i in (0..<k).reversed() {
            alphaArr[i] = 0
            for j in 0..<p { alphaArr[i] += sBase[i * p + j] * q[j] }
            alphaArr[i] *= rhoBase[i]
            for j in 0..<p { q[j] -= alphaArr[i] * yBase[i * p + j] }
        }

        // Initial Hessian (gamma * I)
        var gamma: Float = 1.0
        if k > 0 {
            let idx = (it - 1) % m
            var sy: Float = 0, yy: Float = 0
            for j in 0..<p {
                sy += sBase[idx * p + j] * yBase[idx * p + j]
                yy += yBase[idx * p + j] * yBase[idx * p + j]
            }
            if yy > 0 { gamma = sy / yy }
        }

        // d = gamma * q
        for j in 0..<p { d[j] = gamma * q[j] }

        // Second loop
        for i in 0..<k {
            var beta: Float = 0
            for j in 0..<p { beta += yBase[i * p + j] * d[j] }
            beta *= rhoBase[i]
            for j in 0..<p { d[j] += (alphaArr[i] - beta) * sBase[i * p + j] }
        }

        // Negate and store direction (d = -H*g)
        for j in 0..<p { d[j] = -d[j] }

        // Directional derivative: g·d
        var gd: Float = 0
        for j in 0..<p { gd += gCpu[j] * d[j] }
        if gd >= 0 { break }  // not a descent direction

        // Fused line search: encode ALL 20 trials in 1 command buffer
        let c1: Float = 1e-4
        var step = Float(1.0)
        var wTrial = [Float](repeating: 0, count: p)
        var wNorm2Trials = [Float](repeating: 0, count: maxTrials)
        var trialSteps = [Float](repeating: 0, count: maxTrials)
        var nTrials = 0
        let trialFPtr = trialWbuf.contents().assumingMemoryBound(to: Float.self)

        for t in 0..<maxTrials {
            trialSteps[t] = step
            for j in 0..<p {
                let val = wCpu[j] + step * d[j]
                wTrial[j] = val
                trialFPtr[t * p + j] = val
            }
            var wNorm2V: Float = 0
            for j in 0..<p { wNorm2V += wTrial[j] * wTrial[j] }
            wNorm2Trials[t] = wNorm2V
            nTrials = t + 1
            step *= 0.5
        }

        let lsCB = ctx.commandQueue.makeCommandBuffer()!
        for t in 0..<nTrials {
            let copyBlit = lsCB.makeBlitCommandEncoder()!
            copyBlit.copy(from: trialWbuf, sourceOffset: t * pSize,
                          to: wBuffer, destinationOffset: 0,
                          size: pSize)
            copyBlit.endEncoding()

            gemmLinLS.encode(commandBuffer: lsCB, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixLin)

            let encLoss = lsCB.makeComputeCommandEncoder()!
            encLoss.setComputePipelineState(logLossPpl)
            encLoss.setBuffer(linBuffer, offset: 0, index: 0)
            encLoss.setBuffer(yBuffer, offset: 0, index: 1)
            encLoss.setBuffer(lossBuf, offset: 0, index: 2)
            var nU = UInt32(n)
            encLoss.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            encLoss.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                    threadsPerThreadgroup: lsTg256)
            encLoss.endEncoding()

            let encRed = lsCB.makeComputeCommandEncoder()!
            encRed.setComputePipelineState(rsPpl)
            encRed.setBuffer(lossBuf, offset: 0, index: 0)
            encRed.setBuffer(lsConvBuf, offset: t * lsNg * fs, index: 1)
            var nU32 = UInt32(n)
            var ngU32 = UInt32(lsNg)
            encRed.setBytes(&nU32, length: MemoryLayout<UInt32>.stride, index: 2)
            encRed.setBytes(&ngU32, length: MemoryLayout<UInt32>.stride, index: 3)
            encRed.dispatchThreadgroups(lsNgSize, threadsPerThreadgroup: lsTg256)
            encRed.endEncoding()
        }

        lsCB.commit()
        lsCB.waitUntilCompleted()

        let lsConvPtr = lsConvBuf.contents().assumingMemoryBound(to: Float.self)
        var accepted = false
        for t in 0..<nTrials {
            var lossSum: Float = 0
            for j in 0..<lsNg { lossSum += lsConvPtr[t * lsNg + j] }
            let lossTrial = lossSum / Float(n) + Float(0.5) * alphaN * wNorm2Trials[t]
            if lossTrial <= loss + c1 * trialSteps[t] * gd {
                for j in 0..<p { wCpu[j] = trialFPtr[t * p + j] }
                loss = lossTrial
                step = trialSteps[t]
                accepted = true
                break
            }
        }
        if !accepted { break }

        // Store s, y, rho
        let idx = it % m
        let sOff = idx * p
        let yOff = idx * p
        var sy: Float = 0
        for j in 0..<p {
            sBase[sOff + j] = wTrial[j] - wCpu[j]          // s = w_new - w_old
            yBase[yOff + j] = gCpu[j]                       // y = g_old - g_new (will update below)
            sy += sBase[sOff + j] * yBase[yOff + j]
        }

        // Update w = w_trial
        wCpu = wTrial
        for j in 0..<p { wPtr[j] = wCpu[j] }

        // Compute new gradient + loss
        let oldGpu = gCpu  // save old gradient
        loss = computeGradientLoss()

        var yy: Float = 0
        for j in 0..<p {
            let newGj = gPtr[j]
            let oldGj = oldGpu[j]
            yBase[yOff + j] = newGj - oldGj                // y = g_new - g_old
            gCpu[j] = newGj
            yy += yBase[yOff + j] * yBase[yOff + j]
        }

        // Recompute rho = 1/(s·y)
        sy = 0
        for j in 0..<p { sy += sBase[sOff + j] * yBase[yOff + j] }
        rhoBase[idx] = sy > 0 ? 1.0 / sy : 0
    }

    let coefOut = coef_out.assumingMemoryBound(to: Float.self)
    memcpy(coefOut, wPtr, pSize)
    n_iter_out?.pointee = nIter
    return 0
}

// MARK: - L-BFGS fit: multinomial LogisticRegression (robust to collinear features)

@_cdecl("skmetal_multinomial_lbfgs_fit")
public func skmetal_multinomial_lbfgs_fit(
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
    let pSize = p * fs
    let xRowBytes = p * fs
    let nSize = n * fs
    let totalParams = p * n_classes
    let totalSize = totalParams * fs

    let alpha: Float = 1.0 / C
    let alphaN: Float = alpha / Float(n)  // scaled by n for gradient

    guard let xBuffer = wrapInput(X, length: n * xRowBytes, device: ctx.device),
          let yBuffer = wrapInput(y, length: n * fs, device: ctx.device) else {
        return 1
    }

    let kSize = n_classes * fs  // row bytes for n_classes-wide matrices
    let wSize = totalParams * fs

    guard let wBuffer = ctx.device.makeBuffer(length: wSize, options: .storageModeShared),
          let gBuffer = ctx.device.makeBuffer(length: wSize, options: .storageModeShared),
          let scoresBuffer = ctx.device.makeBuffer(length: n * n_classes * fs, options: .storageModeShared),
          let probBuffer = ctx.device.makeBuffer(length: n * n_classes * fs, options: .storageModeShared),
          let resBuffer = ctx.device.makeBuffer(length: n * n_classes * fs, options: .storageModeShared),
          let lossBuf = ctx.device.makeBuffer(length: nSize, options: .storageModeShared),
          let sumBuf = ctx.device.makeBuffer(length: max(1, (n + 255) / 256) * fs, options: .storageModeShared) else {
        return 1
    }
    memset(wBuffer.contents(), 0, wSize)

    let descX = MPSMatrixDescriptor(dimensions: n, columns: p, rowBytes: xRowBytes, dataType: .float32)
    let descW = MPSMatrixDescriptor(dimensions: p, columns: n_classes, rowBytes: kSize, dataType: .float32)
    let descScores = MPSMatrixDescriptor(dimensions: n, columns: n_classes, rowBytes: kSize, dataType: .float32)
    let descRes = MPSMatrixDescriptor(dimensions: n, columns: n_classes, rowBytes: kSize, dataType: .float32)
    let descG = MPSMatrixDescriptor(dimensions: p, columns: n_classes, rowBytes: kSize, dataType: .float32)
    let matrixX = MPSMatrix(buffer: xBuffer, descriptor: descX)
    let matrixW = MPSMatrix(buffer: wBuffer, descriptor: descW)
    let matrixScores = MPSMatrix(buffer: scoresBuffer, descriptor: descScores)
    let matrixProb = MPSMatrix(buffer: probBuffer, descriptor: descScores)
    let matrixRes = MPSMatrix(buffer: resBuffer, descriptor: descRes)
    let matrixGrad = MPSMatrix(buffer: gBuffer, descriptor: descG)

    // L-BFGS memory (m = 10)
    let m = 10
    let mSize = m * totalParams * fs
    guard let sBuf = ctx.device.makeBuffer(length: mSize, options: .storageModeShared),
          let yBuf = ctx.device.makeBuffer(length: mSize, options: .storageModeShared),
          let rhoBuf = ctx.device.makeBuffer(length: m * fs, options: .storageModeShared) else {
        return 1
    }
    let sBase = sBuf.contents().assumingMemoryBound(to: Float.self)
    let yBase = yBuf.contents().assumingMemoryBound(to: Float.self)
    let rhoBase = rhoBuf.contents().assumingMemoryBound(to: Float.self)

    let wPtr = wBuffer.contents().assumingMemoryBound(to: Float.self)
    let gPtr = gBuffer.contents().assumingMemoryBound(to: Float.self)
    let probPtr = probBuffer.contents().assumingMemoryBound(to: Float.self)
    let sumPtr = sumBuf.contents().assumingMemoryBound(to: Float.self)

    let maxIter = Int(max_iter)
    var nIter: Int32 = 0

    var d = [Float](repeating: 0, count: totalParams)
    var wCpu = [Float](repeating: 0, count: totalParams)
    var gCpu = [Float](repeating: 0, count: totalParams)

    // ---- GPU compute closures ----

    func computeLoss() -> Float {
        let ng = max(1, (n + 255) / 256)
        let cb = ctx.commandQueue.makeCommandBuffer()!
        if let rsPpl = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(rsPpl)
            enc.setBuffer(lossBuf, offset: 0, index: 0)
            enc.setBuffer(sumBuf, offset: 0, index: 1)
            var nU = UInt32(n); var ngU = UInt32(ng)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: ng, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        var loss = Float(0)
        for i in 0..<ng { loss += sumPtr[i] }
        loss /= Float(n)
        if alpha != 0 {
            var wNorm2: Float = 0
            for i in 0..<totalParams { wNorm2 += wPtr[i] * wPtr[i] }
            loss += Float(0.5) * alphaN * Float(n) * wNorm2  // α/2 * ||W||²
        }
        return loss
    }

    func computeGradientLoss() -> Float {
        let cb = ctx.commandQueue.makeCommandBuffer()!

        // 1. scores = X @ W (MPS GEMM)
        let gemmScores = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: false, transposeRight: false,
            resultRows: n, resultColumns: n_classes, interiorColumns: p,
            alpha: 1.0, beta: 0.0)
        gemmScores.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixScores)

        // 2. softmax → probabilities
        let softmax = MPSMatrixSoftMax(device: ctx.device)
        softmax.sourceRows = n; softmax.sourceColumns = n_classes
        softmax.encode(commandBuffer: cb, inputMatrix: matrixScores, resultMatrix: matrixProb)

        // 3. residual = prob - y_enc
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

        // 4. gradient = X^T @ residual / n  (p × n_classes)
        let gemmGrad = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: n_classes, interiorColumns: n,
            alpha: 1.0 / Double(n), beta: 0.0)
        gemmGrad.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixRes, resultMatrix: matrixGrad)

        // 5. L2: grad[c][i] += alpha * W[i][c]
        if alpha != 0 {
            if let pipeline = ctx.getPipeline(name: "multinomial_grad_l2", functionName: "multinomial_grad_l2") {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(gBuffer, offset: 0, index: 0)
                enc.setBuffer(wBuffer, offset: 0, index: 1)
                var a = alphaN * Float(n)  // α (unnormalized, matches L2 loss scaling)
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

        // 6. Cross-entropy loss
        if let pipeline = ctx.getPipeline(name: "cross_entropy_loss", functionName: "cross_entropy_loss") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(probBuffer, offset: 0, index: 0)
            enc.setBuffer(yBuffer, offset: 0, index: 1)
            enc.setBuffer(lossBuf, offset: 0, index: 2)
            var nU = UInt32(n); var cU = UInt32(n_classes)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 7. Reduce sum of loss
        let ng = max(1, (n + 255) / 256)
        if let rsPpl = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(rsPpl)
            enc.setBuffer(lossBuf, offset: 0, index: 0)
            enc.setBuffer(sumBuf, offset: 0, index: 1)
            var nU = UInt32(n); var ngU = UInt32(ng)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: ng, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        cb.commit()
        cb.waitUntilCompleted()

        var loss = Float(0)
        for i in 0..<ng { loss += sumPtr[i] }
        loss /= Float(n)
        if alpha != 0 {
            var wNorm2: Float = 0
            for i in 0..<totalParams { wNorm2 += wPtr[i] * wPtr[i] }
            loss += Float(0.5) * alphaN * Float(n) * wNorm2
        }
        return loss
    }

    func computeLossAt(w: [Float]) -> Float {
        for i in 0..<totalParams { wPtr[i] = w[i] }
        let cb = ctx.commandQueue.makeCommandBuffer()!

        let gemmScores = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: false, transposeRight: false,
            resultRows: n, resultColumns: n_classes, interiorColumns: p,
            alpha: 1.0, beta: 0.0)
        gemmScores.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixScores)

        let softmax = MPSMatrixSoftMax(device: ctx.device)
        softmax.sourceRows = n; softmax.sourceColumns = n_classes
        softmax.encode(commandBuffer: cb, inputMatrix: matrixScores, resultMatrix: matrixProb)

        if let pipeline = ctx.getPipeline(name: "cross_entropy_loss", functionName: "cross_entropy_loss") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(probBuffer, offset: 0, index: 0)
            enc.setBuffer(yBuffer, offset: 0, index: 1)
            enc.setBuffer(lossBuf, offset: 0, index: 2)
            var nU = UInt32(n); var cU = UInt32(n_classes)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        let ng = max(1, (n + 255) / 256)
        if let rsPpl = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(rsPpl)
            enc.setBuffer(lossBuf, offset: 0, index: 0)
            enc.setBuffer(sumBuf, offset: 0, index: 1)
            var nU = UInt32(n); var ngU = UInt32(ng)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: ng, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        var loss = Float(0)
        for i in 0..<ng { loss += sumPtr[i] }
        loss /= Float(n)
        if alpha != 0 {
            var wNorm2: Float = 0
            for i in 0..<totalParams { wNorm2 += w[i] * w[i] }
            loss += Float(0.5) * alphaN * Float(n) * wNorm2
        }
        return loss
    }

    // ---- Main L-BFGS loop ----
    var loss = computeGradientLoss()
    memcpy(&wCpu, wPtr, totalSize)
    memcpy(&gCpu, gPtr, totalSize)

    for it in 0..<maxIter {
        nIter = Int32(it + 1)

        var gNorm: Float = 0
        var wNorm2: Float = 0
        for i in 0..<totalParams {
            gNorm = max(gNorm, abs(gCpu[i]))
            wNorm2 += wCpu[i] * wCpu[i]
        }
        let wNorm = sqrt(wNorm2)
        if gNorm < tol * max(1.0, wNorm) { break }

        let k = min(it, m)
        var alphaArr = [Float](repeating: 0, count: k)
        var q = gCpu
        for i in (0..<k).reversed() {
            alphaArr[i] = 0
            for j in 0..<totalParams { alphaArr[i] += sBase[i * totalParams + j] * q[j] }
            alphaArr[i] *= rhoBase[i]
            for j in 0..<totalParams { q[j] -= alphaArr[i] * yBase[i * totalParams + j] }
        }

        var gamma: Float = 1.0
        if k > 0 {
            let idx = (it - 1) % m
            var sy: Float = 0, yy: Float = 0
            for j in 0..<totalParams {
                sy += sBase[idx * totalParams + j] * yBase[idx * totalParams + j]
                yy += yBase[idx * totalParams + j] * yBase[idx * totalParams + j]
            }
            if yy > 0 { gamma = sy / yy }
        }

        for j in 0..<totalParams { d[j] = gamma * q[j] }
        for i in 0..<k {
            var beta: Float = 0
            for j in 0..<totalParams { beta += yBase[i * totalParams + j] * d[j] }
            beta *= rhoBase[i]
            for j in 0..<totalParams { d[j] += (alphaArr[i] - beta) * sBase[i * totalParams + j] }
        }
        for j in 0..<totalParams { d[j] = -d[j] }

        var gd: Float = 0
        for j in 0..<totalParams { gd += gCpu[j] * d[j] }
        if gd >= 0 { break }

        let maxTrialSteps = 20
        var trialSteps = [Float](repeating: 1.0, count: maxTrialSteps)
        var wTrials = [[Float]](repeating: [Float](repeating: 0, count: totalParams), count: maxTrialSteps)
        var step: Float = 1.0
        for t in 0..<maxTrialSteps {
            trialSteps[t] = step
            for j in 0..<totalParams { wTrials[t][j] = wCpu[j] + step * d[j] }
            step *= 0.5
        }

        let trialNg = max(1, (n + 255) / 256)
        let trialBufSize = maxTrialSteps * trialNg * fs
        let trialWSize = maxTrialSteps * totalParams * fs
        guard let trialSumBuf = ctx.reusableBuffer(length: trialBufSize) else { return 1 }
        let trialWBuffer = ctx.device.makeBuffer(length: trialWSize, options: .storageModeShared)
        guard trialWBuffer != nil else { return 1 }
        let trialWBase = trialWBuffer!.contents().assumingMemoryBound(to: Float.self)
        var wNorms2 = [Float](repeating: 0, count: maxTrialSteps)
        for t in 0..<maxTrialSteps {
            var wn2: Float = 0
            for j in 0..<totalParams {
                trialWBase[t * totalParams + j] = wTrials[t][j]
                wn2 += wTrials[t][j] * wTrials[t][j]
            }
            wNorms2[t] = wn2
        }

        let trialCB = ctx.commandQueue.makeCommandBuffer()!
        for t in 0..<maxTrialSteps {
            let blit = trialCB.makeBlitCommandEncoder()!
            blit.copy(from: trialWBuffer!, sourceOffset: t * totalParams * fs,
                      to: wBuffer, destinationOffset: 0, size: totalParams * fs)
            blit.endEncoding()

            let gemm = MPSMatrixMultiplication(
                device: ctx.device, transposeLeft: false, transposeRight: false,
                resultRows: n, resultColumns: n_classes, interiorColumns: p,
                alpha: 1.0, beta: 0.0)
            gemm.encode(commandBuffer: trialCB, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixScores)

            let softmax = MPSMatrixSoftMax(device: ctx.device)
            softmax.sourceRows = n; softmax.sourceColumns = n_classes
            softmax.encode(commandBuffer: trialCB, inputMatrix: matrixScores, resultMatrix: matrixProb)

            if let pipeline = ctx.getPipeline(name: "cross_entropy_loss", functionName: "cross_entropy_loss") {
                let enc = trialCB.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(probBuffer, offset: 0, index: 0)
                enc.setBuffer(yBuffer, offset: 0, index: 1)
                enc.setBuffer(lossBuf, offset: 0, index: 2)
                var nU = UInt32(n); var cU = UInt32(n_classes)
                enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
                enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 4)
                enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }

            if let rsPpl = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum") {
                let enc = trialCB.makeComputeCommandEncoder()!
                enc.setComputePipelineState(rsPpl)
                enc.setBuffer(lossBuf, offset: 0, index: 0)
                enc.setBuffer(trialSumBuf, offset: t * trialNg * fs, index: 1)
                var nU = UInt32(n); var ngU = UInt32(trialNg)
                enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
                enc.setBytes(&ngU, length: MemoryLayout<UInt32>.stride, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: trialNg, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
        }
        trialCB.commit()
        trialCB.waitUntilCompleted()

        let trialSumBase = trialSumBuf.contents().assumingMemoryBound(to: Float.self)
        let c1: Float = 1e-4
        var wTrial = [Float](repeating: 0, count: totalParams)
        var lossTrial: Float = 0
        var accepted = false
        for t in 0..<maxTrialSteps {
            var ceLoss: Float = 0
            for i in 0..<trialNg { ceLoss += trialSumBase[t * trialNg + i] }
            ceLoss /= Float(n)
            lossTrial = ceLoss
            if alpha != 0 {
                lossTrial += Float(0.5) * alphaN * Float(n) * wNorms2[t]
            }
            if lossTrial <= loss + c1 * trialSteps[t] * gd {
                for j in 0..<totalParams { wTrial[j] = wTrials[t][j] }
                accepted = true
                break
            }
        }
        ctx.recycleBuffer(trialSumBuf)
        if !accepted { break }

        let idx = it % m
        let sOff = idx * totalParams
        let yOff = idx * totalParams
        for j in 0..<totalParams {
            sBase[sOff + j] = wTrial[j] - wCpu[j]
            yBase[yOff + j] = gCpu[j]
        }

        wCpu = wTrial
        for j in 0..<totalParams { wPtr[j] = wCpu[j] }
        let oldGpu = gCpu
        loss = computeGradientLoss()

        for j in 0..<totalParams {
            yBase[yOff + j] = gPtr[j] - oldGpu[j]
            gCpu[j] = gPtr[j]
        }

        var sy: Float = 0
        for j in 0..<totalParams { sy += sBase[sOff + j] * yBase[yOff + j] }
        rhoBase[idx] = sy > 0 ? 1.0 / sy : 0
    }

    let coefOut = W_out.assumingMemoryBound(to: Float.self)
    memcpy(coefOut, wPtr, totalSize)
    n_iter_out?.pointee = nIter
    return 0
}



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
    let isize = MemoryLayout<Int32>.stride
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
    let linPtr = linBuffer.contents().assumingMemoryBound(to: Float.self)
    let lossPtr = lossBuf.contents().assumingMemoryBound(to: Float.self)
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

    // Helper: compute just loss for line search (w already on GPU)
    // Returns loss = Σ(log loss)/n + (α/2)·||w||²
    func computeLossAt(w: [Float]) -> Float {
        // Copy w to GPU
        for i in 0..<p { wPtr[i] = w[i] }
        // lin = X @ w
        let cb = ctx.commandQueue.makeCommandBuffer()!
        let gemmLin = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: false, transposeRight: false,
            resultRows: n, resultColumns: 1, interiorColumns: p,
            alpha: 1.0, beta: 0.0)
        gemmLin.encode(commandBuffer: cb, leftMatrix: matrixX, rightMatrix: matrixW, resultMatrix: matrixLin)

        if let pipeline = ctx.getPipeline(name: "log_loss_binary", functionName: "log_loss_binary") {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(linBuffer, offset: 0, index: 0)
            enc.setBuffer(yBuffer, offset: 0, index: 1)
            enc.setBuffer(lossBuf, offset: 0, index: 2)
            var nU = UInt32(n)
            enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
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
            for i in 0..<p { wNorm2 += w[i] * w[i] }
            loss += Float(0.5) * alphaN * wNorm2
        }
        return loss
    }

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

        // Backtracking Armijo line search
        var step: Float = 1.0
        let c1: Float = 1e-4
        var wTrial = [Float](repeating: 0, count: p)
        var lossTrial: Float = 0
        var accepted = false
        for _ in 0..<20 {
            for j in 0..<p { wTrial[j] = wCpu[j] + step * d[j] }
            lossTrial = computeLossAt(w: wTrial)
            if lossTrial <= loss + c1 * step * gd {
                accepted = true
                break
            }
            step *= 0.5
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

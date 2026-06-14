import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - Fused Ridge: center X in-place + X^T X + X^T y in one command buffer

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

    var nU: UInt32 = UInt32(n)
    var pU: UInt32 = UInt32(p)
    let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
    if let pipeline = ctx.getPipeline(name: "column_means", functionName: "column_means") {
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(meanBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 3)
        let blockCols = 8
        let tgCount = (p + blockCols - 1) / blockCols
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(MTLSize(width: tgCount, height: 1, depth: 1),
                                            threadsPerThreadgroup: tgSize)
    }
    computeEncoder.endEncoding()

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

    let gemmXTX = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: p, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmXTX.encode(commandBuffer: commandBuffer, leftMatrix: matrixX, rightMatrix: matrixX, resultMatrix: matrixXTX)

    let gemmXTy = MPSMatrixMultiplication(
        device: ctx.device, transposeLeft: true, transposeRight: false,
        resultRows: p, resultColumns: 1, interiorColumns: n,
        alpha: 1.0, beta: 0.0)
    gemmXTy.encode(commandBuffer: commandBuffer, leftMatrix: matrixX, rightMatrix: matrixY, resultMatrix: matrixXTy)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
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
          let rsPpl = ctx.getPipeline(name: "reduce_sum", functionName: "reduce_sum"),
          let nsPpl = ctx.getPipeline(name: "norm_sq", functionName: "norm_sq") else {
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
        let cb = ctx.commandQueue.makeCommandBuffer()!
        let gemm = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: true, transposeRight: false,
            resultRows: p, resultColumns: p, interiorColumns: n,
            alpha: 1.0, beta: 0.0)
        gemm.encode(commandBuffer: cb, leftMatrix: mX, rightMatrix: mX, resultMatrix: mXTX)
        cb.commit()
        cb.waitUntilCompleted()
    }

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

    let L: Float = {
        if p == 1 {
            return abs(xtxBuf.contents().load(as: Float.self))
        }
        guard let vBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
              let uBuf = ctx.device.makeBuffer(length: pBufSize, options: .storageModeShared),
              let sumBuf = ctx.device.makeBuffer(length: fs, options: .storageModeShared) else {
            return 1
        }
        let vPtr = vBuf.contents().assumingMemoryBound(to: Float.self)
        let xtxPtr = xtxBuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<p { vPtr[i] = xtxPtr[i * p] }

        let mV = MPSMatrix(buffer: vBuf, descriptor: colDesc)
        let mU = MPSMatrix(buffer: uBuf, descriptor: colDesc)

        let powerIters = 20
        let groupSize = 5
        for batchStart in stride(from: 0, to: powerIters, by: groupSize) {
            let batchEnd = min(batchStart + groupSize, powerIters)
            let cb = ctx.commandQueue.makeCommandBuffer()!

            for _ in batchStart..<batchEnd {
                let gemm = ctx.getMPSGemm(transposeLeft: false, transposeRight: false,
                                           resultRows: p, resultColumns: 1, interiorColumns: p,
                                           alpha: 1.0, beta: 0.0)
                gemm.encode(commandBuffer: cb, leftMatrix: mXTX, rightMatrix: mV, resultMatrix: mU)

                let encNS = cb.makeComputeCommandEncoder()!
                encNS.setComputePipelineState(nsPpl)
                encNS.setBuffer(uBuf, offset: 0, index: 0)
                encNS.setBuffer(vBuf, offset: 0, index: 1)
                var nU32 = UInt32(p)
                encNS.setBytes(&nU32, length: MemoryLayout<UInt32>.stride, index: 2)
                encNS.dispatchThreadgroups(MTLSize(width: (p + 255) / 256, height: 1, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encNS.endEncoding()

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
            }

            cb.commit()
            cb.waitUntilCompleted()

            let uPtr = uBuf.contents().assumingMemoryBound(to: Float.self)
            let sumPtr = sumBuf.contents().assumingMemoryBound(to: Float.self)
            for _ in batchStart..<batchEnd {
                let norm = sqrt(sumPtr[0])
                if norm < 1e-10 { break }
                for i in 0..<p { vPtr[i] = uPtr[i] / norm }
            }
        }

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

    let batchSize = 5
    let checkEvery = 25
    let tg256 = MTLSize(width: 256, height: 1, depth: 1)
    let grd256 = MTLSize(width: (p + 255) / 256, height: 1, depth: 1)
    var t: Float = 1.0
    var it: Int32 = 0

    var itCount = 0
    while itCount < Int(max_iter) {
        let batchEnd = min(itCount + batchSize, Int(max_iter))
        let cb = ctx.commandQueue.makeCommandBuffer()!

        for batchIt in itCount..<batchEnd {
            it = Int32(batchIt + 1)

            let blit1 = cb.makeBlitCommandEncoder()!
            blit1.copy(from: xBuf_g, sourceOffset: 0, to: xPrevBuf, destinationOffset: 0, size: pBufSize)
            blit1.endEncoding()

            let gemm = MPSMatrixMultiplication(
                device: ctx.device, transposeLeft: false, transposeRight: false,
                resultRows: p, resultColumns: 1, interiorColumns: p,
                alpha: 1.0, beta: 0.0)
            gemm.encode(commandBuffer: cb, leftMatrix: mXTX, rightMatrix: mZ, resultMatrix: mGrad)

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

            let blit2 = cb.makeBlitCommandEncoder()!
            blit2.copy(from: zBuf, sourceOffset: 0, to: xTempBuf, destinationOffset: 0, size: pBufSize)
            blit2.endEncoding()

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

            let tPrev = t
            t = (1.0 + sqrt(1.0 + 4.0 * tPrev * tPrev)) / 2.0
            let factor = (tPrev - 1.0) / t

            let encSub = cb.makeComputeCommandEncoder()!
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
        }

        cb.commit()
        cb.waitUntilCompleted()

        itCount = batchEnd
        if batchEnd % checkEvery == 0 || batchEnd >= Int(max_iter) - 1 {
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

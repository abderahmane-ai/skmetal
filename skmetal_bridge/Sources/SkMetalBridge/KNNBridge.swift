import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - KNN vote classification

@_cdecl("skmetal_knn_vote_classify")
public func skmetal_knn_vote_classify(
    indices: UnsafeRawPointer,
    trainLabels: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    N: Int,
    k: Int,
    nTrain: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let idxSize = N * k * MemoryLayout<Int32>.stride
    let predSize = N * MemoryLayout<Float>.stride
    let labelSize = nTrain * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "knn_vote_classify", functionName: "knn_vote_classify"),
          let idxBuffer = wrapInput(indices, length: idxSize, device: ctx.device),
          let tlBuffer = wrapInput(trainLabels, length: labelSize, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    guard let commandBuffer = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return 1 }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(idxBuffer, offset: 0, index: 0)
    encoder.setBuffer(tlBuffer, offset: 0, index: 1)
    encoder.setBuffer(predBuffer, offset: 0, index: 2)
    var nU = UInt32(N); var kU = UInt32(k)
    encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 4)

    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(MTLSize(width: (N + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KNN vote regression

@_cdecl("skmetal_knn_vote_regress")
public func skmetal_knn_vote_regress(
    indices: UnsafeRawPointer,
    trainTargets: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    N: Int,
    k: Int,
    nTrain: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let idxSize = N * k * MemoryLayout<Int32>.stride
    let predSize = N * MemoryLayout<Float>.stride
    let trainSize = nTrain * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "knn_vote_regress", functionName: "knn_vote_regress"),
          let idxBuffer = wrapInput(indices, length: idxSize, device: ctx.device),
          let ttBuffer = wrapInput(trainTargets, length: trainSize, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    guard let commandBuffer = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return 1 }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(idxBuffer, offset: 0, index: 0)
    encoder.setBuffer(ttBuffer, offset: 0, index: 1)
    encoder.setBuffer(predBuffer, offset: 0, index: 2)
    var nU = UInt32(N); var kU = UInt32(k)
    encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 4)

    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(MTLSize(width: (N + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - KNN weighted vote classification

@_cdecl("skmetal_knn_vote_classify_weighted")
public func skmetal_knn_vote_classify_weighted(
    indices: UnsafeRawPointer,
    distances: UnsafeRawPointer,
    trainLabels: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    N: Int,
    k: Int,
    nTrain: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let idxSize = N * k * MemoryLayout<Int32>.stride
    let distSize = N * k * MemoryLayout<Float>.stride
    let predSize = N * MemoryLayout<Float>.stride
    let trainSize = nTrain * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "knn_vote_classify_weighted",
                                          functionName: "knn_vote_classify_weighted"),
          let idxBuffer = wrapInput(indices, length: idxSize, device: ctx.device),
          let distBuffer = wrapInput(distances, length: distSize, device: ctx.device),
          let tlBuffer = wrapInput(trainLabels, length: trainSize, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(idxBuffer, offset: 0, index: 0)
    enc.setBuffer(distBuffer, offset: 0, index: 1)
    enc.setBuffer(tlBuffer, offset: 0, index: 2)
    enc.setBuffer(predBuffer, offset: 0, index: 3)
    var nU = UInt32(N); var kU = UInt32(k)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(MTLSize(width: (N + 255) / 256, height: 1, depth: 1),
                              threadsPerThreadgroup: tg)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - KNN weighted vote regression

@_cdecl("skmetal_knn_vote_regress_weighted")
public func skmetal_knn_vote_regress_weighted(
    indices: UnsafeRawPointer,
    distances: UnsafeRawPointer,
    trainTargets: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    N: Int,
    k: Int,
    nTrain: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let idxSize = N * k * MemoryLayout<Int32>.stride
    let distSize = N * k * MemoryLayout<Float>.stride
    let predSize = N * MemoryLayout<Float>.stride
    let trainSize = nTrain * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "knn_vote_regress_weighted",
                                          functionName: "knn_vote_regress_weighted"),
          let idxBuffer = wrapInput(indices, length: idxSize, device: ctx.device),
          let distBuffer = wrapInput(distances, length: distSize, device: ctx.device),
          let ttBuffer = wrapInput(trainTargets, length: trainSize, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }
    guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(idxBuffer, offset: 0, index: 0)
    enc.setBuffer(distBuffer, offset: 0, index: 1)
    enc.setBuffer(ttBuffer, offset: 0, index: 2)
    enc.setBuffer(predBuffer, offset: 0, index: 3)
    var nU = UInt32(N); var kU = UInt32(k)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 5)
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(MTLSize(width: (N + 255) / 256, height: 1, depth: 1),
                              threadsPerThreadgroup: tg)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Tiled KNN: process training data in tiles to avoid materializing full N×M distance matrix

@_cdecl("skmetal_knn_tiled_kneighbors")
public func skmetal_knn_tiled_kneighbors(
    XQuery: UnsafeRawPointer,
    XTrain: UnsafeRawPointer,
    outIndices: UnsafeMutableRawPointer,
    outValues: UnsafeMutableRawPointer,
    nQ: Int,
    nT: Int,
    d: Int,
    k: Int,
    tileSize: Int,
    metric: Int32
) -> Int32 {
    let ctx = MetalContext.shared
    guard k > 0 else { return 1 }
    let fs = MemoryLayout<Float>.stride
    let isize = MemoryLayout<Int32>.stride
    let querySize = nQ * d * fs
    let trainSize = nT * d * fs
    let kSize = nQ * k * fs
    let kIdxSize = nQ * k * isize

    guard let xQueryBuffer = wrapInput(XQuery, length: querySize, device: ctx.device),
          let xTrainBuffer = wrapInput(XTrain, length: trainSize, device: ctx.device),
          let outValsBuffer = wrapOutput(outValues, length: kSize, device: ctx.device),
          let outIdxsBuffer = wrapOutput(outIndices, length: kIdxSize, device: ctx.device) else {
        return 1
    }

    let isCosine = metric == 2
    let isManhattan = metric == 1

    let rqSize = nQ * fs
    let rtSize = nT * fs
    let rqBuffer = (!isManhattan) ? ctx.device.makeBuffer(length: rqSize, options: .storageModeShared) : nil
    let rtBuffer = (!isManhattan) ? ctx.device.makeBuffer(length: rtSize, options: .storageModeShared) : nil

    guard let cb = ctx.commandQueue.makeCommandBuffer() else { return 1 }

    if !isManhattan, let normPpl = ctx.getPipeline(name: "row_norm_sq", functionName: "row_norm_sq"),
       let rq = rqBuffer, let rt = rtBuffer {
        guard let enc = cb.makeComputeCommandEncoder() else { return 1 }
        enc.setComputePipelineState(normPpl)
        enc.setBuffer(xQueryBuffer, offset: 0, index: 0)
        enc.setBuffer(rq, offset: 0, index: 1)
        var nqU = UInt32(nQ); var dU = UInt32(d)
        enc.setBytes(&nqU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 3)
        let tgNormQ = MTLSize(width: 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(MTLSize(width: (nQ + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: tgNormQ)
        enc.endEncoding()

        guard let enc2 = cb.makeComputeCommandEncoder() else { return 1 }
        enc2.setComputePipelineState(normPpl)
        enc2.setBuffer(xTrainBuffer, offset: 0, index: 0)
        enc2.setBuffer(rt, offset: 0, index: 1)
        var ntU = UInt32(nT)
        enc2.setBytes(&ntU, length: MemoryLayout<UInt32>.stride, index: 2)
        enc2.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 3)
        let tgNormT = MTLSize(width: 256, height: 1, depth: 1)
        enc2.dispatchThreadgroups(MTLSize(width: (nT + 255) / 256, height: 1, depth: 1),
                                  threadsPerThreadgroup: tgNormT)
        enc2.endEncoding()
    }

    let maxTileN = min(tileSize, nT)
    let dotSize = nQ * maxTileN * fs
    let tileValsSize = nQ * k * fs
    let tileIdxsSize = nQ * k * isize

    let dotBuffer = (!isManhattan) ? ctx.device.makeBuffer(length: dotSize, options: .storageModeShared) : nil

    let selectName: String
    let selectFunc: String
    if isManhattan {
        selectName = "knn_select_tile_topk_manhattan"
        selectFunc = "knn_select_tile_topk_manhattan"
    } else {
        selectName = "knn_select_tile_topk"
        selectFunc = "knn_select_tile_topk"
    }
    guard let selectPipeline = ctx.getPipeline(name: selectName, functionName: selectFunc),
          let mergePipeline = ctx.getPipeline(name: "knn_merge_topk", functionName: "knn_merge_topk") else {
        return 1
    }

    let descQ = MPSMatrixDescriptor(dimensions: nQ, columns: d,
                                     rowBytes: d * fs, dataType: .float32)
    let matrixQ = MPSMatrix(buffer: xQueryBuffer, descriptor: descQ)

    let useMPSFindTopK = (k <= 16 && !isManhattan)  // MPSMatrixFindTopK limited to k ≤ 16
    let negDistBuffer = useMPSFindTopK ? ctx.device.makeBuffer(length: dotSize, options: .storageModeShared) : nil

    let numTiles = (nT + tileSize - 1) / tileSize

    // For MPS path: one big buffer per tile-indexed slot (no overwrite)
    let allTValsBuffer = useMPSFindTopK ? ctx.device.makeBuffer(length: numTiles * tileValsSize, options: .storageModeShared) : nil
    let allTIdxsBuffer = useMPSFindTopK ? ctx.device.makeBuffer(length: numTiles * tileIdxsSize, options: .storageModeShared) : nil

    // For old path: per-tile + global + temp
    guard let tValsBuffer = ctx.device.makeBuffer(length: tileValsSize, options: .storageModeShared),
          let tIdxsBuffer = ctx.device.makeBuffer(length: tileIdxsSize, options: .storageModeShared),
          let gValsBuffer = ctx.device.makeBuffer(length: kSize, options: .storageModeShared),
          let gIdxsBuffer = ctx.device.makeBuffer(length: kIdxSize, options: .storageModeShared) else {
        return 1
    }
    let gValsPtr = gValsBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(nQ * k) { gValsPtr[i] = .infinity }
    memset(gIdxsBuffer.contents(), 0, kIdxSize)

    let tempValsBuffer = ctx.device.makeBuffer(length: tileValsSize, options: .storageModeShared)
    let tempIdxsBuffer = ctx.device.makeBuffer(length: tileIdxsSize, options: .storageModeShared)

    let batchSize = min(256, max(1, nQ))
    let tgBatch = MTLSize(width: batchSize, height: 1, depth: 1)
    let gridBatch = MTLSize(width: (nQ + batchSize - 1) / batchSize, height: 1, depth: 1)

    var tileStart = 0
    var tileIdx = 0
    while tileStart < nT {
        let tileEnd = min(tileStart + tileSize, nT)
        let tileN = tileEnd - tileStart

        if isManhattan {
            guard let encSel = cb.makeComputeCommandEncoder() else { return 1 }
            encSel.setComputePipelineState(selectPipeline)
            encSel.setBuffer(xQueryBuffer, offset: 0, index: 0)
            encSel.setBuffer(xTrainBuffer, offset: tileStart * d * fs, index: 1)
            encSel.setBuffer(tValsBuffer, offset: 0, index: 2)
            encSel.setBuffer(tIdxsBuffer, offset: 0, index: 3)
            var nqU = UInt32(nQ); var tnU = UInt32(tileN); var dU = UInt32(d); var kU = UInt32(k)
            encSel.setBytes(&nqU, length: MemoryLayout<UInt32>.stride, index: 4)
            encSel.setBytes(&tnU, length: MemoryLayout<UInt32>.stride, index: 5)
            encSel.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 6)
            encSel.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
            encSel.dispatchThreadgroups(gridBatch, threadsPerThreadgroup: tgBatch)
            encSel.endEncoding()
        } else if useMPSFindTopK {
            guard let rq = rqBuffer, let rt = rtBuffer, let db = dotBuffer,
                  let ndb = negDistBuffer,
                  let negPipeline = ctx.getPipeline(name: "knn_negate_distances",
                                                    functionName: "knn_negate_distances") else { return 1 }

            let trainSlicePtr = xTrainBuffer.contents().advanced(by: tileStart * d * fs)
            guard let trainSliceBuffer = ctx.device.makeBuffer(
                bytesNoCopy: trainSlicePtr,
                length: tileN * d * fs,
                options: .storageModeShared,
                deallocator: nil) else { return 1 }
            let rtSlicePtr = rt.contents().advanced(by: tileStart * fs)
            guard let rtSliceBuffer = ctx.device.makeBuffer(
                bytesNoCopy: rtSlicePtr,
                length: tileN * fs,
                options: .storageModeShared,
                deallocator: nil) else { return 1 }

            let descTSlice = MPSMatrixDescriptor(dimensions: tileN, columns: d,
                                                 rowBytes: d * fs, dataType: .float32)
            let descDot = MPSMatrixDescriptor(dimensions: nQ, columns: tileN,
                                              rowBytes: tileN * fs, dataType: .float32)
            let matrixTSlice = MPSMatrix(buffer: trainSliceBuffer, descriptor: descTSlice)
            let matrixDot = MPSMatrix(buffer: db, descriptor: descDot)

            // 1. GEMM: dot = XQ @ XT[tile].T
            let gemm = MPSMatrixMultiplication(
                device: ctx.device, transposeLeft: false, transposeRight: true,
                resultRows: nQ, resultColumns: tileN, interiorColumns: d,
                alpha: 1.0, beta: 0.0)
            gemm.encode(commandBuffer: cb, leftMatrix: matrixQ, rightMatrix: matrixTSlice,
                        resultMatrix: matrixDot)

            // 2. Negate distances into separate buffer (largest value = closest)
            guard let encNeg = cb.makeComputeCommandEncoder() else { return 1 }
            encNeg.setComputePipelineState(negPipeline)
            encNeg.setBuffer(db, offset: 0, index: 0)
            encNeg.setBuffer(rq, offset: 0, index: 1)
            encNeg.setBuffer(rtSliceBuffer, offset: 0, index: 2)
            encNeg.setBuffer(ndb, offset: 0, index: 3)
            var nqU = UInt32(nQ); var tnU = UInt32(tileN); var csU = UInt32(isCosine ? 1 : 0)
            encNeg.setBytes(&nqU, length: MemoryLayout<UInt32>.stride, index: 4)
            encNeg.setBytes(&tnU, length: MemoryLayout<UInt32>.stride, index: 5)
            encNeg.setBytes(&csU, length: MemoryLayout<UInt32>.stride, index: 6)
            encNeg.dispatchThreads(MTLSize(width: nQ * tileN, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encNeg.endEncoding()

            // 3. MPSMatrixFindTopK per tile (finds k largest values per row)
            guard let atvb = allTValsBuffer, let atib = allTIdxsBuffer else { return 1 }
            let descND = MPSMatrixDescriptor(dimensions: nQ, columns: tileN,
                                             rowBytes: tileN * fs, dataType: .float32)
            let descTV = MPSMatrixDescriptor(dimensions: nQ, columns: k,
                                             rowBytes: k * fs, dataType: .float32)
            let descTI = MPSMatrixDescriptor(dimensions: nQ, columns: k,
                                             rowBytes: k * MemoryLayout<UInt32>.stride, dataType: .uInt32)
            let matrixND = MPSMatrix(buffer: ndb, descriptor: descND)
            let matrixTV = MPSMatrix(buffer: atvb, offset: tileIdx * tileValsSize, descriptor: descTV)
            let matrixTI = MPSMatrix(buffer: atib, offset: tileIdx * tileIdxsSize, descriptor: descTI)

            let findTopK = MPSMatrixFindTopK(device: ctx.device, numberOfTopKValues: k)
            findTopK.indexOffset = tileStart
            findTopK.encode(commandBuffer: cb, inputMatrix: matrixND,
                            resultIndexMatrix: matrixTI, resultValueMatrix: matrixTV)
        } else {
            guard let rq = rqBuffer, let rt = rtBuffer, let db = dotBuffer else { return 1 }

            let trainSlicePtr = xTrainBuffer.contents().advanced(by: tileStart * d * fs)
            guard let trainSliceBuffer = ctx.device.makeBuffer(
                bytesNoCopy: trainSlicePtr,
                length: tileN * d * fs,
                options: .storageModeShared,
                deallocator: nil) else { return 1 }
            let rtSlicePtr = rt.contents().advanced(by: tileStart * fs)
            guard let rtSliceBuffer = ctx.device.makeBuffer(
                bytesNoCopy: rtSlicePtr,
                length: tileN * fs,
                options: .storageModeShared,
                deallocator: nil) else { return 1 }

            let descTSlice = MPSMatrixDescriptor(dimensions: tileN, columns: d,
                                                 rowBytes: d * fs, dataType: .float32)
            let descDot = MPSMatrixDescriptor(dimensions: nQ, columns: tileN,
                                              rowBytes: tileN * fs, dataType: .float32)
            let matrixTSlice = MPSMatrix(buffer: trainSliceBuffer, descriptor: descTSlice)
            let matrixDot = MPSMatrix(buffer: db, descriptor: descDot)

            // 1. GEMM: dot = XQ @ XT[tile].T
            let gemm = MPSMatrixMultiplication(
                device: ctx.device, transposeLeft: false, transposeRight: true,
                resultRows: nQ, resultColumns: tileN, interiorColumns: d,
                alpha: 1.0, beta: 0.0)
            gemm.encode(commandBuffer: cb, leftMatrix: matrixQ, rightMatrix: matrixTSlice,
                        resultMatrix: matrixDot)

            // 2. Fused distance + insertion sort top-k (existing custom kernel)
            guard let encSel = cb.makeComputeCommandEncoder() else { return 1 }
            encSel.setComputePipelineState(selectPipeline)
            encSel.setBuffer(db, offset: 0, index: 0)
            encSel.setBuffer(rq, offset: 0, index: 1)
            encSel.setBuffer(rtSliceBuffer, offset: 0, index: 2)
            encSel.setBuffer(tValsBuffer, offset: 0, index: 3)
            encSel.setBuffer(tIdxsBuffer, offset: 0, index: 4)
            var nqU = UInt32(nQ); var tnU = UInt32(tileN); var kU = UInt32(k)
            encSel.setBytes(&nqU, length: MemoryLayout<UInt32>.stride, index: 5)
            encSel.setBytes(&tnU, length: MemoryLayout<UInt32>.stride, index: 6)
            encSel.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
            encSel.dispatchThreadgroups(gridBatch, threadsPerThreadgroup: tgBatch)
            encSel.endEncoding()
        }

        if !useMPSFindTopK {
            // GPU merge of tile results into global
            guard let encMerge = cb.makeComputeCommandEncoder() else { return 1 }
            encMerge.setComputePipelineState(mergePipeline)
            encMerge.setBuffer(tValsBuffer, offset: 0, index: 0)
            encMerge.setBuffer(tIdxsBuffer, offset: 0, index: 1)
            encMerge.setBuffer(gValsBuffer, offset: 0, index: 2)
            encMerge.setBuffer(gIdxsBuffer, offset: 0, index: 3)
            if let tvb = tempValsBuffer { encMerge.setBuffer(tvb, offset: 0, index: 4) }
            if let tib = tempIdxsBuffer { encMerge.setBuffer(tib, offset: 0, index: 5) }
            var nqU = UInt32(nQ); var kU = UInt32(k)
            encMerge.setBytes(&nqU, length: MemoryLayout<UInt32>.stride, index: 6)
            encMerge.setBytes(&kU, length: MemoryLayout<UInt32>.stride, index: 7)
            var tsU = UInt32(tileStart)
            encMerge.setBytes(&tsU, length: MemoryLayout<UInt32>.stride, index: 8)
            encMerge.dispatchThreadgroups(gridBatch, threadsPerThreadgroup: tgBatch)
            encMerge.endEncoding()
        }

        tileStart += tileSize
        tileIdx += 1
    }

    cb.commit()
    cb.waitUntilCompleted()

    if useMPSFindTopK {
        // CPU merge of tile results (MPSMatrixFindTopK can't accumulate across tiles)
        guard let atvb = allTValsBuffer, let atib = allTIdxsBuffer else { return 1 }
        let outValsPtr = outValsBuffer.contents().assumingMemoryBound(to: Float.self)
        let outIdxsPtr = outIdxsBuffer.contents().assumingMemoryBound(to: Int32.self)
        let tileValsPtr = atvb.contents().assumingMemoryBound(to: Float.self)
        let tileIdxsPtr = atib.contents().assumingMemoryBound(to: UInt32.self)

        for row in 0..<nQ {
            var gv = [Float](repeating: .infinity, count: k)
            var gi = [Int32](repeating: 0, count: k)

            for t in 0..<numTiles {
                let tv = tileValsPtr + t * nQ * k + row * k
                let ti = tileIdxsPtr + t * nQ * k + row * k

                var mv = [Float](repeating: 0, count: k)
                var mi = [Int32](repeating: 0, count: k)
                var a = 0; var b = 0
                for s in 0..<k {
                    if a < k && (b >= k || tv[a] < gv[b]) {
                        mv[s] = tv[a]; mi[s] = Int32(bitPattern: ti[a]); a += 1
                    } else {
                        mv[s] = gv[b]; mi[s] = gi[b]; b += 1
                    }
                }
                gv = mv; gi = mi
            }

            for s in 0..<k {
                if isCosine {
                    outValsPtr[row * k + s] = 1.0 - gv[s]
                } else {
                    outValsPtr[row * k + s] = -gv[s]
                }
                outIdxsPtr[row * k + s] = gi[s]
            }
        }
    } else {
        // GPU path already merged into gValsBuffer/gIdxsBuffer → copy to output
        memcpy(outValsBuffer.contents(), gValsBuffer.contents(), kSize)
        memcpy(outIdxsBuffer.contents(), gIdxsBuffer.contents(), kIdxSize)
    }
    return 0
}

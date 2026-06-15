import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - Shiloach-Vishkin: init parent array

@_cdecl("skmetal_sv_init")
public func skmetal_sv_init(
    parent: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Int32>.stride
    guard let pipeline = ctx.getPipeline(name: "sv_init", functionName: "sv_init"),
          let parentBuffer = wrapOutput(parent, length: byteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(parentBuffer, offset: 0, index: 0)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 1)
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let tgCount = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Shiloach-Vishkin: hook phase

@_cdecl("skmetal_sv_hook")
public func skmetal_sv_hook(
    edges: UnsafeRawPointer,
    parent: UnsafeMutableRawPointer,
    edgeCount: Int,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let edgeByteSize = edgeCount * 2 * MemoryLayout<Int32>.stride
    let parentByteSize = n * MemoryLayout<Int32>.stride
    guard let pipeline = ctx.getPipeline(name: "sv_hook", functionName: "sv_hook"),
          let edgesBuffer = wrapInput(edges, length: edgeByteSize, device: ctx.device),
          let parentBuffer = wrapOutput(parent, length: parentByteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(edgesBuffer, offset: 0, index: 0)
    enc.setBuffer(parentBuffer, offset: 0, index: 1)
    var ecU = UInt32(edgeCount)
    enc.setBytes(&ecU, length: MemoryLayout<UInt32>.stride, index: 2)
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let tgCount = MTLSize(width: (Int(edgeCount) + 255) / 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Shiloach-Vishkin: shortcut phase

@_cdecl("skmetal_sv_shortcut")
public func skmetal_sv_shortcut(
    parent: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Int32>.stride
    guard let pipeline = ctx.getPipeline(name: "sv_shortcut", functionName: "sv_shortcut"),
          let parentBuffer = wrapOutput(parent, length: byteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(parentBuffer, offset: 0, index: 0)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 1)
    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    let tgCount = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Tree Predict (GPU)

@_cdecl("skmetal_tree_predict_all")
public func skmetal_tree_predict_all(
    X: UnsafeRawPointer,
    allTreeValues: UnsafeRawPointer,
    allTreeFeature: UnsafeRawPointer,
    allTreeThreshold: UnsafeRawPointer,
    allTreeLeft: UnsafeRawPointer,
    allTreeRight: UnsafeRawPointer,
    allTreeIsLeaf: UnsafeRawPointer,
    treeOffsets: UnsafeRawPointer,
    treeNNodes: UnsafeRawPointer,
    baseline: UnsafeRawPointer,
    predictions: UnsafeMutableRawPointer,
    n: Int,
    nFeatures: Int,
    nTrees: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let totalNodesPtr = treeNNodes.assumingMemoryBound(to: UInt32.self)
    var totalNodes: UInt32 = 0
    for i in 0..<nTrees { totalNodes += totalNodesPtr[i] }
    let tn = Int(totalNodes)

    let xSize = n * nFeatures * MemoryLayout<Float>.stride
    let arrSize = tn * MemoryLayout<Float>.stride
    let intSize = tn * MemoryLayout<Int32>.stride
    let offSize = nTrees * MemoryLayout<UInt32>.stride
    let predSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "tree_predict_all", functionName: "tree_predict_all"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let tvBuffer = wrapInput(allTreeValues, length: arrSize, device: ctx.device),
          let tfBuffer = wrapInput(allTreeFeature, length: intSize, device: ctx.device),
          let ttBuffer = wrapInput(allTreeThreshold, length: arrSize, device: ctx.device),
          let tlBuffer = wrapInput(allTreeLeft, length: intSize, device: ctx.device),
          let trBuffer = wrapInput(allTreeRight, length: intSize, device: ctx.device),
          let tleafBuffer = wrapInput(allTreeIsLeaf, length: tn * MemoryLayout<UInt8>.stride, device: ctx.device),
          let offBuffer = wrapInput(treeOffsets, length: offSize, device: ctx.device),
          let nnBuffer = wrapInput(treeNNodes, length: offSize, device: ctx.device),
          let blBuffer = wrapInput(baseline, length: MemoryLayout<Float>.stride, device: ctx.device),
          let predBuffer = wrapOutput(predictions, length: predSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(xBuffer, offset: 0, index: 0)
    enc.setBuffer(tvBuffer, offset: 0, index: 1)
    enc.setBuffer(tfBuffer, offset: 0, index: 2)
    enc.setBuffer(ttBuffer, offset: 0, index: 3)
    enc.setBuffer(tlBuffer, offset: 0, index: 4)
    enc.setBuffer(trBuffer, offset: 0, index: 5)
    enc.setBuffer(tleafBuffer, offset: 0, index: 6)
    enc.setBuffer(offBuffer, offset: 0, index: 7)
    enc.setBuffer(nnBuffer, offset: 0, index: 8)
    enc.setBuffer(blBuffer, offset: 0, index: 9)
    enc.setBuffer(predBuffer, offset: 0, index: 10)
    var nU = UInt32(n); var nfU = UInt32(nFeatures); var ntU = UInt32(nTrees)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 11)
    enc.setBytes(&nfU, length: MemoryLayout<UInt32>.stride, index: 12)
    enc.setBytes(&ntU, length: MemoryLayout<UInt32>.stride, index: 13)
    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}



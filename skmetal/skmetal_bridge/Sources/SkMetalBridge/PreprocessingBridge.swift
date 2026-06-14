import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - StandardScaler: per-column mean and variance in one dispatch

@_cdecl("skmetal_scaler_fit")
public func skmetal_scaler_fit(
    X: UnsafeRawPointer,
    meanOut: UnsafeMutableRawPointer,
    varOut: UnsafeMutableRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let statSize = d * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "scaler_fit", functionName: "scaler_fit"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let meanBuffer = wrapOutput(meanOut, length: statSize, device: ctx.device),
          let varBuffer = wrapOutput(varOut, length: statSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(meanBuffer, offset: 0, index: 1)
    encoder.setBuffer(varBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)

    let blockCols = 8
    let tgCount = (d + blockCols - 1) / blockCols

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: tgCount, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - MinMaxScaler: per-column min and max in one dispatch

@_cdecl("skmetal_column_minmax")
public func skmetal_column_minmax(
    X: UnsafeRawPointer,
    minOut: UnsafeMutableRawPointer,
    maxOut: UnsafeMutableRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let xSize = n * d * MemoryLayout<Float>.stride
    let statSize = d * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "column_minmax", functionName: "column_minmax"),
          let xBuffer = wrapInput(X, length: xSize, device: ctx.device),
          let minBuffer = wrapOutput(minOut, length: statSize, device: ctx.device),
          let maxBuffer = wrapOutput(maxOut, length: statSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(minBuffer, offset: 0, index: 1)
    encoder.setBuffer(maxBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)

    let blockColsMM = 8
    let tgCountMM = (d + blockColsMM - 1) / blockColsMM

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: tgCountMM, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - Column transform (RobustScaler / StandardScaler)

@_cdecl("skmetal_column_transform")
public func skmetal_column_transform(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    center: UnsafeRawPointer,
    scale: UnsafeRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * d * MemoryLayout<Float>.stride
    let statSize = d * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "column_transform", functionName: "column_transform"),
          let inBuffer = wrapInput(input, length: matSize, device: ctx.device),
          let outBuffer = wrapOutput(output, length: matSize, device: ctx.device),
          let centerBuffer = wrapInput(center, length: statSize, device: ctx.device),
          let scaleBuffer = wrapInput(scale, length: statSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inBuffer, offset: 0, index: 0)
    encoder.setBuffer(outBuffer, offset: 0, index: 1)
    encoder.setBuffer(centerBuffer, offset: 0, index: 2)
    encoder.setBuffer(scaleBuffer, offset: 0, index: 3)
    var nU = UInt32(n); var dU = UInt32(d)
    encoder.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&dU, length: MemoryLayout<UInt32>.stride, index: 5)

    let tgSize = MTLSize(width: 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(MTLSize(width: (n * d + 255) / 256, height: 1, depth: 1), threadsPerThreadgroup: tgSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

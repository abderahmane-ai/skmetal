import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - Element-Wise Ops (sigmoid, subtract, axpy, norm_sq, scale, negate, fill)

@_cdecl("skmetal_sigmoid")
public func skmetal_sigmoid(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "sigmoid", functionName: "sigmoid"),
          let inputBuffer = wrapInput(input, length: byteSize, device: ctx.device),
          let outputBuffer = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_subtract")
public func skmetal_subtract(
    a: UnsafeRawPointer,
    b: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "subtract", functionName: "subtract"),
          let aBuffer = wrapInput(a, length: byteSize, device: ctx.device),
          let bBuffer = wrapInput(b, length: byteSize, device: ctx.device),
          let outputBuffer = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(outputBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_axpy")
public func skmetal_axpy(
    a: UnsafeMutableRawPointer,
    b: UnsafeRawPointer,
    alpha: Float,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "axpy", functionName: "axpy"),
          let aBuffer = wrapOutput(a, length: byteSize, device: ctx.device),
          let bBuffer = wrapInput(b, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    var alphaF = alpha
    encoder.setBytes(&alphaF, length: MemoryLayout<Float>.stride, index: 2)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_norm_sq")
public func skmetal_norm_sq(
    input: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "norm_sq", functionName: "norm_sq"),
          let inputBuffer = wrapInput(input, length: byteSize, device: ctx.device),
          let outputBuffer = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_add_scalar")
public func skmetal_add_scalar(
    array: UnsafeMutableRawPointer,
    scalar: Float,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "add_scalar", functionName: "add_scalar"),
          let arrayBuffer = wrapOutput(array, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(arrayBuffer, offset: 0, index: 0)
    var scalarV = scalar
    encoder.setBytes(&scalarV, length: MemoryLayout<Float>.stride, index: 1)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_negate")
public func skmetal_negate(
    a: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride
    guard let pipeline = ctx.getPipeline(name: "negate", functionName: "negate"),
          let aBuffer = wrapInput(a, length: byteSize, device: ctx.device),
          let outBuffer = wrapOutput(output, length: byteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(aBuffer, offset: 0, index: 0)
    enc.setBuffer(outBuffer, offset: 0, index: 1)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_fill_f32")
public func skmetal_fill_f32(
    array: UnsafeMutableRawPointer,
    value: Float,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride
    guard let pipeline = ctx.getPipeline(name: "fill_f32", functionName: "fill_f32"),
          let arrBuffer = wrapOutput(array, length: byteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(arrBuffer, offset: 0, index: 0)
    var val = value
    enc.setBytes(&val, length: MemoryLayout<Float>.stride, index: 1)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_scale_f32")
public func skmetal_scale_f32(
    a: UnsafeMutableRawPointer,
    s: Float,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "scale_f32", functionName: "scale_f32"),
          let aBuffer = wrapOutput(a, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    var sV = s
    encoder.setBytes(&sV, length: MemoryLayout<Float>.stride, index: 1)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

// MARK: - IRLS helper functions

@_cdecl("skmetal_irls_weight")
public func skmetal_irls_weight(
    p: UnsafeRawPointer,
    weights: UnsafeMutableRawPointer,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "irls_weight", functionName: "irls_weight"),
          let pBuffer = wrapInput(p, length: byteSize, device: ctx.device),
          let wBuffer = wrapOutput(weights, length: byteSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(pBuffer, offset: 0, index: 0)
    encoder.setBuffer(wBuffer, offset: 0, index: 1)
    var nUint = UInt32(n)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 2)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_scale_rows")
public func skmetal_scale_rows(
    X: UnsafeRawPointer,
    weights: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int,
    d: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * d * MemoryLayout<Float>.stride
    let wSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "scale_rows", functionName: "scale_rows"),
          let xBuffer = wrapInput(X, length: matSize, device: ctx.device),
          let wBuffer = wrapInput(weights, length: wSize, device: ctx.device),
          let oBuffer = wrapOutput(output, length: matSize, device: ctx.device) else {
        return 1
    }

    let commandBuffer = ctx.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(xBuffer, offset: 0, index: 0)
    encoder.setBuffer(wBuffer, offset: 0, index: 1)
    encoder.setBuffer(oBuffer, offset: 0, index: 2)
    var nUint = UInt32(n)
    var dUint = UInt32(d)
    encoder.setBytes(&nUint, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&dUint, length: MemoryLayout<UInt32>.stride, index: 4)

    let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let gridSize = MTLSize(width: (n * d + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_compute_linear_irls")
public func skmetal_compute_linear_irls(
    linear: UnsafeMutableRawPointer,
    weights: UnsafeMutableRawPointer,
    b: Float,
    n: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride
    guard let pipeline = ctx.getPipeline(name: "compute_linear_irls", functionName: "compute_linear_irls"),
          let linearBuffer = wrapOutput(linear, length: byteSize, device: ctx.device),
          let weightBuffer = wrapOutput(weights, length: byteSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(linearBuffer, offset: 0, index: 0)
    enc.setBuffer(weightBuffer, offset: 0, index: 1)
    var bScalar = b
    enc.setBytes(&bScalar, length: MemoryLayout<Float>.stride, index: 2)
    var nU = UInt32(n)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_compute_error_scale")
public func skmetal_compute_error_scale(
    prob: UnsafeRawPointer,
    y: UnsafeRawPointer,
    X: UnsafeRawPointer,
    weights: UnsafeRawPointer,
    error: UnsafeMutableRawPointer,
    X_scaled: UnsafeMutableRawPointer,
    n: Int,
    p: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let byteSize = n * MemoryLayout<Float>.stride
    guard let pipeline = ctx.getPipeline(name: "compute_error_scale", functionName: "compute_error_scale"),
          let probBuffer = wrapInput(prob, length: byteSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: byteSize, device: ctx.device),
          let xBuffer = wrapInput(X, length: n * p * MemoryLayout<Float>.stride, device: ctx.device),
          let wBuffer = wrapInput(weights, length: byteSize, device: ctx.device),
          let eBuffer = wrapOutput(error, length: byteSize, device: ctx.device),
          let xsBuffer = wrapOutput(X_scaled, length: n * p * MemoryLayout<Float>.stride, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(probBuffer, offset: 0, index: 0)
    enc.setBuffer(yBuffer, offset: 0, index: 1)
    enc.setBuffer(xBuffer, offset: 0, index: 2)
    enc.setBuffer(wBuffer, offset: 0, index: 3)
    enc.setBuffer(eBuffer, offset: 0, index: 4)
    enc.setBuffer(xsBuffer, offset: 0, index: 5)
    var nU = UInt32(n); var pU = UInt32(p)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 6)
    enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 7)
    enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - L2 regularization kernels

@_cdecl("skmetal_l2_reg_irls")
public func skmetal_l2_reg_irls(
    Hessian: UnsafeMutableRawPointer,
    gradient: UnsafeMutableRawPointer,
    w: UnsafeRawPointer,
    alpha: Float,
    p: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let hSize = p * p * MemoryLayout<Float>.stride
    let gSize = p * MemoryLayout<Float>.stride
    guard let pipeline = ctx.getPipeline(name: "l2_reg_irls", functionName: "l2_reg_irls"),
          let hBuffer = wrapOutput(Hessian, length: hSize, device: ctx.device),
          let gBuffer = wrapOutput(gradient, length: gSize, device: ctx.device),
          let wBuffer = wrapInput(w, length: gSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(hBuffer, offset: 0, index: 0)
    enc.setBuffer(gBuffer, offset: 0, index: 1)
    enc.setBuffer(wBuffer, offset: 0, index: 2)
    var a = alpha
    enc.setBytes(&a, length: MemoryLayout<Float>.stride, index: 3)
    var pU = UInt32(p)
    enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.dispatchThreads(MTLSize(width: p, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_multinomial_l2_reg")
public func skmetal_multinomial_l2_reg(
    hessians: UnsafeMutableRawPointer,
    gradBatch: UnsafeMutableRawPointer,
    W: UnsafeRawPointer,
    alpha: Float,
    p: Int,
    C: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    guard let pipeline = ctx.getPipeline(name: "multinomial_l2_reg", functionName: "multinomial_l2_reg"),
          let hBuffer = wrapOutput(hessians, length: C * p * p * fs, device: ctx.device),
          let gBuffer = wrapOutput(gradBatch, length: C * p * fs, device: ctx.device),
          let wBuffer = wrapInput(W, length: p * C * fs, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(hBuffer, offset: 0, index: 0)
    enc.setBuffer(gBuffer, offset: 0, index: 1)
    enc.setBuffer(wBuffer, offset: 0, index: 2)
    var a = alpha; var pU = UInt32(p); var cU = UInt32(C)
    enc.setBytes(&a, length: fs, index: 3)
    enc.setBytes(&pU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 5)
    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let tgCount = MTLSize(width: (C + 15) / 16, height: (p + 15) / 16, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

// MARK: - Row max / row sum / softmax

@_cdecl("skmetal_row_max")
public func skmetal_row_max(
    matrix: UnsafeRawPointer,
    maxVals: UnsafeMutableRawPointer,
    n: Int,
    nCols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * nCols * MemoryLayout<Float>.stride
    let maxSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "row_max", functionName: "row_max"),
          let matBuffer = wrapInput(matrix, length: matSize, device: ctx.device),
          let maxBuffer = wrapOutput(maxVals, length: maxSize, device: ctx.device) else {
        return 1
    }

    let tgRowMax = MTLSize(width: 256, height: 1, depth: 1)
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(matBuffer, offset: 0, index: 0)
    enc.setBuffer(maxBuffer, offset: 0, index: 1)
    var nU = UInt32(n); var ncU = UInt32(nCols)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    enc.setBytes(&ncU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                             threadsPerThreadgroup: tgRowMax)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_row_sum")
public func skmetal_row_sum(
    matrix: UnsafeRawPointer,
    sums: UnsafeMutableRawPointer,
    n: Int,
    nCols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * nCols * MemoryLayout<Float>.stride
    let sumSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "row_sum", functionName: "row_sum"),
          let matBuffer = wrapInput(matrix, length: matSize, device: ctx.device),
          let sumBuffer = wrapOutput(sums, length: sumSize, device: ctx.device) else {
        return 1
    }

    let tgRowSum = MTLSize(width: 256, height: 1, depth: 1)
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(matBuffer, offset: 0, index: 0)
    enc.setBuffer(sumBuffer, offset: 0, index: 1)
    var nU = UInt32(n); var ncU = UInt32(nCols)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 2)
    enc.setBytes(&ncU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                             threadsPerThreadgroup: tgRowSum)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_softmax_exp")
public func skmetal_softmax_exp(
    matrix: UnsafeRawPointer,
    maxVals: UnsafeRawPointer,
    output: UnsafeMutableRawPointer,
    n: Int,
    nCols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * nCols * MemoryLayout<Float>.stride
    let maxSize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "softmax_exp", functionName: "softmax_exp"),
          let matBuffer = wrapInput(matrix, length: matSize, device: ctx.device),
          let maxBuffer = wrapInput(maxVals, length: maxSize, device: ctx.device),
          let outBuffer = wrapOutput(output, length: matSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(matBuffer, offset: 0, index: 0)
    enc.setBuffer(maxBuffer, offset: 0, index: 1)
    enc.setBuffer(outBuffer, offset: 0, index: 2)
    var nU = UInt32(n); var ncU = UInt32(nCols)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.setBytes(&ncU, length: MemoryLayout<UInt32>.stride, index: 4)
    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let tgCount = MTLSize(width: (nCols + 15) / 16, height: (n + 15) / 16, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_softmax_normalize_residual")
public func skmetal_softmax_normalize_residual(
    prob: UnsafeMutableRawPointer,
    rowSums: UnsafeRawPointer,
    y: UnsafeRawPointer,
    residual: UnsafeMutableRawPointer,
    n: Int,
    nCols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let matSize = n * nCols * MemoryLayout<Float>.stride
    let sumSize = n * MemoryLayout<Float>.stride
    let ySize = n * MemoryLayout<Float>.stride

    guard let pipeline = ctx.getPipeline(name: "softmax_normalize_residual", functionName: "softmax_normalize_residual"),
          let probBuffer = wrapOutput(prob, length: matSize, device: ctx.device),
          let sumBuffer = wrapInput(rowSums, length: sumSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: ySize, device: ctx.device),
          let resBuffer = wrapOutput(residual, length: matSize, device: ctx.device) else {
        return 1
    }

    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(probBuffer, offset: 0, index: 0)
    enc.setBuffer(sumBuffer, offset: 0, index: 1)
    enc.setBuffer(yBuffer, offset: 0, index: 2)
    enc.setBuffer(resBuffer, offset: 0, index: 3)
    var nU = UInt32(n); var ncU = UInt32(nCols)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 4)
    enc.setBytes(&ncU, length: MemoryLayout<UInt32>.stride, index: 5)
    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let tgCount = MTLSize(width: (nCols + 15) / 16, height: (n + 15) / 16, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

@_cdecl("skmetal_softmax_residual")
public func skmetal_softmax_residual(
    prob: UnsafeRawPointer,
    y: UnsafeRawPointer,
    residual: UnsafeMutableRawPointer,
    n: Int,
    nCols: Int
) -> Int32 {
    let ctx = MetalContext.shared
    let fs = MemoryLayout<Float>.stride
    let matSize = n * nCols * fs
    guard let pipeline = ctx.getPipeline(name: "softmax_residual", functionName: "softmax_residual"),
          let probBuffer = wrapInput(prob, length: matSize, device: ctx.device),
          let yBuffer = wrapInput(y, length: n * fs, device: ctx.device),
          let resBuffer = wrapOutput(residual, length: matSize, device: ctx.device) else {
        return 1
    }
    let cb = ctx.commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(probBuffer, offset: 0, index: 0)
    enc.setBuffer(yBuffer, offset: 0, index: 1)
    enc.setBuffer(resBuffer, offset: 0, index: 2)
    var nU = UInt32(n); var cU = UInt32(nCols)
    enc.setBytes(&nU, length: MemoryLayout<UInt32>.stride, index: 3)
    enc.setBytes(&cU, length: MemoryLayout<UInt32>.stride, index: 4)
    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let tgCount = MTLSize(width: (nCols + 15) / 16, height: (n + 15) / 16, depth: 1)
    enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return 0
}

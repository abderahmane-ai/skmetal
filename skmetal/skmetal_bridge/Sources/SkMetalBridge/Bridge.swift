import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

// MARK: - Shared helpers

func wrapInput(_ ptr: UnsafeRawPointer, length: Int, device: MTLDevice) -> MTLBuffer? {
    let mut = UnsafeMutableRawPointer(mutating: ptr)
    return device.makeBuffer(bytesNoCopy: mut, length: length,
                             options: .storageModeShared, deallocator: nil)
}

func wrapOutput(_ ptr: UnsafeMutableRawPointer, length: Int, device: MTLDevice) -> MTLBuffer? {
    return device.makeBuffer(bytesNoCopy: ptr, length: length,
                             options: .storageModeShared, deallocator: nil)
}

@_cdecl("skmetal_init")
public func skmetal_init() -> Int32 {
    _ = MetalContext.shared
    return 0
}

@_cdecl("skmetal_device_info")
public func skmetal_device_info(
    name: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    maxThreads: UnsafeMutablePointer<Int>?
) -> Int32 {
    let ctx = MetalContext.shared
    let nameStr = ctx.device.name
    name.pointee = strdup(nameStr)
    maxThreads?.pointee = ctx.device.maxThreadsPerThreadgroup.width
    return 0
}

// MARK: - Pipeline warmup

@_cdecl("skmetal_warmup")
public func skmetal_warmup() -> Int32 {
    let ctx = MetalContext.shared
    let pipelineNames: [(name: String, function: String)] = [
        ("reduce_sum", "reduce_sum"),
        ("reduce_mean_var", "reduce_mean_var"),
        ("pairwise_distance_direct", "pairwise_distance_direct"),
        ("pairwise_distance_squared", "pairwise_distance_squared"),
        ("row_norm_sq", "row_norm_sq"),
        ("distance_correct", "distance_correct"),
        ("argmin_rows", "argmin_rows"),
        ("scaler_fit", "scaler_fit"),
        ("column_minmax", "column_minmax"),
        ("irls_weight", "irls_weight"),
        ("scale_rows", "scale_rows"),
        ("scale_f32", "scale_f32"),
        ("sigmoid", "sigmoid"),
        ("subtract", "subtract"),
        ("axpy", "axpy"),
        ("norm_sq", "norm_sq"),
        ("add_scalar", "add_scalar"),
        ("column_means", "column_means"),
        ("center_columns", "center_columns"),
        ("compute_mindists", "compute_mindists"),
        ("kmeans_assign", "kmeans_assign"),
        ("kmeans_assign_partial", "kmeans_assign_partial"),
        ("kmeans_partial_sum", "kmeans_partial_sum"),
        ("kmeans_combine", "kmeans_combine"),
        ("kmeans_combine_normalize", "kmeans_combine_normalize"),
        ("kmeans_normalize", "kmeans_normalize"),
        ("kmeans_inertia", "kmeans_inertia"),
        ("kmeans_shift", "kmeans_shift"),
        ("knn_select_tile_topk", "knn_select_tile_topk"),
        ("knn_select_tile_topk_manhattan", "knn_select_tile_topk_manhattan"),
        ("knn_select_tile_topk_cosine", "knn_select_tile_topk_cosine"),
        ("knn_merge_topk", "knn_merge_topk"),
        ("knn_vote_classify", "knn_vote_classify"),
        ("knn_vote_regress", "knn_vote_regress"),
        ("knn_vote_classify_weighted", "knn_vote_classify_weighted"),
        ("knn_vote_regress_weighted", "knn_vote_regress_weighted"),
        ("soft_threshold", "soft_threshold"),
        ("column_transform", "column_transform"),
        ("transpose_f32", "transpose_f32"),
        ("sv_init", "sv_init"),
        ("sv_hook", "sv_hook"),
        ("sv_shortcut", "sv_shortcut"),
        ("tree_predict", "tree_predict"),
        ("tree_predict_all", "tree_predict_all"),
        ("row_max", "row_max"),
        ("row_sum", "row_sum"),
        ("softmax_exp", "softmax_exp"),
        ("softmax_normalize_residual", "softmax_normalize_residual"),
        ("negate", "negate"),
        ("multinomial_hessians", "multinomial_hessians"),
        ("compute_linear_irls", "compute_linear_irls"),
        ("compute_error_scale", "compute_error_scale"),
        ("l2_reg_irls", "l2_reg_irls"),
        ("multinomial_grad_l2", "multinomial_grad_l2"),
        ("rbf_apply", "rbf_apply"),
        ("fill_f32", "fill_f32"),
        ("softmax_residual", "softmax_residual"),
        ("svc_predict_binary", "svc_predict_binary"),
        ("gemm_simple", "gemm_simple"),
        ("convert_f32_to_f16", "convert_f32_to_f16"),
        ("convert_f16_to_f32", "convert_f16_to_f32"),
    ]

    let cb = ctx.commandQueue.makeCommandBuffer()!

    for (name, funcName) in pipelineNames {
        guard let pipeline = ctx.getPipeline(name: name, functionName: funcName) else {
            continue
        }
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
    }

    cb.commit()
    cb.waitUntilCompleted()

    let aSize = 4 * MemoryLayout<Float>.stride
    let bSize = 4 * MemoryLayout<Float>.stride
    let cSize = 4 * MemoryLayout<Float>.stride
    if let aBuf = ctx.device.makeBuffer(length: aSize, options: .storageModeShared),
       let bBuf = ctx.device.makeBuffer(length: bSize, options: .storageModeShared),
       let cBuf = ctx.device.makeBuffer(length: cSize, options: .storageModeShared) {
        let desc2x2 = MPSMatrixDescriptor(dimensions: 2, columns: 2, rowBytes: 8, dataType: .float32)
        let mA = MPSMatrix(buffer: aBuf, descriptor: desc2x2)
        let mB = MPSMatrix(buffer: bBuf, descriptor: desc2x2)
        let mC = MPSMatrix(buffer: cBuf, descriptor: desc2x2)
        let gemm = MPSMatrixMultiplication(
            device: ctx.device, transposeLeft: false, transposeRight: false,
            resultRows: 2, resultColumns: 2, interiorColumns: 2,
            alpha: 1.0, beta: 0.0)
        let cb2 = ctx.commandQueue.makeCommandBuffer()!
        gemm.encode(commandBuffer: cb2, leftMatrix: mA, rightMatrix: mB, resultMatrix: mC)
        cb2.commit()
        cb2.waitUntilCompleted()
    }

    return 0
}

#include <metal_stdlib>
using namespace metal;

// SIMD-group helpers for multi-level reduction.
// simd_sum, simd_min, simd_max are hardware instructions on Apple GPUs.
// We use simd_shuffle_down for the first level (zero barriers, 5 iterations),
// then threadgroup memory for the second level (waiting on SIMD group leaders).

// Sum reduction - each thread handles strided elements, threadgroup reduces
// Uses two-level reduction: simd_shuffle_down within SIMD group, then
// threadgroup gather + final simd_shuffle_down across group leaders.
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* partial_out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint total_threads = lsz * num_groups;
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += total_threads) {
        local_sum += input[i];
    }

    // Level 1: SIMD group reduction via shuffle (zero barriers, single hardware instruction)
    float simd_result = simd_sum(local_sum);

    // Level 2: threadgroup gather of SIMD group leaders
    threadgroup float shared[32];
    if (simd_lane_id == 0) {
        shared[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (lsz + 31) / 32;
    if (simd_group_id == 0) {
        float tg_partial = (lid < num_simd_groups) ? shared[lid] : 0.0f;
        float tg_result = simd_sum(tg_partial);
        if (lid == 0) {
            partial_out[tid / lsz] = tg_result;
        }
    }
}

// L2 norm squared: Σ input[i]² with SIMD reduction.
// Used by IRLS for GPU-resident convergence detection.
kernel void norm2(
    device const float* input [[buffer(0)]],
    device float* partial_out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    constant uint& write_offset [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint total_threads = lsz * num_groups;
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += total_threads) {
        local_sum += input[i] * input[i];
    }
    float simd_result = simd_sum(local_sum);
    threadgroup float shared[32];
    if (simd_lane_id == 0) {
        shared[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint num_simd_groups = (lsz + 31) / 32;
    if (simd_group_id == 0) {
        float tg_partial = (lid < num_simd_groups) ? shared[lid] : 0.0f;
        float tg_result = simd_sum(tg_partial);
        if (lid == 0) {
            partial_out[write_offset + (tid / lsz)] = tg_result;
        }
    }
}

// Max absolute difference between two vectors with SIMD reduction.
// Used by FISTA for GPU-resident convergence detection.
kernel void max_abs_diff(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* partial_out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& num_groups [[buffer(4)]],
    constant uint& write_offset [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint total_threads = lsz * num_groups;
    float local_max = 0.0f;
    for (uint i = tid; i < n; i += total_threads) {
        float diff = fabs(a[i] - b[i]);
        local_max = fmax(local_max, diff);
    }

    float simd_result = simd_max(local_max);

    threadgroup float shared[32];
    if (simd_lane_id == 0) {
        shared[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd_groups = (lsz + 31) / 32;
    if (simd_group_id == 0) {
        float tg_partial = (lid < num_simd_groups) ? shared[lid] : 0.0f;
        float tg_result = simd_max(tg_partial);
        if (lid == 0) {
            partial_out[write_offset + (tid / lsz)] = tg_result;
        }
    }
}

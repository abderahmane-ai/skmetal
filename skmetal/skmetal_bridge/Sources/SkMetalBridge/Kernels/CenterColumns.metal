#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_COLS = 16;

// Compute mean of each column of X (tall-skinny, n >> p).
// Tiled column approach: each threadgroup processes BLOCK_COLS columns,
// reading BLOCK_COLS contiguous elements per row (coalesced).
// Dispatch: ceil(p / BLOCK_COLS) threadgroups, each with 256 threads.
kernel void column_means(
    device const float* X [[buffer(0)]],
    device float* means [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& p [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    uint col_start = gid * BLOCK_COLS;
    if (col_start >= p) return;
    uint col_end = min(col_start + BLOCK_COLS, p);
    uint active_cols = col_end - col_start;

    float local_sum[BLOCK_COLS];
    for (uint b = 0; b < active_cols; b++) {
        local_sum[b] = 0.0f;
    }

    // Coalesced read: each thread reads BLOCK_COLS contiguous floats per row
    uint total_threads = lsz;
    for (uint i = lid; i < n; i += total_threads) {
        uint base = i * p + col_start;
        for (uint b = 0; b < active_cols; b++) {
            local_sum[b] += X[base + b];
        }
    }

    // Level 1: SIMD-group sum via simd_sum
    for (uint b = 0; b < active_cols; b++) {
        local_sum[b] = simd_sum(local_sum[b]);
    }

    // Level 2: SIMD group 0 writes per-column results to threadgroup
    uint lane_id = lid & 31;
    uint num_simd_groups = (lsz + 31) / 32;
    threadgroup float tg_shared[512];

    if (lane_id == 0) {
        uint sg_idx = lid >> 5;
        if (sg_idx < num_simd_groups) {
            for (uint b = 0; b < active_cols; b++) {
                tg_shared[b * num_simd_groups + sg_idx] = local_sum[b];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Level 3: tree-reduce per column across SIMD groups
    for (uint stride = num_simd_groups >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (uint b = 0; b < active_cols; b++) {
                tg_shared[b * num_simd_groups + lid] += tg_shared[b * num_simd_groups + lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        for (uint b = 0; b < active_cols; b++) {
            means[col_start + b] = tg_shared[b * num_simd_groups] / (float)n;
        }
    }
}

// Subtract column means from each element in-place:
// X[i][j] -= mean[j] for all i, j
// Uses float4 vectorized loads for ~4× memory throughput on aligned columns.
kernel void center_columns(
    device float* X [[buffer(0)]],
    device const float* means [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    uint base = tid * d;
    uint j = 0;
    if (d >= 4) {
        for (; j + 4 <= d; j += 4) {
            float4 v = *reinterpret_cast<device float4*>(X + base + j);
            float4 m = *reinterpret_cast<device const float4*>(means + j);
            v -= m;
            *reinterpret_cast<device float4*>(X + base + j) = v;
        }
    }
    for (; j < d; j++) {
        X[base + j] -= means[j];
    }
}

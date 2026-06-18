#include <metal_stdlib>
using namespace metal;

constexpr constant uint BLOCK_COLS = 16;

// Per-column min and max in a single dispatch using tiled column approach.
// Each threadgroup processes BLOCK_COLS columns, reading BLOCK_COLS
// contiguous elements per row (coalesced).
// X: n*d float32 row-major matrix
// min_out: d float32 (output)
// max_out: d float32 (output)
// Dispatch: ceil(d / BLOCK_COLS) threadgroups, each with 256 threads.
kernel void column_minmax(
    device const float* X [[buffer(0)]],
    device float* min_out [[buffer(1)]],
    device float* max_out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    uint col_start = gid * BLOCK_COLS;
    if (col_start >= d) return;
    uint col_end = min(col_start + BLOCK_COLS, d);
    uint active_cols = col_end - col_start;

    float local_min[16];
    float local_max[16];

    for (uint b = 0; b < active_cols; b++) {
        local_min[b] = FLT_MAX;
        local_max[b] = -FLT_MAX;
    }

    // Coalesced read: each thread processes BLOCK_COLS contiguous elements per row
    uint total_threads = lsz;
    for (uint i = lid; i < n; i += total_threads) {
        uint base = i * d + col_start;
        for (uint b = 0; b < active_cols; b++) {
            float val = X[base + b];
            local_min[b] = min(local_min[b], val);
            local_max[b] = max(local_max[b], val);
        }
    }

    // Level 1: SIMD group reduction for each column (zero barriers)
    for (uint b = 0; b < active_cols; b++) {
        local_min[b] = simd_min(local_min[b]);
        local_max[b] = simd_max(local_max[b]);
    }

    // Level 2: SIMD group 0 writes per-column results to threadgroup
    uint lane_id = lid & 31;
    uint num_simd_groups = (lsz + 31) / 32;
    threadgroup float tg_mins[128];  // max 16 cols * 8 SIMD groups
    threadgroup float tg_maxs[128];

    if (lane_id == 0) {
        uint sg_idx = lid >> 5;
        if (sg_idx < num_simd_groups) {
            for (uint b = 0; b < active_cols; b++) {
                tg_mins[b * num_simd_groups + sg_idx] = local_min[b];
                tg_maxs[b * num_simd_groups + sg_idx] = local_max[b];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduce per column across SIMD groups
    for (uint stride = num_simd_groups / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (uint b = 0; b < active_cols; b++) {
                uint base = b * num_simd_groups;
                uint idx_a = base + lid;
                uint idx_b = base + lid + stride;
                tg_mins[idx_a] = min(tg_mins[idx_a], tg_mins[idx_b]);
                tg_maxs[idx_a] = max(tg_maxs[idx_a], tg_maxs[idx_b]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        for (uint b = 0; b < active_cols; b++) {
            min_out[col_start + b] = tg_mins[b * num_simd_groups];
            max_out[col_start + b] = tg_maxs[b * num_simd_groups];
        }
    }
}

// Apply min-max normalization: X_scaled = (X - min) / (max - min) * (fmax - fmin) + fmin
// One thread per element. Grid: (d, n) total threads.
kernel void minmax_transform(
    device const float* X [[buffer(0)]],
    device float* X_out [[buffer(1)]],
    device const float* min_vals [[buffer(2)]],
    device const float* max_vals [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant float& feature_min [[buffer(6)]],
    constant float& feature_max [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= n || col >= d) return;
    float range = max_vals[col] - min_vals[col];
    float scale = (feature_max - feature_min) / max(range, 1e-10f);
    X_out[row * d + col] = (X[row * d + col] - min_vals[col]) * scale + feature_min;
}

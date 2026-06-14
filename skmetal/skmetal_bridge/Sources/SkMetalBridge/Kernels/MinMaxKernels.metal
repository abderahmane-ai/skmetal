#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_COLS = 8;

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

    float local_min[8];
    float local_max[8];

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

    threadgroup float tg_mins[2048];
    threadgroup float tg_maxs[2048];

    for (uint b = 0; b < active_cols; b++) {
        tg_mins[b * lsz + lid] = local_min[b];
        tg_maxs[b * lsz + lid] = local_max[b];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (uint b = 0; b < active_cols; b++) {
                uint idx_a = b * lsz + lid;
                uint idx_b = idx_a + stride;
                tg_mins[idx_a] = min(tg_mins[idx_a], tg_mins[idx_b]);
                tg_maxs[idx_a] = max(tg_maxs[idx_a], tg_maxs[idx_b]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        for (uint b = 0; b < active_cols; b++) {
            min_out[col_start + b] = tg_mins[b * lsz];
            max_out[col_start + b] = tg_maxs[b * lsz];
        }
    }
}

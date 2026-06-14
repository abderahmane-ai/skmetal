#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_COLS = 8;

// Fused StandardScaler: mean and variance for ALL columns in one dispatch.
// Uses tiled column approach: each threadgroup processes BLOCK_COLS columns.
// Each thread reads BLOCK_COLS contiguous elements per row (coalesced).
// Input:  X[n][d] (row-major)
// Output: mean[d], var[d]
// Dispatch: ceil(d / BLOCK_COLS) threadgroups, each with 256 threads.
kernel void scaler_fit(
    device const float* X [[buffer(0)]],
    device float* mean_out [[buffer(1)]],
    device float* var_out [[buffer(2)]],
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

    float local_mean[8];
    float local_m2[8];
    uint local_count[8];

    for (uint b = 0; b < active_cols; b++) {
        local_mean[b] = 0.0f;
        local_m2[b] = 0.0f;
        local_count[b] = 0;
    }

    // Coalesced: each thread reads BLOCK_COLS contiguous floats per row
    uint total_threads = lsz;
    for (uint i = lid; i < n; i += total_threads) {
        uint base = i * d + col_start;
        for (uint b = 0; b < active_cols; b++) {
            float x = X[base + b];
            local_count[b]++;
            float delta = x - local_mean[b];
            local_mean[b] += delta / (float)local_count[b];
            local_m2[b] += delta * (x - local_mean[b]);
        }
    }

    // Tree reduce across threads for each column in this block
    threadgroup float tg_mean[2048];
    threadgroup float tg_m2[2048];
    threadgroup uint tg_count[2048];

    for (uint b = 0; b < active_cols; b++) {
        tg_mean[b * lsz + lid] = local_mean[b];
        tg_m2[b * lsz + lid] = local_m2[b];
        tg_count[b * lsz + lid] = local_count[b];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (uint b = 0; b < active_cols; b++) {
                uint idx_a = b * lsz + lid;
                uint idx_b = idx_a + stride;
                uint a_c = tg_count[idx_a];
                uint b_c = tg_count[idx_b];
                if (b_c > 0) {
                    float a_m = tg_mean[idx_a];
                    float b_m = tg_mean[idx_b];
                    float a_m2 = tg_m2[idx_a];
                    float b_m2 = tg_m2[idx_b];
                    float delta = b_m - a_m;
                    uint new_count = a_c + b_c;
                    float new_mean = a_m + delta * ((float)b_c / (float)new_count);
                    float new_m2 = a_m2 + b_m2 + delta * delta * (float)a_c * (float)b_c / (float)new_count;
                    tg_mean[idx_a] = new_mean;
                    tg_m2[idx_a] = new_m2;
                    tg_count[idx_a] = new_count;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        for (uint b = 0; b < active_cols; b++) {
            float m = tg_mean[b * lsz];
            float v = tg_m2[b * lsz] / (float)tg_count[b * lsz];
            mean_out[col_start + b] = m;
            var_out[col_start + b] = v;
        }
    }
}

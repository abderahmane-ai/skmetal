#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_COLS = 16;

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

    float local_mean[16];
    float local_m2[16];
    uint local_count[16];

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

    // Level 1: SIMD-group Welford merge via shuffle
    for (uint b = 0; b < active_cols; b++) {
        float m = local_mean[b];
        float m2 = local_m2[b];
        uint cnt = local_count[b];
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float peer_m = simd_shuffle_down(m, offset);
            float peer_m2 = simd_shuffle_down(m2, offset);
            uint peer_cnt = simd_shuffle_down(cnt, offset);
            if (peer_cnt > 0 && cnt > 0) {
                float delta = peer_m - m;
                uint new_cnt = cnt + peer_cnt;
                m += delta * (float)peer_cnt / (float)new_cnt;
                m2 = m2 + peer_m2 + delta * delta * (float)cnt * (float)peer_cnt / (float)new_cnt;
                cnt = new_cnt;
            } else if (peer_cnt > 0) {
                m = peer_m; m2 = peer_m2; cnt = peer_cnt;
            }
        }
        // Broadcast lane 0's result to all lanes in the SIMD group
        local_mean[b] = simd_broadcast(m, 0);
        local_m2[b] = simd_broadcast(m2, 0);
        local_count[b] = simd_broadcast(cnt, 0);
    }

    // Level 2: SIMD group 0 writes per-column results to threadgroup
    uint lane_id = lid & 31;
    uint num_simd_groups = (lsz + 31) / 32;
    threadgroup float tg_mean[128];
    threadgroup float tg_m2[128];
    threadgroup uint tg_count[128];

    if (lane_id == 0) {
        uint sg_idx = lid >> 5;
        if (sg_idx < num_simd_groups) {
            for (uint b = 0; b < active_cols; b++) {
                tg_mean[b * num_simd_groups + sg_idx] = local_mean[b];
                tg_m2[b * num_simd_groups + sg_idx] = local_m2[b];
                tg_count[b * num_simd_groups + sg_idx] = local_count[b];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Level 3: tree-reduce per column across SIMD groups
    for (uint stride = num_simd_groups / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (uint b = 0; b < active_cols; b++) {
                uint base = b * num_simd_groups;
                uint idx_a = base + lid;
                uint idx_b = base + lid + stride;
                uint a_c = tg_count[idx_a];
                uint b_c = tg_count[idx_b];
                if (b_c > 0) {
                    float a_m = tg_mean[idx_a];
                    float a_m2 = tg_m2[idx_a];
                    float b_m = tg_mean[idx_b];
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
            uint base = b * num_simd_groups;
            float m = tg_mean[base];
            float v = tg_m2[base] / (float)tg_count[base];
            mean_out[col_start + b] = m;
            var_out[col_start + b] = v;
        }
    }
}

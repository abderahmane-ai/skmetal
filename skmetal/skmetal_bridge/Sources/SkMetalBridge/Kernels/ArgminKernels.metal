#include <metal_stdlib>
using namespace metal;

// Custom struct for argmin shuffle (shuffle val and idx together)
struct ArgMinPair {
    float val;
    uint idx;
};

// SIMD shuffle for ArgMinPair
ArgMinPair simd_shuffle_argmin(ArgMinPair data, ushort delta) {
    return ArgMinPair{
        simd_shuffle_down(data.val, delta),
        simd_shuffle_down(data.idx, delta)
    };
}

// Argmin per row using SIMD-group shuffle for reduction.
// Input:  matrix[n][k] (row-major)
// Output: indices[n] (uint)
// Each threadgroup handles one row; threads scan column chunks.
kernel void argmin_rows(
    device const float* matrix [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= n) return;
    uint row = gid;

    uint cols_per_thread = (k + lsz - 1) / lsz;
    uint start = lid * cols_per_thread;
    uint end = min(start + cols_per_thread, k);

    float local_min = INFINITY;
    uint local_idx = 0;
    for (uint j = start; j < end; ++j) {
        float val = matrix[row * k + j];
        if (val < local_min) {
            local_min = val;
            local_idx = j;
        }
    }

    // Level 1: SIMD group reduction via shuffle (zero barriers)
    ArgMinPair pair = {local_min, local_idx};
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        ArgMinPair peer = simd_shuffle_argmin(pair, offset);
        if (peer.val < pair.val) {
            pair = peer;
        }
    }

    // Level 2: gather SIMD group leaders into threadgroup
    uint num_simd_groups = (lsz + 31) / 32;
    threadgroup float tg_vals[32];
    threadgroup uint tg_idxs[32];
    if (simd_lane_id == 0) {
        tg_vals[simd_group_id] = pair.val;
        tg_idxs[simd_group_id] = pair.idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Level 3: final argmin across group leaders via tree (only 32 entries)
    if (simd_group_id == 0) {
        float best_val = tg_vals[lid];
        uint best_idx = tg_idxs[lid];
        for (uint stride = num_simd_groups / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                float peer_val = tg_vals[lid + stride];
                if (peer_val < best_val) {
                    best_val = peer_val;
                    best_idx = tg_idxs[lid + stride];
                    tg_vals[lid] = best_val;
                    tg_idxs[lid] = best_idx;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) {
            indices[row] = tg_idxs[0];
        }
    }
}

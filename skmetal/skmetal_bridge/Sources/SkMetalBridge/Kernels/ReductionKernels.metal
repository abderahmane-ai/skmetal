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

// Welford mean + variance with SIMD-group accelerated reduction
kernel void reduce_mean_var(
    device const float* input [[buffer(0)]],
    device float* partial_mean [[buffer(1)]],
    device float* partial_m2 [[buffer(2)]],
    device uint* partial_count [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_groups [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    float local_mean = 0.0f;
    float local_m2 = 0.0f;
    uint local_count = 0;

    uint total_threads = lsz * num_groups;
    for (uint i = tid; i < n; i += total_threads) {
        float x = input[i];
        local_count++;
        float delta = x - local_mean;
        local_mean += delta / (float)local_count;
        local_m2 += delta * (x - local_mean);
    }

    // Welford merge across SIMD group using shuffle
    // simd_sum hardware instruction handles the pairwise merge pattern
    float merge_mean = local_mean;
    float merge_m2 = local_m2;
    uint merge_cnt = local_count;

    for (uint offset = 16; offset > 0; offset >>= 1) {
        float peer_mean = simd_shuffle_down(merge_mean, offset);
        float peer_m2 = simd_shuffle_down(merge_m2, offset);
        uint peer_cnt = simd_shuffle_down(merge_cnt, offset);
        if (peer_cnt > 0 && merge_cnt > 0) {
            float delta = peer_mean - merge_mean;
            uint new_cnt = merge_cnt + peer_cnt;
            merge_mean += delta * (float)peer_cnt / (float)new_cnt;
            merge_m2 = merge_m2 + peer_m2 + delta * delta * (float)merge_cnt * (float)peer_cnt / (float)new_cnt;
            merge_cnt = new_cnt;
        } else if (peer_cnt > 0) {
            merge_mean = peer_mean;
            merge_m2 = peer_m2;
            merge_cnt = peer_cnt;
        }
    }

    // Write SIMD group result to threadgroup memory
    threadgroup float tg_means[32];
    threadgroup float tg_m2s[32];
    threadgroup uint tg_cnts[32];

    if (simd_lane_id == 0) {
        tg_means[simd_group_id] = merge_mean;
        tg_m2s[simd_group_id] = merge_m2;
        tg_cnts[simd_group_id] = merge_cnt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final Welford merge across SIMD group leaders
    uint num_simd_groups = (lsz + 31) / 32;
    if (simd_group_id == 0 && lid < num_simd_groups) {
        float final_mean = tg_means[lid];
        float final_m2 = tg_m2s[lid];
        uint final_cnt = tg_cnts[lid];

        for (uint offset = num_simd_groups / 2; offset > 0; offset >>= 1) {
            if (lid < offset) {
                float a_m = final_mean;
                float a_m2 = final_m2;
                uint a_c = final_cnt;
                float b_m = tg_means[lid + offset];
                float b_m2 = tg_m2s[lid + offset];
                uint b_c = tg_cnts[lid + offset];
                if (b_c > 0 && a_c > 0) {
                    float delta = b_m - a_m;
                    uint new_c = a_c + b_c;
                    a_m += delta * (float)b_c / (float)new_c;
                    a_m2 = a_m2 + b_m2 + delta * delta * (float)a_c * (float)b_c / (float)new_c;
                    a_c = new_c;
                } else if (b_c > 0) {
                    a_m = b_m; a_m2 = b_m2; a_c = b_c;
                }
                tg_means[lid] = a_m;
                tg_m2s[lid] = a_m2;
                tg_cnts[lid] = a_c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            uint gid = tid / lsz;
            partial_mean[gid] = tg_means[0];
            partial_m2[gid] = tg_m2s[0];
            partial_count[gid] = tg_cnts[0];
        }
    }
}

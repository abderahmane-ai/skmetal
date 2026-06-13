#include <metal_stdlib>
using namespace metal;

// Fused StandardScaler: mean and variance for ALL columns in one dispatch.
// Input:  X[n][d] (row-major)
// Output: mean[d], var[d]
// Dispatch: d threadgroups, each processing its column via Welford reduction.
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
    if (gid >= d) return;
    uint col = gid;

    float local_mean = 0.0f;
    float local_m2 = 0.0f;
    uint local_count = 0;

    uint total_threads = lsz;
    for (uint i = lid; i < n; i += total_threads) {
        float x = X[i * d + col];
        local_count++;
        float delta = x - local_mean;
        local_mean += delta / (float)local_count;
        local_m2 += delta * (x - local_mean);
    }

    // Threadgroup Welford reduction
    threadgroup float tg_mean[256];
    threadgroup float tg_m2[256];
    threadgroup uint tg_count[256];

    tg_mean[lid] = local_mean;
    tg_m2[lid] = local_m2;
    tg_count[lid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            uint a_c = tg_count[lid];
            uint b_c = tg_count[lid + stride];
            if (b_c > 0) {
                float a_m = tg_mean[lid];
                float b_m = tg_mean[lid + stride];
                float a_m2 = tg_m2[lid];
                float b_m2 = tg_m2[lid + stride];

                float delta = b_m - a_m;
                uint new_count = a_c + b_c;
                float new_mean = a_m + delta * ((float)b_c / (float)new_count);
                float new_m2 = a_m2 + b_m2 + delta * delta * (float)a_c * (float)b_c / (float)new_count;

                tg_mean[lid] = new_mean;
                tg_m2[lid] = new_m2;
                tg_count[lid] = new_count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        float m = tg_mean[0];
        float v = tg_m2[0] / (float)tg_count[0];
        mean_out[col] = m;
        var_out[col] = v;
    }
}

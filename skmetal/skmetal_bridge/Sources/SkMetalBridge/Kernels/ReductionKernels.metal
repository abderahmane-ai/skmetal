#include <metal_stdlib>
using namespace metal;

// Sum reduction - each thread handles strided elements, threadgroup reduces
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* partial_out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]]
) {
    threadgroup float shared[256];

    uint total_threads = lsz * num_groups;
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += total_threads) {
        local_sum += input[i];
    }

    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        uint gid = tid / lsz;
        partial_out[gid] = shared[0];
    }
}

// Mean + variance using Welford's algorithm
// Each threadgroup outputs partial state: mean, m2, count
kernel void reduce_mean_var(
    device const float* input [[buffer(0)]],
    device float* partial_mean [[buffer(1)]],
    device float* partial_m2 [[buffer(2)]],
    device uint* partial_count [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_groups [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]]
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

    // Threadgroup reduction using shared memory
    threadgroup float shared_mean[256];
    threadgroup float shared_m2[256];
    threadgroup uint shared_count[256];

    shared_mean[lid] = local_mean;
    shared_m2[lid] = local_m2;
    shared_count[lid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float a_mean = shared_mean[lid];
            float a_m2 = shared_m2[lid];
            uint a_count = shared_count[lid];
            float b_mean = shared_mean[lid + stride];
            float b_m2 = shared_m2[lid + stride];
            uint b_count = shared_count[lid + stride];

            if (b_count > 0) {
                float delta = b_mean - a_mean;
                uint new_count = a_count + b_count;
                float new_mean = a_mean + delta * ((float)b_count / (float)new_count);
                float new_m2 = a_m2 + b_m2 + delta * delta * (float)a_count * (float)b_count / (float)new_count;
                shared_mean[lid] = new_mean;
                shared_m2[lid] = new_m2;
                shared_count[lid] = new_count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        uint gid = tid / lsz;
        partial_mean[gid] = shared_mean[0];
        partial_m2[gid] = shared_m2[0];
        partial_count[gid] = shared_count[0];
    }
}

#include <metal_stdlib>
using namespace metal;

// Compute mean of each column of X (tall-skinny, n >> p).
// One threadgroup per column, tree-reduction within threadgroup.
kernel void column_means(
    device const float* X [[buffer(0)]],
    device float* means [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& p [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (gid >= p) return;
    uint col = gid;

    threadgroup float shared[256];
    float sum = 0.0f;

    uint rows_per_thread = (n + lsz - 1) / lsz;
    for (uint i = 0; i < rows_per_thread; ++i) {
        uint row = lid + i * lsz;
        if (row < n) {
            sum += X[row * p + col];
        }
    }

    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        means[col] = shared[0] / (float)n;
    }
}

// Subtract column means from each element in-place:
// X[i][j] -= mean[j] for all i, j
kernel void center_columns(
    device float* X [[buffer(0)]],
    device const float* means [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n * d) return;
    uint j = tid % d;
    X[tid] -= means[j];
}

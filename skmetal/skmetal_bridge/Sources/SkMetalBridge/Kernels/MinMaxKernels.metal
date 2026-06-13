#include <metal_stdlib>
using namespace metal;

// Per-column min and max in a single dispatch using threadgroup tree reduction.
// X: n*d float32 row-major matrix
// min_out: d float32 (output)
// max_out: d float32 (output)
// Dispatch: d threadgroups, each with 256 threads
// Each threadgroup processes one column
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
    if (gid >= d) return;
    uint col = gid;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    for (uint i = lid; i < n; i += lsz) {
        float val = X[i * d + col];
        local_min = min(local_min, val);
        local_max = max(local_max, val);
    }

    threadgroup float mins[256];
    threadgroup float maxs[256];
    mins[lid] = local_min;
    maxs[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            mins[lid] = min(mins[lid], mins[lid + stride]);
            maxs[lid] = max(maxs[lid], maxs[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        min_out[col] = mins[0];
        max_out[col] = maxs[0];
    }
}

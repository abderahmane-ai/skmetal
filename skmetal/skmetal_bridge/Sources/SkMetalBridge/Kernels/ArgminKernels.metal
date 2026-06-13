#include <metal_stdlib>
using namespace metal;

// Argmin per row: find column index of minimum value in each row.
// Input:  matrix[n][k] (row-major)
// Output: indices[n] (uint)
// Uses threadgroup memory for tree reduction.
kernel void argmin_rows(
    device const float* matrix [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (gid >= n) return;
    uint row = gid;

    // Each thread in threadgroup scans a chunk of columns
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

    // Threadgroup tree reduction for min
    threadgroup float shared_vals[256];
    threadgroup uint shared_idxs[256];
    shared_vals[lid] = local_min;
    shared_idxs[lid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (shared_vals[lid + stride] < shared_vals[lid]) {
                shared_vals[lid] = shared_vals[lid + stride];
                shared_idxs[lid] = shared_idxs[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        indices[row] = shared_idxs[0];
    }
}

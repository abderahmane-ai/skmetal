#include <metal_stdlib>
using namespace metal;

// Row-wise squared norm: row_norm[i] = sum(X[i][j]^2) for all j.
// Each thread processes one row.
kernel void row_norm_sq(
    device const float* X [[buffer(0)]],
    device float* norms [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float sum = 0.0f;
    for (uint j = 0; j < d; ++j) {
        float x = X[tid * d + j];
        sum += x * x;
    }
    norms[tid] = sum;
}

// For each point: dist = ||X[i] - C[assignments[i]]||^2
// One thread per point, no threadgroup needed.
kernel void compute_mindists(
    device const float* X [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device const uint* assignments [[buffer(2)]],
    device float* dists [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    uint c = assignments[tid];
    float dist = 0.0f;
    for (uint j = 0; j < d; ++j) {
        float diff = X[tid * d + j] - centroids[c * d + j];
        dist += diff * diff;
    }
    dists[tid] = dist;
}

// Fused: D[i][j] = X_norm[i] + C_norm[j] - 2 * raw_D[i][j]
// raw_D is X @ C^T computed via MPS GEMM
// Output can alias raw_D (in-place correction)
kernel void distance_correct(
    device float* D [[buffer(0)]],
    device const float* X_norm [[buffer(1)]],
    device const float* C_norm [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= n || j >= k) return;
    D[i * k + j] = X_norm[i] + C_norm[j] - 2.0f * D[i * k + j];
}

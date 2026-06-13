#include <metal_stdlib>
using namespace metal;

// Squared Euclidean pairwise distance
// Uses expansion: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*<x_i, x_j>
kernel void pairwise_distance_squared(
    device const float* X [[buffer(0)]],
    device const float* X_norm_sq [[buffer(1)]],
    device float* D [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= n || j >= n) return;

    float dot = 0.0f;
    for (uint k = 0; k < d; ++k) {
        dot += X[i * d + k] * X[j * d + k];
    }

    D[i * n + j] = X_norm_sq[i] + X_norm_sq[j] - 2.0f * dot;
}

// Direct squared Euclidean pairwise distance (no precomputed norms)
kernel void pairwise_distance_direct(
    device const float* X [[buffer(0)]],
    device float* D [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= n || j >= n) return;

    float sum = 0.0f;
    for (uint k = 0; k < d; ++k) {
        float diff = X[i * d + k] - X[j * d + k];
        sum += diff * diff;
    }
    D[i * n + j] = sum;
}

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
    uint base_i = i * d;
    uint base_j = j * d;
    uint k = 0;
    if (d >= 4) {
        for (; k + 4 <= d; k += 4) {
            float4 vi = *reinterpret_cast<device const float4*>(X + base_i + k);
            float4 vj = *reinterpret_cast<device const float4*>(X + base_j + k);
            dot += vi.x * vj.x + vi.y * vj.y + vi.z * vj.z + vi.w * vj.w;
        }
    }
    for (; k < d; ++k) {
        dot += X[base_i + k] * X[base_j + k];
    }

    D[i * n + j] = X_norm_sq[i] + X_norm_sq[j] - 2.0f * dot;
}

// Direct squared Euclidean pairwise distance (no precomputed norms)
// Uses float4 vectorized loads for ~2x bandwidth improvement.
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
    uint base_i = i * d;
    uint base_j = j * d;
    uint k = 0;
    if (d >= 4) {
        for (; k + 4 <= d; k += 4) {
            float4 vi = *reinterpret_cast<device const float4*>(X + base_i + k);
            float4 vj = *reinterpret_cast<device const float4*>(X + base_j + k);
            float4 diff = vi - vj;
            sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
        }
    }
    for (; k < d; ++k) {
        float diff = X[base_i + k] - X[base_j + k];
        sum += diff * diff;
    }
    D[i * n + j] = sum;
}

#include <metal_stdlib>
using namespace metal;

// Compute IRLS weight: w[i] = sqrt(p[i] * (1 - p[i]))
// Stable: clamps p to [eps, 1-eps] to avoid divide-by-zero.
kernel void irls_weight(
    device const float* p [[buffer(0)]],      // probabilities (n,)
    device float* weights [[buffer(1)]],       // output: sqrt(p*(1-p))
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float pi = p[tid];
    pi = clamp(pi, 1e-7f, 1.0f - 1e-7f);
    weights[tid] = sqrt(pi * (1.0f - pi));
}

// Scale each row of X by a weight: output[i][j] = X[i][j] * weight[i]
// Used to compute X_weighted for IRLS Hessian
kernel void scale_rows(
    device const float* X [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n * d) return;
    uint i = tid / d;
    uint j = tid % d;
    output[tid] = X[tid] * weights[i];
}

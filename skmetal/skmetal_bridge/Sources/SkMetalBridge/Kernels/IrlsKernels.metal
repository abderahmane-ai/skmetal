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

// Compute all C multinomial Hessians in one dispatch.
// Grid: C × p × p (c, f, g).
// Each thread computes Hessian_c[f][g] = sum_i X[i][f] * X[i][g] * p_ic * (1-p_ic)
// where p_ic = exp_prob[i][c] / row_sum[i] (normalized probability).
kernel void multinomial_hessians(
    device const float* X [[buffer(0)]],
    device const float* exp_prob [[buffer(1)]],
    device const float* row_sums [[buffer(2)]],
    device float* hessians [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint c = tid.x, f = tid.y, g = tid.z;
    if (c >= C || f >= p || g >= p) return;

    float h = 0.0f;
    for (uint i = 0; i < n; i++) {
        float pc = exp_prob[i * C + c] / row_sums[i];
        float w = pc * (1.0f - pc);
        h += X[i * p + f] * X[i * p + g] * w;
    }
    hessians[c * p * p + f * p + g] = h;
}

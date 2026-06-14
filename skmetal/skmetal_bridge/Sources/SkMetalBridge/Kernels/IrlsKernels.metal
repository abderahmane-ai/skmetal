#include <metal_stdlib>
using namespace metal;

// Compute IRLS weight: w[i] = sqrt(p[i] * (1 - p[i]))
// Stable: clamps p to [eps, 1-eps] to avoid divide-by-zero.
kernel void irls_weight(
    device const float* p [[buffer(0)]],
    device float* weights [[buffer(1)]],
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
    output[tid] = X[tid] * weights[i];
}

// Fused: linear += b → sigmoid → irls_weight, in one dispatch.
// Input: linear = X@w (raw scores)
// Output: linear = sigmoid(linear + b)  (probabilities)
// Output: weights = sqrt(prob * (1 - prob))
kernel void compute_linear_irls(
    device float* linear [[buffer(0)]],
    device float* weights [[buffer(1)]],
    constant float& b [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float val = linear[tid] + b;
    val = clamp(val, -100.0f, 100.0f);
    float prob = 1.0f / (1.0f + exp(-val));
    linear[tid] = prob;
    prob = clamp(prob, 1e-7f, 1.0f - 1e-7f);
    weights[tid] = sqrt(prob * (1.0f - prob));
}

// Fused: error = prob - y + scale rows of X by weight.
// Each thread handles one row: computes the error, then scales all p columns.
// Much better cache locality than separate subtract + scale_rows dispatches.
kernel void compute_error_scale(
    device const float* prob [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* error [[buffer(4)]],
    device float* X_scaled [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    error[tid] = prob[tid] - y[tid];
    float w = weights[tid];
    uint base = tid * p;
    for (uint j = 0; j < p; j++) {
        X_scaled[base + j] = X[base + j] * w;
    }
}

// L2 regularization: Hessian[i,i] += alpha, gradient[i] += alpha * w[i]
kernel void l2_reg_irls(
    device float* Hessian [[buffer(0)]],
    device float* gradient [[buffer(1)]],
    device const float* w [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant uint& p [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p) return;
    Hessian[tid * p + tid] += alpha;
    gradient[tid] += alpha * w[tid];
}

// Compute all C multinomial Hessians in one dispatch.
// Grid: (p*(p+1)/2, C) — packed lower-triangle column-major.
// Storage: LAPACK 'L' packed format (column-major).
kernel void multinomial_hessians(
    device const float* X [[buffer(0)]],
    device const float* exp_prob [[buffer(1)]],
    device float* hessians [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint idx = tid.x;
    uint c = tid.y;
    uint p_packed = p * (p + 1) / 2;
    if (c >= C || idx >= p_packed) return;

    // Convert packed lower-triangular column-major index to (row, col)
    // LAPACK 'L' format: pos(row, col) = row + col*(2p - col - 1)/2 for row >= col
    // offset(col) = col*(2p - col + 1)/2 (first element of column col)
    float col_f = (2.0 * float(p) + 1.0 - sqrt(float(int(2 * p + 1) * int(2 * p + 1) - 8 * int(idx)))) / 2.0;
    uint col = uint(col_f + 1e-4f);
    uint row = idx - col * (2 * p - col + 1) / 2 + col;

    float h = 0.0f;
    for (uint i = 0; i < n; i++) {
        float pc = exp_prob[i * C + c];
        float w = pc * (1.0f - pc);
        h += X[i * p + row] * X[i * p + col] * w;
    }
    if (row == col) h += alpha;
    hessians[c * p_packed + idx] = h;
}

// L2 regularization gradient term (Hessian diagonal is fused into multinomial_hessians):
//   gradient[c][i] += alpha * W[i][c]
// gradient is stored as (C, p) — contiguous per-class
// W is stored as (p, C) — column-major per-class
kernel void multinomial_grad_l2(
    device float* gradient [[buffer(0)]],
    device const float* W [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& p [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint c = tid.x, i = tid.y;
    if (c >= C || i >= p) return;
    uint g_idx = c * p + i;
    gradient[g_idx] += alpha * W[i * C + c];
}

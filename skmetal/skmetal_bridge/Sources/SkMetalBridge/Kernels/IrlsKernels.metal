#include <metal_stdlib>
using namespace metal;

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

// Fused: sigmoid + residual (prob - y) + per-element log loss.
// For L-BFGS gradient and loss computation (binary logistic regression).
// y ∈ {0, 1}, y_adj = 2*y - 1 ∈ {-1, 1}
// Log loss: log(1 + exp(-y_adj * lin)) — numerically stable.
kernel void sigmoid_grad_loss_binary(
    device const float* lin [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* prob [[buffer(2)]],
    device float* residual [[buffer(3)]],
    device float* loss_i [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float z = lin[tid];
    z = clamp(z, -100.0f, 100.0f);
    float p = 1.0f / (1.0f + exp(-z));
    prob[tid] = p;
    residual[tid] = p - y[tid];
    float y_adj = 2.0f * y[tid] - 1.0f;
    float t = -y_adj * z;
    if (t > 0) {
        loss_i[tid] = t + log(1.0f + exp(-t));
    } else {
        loss_i[tid] = log(1.0f + exp(t));
    }
}

// Per-element log loss only (for line search, no sigmoid needed).
// Uses same numerically stable log(1+exp(-y_adj * lin)).
kernel void log_loss_binary(
    device const float* lin [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* loss_i [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float z = lin[tid];
    float y_adj = 2.0f * y[tid] - 1.0f;
    float t = -y_adj * z;
    if (t > 0) {
        loss_i[tid] = t + log(1.0f + exp(-t));
    } else {
        loss_i[tid] = log(1.0f + exp(t));
    }
}

// Per-element cross-entropy loss for multinomial L-BFGS line search.
// y encodes the true class as an integer index ∈ [0, C).
// loss_i[tid] = -log(prob[tid * C + class])
kernel void cross_entropy_loss(
    device const float* prob [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* loss_i [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    uint c = uint(y[tid] + 0.5f);
    float p = prob[tid * C + c];
    p = clamp(p, 1e-7f, 1.0f);
    loss_i[tid] = -log(p);
}

// L2 regularization gradient term (Hessian diagonal is fused into multinomial_hessians):
//   gradient[c][i] += alpha * W[i][c]
// gradient is stored as (C, p) — contiguous per-class
// W is stored as (p, C) — column-master per-class
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

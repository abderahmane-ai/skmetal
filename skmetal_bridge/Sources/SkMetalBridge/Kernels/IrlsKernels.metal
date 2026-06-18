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
    float prob = 1.0f / (1.0f + fast::exp(-val));
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
    // float4 vectorized: 4 elements per iteration
    uint j = 0;
    float4 w4 = float4{w, w, w, w};
    for (; j + 4 <= p; j += 4) {
        float4 x = *reinterpret_cast<device const float4*>(X + base + j);
        *reinterpret_cast<device float4*>(X_scaled + base + j) = w4 * x;
    }
    for (; j < p; j++) {
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
    float p = 1.0f / (1.0f + fast::exp(-z));
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

// Fused L-BFGS gradient + loss kernel for binary logistic regression.
//
// Replaces 5 dispatches (2 MPS GEMM, sigmoid, axpy, reduce_sum) with 1.
// X is read exactly once — vs. twice in the old separate GEMM path.
//
// Design: cooperative SIMD-group row processing (32 threads per row).
// - Adjacent lanes → coalesced float4 reads of X.
// - w in threadgroup memory → broadcast (free).
// - simd_sum() = dot product in one instruction.
// - Gradient in threadgroup memory, atomically added at kernel end.
//
// Threadgroup memory: 2 × p × 4 bytes (grad_tg + w_tg). For p=1000: 8 KB.
// Occupancy: 4 concurrent groups/core at p=500, 2 at p=1000.
//
// Grid: num_simd_groups × 1 × 1. Each group = 32 threads = 1 threadgroup.
kernel void lbfgs_grad_loss_binary_fused(
    device const float* X          [[buffer(0)]],  // (n, p) row-major
    device const float* w          [[buffer(1)]],  // (p,)
    device const float* y          [[buffer(2)]],  // (n,)
    device float* grad_out         [[buffer(3)]],  // (p,) output — zero-init before dispatch
    device float* loss_partial     [[buffer(4)]],  // (num_groups,) one float per group
    constant uint& n               [[buffer(5)]],
    constant uint& p               [[buffer(6)]],
    uint simd_lane     [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]],
    threadgroup float* grad_tg [[threadgroup(0)]],
    threadgroup float* w_tg    [[threadgroup(1)]]
) {
    uint num_groups = simd_groups_per_tg;

    // ---- Load w into threadgroup memory ----
    for (uint j = simd_lane; j < p; j += 32) {
        w_tg[j] = w[j];
    }
    // Zero-init partial gradient
    for (uint j = simd_lane; j < p; j += 32) {
        grad_tg[j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Process rows assigned to this SIMD-group ----
    uint rows_per_group = (n + num_groups - 1) / num_groups;
    uint row_start = simd_group_id * rows_per_group;
    uint row_end = min(row_start + rows_per_group, n);

    if (row_start >= row_end) {
        if (simd_lane == 0) loss_partial[simd_group_id] = 0.0f;
        return;
    }

    float loss_sum = 0.0f;

    for (uint i = row_start; i < row_end; i++) {
        uint base = i * p;

        // Dot product: coalesced float4 reads + simd_sum
        float dot = 0.0f;
        uint j = simd_lane * 4;
        for (; j + 4 <= p; j += 128) {
            float4 xv = *reinterpret_cast<device const float4*>(X + base + j);
            float4 wv = *reinterpret_cast<const threadgroup float4*>(w_tg + j);
            dot += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
        for (; j < p; j += 32) {
            if (j + simd_lane < p) {
                dot += X[base + j + simd_lane] * w_tg[j + simd_lane];
            }
        }

        float lin = simd_sum(dot);

        // Sigmoid + residual + log loss
        float z = clamp(lin, -100.0f, 100.0f);
        float prob = 1.0f / (1.0f + fast::exp(-z));
        float residual = prob - y[i];
        float y_adj = 2.0f * y[i] - 1.0f;
        float t = -y_adj * z;
        loss_sum += (t > 0.0f) ? (t + log(1.0f + exp(-t))) : log(1.0f + exp(t));

        // Gradient: lane k writes to columns k, k+32, k+64, ...
        j = simd_lane;
        for (; j < p; j += 32) {
            grad_tg[j] += residual * X[base + j];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Atomically add threadgroup gradient to global ----
    for (uint j = simd_lane; j < p; j += 32) {
        float val = grad_tg[j];
        if (val != 0.0f) {
            atomic_fetch_add_explicit(
                (device atomic_float*)(grad_out + j), val, memory_order_relaxed);
        }
    }

    // ---- Reduce loss across SIMD lanes ----
    float group_loss = simd_sum(loss_sum);
    if (simd_lane == 0) {
        loss_partial[simd_group_id] = group_loss;
    }
}

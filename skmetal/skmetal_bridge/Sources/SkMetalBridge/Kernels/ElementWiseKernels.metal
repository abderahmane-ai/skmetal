#include <metal_stdlib>
using namespace metal;

// Sigmoid: 1 / (1 + exp(-x))
kernel void sigmoid(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float x = input[tid];
    x = clamp(x, -100.0f, 100.0f);
    output[tid] = 1.0f / (1.0f + exp(-x));
}

// Element-wise: output[i] = a[i] - b[i]
kernel void subtract(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    output[tid] = a[tid] - b[tid];
}

// Element-wise: array[i] += scalar
kernel void add_scalar(
    device float* array [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    array[tid] += scalar;
}

// Element-wise: output[i] = a[i] + alpha * b[i]
// Used for gradient descent step: w -= step * grad
kernel void axpy(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    a[tid] += alpha * b[tid];
}

// RBF kernel: D[i][j] = exp(-gamma * D[i][j])
// D already contains squared distances. In-place.
kernel void rbf_apply(
    device float* D [[buffer(0)]],
    constant float& gamma [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= m || gid.y >= n) return;
    uint idx = gid.y * m + gid.x;
    D[idx] = exp(-gamma * D[idx]);
}

// SVC predict: decision[i] = sum_k dual_coef[k] * exp(-gamma * ||X_test[i] - X_sv[k]||^2) + intercept
// Each thread handles one test point.
kernel void svc_predict_binary(
    device const float* X_test [[buffer(0)]],
    device const float* X_sv [[buffer(1)]],
    device const float* dual_coef [[buffer(2)]],
    device const float* intercept [[buffer(3)]],
    device float* decisions [[buffer(4)]],
    constant uint& n_test [[buffer(5)]],
    constant uint& n_sv [[buffer(6)]],
    constant uint& d [[buffer(7)]],
    constant float& gamma [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_test) return;
    float s = 0.0f;
    for (uint k = 0; k < n_sv; k++) {
        float d2 = 0.0f;
        uint base_test = tid * d;
        uint base_sv = k * d;
        uint l = 0;
        if (d >= 4) {
            for (; l + 4 <= d; l += 4) {
                float4 vx = *reinterpret_cast<device const float4*>(X_test + base_test + l);
                float4 vs = *reinterpret_cast<device const float4*>(X_sv + base_sv + l);
                float4 diff = vx - vs;
                d2 += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
            }
        }
        for (; l < d; l++) {
            float diff = X_test[base_test + l] - X_sv[base_sv + l];
            d2 += diff * diff;
        }
        s += dual_coef[k] * exp(-gamma * d2);
    }
    decisions[tid] = s + intercept[0];
}

// Add scalar to diagonal of a square matrix (L2 regularization for Ridge/IRLS)
kernel void add_diagonal(
    device float* matrix [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= p) return;
    matrix[tid * p + tid] += value;
}

// Residual after softmax: residual[i][j] = prob[i][j] - (j == y[i] ? 1 : 0)
// prob is already normalized (MPSMatrixSoftMax output).
kernel void softmax_residual(
    device const float* prob [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* residual [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& n_cols [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y, col = tid.x;
    if (row >= n || col >= n_cols) return;
    uint idx = row * n_cols + col;
    uint true_class = uint(y[row]);
    float target = (col == true_class) ? 1.0f : 0.0f;
    residual[idx] = prob[idx] - target;
}

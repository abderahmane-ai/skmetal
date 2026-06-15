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

// Transpose f32 matrix (row-major ↔ column-major)
// Each thread copies one element: gid.x = col, gid.y = row
kernel void transpose_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= cols || gid.y >= rows) return;
    output[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
}

// Row-wise max: for each row, find maximum value across columns
// Uses float4 vectorized loads for ~2x bandwidth on aligned rows.
kernel void row_max(
    device const float* matrix [[buffer(0)]],
    device float* max_vals [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& n_cols [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float mx = -INFINITY;
    uint base = tid * n_cols;
    uint j = 0;
    if (n_cols >= 4) {
        for (; j + 4 <= n_cols; j += 4) {
            float4 v = *reinterpret_cast<device const float4*>(matrix + base + j);
            mx = max(mx, v.x);
            mx = max(mx, v.y);
            mx = max(mx, v.z);
            mx = max(mx, v.w);
        }
    }
    for (; j < n_cols; j++) {
        mx = max(mx, matrix[base + j]);
    }
    max_vals[tid] = mx;
}

// Row-wise sum: for each row, compute sum of values
// Uses float4 vectorized loads for ~2x bandwidth on aligned rows.
kernel void row_sum(
    device const float* matrix [[buffer(0)]],
    device float* sums [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& n_cols [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float s = 0.0f;
    uint base = tid * n_cols;
    uint j = 0;
    if (n_cols >= 4) {
        for (; j + 4 <= n_cols; j += 4) {
            float4 v = *reinterpret_cast<device const float4*>(matrix + base + j);
            s += v.x + v.y + v.z + v.w;
        }
    }
    for (; j < n_cols; j++) {
        s += matrix[base + j];
    }
    sums[tid] = s;
}

// Softmax numerator: exp(x - row_max[row]) for each element
kernel void softmax_exp(
    device const float* matrix [[buffer(0)]],
    device const float* max_vals [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& n_cols [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y, col = tid.x;
    if (row >= n || col >= n_cols) return;
    output[row * n_cols + col] = exp(matrix[row * n_cols + col] - max_vals[row]);
}

// Normalize softmax probabilities and compute residual = prob - one_hot
kernel void softmax_normalize_residual(
    device float* prob [[buffer(0)]],
    device const float* row_sums [[buffer(1)]],
    device const float* y [[buffer(2)]],
    device float* residual [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& n_cols [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y, col = tid.x;
    if (row >= n || col >= n_cols) return;
    uint idx = row * n_cols + col;
    float p = prob[idx] / row_sums[row];
    prob[idx] = p;
    uint true_class = uint(y[row]);
    float target = (col == true_class) ? 1.0f : 0.0f;
    residual[idx] = p - target;
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

// Element-wise: output[i] = -a[i]
kernel void negate(
    device const float* a [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    output[tid] = -a[tid];
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

// Fill a buffer with a constant float value
kernel void fill_f32(
    device float* buf [[buffer(0)]],
    constant float& val [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    buf[tid] = val;
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

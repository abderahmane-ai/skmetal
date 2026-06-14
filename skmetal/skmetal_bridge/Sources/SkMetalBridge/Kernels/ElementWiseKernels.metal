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

// L2 norm squared of a vector
kernel void norm_sq(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    output[tid] = input[tid] * input[tid];
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
    for (uint j = 0; j < n_cols; j++) {
        mx = max(mx, matrix[base + j]);
    }
    max_vals[tid] = mx;
}

// Row-wise sum: for each row, compute sum of values
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
    for (uint j = 0; j < n_cols; j++) {
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
    uint row = tid.x, col = tid.y;
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
    uint row = tid.x, col = tid.y;
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

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

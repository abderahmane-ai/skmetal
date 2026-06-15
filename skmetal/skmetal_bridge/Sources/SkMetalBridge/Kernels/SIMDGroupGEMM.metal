#include <metal_stdlib>
using namespace metal;

// simdgroup_matrix GEMM: each simdgroup (32 threads) computes one 8x8 tile of C.
// Requires M%8==0, N%8==0, K%8==0. Non-aligned dimensions fall back to MPS.
// C = A @ B, both row-major, no transpose.
// Grid: ((N+7)/8, (M+7)/8) — gid.x = tile col, gid.y = tile row.
// Each threadgroup = 1 simdgroup = 32 threads. Use grid dispatch.

// Float32 (default precision) variant.
kernel void simdgroup_gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint base_row = gid.y * 8;
    uint base_col = gid.x * 8;

    simdgroup_matrix<float, 8> c = {};

    for (uint kk = 0; kk < K; kk += 8) {
        simdgroup_matrix<float, 8> a, b;
        simdgroup_load(a, A + base_row * K + kk, K);
        simdgroup_load(b, B + kk * N + base_col, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    simdgroup_store(c, C + base_row * N + base_col, N);
}

// Float16 (half precision) variant. Same structure but half type for A/B/C.
// Input/output buffers are half-precision. 2x compute throughput on M-series.
kernel void simdgroup_gemm_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint base_row = gid.y * 8;
    uint base_col = gid.x * 8;

    simdgroup_matrix<half, 8> c = {};

    for (uint kk = 0; kk < K; kk += 8) {
        simdgroup_matrix<half, 8> a, b;
        simdgroup_load(a, A + base_row * K + kk, K);
        simdgroup_load(b, B + kk * N + base_col, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    simdgroup_store(c, C + base_row * N + base_col, N);
}

// Type converters for f32<->f16 (element-wise).
kernel void convert_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) output[tid] = half(input[tid]);
}

kernel void convert_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) output[tid] = float(input[tid]);
}

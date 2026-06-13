#include <metal_stdlib>
using namespace metal;

// Simple GEMM fallback for small matrices or when MPS unavailable
// MPS path is preferred and used from Swift; this is for custom kernels
kernel void gemm_simple(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant bool& transA [[buffer(6)]],
    constant bool& transB [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    if (!transA && !transB) {
        for (uint k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
    } else if (transA && !transB) {
        for (uint k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[k * N + col];
        }
    } else if (!transA && transB) {
        for (uint k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
    } else {
        for (uint k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[col * K + k];
        }
    }
    C[row * N + col] = sum;
}

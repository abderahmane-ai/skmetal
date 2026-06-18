#include <metal_stdlib>
using namespace metal;

// Squared Euclidean pairwise distance via expanded formula:
//   D[i][j] = ||X[i]||² + ||X[j]||² - 2 * X[i]·X[j]
// Row norms must be pre-computed (e.g., via row_norm_sq kernel).
// Cross product X @ X^T must be pre-computed (e.g., via MPS GEMM or simdgroup_gemm).
// This kernel combines the two, dispatching one thread per output element.
// Grid: (n, n), one thread per element. No threadgroup memory, no barriers.
kernel void pairwise_from_cross(
    device const float* X_norm_sq [[buffer(0)]],
    device const float* cross [[buffer(1)]],
    device float* D [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= n || j >= n) return;
    float val = X_norm_sq[i] + X_norm_sq[j] - 2.0f * cross[i * n + j];
    D[i * n + j] = max(val, 0.0f);
}

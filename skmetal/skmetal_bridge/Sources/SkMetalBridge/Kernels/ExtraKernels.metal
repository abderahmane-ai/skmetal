#include <metal_stdlib>
using namespace metal;

// Soft thresholding for Lasso (FISTA):
// w[i] = sign(x) * max(|x| - threshold, 0)
kernel void soft_threshold(
    device float* w [[buffer(0)]],
    device const float* w_temp [[buffer(1)]],
    constant float& threshold [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float x = w_temp[tid];
    float ax = fabs(x);
    if (ax <= threshold) {
        w[tid] = 0.0f;
    } else {
        w[tid] = (x > 0.0f) ? (x - threshold) : (x + threshold);
    }
}

// Column-wise transform for RobustScaler / StandardScaler:
// output[i][j] = (input[i][j] - center[j]) * scale[j]
kernel void column_transform(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* center [[buffer(2)]],
    device const float* scale [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n * d) return;
    uint j = tid % d;
    output[tid] = (input[tid] - center[j]) * scale[j];
}

// Element-wise scale: a[i] *= s (in-place)
kernel void scale_f32(
    device float* a [[buffer(0)]],
    constant float& s [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    a[tid] *= s;
}

// Shiloach-Vishkin connected components — initialize parent array (parent[i] = i)
kernel void sv_init(
    device int* parent [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    parent[tid] = int(tid);
}

// Shiloach-Vishkin hook phase: for each edge (u,v), minimize parent[u], parent[v]
// edges is packed pairs: [u0, v0, u1, v1, ...]  (2 * edge_count ints)
kernel void sv_hook(
    device const int* edges [[buffer(0)]],
    device int* parent [[buffer(1)]],
    constant uint& edge_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= edge_count) return;
    uint u = uint(edges[tid * 2]);
    uint v = uint(edges[tid * 2 + 1]);
    int pu = parent[u];
    int pv = parent[v];
    if (pu < pv) {
        atomic_fetch_min_explicit((device atomic_int*)(parent + v), pu, memory_order_relaxed);
    } else if (pv < pu) {
        atomic_fetch_min_explicit((device atomic_int*)(parent + u), pv, memory_order_relaxed);
    }
}

// Shiloach-Vishkin shortcut phase: parent[i] = parent[parent[i]]
kernel void sv_shortcut(
    device int* parent [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    uint p = uint(parent[tid]);
    if (tid != p) {
        parent[tid] = parent[p];
    }
}

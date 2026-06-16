#include <metal_stdlib>
using namespace metal;

// Assignment step: find nearest centroid for each point
// Grid: n threads
kernel void kmeans_assign(
    device const float* X [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device uint* assignments [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float min_dist = FLT_MAX;
    uint best_k = 0;
    uint base = tid * d;

    for (uint c = 0; c < k; ++c) {
        float dist = 0.0f;
        uint c_base = c * d;
        uint j = 0;
        if (d >= 4) {
            for (; j + 4 <= d; j += 4) {
                float4 vx = *reinterpret_cast<device const float4*>(X + base + j);
                float4 vc = *reinterpret_cast<device const float4*>(centroids + c_base + j);
                float4 diff = vx - vc;
                dist += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
            }
        }
        for (; j < d; ++j) {
            float diff = X[base + j] - centroids[c_base + j];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_k = c;
        }
    }
    assignments[tid] = best_k;
}

// CAS-free centroid accumulation per (group, cluster) — one thread per cell.
// Grid: (k, num_groups) 2D — thread (cluster=c, group=g) accumulates all points
// in group g assigned to cluster c into its private area of global partial buffers.
// Zero contention, zero CAS, zero threadgroup atomics.
// Called AFTER kmeans_assign writes assignments.
// 256 threads per (cluster, group) pair cooperatively accumulate assigned points.
// Each thread handles ceil(d/256) contiguous dimensions — no atomics needed.
// Grid: (k, num_groups) threadgroups, 256 threads per threadgroup.
kernel void kmeans_accumulate(
    device const float* X [[buffer(0)]],
    device const uint* assignments [[buffer(1)]],
    device float* partial_centroids [[buffer(2)]],
    device uint* partial_counts [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& num_groups [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint cluster = gid.x;
    uint group = gid.y;
    if (cluster >= k || group >= num_groups) return;

    uint group_size = (n + num_groups - 1) / num_groups;
    uint start = group * group_size;
    uint end = min(n, start + group_size);

    uint centroid_base = group * k * d + cluster * d;

    uint dims_per_thread = (d + 255) / 256;
    uint my_start = lid.x * dims_per_thread;
    uint my_end = min(d, my_start + dims_per_thread);

    // Zero this thread's dimension range
    for (uint j = my_start; j < my_end; ++j) {
        partial_centroids[centroid_base + j] = 0.0f;
    }
    if (lid.x == 0) {
        partial_counts[group * k + cluster] = 0;
    }

    // Single pass: count matching points and accumulate dims
    uint count = 0;
    for (uint i = start; i < end; ++i) {
        if (assignments[i] == cluster) {
            ++count;
            uint x_base = i * d;
            // float4 vectorized accumulation
            uint j = my_start;
            for (; j + 4 <= my_end; j += 4) {
                float4 vx = *reinterpret_cast<device const float4*>(X + x_base + j);
                device float4* dst = reinterpret_cast<device float4*>(partial_centroids + centroid_base + j);
                *dst = *dst + vx;
            }
            for (; j < my_end; ++j) {
                partial_centroids[centroid_base + j] += X[x_base + j];
            }
        }
    }

    if (lid.x == 0) {
        partial_counts[group * k + cluster] = count;
    }
}

// Fused combine + normalize: sums partial sums across groups and divides by count.
// Grid: (d, k), one thread per (centroid, feature).
kernel void kmeans_combine_normalize(
    device const float* partial_centroids [[buffer(0)]],
    device const uint* partial_counts [[buffer(1)]],
    device float* centroids [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant uint& num_groups [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.y;
    uint f = gid.x;
    if (c >= k || f >= d) return;

    float sum = 0.0f;
    uint count = 0;
    for (uint g = 0; g < num_groups; ++g) {
        uint base = g * k * d + c * d;
        sum += partial_centroids[base + f];
        count += partial_counts[g * k + c];
    }
    centroids[c * d + f] = (count > 0) ? sum / (float)count : 0.0f;
}

// Compute total inertia: Σ ‖X[i] - centroids[assignments[i]]‖²
// Each thread computes one point's distance, reduces via simd_sum + threadgroup.
// Output: partial_sums[gid] = per-threadgroup partial inertia.
// Swift sums partials on CPU (typically < 2000 groups → trivial).
// Grid: n threads, 256 per threadgroup.
kernel void kmeans_inertia(
    device const float* X [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device const uint* assignments [[buffer(2)]],
    device float* partial_sums [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    float dist = 0.0f;
    if (tid < n) {
        uint c = assignments[tid];
        uint base = tid * d;
        uint c_base = c * d;
        uint j = 0;
        if (d >= 4) {
            for (; j + 4 <= d; j += 4) {
                float4 vx = *reinterpret_cast<device const float4*>(X + base + j);
                float4 vc = *reinterpret_cast<device const float4*>(centroids + c_base + j);
                float4 diff = vx - vc;
                dist += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
            }
        }
        for (; j < d; ++j) {
            float diff = X[base + j] - centroids[c_base + j];
            dist += diff * diff;
        }
    }

    // SIMD sum within each SIMD group
    float sg_sum = simd_sum(dist);
    uint lane_id = lid & 31;
    uint num_sg = (lsz + 31) / 32;
    threadgroup float tg_buf[32];
    if (lane_id == 0) {
        tg_buf[lid >> 5] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < num_sg; ++i) {
            total += tg_buf[i];
        }
        partial_sums[gid] = total;
    }
}

// Compute max centroid shift: max_c ‖new_centroids[c] - old_centroids[c]‖²
// One thread per centroid, reduces via simd_max + threadgroup.
// Output: partial_max[gid] = per-threadgroup max squared shift.
// Swift takes sqrt of max across partials.
// Grid: k threads, 256 per threadgroup.
kernel void kmeans_shift(
    device const float* new_centroids [[buffer(0)]],
    device const float* old_centroids [[buffer(1)]],
    device float* partial_max [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    float max_sq = 0.0f;
    if (tid < k) {
        uint base = tid * d;
        float sq = 0.0f;
        uint j = 0;
        if (d >= 4) {
            for (; j + 4 <= d; j += 4) {
                float4 vn = *reinterpret_cast<device const float4*>(new_centroids + base + j);
                float4 vo = *reinterpret_cast<device const float4*>(old_centroids + base + j);
                float4 diff = vn - vo;
                sq += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
            }
        }
        for (; j < d; ++j) {
            float diff = new_centroids[base + j] - old_centroids[base + j];
            sq += diff * diff;
        }
        max_sq = sq;
    }

    float sg_max = simd_max(max_sq);
    uint lane_id = lid & 31;
    uint num_sg = (lsz + 31) / 32;
    threadgroup float tg_buf[32];
    if (lane_id == 0) {
        tg_buf[lid >> 5] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float total_max = 0.0f;
        for (uint i = 0; i < num_sg; ++i) {
            total_max = max(total_max, tg_buf[i]);
        }
        partial_max[gid] = total_max;
    }
}

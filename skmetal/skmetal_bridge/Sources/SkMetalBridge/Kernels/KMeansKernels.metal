#include <metal_stdlib>
using namespace metal;

// Assignment step: find nearest centroid for each point
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

    for (uint c = 0; c < k; ++c) {
        float dist = 0.0f;
        for (uint j = 0; j < d; ++j) {
            float diff = X[tid * d + j] - centroids[c * d + j];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_k = c;
        }
    }
    assignments[tid] = best_k;
}

// Partial centroid update: each threadgroup accumulates local sums in
// threadgroup (SRAM) memory using CAS, drastically reducing device-level
// atomic contention from O(n) to O(num_threadgroups).
// Output: partial_centroids[num_groups * k * d], partial_counts[num_groups * k].
// Max supported: k * d <= 8000 floats (32KB threadgroup memory).
kernel void kmeans_partial_update(
    device const float* X [[buffer(0)]],
    device const uint* assignments [[buffer(1)]],
    device float* partial_centroids [[buffer(2)]],
    device uint* partial_counts [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& num_groups [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint max_kd = 7168; // fits in 32KB threadgroup memory (28672 + 1024 bytes)
    threadgroup float centroids_shared[7168];
    threadgroup uint counts_shared[256];
    uint kd = min(k * d, max_kd);

    if (lid < kd) centroids_shared[lid] = 0.0f;
    if (lid < k && k <= 256) counts_shared[lid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total_threads = lsz * num_groups;
    for (uint i = tid; i < n; i += total_threads) {
        uint c = assignments[i];
        if (c >= k) continue;

        atomic_fetch_add_explicit(
            (threadgroup atomic_uint*)&counts_shared[c], 1,
            memory_order_relaxed);

        for (uint j = 0; j < d; ++j) {
            float x = X[i * d + j];
            threadgroup atomic_int* addr =
                (threadgroup atomic_int*)&centroids_shared[c * d + j];
            int expected = atomic_load_explicit(addr, memory_order_relaxed);
            int desired;
            do {
                desired = as_type<int>(as_type<float>(expected) + x);
            } while (!atomic_compare_exchange_weak_explicit(
                addr, &expected, desired,
                memory_order_relaxed, memory_order_relaxed));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint offset_c = gid * k * d;
    uint offset_n = gid * k;
    if (lid < k * d) {
        partial_centroids[offset_c + lid] = centroids_shared[lid];
    }
    if (lid < k) {
        partial_counts[offset_n + lid] = counts_shared[lid];
    }
}

// Combine partial centroids from all threadgroups (no atomics).
// Input: partial_centroids[num_groups * k * d], partial_counts[num_groups * k].
// Output: centroids[k * d], counts[k].
kernel void kmeans_combine(
    device const float* partial_centroids [[buffer(0)]],
    device const uint* partial_counts [[buffer(1)]],
    device float* centroids [[buffer(2)]],
    device uint* counts [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& num_groups [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= k) return;
    uint c = tid;
    uint count_total = 0;

    for (uint g = 0; g < num_groups; ++g) {
        count_total += partial_counts[g * k + c];
    }
    counts[c] = count_total;

    for (uint j = 0; j < d; ++j) {
        float sum = 0.0f;
        for (uint g = 0; g < num_groups; ++g) {
            sum += partial_centroids[g * k * d + c * d + j];
        }
        centroids[c * d + j] = sum;
    }
}

// Legacy: direct atomic-based update (for backward compatibility)
kernel void kmeans_update(
    device const float* X [[buffer(0)]],
    device const uint* assignments [[buffer(1)]],
    device float* centroids [[buffer(2)]],
    device atomic_uint* counts [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    uint c = assignments[tid];
    if (c >= k) return;
    atomic_fetch_add_explicit(&counts[c], 1, memory_order_relaxed);
    for (uint j = 0; j < d; ++j) {
        float x = X[tid * d + j];
        device atomic_int* addr = (device atomic_int*)&centroids[c * d + j];
        int expected = atomic_load_explicit(addr, memory_order_relaxed);
        int desired;
        do {
            desired = as_type<int>(as_type<float>(expected) + x);
        } while (!atomic_compare_exchange_weak_explicit(
            addr, &expected, desired,
            memory_order_relaxed, memory_order_relaxed));
    }
}

// Normalize centroids by counts (in-place)
kernel void kmeans_normalize(
    device float* centroids [[buffer(0)]],
    device const uint* counts [[buffer(1)]],
    constant uint& k [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= k * d) return;
    uint c = tid / d;
    uint count = counts[c];
    if (count > 0) {
        centroids[tid] /= (float)count;
    }
}

// Fused combine + normalize: sums partial sums across groups and divides by count.
// Replaces two dispatches (combine + normalize) with one.
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

// Fused assign + partial_accumulate in one dispatch.
// Reads X and centroids (from device memory), writes assignments + per-group partial sums.
// Each group zeros its OWN partial area (no cross-group races).
// Combine+normalize is a separate dispatch (kmeans_combine_normalize).
// Saves 1 dispatch per iteration vs old assign+partial_update+combine pipeline.
kernel void kmeans_assign_partial(
    device const float* X [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device float* partial_centroids [[buffer(2)]],
    device uint* partial_counts [[buffer(3)]],
    device uint* assignments [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& num_groups [[buffer(8)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint max_kd = 7168;
    uint kd = min(k * d, max_kd);
    uint total_threads = lsz * num_groups;

    // Zero this group's area of partial_centroids and partial_counts
    uint pc_off = gid * k * d;
    uint pn_off = gid * k;
    for (uint i = lid; i < kd; i += lsz) partial_centroids[pc_off + i] = 0.0f;
    if (lid < k) partial_counts[pn_off + lid] = 0;

    // Threadgroup accumulation buffers (only ~29KB, fits in 32KB)
    threadgroup float accum[7168];
    threadgroup uint accum_counts[256];
    for (uint i = lid; i < kd; i += lsz) accum[i] = 0.0f;
    if (lid < k) accum_counts[lid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Centroids read from device memory (k*d floats, no threadgroup cache needed)
    for (uint i = tid; i < n; i += total_threads) {
        float min_dist = FLT_MAX;
        uint best_c = 0;
        for (uint c = 0; c < k; ++c) {
            float dist = 0.0f;
            for (uint j = 0; j < d; ++j) {
                float diff = X[i * d + j] - centroids[c * d + j];
                dist += diff * diff;
            }
            if (dist < min_dist) { min_dist = dist; best_c = c; }
        }
        assignments[i] = best_c;

        atomic_fetch_add_explicit((threadgroup atomic_uint*)&accum_counts[best_c],
                                  1, memory_order_relaxed);
        for (uint j = 0; j < d; ++j) {
            float x = X[i * d + j];
            threadgroup atomic_int* a = (threadgroup atomic_int*)&accum[best_c * d + j];
            int expected = atomic_load_explicit(a, memory_order_relaxed);
            int desired;
            do {
                desired = as_type<int>(as_type<float>(expected) + x);
            } while (!atomic_compare_exchange_weak_explicit(
                a, &expected, desired, memory_order_relaxed, memory_order_relaxed));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write this group's partial sums to device memory (no atomics, each group owns its area)
    for (uint i = lid; i < kd; i += lsz) {
        partial_centroids[pc_off + i] = accum[i];
    }
    if (lid < k) {
        partial_counts[pn_off + lid] = accum_counts[lid];
    }
}

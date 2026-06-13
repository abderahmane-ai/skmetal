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

// Partial centroid accumulation: per-threadgroup partial sums for a subset of
// clusters, using threadgroup-local CAS atomics (fast in SRAM).
// Dispatch once per cluster batch; each dispatch handles clusters
// [cluster_start, cluster_start + batch_k).
// After all batches: call kmeans_combine_normalize to reduce across groups.
// Grid: num_groups threadgroups, each with 256 threads.
// TG memory: sum_area 28672 bytes (7168 floats) + count_area 1024 bytes (256 uints).
// batch_k * d must be <= 7168 (enforced at dispatch time).
kernel void kmeans_partial_sum(
    device const float* X [[buffer(0)]],
    device const uint* assignments [[buffer(1)]],
    device float* partial_centroids [[buffer(2)]],
    device uint* partial_counts [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& num_groups [[buffer(6)]],
    constant uint& full_k [[buffer(7)]],
    constant uint& cluster_start [[buffer(8)]],
    constant uint& batch_k [[buffer(9)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float sums[7168];
    threadgroup uint counts[256];
    uint area = batch_k * d;

    for (uint i = lid; i < area; i += lsz) sums[i] = 0.0f;
    if (lid < batch_k) counts[lid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total_threads = lsz * num_groups;
    uint cluster_end = cluster_start + batch_k;

    for (uint i = tid; i < n; i += total_threads) {
        uint c = assignments[i];
        if (c < cluster_start || c >= cluster_end) continue;

        uint local_c = c - cluster_start;
        atomic_fetch_add_explicit(
            (threadgroup atomic_uint*)&counts[local_c], 1,
            memory_order_relaxed);

        for (uint j = 0; j < d; ++j) {
            float x = X[i * d + j];
            threadgroup atomic_int* addr =
                (threadgroup atomic_int*)&sums[local_c * d + j];
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

    uint out_base_c = gid * full_k * d + cluster_start * d;
    uint out_base_n = gid * full_k + cluster_start;
    for (uint i = lid; i < area; i += lsz) {
        partial_centroids[out_base_c + i] = sums[i];
    }
    if (lid < batch_k) {
        partial_counts[out_base_n + lid] = counts[lid];
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

// Fused assign + partial_accumulate in one dispatch (2 dispatches per iteration).
// Each threadgroup zeros its OWN area of partial buffers (no per-iteration blit needed).
// k * d must be <= 7168 (28 KB threadgroup memory) and k <= 256.
// For larger problems, use separate kmeans_assign + batched kmeans_partial_sum.
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
    uint kd = k * d;
    uint total_threads = lsz * num_groups;

    // Zero this group's area of the global partial buffers
    uint pc_off = gid * k * d;
    uint pn_off = gid * k;
    for (uint i = lid; i < kd; i += lsz) partial_centroids[pc_off + i] = 0.0f;
    if (lid < k) partial_counts[pn_off + lid] = 0;

    threadgroup float accum[7168];
    threadgroup uint accum_counts[256];
    for (uint i = lid; i < kd; i += lsz) accum[i] = 0.0f;
    if (lid < k) accum_counts[lid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

        atomic_fetch_add_explicit(
            (threadgroup atomic_uint*)&accum_counts[best_c], 1,
            memory_order_relaxed);
        for (uint j = 0; j < d; ++j) {
            float x = X[i * d + j];
            threadgroup atomic_int* a =
                (threadgroup atomic_int*)&accum[best_c * d + j];
            int expected = atomic_load_explicit(a, memory_order_relaxed);
            int desired;
            do {
                desired = as_type<int>(as_type<float>(expected) + x);
            } while (!atomic_compare_exchange_weak_explicit(
                a, &expected, desired,
                memory_order_relaxed, memory_order_relaxed));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < kd; i += lsz) {
        partial_centroids[pc_off + i] = accum[i];
    }
    if (lid < k) {
        partial_counts[pn_off + lid] = accum_counts[lid];
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

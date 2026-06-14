#include <metal_stdlib>
using namespace metal;

// Tile-local top-k selection for Euclidean distance.
// Uses batch dispatch: each threadgroup processes multiple query rows.
// Each thread handles one query row; stride by total_threads for remaining rows.
kernel void knn_select_tile_topk(
    device const float* raw_dot [[buffer(0)]],
    device const float* r_query [[buffer(1)]],
    device const float* r_train [[buffer(2)]],
    device float* out_vals [[buffer(3)]],
    device int* out_idxs [[buffer(4)]],
    constant uint& n_q [[buffer(5)]],
    constant uint& n_t [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint tgpg [[threadgroups_per_grid]]
) {
    uint total_threads = lsz * tgpg;
    for (uint row = gid * lsz + lid; row < n_q; row += total_threads) {
        device float* my_vals = out_vals + row * k;
        device int* my_idxs = out_idxs + row * k;

        for (uint i = 0; i < k; i++) {
            my_vals[i] = INFINITY;
            my_idxs[i] = 0;
        }

        float r_q = r_query[row];
        device const float* D_row = raw_dot + row * n_t;

        for (uint j = 0; j < n_t; j++) {
            float dist = r_q + r_train[j] - 2.0f * D_row[j];
            if (dist < 0.0f) dist = 0.0f;

            if (dist >= my_vals[k - 1]) continue;

            uint pos = k - 1;
            while (pos > 0 && dist < my_vals[pos - 1]) {
                my_vals[pos] = my_vals[pos - 1];
                my_idxs[pos] = my_idxs[pos - 1];
                pos--;
            }
            my_vals[pos] = dist;
            my_idxs[pos] = int(j);
        }
    }
}

// Majority-vote classification from k-nearest neighbor labels.
// Each thread handles one query point (stride-based dispatch).
kernel void knn_vote_classify(
    device const int* indices [[buffer(0)]],
    device const float* train_labels [[buffer(1)]],
    device float* predictions [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= N) return;

    uint counts[256];
    for (uint j = 0; j < 256; j++) {
        counts[j] = 0;
    }

    for (uint i = 0; i < k; i++) {
        int label = int(train_labels[indices[tid * k + i]]);
        if (label >= 0 && label < 256) {
            counts[label]++;
        }
    }

    uint max_count = 0;
    int best_label = 0;
    for (uint j = 0; j < 256; j++) {
        if (counts[j] > max_count) {
            max_count = counts[j];
            best_label = int(j);
        }
    }
    if (max_count == 0) {
        predictions[tid] = train_labels[indices[tid * k]];
        return;
    }
    predictions[tid] = (float)best_label;
}

// Mean regression from k-nearest neighbor targets.
kernel void knn_vote_regress(
    device const int* indices [[buffer(0)]],
    device const float* train_targets [[buffer(1)]],
    device float* predictions [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    float sum = 0.0f;
    for (uint i = 0; i < k; i++) {
        sum += train_targets[indices[tid * k + i]];
    }
    predictions[tid] = sum / (float)k;
}

// Two-way merge of tile-local top-k into global top-k.
// Both lists sorted ascending by distance. Merges k smallest into global.
// Batch dispatch: each threadgroup processes multiple query rows.
kernel void knn_merge_topk(
    device const float* tile_vals [[buffer(0)]],
    device const int* tile_idxs [[buffer(1)]],
    device float* global_vals [[buffer(2)]],
    device int* global_idxs [[buffer(3)]],
    device float* temp_vals [[buffer(4)]],
    device int* temp_idxs [[buffer(5)]],
    constant uint& n_q [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& tile_start [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint tgpg [[threadgroups_per_grid]]
) {
    uint total_threads = lsz * tgpg;
    for (uint row = gid * lsz + lid; row < n_q; row += total_threads) {
        device const float* a_vals = tile_vals + row * k;
        device const int* a_idxs = tile_idxs + row * k;
        device float* b_vals = global_vals + row * k;
        device int* b_idxs = global_idxs + row * k;
        device float* m_vals = temp_vals + row * k;
        device int* m_idxs = temp_idxs + row * k;

        uint i_a = 0, i_b = 0;
        for (uint s = 0; s < k; s++) {
            if (i_a < k && (i_b >= k || a_vals[i_a] < b_vals[i_b])) {
                m_vals[s] = a_vals[i_a];
                m_idxs[s] = a_idxs[i_a] + int(tile_start);
                i_a++;
            } else {
                m_vals[s] = b_vals[i_b];
                m_idxs[s] = b_idxs[i_b];
                i_b++;
            }
        }

        for (uint s = 0; s < k; s++) {
            b_vals[s] = m_vals[s];
            b_idxs[s] = m_idxs[s];
        }
    }
}

// Manhattan (L1) tile-local top-k: direct L1 distance from X data.
// No GEMM or norms needed. Batch dispatch per threadgroup.
kernel void knn_select_tile_topk_manhattan(
    device const float* X_query [[buffer(0)]],
    device const float* X_train [[buffer(1)]],
    device float* out_vals [[buffer(2)]],
    device int* out_idxs [[buffer(3)]],
    constant uint& n_q [[buffer(4)]],
    constant uint& n_t [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint tgpg [[threadgroups_per_grid]]
) {
    uint total_threads = lsz * tgpg;
    for (uint row = gid * lsz + lid; row < n_q; row += total_threads) {
        device float* my_vals = out_vals + row * k;
        device int* my_idxs = out_idxs + row * k;

        for (uint i = 0; i < k; i++) {
            my_vals[i] = INFINITY;
            my_idxs[i] = 0;
        }

        for (uint j = 0; j < n_t; j++) {
            float dist = 0.0f;
            for (uint dim = 0; dim < d; dim++) {
                dist += fabs(X_query[row * d + dim] - X_train[j * d + dim]);
            }

            if (dist >= my_vals[k - 1]) continue;

            uint pos = k - 1;
            while (pos > 0 && dist < my_vals[pos - 1]) {
                my_vals[pos] = my_vals[pos - 1];
                my_idxs[pos] = my_idxs[pos - 1];
                pos--;
            }
            my_vals[pos] = dist;
            my_idxs[pos] = int(j);
        }
    }
}

// Cosine tile-local top-k: uses MPS dot products + precomputed row norms.
// distance = 1 - dot / (sqrt(rq) * sqrt(rt)). Batch dispatch per threadgroup.
kernel void knn_select_tile_topk_cosine(
    device const float* raw_dot [[buffer(0)]],
    device const float* r_query [[buffer(1)]],
    device const float* r_train [[buffer(2)]],
    device float* out_vals [[buffer(3)]],
    device int* out_idxs [[buffer(4)]],
    constant uint& n_q [[buffer(5)]],
    constant uint& n_t [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsz [[threads_per_threadgroup]],
    uint tgpg [[threadgroups_per_grid]]
) {
    uint total_threads = lsz * tgpg;
    for (uint row = gid * lsz + lid; row < n_q; row += total_threads) {
        device float* my_vals = out_vals + row * k;
        device int* my_idxs = out_idxs + row * k;

        for (uint i = 0; i < k; i++) {
            my_vals[i] = INFINITY;
            my_idxs[i] = 0;
        }

        float rq_sqrt = sqrt(r_query[row] + 1e-10f);
        device const float* D_row = raw_dot + row * n_t;

        for (uint j = 0; j < n_t; j++) {
            float rt_sqrt = sqrt(r_train[j] + 1e-10f);
            float dist = 1.0f - D_row[j] / (rq_sqrt * rt_sqrt);

            if (dist >= my_vals[k - 1]) continue;

            uint pos = k - 1;
            while (pos > 0 && dist < my_vals[pos - 1]) {
                my_vals[pos] = my_vals[pos - 1];
                my_idxs[pos] = my_idxs[pos - 1];
                pos--;
            }
            my_vals[pos] = dist;
            my_idxs[pos] = int(j);
        }
    }
}

// Weighted majority-vote classification using distances.
// Supports up to 256 classes. weight = 1 / (distance + eps).
kernel void knn_vote_classify_weighted(
    device const int* indices [[buffer(0)]],
    device const float* distances [[buffer(1)]],
    device const float* train_labels [[buffer(2)]],
    device float* predictions [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= N) return;

    float weights[256];
    uint labels[256];
    uint n_unique = 0;

    for (uint i = 0; i < k; i++) {
        int idx = indices[tid * k + i];
        float label = train_labels[idx];
        uint l = uint(label);
        float w = 1.0f / (distances[tid * k + i] + 1e-10f);

        bool found = false;
        for (uint j = 0; j < n_unique; j++) {
            if (labels[j] == l) {
                weights[j] += w;
                found = true;
                break;
            }
        }
        if (!found && n_unique < 256) {
            labels[n_unique] = l;
            weights[n_unique] = w;
            n_unique++;
        }
    }

    if (n_unique == 0) {
        predictions[tid] = train_labels[indices[tid * k]];
        return;
    }
    float max_w = 0.0f;
    int best = 0;
    for (uint j = 0; j < n_unique; j++) {
        if (weights[j] > max_w) {
            max_w = weights[j];
            best = int(labels[j]);
        }
    }
    predictions[tid] = (float)best;
}

// Weighted mean regression from k-nearest neighbor targets.
// weight = 1 / (distance + eps), prediction = sum(w * y) / sum(w).
kernel void knn_vote_regress_weighted(
    device const int* indices [[buffer(0)]],
    device const float* distances [[buffer(1)]],
    device const float* train_targets [[buffer(2)]],
    device float* predictions [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= N) return;

    float w_sum = 0.0f;
    float val_sum = 0.0f;
    for (uint i = 0; i < k; i++) {
        int idx = indices[tid * k + i];
        float d = distances[tid * k + i];
        float w = 1.0f / (d + 1e-10f);
        w_sum += w;
        val_sum += w * train_targets[idx];
    }
    predictions[tid] = val_sum / w_sum;
}

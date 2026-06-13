#include <metal_stdlib>
using namespace metal;

// Tree predict: one thread per sample, traverse tree to leaf, accumulate value
kernel void tree_predict(
    device const float* X [[buffer(0)]],
    device const float* tree_values [[buffer(1)]],
    device const int* tree_feature [[buffer(2)]],
    device const float* tree_threshold [[buffer(3)]],
    device const int* tree_left [[buffer(4)]],
    device const int* tree_right [[buffer(5)]],
    device const uint8_t* tree_is_leaf [[buffer(6)]],
    device float* predictions [[buffer(7)]],
    constant uint& n [[buffer(8)]],
    constant uint& n_features [[buffer(9)]],
    constant uint& n_nodes [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    uint node = 0;
    while (true) {
        if (tree_is_leaf[node]) {
            predictions[tid] += tree_values[node];
            return;
        }
        int f = tree_feature[node];
        float x = X[tid * n_features + uint(f)];
        if (x <= tree_threshold[node]) {
            node = uint(tree_left[node]);
        } else {
            node = uint(tree_right[node]);
        }
    }
}

// Multi-tree predict: each thread processes ALL trees for its sample.
// tree_offsets[t] = starting node index in flattened arrays for tree t.
// tree_nodes[t] = number of nodes in tree t.
kernel void tree_predict_all(
    device const float* X [[buffer(0)]],
    device const float* all_tree_values [[buffer(1)]],
    device const int* all_tree_feature [[buffer(2)]],
    device const float* all_tree_threshold [[buffer(3)]],
    device const int* all_tree_left [[buffer(4)]],
    device const int* all_tree_right [[buffer(5)]],
    device const uint8_t* all_tree_is_leaf [[buffer(6)]],
    device const uint* tree_offsets [[buffer(7)]],
    device const uint* tree_n_nodes [[buffer(8)]],
    device const float* baseline [[buffer(9)]],
    device float* predictions [[buffer(10)]],
    constant uint& n [[buffer(11)]],
    constant uint& n_features [[buffer(12)]],
    constant uint& n_trees [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float sum = baseline[0];
    uint base_row = tid * n_features;
    for (uint t = 0; t < n_trees; t++) {
        uint offset = tree_offsets[t];
        uint n_nodes = tree_n_nodes[t];
        uint node = offset;
        uint end = offset + n_nodes;
        while (node < end) {
            if (all_tree_is_leaf[node]) {
                sum += all_tree_values[node];
                break;
            }
            int f = all_tree_feature[node];
            float x = X[base_row + uint(f)];
            if (x <= all_tree_threshold[node]) {
                node = offset + uint(all_tree_left[node]);
            } else {
                node = offset + uint(all_tree_right[node]);
            }
        }
    }
    predictions[tid] = sum;
}



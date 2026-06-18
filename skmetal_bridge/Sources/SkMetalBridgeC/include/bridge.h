#ifndef SKMETAL_BRIDGE_H
#define SKMETAL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int skmetal_init(void);
int skmetal_device_info(char* name, int name_capacity, size_t* max_threads, uint8_t* has_unified_memory, uint64_t* recommended_working_set_size);

int skmetal_gemm(
    const void* A, const void* B, void* C,
    size_t M, size_t N, size_t K,
    float alpha, float beta,
    int transpose_A, int transpose_B
);

int skmetal_reduce_sum(const void* input, void* output, size_t n);
int skmetal_reduce_mean_var(const void* input, void* mean_out, void* var_out, size_t n, float eps);

int skmetal_pairwise_distance(const void* X, void* D, size_t n, size_t d);
int skmetal_row_norm_sq(const void* X, void* norms, size_t n, size_t d);
int skmetal_distance_correct(void* D, const void* X_norm, const void* C_norm, size_t n, size_t k);

int skmetal_argmin_rows(const void* matrix, void* indices, size_t n, size_t k);

int skmetal_scaler_fit(const void* X, void* mean_out, void* var_out, size_t n, size_t d);

int skmetal_column_minmax(const void* X, void* min_out, void* max_out, size_t n, size_t d);

int skmetal_irls_weight(const void* p, void* weights, size_t n);
int skmetal_scale_rows(const void* X, const void* weights, void* output, size_t n, size_t d);

int skmetal_kmeans_assign(const void* X, const void* centroids, void* assignments, size_t n, size_t d, size_t k);
int skmetal_kmeans_combine_normalize(const void* partial_centroids, const void* partial_counts, void* centroids, size_t k, size_t d, size_t num_groups);
int skmetal_kmeans_batch_fused(const void* X, void* centroids, void* assignments, size_t n, size_t d, size_t k, size_t num_groups, size_t max_iter, float tol, int32_t* n_iter_out);

int skmetal_sigmoid(const void* input, void* output, size_t n);
int skmetal_subtract(const void* a, const void* b, void* output, size_t n);
int skmetal_axpy(void* a, const void* b, float alpha, size_t n);

int skmetal_compute_mindists(const void* X, const void* centroids, const void* assignments, void* dists, size_t n, size_t d, size_t k);

int skmetal_center_columns(void* X, const void* mean, size_t n, size_t d);

/* NOTE: X buffer is mean-centered in-place as a side effect. */
int skmetal_ridge_fit(void* X, const void* y, void* XTX, void* XTy, void* X_mean_out, size_t n, size_t p);

int skmetal_knn_k_select(const void* D, void* out_indices, void* out_values, size_t N, size_t M, size_t k);
int skmetal_knn_vote_classify(const void* indices, const void* train_labels, void* predictions, size_t N, size_t k);
int skmetal_knn_vote_regress(const void* indices, const void* train_targets, void* predictions, size_t N, size_t k);

int skmetal_soft_threshold(void* w, const void* w_temp, float threshold, size_t n);
int skmetal_column_transform(const void* input, void* output, const void* center, const void* scale, size_t n, size_t d);

int skmetal_minmax_transform(const void* X, void* X_out, const void* min_vals, const void* max_vals, size_t n, size_t d, float feature_min, float feature_max);
int skmetal_warmup(void);

#ifdef __cplusplus
}
#endif

#endif

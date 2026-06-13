#ifndef SKMETAL_BRIDGE_H
#define SKMETAL_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int skmetal_init(void);
int skmetal_device_info(char** name, size_t* max_threads);

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

int skmetal_kmeans_update(const void* X, const void* assignments, void* centroids, void* counts, size_t n, size_t d, size_t k);
int skmetal_kmeans_assign(const void* X, const void* centroids, void* assignments, size_t n, size_t d, size_t k);
int skmetal_kmeans_partial_update(const void* X, const void* assignments, void* partial_centroids, void* partial_counts, size_t n, size_t d, size_t k, size_t num_groups);
int skmetal_kmeans_combine(const void* partial_centroids, const void* partial_counts, void* centroids, void* counts, size_t k, size_t d, size_t num_groups);
int skmetal_kmeans_normalize(void* centroids, const void* counts, size_t k, size_t d);
int skmetal_kmeans_combine_normalize(const void* partial_centroids, const void* partial_counts, void* centroids, size_t k, size_t d, size_t num_groups);
int skmetal_kmeans_iter(const void* X, const void* centroids_in, void* assignments, void* partial_centroids, void* partial_counts, void* centroids_out, size_t n, size_t d, size_t k, size_t num_groups);
int skmetal_kmeans_batch(const void* X, void* centroids, void* assignments, void* partial_centroids, void* partial_counts, size_t n, size_t d, size_t k, size_t num_groups, size_t max_iter);
int skmetal_kmeans_batch_fused(const void* X, void* centroids, void* assignments, size_t n, size_t d, size_t k, size_t num_groups, size_t max_iter);

int skmetal_svd(const void* A, void* U, void* S, void* Vt, size_t m, size_t n, size_t k);

int skmetal_sigmoid(const void* input, void* output, size_t n);
int skmetal_subtract(const void* a, const void* b, void* output, size_t n);
int skmetal_axpy(void* a, const void* b, float alpha, size_t n);
int skmetal_norm_sq(const void* input, void* output, size_t n);

int skmetal_compute_mindists(const void* X, const void* centroids, const void* assignments, void* dists, size_t n, size_t d, size_t k);

int skmetal_center_columns(void* X, const void* mean, size_t n, size_t d);

int skmetal_ridge_fit(void* X, const void* y, void* XTX, void* XTy, void* X_mean_out, size_t n, size_t p);

int skmetal_logreg_irls_iter(const void* X, const void* y, const void* w, float b, void* linear, void* weight, void* X_scaled, void* Hessian, void* gradient, size_t n, size_t p);

int skmetal_warmup(void);

#ifdef __cplusplus
}
#endif

#endif

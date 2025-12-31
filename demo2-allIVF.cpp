#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <map>
#include <set>
#include <sys/time.h>
#include <omp.h>

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/utils/utils.h>
#include <faiss/impl/AuxIndexStructures.h>

using namespace faiss;

// ==========================================
//  Config: 纯净版参数设置
// ==========================================
struct Config {
    // 路径
    const char* base_path = "sift/sift_base.fvecs";
    const char* query_path = "sift/sift_query.fvecs";
    const char* learn_path = "sift/sift_learn.fvecs";
    const char* gt_path = "sift/sift_groundtruth.ivecs";

    // 1. 索引构建
    int nlist = 15625;           // IVF 聚类数
    int M = 32;                 // HNSW 连边数
    int efConstruction = 16;    // HNSW 建图精度

    // 2. 数据划分
    int n_neighbor_check = 16;
    float alpha = 1.1f;

    // 3. 搜索参数
    int k = 10;                // 用户最终需要的 Top-K
    int efSearch = 40;         // HNSW 搜索广度 (同时也是第一阶段返回的候选点数量)
};

// ==========================================
//  Utils
// ==========================================
float* read_fvecs(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "could not open %s\n", fname); abort(); }
    int d; 
    if (fread(&d, 1, sizeof(int), f) != sizeof(int)) { fprintf(stderr, "read error\n"); abort(); }
    *d_out = d;
    fseek(f, 0, SEEK_END); size_t n = ftell(f) / (sizeof(int) + d * sizeof(float)); fseek(f, 0, SEEK_SET);
    *n_out = n;
    float* x = new float[n * d];
    for (size_t i = 0; i < n; i++) {
        int d_tmp; 
        if (fread(&d_tmp, 1, sizeof(int), f) != sizeof(int)) { fprintf(stderr, "read error\n"); abort(); }
        if (fread(x + i * d, sizeof(float), d, f) != d) { fprintf(stderr, "read error\n"); abort(); }
    }
    fclose(f); return x;
}

int* read_ivecs(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "could not open %s\n", fname); abort(); }
    int d; 
    if (fread(&d, 1, sizeof(int), f) != sizeof(int)) { fprintf(stderr, "read error\n"); abort(); }
    *d_out = d;
    fseek(f, 0, SEEK_END); size_t n = ftell(f) / (sizeof(int) + d * sizeof(int)); fseek(f, 0, SEEK_SET);
    *n_out = n;
    int* x = new int[n * d];
    for (size_t i = 0; i < n; i++) {
        int d_tmp; 
        if (fread(&d_tmp, 1, sizeof(int), f) != sizeof(int)) { fprintf(stderr, "read error\n"); abort(); }
        if (fread(x + i * d, sizeof(int), d, f) != d) { fprintf(stderr, "read error\n"); abort(); }
    }
    fclose(f); return x;
}

// ==========================================
//  Main
// ==========================================
int main() {
    // Force Faiss to use OMP instead of BLAS for distance computations
    faiss::distance_compute_blas_threshold = 2000000000;
    omp_set_num_threads(40);
    Config cfg;
    double t0 = getmillisecs();

    // Load Data
    size_t d, nb, nq, nt, ngt_d, ngt_n;
    float* xb = read_fvecs(cfg.base_path, &d, &nb);
    float* xq = read_fvecs(cfg.query_path, &d, &nq);
    float* xt = read_fvecs(cfg.learn_path, &d, &nt);
    int* gt = read_ivecs(cfg.gt_path, &ngt_d, &ngt_n);

    printf("Data Loaded. Base: %ld, Dim: %ld\n", nb, d);

    // ---------------------------------------------------------
    // 1. Train Quantizer
    // ---------------------------------------------------------
    t0 = getmillisecs();
    IndexFlatL2 quantizer(d);
    IndexIVFFlat index_ivf(&quantizer, d, cfg.nlist, METRIC_L2);
    index_ivf.cp.niter = 10; // Reduce KMeans iterations to speed up training
    index_ivf.verbose = true; // Enable verbose to analyze training speed
    index_ivf.train(nt, xt);
    printf("Training done in %.3f ms\n", getmillisecs() - t0);

    // ---------------------------------------------------------
    // 2. Partition (Full Data in IVF, Loose in HNSW)
    // ---------------------------------------------------------
    t0 = getmillisecs();
    double t_part_start = t0;
    
    std::vector<float> dists(nb * cfg.n_neighbor_check);
    std::vector<idx_t> idxs(nb * cfg.n_neighbor_check);
    
    const size_t batch_size = 65536; 
    for (size_t i = 0; i < nb; i += batch_size) {
        size_t current_batch_size = std::min(batch_size, nb - i);
        quantizer.search(current_batch_size, xb + i * d, 
                          cfg.n_neighbor_check, 
                          dists.data() + i * cfg.n_neighbor_check, 
                          idxs.data() + i * cfg.n_neighbor_check);
    }
    
    double t_part_search = getmillisecs();
    printf("Partitioning Step 1 (Quantizer Search): %.3f ms\n", t_part_search - t_part_start);

    // Parallel Partitioning Logic
    // is_tight_vec 仍然用来判断是否加入 HNSW，但不再阻止加入 IVF
    std::vector<uint8_t> is_tight_vec(nb);
    float alpha_sq = cfg.alpha * cfg.alpha;

    #pragma omp parallel for
    for (size_t i = 0; i < nb; ++i) {
        float d1 = dists[i * cfg.n_neighbor_check + 0];
        float dn = dists[i * cfg.n_neighbor_check + (cfg.n_neighbor_check - 1)];
        // 如果比例过大，说明也是游离的/模糊的，标记为 0 (loose)
        is_tight_vec[i] = ((d1 == 0) || (dn / d1 > alpha_sq)) ? 1 : 0;
    }

    // 计算 HNSW 的 Offsets (IVF 不需要 offset 了，因为是全集 1:1)
    std::vector<size_t> hnsw_offsets(nb);
    size_t loose_count = 0;
    
    for (size_t i = 0; i < nb; ++i) {
        hnsw_offsets[i] = loose_count;
        if (!is_tight_vec[i]) {
            loose_count++; // 只有 loose 点才会计入 HNSW 的计数
        }
    }

    // Allocate Memory
    // 修改：IVF 数组大小改为 nb (全量)
    std::vector<idx_t> ivf_ids(nb);
    std::vector<float> ivf_vecs(nb * d);
    std::vector<idx_t> ivf_assign(nb);
    
    // HNSW 数组大小为 loose_count
    std::vector<idx_t> hnsw_ids(loose_count);
    std::vector<float> hnsw_vecs(loose_count * d);

    // Parallel Copy
    #pragma omp parallel for
    for (size_t i = 0; i < nb; ++i) {
        const float* vec = xb + i * d;

        // 1. 无论是否 loose，全部加入 IVF
        ivf_ids[i] = i;
        ivf_assign[i] = idxs[i * cfg.n_neighbor_check + 0]; // 归属最近的质心
        memcpy(ivf_vecs.data() + i * d, vec, d * sizeof(float));

        // 2. 如果是 loose 点，额外加入 HNSW
        if (!is_tight_vec[i]) {
            size_t idx = hnsw_offsets[i];
            hnsw_ids[idx] = i;
            memcpy(hnsw_vecs.data() + idx * d, vec, d * sizeof(float));
        }
    }

    double t_part_copy = getmillisecs();
    printf("Partitioning Step 2 (Data Copy): %.3f ms\n", t_part_copy - t_part_search);
    printf("Partitioning: IVF (Full)=%ld, HNSW (Loose)=%ld\n", nb, loose_count);

    // ---------------------------------------------------------
    // 3. Build Indices
    // ---------------------------------------------------------
    double t_build_start = getmillisecs();
    
    // Build IVF (使用全量数据)
    index_ivf.add_core(nb, ivf_vecs.data(), ivf_ids.data(), ivf_assign.data());
    
    double t_build_ivf = getmillisecs();
    printf("Build IVF done in %.3f ms\n", t_build_ivf - t_build_start);

    // Build HNSW (Loose + Centroids)
    std::vector<float> centroids(cfg.nlist * d);
    quantizer.reconstruct_n(0, cfg.nlist, centroids.data());
    
    idx_t centroid_mask = 1ULL << 60; 
    std::vector<idx_t> centroid_ids(cfg.nlist);
    for(int i=0; i<cfg.nlist; ++i) centroid_ids[i] = centroid_mask | i;

    // 将质心合并到 HNSW 数据中
    hnsw_ids.insert(hnsw_ids.end(), centroid_ids.begin(), centroid_ids.end());
    hnsw_vecs.insert(hnsw_vecs.end(), centroids.begin(), centroids.end());

    IndexHNSWFlat index_hnsw_storage(d, cfg.M);
    index_hnsw_storage.hnsw.efConstruction = cfg.efConstruction;
    IndexIDMap index_hnsw(&index_hnsw_storage);
    index_hnsw.add_with_ids(hnsw_ids.size(), hnsw_vecs.data(), hnsw_ids.data());

    double t_build_hnsw = getmillisecs();
    printf("Build HNSW done in %.3f ms\n", t_build_hnsw - t_build_ivf);
    printf("Indices Built Total: %.3f ms\n", t_build_hnsw - t_build_start);
    // ---------------------------------------------------------
    // 4. Hybrid Search (Simplified Logic)
    // ---------------------------------------------------------
    printf("OMP Max Threads: %d\n", omp_get_max_threads());
    t0 = getmillisecs();
    double t_start = t0;

    // 核心修改：直接让 efSearch 既控制搜索精度，也控制返回数量
    index_hnsw_storage.hnsw.efSearch = cfg.efSearch;
    int k_hnsw_return = cfg.efSearch; 

    // Step 4.1: HNSW Search
    std::vector<float> hnsw_D(nq * k_hnsw_return);
    std::vector<idx_t> hnsw_I(nq * k_hnsw_return);
    index_hnsw.search(nq, xq, k_hnsw_return, hnsw_D.data(), hnsw_I.data());
    
    double t_hnsw = getmillisecs();
    printf("Step 4.1 HNSW Search: %.3f ms\n", t_hnsw - t_start);

    // Step 4.2: Analyze HNSW results to find Centroids
    std::vector<std::vector<idx_t>> query_clusters(nq);
    size_t max_clusters = 0;

    long long total_centroids = 0;
    long long total_loose = 0;

    for (size_t i = 0; i < nq; ++i) {
        std::set<idx_t> clusters;
        for (int j = 0; j < k_hnsw_return; ++j) {
            idx_t id = hnsw_I[i * k_hnsw_return + j];
            if (id != -1) {
                if (id & centroid_mask) {
                    // Found a centroid! Add this cluster to IVF search target
                    clusters.insert(id & ~centroid_mask);
                    total_centroids++;
                } else {
                    total_loose++;
                }
            }
        }
        query_clusters[i].assign(clusters.begin(), clusters.end());
        if (clusters.size() > max_clusters) max_clusters = clusters.size();
    }
    if (max_clusters == 0) max_clusters = 1;
    
    double t_analyze = getmillisecs();
    printf("Step 4.2 Analysis: %.3f ms (Max Clusters: %ld)\n", t_analyze - t_hnsw, max_clusters);

    // Step 4.3: IVF Search on identified clusters
    std::vector<idx_t> ivf_assign_query(nq * max_clusters, -1);
    std::vector<float> ivf_centroid_dis(nq * max_clusters, 0.0f);

    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < query_clusters[i].size(); ++j) {
            ivf_assign_query[i * max_clusters + j] = query_clusters[i][j];
        }
    }

    IVFSearchParameters ivf_params;
    ivf_params.nprobe = max_clusters; 
    
    std::vector<float> ivf_D(nq * cfg.k);
    std::vector<idx_t> ivf_I(nq * cfg.k);

    // Manual Parallelization of IVF Search
    #pragma omp parallel for
    for (size_t i = 0; i < nq; ++i) {
        index_ivf.search_preassigned(1, xq + i * d, cfg.k, 
                                     ivf_assign_query.data() + i * max_clusters, 
                                     ivf_centroid_dis.data() + i * max_clusters, 
                                     ivf_D.data() + i * cfg.k, 
                                     ivf_I.data() + i * cfg.k, 
                                     false, &ivf_params);
    }
                                 
    double t_ivf = getmillisecs();
    printf("Step 4.3 IVF Search: %.3f ms\n", t_ivf - t_analyze);

    // Step 4.4: Merge & Top-K
    std::vector<idx_t> final_I(nq * cfg.k);
    
    #pragma omp parallel for
    for (size_t i = 0; i < nq; ++i) {
        std::vector<std::pair<float, idx_t>> candidates;
        candidates.reserve(k_hnsw_return + cfg.k);

        // 收集 HNSW 结果中的 Loose Points (排除质心)
        for (int j = 0; j < k_hnsw_return; ++j) {
            idx_t id = hnsw_I[i * k_hnsw_return + j];
            if (id != -1 && !(id & centroid_mask)) {
                candidates.push_back({hnsw_D[i * k_hnsw_return + j], id});
            }
        }

        // 收集 IVF 结果
        for (int j = 0; j < cfg.k; ++j) {
            idx_t id = ivf_I[i * cfg.k + j];
            if (id != -1) {
                candidates.push_back({ivf_D[i * cfg.k + j], id});
            }
        }

        // 排序选出 Top-K
        std::sort(candidates.begin(), candidates.end());
        
        // 去重并填充
        int filled = 0;
        std::set<idx_t> seen;
        for (auto& p : candidates) {
            if (seen.find(p.second) == seen.end()) {
                if (filled < cfg.k) {
                    final_I[i * cfg.k + filled] = p.second;
                    filled++;
                }
                seen.insert(p.second);
            }
            if (filled >= cfg.k) break;
        }
        // 补齐 -1 (如果有的话)
        while (filled < cfg.k) {
            final_I[i * cfg.k + filled] = -1;
            filled++;
        }
    }
    double t_merge = getmillisecs();
    printf("Step 4.4 Merge: %.3f ms\n", t_merge - t_ivf);
    
    printf("Search done in %.3f ms\n", t_merge - t_start);
    printf("HNSW Result Stats (Avg): Centroids=%.2f, Loose Points=%.2f\n", 
        (double)total_centroids / nq, (double)total_loose / nq);

    // ---------------------------------------------------------
    // 5. Recall Check
    // ---------------------------------------------------------
    int n_100 = 0;
    for (int i = 0; i < nq; i++) {
        int gt_idx = gt[i * ngt_d];
        for (int j = 0; j < cfg.k; j++) {
            if (final_I[i * cfg.k + j] == gt_idx) { n_100++; break; }
        }
    }
    printf("R@100: %.4f\n", n_100 / (float)nq);

    delete[] xb; delete[] xq; delete[] xt; delete[] gt;
    return 0;
}
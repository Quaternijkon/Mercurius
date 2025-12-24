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
    int nlist = 1024;           // IVF 聚类数
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
    index_ivf.train(nt, xt);
    printf("Training done in %.3f ms\n", getmillisecs() - t0);

    // ---------------------------------------------------------
    // 2. Partition (Tight vs Loose)
    // ---------------------------------------------------------
    t0 = getmillisecs();
    std::vector<float> dists(nb * cfg.n_neighbor_check);
    std::vector<idx_t> idxs(nb * cfg.n_neighbor_check);
    quantizer.search(nb, xb, cfg.n_neighbor_check, dists.data(), idxs.data());

    std::vector<idx_t> ivf_ids;      ivf_ids.reserve(nb);
    std::vector<float> ivf_vecs;     ivf_vecs.reserve(nb * d);
    std::vector<idx_t> ivf_assign;   ivf_assign.reserve(nb);
    std::vector<idx_t> hnsw_ids;     hnsw_ids.reserve(nb / 10);
    std::vector<float> hnsw_vecs;    hnsw_vecs.reserve(nb * d / 10);

    float alpha_sq = cfg.alpha * cfg.alpha;
    int cluster_count = 0, loose_count = 0;

    for (size_t i = 0; i < nb; ++i) {
        float d1 = dists[i * cfg.n_neighbor_check + 0];
        float dn = dists[i * cfg.n_neighbor_check + (cfg.n_neighbor_check - 1)];
        
        bool is_tight = (d1 == 0) || (dn / d1 > alpha_sq);
        const float* vec = xb + i * d;

        if (is_tight) {
            ivf_ids.push_back(i);
            ivf_assign.push_back(idxs[i * cfg.n_neighbor_check + 0]);
            ivf_vecs.insert(ivf_vecs.end(), vec, vec + d);
            cluster_count++;
        } else {
            hnsw_ids.push_back(i);
            hnsw_vecs.insert(hnsw_vecs.end(), vec, vec + d);
            loose_count++;
        }
    }
    printf("Partitioning: IVF=%d, HNSW=%d\n", cluster_count, loose_count);

    // ---------------------------------------------------------
    // 3. Build Indices
    // ---------------------------------------------------------
    // Build IVF
    index_ivf.add_core(cluster_count, ivf_vecs.data(), ivf_ids.data(), ivf_assign.data());

    // Build HNSW (Loose + Centroids)
    std::vector<float> centroids(cfg.nlist * d);
    quantizer.reconstruct_n(0, cfg.nlist, centroids.data());
    
    idx_t centroid_mask = 1ULL << 60; 
    std::vector<idx_t> centroid_ids(cfg.nlist);
    for(int i=0; i<cfg.nlist; ++i) centroid_ids[i] = centroid_mask | i;

    hnsw_ids.insert(hnsw_ids.end(), centroid_ids.begin(), centroid_ids.end());
    hnsw_vecs.insert(hnsw_vecs.end(), centroids.begin(), centroids.end());

    IndexHNSWFlat index_hnsw_storage(d, cfg.M);
    index_hnsw_storage.hnsw.efConstruction = cfg.efConstruction;
    IndexIDMap index_hnsw(&index_hnsw_storage);
    index_hnsw.add_with_ids(hnsw_ids.size(), hnsw_vecs.data(), hnsw_ids.data());

    printf("Indices Built.\n");

    // ---------------------------------------------------------
    // 4. Hybrid Search (Simplified Logic)
    // ---------------------------------------------------------
    t0 = getmillisecs();

    // 核心修改：直接让 efSearch 既控制搜索精度，也控制返回数量
    index_hnsw_storage.hnsw.efSearch = cfg.efSearch;
    int k_hnsw_return = cfg.efSearch; 

    // Step 4.1: HNSW Search
    std::vector<float> hnsw_D(nq * k_hnsw_return);
    std::vector<idx_t> hnsw_I(nq * k_hnsw_return);
    index_hnsw.search(nq, xq, k_hnsw_return, hnsw_D.data(), hnsw_I.data());

    // Step 4.2: Analyze HNSW results to find Centroids
    std::vector<std::vector<idx_t>> query_clusters(nq);
    size_t max_clusters = 0;

    for (size_t i = 0; i < nq; ++i) {
        std::set<idx_t> clusters;
        for (int j = 0; j < k_hnsw_return; ++j) {
            idx_t id = hnsw_I[i * k_hnsw_return + j];
            if (id != -1 && (id & centroid_mask)) {
                // Found a centroid! Add this cluster to IVF search target
                clusters.insert(id & ~centroid_mask);
            }
        }
        query_clusters[i].assign(clusters.begin(), clusters.end());
        if (clusters.size() > max_clusters) max_clusters = clusters.size();
    }
    if (max_clusters == 0) max_clusters = 1;

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
    
    // 注意：search_preassigned 会在指定的 cluster 里搜，
    // 虽然不会把“所有点”都读出来（那样内存太大了），但会扫描所有点并返回 Top-K。
    // 这里我们请求返回 cfg.k 个结果，用于最后的合并
    std::vector<float> ivf_D(nq * cfg.k);
    std::vector<idx_t> ivf_I(nq * cfg.k);
    index_ivf.search_preassigned(nq, xq, cfg.k, ivf_assign_query.data(), ivf_centroid_dis.data(), 
                                 ivf_D.data(), ivf_I.data(), false, &ivf_params);

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
    printf("Search done in %.3f ms\n", getmillisecs() - t0);

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
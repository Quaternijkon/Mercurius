#include <iostream>
#include <vector>
#include <string>
#include <sstream> 
#include <iomanip> 
#include <fstream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <map>
#include <set>
#include <sys/time.h>
#include <omp.h>
#include <sys/stat.h> 
#include <sys/types.h>

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/utils/utils.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>

using namespace faiss;

// ==========================================
//  Utils
// ==========================================
bool file_exists(const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

void ensure_dir(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        if (mkdir(path.c_str(), 0755) != 0) {
            fprintf(stderr, "Error creating directory: %s\n", path.c_str());
            abort();
        }
    }
}

// ==========================================
//  Config
// ==========================================
struct Config {
    // --- 基础配置 ---
    std::string dataset_name = "sift"; 
    std::string base_path = "sift/sift_base.fvecs";
    std::string query_path = "sift/sift_query.fvecs";
    std::string learn_path = "sift/sift_learn.fvecs";
    std::string gt_path = "sift/sift_groundtruth.ivecs";
    std::string index_dir = "indices"; 

    // --- 1. 索引构建参数 (影响文件名) ---
    int nlist = 15625;
    int M = 32;
    int efConstruction = 40;

    // --- 2. 数据划分参数 (影响文件名) ---
    int n_neighbor_check = 40; 
    float alpha = 1.1f;

    // --- 3. 搜索参数 (运行时可变) ---
    int k = 10;                 
    int efSearch = 100;         
    int nprobe = 64;             

    // --- 生成唯一文件名 ---
    std::string get_params_signature() const {
        std::stringstream ss;
        ss << dataset_name 
           << "_nlist" << nlist 
           << "_M" << M 
           << "_efC" << efConstruction 
           << "_nc" << n_neighbor_check 
           << "_alpha" << std::fixed << std::setprecision(2) << alpha;
        return ss.str();
    }

    std::string get_ivf_path() const { return index_dir + "/" + get_params_signature() + "_ivf.bin"; }
    std::string get_hnsw_path() const { return index_dir + "/" + get_params_signature() + "_hnsw.bin"; }
};

// ... read_fvecs 等 IO 函数保持原样 ...
float* read_fvecs(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "could not open %s\n", fname); abort(); }
    int d; if (fread(&d, 1, sizeof(int), f) != sizeof(int)) abort();
    *d_out = d;
    fseek(f, 0, SEEK_END); size_t n = ftell(f) / (sizeof(int) + d * sizeof(float)); fseek(f, 0, SEEK_SET);
    *n_out = n;
    float* x = new float[n * d];
    for (size_t i = 0; i < n; i++) {
        int d_tmp; fread(&d_tmp, 1, sizeof(int), f);
        fread(x + i * d, sizeof(float), d, f);
    }
    fclose(f); return x;
}

int* read_ivecs(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "could not open %s\n", fname); abort(); }
    int d; fread(&d, 1, sizeof(int), f);
    *d_out = d;
    fseek(f, 0, SEEK_END); size_t n = ftell(f) / (sizeof(int) + d * sizeof(int)); fseek(f, 0, SEEK_SET);
    *n_out = n;
    int* x = new int[n * d];
    for (size_t i = 0; i < n; i++) {
        int d_tmp; fread(&d_tmp, 1, sizeof(int), f);
        fread(x + i * d, sizeof(int), d, f);
    }
    fclose(f); return x;
}

// ==========================================
//  Main
// ==========================================
int main() {
    faiss::distance_compute_blas_threshold = 2000000000;
    omp_set_num_threads(40);
    
    Config cfg;
    // cfg.n_neighbor_check = 24; // 如果需要修改参数

    ensure_dir(cfg.index_dir);
    std::string ivf_path = cfg.get_ivf_path();
    std::string hnsw_path = cfg.get_hnsw_path();

    IndexIVFFlat* index_ivf = nullptr;
    Index* index_hnsw_raw = nullptr;
    IndexIDMap* index_hnsw = nullptr;
    size_t d, nb;

    // =========================================================
    // 检查是否存在
    // =========================================================
    if (file_exists(ivf_path) && file_exists(hnsw_path)) {
        // === LOAD ===
        // 为了保持日志整洁，Load 模式下只打印必要信息，不干扰后续 Search 日志
        printf("[LOAD] Loading existing indices: %s\n", ivf_path.c_str());
        index_ivf = dynamic_cast<IndexIVFFlat*>(read_index(ivf_path.c_str()));
        index_hnsw_raw = read_index(hnsw_path.c_str());
        index_hnsw = dynamic_cast<IndexIDMap*>(index_hnsw_raw);

        d = index_ivf->d;
        nb = index_ivf->ntotal;
        // 不打印 Data Loaded，直接进入 Search
    } else {
        // === BUILD ===
        // 1. Load Data
        size_t nt;
        float* xb = read_fvecs(cfg.base_path.c_str(), &d, &nb);
        float* xt = read_fvecs(cfg.learn_path.c_str(), &d, &nt);
        
        // --- 恢复日志格式: Data Loaded ---
        printf("Data Loaded. Base: %ld, Dim: %ld\n", nb, d);

        // 2. Train
        IndexFlatL2 quantizer(d);
        index_ivf = new IndexIVFFlat(&quantizer, d, cfg.nlist, METRIC_L2);
        index_ivf->cp.niter = 10;
        index_ivf->train(nt, xt);

        // 3. Partition
        std::vector<float> dists(nb * cfg.n_neighbor_check);
        std::vector<idx_t> idxs(nb * cfg.n_neighbor_check);
        const size_t batch_size = 65536; 
        for (size_t i = 0; i < nb; i += batch_size) {
            quantizer.search(std::min(batch_size, nb - i), xb + i * d, cfg.n_neighbor_check, 
                              dists.data() + i * cfg.n_neighbor_check, idxs.data() + i * cfg.n_neighbor_check);
        }

        std::vector<uint8_t> is_tight_vec(nb);
        float alpha_sq = cfg.alpha * cfg.alpha;
        #pragma omp parallel for
        for (size_t i = 0; i < nb; ++i) {
            float d1 = dists[i * cfg.n_neighbor_check + 0];
            float dn = dists[i * cfg.n_neighbor_check + (cfg.n_neighbor_check - 1)];
            is_tight_vec[i] = ((d1 == 0) || (dn / d1 > alpha_sq)) ? 1 : 0;
        }

        size_t loose_count = 0;
        std::vector<size_t> hnsw_offsets(nb);
        for (size_t i = 0; i < nb; ++i) {
            hnsw_offsets[i] = loose_count;
            if (!is_tight_vec[i]) loose_count++;
        }

        std::vector<idx_t> ivf_ids(nb);
        std::vector<float> ivf_vecs(nb * d);
        std::vector<idx_t> ivf_assign(nb);
        std::vector<idx_t> hnsw_ids(loose_count);
        std::vector<float> hnsw_vecs(loose_count * d);

        #pragma omp parallel for
        for (size_t i = 0; i < nb; ++i) {
            ivf_ids[i] = i;
            ivf_assign[i] = idxs[i * cfg.n_neighbor_check + 0];
            memcpy(ivf_vecs.data() + i * d, xb + i * d, d * sizeof(float));
            
            if (!is_tight_vec[i]) {
                size_t idx = hnsw_offsets[i];
                hnsw_ids[idx] = i;
                memcpy(hnsw_vecs.data() + idx * d, xb + i * d, d * sizeof(float));
            }
        }

        // --- 恢复日志格式: Partition ---
        printf("Partition: IVF(Full)=%ld, HNSW(Loose)=%ld\n", nb, loose_count);

        // 4. Build Objects
        index_ivf->add_core(nb, ivf_vecs.data(), ivf_ids.data(), ivf_assign.data());

        std::vector<float> centroids(cfg.nlist * d);
        quantizer.reconstruct_n(0, cfg.nlist, centroids.data());
        idx_t centroid_mask = 1ULL << 60; 
        std::vector<idx_t> centroid_ids(cfg.nlist);
        for(int i=0; i<cfg.nlist; ++i) centroid_ids[i] = centroid_mask | i;

        hnsw_ids.insert(hnsw_ids.end(), centroid_ids.begin(), centroid_ids.end());
        hnsw_vecs.insert(hnsw_vecs.end(), centroids.begin(), centroids.end());

        IndexHNSWFlat* index_hnsw_storage = new IndexHNSWFlat(d, cfg.M);
        index_hnsw_storage->hnsw.efConstruction = cfg.efConstruction;
        index_hnsw = new IndexIDMap(index_hnsw_storage);
        index_hnsw->own_fields = true; 
        index_hnsw->add_with_ids(hnsw_ids.size(), hnsw_vecs.data(), hnsw_ids.data());

        // --- 恢复日志格式: Indices Built ---
        printf("Indices Built.\n");
        
        // Save
        write_index(index_ivf, ivf_path.c_str());
        write_index(index_hnsw, hnsw_path.c_str());
        
        delete[] xb; delete[] xt;
    }

    // =========================================================
    // SEARCH
    // =========================================================
    size_t nq, ngt_d, ngt_n;
    float* xq = read_fvecs(cfg.query_path.c_str(), &d, &nq);
    int* gt = read_ivecs(cfg.gt_path.c_str(), &ngt_d, &ngt_n);
    IndexHNSWFlat* hnsw_storage = dynamic_cast<IndexHNSWFlat*>(index_hnsw->index);

    // --- 恢复日志格式: Hybrid Search 头 ---
    printf("Hybrid Search: efSearch=%d (HNSW), nprobe=%d (IVF Cap)\n", cfg.efSearch, cfg.nprobe);
    
    double t0 = getmillisecs(); // 使用你原变量名习惯
    double t_start = t0;

    // 1. HNSW
    hnsw_storage->hnsw.efSearch = cfg.efSearch;
    int k_hnsw_return = cfg.efSearch; 
    std::vector<float> hnsw_D(nq * k_hnsw_return);
    std::vector<idx_t> hnsw_I(nq * k_hnsw_return);
    index_hnsw->search(nq, xq, k_hnsw_return, hnsw_D.data(), hnsw_I.data());
    
    double t_hnsw = getmillisecs();
    // --- 恢复日志格式: Step 5.1 ---
    printf("Step 5.1 HNSW: %.3f ms\n", t_hnsw - t_start);

    // 2. Analyze
    idx_t centroid_mask = 1ULL << 60;
    std::vector<std::vector<idx_t>> query_clusters(nq);
    size_t max_clusters = 0;
    long long total_centroids_found = 0;    // 统计用
    long long total_centroids_searched = 0; // 统计用

    for (size_t i = 0; i < nq; ++i) {
        std::vector<idx_t> clusters;
        std::set<idx_t> seen;
        clusters.reserve(cfg.nprobe);
        for (int j = 0; j < k_hnsw_return; ++j) {
            idx_t id = hnsw_I[i * k_hnsw_return + j];
            if (id != -1 && (id & centroid_mask)) { 
                idx_t real_id = id & ~centroid_mask;
                total_centroids_found++;
                if (seen.find(real_id) == seen.end()) {
                    if (clusters.size() < cfg.nprobe) {
                        clusters.push_back(real_id);
                        seen.insert(real_id);
                        total_centroids_searched++;
                    }
                }
            }
        }
        query_clusters[i] = clusters;
        if (clusters.size() > max_clusters) max_clusters = clusters.size();
    }
    if (max_clusters == 0) max_clusters = 1;

    double t_analyze = getmillisecs();
    // --- 恢复日志格式: Step 5.2 (注意包含 "to search") ---
    printf("Step 5.2 Analyze: %.3f ms (Max Clusters to search: %ld)\n", t_analyze - t_hnsw, max_clusters);

    // 3. IVF
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
    #pragma omp parallel for
    for (size_t i = 0; i < nq; ++i) {
        index_ivf->search_preassigned(1, xq + i * d, cfg.k, 
                                      ivf_assign_query.data() + i * max_clusters, 
                                      ivf_centroid_dis.data() + i * max_clusters, 
                                      ivf_D.data() + i * cfg.k, 
                                      ivf_I.data() + i * cfg.k, 
                                      false, &ivf_params);
    }
    
    double t_ivf = getmillisecs();
    // --- 恢复日志格式: Step 5.3 ---
    printf("Step 5.3 IVF Search: %.3f ms\n", t_ivf - t_analyze);

    // 4. Merge
    std::vector<idx_t> final_I(nq * cfg.k);
    #pragma omp parallel for
    for (size_t i = 0; i < nq; ++i) {
        std::vector<std::pair<float, idx_t>> candidates;
        candidates.reserve(k_hnsw_return + cfg.k);
        for (int j = 0; j < k_hnsw_return; ++j) {
            idx_t id = hnsw_I[i * k_hnsw_return + j];
            float dist = hnsw_D[i * k_hnsw_return + j];
            if (id != -1 && !(id & centroid_mask)) candidates.push_back({dist, id});
        }
        for (int j = 0; j < cfg.k; ++j) {
            idx_t id = ivf_I[i * cfg.k + j];
            if (id != -1) candidates.push_back({ivf_D[i * cfg.k + j], id});
        }
        std::sort(candidates.begin(), candidates.end());
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
        while (filled < cfg.k) final_I[i * cfg.k + filled++] = -1;
    }

    double t_merge = getmillisecs();
    // --- 恢复日志格式: Step 5.4 ---
    printf("Step 5.4 Merge: %.3f ms\n", t_merge - t_ivf);
    
    // --- 恢复日志格式: Total Search Time ---
    printf("Total Search Time: %.3f ms\n", t_merge - t_start);

    // --- 恢复日志格式: Centroid Stats ---
    printf("Centroid Stats (Avg): HNSW Found=%.2f, IVF Searched=%.2f\n", 
           (double)total_centroids_found / nq, (double)total_centroids_searched / nq);

    // Eval
    int n_100 = 0;
    for (int i = 0; i < nq; i++) {
        int gt_idx = gt[i * ngt_d];
        for (int j = 0; j < cfg.k; j++) {
            if (final_I[i * cfg.k + j] == gt_idx) { n_100++; break; }
        }
    }
    // --- 恢复日志格式: R@... ---
    printf("R@%d: %.4f\n", cfg.k, n_100 / (float)nq);

    delete index_ivf; 
    delete index_hnsw; 
    delete[] xq; delete[] gt;
    return 0;
}
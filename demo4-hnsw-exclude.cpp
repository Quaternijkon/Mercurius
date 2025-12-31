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

struct Config {
    // --- IO ---
    std::string dataset_name = "sift"; 
    std::string base_path = "sift/sift_base.fvecs";
    std::string query_path = "sift/sift_query.fvecs";
    std::string learn_path = "sift/sift_learn.fvecs";
    std::string gt_path = "sift/sift_groundtruth.ivecs";
    std::string index_dir = "indices"; 

    // --- Build Params ---
    int nlist = 15625;
    int M = 32;
    int efConstruction = 40;
    
    // 训练时的搜索精度，低精度以加速训练
    int efSearch_train = 16; 

    // --- Partition Params ---
    // 必须 >= 2 才能计算 d_n / d_1 的比值
    int n_neighbor_check = 40; 
    float alpha = 1.1f;       

    // --- Search Params ---
    int k = 10;                 
    int efSearch = 100;         
    int nprobe = 64;             

    std::string get_params_signature() const {
        std::stringstream ss;
        ss << dataset_name << "_nlist" << nlist << "_M" << M << "_opt_deferred";
        return ss.str();
    }
    std::string get_ivf_path() const { return index_dir + "/" + get_params_signature() + "_ivf.bin"; }
    std::string get_hnsw_path() const { return index_dir + "/" + get_params_signature() + "_hnsw.bin"; }
};

// ... (这里请插入之前的 read_fvecs, read_ivecs, file_exists, ensure_dir 实现) ...
// 为了代码完整性，我再次提供 read_fvecs 等核心 IO，确保你可以直接编译
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
    Config cfg;
    ensure_dir(cfg.index_dir);
    
    std::string ivf_path = cfg.get_ivf_path();
    std::string hnsw_path = cfg.get_hnsw_path();

    IndexIVFFlat* index_ivf = nullptr;
    Index* index_hnsw_raw = nullptr;
    IndexIDMap* index_hnsw = nullptr;
    size_t d, nb;

    if (file_exists(ivf_path) && file_exists(hnsw_path)) {
        printf("[LOAD] Loading existing indices...\n");
        index_ivf = dynamic_cast<IndexIVFFlat*>(read_index(ivf_path.c_str()));
        index_hnsw_raw = read_index(hnsw_path.c_str());
        index_hnsw = dynamic_cast<IndexIDMap*>(index_hnsw_raw);
        d = index_ivf->d;
        nb = index_ivf->ntotal; 
    } else {
        // === BUILD PHASE ===
        size_t nt;
        float* xb = read_fvecs(cfg.base_path.c_str(), &d, &nb);
        float* xt = read_fvecs(cfg.learn_path.c_str(), &d, &nt);
        
        printf("Data Loaded. Base: %ld, Dim: %ld\n", nb, d);

        // 1. Train IVF (Fast Mode)
        IndexHNSWFlat quantizer_for_train(d, cfg.M);
        quantizer_for_train.hnsw.efConstruction = cfg.efConstruction;
        quantizer_for_train.hnsw.efSearch = cfg.efSearch_train; 

        index_ivf = new IndexIVFFlat(&quantizer_for_train, d, cfg.nlist, METRIC_L2);
        index_ivf->cp.niter = 10;
        index_ivf->verbose = true;
        
        printf("Training IVF...\n");
        double t_train_start = getmillisecs();
        index_ivf->train(nt, xt);
        printf("Training finished in %.3f ms\n", getmillisecs() - t_train_start);

        // 2. Build Navigation Index (Only Centroids)
        printf("Building Navigation Index (Centroids Only)...\n");
        std::vector<float> centroids(cfg.nlist * d);
        index_ivf->quantizer->reconstruct_n(0, cfg.nlist, centroids.data());

        idx_t centroid_mask = 1ULL << 60; 
        std::vector<idx_t> centroid_ids(cfg.nlist);
        for(int i=0; i<cfg.nlist; ++i) centroid_ids[i] = centroid_mask | i;

        IndexHNSWFlat* hnsw_impl = new IndexHNSWFlat(d, cfg.M);
        hnsw_impl->hnsw.efConstruction = cfg.efConstruction;
        index_hnsw = new IndexIDMap(hnsw_impl);
        index_hnsw->own_fields = true;
        index_hnsw->add_with_ids(cfg.nlist, centroids.data(), centroid_ids.data());

        // 3. Unified Search (One Search, Two Purposes)
        // 这里的搜索结果将被复用：既用于 IVF 分发，也用于游离点判定
        printf("Unified Search (Calculating assignments & distances)...\n");
        double t_search_start = getmillisecs();

        IndexHNSWFlat* hnsw_searcher = dynamic_cast<IndexHNSWFlat*>(index_hnsw->index);
        hnsw_searcher->hnsw.efSearch = cfg.efConstruction; // High precision for distribution

        // 分配内存存储搜索结果 (Top-40)
        std::vector<float> dists(nb * cfg.n_neighbor_check);
        std::vector<idx_t> idxs(nb * cfg.n_neighbor_check);

        // 原生并行搜索
        const size_t batch_size = 65536; 
        for (size_t i = 0; i < nb; i += batch_size) {
            index_hnsw->search(std::min(batch_size, nb - i), xb + i * d, cfg.n_neighbor_check, 
                               dists.data() + i * cfg.n_neighbor_check, idxs.data() + i * cfg.n_neighbor_check);
        }
        printf("Unified Search finished in %.3f ms\n", getmillisecs() - t_search_start);

        // 4. Build IVF (Pure Add, No Search)
        // 在这一步，我们利用搜索结果填充 IVF，此时 HNSW 仍然只包含质心！
        printf("Adding data to IVF (using pre-computed assignments)...\n");
        double t_ivf_add = getmillisecs();

        // 提取最近的质心 ID (Top-1)，并去除 Mask
        std::vector<idx_t> assignments(nb);
        #pragma omp parallel for
        for(size_t i=0; i<nb; ++i) {
            assignments[i] = idxs[i * cfg.n_neighbor_check + 0] & ~centroid_mask;
        }

        // 关键调用：add_core
        // 此时数据被加入 IVF，但 HNSW 索引保持不变
        index_ivf->add_core(nb, xb, nullptr, assignments.data());
        
        printf("IVF Build finished in %.3f ms. (HNSW is still pure)\n", getmillisecs() - t_ivf_add);

        // 5. Update HNSW (Delayed Loose Point Addition)
        // IVF 构建完了，现在我们回过头来利用刚才的 dists 筛选游离点
        printf("Filtering & Updating HNSW with Loose Points...\n");
        double t_loose_start = getmillisecs();
        
        std::vector<idx_t> loose_ids;
        // 预估大小，避免频繁 realloc
        loose_ids.reserve(nb * 0.05); 
        float alpha_sq = cfg.alpha * cfg.alpha;

        for (size_t i = 0; i < nb; ++i) {
            float d1 = dists[i * cfg.n_neighbor_check + 0];
            float dn = dists[i * cfg.n_neighbor_check + (cfg.n_neighbor_check - 1)];
            
            // 复用之前的搜索结果进行判定
            if ((d1 == 0) || (dn / d1 < alpha_sq)) {
                loose_ids.push_back(i);
            }
        }
        
        // 提取游离点向量
        size_t loose_count = loose_ids.size();
        std::vector<float> loose_vecs(loose_count * d);
        #pragma omp parallel for
        for (size_t i = 0; i < loose_count; ++i) {
            memcpy(loose_vecs.data() + i * d, xb + loose_ids[i] * d, d * sizeof(float));
        }

        // 只有在这里，IVF 完工后，才污染 HNSW
        if (loose_count > 0) {
            index_hnsw->add_with_ids(loose_count, loose_vecs.data(), loose_ids.data());
        }
        
        printf("HNSW Updated. Loose Points: %ld (%.2f%%). Time: %.3f ms\n", 
               loose_count, (double)loose_count/nb*100.0, getmillisecs() - t_loose_start);

        printf("Indices Built. Saving...\n");
        write_index(index_ivf, ivf_path.c_str());
        write_index(index_hnsw, hnsw_path.c_str());
        
        delete[] xb; delete[] xt;
    }

    // =========================================================
    // SEARCH PHASE (保持不变)
    // =========================================================
    size_t nq, ngt_d, ngt_n;
    float* xq = read_fvecs(cfg.query_path.c_str(), &d, &nq);
    int* gt = read_ivecs(cfg.gt_path.c_str(), &ngt_d, &ngt_n);
    IndexHNSWFlat* hnsw_storage = dynamic_cast<IndexHNSWFlat*>(index_hnsw->index);

    printf("Hybrid Search: efSearch=%d, nprobe=%d\n", cfg.efSearch, cfg.nprobe);
    double t_start = getmillisecs();

    // 1. HNSW Search
    hnsw_storage->hnsw.efSearch = cfg.efSearch;
    int k_hnsw_return = cfg.efSearch; 
    std::vector<float> hnsw_D(nq * k_hnsw_return);
    std::vector<idx_t> hnsw_I(nq * k_hnsw_return);
    index_hnsw->search(nq, xq, k_hnsw_return, hnsw_D.data(), hnsw_I.data());
    
    double t_hnsw = getmillisecs();
    printf("Step 1 HNSW: %.3f ms\n", t_hnsw - t_start);

    // 2. Analyze
    idx_t centroid_mask = 1ULL << 60;
    std::vector<std::vector<idx_t>> query_clusters(nq);
    size_t max_clusters = 0;
    
    for (size_t i = 0; i < nq; ++i) {
        std::vector<idx_t> clusters;
        std::set<idx_t> seen;
        clusters.reserve(cfg.nprobe);
        for (int j = 0; j < k_hnsw_return; ++j) {
            idx_t id = hnsw_I[i * k_hnsw_return + j];
            if (id != -1 && (id & centroid_mask)) { 
                idx_t real_id = id & ~centroid_mask;
                if (seen.find(real_id) == seen.end()) {
                    if (clusters.size() < cfg.nprobe) {
                        clusters.push_back(real_id);
                        seen.insert(real_id);
                    }
                }
            }
        }
        query_clusters[i] = clusters;
        if (clusters.size() > max_clusters) max_clusters = clusters.size();
    }
    if (max_clusters == 0) max_clusters = 1;

    // 3. IVF Search
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
    printf("Step 3 IVF Search: %.3f ms\n", t_ivf - t_hnsw);

    // 4. Merge
    std::vector<idx_t> final_I(nq * cfg.k);
    #pragma omp parallel for
    for (size_t i = 0; i < nq; ++i) {
        std::vector<std::pair<float, idx_t>> candidates;
        candidates.reserve(k_hnsw_return + cfg.k);
        for (int j = 0; j < k_hnsw_return; ++j) {
            idx_t id = hnsw_I[i * k_hnsw_return + j];
            float dist = hnsw_D[i * k_hnsw_return + j];
            if (id != -1 && !(id & centroid_mask)) {
                candidates.push_back({dist, id});
            }
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
    printf("Step 4 Merge: %.3f ms\n", t_merge - t_ivf);
    printf("Total Search Time: %.3f ms\n", t_merge - t_start);
    
    // Eval
    int n_100 = 0;
    for (int i = 0; i < nq; i++) {
        int gt_idx = gt[i * ngt_d];
        for (int j = 0; j < cfg.k; j++) {
            if (final_I[i * cfg.k + j] == gt_idx) { n_100++; break; }
        }
    }
    printf("R@%d: %.4f\n", cfg.k, n_100 / (float)nq);

    delete index_ivf; 
    delete index_hnsw; 
    delete[] xq; delete[] gt;
    return 0;
}
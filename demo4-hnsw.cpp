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

    // --- 1. 索引构建参数 ---
    int nlist = 15625;
    int M = 32;
    int efConstruction = 40;

    // --- 2. 数据划分参数 ---
    int n_neighbor_check = 40; 
    float alpha = 1.1f;

    // --- 3. 搜索参数 ---
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

// ... IO 函数保持原样 ...
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
        printf("[LOAD] Loading existing indices: %s\n", ivf_path.c_str());
        index_ivf = dynamic_cast<IndexIVFFlat*>(read_index(ivf_path.c_str()));
        index_hnsw_raw = read_index(hnsw_path.c_str());
        index_hnsw = dynamic_cast<IndexIDMap*>(index_hnsw_raw);

        d = index_ivf->d;
        nb = index_ivf->ntotal; // 注意：这里只统计了IVF中的点，加上HNSW中的游离点才是真正的总数，但为了搜索流程这里d最重要
    } else {
        // === BUILD ===
        // 1. Load Data
        size_t nt;
        float* xb = read_fvecs(cfg.base_path.c_str(), &d, &nb);
        float* xt = read_fvecs(cfg.learn_path.c_str(), &d, &nt);
        
        printf("Data Loaded. Base: %ld, Dim: %ld\n", nb, d);

        // 2. Train (使用 HNSW 作为 Quantizer 加速训练)
        // 关键修改：这里使用 IndexHNSWFlat 替代 IndexFlatL2
        IndexHNSWFlat quantizer_for_train(d, cfg.M);
        quantizer_for_train.hnsw.efConstruction = cfg.efConstruction;
        quantizer_for_train.hnsw.efSearch = cfg.efConstruction; // 训练时精度高一点

        index_ivf = new IndexIVFFlat(&quantizer_for_train, d, cfg.nlist, METRIC_L2);
        index_ivf->cp.niter = 10;
        index_ivf->verbose = true; // 开启日志看训练进度
        
        printf("Training IVF with HNSW quantizer...\n");
        double t_train_start = getmillisecs();
        index_ivf->train(nt, xt);
        double t_train_end = getmillisecs();
        printf("Training finished in %.3f ms\n", t_train_end - t_train_start);

        // 3. 构建用于 Partition 的 HNSW 索引（包含质心）
        printf("Building Navigation HNSW Index with Centroids...\n");
        
        // 3.1 提取质心
        std::vector<float> centroids(cfg.nlist * d);
        index_ivf->quantizer->reconstruct_n(0, cfg.nlist, centroids.data());

        // 3.2 准备质心 ID (Masked)
        idx_t centroid_mask = 1ULL << 60; 
        std::vector<idx_t> centroid_ids(cfg.nlist);
        for(int i=0; i<cfg.nlist; ++i) centroid_ids[i] = centroid_mask | i;

        // 3.3 创建并填充 HNSW 索引
        IndexHNSWFlat* hnsw_impl = new IndexHNSWFlat(d, cfg.M);
        hnsw_impl->hnsw.efConstruction = cfg.efConstruction;
        index_hnsw = new IndexIDMap(hnsw_impl);
        index_hnsw->own_fields = true;
        
        // 将质心加入 HNSW
        index_hnsw->add_with_ids(cfg.nlist, centroids.data(), centroid_ids.data());

        // 4. Partition (使用 HNSW 索引加速划分)
        printf("Partitioning data...\n");
        double t_part_start = getmillisecs();

        // 预分配数组
        std::vector<float> dists(nb * cfg.n_neighbor_check);
        std::vector<idx_t> idxs(nb * cfg.n_neighbor_check);
        
        // 使用构建好的 index_hnsw 进行搜索，而不是原来的 Flat quantizer
        // index_hnsw 目前只包含质心，所以搜索结果就是最近的质心
        IndexHNSWFlat* hnsw_storage_build = dynamic_cast<IndexHNSWFlat*>(index_hnsw->index);
        hnsw_storage_build->hnsw.efSearch = cfg.efConstruction; // 保持较高的构建精度

        const size_t batch_size = 65536; 
        for (size_t i = 0; i < nb; i += batch_size) {
            index_hnsw->search(std::min(batch_size, nb - i), xb + i * d, cfg.n_neighbor_check, 
                               dists.data() + i * cfg.n_neighbor_check, idxs.data() + i * cfg.n_neighbor_check);
        }

        // 识别 Loose Points
        std::vector<uint8_t> is_tight_vec(nb);
        float alpha_sq = cfg.alpha * cfg.alpha;
        
        #pragma omp parallel for
        for (size_t i = 0; i < nb; ++i) {
            float d1 = dists[i * cfg.n_neighbor_check + 0];
            float dn = dists[i * cfg.n_neighbor_check + (cfg.n_neighbor_check - 1)];
            is_tight_vec[i] = ((d1 == 0) || (dn / d1 > alpha_sq)) ? 1 : 0;
        }

        // 计算 Offset 以便并行填充
        size_t loose_count = 0;
        std::vector<size_t> hnsw_offsets(nb);
        for (size_t i = 0; i < nb; ++i) {
            hnsw_offsets[i] = loose_count;
            if (!is_tight_vec[i]) loose_count++;
        }

        std::vector<idx_t> ivf_ids(nb);
        std::vector<float> ivf_vecs(nb * d);
        std::vector<idx_t> ivf_assign(nb);
        
        std::vector<idx_t> loose_ids(loose_count);
        std::vector<float> loose_vecs(loose_count * d);

        #pragma omp parallel for
        for (size_t i = 0; i < nb; ++i) {
            // 无论 loose 还是 tight，原始 ID 都是 i
            // 注意：Search 结果 idxs 里的 ID 带有 mask，我们需要去掉 mask 才能作为 assign id
            idx_t masked_centroid_id = idxs[i * cfg.n_neighbor_check + 0];
            idx_t centroid_id = masked_centroid_id & ~centroid_mask;

            ivf_ids[i] = i;
            ivf_assign[i] = centroid_id; 
            memcpy(ivf_vecs.data() + i * d, xb + i * d, d * sizeof(float));
            
            if (!is_tight_vec[i]) {
                size_t idx = hnsw_offsets[i];
                loose_ids[idx] = i; // 游离点使用原始 ID
                memcpy(loose_vecs.data() + idx * d, xb + i * d, d * sizeof(float));
            }
        }
        double t_part_end = getmillisecs();
        printf("Partition finished in %.3f ms. IVF(Full)=%ld, HNSW(Loose)=%ld\n", 
               t_part_end - t_part_start, nb, loose_count);

        // 5. Build Final Structures
        // 5.1 添加所有点到 IVF (策略不变，IVF 包含全量数据的 tight 视角)
        // 注意：index_ivf 的 quantizer 仍然是 HNSW 类型的，但这不影响 add_core，因为它直接操作倒排链表
        index_ivf->add_core(nb, ivf_vecs.data(), ivf_ids.data(), ivf_assign.data());

        // 5.2 添加 Loose points 到 HNSW
        // 此时 index_hnsw 已经包含了质心，现在追加游离点
        if (loose_count > 0) {
            index_hnsw->add_with_ids(loose_count, loose_vecs.data(), loose_ids.data());
        }

        printf("Indices Built.\n");
        
        // Save
        // 注意：index_ivf 保存时会序列化它的 quantizer。
        // 由于我们在 train 时用的是局部变量 quantizer_for_train，
        // index_ivf 内部拷贝了一份。这个 quantizer 是 HNSW 类型。
        // 这对后续 Load 没问题，只要 Load 时候正确识别即可。
        write_index(index_ivf, ivf_path.c_str());
        write_index(index_hnsw, hnsw_path.c_str());
        
        delete[] xb; delete[] xt;
    }

    // =========================================================
    // SEARCH (逻辑保持不变)
    // =========================================================
    size_t nq, ngt_d, ngt_n;
    float* xq = read_fvecs(cfg.query_path.c_str(), &d, &nq);
    int* gt = read_ivecs(cfg.gt_path.c_str(), &ngt_d, &ngt_n);
    IndexHNSWFlat* hnsw_storage = dynamic_cast<IndexHNSWFlat*>(index_hnsw->index);

    printf("Hybrid Search: efSearch=%d (HNSW), nprobe=%d (IVF Cap)\n", cfg.efSearch, cfg.nprobe);
    
    double t0 = getmillisecs();
    double t_start = t0;

    // 1. HNSW Search (搜质心 + 游离点)
    hnsw_storage->hnsw.efSearch = cfg.efSearch;
    int k_hnsw_return = cfg.efSearch; 
    std::vector<float> hnsw_D(nq * k_hnsw_return);
    std::vector<idx_t> hnsw_I(nq * k_hnsw_return);
    index_hnsw->search(nq, xq, k_hnsw_return, hnsw_D.data(), hnsw_I.data());
    
    double t_hnsw = getmillisecs();
    printf("Step 5.1 HNSW: %.3f ms\n", t_hnsw - t_start);

    // 2. Analyze (分离 质心结果 和 游离点结果)
    idx_t centroid_mask = 1ULL << 60;
    std::vector<std::vector<idx_t>> query_clusters(nq);
    size_t max_clusters = 0;
    long long total_centroids_found = 0;    
    long long total_centroids_searched = 0; 

    for (size_t i = 0; i < nq; ++i) {
        std::vector<idx_t> clusters;
        std::set<idx_t> seen;
        clusters.reserve(cfg.nprobe);
        for (int j = 0; j < k_hnsw_return; ++j) {
            idx_t id = hnsw_I[i * k_hnsw_return + j];
            if (id != -1 && (id & centroid_mask)) { // 这是一个质心
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
            // 如果不是 mask，说明是游离点，将在 Merge 阶段处理
        }
        query_clusters[i] = clusters;
        if (clusters.size() > max_clusters) max_clusters = clusters.size();
    }
    if (max_clusters == 0) max_clusters = 1;

    double t_analyze = getmillisecs();
    printf("Step 5.2 Analyze: %.3f ms (Max Clusters to search: %ld)\n", t_analyze - t_hnsw, max_clusters);

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
    printf("Step 5.3 IVF Search: %.3f ms\n", t_ivf - t_analyze);

    // 4. Merge (HNSW中的游离点 + IVF中的紧密点)
    std::vector<idx_t> final_I(nq * cfg.k);
    #pragma omp parallel for
    for (size_t i = 0; i < nq; ++i) {
        std::vector<std::pair<float, idx_t>> candidates;
        candidates.reserve(k_hnsw_return + cfg.k);
        
        // 收集 HNSW 结果中的 游离点
        for (int j = 0; j < k_hnsw_return; ++j) {
            idx_t id = hnsw_I[i * k_hnsw_return + j];
            float dist = hnsw_D[i * k_hnsw_return + j];
            // 只有 不带 centroid_mask 的 ID 才是真实数据点（游离点）
            if (id != -1 && !(id & centroid_mask)) {
                candidates.push_back({dist, id});
            }
        }
        // 收集 IVF 结果
        for (int j = 0; j < cfg.k; ++j) {
            idx_t id = ivf_I[i * cfg.k + j];
            if (id != -1) candidates.push_back({ivf_D[i * cfg.k + j], id});
        }
        
        // 排序并去重
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
    printf("Step 5.4 Merge: %.3f ms\n", t_merge - t_ivf);
    printf("Total Search Time: %.3f ms\n", t_merge - t_start);
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
    printf("R@%d: %.4f\n", cfg.k, n_100 / (float)nq);

    delete index_ivf; 
    delete index_hnsw; 
    delete[] xq; delete[] gt;
    return 0;
}
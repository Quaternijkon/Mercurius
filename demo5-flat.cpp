#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <regex>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h> // 核心依赖：Flat索引
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/io.h>
#include <faiss/impl/FaissAssert.h>
#include <omp.h>

using namespace std;
using namespace faiss;

// ============================================================================
// Utils
// ============================================================================
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void ensure_dir(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

pair<size_t, size_t> get_fbin_meta(const string& filename) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) throw runtime_error("Cannot open file: " + filename);
    int32_t nvecs_raw, dim_raw;
    f.read(reinterpret_cast<char*>(&nvecs_raw), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim_raw), sizeof(int32_t));
    return {static_cast<size_t>(nvecs_raw), static_cast<size_t>(dim_raw)};
}

pair<vector<float>, pair<size_t, size_t>> read_fbin(const string& filename, size_t start_idx = 0, size_t chunk_size = 0) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) throw runtime_error("Cannot open file: " + filename);
    int32_t nvecs_raw, dim_raw;
    f.read(reinterpret_cast<char*>(&nvecs_raw), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim_raw), sizeof(int32_t));
    size_t nvecs = static_cast<size_t>(nvecs_raw);
    size_t dim = static_cast<size_t>(dim_raw);

    size_t num_vectors_in_chunk = nvecs;
    if (chunk_size > 0) {
        size_t end_idx = min(start_idx + chunk_size, nvecs);
        num_vectors_in_chunk = end_idx - start_idx;
        if (num_vectors_in_chunk == 0) return {{}, {0, dim}};
        size_t offset = 8 + start_idx * dim * sizeof(float);
        f.seekg(offset, ios::beg);
    }
    vector<float> data(num_vectors_in_chunk * dim);
    f.read(reinterpret_cast<char*>(data.data()), num_vectors_in_chunk * dim * sizeof(float));
    return {data, {num_vectors_in_chunk, dim}};
}

vector<vector<int32_t>> read_ivecs(const string& filename) {
    ifstream f(filename, ios::binary | ios::ate);
    if (!f.is_open()) throw runtime_error("Cannot open file: " + filename);
    size_t file_size = f.tellg();
    f.seekg(0, ios::beg);
    vector<int32_t> a(file_size / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(a.data()), file_size);
    int32_t d = a[0];
    vector<vector<int32_t>> result(a.size() / (d + 1));
    for (size_t i = 0, j = 0; i < result.size(); ++i) {
        result[i].resize(d);
        ++j; copy(a.begin() + j, a.begin() + j + d, result[i].begin()); j += d;
    }
    return result;
}

// Config
const string DATA_DIR = "./sift";
const string LEARN_FILE = DATA_DIR + "/learn.fbin";
const string BASE_FILE = DATA_DIR + "/base.fbin";
const string QUERY_FILE = DATA_DIR + "/query.fbin";
const string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

const int N_LIST = 15625;          
// [MODIFIED] Removed HNSW params (M_HNSW, EF_CONSTRUCTION, EF_SEARCH_TRAIN)
const int NEIGHBOR_CHECK = 40;     
const float ALPHA = 1.1f;          
const int SEARCH_K = 10;
const int SEARCH_EF = 100; // 在Flat模式下，这代表我们扫描前100个最近邻来寻找质心或游离点
const int SEARCH_NPROBE = 64;

// ============================================================================
// Main
// ============================================================================
int main() {
    omp_set_num_threads(40);
    ensure_dir("indices");

    auto [nb, d] = get_fbin_meta(BASE_FILE);
    cout << "Dataset Info: nb=" << nb << ", d=" << d << endl;

    string base_name = BASE_FILE.substr(BASE_FILE.find_last_of("/") + 1);
    string clean_base_name = regex_replace(base_name.substr(0, base_name.find_last_of(".")), regex("[^a-zA-Z0-9_]"), "_");
    
    // [MODIFIED] Update filenames to reflect FLAT structure
    string IVF_FILE = "indices/" + clean_base_name + "_IVF_FLAT_final.index";

    IndexIVFFlat* index_ivf = nullptr;
    IndexIDMap* index_flat_top = nullptr; // [MODIFIED] Renamed from index_hnsw
    IndexFlatL2* flat_impl = nullptr;     // [MODIFIED] Renamed from hnsw_impl, underlying flat index
    IndexFlatL2* quantizer_train = nullptr; // [MODIFIED] Quantizer is now Flat

    // 1. Training
    {
        cout << "Loading Learn data..." << endl;
        auto [xt_vec, meta_learn] = read_fbin(LEARN_FILE);
        size_t nt = meta_learn.first;
        float* xt = xt_vec.data();

        cout << "Training IVF (Flat Quantizer)..." << endl;
        // [MODIFIED] Use IndexFlatL2 for quantizer
        quantizer_train = new IndexFlatL2(d); 
        
        index_ivf = new IndexIVFFlat(quantizer_train, d, N_LIST, METRIC_L2);
        index_ivf->cp.niter = 10; 
        index_ivf->verbose = true;

        double t_start = get_time_ms();
        index_ivf->train(nt, xt);
        cout << "Training finished in " << get_time_ms() - t_start << " ms." << endl;
    }

    // 2. Init Top-Level Flat Index
    cout << "Initializing Top-Level Flat with Centroids..." << endl;
    {
        vector<float> centroids(N_LIST * d);
        index_ivf->quantizer->reconstruct_n(0, N_LIST, centroids.data());

        idx_t centroid_mask = 1ULL << 60;
        vector<idx_t> centroid_ids(N_LIST);
        for(int i=0; i<N_LIST; ++i) centroid_ids[i] = centroid_mask | i;

        // [MODIFIED] Initialize Flat index and wrap with IDMap
        flat_impl = new IndexFlatL2(d);
        index_flat_top = new IndexIDMap(flat_impl);
        index_flat_top->own_fields = true;
        
        index_flat_top->add_with_ids(N_LIST, centroids.data(), centroid_ids.data());
    }

    // 3. Build Loop
    cout << "Starting Unified Build (Flat Top-Level)..." << endl;
    vector<float> loose_vectors_buffer; 
    vector<idx_t> loose_ids_buffer;
    loose_vectors_buffer.reserve(nb * 0.05 * d); 
    loose_ids_buffer.reserve(nb * 0.05);

    double t_build_start = get_time_ms();
    size_t chunk_size = 100000; 
    float alpha_sq = ALPHA * ALPHA;
    // [MODIFIED] Removed hnsw.efSearch setting

    for (size_t start_idx = 0; start_idx < nb; start_idx += chunk_size) {
        auto [chunk_data, meta_chunk] = read_fbin(BASE_FILE, start_idx, chunk_size);
        size_t current_chunk_size = meta_chunk.first;
        float* xb_chunk = chunk_data.data();
        
        cout << "Processing chunk " << start_idx << " / " << nb << "..." << "\r" << flush;

        // Search in Top Level (Flat)
        vector<float> dists(current_chunk_size * NEIGHBOR_CHECK);
        vector<idx_t> idxs(current_chunk_size * NEIGHBOR_CHECK);
        
        // [MODIFIED] Search using Flat index
        index_flat_top->search(current_chunk_size, xb_chunk, NEIGHBOR_CHECK, dists.data(), idxs.data());

        // Assignments
        vector<idx_t> assignments(current_chunk_size);
        idx_t centroid_mask = 1ULL << 60;
        #pragma omp parallel for
        for (size_t i = 0; i < current_chunk_size; ++i) {
            assignments[i] = idxs[i * NEIGHBOR_CHECK + 0] & ~centroid_mask;
        }

        // Add to IVF
        index_ivf->add_core(current_chunk_size, xb_chunk, nullptr, assignments.data());

        // Filter Loose Points
        #pragma omp parallel
        {
            vector<float> local_vecs;
            vector<idx_t> local_ids;
            #pragma omp for nowait
            for (size_t i = 0; i < current_chunk_size; ++i) {
                float d1 = dists[i * NEIGHBOR_CHECK + 0];
                float dn = dists[i * NEIGHBOR_CHECK + (NEIGHBOR_CHECK - 1)];
                // 游离点判断逻辑保持不变
                if (d1 > 0 && (dn / d1 <= alpha_sq)) {
                    idx_t global_id = start_idx + i; 
                    local_ids.push_back(global_id);
                    for(int k=0; k<d; ++k) local_vecs.push_back(xb_chunk[i*d + k]);
                }
            }
            #pragma omp critical
            {
                loose_ids_buffer.insert(loose_ids_buffer.end(), local_ids.begin(), local_ids.end());
                loose_vectors_buffer.insert(loose_vectors_buffer.end(), local_vecs.begin(), local_vecs.end());
            }
        }
    }
    cout << endl << "IVF Build Finished. Loose points: " << loose_ids_buffer.size() << endl;

    // Update Top Level Index with Loose Points
    if (!loose_ids_buffer.empty()) {
        index_flat_top->add_with_ids(loose_ids_buffer.size(), loose_vectors_buffer.data(), loose_ids_buffer.data());
    }
    cout << "Total Build Time: " << (get_time_ms() - t_build_start) << " ms" << endl;

    // 4. Search & Eval
    cout << "Starting Search (Flat Top-Level)..." << endl;
    auto [xq_vec, meta_query] = read_fbin(QUERY_FILE);
    size_t nq = meta_query.first;
    float* xq = xq_vec.data();

    // Top Level Search (Flat)
    // [MODIFIED] Removed HNSW efSearch settings
    // 我们搜索 SEARCH_EF 个结果，以此作为候选池（Pool），从中筛选 nprobe 个质心
    vector<float> top_D(nq * SEARCH_EF);
    vector<idx_t> top_I(nq * SEARCH_EF);
    index_flat_top->search(nq, xq, SEARCH_EF, top_D.data(), top_I.data());

    // Analyze & IVF Search
    idx_t centroid_mask = 1ULL << 60;
    vector<vector<idx_t>> query_clusters(nq);
    size_t max_clusters = 0;
    
    // 这里的逻辑未变：从 SEARCH_EF 结果中筛选 nprobe 个质心
    for(size_t i=0; i<nq; ++i) {
        vector<idx_t>& clusters = query_clusters[i];
        clusters.reserve(SEARCH_NPROBE);
        for(int j=0; j<SEARCH_EF; ++j) {
            idx_t id = top_I[i*SEARCH_EF + j];
            if ((id != -1) && (id & centroid_mask)) {
                idx_t real_id = id & ~centroid_mask;
                bool found = false;
                for(auto c : clusters) if(c == real_id) { found = true; break; }
                if(!found && clusters.size() < SEARCH_NPROBE) clusters.push_back(real_id);
            }
        }
        if(clusters.size() > max_clusters) max_clusters = clusters.size();
    }
    if (max_clusters == 0) max_clusters = 1;

    vector<idx_t> ivf_assign_query(nq * max_clusters, -1);
    vector<float> ivf_centroid_dis(nq * max_clusters, 0.0f);
    for(size_t i=0; i<nq; ++i) {
        for(size_t j=0; j<query_clusters[i].size(); ++j) ivf_assign_query[i*max_clusters + j] = query_clusters[i][j];
    }

    vector<float> ivf_D(nq * SEARCH_K);
    vector<idx_t> ivf_I(nq * SEARCH_K);
    IVFSearchParameters ivf_params;
    ivf_params.nprobe = max_clusters;
    // 使用 IVF 部分进行搜索
    index_ivf->search_preassigned(nq, xq, SEARCH_K, ivf_assign_query.data(), ivf_centroid_dis.data(), ivf_D.data(), ivf_I.data(), false, &ivf_params);

    // Merge
    vector<idx_t> final_I(nq * SEARCH_K);
    #pragma omp parallel for
    for(size_t i=0; i<nq; ++i) {
        vector<pair<float, idx_t>> candidates;
        candidates.reserve(SEARCH_EF + SEARCH_K);
        
        // 1. 从上层搜索结果中提取“游离点”直接加入候选
        for(int j=0; j<SEARCH_EF; ++j) {
            idx_t id = top_I[i*SEARCH_EF + j];
            // 如果不是 -1 且 不是质心，说明是游离点
            if (id != -1 && !(id & centroid_mask)) {
                candidates.push_back({top_D[i*SEARCH_EF + j], id});
            }
        }
        
        // 2. 加入 IVF 搜索到的结果
        for(int j=0; j<SEARCH_K; ++j) {
            idx_t id = ivf_I[i*SEARCH_K + j];
            if (id != -1) candidates.push_back({ivf_D[i*SEARCH_K + j], id});
        }
        
        // 排序去重
        sort(candidates.begin(), candidates.end());
        int filled = 0;
        for(size_t j=0; j<candidates.size() && filled < SEARCH_K; ++j) {
            if (j > 0 && candidates[j].second == candidates[j-1].second) continue;
            final_I[i*SEARCH_K + filled++] = candidates[j].second;
        }
        while(filled < SEARCH_K) final_I[i*SEARCH_K + filled++] = -1;
    }

    // Eval
    auto gt_vec = read_ivecs(GROUNDTRUTH_FILE);
    int correct = 0;
    for(size_t i=0; i<nq; ++i) {
        for(int j=0; j<SEARCH_K; ++j) {
            if (final_I[i*SEARCH_K+j] == gt_vec[i][0]) {
                correct++;
                break;
            }
        }
    }
    cout << "R@" << SEARCH_K << ": " << (double)correct / nq << endl;

    delete index_ivf;
    delete index_flat_top;
    delete quantizer_train;
    return 0;
}
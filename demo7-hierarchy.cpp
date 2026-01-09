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
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/io.h>
#include <faiss/impl/FaissAssert.h>
#include <omp.h>

using namespace std;
using namespace faiss;

// ============================================================================
// Utils (Same as demo6/4)
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
const string INDEX_DIR = "indices";
const string IVF_INDEX_FILE = INDEX_DIR + "/ivf_ondisk_demo7.index";

const int N_LIST = 15625;          
const int M_HNSW = 32;             
const int EF_CONSTRUCTION = 40;    
const int EF_SEARCH_TRAIN = 16;    
const int NEIGHBOR_CHECK = 40;     
const float ALPHA = 1.1f;          
const int SEARCH_K = 10;
const int SEARCH_EF = 100;
const int SEARCH_NPROBE = 64;

// ============================================================================
// Main
// ============================================================================
int main() {
    omp_set_num_threads(8);
    ensure_dir(INDEX_DIR);

    auto [nb, d] = get_fbin_meta(BASE_FILE);
    cout << "Dataset Info: nb=" << nb << ", d=" << d << endl; 

    IndexIVFFlat* index_ivf = nullptr;
    IndexIDMap* index_hnsw = nullptr;
    IndexHNSWFlat* hnsw_impl = nullptr;
    IndexHNSWFlat* quantizer_train = nullptr;

    // 1. Training
    {
        cout << "Loading Learn data..." << endl;
        auto [xt_vec, meta_learn] = read_fbin(LEARN_FILE);
        size_t nt = meta_learn.first;
        float* xt = xt_vec.data();

        cout << "Training IVF (HNSW Quantizer)..." << endl;
        quantizer_train = new IndexHNSWFlat(d, M_HNSW);
        quantizer_train->hnsw.efConstruction = EF_CONSTRUCTION;
        quantizer_train->hnsw.efSearch = EF_SEARCH_TRAIN; 

        index_ivf = new IndexIVFFlat(quantizer_train, d, N_LIST, METRIC_L2);
        index_ivf->cp.niter = 10; 
        index_ivf->verbose = true;

        double t_start = get_time_ms();
        index_ivf->train(nt, xt);
        cout << "Training finished in " << get_time_ms() - t_start << " ms." << endl;
    }

    // 2. Init HNSW (Centroids)
    cout << "Initializing HNSW with Centroids..." << endl;
    {
        vector<float> centroids(N_LIST * d);
        index_ivf->quantizer->reconstruct_n(0, N_LIST, centroids.data());

        idx_t centroid_mask = 1ULL << 60;
        vector<idx_t> centroid_ids(N_LIST);
        for(int i=0; i<N_LIST; ++i) centroid_ids[i] = centroid_mask | i;

        hnsw_impl = new IndexHNSWFlat(d, M_HNSW);
        hnsw_impl->hnsw.efConstruction = EF_CONSTRUCTION;
        index_hnsw = new IndexIDMap(hnsw_impl);
        index_hnsw->own_fields = true;
        
        index_hnsw->add_with_ids(N_LIST, centroids.data(), centroid_ids.data());
    }

    // 3. Unified Build (RAM first)
    cout << "Starting Unified Build (RAM)..." << endl;
    vector<float> loose_vectors_buffer; 
    vector<idx_t> loose_ids_buffer;
    loose_vectors_buffer.reserve(nb * 0.05 * d); 
    loose_ids_buffer.reserve(nb * 0.05);

    double t_build_start = get_time_ms();
    size_t chunk_size = 100000; 
    float alpha_sq = ALPHA * ALPHA;
    hnsw_impl->hnsw.efSearch = EF_CONSTRUCTION;

    for (size_t start_idx = 0; start_idx < nb; start_idx += chunk_size) {
        auto [chunk_data, meta_chunk] = read_fbin(BASE_FILE, start_idx, chunk_size);
        size_t current_chunk_size = meta_chunk.first;
        float* xb_chunk = chunk_data.data();
        
        cout << "Processing chunk " << start_idx << " / " << nb << "..." << "\r" << flush;

        // Search HNSW
        vector<float> dists(current_chunk_size * NEIGHBOR_CHECK);
        vector<idx_t> idxs(current_chunk_size * NEIGHBOR_CHECK);
        index_hnsw->search(current_chunk_size, xb_chunk, NEIGHBOR_CHECK, dists.data(), idxs.data());

        // Assignments
        vector<idx_t> assignments(current_chunk_size);
        idx_t centroid_mask = 1ULL << 60;
        #pragma omp parallel for
        for (size_t i = 0; i < current_chunk_size; ++i) {
            assignments[i] = idxs[i * NEIGHBOR_CHECK + 0] & ~centroid_mask;
        }

        // Add to IVF (RAM)
        index_ivf->add_core(current_chunk_size, xb_chunk, nullptr, assignments.data());

        // Filter / Collect Loose Points
        #pragma omp parallel
        {
            vector<float> local_vecs;
            vector<idx_t> local_ids;
            #pragma omp for nowait
            for (size_t i = 0; i < current_chunk_size; ++i) {
                float d1 = dists[i * NEIGHBOR_CHECK + 0];
                float dn = dists[i * NEIGHBOR_CHECK + (NEIGHBOR_CHECK - 1)];
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
    cout << endl << "Build Finished. Loose points: " << loose_ids_buffer.size() << endl;

    // Add Loose points to HNSW
    if (!loose_ids_buffer.empty()) {
        index_hnsw->add_with_ids(loose_ids_buffer.size(), loose_vectors_buffer.data(), loose_ids_buffer.data());
    }
    cout << "Total RAM Build Time: " << (get_time_ms() - t_build_start) << " ms" << endl;

    // 4. OFFLOAD TO DISK (Hierarchy Setup)
    cout << "Offloading IVF index to disk: " << IVF_INDEX_FILE << "..." << endl;
    write_index(index_ivf, IVF_INDEX_FILE.c_str());
    
    // Free RAM Index
    delete index_ivf; 
    index_ivf = nullptr;
    // Note: index_hnsw remains in RAM (Top of hierarchy)

    // 5. LOAD FROM DISK (Memory Mapped)
    cout << "Reloading IVF index as Memory Mapped (Out-of-Core)..." << endl;
    {
        // Load with MMAP flag
        Index* raw_index = read_index(IVF_INDEX_FILE.c_str(), IO_FLAG_MMAP);
        index_ivf = dynamic_cast<IndexIVFFlat*>(raw_index);
        if (!index_ivf) {
            cerr << "Error: Loaded index is not IndexIVFFlat!" << endl;
            return 1;
        }
        cout << "IVF Index loaded. Is memory mapped? (Implied by IO_FLAG_MMAP)" << endl;
    }

    // 6. Unified Search (Hybrid: RAM HNSW + Disk IVF)
    cout << "Starting Unified Search..." << endl;
    auto [xq_vec, meta_query] = read_fbin(QUERY_FILE);
    size_t nq = meta_query.first;
    float* xq = xq_vec.data();

    // HNSW Step (RAM)
    hnsw_impl->hnsw.efSearch = SEARCH_EF;
    vector<float> hnsw_D(nq * SEARCH_EF);
    vector<idx_t> hnsw_I(nq * SEARCH_EF);
    
    double t_search_start = get_time_ms();
    index_hnsw->search(nq, xq, SEARCH_EF, hnsw_D.data(), hnsw_I.data());

    // Expand Centroids
    idx_t centroid_mask = 1ULL << 60;
    vector<vector<idx_t>> query_clusters(nq);
    size_t max_clusters = 0;
    for(size_t i=0; i<nq; ++i) {
        vector<idx_t>& clusters = query_clusters[i];
        clusters.reserve(SEARCH_NPROBE);
        for(int j=0; j<SEARCH_EF; ++j) {
            idx_t id = hnsw_I[i*SEARCH_EF + j];
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

    // IVF Step (Disk Access via MMAP)
    vector<float> ivf_D(nq * SEARCH_K);
    vector<idx_t> ivf_I(nq * SEARCH_K);
    IVFSearchParameters ivf_params;
    ivf_params.nprobe = max_clusters;
    
    // Standard preassigned search
    index_ivf->search_preassigned(nq, xq, SEARCH_K, ivf_assign_query.data(), ivf_centroid_dis.data(), 
                                  ivf_D.data(), ivf_I.data(), false, &ivf_params);

    // Merge
    vector<idx_t> final_I(nq * SEARCH_K);
    #pragma omp parallel for
    for(size_t i=0; i<nq; ++i) {
        vector<pair<float, idx_t>> candidates;
        candidates.reserve(SEARCH_EF + SEARCH_K);
        for(int j=0; j<SEARCH_EF; ++j) {
            idx_t id = hnsw_I[i*SEARCH_EF + j];
            if (id != -1 && !(id & centroid_mask)) candidates.push_back({hnsw_D[i*SEARCH_EF + j], id});
        }
        for(int j=0; j<SEARCH_K; ++j) {
            idx_t id = ivf_I[i*SEARCH_K + j];
            if (id != -1) candidates.push_back({ivf_D[i*SEARCH_K + j], id});
        }
        sort(candidates.begin(), candidates.end());
        int filled = 0;
        for(size_t j=0; j<candidates.size() && filled < SEARCH_K; ++j) {
            if (j > 0 && candidates[j].second == candidates[j-1].second) continue;
            final_I[i*SEARCH_K + filled++] = candidates[j].second;
        }
        while(filled < SEARCH_K) final_I[i*SEARCH_K + filled++] = -1;
    }
    double t_search_end = get_time_ms();
    cout << "Total Search Time: " << (t_search_end - t_search_start) << " ms" << endl;
    cout << "Avg Latency: " << (t_search_end - t_search_start) / nq << " ms/query" << endl;

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

    if (index_ivf) delete index_ivf;
    if (index_hnsw) delete index_hnsw;
    if (quantizer_train) delete quantizer_train;

    return 0;
}

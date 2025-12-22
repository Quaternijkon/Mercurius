#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFHNSWHybrid.h>
#include <faiss/utils/utils.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

using namespace std::chrono;

// 路径配置
const std::string DATA_DIR = "./sift";
const std::string LEARN_FILE = DATA_DIR + "/learn.fbin";
const std::string BASE_FILE = DATA_DIR + "/base.fbin";
const std::string QUERY_FILE = DATA_DIR + "/query.fbin";
const std::string GROUNDTRUTH_FILE = DATA_DIR + "/groundtruth.ivecs";

// 读取 fbin
std::pair<std::vector<float>, std::pair<size_t, size_t>> read_fbin(
        const std::string& filename,
        size_t start_idx = 0,
        size_t chunk_size = 0) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int32_t nvecs_raw, dim_raw;
    f.read(reinterpret_cast<char*>(&nvecs_raw), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dim_raw), sizeof(int32_t));
    size_t nvecs = static_cast<size_t>(nvecs_raw);
    size_t dim = static_cast<size_t>(dim_raw);

    size_t num_vectors_in_chunk = nvecs;
    if (chunk_size > 0) {
        size_t end_idx = std::min(start_idx + chunk_size, nvecs);
        num_vectors_in_chunk = end_idx - start_idx;
        size_t offset = 8 + start_idx * dim * sizeof(float);
        f.seekg(offset, std::ios::beg);
    }

    std::vector<float> data(num_vectors_in_chunk * dim);
    f.read(reinterpret_cast<char*>(data.data()), num_vectors_in_chunk * dim * sizeof(float));

    return {data, {nvecs, dim}};
}

// 读取 ivecs (ground truth)
std::pair<std::vector<int>, std::pair<size_t, size_t>> read_ivecs(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<int> data;
    size_t dim = 0;
    size_t nvecs = 0;
    while (true) {
        int dim_raw;
        if (!f.read(reinterpret_cast<char*>(&dim_raw), sizeof(int))) {
            break;
        }
        if (dim == 0) {
            dim = static_cast<size_t>(dim_raw);
        }
        std::vector<int> tmp(dim_raw);
        f.read(reinterpret_cast<char*>(tmp.data()), sizeof(int) * dim_raw);
        data.insert(data.end(), tmp.begin(), tmp.end());
        nvecs++;
    }
    return {data, {nvecs, dim}};
}

double recall_at_k(
        const std::vector<faiss::idx_t>& I,
        const std::vector<int>& gt,
        size_t nq,
        size_t k,
        size_t gt_k) {
    size_t hits = 0;
    for (size_t qi = 0; qi < nq; qi++) {
        std::unordered_set<int> gt_set;
        for (size_t j = 0; j < gt_k; j++) {
            gt_set.insert(gt[qi * gt_k + j]);
        }
        for (size_t j = 0; j < k; j++) {
            if (gt_set.count((int)I[qi * k + j])) {
                hits++;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(nq);
}

int main() {
    const size_t k = 10;
    const size_t nlist = 4096;
    const float alpha = 1.05f;
    const int M = 32;
    const size_t nprobe = 64; // wider probing for better recall
    const size_t batch_size = 1000; // process queries in batches to limit memory/thread pressure

    std::cout << "Loading SIFT1M..." << std::endl;
    auto learn = read_fbin(LEARN_FILE);
    auto base = read_fbin(BASE_FILE);
    auto query = read_fbin(QUERY_FILE);
    auto gt = read_ivecs(GROUNDTRUTH_FILE);

    const size_t d = learn.second.second;
    const size_t nt = learn.second.first;
    const size_t nb = base.second.first;
    const size_t nq = query.second.first;
    const size_t gt_dim = gt.second.second; // usually 100

    std::cout << "dim=" << d << " nt=" << nt << " nb=" << nb << " nq=" << nq
              << " gt_k=" << gt_dim << std::endl;

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFHNSWHybrid index(&quantizer, d, nlist, alpha, M, faiss::METRIC_L2);
    index.verbose = true;
    index.nprobe = nprobe;
    index.max_outlier_fraction = 0.02f; // cap outliers to 2% to keep HNSW small
    index.hnsw_candidates = 128;
    index.hnsw_graph.hnsw.efSearch = 128;

    omp_set_num_threads(std::max(1, omp_get_max_threads() / 2));

    std::cout << "Training..." << std::endl;
    index.train(nt, learn.first.data());

    std::cout << "Adding base vectors..." << std::endl;
    index.add(nb, base.first.data());

    size_t n_core = 0, n_outlier = 0;
    for (auto t : index.point_types) {
        if (t == faiss::IndexIVFHNSWHybrid::PointType::Core) {
            n_core++;
        } else if (t == faiss::IndexIVFHNSWHybrid::PointType::Outlier) {
            n_outlier++;
        }
    }
    size_t n_anchor = index.anchor_count();

    std::cout << "Classification summary (alpha=" << alpha << ")\n";
    std::cout << "  Core points   : " << n_core << "\n";
    std::cout << "  Anchor points : " << n_anchor << "\n";
    std::cout << "  Outlier points: " << n_outlier << "\n";

    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);

    std::cout << "Searching..." << std::endl;
    auto t0 = steady_clock::now();
    for (size_t i0 = 0; i0 < nq; i0 += batch_size) {
        size_t bs = std::min(batch_size, nq - i0);
        index.search(bs, query.first.data() + i0 * d, k, D.data() + i0 * k, I.data() + i0 * k);
    }
    auto t1 = steady_clock::now();
    double secs = duration_cast<duration<double>>(t1 - t0).count();
    double qps = nq / secs;

    double recall = recall_at_k(I, gt.first, nq, k, std::min<size_t>(gt_dim, k));

    std::cout << "Search time: " << secs << " s, QPS: " << qps << std::endl;
    std::cout << "Recall@" << k << ": " << recall << std::endl;

    return 0;
}

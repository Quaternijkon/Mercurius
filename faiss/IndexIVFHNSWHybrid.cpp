#include <faiss/IndexIVFHNSWHybrid.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/distances.h>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <cstring>

namespace faiss {

IndexIVFHNSWHybrid::IndexIVFHNSWHybrid(Index *quantizer, size_t d, size_t nlist, float alpha, int M, MetricType metric)
    : IndexIVF(quantizer, d, nlist, sizeof(float) * d, metric),
      hnsw(d, M, metric),
      alpha(alpha)
{
    own_invlists = true;
    // We do NOT use direct_map as per requirements
    // direct_map.set_type(DirectMap::NoMap, invlists, ntotal); // Default is NoMap
}

void IndexIVFHNSWHybrid::encode_vectors(idx_t n, const float* x, const idx_t* list_nos, uint8_t* codes, bool include_listnos) const {
    FAISS_THROW_IF_NOT(!include_listnos);
    memcpy(codes, x, n * code_size);
}

void IndexIVFHNSWHybrid::reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);
    memcpy(recons, code, code_size);
}

void IndexIVFHNSWHybrid::train(idx_t n, const float* x) {
    if (verbose) printf("Training hybrid index...\n");
    IndexIVF::train(n, x);
    // HNSWFlat doesn't require training, but we call it for consistency
    hnsw.train(n, x); 
    
    // After training, we have centroids. Add them as Anchors.
    add_centroids_as_anchors();
}

void IndexIVFHNSWHybrid::add_centroids_as_anchors() {
    if (verbose) printf("Adding %ld centroids as Anchor Points...\n", nlist);
    std::vector<float> centroid(d);
    for (size_t i = 0; i < nlist; i++) {
        quantizer->reconstruct(i, centroid.data());
        hnsw.add(1, centroid.data());
        // Mask the ID to indicate it's an Anchor
        hnsw_id_map.push_back((idx_t)i | ANCHOR_MASK);
    }
}

void IndexIVFHNSWHybrid::add(idx_t n, const float* x) {
    if (verbose) printf("Adding %ld vectors...\n", n);
    
    // 1. Search Quantizer for top 2 centroids (Optimization)
    std::vector<float> distances(n * 2);
    std::vector<idx_t> labels(n * 2);
    
    // Use quantizer->search directly as requested
    quantizer->search(n, x, 2, distances.data(), labels.data());

    // 2. Classification & Batching
    std::vector<float> core_vectors;
    std::vector<idx_t> core_ids;
    std::vector<idx_t> core_cluster_ids;

    std::vector<float> outlier_vectors;
    std::vector<idx_t> outlier_ids;

    // Reserve memory
    core_vectors.reserve(n * d);
    core_ids.reserve(n);
    core_cluster_ids.reserve(n);
    outlier_vectors.reserve(n * d);
    outlier_ids.reserve(n);

    idx_t start_id = ntotal;

    for (idx_t i = 0; i < n; i++) {
        float d1 = distances[2 * i];
        float d2 = distances[2 * i + 1];
        idx_t cluster_id = labels[2 * i];
        
        bool is_core = false;
        if (metric_type == METRIC_INNER_PRODUCT) {
            // Cosine similarity ratio test equivalent:
            // sim2 < 1 - alpha^2 * (1 - sim1)
            float alpha_sq = alpha * alpha;
            if (d2 < 1.0f - alpha_sq * (1.0f - d1)) is_core = true;
        } else {
            // L2 distance ratio test
            float s1 = sqrt(d1);
            float s2 = sqrt(d2);
            if (s1 == 0) is_core = true;
            else if (s2 / s1 > alpha) is_core = true;
        }
        
        const float* vec = x + i * d;
        idx_t global_id = start_id + i;
        
        if (is_core) {
            core_vectors.insert(core_vectors.end(), vec, vec + d);
            core_ids.push_back(global_id);
            core_cluster_ids.push_back(cluster_id);
        } else {
            outlier_vectors.insert(outlier_vectors.end(), vec, vec + d);
            outlier_ids.push_back(global_id);
        }
    }

    // 3. Batch Add Core Points
    size_t n_core = core_ids.size();
    if (n_core > 0) {
        // Use IndexIVF::add_core to leverage existing parallel implementation
        IndexIVF::add_core(n_core, core_vectors.data(), core_ids.data(), core_cluster_ids.data());
    }

    // 4. Batch Add Outlier Points
    size_t n_outliers = outlier_ids.size();
    if (n_outliers > 0) {
        hnsw.add(n_outliers, outlier_vectors.data());
        hnsw_id_map.insert(hnsw_id_map.end(), outlier_ids.begin(), outlier_ids.end());
        
        // Manually update ntotal for outliers since they are not added via IndexIVF methods
        ntotal += n_outliers;
    }

    if (verbose) printf("Classified: %ld Core Points, %ld Outlier Points.\n", n_core, n_outliers);
}

void IndexIVFHNSWHybrid::search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels, const SearchParameters* params) const {
    // 1. Search HNSW
    int k_hnsw = std::max(k, (idx_t)10); // Search a bit more to find anchors
    std::vector<float> D_hnsw(n * k_hnsw);
    std::vector<idx_t> I_hnsw(n * k_hnsw);
    
    hnsw.search(n, x, k_hnsw, D_hnsw.data(), I_hnsw.data());
    
    for (idx_t i = 0; i < n; i++) {
        std::unordered_set<idx_t> candidates;
        std::vector<std::pair<float, idx_t>> final_results;
        
        const float* q = x + i * d;

        // 2. Expansion: Collect HNSW results & Expand Anchors
        for (int j = 0; j < k_hnsw; j++) {
            idx_t hnsw_internal_id = I_hnsw[i * k_hnsw + j];
            if (hnsw_internal_id < 0) continue;
            
            idx_t mapped_id = hnsw_id_map[hnsw_internal_id];
            
            if (mapped_id & ANCHOR_MASK) {
                // It's an Anchor Point
                idx_t cluster_id = mapped_id & ~ANCHOR_MASK;
                
                // Add ALL Core Points from this cluster
                size_t list_size = invlists->list_size(cluster_id);
                const idx_t* list_ids = invlists->get_ids(cluster_id);
                
                // We can iterate and compute distances immediately to avoid storing vectors
                // or store IDs and compute later. Let's compute immediately for simplicity.
                for (size_t l = 0; l < list_size; l++) {
                    idx_t core_id = list_ids[l];
                    if (candidates.find(core_id) == candidates.end()) {
                        candidates.insert(core_id);
                        
                        // Reconstruct from IVF list
                        std::vector<float> vec(d);
                        reconstruct_from_offset(cluster_id, l, vec.data());
                        
                        float dist = fvec_L2sqr(q, vec.data(), d);
                        final_results.push_back({dist, core_id});
                    }
                }
            } else {
                // It's an Outlier Point
                idx_t outlier_id = mapped_id;
                if (candidates.find(outlier_id) == candidates.end()) {
                    candidates.insert(outlier_id);
                    
                    // Reconstruct from HNSW
                    std::vector<float> vec(d);
                    hnsw.reconstruct(hnsw_internal_id, vec.data());
                    
                    float dist = fvec_L2sqr(q, vec.data(), d);
                    final_results.push_back({dist, outlier_id});
                }
            }
        }
        
        // 3. Final Ranking
        std::sort(final_results.begin(), final_results.end());
        
        for (int j = 0; j < k; j++) {
            if (j < final_results.size()) {
                distances[i * k + j] = final_results[j].first;
                labels[i * k + j] = final_results[j].second;
            } else {
                distances[i * k + j] = std::numeric_limits<float>::max();
                labels[i * k + j] = -1;
            }
        }
    }
}

} // namespace faiss

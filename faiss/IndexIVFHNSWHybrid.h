/*
 * @Author: quaternijkon quaternijkon@mail.ustc.edu.cn
 * @Date: 2025-12-22 09:27:37
 * @LastEditors: quaternijkon quaternijkon@mail.ustc.edu.cn
 * @LastEditTime: 2025-12-23 07:01:48
 * @FilePath: /Mercurius/faiss/faiss/IndexIVFHNSWHybrid.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef FAISS_INDEX_IVF_HNSW_HYBRID_H
#define FAISS_INDEX_IVF_HNSW_HYBRID_H

#include <faiss/IndexIVF.h>
#include <faiss/IndexHNSW.h>
#include <vector>

namespace faiss {

struct IndexIVFHNSWHybrid : IndexIVF {
    
    IndexHNSWFlat hnsw;
    float alpha;
    
    // Map HNSW Internal ID -> Global ID (for Outliers) or Masked Cluster ID (for Anchors)
    std::vector<idx_t> hnsw_id_map;

    // Bit mask to identify Anchor Points (using bit 62 to avoid sign bit issues)
    static const idx_t ANCHOR_MASK = 1ULL << 62;

    IndexIVFHNSWHybrid(Index *quantizer, size_t d, size_t nlist, float alpha, int M = 32, MetricType metric = METRIC_L2);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels, const SearchParameters* params = nullptr) const override;

    // Helper to add virtual centroids as anchors
    void add_centroids_as_anchors();

    // Required implementation for IndexIVF to handle raw vector storage
    void encode_vectors(idx_t n, const float* x, const idx_t* list_nos, uint8_t* codes, bool include_listnos = false) const override;
    
    // Required implementation for IndexIVF to reconstruct vectors from invlists
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons) const override;
};

} // namespace faiss

#endif

/*
 * @Author: quaternijkon quaternijkon@mail.ustc.edu.cn
 * @Date: 2025-12-18 09:50:51
 * @LastEditors: quaternijkon quaternijkon@mail.ustc.edu.cn
 * @LastEditTime: 2025-12-19 09:29:41
 * @FilePath: /Mercurius/faiss/faiss/IndexIVFHNSWHybrid.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Hybrid IVF + HNSW index.
 * Core points live in IVF lists; outliers and anchors live in an HNSW graph.
 */

#pragma once

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/utils/utils.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace faiss {

struct IndexIVFHNSWHybrid : IndexIVFFlat {
    enum class PointType : uint8_t { Core = 0, Outlier = 1, Anchor = 2 };

    float alpha = 1.01f; // ratio threshold
    float max_outlier_fraction = 0.1f; // cap to keep HNSW build manageable
    IndexHNSWFlat hnsw_graph;

    static constexpr uint64_t kAnchorFlag = (uint64_t(1) << 63);

    // Per-id metadata
    std::vector<PointType> point_types; // indexed by user id (assumes dense ids)

    // For core points: list id that stored the vector; -1 for outliers.
    std::vector<idx_t> id_to_list;

    // Graph label -> encoded id (outlier id, or anchor flag | list_no)
    std::vector<idx_t> hnsw_ids;

    // encoded id -> graph node for O(1) lookup
    std::unordered_map<idx_t, idx_t> id_to_graph;

    // Anchor bookkeeping per list
    std::vector<bool> anchor_inserted; // whether centroid for list is in HNSW

    // How many HNSW neighbors to pull before expansion
    idx_t hnsw_candidates = 64;

    IndexIVFHNSWHybrid(
            Index* quantizer,
            size_t d,
            size_t nlist,
            float alpha = 1.2f,
            int M = 32,
            MetricType metric = METRIC_L2);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override {
        add_with_ids(n, x, nullptr);
    }
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    size_t anchor_count() const {
        return std::count(anchor_inserted.begin(), anchor_inserted.end(), true);
    }

private:
    bool reconstruct_outlier(idx_t encoded_id, std::vector<float>& buffer) const;
    void push_hnsw_point(const float* x, idx_t encoded_id);
    bool fetch_vector(idx_t id, std::vector<float>& buffer) const;
    void maybe_add_anchor(size_t list_no);
};

} // namespace faiss

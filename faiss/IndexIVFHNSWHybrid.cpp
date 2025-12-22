/*
 * Hybrid IVF + HNSW index implementation.
 */

#include "faiss/IndexIVFHNSWHybrid.h"

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <unordered_set>

namespace faiss {

using ScopedIds = InvertedLists::ScopedIds;
using ScopedCodes = InvertedLists::ScopedCodes;

IndexIVFHNSWHybrid::IndexIVFHNSWHybrid(
        Index* quantizer,
        size_t d,
        size_t nlist,
        float alpha,
        int M,
        MetricType metric)
        : IndexIVFFlat(quantizer, d, nlist, metric),
          alpha(alpha),
          hnsw_graph(d, M, metric) {
    by_residual = false;
    direct_map.set_type(DirectMap::NoMap, invlists, ntotal);
    anchor_inserted.assign(nlist, false);
    hnsw_graph.own_fields = true;
    hnsw_graph.keep_max_size_level0 = true;
    hnsw_graph.hnsw.efSearch = std::max<int>(hnsw_graph.hnsw.efSearch, 2 * (int)hnsw_candidates);
}

void IndexIVFHNSWHybrid::train(idx_t n, const float* x) {
    IndexIVFFlat::train(n, x);
    anchor_inserted.assign(nlist, false);
    point_types.clear();
    id_to_list.clear();
    id_to_graph.clear();
    direct_map.set_type(DirectMap::NoMap, invlists, ntotal);
}

void IndexIVFHNSWHybrid::push_hnsw_point(const float* x, idx_t encoded_id) {
    idx_t graph_id = hnsw_graph.ntotal;
    hnsw_graph.add(1, x);
    hnsw_ids.push_back(encoded_id);
    id_to_graph[encoded_id] = graph_id;
}

void IndexIVFHNSWHybrid::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);
    std::vector<float> dist(n * 2);
    std::vector<idx_t> assign(n * 2);
    quantizer->search(n, x, 2, dist.data(), assign.data());

    idx_t old_ntotal = ntotal;
    idx_t max_id = xids ? *std::max_element(xids, xids + n) : old_ntotal + n - 1;
    if ((idx_t)point_types.size() <= max_id) {
        point_types.resize(max_id + 1, PointType::Core);
    }
    if ((idx_t)id_to_list.size() <= max_id) {
        id_to_list.resize(max_id + 1, -1);
    }

    std::vector<float> ratios(n, 0.0f);
    std::vector<idx_t> coarse_idx(n, -1);
    size_t n_core = 0, n_outlier = 0;
    float alpha_eff = alpha;

    for (idx_t i = 0; i < n; i++) {
        idx_t user_id = xids ? xids[i] : old_ntotal + i;
        float s1 = dist[2 * i];
        float s2 = dist[2 * i + 1];
        idx_t list_no = assign[2 * i];

        bool core = false;
        if (list_no >= 0) {
            float ratio;
            if (is_similarity_metric(metric_type)) {
                ratio = s1 / (s2 + 1e-6f);
            } else {
                float denom = std::max(s1, 1e-6f);
                ratio = (assign[2 * i + 1] < 0) ?
                        std::numeric_limits<float>::infinity() :
                        s2 / denom;
            }
            ratios[i] = ratio;
            core = ratio > alpha_eff;
        }

        if (core) {
            coarse_idx[i] = list_no;
            point_types[user_id] = PointType::Core;
            id_to_list[user_id] = list_no;
            n_core++;
        } else {
            point_types[user_id] = PointType::Outlier;
            id_to_list[user_id] = -1;
            n_outlier++;
        }
    }

    if (max_outlier_fraction > 0.0f) {
        size_t max_outliers = (size_t)std::ceil(max_outlier_fraction * (double)n);
        if (n_outlier > max_outliers) {
            std::vector<float> ratios_copy = ratios;
            size_t kth = std::min(max_outliers, (size_t)ratios_copy.size() - 1);
            std::nth_element(ratios_copy.begin(), ratios_copy.begin() + kth, ratios_copy.end());
            alpha_eff = ratios_copy[kth];

            if (verbose) {
                printf(
                        "Hybrid add: outliers %zu exceed cap %zu, tightening alpha from %.4f to %.4f\n",
                        n_outlier,
                        max_outliers,
                        alpha,
                        alpha_eff);
                fflush(stdout);
            }

            n_core = 0;
            n_outlier = 0;
            std::fill(coarse_idx.begin(), coarse_idx.end(), -1);
            for (idx_t i = 0; i < n; i++) {
                idx_t user_id = xids ? xids[i] : old_ntotal + i;
                bool core = ratios[i] > alpha_eff;
                idx_t list_no = assign[2 * i];
                if (core && list_no >= 0) {
                    coarse_idx[i] = list_no;
                    point_types[user_id] = PointType::Core;
                    id_to_list[user_id] = list_no;
                    n_core++;
                } else {
                    point_types[user_id] = PointType::Outlier;
                    id_to_list[user_id] = -1;
                    n_outlier++;
                }
            }
        }
    }

    IndexIVF::add_core(n, x, xids, coarse_idx.data());

    auto t_after_core = std::chrono::steady_clock::now();
    if (verbose) {
        printf(
                "Hybrid add: add_core complete (n=%jd) core=%zu outlier=%zu time=%.3fs\n",
                (intmax_t)n,
                n_core,
                n_outlier,
                std::chrono::duration<double>(t_after_core.time_since_epoch()).count());
        fflush(stdout);
    }

    size_t outliers_pushed = 0;
    auto t_outlier_start = std::chrono::steady_clock::now();
    for (idx_t i = 0; i < n; i++) {
        idx_t user_id = xids ? xids[i] : old_ntotal + i;
        PointType pt = point_types[user_id];
        if (pt == PointType::Outlier) {
            push_hnsw_point(x + i * d, user_id);
            outliers_pushed++;
            if (verbose && (outliers_pushed % 10000 == 0)) {
                double secs = std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - t_outlier_start)
                                      .count();
                printf(
                        "Hybrid add: HNSW outliers %zu/%zu (%.2fs)\n",
                        outliers_pushed,
                        n_outlier,
                        secs);
                fflush(stdout);
            }
        }
    }

    if (verbose) {
        double secs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t_outlier_start)
                              .count();
        printf(
                "Hybrid add: HNSW outliers done %zu/%zu (%.2fs)\n",
                outliers_pushed,
                n_outlier,
                secs);
        fflush(stdout);
    }

    // insert anchors (centroids) for lists that now have data
    size_t anchors_before = std::count(anchor_inserted.begin(), anchor_inserted.end(), true);
    size_t anchors_added = 0;
    auto t_anchor_start = std::chrono::steady_clock::now();
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        bool was_inserted = anchor_inserted[list_no];
        maybe_add_anchor(list_no);
        if (!was_inserted && anchor_inserted[list_no]) {
            anchors_added++;
            if (verbose && anchors_added % 512 == 0) {
                double secs = std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - t_anchor_start)
                                      .count();
                printf(
                        "Hybrid add: anchors %zu added so far (%.2fs)\n",
                        anchors_added,
                        secs);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        double secs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t_anchor_start)
                              .count();
        printf(
                "Hybrid add: anchors done added=%zu total=%zu (%.2fs)\n",
                anchors_added,
                anchors_before + anchors_added,
                secs);
            fflush(stdout);
    }

    if (verbose) {
        printf(
                "Hybrid add: core=%zu, anchors_added=%zu, outliers=%zu\n",
                n_core,
                std::count(anchor_inserted.begin(), anchor_inserted.end(), true),
                n_outlier);
    }
}

bool IndexIVFHNSWHybrid::reconstruct_outlier(idx_t encoded_id, std::vector<float>& buffer) const {
    auto it = id_to_graph.find(encoded_id);
    if (it == id_to_graph.end()) {
        return false;
    }
    buffer.resize(d);
    hnsw_graph.reconstruct(it->second, buffer.data());
    return true;
}

void IndexIVFHNSWHybrid::maybe_add_anchor(size_t list_no) {
    if (anchor_inserted[list_no]) {
        return;
    }
    if (invlists->list_size(list_no) == 0) {
        return; // empty list, no anchor
    }
    std::vector<float> centroid(d);
    quantizer->reconstruct(list_no, centroid.data());
    idx_t encoded_anchor = (idx_t)(kAnchorFlag | (uint64_t)list_no);
    push_hnsw_point(centroid.data(), encoded_anchor);
    anchor_inserted[list_no] = true;
}

bool IndexIVFHNSWHybrid::fetch_vector(idx_t id, std::vector<float>& buffer) const {
    if (id < 0 || (size_t)id >= point_types.size()) {
        return false;
    }

    if (point_types[id] == PointType::Outlier) {
        return reconstruct_outlier(id, buffer);
    }

    idx_t list_no = (id < (idx_t)id_to_list.size()) ? id_to_list[id] : -1;
    if (list_no < 0 || (size_t)list_no >= nlist) {
        return false;
    }

    ScopedIds sids(invlists, list_no);
    ScopedCodes scodes(invlists, list_no);
    const idx_t* ids = sids.get();
    const uint8_t* codes = scodes.get();
    size_t lsz = invlists->list_size(list_no);
    for (size_t t = 0; t < lsz; t++) {
        if (ids[t] == id) {
            buffer.resize(d);
            const float* vec = reinterpret_cast<const float*>(codes + t * code_size);
            std::copy(vec, vec + d, buffer.data());
            return true;
        }
    }
    return false;
}

void IndexIVFHNSWHybrid::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    // 1) Run standard IVFFlat search to get strong core recall quickly.
    IndexIVFFlat::search(n, x, k, distances, labels, params);

    // 2) Optionally refine with HNSW outliers (anchors skipped) and merge.
    idx_t graph_k = hnsw_graph.ntotal == 0
            ? 0
            : std::min<idx_t>(hnsw_graph.ntotal, std::max<idx_t>(k * 2, hnsw_candidates));
    if (graph_k == 0) {
        return;
    }

    std::vector<float> graph_dis(graph_k);
    std::vector<idx_t> graph_lab(graph_k);
    std::vector<float> buffer(d);

    for (idx_t qi = 0; qi < n; qi++) {
        const float* q = x + qi * d;

        hnsw_graph.search(1, q, graph_k, graph_dis.data(), graph_lab.data(), params);

        for (idx_t j = 0; j < graph_k; j++) {
            idx_t glab = graph_lab[j];
            if (glab < 0 || glab >= (idx_t)hnsw_ids.size()) {
                continue;
            }
            idx_t encoded = hnsw_ids[glab];
            if ((uint64_t(encoded) & kAnchorFlag) != 0) {
                continue; // skip anchors in final results
            }

            if (!fetch_vector(encoded, buffer)) {
                continue;
            }
            float dis = metric_type == METRIC_L2 ? fvec_L2sqr(q, buffer.data(), d)
                                                  : fvec_inner_product(q, buffer.data(), d);

            // insert into existing top-k if better
            // Find worst current result
            idx_t worst_idx = -1;
            float worst_dis = metric_type == METRIC_L2 ? -1.0f : 1e30f;
            for (idx_t r = 0; r < k; r++) {
                float cur = distances[qi * k + r];
                if (labels[qi * k + r] == -1) {
                    worst_idx = r;
                    worst_dis = metric_type == METRIC_L2 ? std::numeric_limits<float>::infinity()
                                                         : -std::numeric_limits<float>::infinity();
                    break;
                }
                if (metric_type == METRIC_L2) {
                    if (cur > worst_dis) {
                        worst_dis = cur;
                        worst_idx = r;
                    }
                } else {
                    if (cur < worst_dis) {
                        worst_dis = cur;
                        worst_idx = r;
                    }
                }
            }

            bool better = metric_type == METRIC_L2 ? dis < worst_dis : dis > worst_dis;
            if (worst_idx == -1 || better) {
                labels[qi * k + (worst_idx == -1 ? k - 1 : worst_idx)] = encoded;
                distances[qi * k + (worst_idx == -1 ? k - 1 : worst_idx)] = dis;
            }
        }

        // keep outputs sorted
        std::vector<std::pair<float, idx_t>> pairs(k);
        for (idx_t r = 0; r < k; r++) {
            pairs[r] = {distances[qi * k + r], labels[qi * k + r]};
        }
        auto cmp_l2 = [](const std::pair<float, idx_t>& a, const std::pair<float, idx_t>& b) {
            return a.first < b.first;
        };
        auto cmp_ip = [](const std::pair<float, idx_t>& a, const std::pair<float, idx_t>& b) {
            return a.first > b.first;
        };
        std::sort(pairs.begin(), pairs.end(), metric_type == METRIC_L2 ? cmp_l2 : cmp_ip);
        for (idx_t r = 0; r < k; r++) {
            distances[qi * k + r] = pairs[r].first;
            labels[qi * k + r] = pairs[r].second;
        }
    }
}

} // namespace faiss

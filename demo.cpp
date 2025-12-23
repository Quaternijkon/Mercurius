/*
 * @Author: quaternijkon quaternijkon@mail.ustc.edu.cn
 * @Date: 2025-12-22 08:52:24
 * @LastEditors: quaternijkon quaternijkon@mail.ustc.edu.cn
 * @LastEditTime: 2025-12-22 12:05:23
 * @FilePath: /Mercurius/faiss/demo.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>
#include <vector>
#include <random>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFHNSWHybrid.h>
#include <faiss/utils/utils.h>

using namespace faiss;

// Data Generation Helper
void generate_data(size_t d, size_t n, std::vector<float>& data) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<> dist(0, 1);
    data.resize(n * d);
    for (size_t i = 0; i < n * d; i++) {
        data[i] = dist(rng);
    }
}

int main() {
    size_t d = 128;      // Dimension
    size_t nb = 100000;  // Database size
    size_t nq = 10;      // Query size
    size_t nlist = 1000; // Number of clusters
    float alpha = 1.01;  // Ratio threshold

    std::vector<float> xb;
    generate_data(d, nb, xb);
    fvec_renorm_L2(d, nb, xb.data());
    
    std::vector<float> xq;
    generate_data(d, nq, xq);
    fvec_renorm_L2(d, nq, xq.data());

    // Quantizer
    IndexFlatIP quantizer(d);
    
    // Hybrid Index
    IndexIVFHNSWHybrid index(&quantizer, d, nlist, alpha, 32, METRIC_INNER_PRODUCT);
    index.verbose = true;
    
    // Train
    index.train(nb, xb.data());
    
    // Add
    index.add(nb, xb.data());
    
    // Search
    int k = 5;
    std::vector<idx_t> I(nq * k);
    std::vector<float> D(nq * k);
    
    printf("\nSearching...\n");
    index.search(nq, xq.data(), k, D.data(), I.data());
    
    printf("\nResults:\n");
    for (int i = 0; i < nq; i++) {
        printf("Query %d:\n", i);
        for (int j = 0; j < k; j++) {
            printf("  Rank %d: ID=%ld, Dist=%f\n", j, I[i * k + j], D[i * k + j]);
        }
    }

    return 0;
}

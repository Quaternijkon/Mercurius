# Copilot usage notes

- Purpose: Faiss provides C++17 vector-search indices (L2, dot, cosine) with GPU backends and Python/C APIs; CPU/GPU indices live in `faiss/` (see `Index*.{cpp,h}`, `impl/`, `invlists/`, `utils/`, `gpu/`). Benchmarks in `benchs/`, demos in `demos/`, perf micro-benchmarks in `perf_tests/`, C API in `c_api/`, and Python binding sources in `faiss/python/`.
- Default build (CPU-only): `cmake -B build .` then `make -C build -j faiss`; enable SIMD targets with `-DFAISS_OPT_LEVEL=avx2|avx512|avx512_spr` before building `faiss_avx2/faiss_avx512/faiss_avx512_spr` targets.
- GPU build: pass `-DFAISS_ENABLE_GPU=ON` and optionally `-DCMAKE_CUDA_ARCHITECTURES="75;80;90"`; AMD GPUs need `-DFAISS_ENABLE_ROCM=ON`. cuVS integration requires `-DFAISS_ENABLE_CUVS=ON` and `libcuvs=25.10` (Conda install in `INSTALL.md`).
- Intel SVS integration: use `-DFAISS_ENABLE_SVS=ON` to pull SVS and add graph-based indices; ensure the generated `libsvs_runtime.so` is on `LD_LIBRARY_PATH` when using the C++ libs.
- Python bindings: configure with `-DFAISS_ENABLE_PYTHON=ON`; build via `make -C build -j swigfaiss`, then `(cd build/faiss/python && python setup.py install)`.
- C API: enable with `-DFAISS_ENABLE_C_API=ON`; C headers/sources are under `c_api/` (see `c_api/INSTALL.md` for packaging details).
- Testing (C++): configure with `-DBUILD_TESTING=ON` and run `make -C build test`.
- Testing (Python): after building bindings, run from repo root `PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)/" pytest tests/test_*.py`.
- Examples: build demos with `make -C build demo_ivfpq_indexing` (CPU) or `demo_ivfpq_indexing_gpu` (GPU) then run from repo root; `demos/demo_auto_tune.py` exercises multiple index types (set `keys_to_test` and `use_gpu=True` for GPU runs).
- Common CMake toggles: `-DFAISS_ENABLE_C_API=ON|OFF`, `-DBUILD_SHARED_LIBS=ON|OFF`, `-DFAISS_USE_LTO=ON` for LTO, `-DBLA_VENDOR=Intel10_64_dyn` plus `-DMKL_LIBRARIES=...` to force MKL, `-DPython_EXECUTABLE=...` for non-default Python.
- Style: C++/Python use 4-space indent, 80-character lines; keep code C++17; follow existing `Index*` patterns (factory in `index_factory.cpp`, serialization in `index_io.h`, GPU counterparts under `faiss/gpu/`).
- Data flow: user builds `faiss::Index` variants (flat, IVF, PQ, HNSW/NSG, additive/fast-scan codecs) stored in `faiss/Index*.{cpp,h}`; training adds centroids/codebooks; searches route through metric-aware distance/quantization kernels; GPU classes (`GpuIndex*`) mirror CPU APIs and automatically move data when inputs are on CPU.
- Benchmarks: large-scale scripts live in `benchs/` (eg. IVF/PQ/HNSW runners); SIFT1M benchmarking via `make -C build demo_sift1M && ./build/demos/demo_sift1M` (requires dataset download per `README`).
- Troubleshooting: build flags and platform notes are centralized in `INSTALL.md`; prefer MKL or tuned BLAS for performance and reduce `-j` if OOM during `make`.
- Contribution norms: fork from `main`, add tests for new features, keep APIs documented, and respect coding style; CLA required (see `CONTRIBUTING.md`).
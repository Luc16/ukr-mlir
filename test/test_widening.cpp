#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <cstring>

// Dynamic Types based on compilation flags
#if TYPE_IN == 8
using T_in = int8_t;
#elif TYPE_IN == 16
using T_in = _Float16;
#elif TYPE_IN == 17
using T_in = __bf16; 
#elif TYPE_IN == 32
using T_in = float;
#elif TYPE_IN == 33
using T_in = int32_t; // Support for pure i32 input
#else
using T_in = double;
#endif

// Map VNNI output (i32) or standard FP outputs
#if TYPE_OUT == 16
using T_out = _Float16;
#elif TYPE_OUT == 17
using T_out = __bf16;
#elif TYPE_OUT == 32 && TYPE_IN != 8 && TYPE_IN != 33
using T_out = float;
#elif TYPE_OUT == 33 || TYPE_IN == 8 || TYPE_IN == 33
using T_out = int32_t; // Support for i32 output
#else
using T_out = double;
#endif

// MLIR ABI MemRefs
template<typename T> struct MemRef0D { T* basePtr; T* data; int64_t offset; };
template<typename T> struct MemRef1D { T* basePtr; T* data; int64_t offset; int64_t sizes[1]; int64_t strides[1]; };
template<typename T> struct MemRef2D { T* basePtr; T* data; int64_t offset; int64_t sizes[2]; int64_t strides[2]; };

// External MLIR Function (Mixed Precision)
extern "C" void MLIR_CIFACE_FUNC(
    int64_t m, int64_t n, int64_t k,
    MemRef0D<T_out>* alpha,
    MemRef2D<T_in>* a,
    MemRef2D<T_in>* b,
    MemRef0D<T_out>* beta,
    MemRef1D<T_out>* c,
    int64_t rs_c, int64_t cs_c,
    int64_t data, int64_t cntx
);

// Naive C++ Ground Truth (Perfectly matches edge cases and panel packing)
void naive_ukernel(int64_t m, int64_t n, int64_t k, T_out alpha, T_in* A, T_in* B, T_out beta, T_out* C, int64_t rs_c, int64_t cs_c) {
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            T_out sum = 0;
            for (int64_t p = 0; p < k; ++p) {
#if TYPE_IN == 8 || TYPE_IN == 33
                // VNNI / Pure Integer Dot Product (No float cast needed)
                sum += (T_out)A[p * TEST_MR + i] * (T_out)B[p * TEST_NR + j];
#else
                // Widened Floating Point Dot Product 
                sum += (T_out)( (float)A[p * TEST_MR + i] * (float)B[p * TEST_NR + j] );
#endif
            }
            C[i * rs_c + j * cs_c] = alpha * sum + beta * C[i * rs_c + j * cs_c];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./test_ukernel <k> <iterations>\n";
        return 1;
    }

    int64_t m = TEST_MR;
    int64_t n = TEST_NR;
    int64_t k = std::stoll(argv[1]);
    int iterations = std::stoi(argv[2]);

    int64_t rs_c = 1;
    int64_t cs_c = TEST_MR;
    #ifdef IS_ROW_MAJOR
        rs_c = TEST_NR;
        cs_c = 1;
    #endif

    size_t size_A = TEST_MR * k * sizeof(T_in);
    size_t size_B = TEST_NR * k * sizeof(T_in);
    size_t size_C = TEST_MR * TEST_NR * sizeof(T_out);

    T_in* A = (T_in*)std::aligned_alloc(64, size_A);
    T_in* B = (T_in*)std::aligned_alloc(64, size_B);
    T_out* C_mlir = (T_out*)std::aligned_alloc(64, size_C);
    T_out* C_naive = (T_out*)std::aligned_alloc(64, size_C);

    std::random_device rd;
    std::mt19937 gen(rd());
    
#if TYPE_IN == 8 || TYPE_IN == 33
    // Random integers (Keep them small to avoid i32 overflow during accumulation)
    std::uniform_int_distribution<int> dist(-5, 5);
#else
    // Random floats for fp16/bf16/f32
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
#endif

    for (int i = 0; i < TEST_MR * k; ++i) {
        A[i] = (T_in)dist(gen);
    }
    for (int i = 0; i < TEST_NR * k; ++i) {
        B[i] = (T_in)dist(gen);
    }
    
    std::memset(C_mlir, 0, size_C);
    std::memset(C_naive, 0, size_C);

    T_out alpha = (T_out)1; // For integers, this resolves to 1
    T_out beta = (T_out)0;  // For integers, this resolves to 0

    MemRef0D<T_out> mr_alpha = {&alpha, &alpha, 0};
    MemRef0D<T_out> mr_beta  = {&beta, &beta, 0};
    MemRef2D<T_in> mr_A = { A, A, 0, {k, TEST_MR}, {TEST_MR, 1} };
    MemRef2D<T_in> mr_B = { B, B, 0, {k, TEST_NR}, {TEST_NR, 1} };
    MemRef1D<T_out> mr_C = { C_mlir, C_mlir, 0, {TEST_MR * TEST_NR}, {1} };

    naive_ukernel(m, n, k, alpha, A, B, beta, C_naive, rs_c, cs_c);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        MLIR_CIFACE_FUNC(m, n, k, &mr_alpha, &mr_A, &mr_B, &mr_beta, &mr_C, rs_c, cs_c, 0, 0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff_mlir = end - start;
    double gflops_mlir = (2.0 * m * n * k * iterations) / (diff_mlir.count() * 1e9);

    float max_diff = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float val_mlir = (float)C_mlir[i * rs_c + j * cs_c];
            float val_naive = (float)C_naive[i * rs_c + j * cs_c];
            max_diff = std::max(max_diff, std::abs(val_mlir - val_naive));
        }
    }

    std::cout << "Benchmarking Micro-Kernel: " << m << "x" << n << "x" << k << "\n";
    std::cout << "Max difference vs Naive: " << max_diff << "\n";
    std::cout << "MLIR Performance : " << std::fixed << std::setprecision(2) << gflops_mlir << " GFLOPS\n";

    if (max_diff < 1e-2) std::cout << "Result check: PASSED\n";
    else std::cout << "Result check: FAILED\n";

    std::free(A); std::free(B); std::free(C_mlir); std::free(C_naive);
    return 0;
}

#include <blis.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>

// Only include the wrapper if we are compiling a custom MLIR kernel
#ifdef RUN_CUSTOM
  #include "wrapper.h"
#endif

// Default to float if not specified (BLIS macros need this)
#ifdef USE_FLOAT
  #define BLIS_DATATYPE BLIS_FLOAT
  using RealType = float;
#else
  #define BLIS_DATATYPE BLIS_DOUBLE
  using RealType = double;
#endif

double run_gemm(const cntx_t* cntx, dim_t m, dim_t n, dim_t k, int iterations, obj_t* A, obj_t* B, obj_t* C) {
    obj_t alpha_obj, beta_obj;
    bli_obj_create_1x1(BLIS_DATATYPE, &alpha_obj);
    bli_obj_create_1x1(BLIS_DATATYPE, &beta_obj);

    if constexpr (std::is_same<RealType, double>::value) {
        bli_setsc(1.0, 0.0, &alpha_obj);
        bli_setsc(1.0, 0.0, &beta_obj);
    } else {
        bli_setsc(1.0f, 0.0f, &alpha_obj);
        bli_setsc(1.0f, 0.0f, &beta_obj);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++) {
        bli_gemm_ex(&alpha_obj, A, B, &beta_obj, C, cntx, NULL);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double flops = 2.0 * m * n * k * iterations;
    return flops / (diff.count() * 1e9);
}

int main(int argc, char** argv) {
    if (argc < 10) {
        std::cerr << "Usage: ./runner <M> <N> <K> <iterations> <is_custom> <mr> <nr> <arch> <precision> <layout>\n";
        return 1;
    }

    dim_t M = std::stoll(argv[1]);
    dim_t N = std::stoll(argv[2]);
    dim_t K = std::stoll(argv[3]);
    
    int iterations         = std::atoi(argv[4]);
    int is_custom          = std::atoi(argv[5]);
    std::string mr_str     = argv[6];
    std::string nr_str     = argv[7];
    std::string arch       = argv[8];
    std::string precision  = argv[9];
    std::string layout     = (argc > 10) ? argv[10] : "row";

    bli_init();

    obj_t A, B, C, C_blis;
    bli_obj_create(BLIS_DATATYPE, M, K, 1, M, &A);
    bli_obj_create(BLIS_DATATYPE, K, N, 1, K, &B);
    bli_obj_create(BLIS_DATATYPE, M, N, 1, M, &C);
    bli_obj_create(BLIS_DATATYPE, M, N, 1, M, &C_blis);

    bli_randm(&A);
    bli_randm(&B);
    bli_setm(&BLIS_ZERO, &C);
    bli_setm(&BLIS_ZERO, &C_blis);

    const cntx_t* base_cntx = bli_gks_query_cntx();
    cntx_t active_cntx = *base_cntx;

    // --- Extract Default BLIS Micro-Kernel Dimensions ---
    auto blis_mr_blksz = bli_cntx_get_blksz(BLIS_MR, base_cntx);
    auto blis_nr_blksz = bli_cntx_get_blksz(BLIS_NR, base_cntx);
    
    dim_t blis_mr = bli_blksz_get_def(BLIS_DATATYPE, blis_mr_blksz);
    dim_t blis_nr = bli_blksz_get_def(BLIS_DATATYPE, blis_nr_blksz);
    // ----------------------------------------------------

    double gflops_blis = run_gemm(&active_cntx, M, N, K, iterations, &A, &B, &C_blis);

    if (is_custom) {
        // Register the custom MLIR UKernel and block sizes
        bli_cntx_set_ukr_dt((void_fp)TARGET_RUN_FUNC, BLIS_DATATYPE, BLIS_GEMM_UKR, &active_cntx);

        blksz_t mr_bs, nr_bs;
        bli_blksz_init(&mr_bs, TEST_MR, TEST_MR, TEST_MR, TEST_MR, TEST_MR, TEST_MR, TEST_MR, TEST_MR); 
        bli_blksz_init(&nr_bs, TEST_NR, TEST_NR, TEST_NR, TEST_NR, TEST_NR, TEST_NR, TEST_NR, TEST_NR);

        bli_cntx_set_blksz(BLIS_MR, &mr_bs, BLIS_MR, &active_cntx);
        bli_cntx_set_blksz(BLIS_NR, &nr_bs, BLIS_NR, &active_cntx);
    }

    double gflops = run_gemm(&active_cntx, M, N, K, iterations, &A, &B, &C);

    std::cout << "Version with Regular BLIS: M: " << M << ", N: " << N << ", K: " << K 
              << ", native_mr: " << blis_mr << ", native_nr: " << blis_nr 
              << ", Performance: " << gflops_blis << " GFLOPS\n";
              
    std::cout << "Version with MLIR micro kernel: M: " << M << ", N: " << N << ", K: " << K 
              << ", custom_mr: " << mr_str << ", custom_nr: " << nr_str 
              << ", Performance: " << gflops << " GFLOPS\n";

    bli_obj_free(&A);
    bli_obj_free(&B);
    bli_obj_free(&C);
    bli_obj_free(&C_blis);

    bli_finalize();
    return 0;
}

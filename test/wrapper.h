#pragma once

#include <blis.h>
#include <stdint.h>

#define CONCAT_INNER(A,B) A##B
#define CONCAT(A,B) CONCAT_INNER(A,B)

#ifdef USE_FLOAT
using RealType = float;
#else
using RealType = double;
#endif

// ------------------------------------------------------------
// MLIR MemRef ABI
// ------------------------------------------------------------
template<typename T>
struct MemRef0D {
  T* basePtr;
  T* data;
  int64_t offset;
};

template<typename T>
struct MemRef2D {
  T* basePtr;
  T* data;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

template<typename T>
struct MemRef1D {
  T* basePtr;
  T* data;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

// ------------------------------------------------------------
// Kernel naming
// ------------------------------------------------------------
#define MLIR_CIFACE_FUNC CONCAT(_mlir_ciface_, TEST_KERNEL_FUNC)
#define TARGET_RUN_FUNC CONCAT(wrapper_, TEST_KERNEL_FUNC)

// ------------------------------------------------------------
// MLIR kernel symbol
// ------------------------------------------------------------
extern "C" void MLIR_CIFACE_FUNC(
    int64_t m, int64_t n, int64_t k,
    MemRef0D<RealType>* alpha,
    MemRef2D<RealType>* a,
    MemRef2D<RealType>* b,
    MemRef0D<RealType>* beta,
    MemRef1D<RealType>* c,
    int64_t rs_c,
    int64_t cs_c,
    int64_t data,
    int64_t cntx
    );

extern "C" inline void TARGET_RUN_FUNC(
    dim_t m, dim_t n, dim_t k,
    const void* alpha_p,
    const void* a_p,
    const void* b_p,
    const void* beta_p,
    void* c_p,
    inc_t rs_c,
    inc_t cs_c,
    const auxinfo_t* data,
    const cntx_t* cntx
    )
{
  if (k == 0) return;
  RealType* alpha = (RealType*)alpha_p;
  RealType* beta  = (RealType*)beta_p;
  RealType* a = (RealType*)a_p;
  RealType* b = (RealType*)b_p;
  RealType* c = (RealType*)c_p;

  bool isRowMajor = (cs_c == 1);

// #ifdef USE_FLOAT
//   GEMM_UKR_SETUP_CT_ANY( s, TEST_MR, TEST_NR, isRowMajor );
// #else
//   GEMM_UKR_SETUP_CT_ANY( d, TEST_MR, TEST_NR, isRowMajor );
// #endif


  MemRef0D<RealType> mr_alpha = {alpha, alpha, 0};
  MemRef0D<RealType> mr_beta  = {beta, beta, 0};

  MemRef2D<RealType> mr_A = {
    a, a, 0,
    {k, TEST_MR},
    {TEST_MR, 1}
  };

  MemRef2D<RealType> mr_B = {
    b, b, 0,
    {k, TEST_NR},
    {TEST_NR, 1}
  };

  MemRef1D<RealType> mr_C = {
    c, c, 0,
    {m * rs_c + n * cs_c}, // Safe boundary size
    {1}                    // Native stride
  };

  MLIR_CIFACE_FUNC(
      m, n, k, &mr_alpha, &mr_A, &mr_B, &mr_beta, &mr_C,
      rs_c, cs_c, (int64_t)data, (int64_t)cntx
      );

// #ifdef USE_FLOAT
//   GEMM_UKR_FLUSH_CT( s );
// #else
//   GEMM_UKR_FLUSH_CT( d );
// #endif
}

#include <immintrin.h>

#include "waxpby.h"

/**
 * @brief Compute the update of a vector with the sum of two scaled vectors
 * 
 * @param n Number of vector elements
 * @param alpha Scalars applied to x
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */
int waxpby (const int n, const double * const x, const double beta, const double * const y, double * const w) {  
  const int loopFactor = 4;
  const int loopN = (n/loopFactor)*loopFactor;

  __m256d v_beta = _mm256_set1_pd(beta);
  
  #pragma omp parallel for
  for (int i=0; i<loopN; i+=loopFactor) {
    __m256d v_x = _mm256_load_pd(x+i);
    __m256d v_y = _mm256_load_pd(y+i);
    __m256d v_w = _mm256_add_pd(v_x, _mm256_mul_pd(v_beta, v_y));
    _mm256_store_pd(w+i, v_w);
  }

  for (int i=loopN; i<n; i++) {
    w[i] = x[i]+(beta*y[i]);
  }

  return 0;
}

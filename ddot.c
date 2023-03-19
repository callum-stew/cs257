#include <immintrin.h>

#include "ddot.h"

/**
 * @brief Compute the dot product of two vectors
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */
int ddot (const int n, const double * const x, const double * const y, double * const result) {  
  double local_result = 0.0;
  __m256d v_r_total = _mm256_setzero_pd();

  const int loopFactor = 8;
  const int loopN = (n/loopFactor)*loopFactor;

  #pragma omp parallel shared(v_r_total)
  {
     __m256d v_r = _mm256_setzero_pd();

    #pragma omp for
    for (int i=0; i<loopN; i+=loopFactor) {
      __m256d v_x1 = _mm256_load_pd(x+i);
      __m256d v_y1 = _mm256_load_pd(y+i);
      __m256d v_x2 = _mm256_load_pd(x+i+4);
      __m256d v_y2 = _mm256_load_pd(y+i+4);
      v_r = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(v_x1, v_y1), _mm256_mul_pd(v_x2, v_y2)), v_r);
    }
    #pragma omp critical
    {
      v_r_total = _mm256_add_pd(v_r, v_r_total);
    }
  }

  for (int i=loopN; i<n; i++) {
    local_result += x[i]*y[i];
  }

  double r_array[4] __attribute__((aligned(32)));
  _mm256_store_pd(r_array, v_r_total);
  local_result += r_array[0] + r_array[1] + r_array[2] + r_array[3];

  *result = local_result;
  return 0;
}

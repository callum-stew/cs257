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
  __m256d v_r = _mm256_set1_pd(0.0);
  __m256d v_r_total = _mm256_set1_pd(0.0);

  const int loopFactor = 4;
  const int loopN = (n/loopFactor)*loopFactor;

  #pragma omp parallel firstprivate(v_r) shared(v_r_total)
  {
    #pragma omp for
    for (int i=0; i<loopN; i+=loopFactor) {
      __m256d v_x = _mm256_load_pd(x+i);
      __m256d v_y = _mm256_load_pd(y+i);
      v_r = _mm256_add_pd(_mm256_mul_pd(v_x, v_y), v_r);
    }
    #pragma omp critical
    {
      v_r_total = _mm256_add_pd(v_r, v_r_total);
    }
  }

  for (int i=loopN; i<n; i++) {
    local_result += x[i]*y[i];
  }

  double * r_array = _mm_malloc(sizeof(double)*4, 32);
  _mm256_store_pd(r_array, v_r_total);
  local_result += r_array[0] + r_array[1] + r_array[2] + r_array[3];
  _mm_free(r_array);

  *result = local_result;
  return 0;
}

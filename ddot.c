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
int ddot (const int n, const float * const x, const float * const y, float * const result) {  
  float local_result = 0.0;
  __m256 v_r_total = _mm256_setzero_ps();

  const int loopFactor = 8;
  const int loopN = (n/loopFactor)*loopFactor;

  #pragma omp parallel shared(v_r_total)
  {
     __m256 v_r = _mm256_setzero_ps();

    #pragma omp for
    for (int i=0; i<loopN; i+=loopFactor) {
      __m256 v_x1 = _mm256_load_ps(x+i);
      __m256 v_y1 = _mm256_load_ps(y+i);
      v_r = _mm256_add_ps(_mm256_mul_ps(v_x1, v_y1), v_r);
    }
    #pragma omp critical
    {
      v_r_total = _mm256_add_ps(v_r, v_r_total);
    }
  }

  for (int i=loopN; i<n; i++) {
    local_result += x[i]*y[i];
  }

  float r_array[4] __attribute__((aligned(32)));
  _mm256_store_ps(r_array, v_r_total);
  local_result += r_array[0] + r_array[1] + r_array[2] + r_array[3];

  *result = local_result;
  return 0;
}

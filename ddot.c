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
  __m256 v_r = _mm256_set1_ps(0.0);
  __m256 v_r_total = _mm256_set1_ps(0.0);

  const int loopFactor = 8;
  const int loopN = (n/loopFactor)*loopFactor;

  #pragma omp parallel firstprivate(v_r) shared(v_r_total)
  {
    #pragma omp for
    for (int i=0; i<loopN; i+=loopFactor) {
      __m256 v_x = _mm256_load_ps(x+i);
      __m256 v_y = _mm256_load_ps(y+i);
      v_r = _mm256_add_ps(_mm256_mul_ps(v_x, v_y), v_r);
    }
    #pragma omp critical
    {
      v_r_total = _mm256_add_ps(v_r, v_r_total);
    }
  }

  for (int i=loopN; i<n; i++) {
    local_result += x[i]*y[i];
  }

  float * r_array = _mm_malloc(sizeof(float)*loopFactor, 32);
  _mm256_store_ps(r_array, v_r_total);
  local_result += r_array[0] + r_array[1] + r_array[2] + r_array[3];
  _mm_free(r_array);

  *result = local_result;
  return 0;
}

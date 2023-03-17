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
  #pragma omp parallel reduction(+:local_result)
  {
    if (y==x){
      #pragma omp for nowait
      for (int i=0; i<n; i++) {
        local_result += x[i]*x[i];
      }

    } else {
      #pragma omp for nowait
      for (int i=0; i<n; i++) {
        local_result += x[i]*y[i];
      }
    }
  }

  *result = local_result;
  return 0;
}

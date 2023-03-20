#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>

#include "sparsemv.h"

/**
 * @brief Compute matrix vector product (y = A*x)
 * 
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */
int sparsemv(struct mesh *A, const double * const x, double * const y)
{
  const int nrow = (const int) A->local_nrow;

  //#pragma omp parallel for
  for (int i=0; i<nrow; i++) {
      double sum = 0.0;
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
      const int cur_nnz = (const int) A->nnz_in_row[i];
      int j;
      const int loopN = (cur_nnz/4)*4;

      __m256d sum_vec = _mm256_setzero_pd();

      for (j=0; j<loopN; j+=4) {
        __m256d val_vec = _mm256_loadu_pd(&cur_vals[j]);
        __m256d x_vec = _mm256_loadu_pd(&x[cur_inds[j]]);
        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(val_vec, x_vec));
      }

      double tmp[4] __attribute__((aligned(32))) = {0.0};
      _mm256_store_pd(tmp, sum_vec);
      sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];

      for (; j<cur_nnz; j++) {
        sum += (cur_vals[j]*x[cur_inds[j]]);
      }
      y[i] = sum;
  }

  return 0;
}

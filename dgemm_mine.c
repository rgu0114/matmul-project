const char* dgemm_desc = "My awesome dgemm.";

// Tune this number by trying lots of different stuff
#define BLOCK_SIZE 48

#include <immintrin.h>  // Header for AVX intrinsics

/* V5: Copy Optimization */

/* V4: SIMD */
/** Performs mat_mul on a sub-block of the input matrices using SIMD */
void do_block(const int M, const int block_size, const double *A, const double *B, double *C,
              const int i_start, const int j_start, const int k_start) {
    for (int j = j_start; j < j_start + block_size && j < M; ++j) {
        // Step size = 4
        int i = i_start;
        for (; i <= i_start + block_size - 4 && i <= M - 4; i += 4) {
            // Load 4 elements of C into an AVX register
            __m256d c_vec = _mm256_loadu_pd(&C[j * M + i]);

            for (int k = k_start; k < k_start + block_size && k < M; ++k) {
                // Load 4 elements from A and broadcast a single element from B
                __m256d a_vec = _mm256_loadu_pd(&A[k * M + i]);
                __m256d b_val = _mm256_broadcast_sd(&B[j * M + k]);

                // c_vec += a_vec * b_val
                c_vec = _mm256_fmadd_pd(a_vec, b_val, c_vec);
            }

            _mm256_storeu_pd(&C[j * M + i], c_vec);
        }

        // Handle the remaining elements (when i % 4 != 0)
        for (; i < i_start + block_size && i < M; ++i) {
            double cij = C[j * M + i];
            for (int k = k_start; k < k_start + block_size && k < M; ++k) {
                cij += A[k * M + i] * B[j * M + k];
            }
            C[j * M + i] = cij;
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C) {
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int k = 0; k < M; k += BLOCK_SIZE) {
            for (int j = 0; j < M; j += BLOCK_SIZE) {
                do_block(M, BLOCK_SIZE, A, B, C, i, j, k);
            }
        }
    }
}


/* V3: Loop orders */
/** Performs mat_mul on a sub-block of the input matrices */
// void do_block(const int M, const int block_size, const double *A, const double *B, double *C,
//               const int i_start, const int j_start, const int k_start) {    
//     for (int j = j_start; j < j_start + block_size && j < M; ++j) {
//         for (int i = i_start; i < i_start + block_size && i < M; ++i) {
//             double cij = C[j * M + i];
//             for (int k = k_start; k < k_start + block_size && k < M; ++k) {
//                 cij += A[k * M + i] * B[j * M + k];
//             }
//             C[j * M + i] = cij;
//         }
//     }
// }

// void square_dgemm(const int M, const double *A, const double *B, double *C) {
//     for (int i = 0; i < M; i += BLOCK_SIZE) {
//         for (int k = 0; k < M; k += BLOCK_SIZE) {
//             for (int j = 0; j < M; j += BLOCK_SIZE) {
//                 do_block(M, BLOCK_SIZE, A, B, C, i, j, k);
//             }
//         }
//     }
// }

/* V2: Blocking */

/** Performs mat_mul on a sub-block of the input matrices */
// void do_block(const int M, const int block_size, const double *A, const double *B, double *C,
//               const int i_start, const int j_start, const int k_start) {
//     for (int i = i_start; i < i_start + block_size && i < M; ++i) {
//         for (int j = j_start; j < j_start + block_size && j < M; ++j) {
//             double cij = C[j * M + i];
//             for (int k = k_start; k < k_start + block_size && k < M; ++k) {
//                 cij += A[k * M + i] * B[j * M + k];
//             }
//             C[j * M + i] = cij;
//         }
//     }
// }

// void square_dgemm(const int M, const double *A, const double *B, double *C) {
//     for (int i = 0; i < M; i += BLOCK_SIZE) {
//         for (int j = 0; j < M; j += BLOCK_SIZE) {
//             for (int k = 0; k < M; k += BLOCK_SIZE) {
//                 do_block(M, BLOCK_SIZE, A, B, C, i, j, k);
//             }
//         }
//     }
// }


/* V1: Naive Implementation */
// void square_dgemm(const int M, const double *A, const double *B, double *C)
// {
//     int i, j, k;
//     for (i = 0; i < M; ++i) {
//         for (j = 0; j < M; ++j) {
//             double cij = C[j*M+i];
//             for (k = 0; k < M; ++k)
//                 cij += A[k*M+i] * B[j*M+k];
//             C[j*M+i] = cij;
//         }
//     }
// }
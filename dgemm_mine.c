#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <stdint.h>     
#include <assert.h> 
#include <omp.h>

const char* dgemm_desc = "My awesome dgemm.";

// Tune this number by trying lots of different stuff
#define BLOCK_SIZE 48

/* V6: Aligned SIMD (+loop reordering and copy opt) */
void do_block(const int M, const int block_size, double* A_block, double* B_block,
    const double* A, const double* B, double* C, const int i_start,
    const int j_start, const int k_start)
{
    for (int k = k_start; k < k_start + block_size && k < M; ++k) {
        for (int i = i_start; i < i_start + block_size && i < M; ++i) {
            A_block[(k - k_start) * block_size + (i - i_start)] = A[k * M + i];
        }
    }

    for (int j = j_start; j < j_start + block_size && j < M; ++j) {
        for (int k = k_start; k < k_start + block_size && k < M; ++k) {
            B_block[(j - j_start) * block_size + (k - k_start)] = B[j * M + k];
        }
    }

    for (int j = j_start; j < j_start + block_size && j < M; ++j) {
        for (int k = k_start; k < k_start + block_size && k < M; ++k) {
            int i = i_start;
            for (; i < i_start + block_size - 8 && i < M - 8; i += 8) {
                double* aligned_C_ptr = &C[j * M + i];
                __m512d c_vec = _mm512_load_pd(aligned_C_ptr);

                __m512d a_vec = _mm512_load_pd(&A_block[(k - k_start) * block_size + (i - i_start)]);
                __m512d b_val = _mm512_set1_pd(B_block[(j - j_start) * block_size + (k - k_start)]);

                c_vec = _mm512_fmadd_pd(a_vec, b_val, c_vec);
                _mm512_storeu_pd(aligned_C_ptr, c_vec);
            }

            // Handle the remaining elements (when i % 8 != 0)
            for (; i < i_start + block_size && i < M; ++i) {
                C[j * M + i] += A_block[(k - k_start) * block_size + (i - i_start)] * B_block[(j - j_start) * block_size + (k - k_start)];
            }
        }
    }
}

void square_dgemm(const int M, const double* A, const double* B, double* C)
{
    double* A_block = (double*)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double* B_block = (double*)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    assert(((uintptr_t) A_block % 64) == 0);
    assert(((uintptr_t) B_block % 64) == 0);

    #pragma omp parallel for collapse(3) private (A_block, B_block)
    for (int j = 0; j < M; j += BLOCK_SIZE) {
        for (int k = 0; k < M; k += BLOCK_SIZE) {
            for (int i = 0; i < M; i += BLOCK_SIZE) {
                do_block(M, BLOCK_SIZE, A_block, B_block, A, B, C, i, j, k);
            }
        }
    }

    free(A_block);
    free(B_block);
}

/* V5: SIMD */
/** Performs mat_mul on a sub-block of the input matrices using SIMD */
// void do_block(const int M, const int block_size, const double *A, const double *B, double *C,
//               const int i_start, const int j_start, const int k_start) {
//     for (int j = j_start; j < j_start + block_size && j < M; ++j) {
//         // Step size = 8
//         int i = i_start;
//         for (; i <= i_start + block_size - 8 && i <= M - 8; i += 8) {
//             // Load 8 elements of C into an AVX register
//             __m512d c_vec = _mm512_loadu_pd(&C[j * M + i]);

//             for (int k = k_start; k < k_start + block_size && k < M; ++k) {
//                 // Load 8 elements from A and broadcast a single element from B
//                 __m512d a_vec = _mm512_loadu_pd(&A[k * M + i]);
//                 __m512d b_val = _mm512_set1_pd(B[j * M + k]);

//                 // c_vec += a_vec * b_val
//                 c_vec = _mm512_fmadd_pd(a_vec, b_val, c_vec);
//             }

//             _mm512_storeu_pd(&C[j * M + i], c_vec);
//         }

//         // Handle the remaining elements (when i % 8 != 0)
//         for (; i < i_start + block_size && i < M; ++i) {
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

/* V4: Copy optimization */
// void do_block(const int M, const int block_size, double* A_block, double* B_block,
//     const double* A, const double* B, double* C, const int i_start,
//     const int j_start, const int k_start)
// {

//     memset(A_block, 0, block_size * block_size * sizeof(double));
//     memset(B_block, 0, block_size * block_size * sizeof(double));

//     for (int k = k_start; k < k_start + block_size && k < M; ++k) {
//         for (int i = i_start; i < i_start + block_size && i < M; ++i) {
//             A_block[(k - k_start) * block_size + (i - i_start)] = A[k * M + i];
//         }
//     }

//     for (int j = j_start; j < j_start + block_size && j < M; ++j) {
//         for (int k = k_start; k < k_start + block_size && k < M; ++k) {
//             B_block[(j - j_start) * block_size + (k - k_start)] = B[j * M + k];
//         }
//     }

//     for (int j = j_start; j < j_start + block_size && j < M; ++j) {
//         for (int i = i_start; i < i_start + block_size && i < M; ++i) {
//             double cij = C[j * M + i];
//             for (int k = k_start; k < k_start + block_size && k < M; ++k) {
//                 cij += A_block[(k - k_start) * block_size + (i - i_start)] * B_block[(j - j_start) * block_size + (k - k_start)];
//             }
//             C[j * M + i] = cij;
//         }
//     }
// }

// void square_dgemm(const int M, const double* A, const double* B, double* C)
// {
//     double* A_block = (double*)aligned_alloc(BLOCK_SIZE * sizeof(double), BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
//     double* B_block = (double*)aligned_alloc(BLOCK_SIZE * sizeof(double), BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

//     for (int i = 0; i < M; i += BLOCK_SIZE) {
//         for (int j = 0; j < M; j += BLOCK_SIZE) {
//             for (int k = 0; k < M; k += BLOCK_SIZE) {
//                 do_block(M, BLOCK_SIZE, A_block, B_block, A, B, C, i, j, k);
//             }
//         }
//     }

//     free(A_block);
//     free(B_block);
// }

/* V3: Loop orders */
/** Performs mat_mul on a sub-block of the input matrices */
// void do_block(const int M, const int block_size, const double* A, const double* B, double* C,
//     const int i_start, const int j_start, const int k_start)
// {
//     for (int j = j_start; j < j_start + block_size && j < M; ++j) {
//         for (int k = k_start; k < k_start + block_size && k < M; ++k) {
//             for (int i = i_start; i < i_start + block_size && i < M; ++i) {
//                 C[j * M + i] += A[k * M + i] * B[j * M + k];
//             }
//         }
//     }
// }

// void square_dgemm(const int M, const double* A, const double* B, double* C)
// {
//     for (int j = 0; j < M; j += BLOCK_SIZE) {
//         for (int k = 0; k < M; k += BLOCK_SIZE) {
//             for (int i = 0; i < M; i += BLOCK_SIZE) {
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
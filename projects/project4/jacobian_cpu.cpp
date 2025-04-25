#include <iostream>
#include <cstdlib> // malloc, free, atoi, rand
#include <cmath>
#include <iomanip> // setprecision
#include <omp.h>
#include "mm_utils.hpp" // provides initDiagDomNearIdentityMatrix(int, TYPE*)
constexpr *double* TOLERANCE = 0.001;
constexpr *int* DEF_SIZE = 1000;
constexpr *int* MAX_ITERS = 15000;
constexpr *double* LARGE = 1e6;
typedef *double* TYPE;
*int* main(*int* *argc*, *char*** *argv*) {
 // 1) Problem size
*int* N = (*argc* == 2) ? std::atoi(*argv*[1]) : DEF_SIZE;
std::cout << "Matrix dimension (N) = " << N << "\n";
std::cout << "Threads: " << omp_get_max_threads() << "\n";
 // 2) Allocate with raw pointers
TYPE* A = static_cast<TYPE*>(std::malloc(sizeof(TYPE) * N * N));
TYPE* b = static_cast<TYPE*>(std::malloc(sizeof(TYPE) * N));
TYPE* xnew = static_cast<TYPE*>(std::malloc(sizeof(TYPE) * N));
TYPE* xold = static_cast<TYPE*>(std::malloc(sizeof(TYPE) * N));
if (!A || !b || !xnew || !xold) {
std::cerr << "Allocation failed\n";
return 1;
 }
 // 3) Initialize matrix & vectors
std::srand(42);
initDiagDomNearIdentityMatrix(N, A);
for (*int* i = 0; i < N; ++i) {
b[i] = static_cast<TYPE>(std::rand() % 51) / 100.0;
xnew[i] = xold[i] = 0.0;
 }
 // 4) Jacobi iteration on CPU
*double* start = omp_get_wtime();
TYPE conv = LARGE;
*int* iters = 0;
while (conv > TOLERANCE && iters < MAX_ITERS) {
++iters;
 // (a) Jacobi update
 // Distributes loop iterations across available CPU threads
 // 'parallel for' creates a team of threads, each executing a portion of iterations
 // 'schedule(static)' divides iterations evenly among threads (default chunk size)
#pragma omp parallel for schedule(static)
for (*int* i = 0; i < N; ++i) {
TYPE sum = 0;
for (*int* j = 0; j < N; ++j) {
if (i != j) sum += A[i*N + j] * xold[j];
 }
xnew[i] = (b[i] - sum) / A[i*N + i];
 }
 // (b) Convergence norm
TYPE loc_sq = 0;
 // Similar to above, but includes a reduction operation
 // 'reduction(+:loc_sq)' safely accumulates partial sums from all threads
 // Each thread maintains a private copy of loc_sq during computation
 // At the end of the loop, all private copies are combined using addition
#pragma omp parallel for reduction(+:loc_sq) schedule(static)
for (*int* i = 0; i < N; ++i) {
TYPE d = xnew[i] - xold[i];
loc_sq += d*d;
 }
conv = std::sqrt(loc_sq);
 // (c) Swap xold ← xnew
 // Parallelizes the array update operation
 // Each thread handles a subset of the array copying
#pragma omp parallel for schedule(static)
for (*int* i = 0; i < N; ++i) {
xold[i] = xnew[i];
 }
 }
*double* elapsed = omp_get_wtime() - start;
std::cout << "Converged in " << iters
<< " iterations, time = " << elapsed << " s, final conv = "
<< conv << "\n";
 // 5) Compute GFLOP/s
*double* flops_per_iter = 2.0 * N * N + 2.0 * N; // as before
*double* total_flops = flops_per_iter * static_cast<*double*>(iters);
*double* gflops = total_flops / (elapsed * 1e9);
std::cout << std::fixed << std::setprecision(3)
<< "Estimated performance: " << gflops << " GFLOP/s\n";
 // 6) Verify solution
TYPE err = 0, checksum = 0;
for (*int* i = 0; i < N; ++i) {
TYPE tmp = 0;
for (*int* j = 0; j < N; ++j) {
tmp += A[i*N + j] * xnew[j];
 }
TYPE d = tmp - b[i];
err += d*d;
checksum += xnew[i];
 }
err = std::sqrt(err);
std::cout << "Error = " << err
<< ", checksum = " << checksum << "\n";
if (err > TOLERANCE)
std::cout << "WARNING: error exceeds tolerance!\n";
 // 7) Free memory
std::free(A);
std::free(b);
std::free(xnew);
std::free(xold);
return 0;
}
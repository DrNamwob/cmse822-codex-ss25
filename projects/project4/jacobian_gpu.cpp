#include <iostream>
#include <cstdlib>    // malloc, free, atoi, rand, system
#include <cmath>
#include <iomanip>    // for std::setprecision
#include <fstream>
#include <string>
#include <omp.h>
#include "mm_utils.hpp"   // provides initDiagDomNearIdentityMatrix(int, TYPE*)

constexpr double TOLERANCE = 0.001;
constexpr int    DEF_SIZE  = 1000;
constexpr int    MAX_ITERS = 100000;
constexpr double LARGE     = 1e6;

typedef double TYPE;

int main(int argc, char** argv) {
    // Determine problem size
    int N = (argc == 2) ? std::atoi(argv[1]) : DEF_SIZE;
    std::cout << "Matrix dimension (N) = " << N << "\n";

    // Prepare output directory and CSV path
    const std::string out_dir = "/mnt/scratch/bowmand8/cmse822-codex-ss25/projects/project4/Timing_Results";
    const std::string csv_path = out_dir + "/results_gpu.csv";
    // Create directory if it doesn't exist
    std::system((std::string("mkdir -p ") + out_dir).c_str());

    // 1) Allocate with raw pointers
    TYPE* A    = static_cast<TYPE*>(std::malloc(sizeof(TYPE) * N * N));
    TYPE* b    = static_cast<TYPE*>(std::malloc(sizeof(TYPE) * N));
    TYPE* xnew = static_cast<TYPE*>(std::malloc(sizeof(TYPE) * N));
    TYPE* xold = static_cast<TYPE*>(std::malloc(sizeof(TYPE) * N));

    if (!A || !b || !xnew || !xold) {
        std::cerr << "Allocation failed\n";
        return 1;
    }

    // 2) Initialize matrix & vectors
    std::srand(42);
    initDiagDomNearIdentityMatrix(N, A);
    for (int i = 0; i < N; ++i) {
        b[i]    = static_cast<TYPE>(std::rand() % 51) / 100.0;
        xnew[i] = xold[i] = 0.0;
    }

    // 3) Jacobi iteration on GPU
    double start = omp_get_wtime();
    TYPE   conv  = LARGE;
    int    iters = 0;

    // Create a data environment on the GPU and map host arrays to device memory
    // - 'to' transfers A and b to the GPU (read-only)
    // - 'tofrom' transfers xnew and xold to GPU initially and back to host at end of region
    #pragma omp target data map(to:   A[0:N*N], b[0:N]) \
                            map(tofrom: xnew[0:N], xold[0:N])
    {
      while (conv > TOLERANCE && iters < MAX_ITERS) {
        ++iters;
        // a) Jacobi update
        // Launch parallel teams on the GPU and distribute loop iterations
        // 'teams distribute' creates multiple teams of threads
        // 'parallel for' further parallelizes within each team
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < N; ++i) {
          TYPE sum = 0;
          for (int j = 0; j < N; ++j)
            if (i != j) sum += A[i*N + j] * xold[j];
          xnew[i] = (b[i] - sum) / A[i*N + i];
        }
        // b) Convergence norm
        TYPE loc_sq = 0;
        // Similar to above but with reduction operator
        // 'reduction(+:loc_sq)' safely accumulates partial sums from all threads
        #pragma omp target teams distribute parallel for reduction(+:loc_sq)
        for (int i = 0; i < N; ++i) {
          TYPE d = xnew[i] - xold[i];
          loc_sq += d * d;
        }
        // Transfer computed reduction value back to host memory
        // This is needed since 'loc_sq' is used by host code for convergence check
        #pragma omp target update from(loc_sq)
        conv = std::sqrt(loc_sq);
        // c) Swap xold ← xnew
        // Parallelize the array update across GPU threads
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < N; ++i)
          xold[i] = xnew[i];
      }
      // Explicitly transfer the final solution back to host memory
      // This ensures we have the latest values even if the implicit transfer at region end fails
      #pragma omp target update from(xnew[0:N])
    }

    double elapsed = omp_get_wtime() - start;
    std::cout << "Converged in " << iters
              << " iters, time = " << elapsed << " s, final conv = "
              << conv << "\n";

    // 4) Compute GFLOP/s
    double flops_per_iter = 2.0 * N * N + 2.0 * N;
    double total_flops    = flops_per_iter * static_cast<double>(iters);
    double gflops         = total_flops / (elapsed * 1e9);

    std::cout << std::fixed << std::setprecision(3)
              << "Estimated performance: " << gflops << " GFLOP/s\n";

    // 5) Append result to CSV
    std::ofstream ofs(csv_path, std::ios::app);
    if (!ofs) {
        std::cerr << "Error: could not open " << csv_path << " for writing\n";
    } else {
        ofs << N << "," << std::fixed << std::setprecision(3)
            << gflops << "\n";
        ofs.close();
    }

    // 6) Verify solution
    TYPE err = 0, checksum = 0;
    for (int i = 0; i < N; ++i) {
      TYPE tmp = 0;
      for (int j = 0; j < N; ++j)
        tmp += A[i*N + j] * xnew[j];
      TYPE d = tmp - b[i];
      err      += d * d;
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
#include <iostream>
#include <chrono>
#include <cstring>
#include <omp.h> // Include OpenMP header
#include <cmath>

int main(int argc, char* argv[]) {
    // Default size
    long long N = 100000;
    
    // Parse command line argument for vector size if provided
    if (argc > 1) {
        try {
            N = std::stoll(argv[1]);
            if (N <= 0) {
                std::cerr << "Vector size must be positive. Using default size: " << N << std::endl;
                N = 100000;
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid size argument. Using default size: " << N << std::endl;
        }
    }
    
    std::cout << "Vector size: " << N << std::endl;
    
    // Tolerance for comparison
    const double TOL = 1e-4;
    
    // Use raw pointers instead of std::vector
    double *a = new double[N];
    double *b = new double[N];
    double *c = new double[N];
    double *res = new double[N];
    
    // Initialize arrays on the host
    #pragma omp parallel for
    for (long long i = 0; i < N; ++i) {
        a[i] = static_cast<double>(i % 1000); // Use modulo to avoid large values
        b[i] = 2.0 * static_cast<double>(i % 1000);
        res[i] = a[i] + b[i]; // Reference result for verification
    }
    
    // Print device information
    int num_devices = omp_get_num_devices();
    std::cout << "Number of available devices: " << num_devices << std::endl;
    std::cout << "Default device: " << omp_get_default_device() << std::endl;
    
    // Add two vectors using GPU and measure time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Transfer data to device, execute on device, then transfer results back
    #pragma omp target data map(to: a[0:N], b[0:N]) map(from: c[0:N])
    {
        #pragma omp target teams distribute parallel for
        for (long long i = 0; i < N; ++i) {
            c[i] = a[i] + b[i];
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Test results on host (only check a sample if N is very large)
    int err = 0;
    long long check_count = (N > 1000000) ? 1000000 : N; // Limit verification for very large vectors
    double sample_factor = static_cast<double>(N) / check_count;
    
    #pragma omp parallel for reduction(+:err)
    for (long long i = 0; i < check_count; ++i) {
        long long idx = static_cast<long long>(i * sample_factor);
        if (idx >= N) idx = N - 1;
        double diff = c[idx] - res[idx];
        if (diff * diff > TOL) {
            ++err;
        }
    }
    
    std::cout << "Vectors added with " << err << " errors in sample check\n";
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    double gflops = (2.0 * N / elapsed.count()) / 1e9;
    std::cout << "Performance: " << gflops << " GFLOP/s\n";
    
    // Print a few values for verification
    if (N >= 5) {
        std::cout << "\nSome result values for verification:" << std::endl;
        for (int i = 0; i < 5; i++) {
            std::cout << "c[" << i << "] = " << c[i] << ", expected: " << res[i] << std::endl;
        }
    }
    
    // Clean up memory
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] res;
    
    return 0;
}
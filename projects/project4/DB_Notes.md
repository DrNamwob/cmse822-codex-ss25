module purge

module load NVHPC/23.7-CUDA-12.1.1

nvc++ -O2 -mp=gpu -gpu=ccnative -o vecadd_gpu vecadd.cpp





mp=mulitcore tells the compiler to compile for CPU.

nvc++ -O2 -mp=multicore -o jacobi_cpu claude_jacobian_cpu.cpp
export OMP_NUM_THREADS=16    # or however many cores you want to use
./jacobi_cpu 2000




std::vector<double> a[N], b[N] etc.

But we need to define "raw pointers" to the data??

<double> a_ptr = a.data():



CPU scaling:

vector size = each line

CPU threads on x axis

FLOPs on Y axis



GPU vs CPU plots:

X axis = vector size

y axis = FLOPS

several line for CPUs for CPU with different thread counts
the other line is GPU (different colors)





Jacobian Solver:









1. Step 3: Initial OpenMP Parallelization
This version introduces basic GPU parallelization using OpenMP target directives. The key features are:

Uses target directive to offload computation to GPU
Maps data from host to device for each iteration
Uses teams distribute parallel for to parallelize computation
Uses reduction for convergence calculation
Main limitation: Transfers data between CPU and GPU every iteration

2. Step 4: Improving Data Transfer with Target Data Regions
This version addresses the data transfer bottleneck by:

Creating a persistent data region with target data
Transferring matrices A and b to the device only once
Only transferring the convergence value each iteration
Performing vector swap directly on the device
Transferring final results back only at the end
Typically achieves 5-7x speedup over the initial version

3. Step 5: Eliminating Branches (Branchless Implementation)
This version improves GPU execution efficiency by:

Replacing the if (i != j) conditional with an arithmetic expression
Using static_cast<TYPE>(i != j) as a multiplier (1 when true, 0 when false)
Eliminating thread divergence in GPU execution
Typically provides ~30% additional performance improvement

4. Step 6: Optimizing Memory Access Patterns (Coalescing)
This version optimizes memory access patterns for GPU architecture by:

Changing matrix indexing from A[i*Ndim + j] to A[j*Ndim + i]
Ensuring consecutive threads access consecutive memory locations
Enabling coalesced memory access for better memory bandwidth utilization
Typically delivers an additional ~50% performance improvement

Each of these scripts can be compiled with:



g++ -O3 -fopenmp -foffload=nvptx-none jacobi_solver.cpp -o jacobi_solver


./jacobi_solver       # Run with default size
./jacobi_solver 2000  # Run with size 2000
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// =============================================================================
// UNFUSED LayerNorm: 5 separate kernels, 5 global memory round-trips
// =============================================================================

// Kernel 1: Compute mean per row
__global__ void compute_mean(const float* input, float* mean, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += input[row * D + i];
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    // Block reduction via shared memory
    __shared__ float shared[32];  // one per warp
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize) {
        sum = shared[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (threadIdx.x == 0)
        mean[row] = sum / D;
}

// Kernel 2: Subtract mean
__global__ void subtract_mean(const float* input, const float* mean,
                               float* centered, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * D) {
        int row = idx / D;
        centered[idx] = input[idx] - mean[row];
    }
}

// Kernel 3: Compute variance
__global__ void compute_variance(const float* centered, float* var, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = centered[row * D + i];
        sum += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize) {
        sum = shared[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (threadIdx.x == 0)
        var[row] = sum / D;
}

// Kernel 4: Normalize
__global__ void normalize(const float* centered, const float* var,
                           float* normalized, int N, int D, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * D) {
        int row = idx / D;
        normalized[idx] = centered[idx] / sqrtf(var[row] + eps);
    }
}

// Kernel 5: Scale and shift (affine)
__global__ void scale_shift(const float* normalized, const float* gamma,
                             const float* beta, float* output, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * D) {
        int col = idx % D;
        output[idx] = gamma[col] * normalized[idx] + beta[col];
    }
}

// Launch all 5 unfused kernels
void layernorm_unfused(const float* input, const float* gamma, const float* beta,
                        float* output, float* mean, float* var, float* centered,
                        float* normalized, int N, int D, float eps) {
    int threads = 256;

    // Kernel 1: mean (one block per row)
    compute_mean<<<N, threads>>>(input, mean, N, D);

    // Kernel 2: subtract mean
    int total = N * D;
    int blocks = (total + threads - 1) / threads;
    subtract_mean<<<blocks, threads>>>(input, mean, centered, N, D);

    // Kernel 3: variance
    compute_variance<<<N, threads>>>(centered, var, N, D);

    // Kernel 4: normalize
    normalize<<<blocks, threads>>>(centered, var, normalized, N, D, eps);

    // Kernel 5: scale + shift
    scale_shift<<<blocks, threads>>>(normalized, gamma, beta, output, N, D);
}


// =============================================================================
// FUSED LayerNorm: ONE kernel, ONE global memory read, ONE write
// =============================================================================

__global__ void layernorm_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int D, float eps
) {
    // One block per row
    int row = blockIdx.x;
    if (row >= N) return;

    const float* row_in = input + row * D;
    float* row_out = output + row * D;

    int num_warps = blockDim.x / warpSize;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    __shared__ float shared[32];

    // ---- Pass 1: Compute mean ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        local_sum += row_in[i];
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    // Block-level reduction
    if (lane == 0) shared[warp_id] = local_sum;
    __syncthreads();

    // First warp reduces across warps
    float mean;
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x == 0) shared[0] = val / D;
    }
    __syncthreads();
    mean = shared[0];

    // ---- Pass 2: Compute variance ----
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_var += __shfl_down_sync(0xFFFFFFFF, local_var, offset);

    if (lane == 0) shared[warp_id] = local_var;
    __syncthreads();

    float variance;
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x == 0) shared[0] = val / D;
    }
    __syncthreads();
    variance = shared[0];

    // ---- Output: normalize + scale + shift in one pass ----
    float inv_std = rsqrtf(variance + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float normalized = (row_in[i] - mean) * inv_std;
        row_out[i] = gamma[i] * normalized + beta[i];
    }
}

void layernorm_fused(const float* input, const float* gamma, const float* beta,
                      float* output, int N, int D, float eps) {
    int threads = 256;
    layernorm_fused_kernel<<<N, threads>>>(input, gamma, beta, output, N, D, eps);
}


// =============================================================================
// Benchmarking
// =============================================================================

float benchmark_unfused(const float* input, const float* gamma, const float* beta,
                         float* output, int N, int D, float eps, int warmup, int iters) {
    // Allocate temporaries needed by unfused version
    float *mean, *var, *centered, *normalized;
    cudaMalloc(&mean, N * sizeof(float));
    cudaMalloc(&var, N * sizeof(float));
    cudaMalloc(&centered, N * D * sizeof(float));
    cudaMalloc(&normalized, N * D * sizeof(float));

    // Warmup
    for (int i = 0; i < warmup; i++)
        layernorm_unfused(input, gamma, beta, output, mean, var, centered, normalized, N, D, eps);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; i++)
        layernorm_unfused(input, gamma, beta, output, mean, var, centered, normalized, N, D, eps);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(mean);
    cudaFree(var);
    cudaFree(centered);
    cudaFree(normalized);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}

float benchmark_fused(const float* input, const float* gamma, const float* beta,
                       float* output, int N, int D, float eps, int warmup, int iters) {
    // Warmup
    for (int i = 0; i < warmup; i++)
        layernorm_fused(input, gamma, beta, output, N, D, eps);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; i++)
        layernorm_fused(input, gamma, beta, output, N, D, eps);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}


// =============================================================================
// Main
// =============================================================================

bool verify(float* a, float* b, int n, float tol = 1e-3f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            printf("Mismatch at %d: %f vs %f (diff=%f)\n", i, a[i], b[i], fabsf(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

int main() {
    // Simulate a transformer layer: batch*seq_len=2048 rows, hidden_dim=4096
    int N = 2048;   // number of rows (batch × seq_len)
    int D = 4096;   // hidden dimension
    float eps = 1e-5f;
    int warmup = 5;
    int iters = 20;

    size_t data_bytes = N * D * sizeof(float);
    size_t param_bytes = D * sizeof(float);

    // Host allocations
    float* h_input = (float*)malloc(data_bytes);
    float* h_gamma = (float*)malloc(param_bytes);
    float* h_beta = (float*)malloc(param_bytes);
    float* h_output_unfused = (float*)malloc(data_bytes);
    float* h_output_fused = (float*)malloc(data_bytes);

    // Initialize with random data
    srand(42);
    for (int i = 0; i < N * D; i++)
        h_input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < D; i++) {
        h_gamma[i] = 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_beta[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // Device allocations
    float *d_input, *d_gamma, *d_beta, *d_output_unfused, *d_output_fused;
    cudaMalloc(&d_input, data_bytes);
    cudaMalloc(&d_gamma, param_bytes);
    cudaMalloc(&d_beta, param_bytes);
    cudaMalloc(&d_output_unfused, data_bytes);
    cudaMalloc(&d_output_fused, data_bytes);

    cudaMemcpy(d_input, h_input, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_bytes, cudaMemcpyHostToDevice);

    // ---- Benchmark ----
    printf("LayerNorm Fusion Benchmark\n");
    printf("Rows (N): %d, Hidden dim (D): %d\n", N, D);
    printf("Data size: %.1f MB\n", (float)data_bytes / 1024 / 1024);
    printf("Warmup: %d, Iterations: %d\n\n", warmup, iters);

    printf("Running unfused benchmark...\n"); fflush(stdout);
    float ms_unfused = benchmark_unfused(d_input, d_gamma, d_beta, d_output_unfused,
                                          N, D, eps, warmup, iters);
    printf("Unfused done: %.3f ms\n", ms_unfused); fflush(stdout);

    printf("Running fused benchmark...\n"); fflush(stdout);
    float ms_fused = benchmark_fused(d_input, d_gamma, d_beta, d_output_fused,
                                      N, D, eps, warmup, iters);
    printf("Fused done: %.3f ms\n", ms_fused); fflush(stdout);

    // ---- Verify correctness ----
    cudaMemcpy(h_output_unfused, d_output_unfused, data_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fused, d_output_fused, data_bytes, cudaMemcpyDeviceToHost);

    bool correct = verify(h_output_unfused, h_output_fused, N * D);

    // ---- Report ----
    printf("Results:\n");
    printf("  Unfused (5 kernels): %.3f ms\n", ms_unfused);
    printf("  Fused   (1 kernel):  %.3f ms\n", ms_fused);
    printf("  Speedup:             %.2fx\n", ms_unfused / ms_fused);
    printf("  Correctness:         %s\n", correct ? "PASS ✓" : "FAIL ✗");

    // Memory traffic analysis
    float unfused_bytes = (float)data_bytes * 9;  // input read 3x, centered read 2x, 
                                                    // normalized read 1x, writes for each
    float fused_bytes = (float)data_bytes * 2;     // one read + one write
    printf("\nMemory traffic (estimated):\n");
    printf("  Unfused: %.0f MB (input read multiple times + intermediates)\n",
           unfused_bytes / 1024 / 1024);
    printf("  Fused:   %.0f MB (one read + one write)\n",
           fused_bytes / 1024 / 1024);
    printf("  Reduction: %.1fx less memory traffic\n", unfused_bytes / fused_bytes);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output_unfused);
    cudaFree(d_output_fused);
    free(h_input);
    free(h_gamma);
    free(h_beta);
    free(h_output_unfused);
    free(h_output_fused);

    return 0;
}

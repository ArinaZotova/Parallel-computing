%%writefile matmul.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;

const int N = 1024;
const int BLOCK_SIZE = 16;

__global__ void MatrMulClassic(float *a, float *b, float *c, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void MatrMulBlock(float *a, float *b, float *c, int n) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int m = 0; m < n / BLOCK_SIZE; ++m) {
        shared_A[ty][tx] = a[row * n + (m * BLOCK_SIZE + tx)];
        shared_B[ty][tx] = b[(m * BLOCK_SIZE + ty) * n + col];
        
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

float calcGFLOPS(float time_ms, int n) {
    double ops = 2.0 * n * n * n; 
    return (ops / (time_ms / 1000.0)) / 1e9;
}

int main() {
    size_t size = N * N * sizeof(float);

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C_gpu_classic = new float[N * N];
    float *h_C_gpu_block = new float[N * N];

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float time_classic, time_block;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    cout << "Сравнение алгоритмов умножения матриц (" << N << "x" << N << ")" << endl;

    cudaEventRecord(start);
    MatrMulClassic<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_classic, start, stop);
    
    cudaMemcpy(h_C_gpu_classic, d_C, size, cudaMemcpyDeviceToHost);
    
    cout << "[GPU] Классическое умножение:" << endl;
    cout << "Время: " << time_classic << " мс \t Производительность: " << calcGFLOPS(time_classic, N) << " GFLOPS" << endl;

    cudaEventRecord(start);
    MatrMulBlock<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_block, start, stop);
    
    cudaMemcpy(h_C_gpu_block, d_C, size, cudaMemcpyDeviceToHost);
    
    cout << "[GPU] Блочное умножение (Shared Memory):" << endl;
    cout << "Время: " << time_block << " мс \t Производительность: " << calcGFLOPS(time_block, N) << " GFLOPS" << endl;
    
    cout << "Ускорение блочного метода: в " << time_classic / time_block << " раз(а)" << endl;

    delete[] h_A; delete[] h_B; delete[] h_C_gpu_classic; delete[] h_C_gpu_block;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

#include "kernel.h"
#include <cuda_runtime.h>

__global__ void VecAddKernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void runCudaVecAdd(float *a, float *b, float *c, int n) {
    float *dev_a, *dev_b, *dev_c;
    size_t size = n * sizeof(float);
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    VecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
}

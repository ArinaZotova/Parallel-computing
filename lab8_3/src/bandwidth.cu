#include <iostream>
#include <cuda_runtime.h>

using namespace std;

float calcBandwidth(size_t bytes, float time_ms) {
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
}

int main() {
    int N = 64 * 1024 * 1024;
    size_t size = N * sizeof(float);

    cout << "Размер передаваемых данных: " << size / (1024 * 1024) << " МБ" << endl;

    float *h_pageable, *h_pinned, *d_data;

    h_pageable = new float[N];
    cudaMallocHost((void**)&h_pinned, size);
    cudaMalloc((void**)&d_data, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms;

    cudaEventRecord(start);
    cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cout << "Pageable Host -> Device: \t" << calcBandwidth(size, time_ms) << " ГБ/с (" << time_ms << " мс)" << endl;

    cudaEventRecord(start);
    cudaMemcpy(h_pageable, d_data, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cout << "Pageable Device -> Host: \t" << calcBandwidth(size, time_ms) << " ГБ/с (" << time_ms << " мс)" << endl;

    cudaEventRecord(start);
    cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cout << "Pinned Host -> Device: \t\t" << calcBandwidth(size, time_ms) << " ГБ/с (" << time_ms << " мс)" << endl;

    cudaEventRecord(start);
    cudaMemcpy(h_pinned, d_data, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cout << "Pinned Device -> Host: \t\t" << calcBandwidth(size, time_ms) << " ГБ/с (" << time_ms << " мс)" << endl;

    delete[] h_pageable;
    cudaFreeHost(h_pinned);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

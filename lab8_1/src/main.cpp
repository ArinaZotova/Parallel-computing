#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "kernel.h"

using namespace std;

void vecAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

int main() {
    int N = 100000;
    float *a = new float[N], *b = new float[N], *c_cpu = new float[N], *c_gpu = new float[N];
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }
    vecAddCPU(a, b, c_cpu, N);
    runCudaVecAdd(a, b, c_gpu, N);
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i) maxError = max(maxError, abs(c_cpu[i] - c_gpu[i]));
    cout << "Max error: " << maxError << endl;
    if (maxError < 1e-5) cout << "Success!" << endl;
    delete[] a; delete[] b; delete[] c_cpu; delete[] c_gpu;
    return 0;
}

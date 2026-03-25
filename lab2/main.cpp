#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace std;

double calc_gflops(int N, double time_sec) {
    if (time_sec <= 0) return 0;
    return (2.0 * N * N * N) / (time_sec * 1e9);
}

void multiply_classic(int N, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) {
                s += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

void multiply_transpose(int N, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    vector<float> BT(N * N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BT[i * N + j] = B[j * N + i];
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) {
                s += A[i * N + k] * BT[j * N + k];
            }
            C[i * N + j] = s;
        }
    }
}

void multiply_buffer(int N, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    int M = 16; 
    vector<float> buffer(N);
    
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            buffer[k] = B[k * N + j];
        }
        
        for (int i = 0; i < N; i++) {
            float sum = 0.0f;
            int k = 0;
            for (; k <= N - M; k += M) {
                for (int step = 0; step < M; step++) {
                    sum += A[i * N + (k + step)] * buffer[k + step];
                }
            }
            for (; k < N; k++) {
                sum += A[i * N + k] * buffer[k];
            }
            C[i * N + j] = sum;
        }
    }
}

void multiply_block(int N, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    int S = 8;
    fill(C.begin(), C.end(), 0.0f);
    
    for (int i0 = 0; i0 < N; i0 += S) {
        for (int j0 = 0; j0 < N; j0 += S) {
            for (int k0 = 0; k0 < N; k0 += S) {
                for (int i = i0; i < min(i0 + S, N); i++) {
                    for (int j = j0; j < min(j0 + S, N); j++) {
                        float s = C[i * N + j];
                        for (int k = k0; k < min(k0 + S, N); k++) {
                            s += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = s;
                    }
                }
            }
        }
    }
}

int main() {
    
    vector<int> N_values = {128, 256, 512, 1024, 2048};
    
    cout << fixed << setprecision(4);
    cout << "Final comparison of all matrix multiplication algorithms\n";
    cout << setw(6) << "N" 
         << setw(14) << "Classic" 
         << setw(14) << "Transpose" 
         << setw(14) << "Buffer(M=16)" 
         << setw(14) << "Block" << "\n";

    for (int N : N_values) {
        vector<float> A(N * N, 1.0f), B(N * N, 2.0f), C(N * N);
        double t_class = 0, t_trans = 0, t_buf = 0, t_block = 0;

        auto start = chrono::high_resolution_clock::now();
        multiply_classic(N, A, B, C);
        auto end = chrono::high_resolution_clock::now();
        t_class = calc_gflops(N, chrono::duration<double>(end - start).count());

        start = chrono::high_resolution_clock::now();
        multiply_transpose(N, A, B, C);
        end = chrono::high_resolution_clock::now();
        t_trans = calc_gflops(N, chrono::duration<double>(end - start).count());

        start = chrono::high_resolution_clock::now();
        multiply_buffer(N, A, B, C);
        end = chrono::high_resolution_clock::now();
        t_buf = calc_gflops(N, chrono::duration<double>(end - start).count());

        start = chrono::high_resolution_clock::now();
        multiply_block(N, A, B, C);
        end = chrono::high_resolution_clock::now();
        t_block = calc_gflops(N, chrono::duration<double>(end - start).count());

        cout << setw(6) << N 
             << setw(14) << t_class 
             << setw(14) << t_trans 
             << setw(14) << t_buf 
             << setw(14) << t_block << "\n";
    }
    
    
    return 0;
}

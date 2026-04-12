#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>

using namespace std;

uint32_t fast_rand(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void test_cache(size_t max_size, size_t step, int mode) {
    int iters = 100;
    
    cout << "Size(KB);Time(ns)\n";

    for (size_t cur_size = step; cur_size <= max_size; cur_size += step) {
        size_t n = cur_size / sizeof(int);
        
        vector<int> arr(n, 1);
        vector<uint32_t> idx(n);
        
        if (mode == 3) {
            iota(idx.begin(), idx.end(), 0);
            mt19937 g(42);
            shuffle(idx.begin(), idx.end(), g);
        }

        uint32_t state = 42;
        volatile long long total_sum = 0; 

        auto start = chrono::high_resolution_clock::now();

        for (int k = 0; k < iters; ++k) {
            long long sum = 0;
            
            if (mode == 1) {
                for (size_t i = 0; i < n; ++i) {
                    sum += arr[i];
                }
            }
            else if (mode == 2) {
                for (size_t i = 0; i < n; ++i) {
                    sum += arr[fast_rand(state) % n];
                }
            }
            else if (mode == 3) {
                for (size_t i = 0; i < n; ++i) {
                    sum += arr[idx[i]];
                }
            }
            total_sum += sum;
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, nano> time_span = end - start;

        double time_per_el = time_span.count() / (n * iters);
        cout << cur_size / 1024 << ";" << fixed << setprecision(3) << time_per_el << "\n";
    }
}

int main() {
    int mode;
    cout << "Rezhim (1-posl, 2-sluch, 3-indeksi): ";
    cin >> mode;

    int range;
    cout << "Diapazon (1-L1/L2, 2-L3, 3-RAM): ";
    cin >> range;

    size_t kb = 1024;
    size_t mb = 1024 * 1024;
    size_t max_s = 0;
    size_t step_s = 0;

    if (range == 1) { 
        max_s = 2 * mb; 
        step_s = 1 * kb; 
    } else if (range == 2) { 
        max_s = 32 * mb; 
        step_s = 512 * kb; 
    } else { 
        max_s = 150 * mb; 
        step_s = 5 * mb; 
    }

    test_cache(max_s, step_s, mode);

    system("pause");
    return 0;
}

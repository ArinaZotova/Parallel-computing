#include <iostream>
#include <windows.h>
#include <intrin.h>

using namespace std;

double calculatePi(long long num_steps) {
    double step = 1.0 / (double)num_steps;
    double pi = 0.0;
    for (long long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        pi += 4.0 / (1.0 + x * x);
    }
    return pi * step;
}

int main() {
    setlocale(LC_ALL, "Russian");
    
    const long long NUM_STEPS = 100000000; 
    
    const int K = 5; 

    cout << "Алгоритм: Вычисление Пи (Шагов: " << NUM_STEPS << ")" << endl;
    cout << "Замер\tGetTickCount(мс)\tRDTSC(тактов)\tQPC(мс)" << endl;

    for (int k = 0; k < K; k++) {

        DWORD startTick = GetTickCount();
        volatile double pi1 = calculatePi(NUM_STEPS); 
        DWORD endTick = GetTickCount();
        DWORD timeTick = endTick - startTick;
        
        unsigned __int64 startTSC = __rdtsc();
        volatile double pi2 = calculatePi(NUM_STEPS);
        unsigned __int64 endTSC = __rdtsc();
        unsigned __int64 timeTSC = endTSC - startTSC;
        

        LARGE_INTEGER freq, startQPC, endQPC;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&startQPC);
        volatile double pi3 = calculatePi(NUM_STEPS);
        QueryPerformanceCounter(&endQPC);
        

        double timeQPC = (double)(endQPC.QuadPart - startQPC.QuadPart) * 1000.0 / freq.QuadPart;
        

        cout << k + 1 << "\t" 
             << timeTick << "\t\t\t" 
             << timeTSC << "\t\t" 
             << timeQPC << endl;
    }
    
    return 0;
}

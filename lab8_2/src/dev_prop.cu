%%writefile dev_prop.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        cout << "CUDA-совместимые устройства не найдены." << endl;
        return 1;
    }

    cout << "Найдено устройств: " << deviceCount << endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        cout << "Устройство №" << i << ": " << prop.name << endl;
        cout << "Вычислительная способность (Compute Capability): " << prop.major << "." << prop.minor << endl;
        cout << "Количество потоковых мультипроцессоров (SM): " << prop.multiProcessorCount << endl;
        cout << "Объем глобальной памяти: " << prop.totalGlobalMem / (1024 * 1024) << " МБ" << endl;
        cout << "Объем константной памяти: " << prop.totalConstMem / 1024 << " КБ" << endl;
        cout << "Объем разделяемой памяти на блок: " << prop.sharedMemPerBlock / 1024 << " КБ" << endl;
        cout << "Регистров на блок: " << prop.regsPerBlock << endl;
        cout << "Размер WARP'а: " << prop.warpSize << " потоков" << endl;
        cout << "Максимальное число потоков в блоке: " << prop.maxThreadsPerBlock << endl;
        cout << "Максимальные размеры блока: " 
             << prop.maxThreadsDim[0] << " x " 
             << prop.maxThreadsDim[1] << " x " 
             << prop.maxThreadsDim[2] << endl;
        cout << "Максимальные размеры сетки: " 
             << prop.maxGridSize[0] << " x " 
             << prop.maxGridSize[1] << " x " 
             << prop.maxGridSize[2] << endl;
        cout << "Тактовая частота ядра: " << prop.clockRate / 1000 << " МГц" << endl;
        cout << "Тактовая частота памяти: " << prop.memoryClockRate / 1000 << " МГц" << endl;
        cout << "Ширина шины памяти: " << prop.memoryBusWidth << " бит" << endl;
    }

    return 0;
}

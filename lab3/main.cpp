#include <iostream>
#include <bitset>
#include <cstring>

#if defined(__GNUC__) || defined(__GNUG__)
    #include <cpuid.h>
    void get_cpuid(int cpuInfo[4], int function_id) {
        __cpuid(function_id, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
    }
    void get_cpuidex(int cpuInfo[4], int function_id, int subfunction_id) {
        __cpuid_count(function_id, subfunction_id, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
    }
#else
    #include <intrin.h>
    void get_cpuid(int cpuInfo[4], int function_id) {
        __cpuid(cpuInfo, function_id);
    }
    void get_cpuidex(int cpuInfo[4], int function_id, int subfunction_id) {
        __cpuidex(cpuInfo, function_id, subfunction_id);
    }
#endif

using namespace std;

int main() {
    setlocale(LC_ALL, "Russian");
    
    int cpuInfo[4] = { -1 };

    cout << " Информация о процессоре (CPUID) " << endl;

    get_cpuid(cpuInfo, 0);
    int maxFunc = cpuInfo[0]; 
    
    char vendor[13];
    memset(vendor, 0, sizeof(vendor));
    *reinterpret_cast<int*>(vendor) = cpuInfo[1];     
    *reinterpret_cast<int*>(vendor + 4) = cpuInfo[3]; 
    *reinterpret_cast<int*>(vendor + 8) = cpuInfo[2]; 
    
    cout << "Производитель (Vendor ID): " << vendor << endl;
    cout << "Макс. базовая функция: " << maxFunc << endl;

    get_cpuid(cpuInfo, 0x80000000);
    unsigned int maxExt = cpuInfo[0]; 

    if (maxExt >= 0x80000004) {
        char brand[65];
        memset(brand, 0, sizeof(brand));
        get_cpuid(cpuInfo, 0x80000002);
        memcpy(brand, cpuInfo, sizeof(cpuInfo));
        
        get_cpuid(cpuInfo, 0x80000003);
        memcpy(brand + 16, cpuInfo, sizeof(cpuInfo));
        
        get_cpuid(cpuInfo, 0x80000004);
        memcpy(brand + 32, cpuInfo, sizeof(cpuInfo));
        
        cout << "Название процессора: " << brand << endl;
    }

    if (maxFunc >= 1) {
        get_cpuid(cpuInfo, 1);
        
        bitset<32> edx(cpuInfo[3]);
        bitset<32> ecx(cpuInfo[2]);

        cout << "\n Поддерживаемые технологии " << endl;
        cout << "FPU: " << (edx[0] ? "Да" : "Нет") << endl;
        cout << "MMX: " << (edx[23] ? "Да" : "Нет") << endl;
        cout << "SSE: " << (edx[25] ? "Да" : "Нет") << endl;
        cout << "SSE2: " << (edx[26] ? "Да" : "Нет") << endl;
        cout << "HTT: " << (edx[28] ? "Да" : "Нет") << endl;

        cout << "SSE3: " << (ecx[0] ? "Да" : "Нет") << endl;
        cout << "SSSE3: " << (ecx[9] ? "Да" : "Нет") << endl;
        cout << "FMA3: " << (ecx[12] ? "Да" : "Нет") << endl;
        cout << "SSE4.1: " << (ecx[19] ? "Да" : "Нет") << endl;
        cout << "SSE4.2: " << (ecx[20] ? "Да" : "Нет") << endl;
        cout << "AVX: " << (ecx[28] ? "Да" : "Нет") << endl;
    }

    if (maxFunc >= 7) {
        get_cpuidex(cpuInfo, 7, 0);
        bitset<32> ebx(cpuInfo[1]); 
        
        cout << "\n Дополнительные расширения " << endl;
        cout << "AVX2: " << (ebx[5] ? "Да" : "Нет") << endl;       
        cout << "AVX512-f: " << (ebx[16] ? "Да" : "Нет") << endl; 
        cout << "SHA: " << (ebx[29] ? "Да" : "Нет") << endl;      
    }

    system("pause"); 
    return 0;
}

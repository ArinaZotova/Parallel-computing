# Лабораторная работа 8.1
Разработка программ CUDA с использованием компилятора командной строки.

## Структура
- `src/kernel.cu` — CUDA-ядро и обертка.
- `src/kernel.h` — заголовочный файл.
- `src/main.cpp` — основной файл программы на C++.

## Сборка и запуск (Linux/Colab)
```bash
nvcc -c src/kernel.cu -o kernel.o
g++ -c src/main.cpp -Isrc/ -o main.o
nvcc kernel.o main.o -o main_app
./main_app

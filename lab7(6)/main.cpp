#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <immintrin.h>
#include <cstring>

void roberts_scalar(const unsigned char* src, unsigned char* dst, int w, int h) {
    for (int i = 0; i < h - 1; ++i) {
        for (int j = 0; j < w - 1; ++j) {
            float a = src[i * w + j];
            float b = src[i * w + j + 1];
            float c = src[(i + 1) * w + j];
            float d = src[(i + 1) * w + j + 1];

            float hh = a - d;
            float hv = b - c;

            float val = std::sqrt(hh * hh + hv * hv);
            if (val > 255.0f) val = 255.0f;

            dst[i * w + j] = static_cast<unsigned char>(val);
        }
    }
}

void roberts_simd(const unsigned char* src, unsigned char* dst, int w, int h) {
  
    __m128 max_val = _mm_set1_ps(255.0f);

    for (int i = 0; i < h - 1; ++i) {
        int j = 0;

        for (; j <= w - 1 - 4; j += 4) {
          
            int a_val, b_val, c_val, d_val;
            std::memcpy(&a_val, &src[i * w + j], 4);
            std::memcpy(&b_val, &src[i * w + j + 1], 4);
            std::memcpy(&c_val, &src[(i + 1) * w + j], 4);
            std::memcpy(&d_val, &src[(i + 1) * w + j + 1], 4);

            __m128i a_bytes = _mm_cvtsi32_si128(a_val);
            __m128i b_bytes = _mm_cvtsi32_si128(b_val);
            __m128i c_bytes = _mm_cvtsi32_si128(c_val);
            __m128i d_bytes = _mm_cvtsi32_si128(d_val);

            __m128i a_ints = _mm_cvtepu8_epi32(a_bytes);
            __m128i b_ints = _mm_cvtepu8_epi32(b_bytes);
            __m128i c_ints = _mm_cvtepu8_epi32(c_bytes);
            __m128i d_ints = _mm_cvtepu8_epi32(d_bytes);

            __m128 a_f = _mm_cvtepi32_ps(a_ints);
            __m128 b_f = _mm_cvtepi32_ps(b_ints);
            __m128 c_f = _mm_cvtepi32_ps(c_ints);
            __m128 d_f = _mm_cvtepi32_ps(d_ints);

            __m128 hh = _mm_sub_ps(a_f, d_f);
            __m128 hv = _mm_sub_ps(b_f, c_f);

            __m128 hh2 = _mm_mul_ps(hh, hh);
            __m128 hv2 = _mm_mul_ps(hv, hv);

            __m128 sum = _mm_add_ps(hh2, hv2);
            __m128 result = _mm_sqrt_ps(sum);

            result = _mm_min_ps(result, max_val);

            __m128i res_ints = _mm_cvttps_epi32(result);
            __m128i res_shorts = _mm_packus_epi32(res_ints, res_ints);
            __m128i res_bytes = _mm_packus_epi16(res_shorts, res_shorts);

            int final_result = _mm_cvtsi128_si32(res_bytes);
            std::memcpy(&dst[i * w + j], &final_result, 4);
        }

        for (; j < w - 1; ++j) {
            float a = src[i * w + j];
            float b = src[i * w + j + 1];
            float c = src[(i + 1) * w + j];
            float d = src[(i + 1) * w + j + 1];
            float hh = a - d;
            float hv = b - c;
            float val = std::sqrt(hh * hh + hv * hv);
            if (val > 255.0f) val = 255.0f;
            dst[i * w + j] = static_cast<unsigned char>(val);
        }
    }
}

int main() {
    int w = 2048;
    int h = 2048;

    std::vector<unsigned char> src(w * h, 100);
    std::vector<unsigned char> dst_scalar(w * h, 0);
    std::vector<unsigned char> dst_simd(w * h, 0);

    for (int i = 0; i < h; i++) {
        src[i * w + w / 2] = 255;
        src[i * w + w / 4] = 50;
    }

    std::cout << "Starting Scalar..." << std::endl;
    clock_t start_scalar = clock();
    roberts_scalar(src.data(), dst_scalar.data(), w, h);
    clock_t end_scalar = clock();
    double time_scalar = (double)(end_scalar - start_scalar) / CLOCKS_PER_SEC;

    std::cout << "Starting SIMD..." << std::endl;
    clock_t start_simd = clock();
    roberts_simd(src.data(), dst_simd.data(), w, h);
    clock_t end_simd = clock();
    double time_simd = (double)(end_simd - start_simd) / CLOCKS_PER_SEC;

    bool correct = true;
    for (int i = 0; i < w * h; ++i) {
        if (dst_scalar[i] != dst_simd[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Scalar Time: " << time_scalar << " seconds" << std::endl;
    std::cout << "SIMD Time:   " << time_simd << " seconds" << std::endl;
    std::cout << "Acceleration: " << time_scalar / time_simd << " x" << std::endl;
    std::cout << "Result Check: " << (correct ? "PASSED (Images are identical)" : "FAILED (Mismatch found)") << std::endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <stdint.h>
#include <immintrin.h>
#include <mmintrin.h>
#include <x86intrin.h>

int32_t dot_product_cpp(const int8_t* A, const int8_t* B, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<int16_t>(A[i]) * static_cast<int16_t>(B[i]);
    }
    return sum;
}

int32_t dot_product_mmx(const int8_t* A, const int8_t* B, int n) {
    __m64 v_sum = _mm_setzero_si64();
    __m64 v_zero = _mm_setzero_si64();
    for (int i = 0; i < n; i += 8) {
        __m64 a = *reinterpret_cast<const __m64*>(&A[i]);
        __m64 b = *reinterpret_cast<const __m64*>(&B[i]);
        __m64 sign_a = _mm_cmpgt_pi8(v_zero, a);
        __m64 a_lo = _mm_unpacklo_pi8(a, sign_a);
        __m64 a_hi = _mm_unpackhi_pi8(a, sign_a);
        __m64 sign_b = _mm_cmpgt_pi8(v_zero, b);
        __m64 b_lo = _mm_unpacklo_pi8(b, sign_b);
        __m64 b_hi = _mm_unpackhi_pi8(b, sign_b);
        __m64 prod_lo = _mm_madd_pi16(a_lo, b_lo);
        __m64 prod_hi = _mm_madd_pi16(a_hi, b_hi);
        v_sum = _mm_add_pi32(v_sum, prod_lo);
        v_sum = _mm_add_pi32(v_sum, prod_hi);
    }
    __m64 high = _mm_srli_si64(v_sum, 32);
    v_sum = _mm_add_pi32(v_sum, high);
    int32_t final_sum = _mm_cvtsi64_si32(v_sum);
    _mm_empty();
    return final_sum;
}

int32_t dot_product_sse2(const int8_t* A, const int8_t* B, int n) {
    __m128i v_sum = _mm_setzero_si128();
    __m128i v_zero = _mm_setzero_si128();
    for (int i = 0; i < n; i += 16) {
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&A[i]));
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&B[i]));
        __m128i sign_a = _mm_cmpgt_epi8(v_zero, a);
        __m128i a_lo = _mm_unpacklo_epi8(a, sign_a);
        __m128i a_hi = _mm_unpackhi_epi8(a, sign_a);
        __m128i sign_b = _mm_cmpgt_epi8(v_zero, b);
        __m128i b_lo = _mm_unpacklo_epi8(b, sign_b);
        __m128i b_hi = _mm_unpackhi_epi8(b, sign_b);
        __m128i prod_lo = _mm_madd_epi16(a_lo, b_lo);
        __m128i prod_hi = _mm_madd_epi16(a_hi, b_hi);
        v_sum = _mm_add_epi32(v_sum, prod_lo);
        v_sum = _mm_add_epi32(v_sum, prod_hi);
    }
    v_sum = _mm_add_epi32(v_sum, _mm_shuffle_epi32(v_sum, _MM_SHUFFLE(1, 0, 3, 2)));
    v_sum = _mm_add_epi32(v_sum, _mm_shuffle_epi32(v_sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v_sum);
}

int32_t dot_product_avx2(const int8_t* A, const int8_t* B, int n) {
    __m256i v_sum = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&A[i]));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[i]));
        __m128i a_lo128 = _mm256_castsi256_si128(a);
        __m128i b_lo128 = _mm256_castsi256_si128(b);
        __m256i a_lo = _mm256_cvtepi8_epi16(a_lo128);
        __m256i b_lo = _mm256_cvtepi8_epi16(b_lo128);
        __m128i a_hi128 = _mm256_extracti128_si256(a, 1);
        __m128i b_hi128 = _mm256_extracti128_si256(b, 1);
        __m256i a_hi = _mm256_cvtepi8_epi16(a_hi128);
        __m256i b_hi = _mm256_cvtepi8_epi16(b_hi128);
        __m256i prod_lo = _mm256_madd_epi16(a_lo, b_lo);
        __m256i prod_hi = _mm256_madd_epi16(a_hi, b_hi);
        v_sum = _mm256_add_epi32(v_sum, prod_lo);
        v_sum = _mm256_add_epi32(v_sum, prod_hi);
    }
    __m128i hi = _mm256_extracti128_si256(v_sum, 1);
    __m128i lo = _mm256_castsi256_si128(v_sum);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(lo);
}

int main() {
    const int N = 1024;
    const int ITERATIONS = 1000000; 
    std::vector<int8_t> A(N, 1);
    std::vector<int8_t> B(N, 1);

    unsigned long long start, end;
    volatile int32_t res;

    std::cout << "Starting benchmarks (N=" << N << ", " << ITERATIONS << " iterations)...\n\n";

    // 1. C++
    start = __rdtsc();
    for(int i = 0; i < ITERATIONS; i++) res = dot_product_cpp(A.data(), B.data(), N);
    end = __rdtsc();
    std::cout << "C++   | Ticks: " << (end - start) / ITERATIONS << "\n";

    // 2. MMX
    start = __rdtsc();
    for(int i = 0; i < ITERATIONS; i++) res = dot_product_mmx(A.data(), B.data(), N);
    end = __rdtsc();
    std::cout << "MMX   | Ticks: " << (end - start) / ITERATIONS << "\n";

    // 3. SSE2
    start = __rdtsc();
    for(int i = 0; i < ITERATIONS; i++) res = dot_product_sse2(A.data(), B.data(), N);
    end = __rdtsc();
    std::cout << "SSE2  | Ticks: " << (end - start) / ITERATIONS << "\n";

    // 4. AVX2
    start = __rdtsc();
    for(int i = 0; i < ITERATIONS; i++) res = dot_product_avx2(A.data(), B.data(), N);
    end = __rdtsc();
    std::cout << "AVX2  | Ticks: " << (end - start) / ITERATIONS << "\n";

    return 0;
}

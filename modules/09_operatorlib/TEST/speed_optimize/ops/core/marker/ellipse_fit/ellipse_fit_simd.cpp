/**
 * @file ellipse_fit_simd.cpp
 * @brief E2 内点扫描 SSE2 实现——表达式序与 ellipse_fit_cpu.cpp 标量原式逐句对应
 */

#include "ellipse_fit_simd.h"

#include <immintrin.h>
#include <cstddef>
#include <cmath>

namespace calib {

namespace {

// 标量原式（与 ellipse_fit_cpu.cpp RANSAC/nearPts 扫描逐字符同式）——SIMD 主体
// 与尾部余数共用同一数值契约
inline bool inlierScalar(const cv::Point2f& pt, float cx, float cy,
                         float cosA, float sinA, float a2, float b2, float thr)
{
    float dx = pt.x - cx;
    float dy = pt.y - cy;
    float x_rot = dx * cosA + dy * sinA;
    float y_rot = -dx * sinA + dy * cosA;
    float dist = std::abs((x_rot * x_rot) / a2 + (y_rot * y_rot) / b2 - 1.0f);
    return dist < thr;
}

} // anonymous namespace

int countInliersSimd(const cv::Point2f* pts, int n,
                     float cx, float cy, float cosA, float sinA,
                     float a2, float b2, float thr,
                     uint8_t* mask)
{
    int count = 0;
    int k = 0;

    const __m128 vcos = _mm_set1_ps(cosA);
    const __m128 vsin = _mm_set1_ps(sinA);
    const __m128 vcx = _mm_set1_ps(cx);
    const __m128 vcy = _mm_set1_ps(cy);
    const __m128 va2 = _mm_set1_ps(a2);
    const __m128 vb2 = _mm_set1_ps(b2);
    const __m128 vthr = _mm_set1_ps(thr);
    const __m128 vone = _mm_set1_ps(1.0f);
    const __m128 signMask = _mm_set1_ps(-0.0f);   // abs：清符号位（精确）

    for (; k + 4 <= n; k += 4) {
        // AoS→SoA 解交织（128-bit 两路 shuffle）
        const __m128 xy0 = _mm_loadu_ps(&pts[k].x);       // x0 y0 x1 y1
        const __m128 xy1 = _mm_loadu_ps(&pts[k + 2].x);   // x2 y2 x3 y3
        const __m128 vx = _mm_shuffle_ps(xy0, xy1, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 vy = _mm_shuffle_ps(xy0, xy1, _MM_SHUFFLE(3, 1, 3, 1));

        const __m128 dx = _mm_sub_ps(vx, vcx);
        const __m128 dy = _mm_sub_ps(vy, vcy);
        // x_rot = dx*cos + dy*sin（两乘一加，与标量同序；加法操作数序对舍入无关）
        const __m128 xr = _mm_add_ps(_mm_mul_ps(dx, vcos), _mm_mul_ps(dy, vsin));
        // y_rot = -dx*sin + dy*cos（取负＝清符号位，精确）
        const __m128 negdx = _mm_xor_ps(dx, signMask);
        const __m128 yr = _mm_add_ps(_mm_mul_ps(negdx, vsin), _mm_mul_ps(dy, vcos));

        const __m128 x2 = _mm_mul_ps(xr, xr);
        const __m128 y2 = _mm_mul_ps(yr, yr);
        const __m128 t1 = _mm_div_ps(x2, va2);   // 真除（IEEE 逐元素舍入）
        const __m128 t2 = _mm_div_ps(y2, vb2);
        const __m128 d = _mm_sub_ps(_mm_add_ps(t1, t2), vone);
        const __m128 absd = _mm_andnot_ps(signMask, d);

        const __m128 cmp = _mm_cmplt_ps(absd, vthr);   // 有序 LT（NaN→0）
        const unsigned bits = static_cast<unsigned>(_mm_movemask_ps(cmp));
        if (mask) {
            mask[k]     = static_cast<uint8_t>((bits >> 0) & 1u);
            mask[k + 1] = static_cast<uint8_t>((bits >> 1) & 1u);
            mask[k + 2] = static_cast<uint8_t>((bits >> 2) & 1u);
            mask[k + 3] = static_cast<uint8_t>((bits >> 3) & 1u);
        }
        count += static_cast<int>((bits & 1u) + ((bits >> 1) & 1u)
                                  + ((bits >> 2) & 1u) + ((bits >> 3) & 1u));
    }

    // 尾部：与标量原式逐字符相同的表达式
    for (; k < n; ++k) {
        const bool inl = inlierScalar(pts[k], cx, cy, cosA, sinA, a2, b2, thr);
        if (mask) mask[k] = inl ? 1 : 0;
        if (inl) ++count;
    }
    return count;
}

} // namespace calib

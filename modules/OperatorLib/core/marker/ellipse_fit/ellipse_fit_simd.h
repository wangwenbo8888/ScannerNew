/**
 * @file ellipse_fit_simd.h
 * @brief 椭圆拟合内点扫描 SSE2 向量化（E2）——与标量逐位一致的判定位实现
 *
 * 数值契约（对拍红线）：
 *  - 逐点独立（无跨点归约）→ lane＝点的 SIMD 切法位同
 *  - divps 保留真除（IEEE-754 逐元素舍入＝标量 float/float；禁倒数乘）
 *  - mulps/subps/addps 分立（SSE2 无 FMA；表达式求值序与标量逐句对应）
 *  - cmplt 有序比较：NaN 两边均 false＝标量 dist<thr 行为
 *  - 尾部（n%4）走与标量原式逐字符相同的余数处理
 */

#pragma once

#include <opencv2/core/types.hpp>
#include <cstdint>

namespace calib {

/// 对 n 个点计算 |x_rot²/a2 + y_rot²/b2 − 1| < thr 的内点数；
/// mask 非空时逐点写 0/1（与标量判定逐位同）。SSE2（x64 基线，无 /arch 依赖）。
int countInliersSimd(const cv::Point2f* pts, int n,
                     float cx, float cy, float cosA, float sinA,
                     float a2, float b2, float thr,
                     uint8_t* mask);

} // namespace calib

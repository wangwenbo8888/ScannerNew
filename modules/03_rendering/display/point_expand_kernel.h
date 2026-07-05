#pragma once

#include <cstddef>

namespace calib {

// ============================================================
// CUDA 展点 Kernel (位置+法线) -> 法线朝向四边形 (4 顶点)
// ============================================================

// 展点（无剔除，全部展开）
void launchExpandLaserPoints(
    float*        d_outPos,
    float*        d_outUv,
    float*        d_outNormal,
    const float*  d_xyz,
    const float*  d_normal,
    float         halfSize,
    size_t        pointCount,
    void*         stream);

} // namespace calib

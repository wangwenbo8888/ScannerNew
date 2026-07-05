#pragma once

#include <cmath>

namespace calib {

// ============================================================
// 展点数学共享模块 (header-only, 无依赖)
// laser_cloud_renderer / marker_cloud_renderer 共用
// ============================================================

// 展开后顶点 (32 bytes)
struct ExpandVertex {
    float x  = 0.0f;
    float y  = 0.0f;
    float z  = 0.0f;
    float u  = 0.0f;
    float v  = 0.0f;
    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 0.0f;
};

static_assert(sizeof(ExpandVertex) == 32, "ExpandVertex must be 32 bytes");

// ============================================================
// CPU 端展开: 点(位置+法线) -> 法线朝向四边形 (4 顶点)
// ============================================================

inline void expandMarkerPointCPU(
    ExpandVertex out[4],
    float px, float py, float pz,
    float nx, float ny, float nz,
    float halfSize)
{
    // 归一化法线
    float nLen = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (nLen < 1e-10f) { nx = 0.f; ny = 0.f; nz = 1.f; }
    else { nx /= nLen; ny /= nLen; nz /= nLen; }

    // 正交基
    float ux, uy, uz;
    if (std::fabs(ny) < 0.99f) { ux = 0.f; uy = 1.f; uz = 0.f; }
    else                        { ux = 1.f; uy = 0.f; uz = 0.f; }

    float Tx = uy*nz - uz*ny;
    float Ty = uz*nx - ux*nz;
    float Tz = ux*ny - uy*nx;
    float tLen = std::sqrt(Tx*Tx + Ty*Ty + Tz*Tz);
    if (tLen < 1e-10f) { Tx = 1.f; Ty = 0.f; Tz = 0.f; }
    else { Tx /= tLen; Ty /= tLen; Tz /= tLen; }

    float Bx = ny*Tz - nz*Ty;
    float By = nz*Tx - nx*Tz;
    float Bz = nx*Ty - ny*Tx;

    float hx = Tx*halfSize, hy = Ty*halfSize, hz = Tz*halfSize;
    float bx = Bx*halfSize, by = By*halfSize, bz = Bz*halfSize;

    out[0] = { px - hx - bx, py - hy - by, pz - hz - bz, -1.f, -1.f, nx, ny, nz };
    out[1] = { px + hx - bx, py + hy - by, pz + hz - bz,  1.f, -1.f, nx, ny, nz };
    out[2] = { px + hx + bx, py + hy + by, pz + hz + bz,  1.f,  1.f, nx, ny, nz };
    out[3] = { px - hx + bx, py - hy + by, pz - hz + bz, -1.f,  1.f, nx, ny, nz };
}

} // namespace calib

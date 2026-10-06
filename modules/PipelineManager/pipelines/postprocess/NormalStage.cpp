// ============================================================================
// NormalStage.cpp — 09 laser_cloud_normal_cpu 适配实现（决策与契约见头注）
// ============================================================================
#include "pipelines/postprocess/NormalStage.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "scanning/fusion/laser_cloud_fuse/laser_cloud_fuse_cpu.h"
#include "scanning/fusion/laser_cloud_normal/laser_cloud_normal_cpu.h"

namespace Scanner::pipeline {
namespace {

/// 自适应体素边长：cbrt(包围盒体积/N)，夹 [diag/1024, diag/32]（防极端云
/// 退化成单体素或百万碎体素；单位球 1000 点 → diag/32≈0.11，5×5×5 邻域
/// 跨度 ~0.54，PCA 中位误差实测 < 3°）
float chooseVoxelSize(const std::vector<float>& xyz) {
    float minx = xyz[0], maxx = xyz[0];
    float miny = xyz[1], maxy = xyz[1];
    float minz = xyz[2], maxz = xyz[2];
    for (size_t i = 3; i < xyz.size(); i += 3) {
        minx = std::fmin(minx, xyz[i]);      maxx = std::fmax(maxx, xyz[i]);
        miny = std::fmin(miny, xyz[i + 1]);  maxy = std::fmax(maxy, xyz[i + 1]);
        minz = std::fmin(minz, xyz[i + 2]);  maxz = std::fmax(maxz, xyz[i + 2]);
    }
    const float ex = maxx - minx, ey = maxy - miny, ez = maxz - minz;
    const double diag = std::sqrt(static_cast<double>(ex) * ex +
                                  static_cast<double>(ey) * ey +
                                  static_cast<double>(ez) * ez);
    if (diag <= 0.0)
        return 1.0f;                        // 全同点：任意正体素（单体素 fallback 如实降级）
    const double lo = diag / 1024.0, hi = diag / 32.0;
    const double v = std::cbrt(static_cast<double>(ex) * ey * ez /
                               static_cast<double>(xyz.size() / 3));
    return static_cast<float>(std::min(std::max(v, lo), hi));
}

} // namespace

Result NormalStage::run(MeshData& io, const CancelToken& cancel) {
    const size_t n = io.pointCount();
    if (n == 0)
        return Result::fail("NormalStage: 空点云，无法重算法线");
    if (cancel.cancelled())
        return Result::degraded("NormalStage 已取消（执行前检查点）");

    // —— xyz(host float 数组) → 临时融合累积器（09 算子输入形态；每 run 新建，
    //    累积器自持久化不可跨 run 复用）——
    std::vector<cv::Point3f> pts(n);
    for (size_t i = 0; i < n; ++i)
        pts[i] = cv::Point3f(io.xyz[3 * i], io.xyz[3 * i + 1], io.xyz[3 * i + 2]);

    calib::LaserCloudFuseCPUParams fp;
    fp.voxelSize = chooseVoxelSize(io.xyz);
    calib::LaserCloudFuseCPU fuse(fp);
    fuse.Execute(pts);

    const size_t voxels = fuse.GetFusedPointCount();
    if (voxels == 0)
        return Result::fail("NormalStage: 融合后零体素");

    // —— 全体素法线重算（GBA 重融合已更新位置故重算全云）——
    calib::LaserCloudNormalCPUParams np;
    np.kernelRadius = 2;    // 5×5×5 邻域（千点级密度下 3×3×3 常不足 minNeighbors）
    np.minNeighbors = 3;
    calib::LaserCloudNormalCPU normalOp(np);
    auto nr = normalOp.Execute(fuse, 0, voxels);
    if (!nr.success)
        return Result::fail("NormalStage: 09 法线算子失败——" + nr.message);

    // —— 逐点映射回写：取就近体素代表点法线（所在体素必在 3×3×3 邻域内，
    //    代表点即该体素首个落点；就近即所在体素或贴面邻域——法向差异体素级）
    io.normals.assign(n * 3, 0.0f);
    std::vector<const calib::CloudPoint*> neigh;
    neigh.reserve(27);
    for (size_t i = 0; i < n; ++i) {
        if ((i & 1023) == 0 && cancel.cancelled())
            return Result::degraded("NormalStage 已取消（法线映射回写途中）");
        const cv::Point3f p(pts[i]);
        neigh.clear();
        fuse.GatherVoxelNeighbors(p, 1, neigh);
        if (neigh.empty())
            continue;                       // 防御：理论不可达（本帧灌入所在体素必有代表点）
        const calib::CloudPoint* best = neigh[0];
        float bestD2 = (best->x - p.x) * (best->x - p.x) +
                       (best->y - p.y) * (best->y - p.y) +
                       (best->z - p.z) * (best->z - p.z);
        for (size_t k = 1; k < neigh.size(); ++k) {
            const calib::CloudPoint* c = neigh[k];
            const float d2 = (c->x - p.x) * (c->x - p.x) +
                             (c->y - p.y) * (c->y - p.y) +
                             (c->z - p.z) * (c->z - p.z);
            if (d2 < bestD2) {
                bestD2 = d2;
                best = c;
            }
        }
        io.normals[3 * i] = best->nx;
        io.normals[3 * i + 1] = best->ny;
        io.normals[3 * i + 2] = best->nz;
    }

    // —— 质量如实：退化比例对齐算子阈值（<10% ok；≥10% degraded；≥50% 加重措辞）——
    const auto& st = nr.statistics;
    const size_t attempted = st.processedCount + st.fallbackCount;
    const double fallbackRate = attempted > 0
        ? static_cast<double>(st.fallbackCount) / static_cast<double>(attempted)
        : 0.0;
    char rate[32];
    std::snprintf(rate, sizeof(rate), "%.1f%%", fallbackRate * 100.0);
    const std::string stats = "points=" + std::to_string(n) +
                              " voxels=" + std::to_string(voxels) +
                              " fallback=" + std::to_string(st.fallbackCount) +
                              " (" + rate + ")";
    if (fallbackRate >= 0.5)
        return Result::degraded("NormalStage: 法线退化比例过高，多数点使用回退法线——" + stats);
    if (fallbackRate >= 0.1)
        return Result::degraded("NormalStage: 部分点邻域稀疏使用回退法线——" + stats);
    return Result::ok("NormalStage: 法线重算完成 " + stats);
}

} // namespace Scanner::pipeline

#pragma once
// ============================================================================
// bp_uniform_grid.h — 均匀体素栅格（PCMP HybridBallPivotV2 uniform_grid.h 移植）
// 体素=cellSize（通常最大球半径）→ 半径查询扫 3×3×3 体素 O(1)。
// Open3D 剥离：原文件本就零 Open3D 依赖，仅入命名空间。
// ============================================================================
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include <cstddef>

namespace calib {
namespace bp {

class UniformGrid {
public:
    UniformGrid() = default;

    void build(const std::vector<Eigen::Vector3d>& points, double cellSize) {
        points_ = &points;
        cellSize_ = cellSize;
        invCellSize_ = (cellSize > 0) ? 1.0 / cellSize : 1.0;

        size_t N = points.size();
        if (N == 0) return;

        bboxMin_ = points[0];
        Eigen::Vector3d bboxMax = points[0];
        for (const auto& p : points) {
            for (int d = 0; d < 3; ++d) {
                if (p(d) < bboxMin_(d)) bboxMin_(d) = p(d);
                if (p(d) > bboxMax(d)) bboxMax(d) = p(d);
            }
        }

        gridDim_.x() = static_cast<int>((bboxMax.x() - bboxMin_.x()) * invCellSize_) + 2;
        gridDim_.y() = static_cast<int>((bboxMax.y() - bboxMin_.y()) * invCellSize_) + 2;
        gridDim_.z() = static_cast<int>((bboxMax.z() - bboxMin_.z()) * invCellSize_) + 2;

        // 体素数上限保护（与沙盒同源：>5000 万体素自动增大 cellSize）
        const size_t MAX_VOXELS = 50000000;
        size_t numVoxels = static_cast<size_t>(gridDim_.x()) * gridDim_.y() * gridDim_.z();
        if (numVoxels > MAX_VOXELS) {
            Eigen::Vector3d extent = bboxMax - bboxMin_;
            double maxExtent = extent.maxCoeff();
            int MAX_DIM = 300;
            double safeCellSize = maxExtent / MAX_DIM;
            cellSize_ = std::max(cellSize_, safeCellSize);
            invCellSize_ = 1.0 / cellSize_;
            gridDim_.x() = static_cast<int>(extent.x() * invCellSize_) + 2;
            gridDim_.y() = static_cast<int>(extent.y() * invCellSize_) + 2;
            gridDim_.z() = static_cast<int>(extent.z() * invCellSize_) + 2;
            numVoxels = static_cast<size_t>(gridDim_.x()) * gridDim_.y() * gridDim_.z();
        }

        cells_.assign(numVoxels, {});
        for (size_t i = 0; i < N; ++i) {
            int cx, cy, cz;
            pointToCell(points[i], cx, cy, cz);
            cells_[cellLinearIndex(cx, cy, cz)].push_back(static_cast<int>(i));
        }
    }

    /// 半径查询（含 query 自身；接口兼容 KDTreeFlann::SearchRadius 子集用法）
    int searchRadius(const Eigen::Vector3d& query, double radius,
                     std::vector<int>& indices,
                     std::vector<double>& dists2) const {
        indices.clear();
        dists2.clear();
        if (!points_ || points_->empty()) return 0;

        double radius2 = radius * radius;
        int cx0, cy0, cz0, cx1, cy1, cz1;
        pointToCell(query - Eigen::Vector3d(radius, radius, radius), cx0, cy0, cz0);
        pointToCell(query + Eigen::Vector3d(radius, radius, radius), cx1, cy1, cz1);
        cx0 = clampInt(cx0, 0, gridDim_.x() - 1);
        cy0 = clampInt(cy0, 0, gridDim_.y() - 1);
        cz0 = clampInt(cz0, 0, gridDim_.z() - 1);
        cx1 = clampInt(cx1, 0, gridDim_.x() - 1);
        cy1 = clampInt(cy1, 0, gridDim_.y() - 1);
        cz1 = clampInt(cz1, 0, gridDim_.z() - 1);

        for (int cz = cz0; cz <= cz1; ++cz)
            for (int cy = cy0; cy <= cy1; ++cy)
                for (int cx = cx0; cx <= cx1; ++cx) {
                    const auto& cell = cells_[cellLinearIndex(cx, cy, cz)];
                    for (int ptIdx : cell) {
                        double dx = (*points_)[ptIdx].x() - query.x();
                        double dy = (*points_)[ptIdx].y() - query.y();
                        double dz = (*points_)[ptIdx].z() - query.z();
                        double d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 <= radius2) {
                            indices.push_back(ptIdx);
                            dists2.push_back(d2);
                        }
                    }
                }
        return static_cast<int>(indices.size());
    }

private:
    const std::vector<Eigen::Vector3d>* points_ = nullptr;
    double cellSize_ = 1.0;
    double invCellSize_ = 1.0;
    Eigen::Vector3d bboxMin_ = Eigen::Vector3d::Zero();
    Eigen::Vector3i gridDim_ = Eigen::Vector3i::Zero();
    std::vector<std::vector<int>> cells_;

    inline void pointToCell(const Eigen::Vector3d& p, int& cx, int& cy, int& cz) const {
        cx = static_cast<int>((p.x() - bboxMin_.x()) * invCellSize_);
        cy = static_cast<int>((p.y() - bboxMin_.y()) * invCellSize_);
        cz = static_cast<int>((p.z() - bboxMin_.z()) * invCellSize_);
    }
    inline size_t cellLinearIndex(int cx, int cy, int cz) const {
        return static_cast<size_t>(cz) * gridDim_.x() * gridDim_.y() +
               static_cast<size_t>(cy) * gridDim_.x() +
               static_cast<size_t>(cx);
    }
    static inline int clampInt(int v, int lo, int hi) {
        return (v < lo) ? lo : ((v > hi) ? hi : v);
    }
};

} // namespace bp
} // namespace calib

#include "laser_cloud_normal_cpu.h"
#include "common/calib_logging.h"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>

using namespace calib;

OperatorInfo getLaserCloudNormalCPUInfo() {
    return OperatorInfo{"LaserCloudNormalCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(03, LaserCloudNormalCPU);

// ============================================================
// 匿名命名空间：Jacobi 特征分解 + 协方差计算
// ============================================================

namespace {

/// 3x3 对称矩阵 Jacobi 特征分解。
/// 输入 cov[6] = {Cxx, Cxy, Cxz, Cyy, Cyz, Czz}
/// 输出 eigenvalues[3], eigenvectors[9]（列优先：eigenvectors[col*3+row]）
/// 返回最小特征值的索引
int jacobiEigen3x3(const float cov[6],
                   float eigenvalues[3],
                   float eigenvectors[9]) {
    float A[3][3] = {
        {cov[0], cov[1], cov[2]},
        {cov[1], cov[3], cov[4]},
        {cov[2], cov[4], cov[5]}
    };
    float V[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

    for (int sweep = 0; sweep < 12; ++sweep) {
        int p = 0, q = 1;
        float maxOff = std::fabs(A[0][1]);
        if (std::fabs(A[0][2]) > maxOff) { p = 0; q = 2; maxOff = std::fabs(A[0][2]); }
        if (std::fabs(A[1][2]) > maxOff) { p = 1; q = 2; maxOff = std::fabs(A[1][2]); }

        if (maxOff < 1e-12f) break;

        float app = A[p][p], aqq = A[q][q], apq = A[p][q];

        float t;
        float tau = (aqq - app) / (2.0f * apq);
        if (tau >= 0.0f)
            t = 1.0f / (tau + std::sqrt(1.0f + tau * tau));
        else
            t = -1.0f / (-tau + std::sqrt(1.0f + tau * tau));
        float c = 1.0f / std::sqrt(1.0f + t * t);
        float s = t * c;

        for (int i = 0; i < 3; ++i) {
            float aip = A[i][p], aiq = A[i][q];
            A[i][p] = c * aip - s * aiq;
            A[i][q] = s * aip + c * aiq;
        }
        for (int j = 0; j < 3; ++j) {
            float apj = A[p][j], aqj = A[q][j];
            A[p][j] = c * apj - s * aqj;
            A[q][j] = s * apj + c * aqj;
        }
        for (int i = 0; i < 3; ++i) {
            float vip = V[i][p], viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
        }
    }

    eigenvalues[0] = A[0][0];
    eigenvalues[1] = A[1][1];
    eigenvalues[2] = A[2][2];

    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
            eigenvectors[col * 3 + row] = V[row][col];

    int minIdx = 0;
    if (eigenvalues[1] < eigenvalues[0]) minIdx = 1;
    if (eigenvalues[2] < eigenvalues[minIdx]) minIdx = 2;
    return minIdx;
}

void computeCovariance3x3(const std::vector<const CloudPoint*>& pts, float cov[6]) {
    size_t n = pts.size();
    float cx = 0, cy = 0, cz = 0;
    for (size_t i = 0; i < n; ++i) {
        cx += pts[i]->x;
        cy += pts[i]->y;
        cz += pts[i]->z;
    }
    float invN = 1.0f / static_cast<float>(n);
    cx *= invN; cy *= invN; cz *= invN;

    float cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
    for (size_t i = 0; i < n; ++i) {
        float dx = pts[i]->x - cx;
        float dy = pts[i]->y - cy;
        float dz = pts[i]->z - cz;
        cxx += dx * dx; cyy += dy * dy; czz += dz * dz;
        cxy += dx * dy; cxz += dx * dz; cyz += dy * dz;
    }
    cov[0] = cxx * invN;
    cov[1] = cxy * invN;
    cov[2] = cxz * invN;
    cov[3] = cyy * invN;
    cov[4] = cyz * invN;
    cov[5] = czz * invN;
}

} // anonymous namespace

// ============================================================
// Params
// ============================================================

void LaserCloudNormalCPUParams::validate() const {
    if (kernelRadius < 1)
        throw std::invalid_argument("kernelRadius must be >= 1");
    if (minNeighbors < 3)
        throw std::invalid_argument("minNeighbors must be >= 3");
}

nlohmann::json LaserCloudNormalCPUParams::toJson() const {
    nlohmann::json j;
    j["kernelRadius"] = kernelRadius;
    j["minNeighbors"] = minNeighbors;
    j["fallbackNx"] = fallbackNx;
    j["fallbackNy"] = fallbackNy;
    j["fallbackNz"] = fallbackNz;
    return j;
}

LaserCloudNormalCPUParams LaserCloudNormalCPUParams::fromJson(const nlohmann::json& j) {
    LaserCloudNormalCPUParams p;
    if (j.contains("kernelRadius")) p.kernelRadius = j.at("kernelRadius").get<int>();
    if (j.contains("minNeighbors")) p.minNeighbors = j.at("minNeighbors").get<int>();
    if (j.contains("fallbackNx")) p.fallbackNx = j.at("fallbackNx").get<float>();
    if (j.contains("fallbackNy")) p.fallbackNy = j.at("fallbackNy").get<float>();
    if (j.contains("fallbackNz")) p.fallbackNz = j.at("fallbackNz").get<float>();
    return p;
}

// ============================================================
// Impl
// ============================================================

struct LaserCloudNormalCPU::Impl {
    LaserCloudNormalCPUParams params_;
    LaserCloudNormalCPUStats stats_;
    std::vector<const CloudPoint*> neighborBuf_;

    explicit Impl(const LaserCloudNormalCPUParams& p) : params_(p) {
        params_.validate();
        neighborBuf_.reserve(125);  // 5x5x5 max
    }

    LaserCloudNormalCPUResult computeImpl(LaserCloudFuseCPU& fuse,
                     size_t beginIdx, size_t endIdx) {
        LaserCloudNormalCPUResult result;
        LaserCloudNormalCPUStats stats;

        if (beginIdx >= endIdx) {
            result.success = true;
            result.message = "No new voxels";
            result.qualityFlag = calib::QualityFlag::Warning;
            result.statistics = stats;
            return result;
        }

        auto start = std::chrono::high_resolution_clock::now();

        const int initialRadius = params_.kernelRadius;
        const int minNeigh = params_.minNeighbors;

        for (size_t i = beginIdx; i < endIdx; ++i) {
            CloudPoint* cp = fuse.FusedPointPtr(i);
            cv::Point3f pos(cp->x, cp->y, cp->z);

            size_t nNeigh = fuse.GatherVoxelNeighbors(pos, initialRadius, neighborBuf_);

            // 自适应扩大
            if (static_cast<int>(nNeigh) < minNeigh && initialRadius < 4) {
                nNeigh = fuse.GatherVoxelNeighbors(pos, initialRadius + 1, neighborBuf_);
                ++stats.expandedCount;
            }

            // 退化处理
            if (static_cast<int>(nNeigh) < minNeigh) {
                cp->nx = params_.fallbackNx;
                cp->ny = params_.fallbackNy;
                cp->nz = params_.fallbackNz;
                ++stats.fallbackCount;
                continue;
            }

            // 协方差 + 特征分解
            float cov[6];
            computeCovariance3x3(neighborBuf_, cov);

            float eigenvalues[3], eigenvectors[9];
            int minIdx = jacobiEigen3x3(cov, eigenvalues, eigenvectors);

            float nx = eigenvectors[minIdx * 3 + 0];
            float ny = eigenvectors[minIdx * 3 + 1];
            float nz = eigenvectors[minIdx * 3 + 2];

            // 归一化
            float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-10f) {
                float invLen = 1.0f / len;
                cp->nx = nx * invLen;
                cp->ny = ny * invLen;
                cp->nz = nz * invLen;
                ++stats.processedCount;
            } else {
                cp->nx = params_.fallbackNx;
                cp->ny = params_.fallbackNy;
                cp->nz = params_.fallbackNz;
                ++stats.fallbackCount;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        stats.totalTimeMs = std::chrono::duration<double, std::milli>(end - start).count();

        result.success = true;
        result.message = "Normal estimation completed";

        size_t totalAttempted = stats.processedCount + stats.fallbackCount;
        double fallbackRate = totalAttempted > 0
            ? static_cast<double>(stats.fallbackCount) / static_cast<double>(totalAttempted)
            : 0.0;
        result.qualityFlag = fallbackRate < 0.1  ? calib::QualityFlag::Normal :
                             fallbackRate < 0.5  ? calib::QualityFlag::Degraded :
                                                  calib::QualityFlag::Warning;
        result.statistics = stats;
        stats_ = stats;

        CALIB_LOG_DEBUG("compute: voxels={} processed={} expanded={} fallback={} t={:.3f}ms",
                        endIdx - beginIdx, stats.processedCount, stats.expandedCount,
                        stats.fallbackCount, stats.totalTimeMs);

        return result;
    }
};

// ============================================================
// Constructor / Destructor
// ============================================================

LaserCloudNormalCPU::LaserCloudNormalCPU(const LaserCloudNormalCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserCloudNormalCPU initialized (kernelRadius={}, minNeighbors={})",
                   params.kernelRadius, params.minNeighbors);
}

LaserCloudNormalCPU::~LaserCloudNormalCPU() = default;

// ============================================================
// Execute
// ============================================================

LaserCloudNormalCPUResult LaserCloudNormalCPU::Execute(LaserCloudFuseCPU& fuse,
                                   size_t beginIdx, size_t endIdx) {
    return pImpl_->computeImpl(fuse, beginIdx, endIdx);
}

LaserCloudNormalCPUResult LaserCloudNormalCPU::Execute(LaserCloudFuseCPU& fuse,
                                   const LaserCloudFuseCPUResult& fuseResult) {
    size_t total = fuse.GetFusedPointCount();
    size_t begin = total - fuseResult.statistics.newVoxelCount;
    return pImpl_->computeImpl(fuse, begin, total);
}

// ============================================================
// Destroy
// ============================================================

void LaserCloudNormalCPU::Destroy() {
}

// ============================================================
// accessors
// ============================================================

void LaserCloudNormalCPU::SetParams(const LaserCloudNormalCPUParams& params) {
    params.validate();
    pImpl_->params_ = params;
}

const LaserCloudNormalCPUParams& LaserCloudNormalCPU::GetParams() const {
    return pImpl_->params_;
}

const LaserCloudNormalCPUStats& LaserCloudNormalCPU::GetStatistics() const noexcept {
    return pImpl_->stats_;
}

void LaserCloudNormalCPU::ResetStatistics() noexcept {
    pImpl_->stats_ = LaserCloudNormalCPUStats();
}

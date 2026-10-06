/**
 * @file pose_estimate_cpu.cpp
 * @brief 设备姿态CPU算子 - 实现文件
 *
 * 算法流程：
 *   1. 根据网格标记点 + 原点/轴配置建立世界坐标系变换 T_world
 *   2. 预计算目标姿态的4×4齐次矩阵 T_target[]
 *   3. 运行时接收相机R+T，变换到世界坐标系
 *   4. 对每个目标计算 Δ变换矩阵，提取位置误差和旋转误差
 *   5. 阈值判定，触发回调通知
 */

#include "pose_estimate_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <cmath>
#include <chrono>
#include <atomic>
#include <cassert>
#include <limits>

using namespace calib;

OperatorInfo getPoseEstimateCPUInfo() {
    return OperatorInfo{"PoseEstimateCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(13, PoseEstimateCPU);

// ============================================================
// 内部辅助函数
// ============================================================

namespace {

/** @brief 欧拉角(ZYX内旋,度) → 3×3旋转矩阵 */
cv::Matx33d eulerToMatrix(double rxDeg, double ryDeg, double rzDeg) {
    double rx = rxDeg * CV_PI / 180.0;
    double ry = ryDeg * CV_PI / 180.0;
    double rz = rzDeg * CV_PI / 180.0;

    double cx = std::cos(rx), sx = std::sin(rx);
    double cy = std::cos(ry), sy = std::sin(ry);
    double cz = std::cos(rz), sz = std::sin(rz);

    // R = Rz * Ry * Rx (ZYX intrinsic)
    return cv::Matx33d(
        cz*cy,  cz*sy*sx - sz*cx,  cz*sy*cx + sz*sx,
        sz*cy,  sz*sy*sx + cz*cx,  sz*sy*cx - cz*sx,
        -sy,    cy*sx,             cy*cx
    );
}

/** @brief 从3×3旋转矩阵提取旋转角度(度)
 *  使用迹公式：θ = arccos((trace(R) - 1) / 2)
 */
double rotationAngleDeg(const cv::Matx33d& R) {
    double trace = R(0, 0) + R(1, 1) + R(2, 2);
    double cosTheta = (trace - 1.0) / 2.0;
    // 钳位到 [-1, 1] 防止浮点误差导致 acos 返回 NaN
    cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
    double thetaRad = std::acos(cosTheta);
    return thetaRad * 180.0 / CV_PI;
}

/** @brief 构建4×4齐次变换矩阵 */
cv::Matx44d makeTransform(const cv::Matx33d& R, const cv::Vec3d& T) {
    cv::Matx44d M = cv::Matx44d::eye();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            M(i, j) = R(i, j);
        M(i, 3) = T(i);
    }
    return M;
}

/** @brief 4×4矩阵求逆(刚体变换快速求逆: R⁻¹=Rᵀ, T⁻¹=-RᵀT) */
cv::Matx44d invertTransform(const cv::Matx44d& M) {
    cv::Matx33d R;
    cv::Vec3d T;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            R(i, j) = M(i, j);
        T(i) = M(i, 3);
    }
    cv::Matx33d Rt = R.t();
    cv::Vec3d Tinv = -Rt * T;
    return makeTransform(Rt, Tinv);
}

/** @brief 从4×4矩阵提取平移向量和旋转矩阵 */
void extractFromTransform(const cv::Matx44d& M, cv::Vec3d& t, cv::Matx33d& R) {
    for (int i = 0; i < 3; ++i) {
        t(i) = M(i, 3);
        for (int j = 0; j < 3; ++j)
            R(i, j) = M(i, j);
    }
}

/** @brief 归一化向量 */
cv::Vec3d normalizeVec(const cv::Vec3d& v) {
    double n = std::sqrt(v(0)*v(0) + v(1)*v(1) + v(2)*v(2));
    if (n < 1e-15) return cv::Vec3d(0, 0, 1);
    return v / n;
}

} // anonymous namespace

// ============================================================
// PoseEstimateCPU::Impl
// ============================================================

struct PoseEstimateCPU::Impl {
    PoseEstimateCPUParams params_;
    PoseEstimateStats stats_;
    PoseCallback callback_;
    bool warmedUp_ = false;
    bool worldCoordBuilt_ = false;

    cv::Matx44d worldTransform_;                    // 原始坐标→世界坐标的变换
    std::vector<cv::Matx44d> targetTransforms_;     // 预计算的目标4×4矩阵

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const PoseEstimateCPUParams& params)
        : params_(params)
    {
        if (!params_.gridPoints.empty()) {
            params_.validate();
        }
    }

    ~Impl() = default;

    void buildWorldCoordinateSystem() {
        const auto& grid = params_.gridPoints;
        int oRow = params_.originRow;
        int oCol = params_.originCol;

        cv::Point3d origin = grid[oRow][oCol];

        // 行方向和列方向向量
        cv::Vec3d rowDir(
            grid[oRow + 1][oCol].x - origin.x,
            grid[oRow + 1][oCol].y - origin.y,
            grid[oRow + 1][oCol].z - origin.z
        );
        cv::Vec3d colDir(
            grid[oRow][oCol + 1].x - origin.x,
            grid[oRow][oCol + 1].y - origin.y,
            grid[oRow][oCol + 1].z - origin.z
        );

        rowDir = normalizeVec(rowDir);
        colDir = normalizeVec(colDir);

        cv::Vec3d xAxis, yAxis, zAxis;
        if (params_.rowAxis == "X") {
            xAxis = rowDir;
            yAxis = colDir;
        } else {
            xAxis = colDir;
            yAxis = rowDir;
        }

        // Z = X × Y（右手定则）
        zAxis = xAxis.cross(yAxis);
        zAxis = normalizeVec(zAxis);

        // 正交化：Y = Z × X
        yAxis = zAxis.cross(xAxis);
        yAxis = normalizeVec(yAxis);

        // 如果 faceNormal == "-Z"，翻转Z轴并重新正交化Y
        if (params_.faceNormal == "-Z") {
            zAxis = -zAxis;
            yAxis = zAxis.cross(xAxis);
            yAxis = normalizeVec(yAxis);
        }

        // 构建3×3旋转矩阵 [xAxis | yAxis | zAxis] 和4×4齐次矩阵
        cv::Matx33d R(
            xAxis(0), yAxis(0), zAxis(0),
            xAxis(1), yAxis(1), zAxis(1),
            xAxis(2), yAxis(2), zAxis(2)
        );
        cv::Vec3d T(origin.x, origin.y, origin.z);
        worldTransform_ = makeTransform(R, T);

        CALIB_LOG_INFO("World coordinate system built: origin=({:.2f},{:.2f},{:.2f}), rowAxis={}",
                       origin.x, origin.y, origin.z, params_.rowAxis);
    }

    void buildTargetTransforms() {
        targetTransforms_.clear();
        targetTransforms_.reserve(params_.poseTargets.size());
        for (const auto& target : params_.poseTargets) {
            cv::Matx33d Rtgt = eulerToMatrix(target.rx, target.ry, target.rz);
            cv::Vec3d Ttgt(target.tx, target.ty, target.tz);
            targetTransforms_.push_back(makeTransform(Rtgt, Ttgt));
        }
    }

    void ensureWorldCoordBuilt() {
        if (!worldCoordBuilt_) {
            buildWorldCoordinateSystem();
            buildTargetTransforms();
            worldCoordBuilt_ = true;
        }
    }

    PoseEstimateCPUResult ExecuteImpl(const cv::Matx44d& cameraPose) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "Concurrent call detected - NOT thread-safe!");
        struct ScopedFlag {
            std::atomic<bool>* flag;
            ScopedFlag(std::atomic<bool>* f) : flag(f) { flag->store(true); }
            ~ScopedFlag() { flag->store(false); }
        };
        ScopedFlag guard(&inProcess_);
#endif

        auto t0 = std::chrono::high_resolution_clock::now();

        PoseEstimateCPUResult result;
        stats_.estimateCallCount++;

        if (params_.gridPoints.empty()) {
            result.success = false;
            result.message = "No grid points configured";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        if (params_.poseTargets.empty()) {
            result.success = false;
            result.message = "No pose targets configured";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        ensureWorldCoordBuilt();

        // 当前姿态变换到世界坐标系：T_current = T_world × cameraPose
        cv::Matx44d currentPose = worldTransform_ * cameraPose;
        result.currentPose = currentPose;

        // 逐目标比较
        result.matches.resize(params_.poseTargets.size());
        bool anyMatched = false;
        int bestIdx = -1;
        double bestScore = std::numeric_limits<double>::max();

        for (size_t i = 0; i < params_.poseTargets.size(); ++i) {
            const auto& target = params_.poseTargets[i];
            const auto& T_target = targetTransforms_[i];

            // T_delta = T_target⁻¹ × T_current
            cv::Matx44d T_delta = invertTransform(T_target) * currentPose;

            cv::Vec3d deltaT;
            cv::Matx33d deltaR;
            extractFromTransform(T_delta, deltaT, deltaR);

            double posError = std::sqrt(deltaT(0)*deltaT(0) + deltaT(1)*deltaT(1) + deltaT(2)*deltaT(2));
            double rotError = rotationAngleDeg(deltaR);

            bool matched = (posError <= target.posThreshold) && (rotError <= target.rotThreshold);

            auto& match = result.matches[i];
            match.targetIndex = static_cast<int>(i);
            match.targetName = target.name;
            match.matched = matched;
            match.positionError = posError;
            match.rotationError = rotError;
            match.positionThreshold = target.posThreshold;
            match.rotationThreshold = target.rotThreshold;

            if (matched) {
                anyMatched = true;
                // 加权分数（归一化到阈值）
                double score = (posError * posError) / (target.posThreshold * target.posThreshold)
                             + (rotError * rotError) / (target.rotThreshold * target.rotThreshold);
                if (score < bestScore) {
                    bestScore = score;
                    bestIdx = static_cast<int>(i);
                }
            }
        }

        result.anyMatched = anyMatched;
        result.bestMatch = bestIdx;
        result.success = true;
        result.message = anyMatched ? "Pose matched" : "No target matched";
        result.qualityFlag = calib::QualityFlag::Normal;

        // 统计
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_.totalTimeMs += elapsedMs;
        stats_.targetCount = params_.poseTargets.size();
        stats_.matchedCount = 0;
        for (const auto& m : result.matches)
            if (m.matched) stats_.matchedCount++;
        result.statistics = stats_;

        CALIB_LOG_DEBUG("estimate(): matched={}, bestMatch={}, posErr={:.3f}mm, rotErr={:.3f}deg",
                        anyMatched, bestIdx,
                        bestIdx >= 0 ? result.matches[bestIdx].positionError : -1.0,
                        bestIdx >= 0 ? result.matches[bestIdx].rotationError : -1.0);

        // 触发回调（仅在匹配时）
        if (anyMatched && callback_) {
            try {
                callback_(result);
            } catch (...) {
                CALIB_LOG_WARN("Pose callback threw an exception");
            }
        }

        return result;
    }

    void SetParams(const PoseEstimateCPUParams& params) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() while processing - NOT thread-safe!");
#endif
        params_ = params;
        if (!params_.gridPoints.empty()) {
            params_.validate();
        }
        worldCoordBuilt_ = false;  // 标记需要重建
        warmedUp_ = false;
    }
};

// ============================================================
// PoseEstimateCPU 公开接口
// ============================================================

PoseEstimateCPU::PoseEstimateCPU(const PoseEstimateCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("PoseEstimateCPU initialized (grid={}x{}, targets={})",
                   params.gridPoints.empty() ? 0 : params.gridPoints.size(),
                   params.gridPoints.empty() ? 0 : params.gridPoints[0].size(),
                   params.poseTargets.size());
}

PoseEstimateCPU::~PoseEstimateCPU() = default;

PoseEstimateCPUResult PoseEstimateCPU::Execute(const cv::Matx33d& R, const cv::Vec3d& T) {
    CALIB_LOG_DEBUG("estimate(R,T) called");
    cv::Matx44d pose = makeTransform(R, T);
    return pImpl_->ExecuteImpl(pose);
}

PoseEstimateCPUResult PoseEstimateCPU::Execute(const cv::Matx44d& pose) {
    CALIB_LOG_DEBUG("estimate(Matx44d) called");
    return pImpl_->ExecuteImpl(pose);
}

void PoseEstimateCPU::SetCallback(PoseCallback callback) {
    pImpl_->callback_ = std::move(callback);
}

void PoseEstimateCPU::Warmup(int maxTargetCount) {
    pImpl_->warmedUp_ = true;
    CALIB_LOG_INFO("warmup() completed: maxTargetCount={}", maxTargetCount);
}

void PoseEstimateCPU::Warmup(const calib::WarmupConfig& config) {
    Warmup(config.maxPointCount);
}

void PoseEstimateCPU::SetParams(const PoseEstimateCPUParams& params) {
    CALIB_LOG_INFO("setParams() called");
    pImpl_->SetParams(params);
}

const PoseEstimateCPUParams& PoseEstimateCPU::GetParams() const {
    return pImpl_->params_;
}

const PoseEstimateStats& PoseEstimateCPU::GetStatistics() const noexcept {
    return pImpl_->stats_;
}

void PoseEstimateCPU::ResetStatistics() noexcept {
    pImpl_->stats_ = PoseEstimateStats{};
}

void PoseEstimateCPU::Destroy() {
}

/**
 * @file pose_estimate_cpu.h
 * @brief 设备姿态CPU算子 - 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：扫描仪姿态判�?�? * 平台：CPU
 *
 * 功能：根据标定板网格标记点建立世界坐标系，比较相机姿态与
 *       预设目标姿态，在阈值范围内触发通知�? *
 * 输入：用户配置的网格3D�?+ 目标姿态，运行时传入相机R+T
 * 输出：匹配结�?+ 回调通知
 *
 * 精度容差档次：档次②（~0.05mm 位置，~0.05° 角度�? */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct WarmupConfig;

// ============================================================
// PoseTarget �?单个目标姿�?// ============================================================

/**
 * @brief 单个目标姿�? *
 * 欧拉角约定：ZYX内旋（intrinsic），单位：度
 * 异常行为：fromJson() 未知字段 -> 忽略（不抛异常），保证前向兼�? */
struct PoseTarget {
    std::string name;
    double tx = 0.0;              ///< 目标平移X (mm)
    double ty = 0.0;              ///< 目标平移Y (mm)
    double tz = 0.0;              ///< 目标平移Z (mm)
    double rx = 0.0;              ///< 目标旋转X (�? ZYX欧拉�?
    double ry = 0.0;              ///< 目标旋转Y (�?
    double rz = 0.0;              ///< 目标旋转Z (�?
    double posThreshold = 10.0;   ///< 位置容差 (mm)
    double rotThreshold = 5.0;    ///< 旋转容差 (�?

    void validate() const {
        if (posThreshold <= 0.0)
            throw std::invalid_argument("PoseTarget::posThreshold must be > 0");
        if (rotThreshold <= 0.0)
            throw std::invalid_argument("PoseTarget::rotThreshold must be > 0");
    }

    nlohmann::json toJson() const {
        return {
            {"name", name},
            {"tx", tx}, {"ty", ty}, {"tz", tz},
            {"rx", rx}, {"ry", ry}, {"rz", rz},
            {"posThreshold", posThreshold},
            {"rotThreshold", rotThreshold}
        };
    }

    static PoseTarget fromJson(const nlohmann::json& j) {
        PoseTarget t;
        if (j.contains("name"))         t.name = j.at("name").get<std::string>();
        if (j.contains("tx"))           t.tx = j.at("tx").get<double>();
        if (j.contains("ty"))           t.ty = j.at("ty").get<double>();
        if (j.contains("tz"))           t.tz = j.at("tz").get<double>();
        if (j.contains("rx"))           t.rx = j.at("rx").get<double>();
        if (j.contains("ry"))           t.ry = j.at("ry").get<double>();
        if (j.contains("rz"))           t.rz = j.at("rz").get<double>();
        if (j.contains("posThreshold")) t.posThreshold = j.at("posThreshold").get<double>();
        if (j.contains("rotThreshold")) t.rotThreshold = j.at("rotThreshold").get<double>();
        return t;
    }
};

// ============================================================
// PoseEstimateCPUParams �?算子参数
// ============================================================

/**
 * @brief 设备姿态估计参�? */
struct PoseEstimateCPUParams {
    std::vector<std::vector<cv::Point3d>> gridPoints;  ///< 网格3D坐标(行×列)
    int originRow = 0;                  ///< 原点行索引
    int originCol = 0;                  ///< 原点列索引
    std::string rowAxis = "X";          ///< 行方向→"X"或"Y"
    std::string faceNormal = "Z";       ///< 面法线→"Z"�?-Z"
    std::vector<PoseTarget> poseTargets; ///< 目标姿态列表
    bool collectStatistics = true;

    void validate() const {
        if (gridPoints.empty())
            throw std::invalid_argument("gridPoints must not be empty");
        if (gridPoints.size() < 2)
            throw std::invalid_argument("gridPoints must have at least 2 rows");
        if (gridPoints[0].size() < 2)
            throw std::invalid_argument("gridPoints must have at least 2 columns");
        if (originRow < 0 || originRow >= static_cast<int>(gridPoints.size()))
            throw std::invalid_argument("originRow out of range");
        if (originCol < 0 || originCol >= static_cast<int>(gridPoints[0].size()))
            throw std::invalid_argument("originCol out of range");
        if (rowAxis != "X" && rowAxis != "Y")
            throw std::invalid_argument("rowAxis must be 'X' or 'Y'");
        if (faceNormal != "Z" && faceNormal != "-Z")
            throw std::invalid_argument("faceNormal must be 'Z' or '-Z'");
        // 检查所有行长度一致
        size_t cols = gridPoints[0].size();
        for (size_t i = 1; i < gridPoints.size(); ++i) {
            if (gridPoints[i].size() != cols)
                throw std::invalid_argument("gridPoints rows have inconsistent column counts");
        }
        // 原点不能在最后一�?列（需�?1方向建立轴）
        if (originRow + 1 >= static_cast<int>(gridPoints.size()))
            throw std::invalid_argument("originRow must not be on the last row (need row+1 for axis)");
        if (originCol + 1 >= static_cast<int>(gridPoints[0].size()))
            throw std::invalid_argument("originCol must not be on the last column (need col+1 for axis)");
        for (const auto& t : poseTargets)
            t.validate();
    }

    nlohmann::json toJson() const {
        nlohmann::json j;
        nlohmann::json gpArr = nlohmann::json::array();
        for (const auto& row : gridPoints) {
            nlohmann::json rowArr = nlohmann::json::array();
            for (const auto& pt : row) {
                rowArr.push_back({{"x", pt.x}, {"y", pt.y}, {"z", pt.z}});
            }
            gpArr.push_back(rowArr);
        }
        j["gridPoints"] = gpArr;
        j["originRow"] = originRow;
        j["originCol"] = originCol;
        j["rowAxis"] = rowAxis;
        j["faceNormal"] = faceNormal;
        nlohmann::json targetsArr = nlohmann::json::array();
        for (const auto& t : poseTargets)
            targetsArr.push_back(t.toJson());
        j["poseTargets"] = targetsArr;
        j["collectStatistics"] = collectStatistics;
        return j;
    }

    static PoseEstimateCPUParams fromJson(const nlohmann::json& j) {
        PoseEstimateCPUParams p;
        if (j.contains("gridPoints") && j.at("gridPoints").is_array()) {
            for (const auto& row : j.at("gridPoints")) {
                std::vector<cv::Point3d> rowPts;
                for (const auto& pt : row) {
                    double x = pt.contains("x") ? pt.at("x").get<double>() : 0.0;
                    double y = pt.contains("y") ? pt.at("y").get<double>() : 0.0;
                    double z = pt.contains("z") ? pt.at("z").get<double>() : 0.0;
                    rowPts.emplace_back(x, y, z);
                }
                p.gridPoints.push_back(std::move(rowPts));
            }
        }
        if (j.contains("originRow")) p.originRow = j.at("originRow").get<int>();
        if (j.contains("originCol")) p.originCol = j.at("originCol").get<int>();
        if (j.contains("rowAxis")) p.rowAxis = j.at("rowAxis").get<std::string>();
        if (j.contains("faceNormal")) p.faceNormal = j.at("faceNormal").get<std::string>();
        if (j.contains("poseTargets") && j.at("poseTargets").is_array()) {
            for (const auto& tj : j.at("poseTargets"))
                p.poseTargets.push_back(PoseTarget::fromJson(tj));
        }
        if (j.contains("collectStatistics"))
            p.collectStatistics = j.at("collectStatistics").get<bool>();
        return p;
    }
};

// ============================================================
// Result 数据结构
// ============================================================

/**
 * @brief 单目标匹配结�? */
struct PoseMatch {
    int targetIndex = -1;              ///< 目标索引
    std::string targetName;            ///< 目标名称
    bool matched = false;              ///< 是否匹配
    double positionError = 0.0;        ///< 位置误差 (mm)
    double rotationError = 0.0;        ///< 旋转误差 (�?
    double positionThreshold = 0.0;
    double rotationThreshold = 0.0;
};

/**
 * @brief 统计信息
 */
struct PoseEstimateStats {
    double totalTimeMs = 0.0;
    double coordBuildTimeMs = 0.0;
    double matchTimeMs = 0.0;
    size_t targetCount = 0;
    size_t matchedCount = 0;
    size_t estimateCallCount = 0;
};

/**
 * @brief 设备姿态估计结�? */
struct PoseEstimateCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    cv::Matx44d currentPose = cv::Matx44d::eye();
    std::vector<PoseMatch> matches;
    bool anyMatched = false;
    int bestMatch = -1;

    PoseEstimateStats statistics;

    PoseEstimateCPUResult() = default;
    ~PoseEstimateCPUResult() = default;

    PoseEstimateCPUResult(PoseEstimateCPUResult&&) = default;
    PoseEstimateCPUResult& operator=(PoseEstimateCPUResult&&) = default;

    PoseEstimateCPUResult(const PoseEstimateCPUResult&) = delete;
    PoseEstimateCPUResult& operator=(const PoseEstimateCPUResult&) = delete;
};

// ============================================================
// PoseEstimateCPU �?核心算子�?// ============================================================

/**
 * @brief 设备姿态CPU算子（CPU 实现�? *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 estimate() �?setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 * - Debug 模式下维�?inProcess_ 原子变量进行断言检�? * - Release 模式下并发调用为未定义行为（数据竞争�? */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 相机 R/T 按调用传入；SetParams 缓存网格点/目标姿态/回调等只读配置；另有统计遥测(getStatistics/resetStatistics)但不影响计算结果。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API PoseEstimateCPU {
public:
    static constexpr const char* kLogTag = "13-PoseEstimateCPU";

    explicit PoseEstimateCPU(const PoseEstimateCPUParams& params = {});
    ~PoseEstimateCPU();

    PoseEstimateCPU(const PoseEstimateCPU&) = delete;
    PoseEstimateCPU& operator=(const PoseEstimateCPU&) = delete;

    /**
     * @brief 评估相机姿�?�?传入R(3×3) + T(3×1)
     */
    PoseEstimateCPUResult Execute(const cv::Matx33d& R, const cv::Vec3d& T);

    PoseEstimateCPUResult Execute(const cv::Matx44d& pose);

    /**
     * @brief 回调类型定义
     */
    using PoseCallback = std::function<void(const PoseEstimateCPUResult&)>;

    /**
     * @brief 注册回调（匹配时触发�?     */
    void SetCallback(PoseCallback callback);

    void Warmup(int maxTargetCount);
    void Warmup(const WarmupConfig& config);

    void SetParams(const PoseEstimateCPUParams& params);
    const PoseEstimateCPUParams& GetParams() const;

    void Destroy();

    const PoseEstimateStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getPoseEstimateCPUInfo();

} // namespace calib
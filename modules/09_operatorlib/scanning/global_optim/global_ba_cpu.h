#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>
#include <utility>
#include "common/calib_types.h"
#include "common/quality_flag.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct WarmupConfig;

// 单帧观测:本机系3D点 + 全局ID(globalId 由前置全局ID模块赋予)
struct GlobalBAMarkerObs {
    cv::Point3d local;     // point_reconstruct 输出的本机系坐标(= 观测 z_ij)
    int         globalId = -1;  // 同物理标记点 = 同 globalId
};

struct GlobalBAFrame {
    uint64_t frameId = 0;
    cv::Matx33d R_init = cv::Matx33d::eye();   // 绝对位姿初值(相邻帧R/T链乘累积)
    cv::Vec3d  t_init = cv::Vec3d::zeros();
    std::vector<GlobalBAMarkerObs> markerObs;
};

struct GlobalBAInput {
    std::vector<GlobalBAFrame> frames;

    // P4-T24 高精度已有點软先验（客户端扫描流水线.md §5.3）:
    // 对命中 highPrecisionGlobalIds 的全局点 X 加残差 ‖X−X_existing‖²/σ²（σ 极小=高权重）,
    // 防止已有点被扫描观测挪动。X_existing 为原始全局系坐标（3n 展平 x,y,z）;
    // priorSigma 每点一个, 空/缺项回退 params.defaultPriorSigma。
    // id 无对应点 / X_existing 长度不足 → 忽略该 id 并 warn（不 fail, 鲁棒）。
    std::vector<int>    highPrecisionGlobalIds;  // 高精度点 globalId 集
    std::vector<double> X_existing;              // 对应先验位置（3n, 展平 x,y,z）
    std::vector<double> priorSigma;              // 对应 σ（每点一个; 空/缺则用 params 默认）
};

struct GlobalBAParams {
    bool   enablePoseGraphPreopt = true;
    double sigmaObserved         = 0.01;   // mm
    double tukeyC                = 0.03;   // =3σ
    double chiSquareGate         = 11.34;  // 3-DoF 99%
    double tolerance             = 1e-10;
    int    maxIterations         = 200;
    int    minCovisForLoopEdge   = 5;
    int    loopFrameGap          = 30;
    int    minPointsPerFrame     = 3;
    bool   centerOrigin          = true;
    // P4-T24: 高精度已有點软先验开关与默认 σ（单位 mm, 文档 §5.3 建议 0.001 量级;
    // σ 极小=高权重。useSoftPrior=false 或 ids 空 → 行为与无先验完全一致）
    bool   useSoftPrior          = true;
    double defaultPriorSigma     = 0.001;

    // ── 261004 真机事故加固：求解时长上限 + 取消钩子 ──
    //
    // 【为什么要这两个字段】261004 真机会话（2020 观测帧）终局遍在「GBA 批算」
    // 卡 5 分钟以上不返回：本算子最多跑 4 个 Ceres 阶段（plain / 预清洗重解 /
    // Tukey / chi2 重解），每阶段 maxIterations=200 且 tolerance=1e-10 极紧——
    // 容差实际不可能触发早退，等价于每阶段固定跑满 200 次迭代；千帧级问题的
    // 单次稀疏 Cholesky 迭代秒级 → 总时长无上界（数十分钟级）。且 Solve() 是
    // 不可打断的黑盒，用户关程序时 AppContext 析构 join 收尾线程 → 整个进程
    // 挂死在优化器里（实测挂 3 分钟+ 被 Task 杀）。
    //
    // 【修复策略】
    // 1) maxSolveSecondsPerPhase：每 Ceres 阶段（含 PGO 预优化）的时间上限。
    //    到时后 Ceres 以 NO_CONVERGENCE 终止并保留当前最优迭代值（参数块
    //    已更新——非 FAILURE 语义），下游按既有口径标记 Degraded 继续走，
    //    不会丢结果。0 = 不限（离线批量工具 gba_dataset_runner 可显式置 0
    //    换取充分收敛；生产线默认 60s 兜底）。
    // 2) cancelRequested：每次迭代末回调查询；返回 true → SOLVER_ABORT 中止
    //    求解（USER_FAILURE 语义=不采用半程解）。空函数=无取消源（离线工具/
    //    测试默认空）。上层（07）把 CancelToken 接进来后，关程序可在一次
    //    迭代内（秒级）打断优化器，进程不再挂死。
    double maxSolveSecondsPerPhase = 60.0;   // 秒；0=不限
    std::function<bool()> cancelRequested;   // 空=不可取消

    void validate() const;
    nlohmann::json toJson() const;
    static GlobalBAParams fromJson(const nlohmann::json& j);
};

struct GlobalBAStats {
    bool   loopDetected     = false;
    int    ceresIterations  = 0;
    double initialRMSE      = 0.0;   // mm
    double finalRMSE        = 0.0;   // mm
    std::vector<int> outlierObsIds;
    double loopClosureResidual = 0.0;
};

struct FramePose {
    uint64_t frameId = 0;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d  t = cv::Vec3d::zeros();
};

struct GlobalMarker {
    cv::Point3d X;
    int         globalId = -1;
    int         covisCount = 0;
};

struct GlobalBAResult {
    bool   success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<FramePose>    optimizedPoses;
    std::vector<GlobalMarker> optimizedMarkers;
    GlobalBAStats statistics;

    GlobalBAResult() = default;
    ~GlobalBAResult() = default;
    GlobalBAResult(GlobalBAResult&&) = default;
    GlobalBAResult& operator=(GlobalBAResult&&) = default;
    GlobalBAResult(const GlobalBAResult&) = delete;
    GlobalBAResult& operator=(const GlobalBAResult&) = delete;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 全量数据按调用传入,实例不持有跨调用累积状态;SetParams 缓存只读配置。
// 重置接口: N/A
// 并发策略: 每实例非线程安全,多实例并行各自独占
// ==============================
class SCANNER_API GlobalBundleAdjustmentCPU {
public:
    static constexpr const char* kLogTag = "13-GlobalBundleAdjustmentCPU";

    explicit GlobalBundleAdjustmentCPU(const GlobalBAParams& params = {});
    ~GlobalBundleAdjustmentCPU();

    GlobalBundleAdjustmentCPU(const GlobalBundleAdjustmentCPU&) = delete;
    GlobalBundleAdjustmentCPU& operator=(const GlobalBundleAdjustmentCPU&) = delete;

    GlobalBAResult Execute(const GlobalBAInput& input);

    void SetParams(const GlobalBAParams& params);
    const GlobalBAParams& GetParams() const;
    const GlobalBAStats& GetStatistics() const noexcept;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getGlobalBundleAdjustmentCPUInfo();

} // namespace calib

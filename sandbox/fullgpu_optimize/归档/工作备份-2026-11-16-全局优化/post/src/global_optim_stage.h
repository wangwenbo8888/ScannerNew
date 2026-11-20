#pragma once
// ============================================================================
// global_optim_stage.h — 回放收尾全局优化（GBA）阶段【2026-11-16 方案定稿】
// ============================================================================
// 职责（对应定稿：丙底座＋甲式重融合，三处修正）：
//   ①装配：obs 快照 → GlobalBAInput（gid≥0 过滤＋每帧观测数≥minPointsPerFrame，
//     不足整帧剔除并记 excludedFrames）
//   ②求解：GlobalBundleAdjustmentCPU（主树算子原样拷贝，零查表纯 CPU）
//   ③统计：规约对齐（Umeyama 无尺度，消除 centerOrigin 整体重定位差）后的
//     Δt/Δrot 分布（raw＋aligned 双份）＋R1C 门①②自动评估
//   ④产物：gba_result.json＋gba_trajectory.ply（优化前后双轨迹对照）＋控制台段
//   ⑤重融合：核心是纯函数可单测；重融合本体＝refuseWithOp()（Harness 持有的
//     LaserCloudFuseCuda 实例 Clear 后按优化位姿逐帧重 Execute——修正③对账断言
//     由调用方（Harness::gbaRefuse）先做）
// 排序契约（R3I 定谳）：主循环→shutdownExecutors(fc_.reset())→本阶段→现有导出块。
// fail-safe：config 无 globalOptim 节/enabled=false ＝ 本阶段完全不执行。
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>       // GpuMat/Stream（重融合用）

#include "gba_records.h"
#include "global_ba_cpu.h"             // ops/scanning/global_optim（vendor 原样拷贝）
#include "laser_cloud_fuse_cuda.h"     // ops/scanning/fusion（沙箱已在编）

namespace scr {

struct GbaPose {
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d T = cv::Vec3d(0, 0, 0);
    GbaPose() = default;
    GbaPose(const cv::Matx33d& r, const cv::Vec3d& t) : R(r), T(t) {}
};

struct GbaStageOptions {
    bool refuseLaser = true;          // false＝仅诊断（丙模式），不做重融合
    bool keepUnoptimizedFrames = true; // GBA 被剔帧按原位姿重融合（保帧覆盖）
    std::string outDir;               // 空＝不落盘（单测）
    std::string reportPath;           // 空＝<outDir>/gba_result.json
};

struct GbaDeltaStats {
    double medDt = 0, p95Dt = 0, maxDt = 0;   // mm
    double medDr = 0, p95Dr = 0, maxDr = 0;   // deg
    bool valid = false;
};

struct GbaStageResult {
    bool executed = false;      // 输入达到最低门槛（≥2 帧有效观测）
    bool success = false;       // GBA 求解成功
    std::string message;        // 失败/降级原因（qualityFlag 语义外的人话）
    int framesSent = 0;         // 送入 GBA 的帧数
    int framesKept = 0;         // GBA 保留（optimizedPoses）帧数
    std::vector<uint64_t> excludedFrames;  // 装配期剔除（观测不足）帧
    double initialRMSE = -1, finalRMSE = -1;  // mm（GlobalBAStats）
    GbaDeltaStats deltaRaw, deltaAligned;
    bool rmseGate = false, poseGate = false;  // R1C 门①②
    std::map<uint64_t, GbaPose> poses;        // frameId→优化位姿（success 时）
    nlohmann::json report;                    // gba_result.json 全文
};

/// 装配（纯函数，单测暴露）：gid<0 过滤＋每帧有效观测数门；不足帧整帧剔除
calib::GlobalBAInput assembleGbaInput(const std::vector<GbaObsRecord>& obs,
                                      const calib::GlobalBAParams& params,
                                      std::vector<uint64_t>& excludedFrames);

/// 位姿对分布统计（pairs＝(before, after)；dt=mm, drot=deg）
GbaDeltaStats poseDeltaStats(const std::vector<std::pair<GbaPose, GbaPose>>& pairs);

/// Umeyama（无尺度）刚体对齐：求把 after 平移点集对齐到 before 的 (Rg, tg)。
/// 返回后 aligned_after = Rg·after ＋ tg（旋转部分同理左乘 Rg）。
/// 对应数<3 返回恒等。
void gaugeAlign(const std::vector<std::pair<GbaPose, GbaPose>>& pairs,
                cv::Matx33d& Rg, cv::Vec3d& tg);

/// 阶段核心（纯 CPU，不含重融合）：装配→GBA→统计→门→json/轨迹 PLY→汇总段
GbaStageResult runGbaCore(const std::vector<GbaObsRecord>& obs,
                          const std::vector<GbaLaserRecord>* laser,  // 可空＝跳过对账
                          const calib::GlobalBAParams& params,
                          const GbaStageOptions& opts);

/// 双轨迹对照 PLY（前 N 红＋后 N 绿＋两条折线 edge）；N1≠N2 时按较短的画折线
bool writeGbaTrajectoryPly(const std::string& path,
                           const std::vector<GbaPose>& before,
                           const std::vector<GbaPose>& after);

/// 重融合统计
struct GbaRefuseStats {
    bool executed = false;        // Clear+重融合是否真的发生（false＝对账失败保原云）
    std::string reason;           // 未执行原因
    int framesFused = 0;          // 实际重融合帧数
    int framesNoPose = 0;         // GBA 无位姿的帧数（keepUnoptimized=true 时按原位姿融合）
    int framesTotal = 0;          // 激光档总帧数
};

/// 重融合本体（单测可直接喂算子实例；Harness::gbaRefuse 薄封装调用）：
/// ①Clear（算子自 Finish）②逐帧 1×N CV_32FC3 上传＋Execute(R,T,stream)
/// ③位姿缺帧按 keepUnoptimized 决定原位姿兜底/跳帧。
/// ⚠ 调用前置条件：五链执行器已销毁（fcActive_==false，保证单写者语义＋主流空闲）。
GbaRefuseStats refuseWithOp(calib::LaserCloudFuseCuda& op,
                            const std::vector<GbaLaserRecord>& laserArch,
                            const std::map<uint64_t, GbaPose>& poses,
                            bool keepUnoptimized,
                            cv::cuda::Stream& stream);

}  // namespace scr

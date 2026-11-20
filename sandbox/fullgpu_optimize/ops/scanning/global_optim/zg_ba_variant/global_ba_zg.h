#pragma once
// ============================================================================
// global_ba_zg.h — 全局平差备用变体（借对标软件 ZG 的求解思路, 接口与
//                 GlobalBundleAdjustmentCPU 同源数据面）
// ============================================================================
// 【定位】备用算子: 不替换 global_ba_cpu（三层鲁棒+PGO 主路），提供一条
// "ZG 思路"的单阶段参考实现, 用于 A/B 对比评测。借的三点（来源=对标复现
// 工程与逆向结论, 见 prodline_benchmark/ZG/operator_porting/bundle_adjustment）:
//   ① Schur 消点求解器: Ceres ITERATIVE_SCHUR + JACOBI 预条件（ZG BA_vc140u
//      2.0.0.0 实证配置; 主路 GBA 用 SPARSE_NORMAL_CHOLESKY）——标志点 BA
//      "每观测只碰 1 帧+1 点"正是消点结构；
//   ② 入网门控: 解算前按初始残差的稳健尺度（MAD）预剔粗差观测
//      （ZG 标定链"采帧准入/质量门家族"思想在 BA 侧的对应物）；
//   ③ 后验 σ₀ 报表: σ₀̂=sqrt(2C/dof)（单位权后验中误差）——ZG BA 三档报表
//      口径; 主路 GBA 只报 RMSE。
// 保留 ZG 单层 Huber（δ=4σ, BA_vc140u "Huber 4.0/1.0" σ 归一口径）作缺省,
// 与主路三层(plain→Tukey→χ²)形成对比面。
//
// 【接口】输入 GlobalBAInput/GlobalBAParams（与主路完全同型, 含 P4-T24 软先验
// 支持）; 输出 ZgBaResult = GlobalBAResult + ZG 借鉴项报表。不动现有任何文件。
// ============================================================================
#include <string>
#include <vector>

#include "global_ba_cpu.h"  // GlobalBAInput/GlobalBAParams/GlobalBAResult/GlobalBAStats（共用类型）

namespace calib {

// ZG 借鉴项开关与常数（求解主参数沿用 GlobalBAParams:
// sigmaObserved/maxIterations/tolerance/cancelRequested/progressCb/useSoftPrior 等）
struct ZgBaOptions {
    bool   useSchur          = true;   // ITERATIVE_SCHUR+JACOBI; false→SPARSE_NORMAL_CHOLESKY
    double huberDeltaSigma   = 4.0;    // Huber δ, σ 倍数（ZG 口径 4.0）
    bool   usePreGate        = true;   // 入网门控（MAD 稳健尺度预剔粗差）
    double preGateK          = 6.0;    // 门 = max(preGateK×MAD 尺度, floor×σ)
    double preGateFloorSigma = 5.0;    // 门下限（防 MAD 过小误杀好观测）

    void validate() const;
};

// ZG 借鉴项报表（挂在 GlobalBAResult 旁）
struct ZgBaResult {
    GlobalBAResult base;               // 常规产物（poses/markers/stats/message）
    double sigma0InitDimless  = -1.0;  // 后验 σ₀̂(单位权) — 解算前
    double sigma0FinalDimless = -1.0;  // 后验 σ₀̂(单位权) — 解算后（≈1 即 σ 定标正确）
    double sigma0FinalMm      = -1.0;  // 毫米口径后验 σ₀（sqrt(Σ‖r‖²/dof)）
    int     dof               = 0;     // 自由度 = 3N_obs − 6(N_f−1) − 3N_p
    int     obsTotal          = 0;
    int     obsKept           = 0;     // 入网门后参与解算的观测数
    int     obsGated          = 0;     // 被门控剔除数（= base.statistics.outlierObsIds）
    std::string solverName;            // "ITERATIVE_SCHUR+JACOBI" / "SPARSE_NORMAL_CHOLESKY"
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 全量数据按调用传入; 与主路 GlobalBundleAdjustmentCPU 同款。
// 重置接口: N/A
// 并发策略: 每实例非线程安全, 多实例并行各自独占
// ==============================
class SCANNER_API GlobalBundleAdjustmentZG {
public:
    static constexpr const char* kLogTag = "13Z-GlobalBundleAdjustmentZG";

    GlobalBundleAdjustmentZG() = default;
    ~GlobalBundleAdjustmentZG() = default;
    GlobalBundleAdjustmentZG(const GlobalBundleAdjustmentZG&) = delete;
    GlobalBundleAdjustmentZG& operator=(const GlobalBundleAdjustmentZG&) = delete;

    ZgBaResult Execute(const GlobalBAInput& input,
                       const GlobalBAParams& params,
                       const ZgBaOptions& opts = {});

    // 便捷 A/B: 同一输入跑主路与本变体, 返回两者 success（评测工具用）
    static bool RunAb(const GlobalBAInput& input, const GlobalBAParams& params,
                      const ZgBaOptions& opts,
                      GlobalBAResult& outMain, ZgBaResult& outZg);
};

OperatorInfo getGlobalBundleAdjustmentZGInfo();

}  // namespace calib

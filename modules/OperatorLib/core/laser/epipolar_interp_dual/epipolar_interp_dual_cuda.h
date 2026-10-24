/**
 * @file epipolar_interp_dual_cuda.h
 * @brief 激光中心点极线插值CUDA算子·双模式版 - 公开头文件（纯 C++，不含 CUDA 类型）
 *
 * 一个算子覆盖两类应用：
 *   1) 标定（激光虚拟相机标定）—— 输入点带真实线号，同一 (线号, 极线行) 至多一点；
 *      按 (line_id, y) 分段逐 (线, 行) 双窗插值，输出保留线号（供后计算）。
 *      算法与 epipolar_interp_opt 逐位一致（optEpipolarInterp 复刻）。
 *   2) 扫描（复杂表面）—— 前端无法知道线号，同一极线行有多个激光点；
 *      按 y 排序后逐极线行取窗口内点，按 x 间隙聚类（每簇≈一条线穿过该行），
 *      簇内按“就近原则”取上/下窗点配对插值；输出统一 fid=0（无线号），
 *      线号由后续 laser_match_scan 查表匹配确定。
 *
 * 模式选择：params.mode = Auto（默认，输入线号全部相同→扫描，否则→标定）
 *                      / Labeled / Scan（显式指定）。
 */

#pragma once

#include "common/epipolar_types.h"   // EpipolarInterpResult（2026-09-05 抽至公共头，不再依赖原版算子头）
#include "common/scanner_api.h"      // SCANNER_API 导出宏
#include "common/version.h"          // OperatorInfo
#include "common/calib_warmup_config.h"  // WarmupConfig

#include <nlohmann/json.hpp>
#include <memory>
#include <stdexcept>

namespace calib {

// ============================================================================
// InterpMode
// ============================================================================

enum class InterpMode {
    Auto = 0,     // 自动：输入 line_ids 全部相同（含全 0）→ Scan，否则 Labeled
    Labeled = 1,  // 标定：按线号分段（同 epipolar_interp_opt）
    Scan = 2      // 扫描：无号，按行 x 间隙聚类 + 就近配对
};

// ============================================================================
// EpipolarInterpDualParams
// ============================================================================

struct EpipolarInterpDualParams {
    /// 极线栅格行距（像素）：y_k = k * epipolar_row_step
    float epipolar_row_step = 0.7f;
    /// 双窗半径（像素）：上窗 [y_k - w, y_k)，下窗 (y_k, y_k + w]；-1 = 取 epipolar_row_step
    float window_offset = -1.0f;
    /// 配对 X 约束（像素）：上/下窗配对点的 |Δx| 上限（防跨线配对）
    float max_x_diff = 1.0f;
    /// [Scan] 聚类间隙阈值（像素）：行内按 x 排序后相邻点间隙超过该值即分簇
    float scan_x_gap = 5.0f;
    /// [Scan] 行间簇链接横向阈值（像素）：gap 行内 |Δx| ≤ 该值的最近簇并段（0 = 禁用，line_ids 保持簇 ID）
    float scan_link_dx = 3.0f;
    /// [Scan] 行间簇链接断口容忍（行数）
    int scan_link_gap_rows = 10;
    /// 模式
    InterpMode mode = InterpMode::Auto;
    int deviceId = 0;

    float effectiveWindow() const {
        return window_offset > 0.0f ? window_offset : epipolar_row_step;
    }

    void validate() const {
        if (epipolar_row_step <= 0.0f)
            throw std::invalid_argument("EpipolarInterpDualParams::epipolar_row_step must be > 0");
        if (window_offset <= 0.0f && window_offset != -1.0f)
            throw std::invalid_argument("EpipolarInterpDualParams::window_offset must be > 0 or -1 (auto)");
        if (max_x_diff <= 0.0f)
            throw std::invalid_argument("EpipolarInterpDualParams::max_x_diff must be > 0");
        if (scan_x_gap <= 0.0f)
            throw std::invalid_argument("EpipolarInterpDualParams::scan_x_gap must be > 0");
        if (scan_link_dx < 0.0f)
            throw std::invalid_argument("EpipolarInterpDualParams::scan_link_dx must be >= 0 (0 = off)");
        if (scan_link_gap_rows < 1)
            throw std::invalid_argument("EpipolarInterpDualParams::scan_link_gap_rows must be >= 1");
        if (deviceId < 0)
            throw std::invalid_argument("EpipolarInterpDualParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"epipolar_row_step", epipolar_row_step},
            {"window_offset", window_offset},
            {"max_x_diff", max_x_diff},
            {"scan_x_gap", scan_x_gap},
            {"scan_link_dx", scan_link_dx},
            {"scan_link_gap_rows", scan_link_gap_rows},
            {"mode", static_cast<int>(mode)},
            {"deviceId", deviceId}
        };
    }

    static EpipolarInterpDualParams fromJson(const nlohmann::json& j) {
        EpipolarInterpDualParams p;
        if (j.contains("epipolar_row_step"))
            p.epipolar_row_step = j.at("epipolar_row_step").get<float>();
        if (j.contains("window_offset"))
            p.window_offset = j.at("window_offset").get<float>();
        if (j.contains("max_x_diff"))
            p.max_x_diff = j.at("max_x_diff").get<float>();
        if (j.contains("scan_x_gap"))
            p.scan_x_gap = j.at("scan_x_gap").get<float>();
        if (j.contains("scan_link_dx"))
            p.scan_link_dx = j.at("scan_link_dx").get<float>();
        if (j.contains("scan_link_gap_rows"))
            p.scan_link_gap_rows = j.at("scan_link_gap_rows").get<int>();
        if (j.contains("mode"))
            p.mode = static_cast<InterpMode>(j.at("mode").get<int>());
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// EpipolarInterpDualCuda
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态（缓冲容量缓存为唯一内部状态, grow-only）
// 说明: Labeled 路径算法与 epipolar_interp_opt 逐位一致；Scan 路径输出 fid=0。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================

class SCANNER_API EpipolarInterpDualCuda {
public:
    static constexpr const char* kLogTag = "06-EpipolarInterpDualCuda";

    explicit EpipolarInterpDualCuda(const EpipolarInterpDualParams& params = {});
    ~EpipolarInterpDualCuda();

    EpipolarInterpDualCuda(const EpipolarInterpDualCuda&) = delete;
    EpipolarInterpDualCuda& operator=(const EpipolarInterpDualCuda&) = delete;

    // line_ids: Labeled 模式必须有效；Scan 模式忽略其取值（仅需非空同长度）
    // 输出 d_interp_line_ids:
    //   Labeled = 输入线号透传；Scan = 段 ID（行间簇链接合并，scan_link_dx>0 时）
    //             或簇 ID（row*SCAN_SLOTS+slot 编码，scan_link_dx=0 时）
    EpipolarInterpResult Execute(const cv::cuda::GpuMat& d_points,
                                 const cv::cuda::GpuMat& d_line_ids,
                                 cv::cuda::Stream& stream);

    EpipolarInterpResult Execute(const cv::cuda::GpuMat& d_points,
                                 const cv::cuda::GpuMat& d_line_ids);

    /// 【E4·五链延迟完成】Scan 专用提交段（零宿主同步）：输入＝容量宽视图＋
    /// 设备侧计数指针（有效点数 [0,*dcnt)，尾段为陈旧数据——内部排序键尾戳
    /// 0xFFFFFFFF 保容量排序正确）；全部内核按容量发射。计数 D2H 入 pinned，
    /// 不同步。调用方须先排空流再调 FinishInterp。
    /// ⚠ 仅支持 Scan 模式（Auto 判别在容量视图下不可判定——assert 拒绝）
    EpipolarInterpResult ExecuteDeferred(const cv::cuda::GpuMat& d_points,
                                         const cv::cuda::GpuMat& d_line_ids,
                                         const int* d_deviceCount,
                                         cv::cuda::Stream& stream);
    /// 完成段（调用方已排空流）：读 pinned 计数 → 建精确 [0,N) 视图＋段链接
    /// （链接网格按精确 N——内核零改动）＋interpCount/success 填真值
    void FinishInterp(EpipolarInterpResult& result);

    /// 【前段提前提交配套 2026-10-23】延迟期访问口（ExecuteDeferred 成功后、
    /// FinishInterp 前可用；浅引用同帧生命周期）——供 v3 前段容量提交取输入：
    /// points＝d_scan_cpts_ 全幅；ids＝链接开→d_link_label_（容量化后满幅），
    /// 否则 d_scan_cfids_；count＝d_scan_ccount_（设备输出计数——插值补点后
    /// 输出数可大于输入数，勿用输入计数）。
    cv::cuda::GpuMat GetCapacityPointsView() const;
    cv::cuda::GpuMat GetCapacityIdsView() const;
    const int* GetOutputCountPtr() const;

    void Destroy();
    void Warmup(int pointCount);
    void Warmup(const WarmupConfig& config);
    void SetParams(const EpipolarInterpDualParams& params);

    const EpipolarInterpDualParams& GetParams() const;

    /// 【测速工具】最近一次 Execute 的内部阶段耗时串（" 相名=毫秒 …"），
    /// 供 gpu_stress 聚合；失效条件：下一次 Execute 覆写
    std::string LastTimingBreakdown() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getEpipolarInterpDualInfo();

} // namespace calib

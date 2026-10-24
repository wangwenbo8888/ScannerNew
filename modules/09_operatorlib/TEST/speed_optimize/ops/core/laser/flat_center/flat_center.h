/**
 * @file flat_center.h
 * @brief 平顶激光中心亚像素提取算子（flat_center）- 公开头文件
 *
 * 设计文档：docs/plans/2026-10-18-flat-center-extractor-design.md（v4.4）
 *
 * 定位：steger_fast 的替代/备用算子（沙箱 speed_optimize 专用，09 原树零改动）。
 *   - 动机①形状对口：真实激光为平顶剖面，steger 原始出点 ~25-30% 为平顶双肩
 *     双点（每眼每帧 ~4.2k，README:170 实测）靠事后去重补救；本算子逐段单点
 *     从结构上消灭双点。
 *   - 动机②速度：单 kernel 逐列逐段两次行走，拆掉全图卷积/Hessian/去重三座
 *     大山（基线 ≈8.88ms/眼）。
 *
 * 契约：与 StegerExtractorFast 同构（pImpl＋BUILD_CUDA 双路＋Stream& 流亲和）。
 * 结果复用 StegerResult/GroupMode（common/steger_types.h——steger 族＋flat_center
 * 共用类型驻点，借用已在彼头登记）。
 *
 * 算法（设计 §3）：逐列切段→段内剖面分类（平顶/尖峰）→平顶走半高边缘中点、
 * 尖峰走去背景重心→统一自检门→每段恰出 1 点 (x, cy)（整数格点口径，与 steger
 * kernel px=x+t·nx 同锚定约定）。六门弃点（minRun/contrast/bright/edge/gate/
 * width）→ Degraded＋按门计数消息；点集逐位可复现（每列单线程），点序＝
 * atomicAdd 到达序非确定（golden 按排序比对，既有工程口径）。
 */

#pragma once

#include <memory>
#include <string>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "common/steger_types.h"   // StegerResult / GroupMode（借用驻点）
#include "common/scanner_api.h"    // SCANNER_API
#include "common/version.h"        // OperatorInfo

namespace calib {

struct WarmupConfig;

// ============================================================================
// FlatCenterParams
// ============================================================================
struct FlatCenterParams {
    float lowThreshold = 8.0f;    // 【灰度域】段内最大亮度下限。⚠️ 真实掩膜惰性门
                                  // （dilate₉ 段必含 laserRaw 核心像素 ≥~75）——仅合成
                                  // 掩膜/防回归有效；与 steger_fast 的 lowThreshold=0.5
                                  // （|λ| 域）无可比性，勿照搬数值
    float halfRatio = 0.5f;       // 半高比例（平顶路径找沿基准）
    float plateauRatio = 0.95f;   // 平台判定：亮度 ≥ plateauRatio×vmax 计入平台
    int plateauMinWidth = 3;      // 平台像素数 ≥ 此值 → 判平顶（=2 会使 σ4 高斯被误判）
    int minRunLength = 3;         // 段长下限（含）。⚠️ 真实掩膜惰性门（膨胀段高 ≥9px）
    float minContrast = 2.0f;     // vmax−vmin 下限。⚠️ 真实掩膜惰性门（段端衬底使
                                  // vmin≈背景）；保 Σw>0 防除零为主用途
    float outlierGatePx = 1.0f;   // 自检门：|中心−参照| 上限（平顶参照=括号重心，
                                  // 尖峰参照=锚点 yPeak，设计 §3.6）
    float maxLineWidth = 0.0f;    // 【半高全宽口径】yR−yL 上限（0=off，默认关）。平顶
                                  // 路径专用门：过曝变宽成亮斑的病态点丢弃（语义对齐
                                  // steger maxLineWidth 但口径不同——steger 量平滑后
                                  // w=sqrt(g/|λ|)≈2.4（正常线），本口径正常线 FWHM
                                  // ≈16px，标定勿照搬数值）
    bool smooth3 = true;          // 3 点滑动平均（统计遍/找沿遍/括号重心同款口径；
                                  // 段端读段外原始灰度、图像边界钳位）
    int maxLabels = 256;          // 契约兼容保留；K2 hist 沿用 steger 硬编码 4096 bins
    int deviceId = 0;

    void validate() const {
        if (lowThreshold < 0.0f)
            throw std::invalid_argument("FlatCenterParams::lowThreshold must be >= 0");
        if (!(halfRatio > 0.0f && halfRatio < 1.0f))
            throw std::invalid_argument("FlatCenterParams::halfRatio must be in (0, 1)");
        if (!(plateauRatio > 0.0f && plateauRatio <= 1.0f))
            throw std::invalid_argument("FlatCenterParams::plateauRatio must be in (0, 1]");
        if (plateauMinWidth < 1)
            throw std::invalid_argument("FlatCenterParams::plateauMinWidth must be >= 1");
        if (minRunLength < 1)
            throw std::invalid_argument("FlatCenterParams::minRunLength must be >= 1");
        if (minContrast < 0.0f)
            throw std::invalid_argument("FlatCenterParams::minContrast must be >= 0");
        if (!(outlierGatePx > 0.0f))
            throw std::invalid_argument("FlatCenterParams::outlierGatePx must be > 0");
        if (maxLineWidth < 0.0f)
            throw std::invalid_argument("FlatCenterParams::maxLineWidth must be >= 0 (0=off)");
        if (maxLabels < 1 || maxLabels > 4096)
            throw std::invalid_argument("FlatCenterParams::maxLabels must be [1, 4096]");
        if (deviceId < 0)
            throw std::invalid_argument("FlatCenterParams::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"lowThreshold", lowThreshold},
            {"halfRatio", halfRatio},
            {"plateauRatio", plateauRatio},
            {"plateauMinWidth", plateauMinWidth},
            {"minRunLength", minRunLength},
            {"minContrast", minContrast},
            {"outlierGatePx", outlierGatePx},
            {"maxLineWidth", maxLineWidth},
            {"smooth3", smooth3},
            {"maxLabels", maxLabels},
            {"deviceId", deviceId}
        };
    }

    static FlatCenterParams fromJson(const nlohmann::json& j) {
        FlatCenterParams p;
        if (j.contains("lowThreshold")) p.lowThreshold = j.at("lowThreshold").get<float>();
        if (j.contains("halfRatio")) p.halfRatio = j.at("halfRatio").get<float>();
        if (j.contains("plateauRatio")) p.plateauRatio = j.at("plateauRatio").get<float>();
        if (j.contains("plateauMinWidth")) p.plateauMinWidth = j.at("plateauMinWidth").get<int>();
        if (j.contains("minRunLength")) p.minRunLength = j.at("minRunLength").get<int>();
        if (j.contains("minContrast")) p.minContrast = j.at("minContrast").get<float>();
        if (j.contains("outlierGatePx")) p.outlierGatePx = j.at("outlierGatePx").get<float>();
        if (j.contains("maxLineWidth")) p.maxLineWidth = j.at("maxLineWidth").get<float>();
        if (j.contains("smooth3")) p.smooth3 = j.at("smooth3").get<bool>();
        if (j.contains("maxLabels")) p.maxLabels = j.at("maxLabels").get<int>();
        if (j.contains("deviceId")) p.deviceId = j.at("deviceId").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// FlatCenterExtractor
// ============================================================================
class SCANNER_API FlatCenterExtractor {
public:
    static constexpr const char* kLogTag = "10-FlatCenterExtractor";
    // 【E5】快路出点容量上限（＝interp 定容口径 65536；实测 N max≈34k×1.9 余量）。
    // K1 出点 atomicAdd 槽位≥此值不写（计数照加——finishFlat 以 min(N,cap) 报数）
    static constexpr int kOutCap = 65536;

    explicit FlatCenterExtractor(const FlatCenterParams& params = {});
    ~FlatCenterExtractor();

    FlatCenterExtractor(const FlatCenterExtractor&) = delete;
    FlatCenterExtractor& operator=(const FlatCenterExtractor&) = delete;

    /// ByLabel：掩膜 CV_32SC1（逐像素线编号）
    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask,
                         cv::cuda::Stream& stream);

    /// Flat/ByLabel 分派：Flat 需 CV_8UC1、ByLabel 需 CV_32SC1
    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_mask,
                         cv::cuda::Stream& stream,
                         GroupMode groupMode);

    /// 便捷重载×2（语义＝新建异步 cv::cuda::Stream，与 steger_fast 同款）
    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask);
    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_mask,
                         GroupMode groupMode);

    /// 【flat 优化·合并唤醒】延迟完成版（Flat 路径专用）：提交全部 GPU 工作＋
    /// stats 回读入队，但**不做同步等待**——调用方在同一条流上做一次
    /// waitForCompletion()（左右眼两次调用可共用一次唤醒）后，再调
    /// FinishDeferred 补齐 [0,N) 视图与计数。host 输出模式
    /// （SetCollectHostPoints=true）不支持——走原 Execute。
    StegerResult ExecuteDeferred(const cv::cuda::GpuMat& d_grayImage,
                                 const cv::cuda::GpuMat& d_binaryMask,
                                 cv::cuda::Stream& stream);
    /// 收尾：读已回传 stats（调用方须先同步流）、重建 [0,N) ROI、填计数与质量
    void FinishDeferred(StegerResult& result);

    /// 【E5】设备侧计数指针（d_stats_[0]——K1 atomicAdd 出点数；供下游容量化
    /// 内核作设备计数界，如 interp ExecuteDeferred 的 dcnt）
    const int* GetDeviceCountPtr() const;

    void Destroy();
    void Warmup(int rows, int cols);
    void Warmup(const WarmupConfig& config);
    void SetParams(const FlatCenterParams& params);
    const FlatCenterParams& GetParams() const;

    /// 控制 host 侧 centerPoints 收集（D2H＋CPU 拷贝）。false=只出 GPU 视图
    /// （d_centerPoints/d_line_ids）。**默认 false**（沙箱速度口径；steger_fast
    /// 沙箱版默认 true——两算子默认不同，属设计 §0 决策）
    void SetCollectHostPoints(bool collect);

    /// 【测速工具】最近一次 Execute 分步耗时（毫秒，3 项：0=Total 1=Scan
    /// 2=Group/Out）。需构造前置 JMW_FLATTIME=1（bench 模式）；否则全零
    void LastBenchMs(float out[3]) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getFlatCenterInfo();

} // namespace calib

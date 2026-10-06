#pragma once
// ============================================================================
// epipolar_pair_cpu.h — 矫正后同行配对（激光左右点立体匹配，无表模式）
//
// 场景：激光左右中心线点经矫正+极线插值后（y 已对齐），按"同行最近"配对
// 成对 (xL,yL)/(xR,yR)——标准立体方法，不依赖激光平面标定表（mapData 查
// 表路线的旁路，2026-09-01 收编入 09——此前内联在 07 ScanChains，违
// "调度底座零算子知识"红线，还账搬运）。
// 平台：CPU（点数百级，µs~ms 级）。
// 契约：三元组（Params/Result/Operator）+ Execute()；每实例非线程安全。
// ============================================================================
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "common/calib_types.h"
#include "scanner_api.h"   // SCANNER_API

namespace calib {

/// 配对参数
struct EpipolarPairParams {
    float yTolerance = 0.75f;    ///< 同行判定 |yL-yR| 上限（px，矫正后残余）
    float dispMin = 0.5f;        ///< 正视差下限（px）
    float dispMax = 150.0f;      ///< 正视差上限（px）
    int minPairs = 8;            ///< 低于此数判失败（帧降级由调用方处理）

    void validate() const {
        if (yTolerance <= 0.0f)
            throw std::invalid_argument("EpipolarPairParams::yTolerance must be > 0");
        if (dispMin <= 0.0f || dispMax <= dispMin)
            throw std::invalid_argument("EpipolarPairParams: need 0 < dispMin < dispMax");
        if (minPairs < 0)
            throw std::invalid_argument("EpipolarPairParams::minPairs must be >= 0");
    }
};

/// 配对结果：成对的左右 2D 点 + 左侧线号
struct EpipolarPairResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<cv::Point2f> matchedLeft;
    std::vector<cv::Point2f> matchedRight;
    std::vector<int> matchedIds;    ///< 左点线号透传
};

class SCANNER_API EpipolarPairCPU {
public:
    static constexpr const char* kLogTag = "09-EpipolarPairCPU";

    explicit EpipolarPairCPU(const EpipolarPairParams& params = {});
    ~EpipolarPairCPU();
    EpipolarPairCPU(const EpipolarPairCPU&) = delete;
    EpipolarPairCPU& operator=(const EpipolarPairCPU&) = delete;

    /// left/right：矫正后 2D 点（CV_32FC2 语义的 host 向量）；leftIds：左点
    /// 线号。配对：右点按 y 分桶（2px 粒度），左点在桶内找 y 最近且视差
    /// 落 (dispMin, dispMax] 的点。
    EpipolarPairResult Execute(const std::vector<cv::Point2f>& left,
                               const std::vector<cv::Point2f>& right,
                               const std::vector<int>& leftIds);

    void SetParams(const EpipolarPairParams& params);
    const EpipolarPairParams& GetParams() const { return params_; }

private:
    EpipolarPairParams params_;
};

} // namespace calib

#pragma once
// ============================================================================
// TempTableTypes.h — 温度补偿表输入类型（查表器消费的表结构）
//
// 立体侧 06 持档数据（Matx 快照直填）；激光映射侧 07/09 实际持有 GpuMat 表，
// 06 只按温选档给索引（Ref 后缀），不持有表数据。
// ============================================================================

#include <vector>
#include <opencv2/core.hpp>

namespace Scanner::data {

// 单温度档（立体侧）
struct StereoTempTier {
    double tempC = 0.0;                    // 档温（℃）
    cv::Matx33d R1 = cv::Matx33d::zeros();
    cv::Matx33d R2 = cv::Matx33d::zeros();
    cv::Matx34d P1 = cv::Matx34d::zeros();
    cv::Matx34d P2 = cv::Matx34d::zeros();
    cv::Matx44d Q  = cv::Matx44d::zeros();
};

// 单温度档（激光映射侧）——仅档温，表数据由 07/09 侧持有
struct PlaneMapTempTierRef {
    double tempC = 0.0;                    // 档温（℃）
};

struct StereoTempTable {
    std::vector<StereoTempTier> tiers;     // 按 tempC 升序（调用方约定；实现内亦防御排序）
};

struct PlaneMapTempTableRef {
    std::vector<PlaneMapTempTierRef> tiers;
};

} // namespace Scanner::data

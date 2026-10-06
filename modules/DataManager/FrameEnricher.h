#pragma once
// ============================================================================
// FrameEnricher.h — 出口查表器（DataPlane · 扫描工作流 02-⑤）
//
// 按帧温每帧查两张补偿表（立体矫正表 + 激光平面映射表），选最近档（阶梯
// 0.2℃）填 CalibSnapshot，随帧附带产出 EnhancedFrame——算子无查表动作。
//
// 选档规则：tiers 按 tempC 升序（实现内防御排序）；取 |t - tier.tempC| 最小
// 者，并列取低温档（确定性）。
// 降级规则：单表空 → 对应 tier=-1 且返回 warning；两表全空 → fail；
//           帧温越出表范围 → clamp 首/末档 + warning；命中 → ok。
// 输入契约：空 Mat 输入透传空帧（上游 08/02 侧保证非空）。
// ============================================================================

#include "base/types.h"
#include "EnhancedFrame.h"
#include "TempTableTypes.h"

namespace Scanner::data {

/// 出口查表：填充 out（gray 深拷贝 + snapshot 档数据/档索引 + 帧温/帧号）。
Result enrich(const cv::Mat& grayL, const cv::Mat& grayR, double temperatureC,
              const StereoTempTable& stereoTable, const PlaneMapTempTableRef& laserTable,
              uint64_t frameId, EnhancedFrame& out);

} // namespace Scanner::data

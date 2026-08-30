// ScanSessionData.cpp — 02 扫描会话件实现（装表/pushFrame 出口查表入环/丢帧计数）
#include "ScanSessionData.h"

#include "FrameEnricher.h"

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#include <memory>
#include <utility>

namespace Scanner::data {

Scanner::Result ScanSessionData::assemble(const CalibrationRepository& repo) {
    stereoTable_ = repo.stereoTempTable();
    laserTiers_ = repo.planeMapTiers();
    if (stereoTable_.tiers.empty() && laserTiers_.tiers.empty()) {
        return Result::fail("标定表未就绪：立体温度表与激光档表皆空");
    }
    return Result::ok();
}

Scanner::Result ScanSessionData::pushFrame(const cv::Mat& grayL, const cv::Mat& grayR,
                                           double temperatureC, uint64_t frameId) {
    EnhancedFrame out;
    const Result r =
        enrich(grayL, grayR, temperatureC, stereoTable_, laserTiers_, frameId, out);
    if (!r.success) {
        const uint64_t n = droppedFrames_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n == 1) {   // 流程不走点：首丢告警（后续静默计数——会话账本汇总）
            JMW_LOG_WARN("06-ScanSession",
                "[ScanSession] enrich 失败首帧丢弃（后续静默计数）: 帧{} 温{:.1f}℃ {}",
                frameId, temperatureC, r.message);
        }
        return r;                           // fail 透传，不写环
    }
    ring_.write(std::make_shared<EnhancedFrame>(std::move(out)));
    return r;                               // ok/warning 透传（warning 帧已入环）
}

} // namespace Scanner::data

#pragma once
// ScanSessionData.h — 02 扫描会话件（设计 §4.2；02-②构造、⑩逆序销毁前先停07）
// 只管「装表＋ring＋pushFrame」；配置归 07 ScanConfig（§4.1，06 不造）。
// pushFrame 限采集回调单线程调用（SlotRing write 单生产者约定）。
// 注：槽位构造期定死（SlotRing 无 resize）——ringSlots 构造参数，assemble 只装表。
#include "CalibrationRepository.h"
#include "EnhancedFrame.h"
#include "SlotRing.h"
#include "TempTableTypes.h"
#include "base/types.h"
#include <atomic>
#include <cstdint>
#include <opencv2/core.hpp>

namespace Scanner::data {

class ScanSessionData {
public:
    explicit ScanSessionData(size_t ringSlots = 16)
        : ringSlots_(ringSlots),
          ring_(ringSlots, SlotRing<EnhancedFrame>::WriterMode::Overwrite) {}

    // 装表：仓库两 getter 拷入。两表档皆空 → fail「标定表未就绪」（对齐 FrameEnricher
    // 契约：两表全空 enrich 必 fail）；单表空装配放行——pushFrame 期 warning 降级
    //（tier=-1 档索引自证），帧仍产出。
    Scanner::Result assemble(const CalibrationRepository& repo);

    // 帧唯一入口：enrich（出口查表）→ ring.write。enrich fail → 丢帧不写环＋
    // droppedFrames 计数（挂 10 故障桥）＋Result 透传；warning（越界 clamp/单表空）
    // → Result 透传且帧仍入环（快照档数据自证降级）。
    Scanner::Result pushFrame(const cv::Mat& grayL, const cv::Mat& grayR,
                              double temperatureC, uint64_t frameId);

    SlotRing<EnhancedFrame>& ring() { return ring_; }
    size_t ringCapacity() const { return ringSlots_; }
    uint64_t droppedFrames() const { return droppedFrames_.load(std::memory_order_relaxed); }

private:
    StereoTempTable stereoTable_;          // 装配期内存 Matx 档
    PlaneMapTempTableRef laserTiers_;      // 档温索引（网格数据归 §8-1 协调项）
    size_t ringSlots_;
    SlotRing<EnhancedFrame> ring_;
    std::atomic<uint64_t> droppedFrames_{0};   // enrich fail 丢帧计数
};

} // namespace Scanner::data

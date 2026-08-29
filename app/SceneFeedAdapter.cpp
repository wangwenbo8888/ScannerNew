// ============================================================================
// SceneFeedAdapter.cpp — 实现见头注释（渲染加固计划 P2）
// ============================================================================
#include "SceneFeedAdapter.h"

#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"   // calib::MarkerCloudPoint

#include <cmath>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#include <QMetaType>

Q_DECLARE_METATYPE(std::vector<cv::Point3f>)

SceneFeedAdapter::SceneFeedAdapter(QObject* parent) : QObject(parent) {
    // queued signal 跨线程传自定义类型须注册（一次性）
    qRegisterMetaType<std::vector<cv::Point3f>>("std::vector<cv::Point3f>");
}

void SceneFeedAdapter::pushPostureView(const Scanner::Pose& /*live*/,
                                       int /*confirmedCount*/,
                                       const std::vector<uint8_t>& /*markerDetected*/) {
    // TODO(01 接入期): A 姿态实时视图推送——CalibrationWorkflow deps.sceneFeed 接线时落地
    JMW_LOG_DEBUG("app-SceneFeed", "[SceneFeedAdapter] pushPostureView（01 接入期待接线）");
}

void SceneFeedAdapter::pushCloudSnapshot(Scanner::pipeline::CloudViewHandle cloud) {
    pushedClouds_.fetch_add(1, std::memory_order_relaxed);
    if (frozen_.load(std::memory_order_acquire)) {
        droppedByFreeze_.fetch_add(1, std::memory_order_relaxed);
        return;                                   // 冻结期：丢 ingest 保末帧
    }

    // ★调用线程（FuseConsumer/D）立即值拷贝——跨线程零共享引用。
    // hostMarker 实指融合算子内部 std::vector<calib::MarkerCloudPoint> 的稳定
    // 地址（FuseConsumer 适配器保证调用期间有效）；数百点级，廉价。
    // deviceLaser 恒 nullptr（契约现状——激光互操作 P4 另立计划）。
    std::vector<cv::Point3f> pts;
    if (cloud.hostMarker) {
        const auto& src =
            *static_cast<const std::vector<calib::MarkerCloudPoint>*>(cloud.hostMarker);
        pts.reserve(src.size());
        for (const auto& p : src) {
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
                pts.emplace_back(p.x, p.y, p.z);  // 入口先滤一道（P1 同款防御）
        }
    }
    emit markerCloudUpdated(pts);                 // queued → UI 线程
}

void SceneFeedAdapter::notifyFreeze(bool frozen) {
    const bool was = frozen_.exchange(frozen, std::memory_order_acq_rel);
    if (was != frozen)
        emit freezeChanged(frozen);               // queued → UI（可挂状态条提示）
}

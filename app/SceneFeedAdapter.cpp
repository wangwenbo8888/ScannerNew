// ============================================================================
// SceneFeedAdapter.cpp — 实现见头注释（渲染加固计划 P2）
// ============================================================================
#include "SceneFeedAdapter.h"

#include "PointCloudBuffer.h"   // Scanner::data::MarkerRecord（latestMarkers 缓存）
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

std::vector<Scanner::data::MarkerRecord> SceneFeedAdapter::latestMarkers() const {
    std::lock_guard<std::mutex> lock(markersMtx_);
    return latestMarkers_;
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
    std::vector<Scanner::data::MarkerRecord> recs;
    if (cloud.hostMarker) {
        const auto& src =
            *static_cast<const std::vector<calib::MarkerCloudPoint>*>(cloud.hostMarker);
        pts.reserve(src.size());
        recs.reserve(src.size());
        uint32_t idx = 0;                         // globalId=下标（07 seed 对接口径同源）
        for (const auto& p : src) {
            if (!(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)))
                continue;                         // 入口先滤一道（P1 同款防御）
            pts.emplace_back(p.x, p.y, p.z);
            Scanner::data::MarkerRecord r;
            r.globalId = idx++;
            r.pos = cv::Point3f(p.x, p.y, p.z);
            r.normal = cv::Vec3f(p.nx, p.ny, p.nz);
            recs.push_back(std::move(r));
        }
    }
    {   // 末次快照缓存（扫描合账落 06 仓库的取数口——A 模式标志点点云出口）
        std::lock_guard<std::mutex> lock(markersMtx_);
        latestMarkers_ = std::move(recs);
    }
    emit markerCloudUpdated(pts);                 // queued → UI 线程

    // 激光 host 块（n×3 float 平铺——调用线程立即值拷贝后 queued）
    if (cloud.hostLaser && cloud.laserCount > 0) {
        std::vector<cv::Point3f> laser;
        laser.reserve(cloud.laserCount);
        for (size_t i = 0; i < cloud.laserCount; ++i) {
            const float x = cloud.hostLaser[i * 3 + 0];
            const float y = cloud.hostLaser[i * 3 + 1];
            const float z = cloud.hostLaser[i * 3 + 2];
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z))
                laser.emplace_back(x, y, z);
        }
        emit laserCloudUpdated(laser);            // queued → UI 线程
    }
}

void SceneFeedAdapter::notifyFreeze(bool frozen) {
    const bool was = frozen_.exchange(frozen, std::memory_order_acq_rel);
    if (was != frozen)
        emit freezeChanged(frozen);               // queued → UI（可挂状态条提示）
}

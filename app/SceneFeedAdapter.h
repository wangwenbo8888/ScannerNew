#pragma once
// ============================================================================
// SceneFeedAdapter.h — ISceneFeed 首个实现（渲染加固计划 P2；兑现 02 侧 TODO）
//
// 线程纪律（计划 §2.1 铁律）：
//   - pushCloudSnapshot 在流水线线程（FuseConsumer/D）被调——**调用线程立即
//     值拷贝** marker 点集（数百点级，廉价），之后只传值；严禁跨线程持有
//     融合器内部 vector 的引用（消费方随后仍在写同一存储）。
//   - UI 侧经 Qt queued signal 收值（MainWindow 连到 OSGWidget::loadMarkerPoints）。
//   - notifyFreeze：原子标志——冻结期 ingest 丢弃（画面保持末帧）。
//   - deviceLaser 恒 nullptr（契约现状——激光互操作另立计划，P4 接口位）。
//   - pushPostureView：A 姿态实时视图——01 侧接入期接线（暂记日志 TODO）。
// ============================================================================
#include <atomic>
#include <mutex>
#include <vector>

#include <QObject>

#include "opencv2/core.hpp"

#include "pipelines/ISceneFeed.h"

namespace Scanner::data { struct MarkerRecord; }   // 06 标志点记录（latestMarkers 缓存）

class SceneFeedAdapter : public QObject, public Scanner::pipeline::ISceneFeed {
    Q_OBJECT
public:
    explicit SceneFeedAdapter(QObject* parent = nullptr);

    // —— ISceneFeed（流水线线程调用）——
    void pushPostureView(const Scanner::Pose& live, int confirmedCount,
                         const std::vector<uint8_t>& markerDetected) override;
    void pushCloudSnapshot(Scanner::pipeline::CloudViewHandle cloud) override;
    void notifyFreeze(bool frozen) override;

    // —— 观测（任意线程）——
    bool isFrozen() const { return frozen_.load(std::memory_order_acquire); }
    uint64_t pushedClouds() const { return pushedClouds_.load(std::memory_order_relaxed); }
    uint64_t droppedByFreeze() const { return droppedByFreeze_.load(std::memory_order_relaxed); }

    /// 最近一次推送的标志点点云快照（含法线；任意线程）。空=尚无推送。
    /// 扫描合账时由 app 落 06 PointCloudBuffer（A 模式的"生成标志点点云"出口）
    std::vector<Scanner::data::MarkerRecord> latestMarkers() const;

signals:
    // UI 线程消费（MainWindow 接 OSGWidget::loadMarkerPoints；metatype 在 cpp 注册）
    void markerCloudUpdated(const std::vector<cv::Point3f>& points);
    void freezeChanged(bool frozen);

private:
    std::atomic<bool>     frozen_{false};
    std::atomic<uint64_t> pushedClouds_{0};
    std::atomic<uint64_t> droppedByFreeze_{0};

    mutable std::mutex markersMtx_;                          // latestMarkers 缓存锁
    std::vector<Scanner::data::MarkerRecord> latestMarkers_; // 末次推送快照（推线程写/任意读）
};

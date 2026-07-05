#pragma once


#include "laser_cloud_renderer.h"
#include "marker_cloud_renderer.h"
#include <memory>

namespace calib {

// ============================================================
// 扫描查看�?�?OSG Viewer 封装
//
// 创建窗口、管理场景图（激�?Geode + 标记�?Geode）�?// 管理 Shader 程序、驱动渲染循环�?// ============================================================
class ScannerViewer {
public:
    static constexpr const char* kLogTag = "Display-ScannerViewer";

    /// @param maxLaserPoints  激光点云最大点数（预分配VBO）
    /// @param maxMarkers      标记点最大数量
    ScannerViewer(size_t maxLaserPoints = 1 << 20,  // 100万
                  size_t maxMarkers = 256);
    ~ScannerViewer();

    ScannerViewer(const ScannerViewer&) = delete;
    ScannerViewer& operator=(const ScannerViewer&) = delete;

    /// 初始化 Viewer + 窗口（调用后进入 OSG 事件循环）
    void init(int width = 1280, int height = 720);

    /// 渲染一帧（包含 CUDA interop map/unmap）
    /// @return false = 窗口关闭
    bool frame();

    /// 窗口是否已关闭
    bool done() const;

    /// 更新激光点云数据（仅存指针，展点在 draw �?GPU 零拷贝完成）
    LaserCloudRenderer*  laserRenderer();
    MarkerCloudRenderer* markerRenderer();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace calib
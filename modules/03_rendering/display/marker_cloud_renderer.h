#pragma once


#include "point_expand_math.h"
#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"

#include <osg/ref_ptr>
#include <osg/Geometry>
#include <vector>
#include <cstddef>
#include <memory>

namespace calib {

// ============================================================
// 标记点云渲染器（�?CPU�?//
// 消费融合-02M 输出的全局标记点集，展点→VBO→OSG 渲染�?// 自身不再�?R/T 变换和去重，只做展点数学�?VBO 管理�?//
// 数据路径�?//   marker_cloud_fuse_cpu(融合-02M) �?update(fusedPoints) �?flush() �?OSG渲染
// ============================================================
class MarkerCloudRenderer {
public:
    static constexpr const char* kLogTag = "Display-MarkerCloudRenderer";
    static constexpr float RING_RATIO = 10.0f / 6.0f;

    explicit MarkerCloudRenderer(size_t maxMarkers = 1024);
    ~MarkerCloudRenderer();

    MarkerCloudRenderer(const MarkerCloudRenderer&) = delete;
    MarkerCloudRenderer& operator=(const MarkerCloudRenderer&) = delete;

    /// 更新数据源：消费融合-02M 输出的全局标记点集
    /// @param fusedPoints  融合后的全局标记点（经 R/T 变换 + 去重）
    void update(const std::vector<MarkerCloudPoint>& fusedPoints);

    /// 展开累积点为四边形，上传 OSG VBO（每帧渲染前调用）
    void flush();

    /// 获取 OSG Geometry 节点
    osg::Geometry* getGeometry() const;

    /// 清空（隐藏渲染）
    void clear();

    /// 当前数据点数
    size_t pointCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace calib
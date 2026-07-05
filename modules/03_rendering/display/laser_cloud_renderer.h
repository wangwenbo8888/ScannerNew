#pragma once


#include <osg/ref_ptr>
#include <osg/Drawable>
#include <cstddef>
#include <memory>

namespace calib {

// 前向声明
struct LaserCloudFuseDeviceContext;

// ============================================================
// 激光点云渲染器 V2 �?CUDA-GL 零拷�?//
// 数据全程�?GPU�?//   d_fusedXyz(GPU) �?kernel �?GL VBO(GPU, mapped) �?渲染
//   �?�?PCIe 传输
// ============================================================
class LaserCloudRenderer {
public:
    static constexpr const char* kLogTag = "Display-LaserCloudRenderer";

    explicit LaserCloudRenderer(size_t maxPoints);
    ~LaserCloudRenderer();

    LaserCloudRenderer(const LaserCloudRenderer&) = delete;
    LaserCloudRenderer& operator=(const LaserCloudRenderer&) = delete;

    /// 设置本帧融合数据（仅存储指针，实际展点在 draw 时发生）
    void update(const LaserCloudFuseDeviceContext& ctx);

    /// 获取 OSG Drawable 节点
    osg::Drawable* getDrawable() const;

    void clear();
    size_t maxPoints() const;
    size_t activePoints() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace calib
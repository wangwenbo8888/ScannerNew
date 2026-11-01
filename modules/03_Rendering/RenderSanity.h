#pragma once
// ============================================================================
// RenderSanity.h — 渲染快照摄入防御（纯函数；渲染加固计划 P1.1）
//
// 职责：loadFromPointCloudBuffer 摄入前的四道闸——
//   版本短路（未变不重建）/ 空数据 / NaN·Inf 滤除（颜色同步）/
//   包围盒 sanity（半径越界拒绝）/ 点数预算均匀抽稀。
// 拒绝语义：一律「保留现画面」由调用方执行；本函数只做就地净化与判定。
// 依赖仅 OpenCV core 类型——无 Qt/OSG，无头环境可单测。
//
// 03 事件码表（RenderEvent；经 OSGWidget::setFaultSink 注出的轻量出口，
// app 装配桥接 EventBus/日志——03 不直接依赖 base，保持可测）：
//   0x0301 快照拒绝 / 0x0302 构建降级（保留旧景）/ 0x0303 渲染挂起 /
//   0x0304 挂起恢复 / 0x0311 帧超时抽稀 / 0x0312 降级级别变化
// ============================================================================
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace Scanner::render {

enum class RenderEvent : int32_t {
    SnapshotRejected = 0x0301,   // 快照拒绝（版本未变/全空/全无效/尺寸不配/包围盒越界）
    BuildDegraded    = 0x0302,   // 场景构建异常（bad_alloc 等）——旧画面保留
    RenderSuspended  = 0x0303,   // 渲染帧异常，循环已挂起（UI 存活）
    RenderResumed    = 0x0304,   // 挂起恢复
    FrameOverbudget  = 0x0311,   // 帧超时自动抽稀（P2 接入）
    DegradeChanged   = 0x0312,   // 降级级别变化（含摄入截断）
};

struct IngestDecision {
    bool accept = false;         // false → 调用方丢弃本次更新，保留现画面
    bool truncated = false;      // true → 点数超预算被均匀抽稀（keptCount 条）
    size_t keptCount = 0;        // 净化/抽稀后点数
    size_t nanFiltered = 0;      // 滤除的 NaN/Inf 点数
    const char* reason = "";     // 拒绝原因（accept=true 时空串）
};

inline bool pointIsFinite(const cv::Point3f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

/// 就地净化快照并给出摄入判定。points/colors 可被修改（滤除/抽稀后 resize）。
/// colors 为空＝无颜色语义（跳过颜色同步）；非空须与 points 等长（否则拒绝）。
inline IngestDecision sanitizeSnapshot(std::vector<cv::Point3f>& points,
                                       std::vector<cv::Vec3b>& colors,
                                       uint64_t newVersion, uint64_t lastVersion,
                                       size_t maxPoints, double maxExtentMm) {
    IngestDecision d;
    if (newVersion == lastVersion) { d.reason = "unchanged"; return d; }
    if (points.empty())            { d.reason = "empty";     return d; }

    const bool hasColors = !colors.empty();
    if (hasColors && colors.size() != points.size()) {
        d.reason = "size-mismatch";
        return d;
    }

    // NaN/Inf 滤除（就地压缩；颜色同步搬移）
    size_t w = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (!pointIsFinite(points[i])) { ++d.nanFiltered; continue; }
        if (w != i) {
            points[w] = points[i];
            if (hasColors) colors[w] = colors[i];
        }
        ++w;
    }
    points.resize(w);
    if (hasColors) colors.resize(w);
    if (w == 0) { d.reason = "all-invalid"; return d; }

    // 包围盒 sanity：原点最远点距离（渲染系单位 mm）——越界整份拒绝
    double r2max = 0.0;
    for (const auto& p : points) {
        const double r2 = double(p.x) * p.x + double(p.y) * p.y + double(p.z) * p.z;
        if (r2 > r2max) r2max = r2;
    }
    if (std::sqrt(r2max) > maxExtentMm) { d.reason = "extent"; return d; }

    // 超预算均匀抽稀：k*w/max 索引映射——恰留 maxPoints 条、首尾保留、分布均匀
    if (w > maxPoints) {
        std::vector<cv::Point3f> kept;
        std::vector<cv::Vec3b> keptColors;
        kept.reserve(maxPoints);
        if (hasColors) keptColors.reserve(maxPoints);
        for (size_t k = 0; k < maxPoints; ++k) {
            const size_t i = (k + 1 == maxPoints)
                                 ? w - 1                       // 末点显式保留（契约：首尾都在）
                                 : static_cast<size_t>(
                                       static_cast<uint64_t>(k) * w / maxPoints);
            kept.push_back(points[i]);
            if (hasColors) keptColors.push_back(colors[i]);
        }
        points.swap(kept);
        if (hasColors) colors.swap(keptColors);
        d.truncated = true;
        w = maxPoints;
    }

    d.accept = true;
    d.keptCount = w;
    d.reason = "";
    return d;
}

} // namespace Scanner::render

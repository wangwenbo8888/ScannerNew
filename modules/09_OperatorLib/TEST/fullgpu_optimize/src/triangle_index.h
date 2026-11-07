#pragma once
// ============================================================================
// triangle_index.h — 全局标记点库三角形索引（frame_fuse 优化步5）
// ============================================================================
// 结构（2026-10-14 三代理审查修订版）：
//   点表（调用方持有）→ 按【有向】法线种子聚类分面（粗门 60°；正反面天然分离，
//   曲面连锁由"到种子夹角"限制组直径）→ 面内拟合平面投影 → Delaunay 三角剖分
//   （每条边最多两三角形共用；近共线/近重复点导致的退化由最短高门与异常兜底）
//   → 三角形记录{三顶点, 三边排序} → (最短边, 次短边) 排序数组二分索引。
// 查询：当前帧三角形的 (s1,s2) → 二分取候选段 → 次边筛 → 顶点投票 → top-K 库点
//   （返回给调用方做"子集喂算子"，算子保持无状态——状态生命周期＝调用方）。
// 维护策略：全量重建（几百点毫秒级）。库点位置会被新观测刷新——就地改坐标会使
//   索引悬挂，故采用"写库即标脏、下次使用时重建"，不维护增量账本。
// 确定性：全流程无随机源；投票并列取小下标。
// ============================================================================
#include <opencv2/core.hpp>
#include <algorithm>
#include <cstddef>
#include <vector>

namespace scr {

class TriangleLibraryIndex {
public:
    struct BuildStat {
        int pointCount = 0;
        int groupCount = 0;      // 有效分面数（≥3 点才建剖分）
        int triangleCount = 0;
        double buildMs = 0.0;
    };
    struct QueryStat {
        int curTriangleCount = 0;
        int candidateTriangleCount = 0;   // 过边门候选（诊断候选爆炸）
        double queryMs = 0.0;
    };

    /// 全量重建（positions/normals 尺寸须一致；法线须单位化且朝向已规整）
    BuildStat rebuild(const std::vector<cv::Point3d>& positions,
                      const std::vector<cv::Vec3d>& normals);

    /// 当前帧查询：返回投票选出的库内点下标（≤maxCount；有效票 <3 返回空）。
    /// cur 点近共面假设成立时用 Delaunay 三角形；剖分异常时退化为全部三点组合。
    std::vector<int> query(const std::vector<cv::Point3d>& curPositions,
                           int maxCount, QueryStat* stat = nullptr) const;

    bool empty() const { return tris_.empty(); }
    size_t triangleCount() const { return tris_.size(); }
    int lastBuildPointCount() const { return lastPointCount_; }

private:
    struct Tri {
        int v[3] = {0, 0, 0};       // 库点下标
        double s[3] = {0, 0, 0};    // 三边排序（短→长）
    };
    struct EdgeKey {
        double minEdge = 0.0;
        double secondEdge = 0.0;
        int triId = 0;
    };
    std::vector<Tri> tris_;
    std::vector<EdgeKey> edgeKeys_;       // 按 minEdge 升序
    int lastPointCount_ = 0;

    // 查询容差与算子 triangle 参数保持一致（绝对 1mm / 相对 2%）
    static double sideTolOf(double edgeLen) { return std::max(1.0, 0.02 * edgeLen); }
};

} // namespace scr

// ============================================================================
// SelectionService.cpp — 三栏选择引擎实现（契约见 SelectionService.h）
// ============================================================================
#include "SelectionService.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Scanner::edit {

void SelectionService::setCamera(const cv::Matx44d& view, const cv::Matx44d& proj,
                                 int viewportW, int viewportH) {
    view_ = view;
    proj_ = proj;
    vpW_ = std::max(1, viewportW);
    vpH_ = std::max(1, viewportH);
}

bool SelectionService::project(const cv::Point3f& p, cv::Point2f& screen,
                               double& viewZ) const {
    const cv::Vec4d viewPos = view_ * cv::Vec4d(p.x, p.y, p.z, 1.0);
    viewZ = viewPos(2);
    const cv::Vec4d clip = proj_ * viewPos;
    const double w = clip(3);
    if (!(w > 0.0)) return false;                       // 在后方或退化
    const double ndcX = clip(0) / w;
    const double ndcY = clip(1) / w;
    screen = cv::Point2f(static_cast<float>((ndcX + 1.0) * 0.5 * vpW_),
                         static_cast<float>((1.0 - ndcY) * 0.5 * vpH_));
    return true;
}

bool SelectionService::pointInPolygon(const cv::Point2f& pt,
                                      const std::vector<cv::Point2f>& poly) {
    // 偶奇规则射线法（x 轴向右射线）
    bool inside = false;
    const size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const cv::Point2f& a = poly[i];
        const cv::Point2f& b = poly[j];
        if (((a.y > pt.y) != (b.y > pt.y)) &&
            (pt.x < (b.x - a.x) * (pt.y - a.y) / (b.y - a.y + 1e-30) + a.x)) {
            inside = !inside;
        }
    }
    return inside;
}

SelectionResult SelectionService::select(const std::vector<cv::Point2f>& polygon,
                                         const SelectionInput& cand,
                                         DepthMode mode, ObjectType type) const {
    SelectionResult out;
    if (polygon.size() < 3) return out;                 // 非闭合曲线

    // 候选展开（栏3 过滤先行——遮挡判定仅在可选中候选之间）
    struct Cand {
        uint32_t idx;
        bool marker;
        const cv::Point3f* p;
    };
    std::vector<Cand> cands;
    cands.reserve(cand.markerPts.size() + cand.laserPts.size());
    if (type != ObjectType::LaserOnly) {
        for (size_t i = 0; i < cand.markerPts.size(); ++i)
            cands.push_back({static_cast<uint32_t>(i), true, &cand.markerPts[i]});
    }
    if (type != ObjectType::MarkerOnly) {
        for (size_t i = 0; i < cand.laserPts.size(); ++i)
            cands.push_back({static_cast<uint32_t>(i), false, &cand.laserPts[i]});
    }

    // 投影包含测试；记录深度（前向距离＝-viewZ）
    struct Hit {
        uint32_t idx;
        bool marker;
        double depth;
    };
    std::vector<Hit> hits;
    hits.reserve(cands.size());
    double minDepth = std::numeric_limits<double>::max();
    for (const auto& c : cands) {
        cv::Point2f s;
        double vz = 0.0;
        if (!project(*c.p, s, vz)) continue;
        if (!pointInPolygon(s, polygon)) continue;
        const double depth = -vz;                       // 前向为正
        hits.push_back({c.idx, c.marker, depth});
        minDepth = std::min(minDepth, depth);
    }

    // 栏2：只选第一层——深度带宽过滤
    if (mode == DepthMode::FirstLayer) {
        std::vector<Hit> kept;
        kept.reserve(hits.size());
        for (const auto& h : hits)
            if (h.depth <= minDepth + params_.firstLayerTolerance) kept.push_back(h);
        hits.swap(kept);
    }

    // 按类型拆分＋升序
    for (const auto& h : hits) {
        if (h.marker) out.markerIdx.push_back(h.idx);
        else          out.laserIdx.push_back(h.idx);
    }
    std::sort(out.markerIdx.begin(), out.markerIdx.end());
    std::sort(out.laserIdx.begin(), out.laserIdx.end());
    return out;
}

} // namespace Scanner::edit

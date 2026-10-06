#pragma once
// ============================================================================
// SelectionService.h — 05 三栏选择引擎（设计定案 D3；实施计划 P1 2026-09-05）
//
// 职责：屏幕闭合曲线 → 空间点集下标（作用于候选快照，不触任何账本）。
// 三栏模型（正交开关，调用方组合后经参数传入）：
//   栏1 工具类型（套索/多段线）——只影响曲线形状的采集方式，本服务统一吃
//       「闭合屏幕折线」（>=3 点），不区分工具；
//   栏2 选择类型（贯穿/只选第一层）——贯穿=投影落入曲线即选；只选第一层=
//       候选内视图向深度最近层（minDepth + firstLayerTolerance 带宽内），
//       遮挡判定仅在「可选中候选」之间（跨类型遮挡不判——栏3 先行过滤）；
//   栏3 对象类型（激光点/标记点/两者）——候选集过滤。
// 相机约定（列向量数学惯例）：p_clip = P·V·[p,1]；w<=0（在后方）弃；
//   NDC=clip.xy/w；screen=((ndc.x+1)/2·W, (1-ndc.y)/2·H)（屏幕 y 向下）。
//   03 侧负责从 OSG 行主序矩阵转置适配（其 computeRenderingMVP 同款知识）。
// 线程模型：编辑会话 UI 单线程属主（无锁）。
// ============================================================================
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "base/types.h"

namespace Scanner::edit {

enum class DepthMode { Through, FirstLayer };      // 栏2：贯穿 / 只选第一层
enum class ObjectType { LaserOnly, MarkerOnly, Both };  // 栏3：对象类型

// 候选快照：下标＝账本身份（编辑会话期融合云只删不增——下标稳定）
struct SelectionInput {
    std::vector<cv::Point3f> markerPts;    // 标记点（融合累积器快照）
    std::vector<cv::Point3f> laserPts;     // 激光点（融合累积器快照）
};

struct SelectionResult {
    std::vector<uint32_t> markerIdx;       // 升序选中下标
    std::vector<uint32_t> laserIdx;
    bool empty() const { return markerIdx.empty() && laserIdx.empty(); }
};

class SelectionService {
public:
    struct Params {
        double firstLayerTolerance = 1.0;  // mm——第一层深度带宽
    };

    void setParams(const Params& p) { params_ = p; }
    const Params& params() const { return params_; }

    // 相机状态（渲染侧每圈选前刷新）；viewport 为画布像素尺寸
    void setCamera(const cv::Matx44d& view, const cv::Matx44d& proj,
                   int viewportW, int viewportH);

    // polygon＝闭合屏幕折线（px，>=3 点，首尾不必重合）；返回各组选中下标（升序）
    SelectionResult select(const std::vector<cv::Point2f>& polygon,
                           const SelectionInput& cand,
                           DepthMode mode, ObjectType type) const;

private:
    // 世界点 → 屏幕像素；viewZ＝视图空间 z（负＝前方）；false＝在后方/退化
    bool project(const cv::Point3f& p, cv::Point2f& screen, double& viewZ) const;
    static bool pointInPolygon(const cv::Point2f& pt, const std::vector<cv::Point2f>& poly);

    Params params_;
    cv::Matx44d view_ = cv::Matx44d::eye();
    cv::Matx44d proj_ = cv::Matx44d::eye();
    int vpW_ = 1;
    int vpH_ = 1;
};

} // namespace Scanner::edit

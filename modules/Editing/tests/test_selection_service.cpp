// ============================================================================
// test_selection_service.cpp — 三栏选择引擎单测（实施计划 P1）
// 相机用正交恒等约定：V=I、P=I → 世界 (x,y)∈[-1,1] 映射画布；深度＝-z。
// ============================================================================
#include <gtest/gtest.h>

#include "SelectionService.h"

using namespace Scanner::edit;

namespace {
constexpr int kW = 100, kH = 100;

// 全屏方形（画布 100×100 → 世界 [-1,1]）
std::vector<cv::Point2f> fullSquare() {
    return {{5, 5}, {95, 5}, {95, 95}, {5, 95}};
}

SelectionService makeSvc() {
    SelectionService s;
    s.setCamera(cv::Matx44d::eye(), cv::Matx44d::eye(), kW, kH);
    return s;
}
} // namespace

// —— 投影包含：内/外点 ——
TEST(Selection, InsideOutside) {
    auto svc = makeSvc();
    SelectionInput in;
    in.markerPts = {{0, 0, -10}, {0.8f, 0.8f, -10}, {2.0f, 0, -10}};  // 前两内一外
    auto r = svc.select(fullSquare(), in, DepthMode::Through, ObjectType::MarkerOnly);
    ASSERT_EQ(r.markerIdx.size(), 2u);
    EXPECT_EQ(r.markerIdx[0], 0u);
    EXPECT_EQ(r.markerIdx[1], 1u);
    EXPECT_TRUE(r.laserIdx.empty());
}

// —— 栏3 对象类型过滤 ——
TEST(Selection, ObjectTypeFilter) {
    auto svc = makeSvc();
    SelectionInput in;
    in.markerPts = {{0, 0, -10}};
    in.laserPts  = {{0.5f, 0.5f, -10}, {0, 0, -30}};
    auto rM = svc.select(fullSquare(), in, DepthMode::Through, ObjectType::MarkerOnly);
    ASSERT_EQ(rM.markerIdx.size(), 1u);
    EXPECT_TRUE(rM.laserIdx.empty());                 // 激光不选
    auto rB = svc.select(fullSquare(), in, DepthMode::Through, ObjectType::Both);
    EXPECT_EQ(rB.markerIdx.size(), 1u);
    EXPECT_EQ(rB.laserIdx.size(), 2u);
}

// —— 栏2 只选第一层：同屏不同深取近 ——
TEST(Selection, FirstLayerDepthOcclusion) {
    auto svc = makeSvc();
    svc.setParams({1.0});                             // 1mm 带宽
    SelectionInput in;
    in.markerPts = {{0, 0, -10}, {0, 0, -10.5}, {0, 0, -50}};
    auto r = svc.select(fullSquare(), in, DepthMode::FirstLayer, ObjectType::MarkerOnly);
    // 深度 10 与 10.5 在带宽内＝第一层；50 被遮挡剔除
    ASSERT_EQ(r.markerIdx.size(), 2u);
    EXPECT_EQ(r.markerIdx[0], 0u);
    EXPECT_EQ(r.markerIdx[1], 1u);

    // 贯穿模式三个都选
    auto rT = svc.select(fullSquare(), in, DepthMode::Through, ObjectType::MarkerOnly);
    EXPECT_EQ(rT.markerIdx.size(), 3u);
}

// —— 遮挡仅在可选中候选之间（MarkerOnly 时激光不参与遮挡） ——
TEST(Selection, OcclusionAmongSelectableOnly) {
    auto svc = makeSvc();
    SelectionInput in;
    in.markerPts = {{0, 0, -50}};                     // 标记点在深 50
    in.laserPts  = {{0, 0, -10}};                     // 激光点在浅 10（不可选）
    auto r = svc.select(fullSquare(), in, DepthMode::FirstLayer, ObjectType::MarkerOnly);
    EXPECT_EQ(r.markerIdx.size(), 1u);                // 激光不遮挡标记（栏3 先过滤）
}

// —— 在后方的点弃（w<=0）——
TEST(Selection, BehindCameraRejected) {
    auto svc = makeSvc();
    SelectionInput in;
    in.markerPts = {{0, 0, 10}};                      // z=+10 在相机后方（恒等 P 下 w=1>0）
    // 恒等投影无法表达后方剔除——改用真透视矩阵验证 w<=0 路径
    cv::Matx44d proj(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0);  // clip.w=-z
    SelectionService s2;
    s2.setCamera(cv::Matx44d::eye(), proj, kW, kH);
    auto r = s2.select(fullSquare(), in, DepthMode::Through, ObjectType::MarkerOnly);
    EXPECT_TRUE(r.markerIdx.empty());                 // z=+10 → w=-10<0 弃
}

// —— 非法输入：折线不足 3 点 ——
TEST(Selection, PolygonTooShort) {
    auto svc = makeSvc();
    SelectionInput in;
    in.markerPts = {{0, 0, -10}};
    auto r = svc.select({{10, 10}, {50, 50}}, in, DepthMode::Through, ObjectType::Both);
    EXPECT_TRUE(r.empty());
}

// —— 三角形折线（凹多边形也适用偶奇规则） ——
TEST(Selection, TrianglePolygon) {
    auto svc = makeSvc();
    std::vector<cv::Point2f> tri = {{50, 10}, {10, 90}, {90, 90}};
    SelectionInput in;
    in.laserPts = {{0, 0.2f, -10}, {0, -0.9f, -10}};  // 世界 y=0.2 屏(50,40)内 / y=-0.9 屏(50,95)外
    auto r = svc.select(tri, in, DepthMode::Through, ObjectType::LaserOnly);
    ASSERT_EQ(r.laserIdx.size(), 1u);
    EXPECT_EQ(r.laserIdx[0], 0u);
}

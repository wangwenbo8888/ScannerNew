// ============================================================================
// test_render_sanity.cpp — RenderSanity 纯函数单测（渲染加固计划 P1.5）
// 无 Qt/OSG 依赖；覆盖四道闸＋颜色同步＋抽稀均匀性
// ============================================================================
#include <gtest/gtest.h>

#include "RenderSanity.h"

#include <cmath>

using Scanner::render::IngestDecision;
using Scanner::render::RenderEvent;
using Scanner::render::sanitizeSnapshot;

namespace {
constexpr size_t kBudget = 100;         // 测试用小预算
constexpr double kExtent = 1.0e6;       // mm

std::vector<cv::Point3f> makePoints(size_t n, float scale = 1.0f) {
    std::vector<cv::Point3f> pts;
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i)
        pts.emplace_back(float(i) * scale, 1.0f, 2.0f);
    return pts;
}
} // namespace

// —— 版本闸 ——
TEST(RenderSanity, VersionUnchangedRejected) {
    auto pts = makePoints(10);
    std::vector<cv::Vec3b> cols;
    const auto d = sanitizeSnapshot(pts, cols, /*newVersion=*/7, /*lastVersion=*/7,
                                    kBudget, kExtent);
    EXPECT_FALSE(d.accept);
    EXPECT_STREQ(d.reason, "unchanged");
    EXPECT_EQ(pts.size(), 10u);              // 拒绝路径不净化（保留原样）
}

// —— 空数据闸 ——
TEST(RenderSanity, EmptyRejected) {
    std::vector<cv::Point3f> pts;
    std::vector<cv::Vec3b> cols;
    const auto d = sanitizeSnapshot(pts, cols, 2, 1, kBudget, kExtent);
    EXPECT_FALSE(d.accept);
    EXPECT_STREQ(d.reason, "empty");
}

// —— 全无效（NaN/Inf）——
TEST(RenderSanity, AllInvalidRejected) {
    std::vector<cv::Point3f> pts(5, cv::Point3f(
        std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f));
    std::vector<cv::Vec3b> cols;
    const auto d = sanitizeSnapshot(pts, cols, 2, 1, kBudget, kExtent);
    EXPECT_FALSE(d.accept);
    EXPECT_STREQ(d.reason, "all-invalid");
    EXPECT_EQ(d.nanFiltered, 5u);
    EXPECT_TRUE(pts.empty());
}

// —— 部分无效滤除＋颜色同步 ——
TEST(RenderSanity, PartialNaNFilteredWithColorSync) {
    std::vector<cv::Point3f> pts = {
        {0, 0, 0}, {std::numeric_limits<float>::infinity(), 1, 1},
        {2, 2, 2}, {std::numeric_limits<float>::quiet_NaN(), 3, 3},
        {4, 4, 4}};
    std::vector<cv::Vec3b> cols = {{10, 10, 10}, {20, 20, 20}, {30, 30, 30},
                                   {40, 40, 40}, {50, 50, 50}};
    const auto d = sanitizeSnapshot(pts, cols, 2, 1, kBudget, kExtent);
    ASSERT_TRUE(d.accept);
    EXPECT_EQ(d.nanFiltered, 2u);
    EXPECT_EQ(pts.size(), 3u);
    EXPECT_EQ(cols.size(), 3u);
    // 保留序：0/2/4 号点与其颜色配对未被错位
    EXPECT_EQ(pts[0], cv::Point3f(0, 0, 0));
    EXPECT_EQ(cols[0], cv::Vec3b(10, 10, 10));
    EXPECT_EQ(pts[1], cv::Point3f(2, 2, 2));
    EXPECT_EQ(cols[1], cv::Vec3b(30, 30, 30));
    EXPECT_EQ(pts[2], cv::Point3f(4, 4, 4));
    EXPECT_EQ(cols[2], cv::Vec3b(50, 50, 50));
}

// —— 颜色尺寸不配拒绝 ——
TEST(RenderSanity, ColorSizeMismatchRejected) {
    auto pts = makePoints(4);
    std::vector<cv::Vec3b> cols(3);          // 故意短
    const auto d = sanitizeSnapshot(pts, cols, 2, 1, kBudget, kExtent);
    EXPECT_FALSE(d.accept);
    EXPECT_STREQ(d.reason, "size-mismatch");
}

// —— 包围盒越界拒绝（一个野点毁整份）——
TEST(RenderSanity, ExtentRejectsWildPoint) {
    auto pts = makePoints(10);
    pts[5] = cv::Point3f(9.9e6f, 0.f, 0.f);  // 超出 1e6 mm
    std::vector<cv::Vec3b> cols;
    const auto d = sanitizeSnapshot(pts, cols, 2, 1, kBudget, kExtent);
    EXPECT_FALSE(d.accept);
    EXPECT_STREQ(d.reason, "extent");
}

// —— 超预算均匀抽稀 ——
TEST(RenderSanity, OverBudgetThinnedUniformly) {
    constexpr size_t n = 1000;
    auto pts = makePoints(n);
    std::vector<cv::Vec3b> cols(n, cv::Vec3b(7, 8, 9));
    const auto d = sanitizeSnapshot(pts, cols, 2, 1, kBudget, kExtent);
    ASSERT_TRUE(d.accept);
    EXPECT_TRUE(d.truncated);
    EXPECT_EQ(d.keptCount, kBudget);
    EXPECT_EQ(pts.size(), kBudget);
    // 首尾保留＋均匀：索引 i = k*n/max 单调递增
    EXPECT_EQ(pts.front(), cv::Point3f(0.f, 1.f, 2.f));
    EXPECT_EQ(pts.back(), cv::Point3f(float(n - 1), 1.f, 2.f));
    EXPECT_EQ(cols.size(), kBudget);
    // 单调性（分布无乱序）
    for (size_t k = 1; k < pts.size(); ++k)
        EXPECT_LT(pts[k - 1].x, pts[k].x);
}

// —— 正常通过 ——
TEST(RenderSanity, NormalAcceptNoColor) {
    auto pts = makePoints(10);
    std::vector<cv::Vec3b> cols;             // 无颜色语义
    const auto d = sanitizeSnapshot(pts, cols, 2, 1, kBudget, kExtent);
    ASSERT_TRUE(d.accept);
    EXPECT_FALSE(d.truncated);
    EXPECT_EQ(d.keptCount, 10u);
    EXPECT_EQ(d.nanFiltered, 0u);
    EXPECT_STREQ(d.reason, "");
}

// —— 事件码表值稳定（app 桥接契约）——
TEST(RenderSanity, EventCodesStable) {
    EXPECT_EQ(static_cast<int32_t>(RenderEvent::SnapshotRejected), 0x0301);
    EXPECT_EQ(static_cast<int32_t>(RenderEvent::BuildDegraded),    0x0302);
    EXPECT_EQ(static_cast<int32_t>(RenderEvent::RenderSuspended),  0x0303);
    EXPECT_EQ(static_cast<int32_t>(RenderEvent::RenderResumed),    0x0304);
    EXPECT_EQ(static_cast<int32_t>(RenderEvent::FrameOverbudget),  0x0311);
    EXPECT_EQ(static_cast<int32_t>(RenderEvent::DegradeChanged),   0x0312);
}

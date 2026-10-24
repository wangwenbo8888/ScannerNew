/**
 * @file test_image_merge_cpu.cpp
 * @brief 标记点图像合并算子 - 单元测试
 *
 * 测试覆盖：
 * - 参数校验（validate / JSON 序列化/反序列化）
 * - 结果结构体默认值与移动语义
 * - 空输入（空子图列表、空 ROI 列表）
 * - 数量不匹配（子图数 != ROI 数）
 * - 单子图合并坐标精度验证
 * - 多子图合并完整性验证
 * - 边缘点属性保持（angle/amplitude 不变）
 * - groupIds 分组信息正确性验证
 * - warmup / setParams / getParams
 * - WarmupConfig 统一接口
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "../image_merge_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;


class ImageMergeCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = ImageMergeCPUParams{};
    }

    ImageMergeCPUParams params_;
};

// ============================================================
// 参数校验测试
// ============================================================
TEST_F(ImageMergeCPUTest, DefaultParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
}

// ============================================================
// JSON 序列化测试
// ============================================================
TEST_F(ImageMergeCPUTest, JsonRoundtrip) {
    auto j = params_.toJson();
    auto restored = ImageMergeCPUParams::fromJson(j);
    EXPECT_NO_THROW(restored.validate());
}

TEST_F(ImageMergeCPUTest, JsonEmptyObject) {
    nlohmann::json j = {};
    auto restored = ImageMergeCPUParams::fromJson(j);
    EXPECT_NO_THROW(restored.validate());
}

// ============================================================
// Result 结构体测试
// ============================================================
TEST_F(ImageMergeCPUTest, ResultDefaultValues) {
    ImageMergeCPUResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_EQ(result.mergedEdgeCount, 0);
    EXPECT_TRUE(result.mergedEdgePoints.empty());
    EXPECT_TRUE(result.groupIds.empty());
    EXPECT_EQ(result.groupCount, 0);
}

TEST_F(ImageMergeCPUTest, ResultMoveSemantics) {
    ImageMergeCPUResult result1;
    result1.success = true;
    result1.message = "test";
    result1.mergedEdgeCount = 5;
    result1.mergedEdgePoints.resize(5);

    ImageMergeCPUResult result2 = std::move(result1);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.message, "test");
    EXPECT_EQ(result2.mergedEdgeCount, 5);
    EXPECT_EQ(result2.mergedEdgePoints.size(), 5u);
}

// ============================================================
// WarmupConfig 测试
// ============================================================
TEST_F(ImageMergeCPUTest, WarmupConfigForImage) {
    auto config = calib::WarmupConfig::forImage(720, 1280);
    EXPECT_EQ(config.rows, 720);
    EXPECT_EQ(config.cols, 1280);
}

// ============================================================
// 构造/析构测试
// ============================================================
TEST_F(ImageMergeCPUTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW({ ImageMergeCPU op(params_); });
}

// ============================================================
// warmup 测试
// ============================================================
TEST_F(ImageMergeCPUTest, WarmupAndMerge) {
    ImageMergeCPU op(params_);
    op.Warmup(100, 100);

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = {
        { EdgePoint{10.5, 20.3, 0.0, 100.0, 10, 20} }
    };
    std::vector<cv::Rect> roiRects = { cv::Rect(50, 60, 30, 30) };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.mergedEdgeCount, 1);
}

TEST_F(ImageMergeCPUTest, WarmupWithConfig) {
    ImageMergeCPU op(params_);
    auto config = calib::WarmupConfig::forImage(100, 100);
    EXPECT_NO_THROW(op.Warmup(config));
}

// ============================================================
// merge() 空输入测试
// ============================================================
TEST_F(ImageMergeCPUTest, MergeBothEmptyReturnsWarning) {
    ImageMergeCPU op(params_);
    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage;
    std::vector<cv::Rect> roiRects;
    auto result = op.Execute(edgePointsPerSubImage, roiRects);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.mergedEdgeCount, 0);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
}

TEST_F(ImageMergeCPUTest, MergeEmptyEdgePointsWithRoiReturnsError) {
    ImageMergeCPU op(params_);
    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage;
    std::vector<cv::Rect> roiRects = { cv::Rect(0, 0, 10, 10) };
    auto result = op.Execute(edgePointsPerSubImage, roiRects);
    EXPECT_FALSE(result.success);
}

TEST_F(ImageMergeCPUTest, MergeEdgePointsWithoutRoiReturnsError) {
    ImageMergeCPU op(params_);
    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = {
        { EdgePoint{5.0, 5.0, 0.0, 50.0, 5, 5} }
    };
    std::vector<cv::Rect> roiRects;
    auto result = op.Execute(edgePointsPerSubImage, roiRects);
    EXPECT_FALSE(result.success);
}

TEST_F(ImageMergeCPUTest, MergeCountMismatchReturnsError) {
    ImageMergeCPU op(params_);
    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = {
        { EdgePoint{5.0, 5.0, 0.0, 50.0, 5, 5} }
    };
    std::vector<cv::Rect> roiRects = {
        cv::Rect(0, 0, 10, 10),
        cv::Rect(20, 20, 10, 10)
    };
    auto result = op.Execute(edgePointsPerSubImage, roiRects);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

// ============================================================
// merge() 单子图合并测试
// ============================================================
TEST_F(ImageMergeCPUTest, MergeSingleSubImage) {
    ImageMergeCPU op(params_);

    EdgePoint ep1{10.5, 20.3, 1.57, 100.0, 10, 20};
    EdgePoint ep2{15.7, 25.1, -0.5, 80.0, 15, 25};

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = { {ep1, ep2} };
    std::vector<cv::Rect> roiRects = { cv::Rect(100, 200, 50, 50) };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.mergedEdgeCount, 2);
    EXPECT_EQ(result.mergedEdgePoints.size(), 2u);

    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].x, 110.5);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].y, 220.3);
    EXPECT_EQ(result.mergedEdgePoints[0].pixelX, 110);
    EXPECT_EQ(result.mergedEdgePoints[0].pixelY, 220);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].angle, 1.57);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].amplitude, 100.0);

    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[1].x, 115.7);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[1].y, 225.1);
    EXPECT_EQ(result.mergedEdgePoints[1].pixelX, 115);
    EXPECT_EQ(result.mergedEdgePoints[1].pixelY, 225);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[1].angle, -0.5);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[1].amplitude, 80.0);
}

// ============================================================
// merge() 多子图合并测试
// ============================================================
TEST_F(ImageMergeCPUTest, MergeMultipleSubImages) {
    ImageMergeCPU op(params_);

    EdgePoint ep1{5.0, 5.0, 0.0, 50.0, 5, 5};
    EdgePoint ep2{10.0, 10.0, 1.0, 60.0, 10, 10};
    EdgePoint ep3{3.0, 7.0, 2.0, 70.0, 3, 7};

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = {
        {ep1, ep2},
        {ep3}
    };
    std::vector<cv::Rect> roiRects = {
        cv::Rect(100, 200, 50, 50),
        cv::Rect(300, 400, 50, 50)
    };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.mergedEdgeCount, 3);
    EXPECT_EQ(result.mergedEdgePoints.size(), 3u);
    EXPECT_EQ(result.groupCount, 2);
    ASSERT_EQ(result.groupIds.size(), 3u);

    EXPECT_EQ(result.groupIds[0], 0);
    EXPECT_EQ(result.groupIds[1], 0);
    EXPECT_EQ(result.groupIds[2], 1);

    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].x, 105.0);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].y, 205.0);

    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[1].x, 110.0);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[1].y, 210.0);

    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[2].x, 303.0);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[2].y, 407.0);
}

// ============================================================
// merge() groupIds 分组信息测试
// ============================================================
TEST_F(ImageMergeCPUTest, GroupIdsSingleSubImage) {
    ImageMergeCPU op(params_);

    EdgePoint ep1{5.0, 5.0, 0.0, 50.0, 5, 5};
    EdgePoint ep2{10.0, 10.0, 1.0, 60.0, 10, 10};

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = { {ep1, ep2} };
    std::vector<cv::Rect> roiRects = { cv::Rect(100, 200, 50, 50) };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.groupCount, 1);
    ASSERT_EQ(result.groupIds.size(), 2u);
    EXPECT_EQ(result.groupIds[0], 0);
    EXPECT_EQ(result.groupIds[1], 0);
}

TEST_F(ImageMergeCPUTest, GroupIdsMultipleSubImages) {
    ImageMergeCPU op(params_);

    EdgePoint ep1{5.0, 5.0, 0.0, 50.0, 5, 5};
    EdgePoint ep2{10.0, 10.0, 1.0, 60.0, 10, 10};
    EdgePoint ep3{3.0, 7.0, 2.0, 70.0, 3, 7};
    EdgePoint ep4{8.0, 9.0, 0.5, 40.0, 8, 9};

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = {
        {ep1, ep2},
        {ep3},
        {ep4}
    };
    std::vector<cv::Rect> roiRects = {
        cv::Rect(100, 200, 50, 50),
        cv::Rect(300, 400, 50, 50),
        cv::Rect(500, 600, 50, 50)
    };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.groupCount, 3);
    ASSERT_EQ(result.groupIds.size(), 4u);

    EXPECT_EQ(result.groupIds[0], 0);
    EXPECT_EQ(result.groupIds[1], 0);
    EXPECT_EQ(result.groupIds[2], 1);
    EXPECT_EQ(result.groupIds[3], 2);
}

TEST_F(ImageMergeCPUTest, GroupIdsSizeMatchesEdgePoints) {
    ImageMergeCPU op(params_);

    EdgePoint ep{5.0, 5.0, 0.0, 50.0, 5, 5};

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = {
        {ep},
        {},
        {ep, ep}
    };
    std::vector<cv::Rect> roiRects = {
        cv::Rect(100, 200, 50, 50),
        cv::Rect(300, 400, 50, 50),
        cv::Rect(500, 600, 50, 50)
    };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.groupIds.size(), result.mergedEdgePoints.size());
    EXPECT_EQ(result.groupCount, 3);

    EXPECT_EQ(result.groupIds[0], 0);
    EXPECT_EQ(result.groupIds[1], 2);
    EXPECT_EQ(result.groupIds[2], 2);
}

// ============================================================
// merge() 空子图内无边缘点测试
// ============================================================
TEST_F(ImageMergeCPUTest, MergeSubImageWithNoEdgePoints) {
    ImageMergeCPU op(params_);

    EdgePoint ep{5.0, 5.0, 0.0, 50.0, 5, 5};

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = {
        {ep},
        {}
    };
    std::vector<cv::Rect> roiRects = {
        cv::Rect(100, 200, 50, 50),
        cv::Rect(300, 400, 50, 50)
    };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.mergedEdgeCount, 1);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].x, 105.0);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].y, 205.0);
}

TEST_F(ImageMergeCPUTest, MergeAllSubImagesEmptyReturnsWarning) {
    ImageMergeCPU op(params_);

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = {
        {},
        {}
    };
    std::vector<cv::Rect> roiRects = {
        cv::Rect(100, 200, 50, 50),
        cv::Rect(300, 400, 50, 50)
    };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.mergedEdgeCount, 0);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
}

// ============================================================
// merge() 坐标偏移精度测试
// ============================================================
TEST_F(ImageMergeCPUTest, CoordinateOffsetPrecision) {
    ImageMergeCPU op(params_);

    EdgePoint ep{123.456789, 987.654321, 1.234567, 50.0, 123, 987};

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = { {ep} };
    std::vector<cv::Rect> roiRects = { cv::Rect(1000, 2000, 500, 500) };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.mergedEdgeCount, 1);

    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].x, 1123.456789);
    EXPECT_DOUBLE_EQ(result.mergedEdgePoints[0].y, 2987.654321);
    EXPECT_EQ(result.mergedEdgePoints[0].pixelX, 1123);
    EXPECT_EQ(result.mergedEdgePoints[0].pixelY, 2987);
}

TEST_F(ImageMergeCPUTest, AttributesPreserved) {
    ImageMergeCPU op(params_);

    EdgePoint ep{5.0, 5.0, 2.718, 42.5, 5, 5};

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = { {ep} };
    std::vector<cv::Rect> roiRects = { cv::Rect(100, 200, 50, 50) };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    ASSERT_TRUE(result.success);
    const auto& merged = result.mergedEdgePoints[0];
    EXPECT_DOUBLE_EQ(merged.angle, 2.718);
    EXPECT_DOUBLE_EQ(merged.amplitude, 42.5);
}

// ============================================================
// setParams / getParams 测试
// ============================================================
TEST_F(ImageMergeCPUTest, SetParamsAndGetParams) {
    ImageMergeCPU op(params_);

    ImageMergeCPUParams newParams;
    op.SetParams(newParams);
    const auto& current = op.GetParams();
    EXPECT_NO_THROW(current.validate());
}

// ============================================================
// 精度测试：与手动偏移结果一致性
// ============================================================
class ImageMergeCPUPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = ImageMergeCPUParams{};
    }

    ImageMergeCPUParams params_;
};

TEST_F(ImageMergeCPUPrecisionTest, ManualOffsetConsistency) {
    ImageMergeCPU op(params_);

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage;
    std::vector<cv::Rect> roiRects;

    std::vector<EdgePoint> subEdges1;
    for (int i = 0; i < 20; ++i) {
        EdgePoint ep;
        ep.x = static_cast<double>(i) * 1.5;
        ep.y = static_cast<double>(i) * 2.5;
        ep.angle = static_cast<double>(i) * 0.1;
        ep.amplitude = static_cast<double>(i) * 10.0;
        ep.pixelX = i;
        ep.pixelY = i * 2;
        subEdges1.push_back(ep);
    }
    edgePointsPerSubImage.push_back(subEdges1);
    roiRects.push_back(cv::Rect(100, 200, 50, 50));

    std::vector<EdgePoint> subEdges2;
    for (int i = 0; i < 15; ++i) {
        EdgePoint ep;
        ep.x = static_cast<double>(i) * 0.5 + 10.0;
        ep.y = static_cast<double>(i) * 1.5 + 5.0;
        ep.angle = static_cast<double>(i) * 0.2 + 1.0;
        ep.amplitude = static_cast<double>(i) * 5.0 + 20.0;
        ep.pixelX = i + 10;
        ep.pixelY = i + 5;
        subEdges2.push_back(ep);
    }
    edgePointsPerSubImage.push_back(subEdges2);
    roiRects.push_back(cv::Rect(500, 600, 30, 30));

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.mergedEdgeCount, 35);

    for (int i = 0; i < 20; ++i) {
        const auto& merged = result.mergedEdgePoints[i];
        EXPECT_DOUBLE_EQ(merged.x, static_cast<double>(i) * 1.5 + 100.0);
        EXPECT_DOUBLE_EQ(merged.y, static_cast<double>(i) * 2.5 + 200.0);
        EXPECT_DOUBLE_EQ(merged.angle, static_cast<double>(i) * 0.1);
        EXPECT_DOUBLE_EQ(merged.amplitude, static_cast<double>(i) * 10.0);
        EXPECT_EQ(merged.pixelX, i + 100);
        EXPECT_EQ(merged.pixelY, i * 2 + 200);
    }

    for (int i = 0; i < 15; ++i) {
        const auto& merged = result.mergedEdgePoints[20 + i];
        EXPECT_DOUBLE_EQ(merged.x, static_cast<double>(i) * 0.5 + 10.0 + 500.0);
        EXPECT_DOUBLE_EQ(merged.y, static_cast<double>(i) * 1.5 + 5.0 + 600.0);
        EXPECT_DOUBLE_EQ(merged.angle, static_cast<double>(i) * 0.2 + 1.0);
        EXPECT_DOUBLE_EQ(merged.amplitude, static_cast<double>(i) * 5.0 + 20.0);
        EXPECT_EQ(merged.pixelX, i + 10 + 500);
        EXPECT_EQ(merged.pixelY, i + 5 + 600);
    }
}

TEST_F(ImageMergeCPUPrecisionTest, ZeroOffsetIdentity) {
    ImageMergeCPU op(params_);

    EdgePoint ep{42.0, 84.0, 1.0, 100.0, 42, 84};

    std::vector<std::vector<EdgePoint>> edgePointsPerSubImage = { {ep} };
    std::vector<cv::Rect> roiRects = { cv::Rect(0, 0, 100, 100) };

    auto result = op.Execute(edgePointsPerSubImage, roiRects);

    ASSERT_TRUE(result.success);
    const auto& merged = result.mergedEdgePoints[0];
    EXPECT_DOUBLE_EQ(merged.x, ep.x);
    EXPECT_DOUBLE_EQ(merged.y, ep.y);
    EXPECT_EQ(merged.pixelX, ep.pixelX);
    EXPECT_EQ(merged.pixelY, ep.pixelY);
    EXPECT_DOUBLE_EQ(merged.angle, ep.angle);
    EXPECT_DOUBLE_EQ(merged.amplitude, ep.amplitude);
}

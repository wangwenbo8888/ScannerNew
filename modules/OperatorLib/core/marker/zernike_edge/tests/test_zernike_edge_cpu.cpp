/**
 * @file test_zernike_edge_cpu.cpp
 * @brief Zernike椭圆边缘亚像素提取算子 - 单元测试
 *
 * 测试覆盖：
 * - 参数校验（合法参数、JSON 序列化/反序列化、非法参数拒绝）
 * - 构造/析构
 * - warmup 预热
 * - extract 正常处理（合成阶跃边缘、合成椭圆）
 * - extract 空图像/错误类型输入
 * - extract 全黑图像（无边缘）
 * - 亚像素精度验证
 * - 模板切换 5x5 vs 7x7
 * - setParams / getParams 动态更新
 * - Result 结构体移动语义
 * - warmup(WarmupConfig) 统一配置接口
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../zernike_edge_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;


class ZernikeEdgeCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = ZernikeEdgeCPUParams{};
    }

    ZernikeEdgeCPUParams params_;
};

static cv::Mat createStepEdgeImage(int rows, int cols, int edgeX)
{
    cv::Mat img(rows, cols, CV_8UC1, cv::Scalar(50));
    for (int y = 0; y < rows; ++y) {
        for (int x = edgeX; x < cols; ++x) {
            img.at<uchar>(y, x) = 200;
        }
    }
    return img;
}

static cv::Mat createEllipseImage(int rows, int cols, cv::Point2d center,
                                   double radiusX, double radiusY)
{
    cv::Mat img = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::ellipse(img, center, cv::Size(static_cast<int>(radiusX), static_cast<int>(radiusY)),
                0, 0, 360, cv::Scalar(255), -1);
    cv::GaussianBlur(img, img, cv::Size(5, 5), 1.5);
    return img;
}

// ============================================================
// 参数校验测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, DefaultParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
    EXPECT_EQ(params_.templateSize, 5);
    EXPECT_DOUBLE_EQ(params_.cannyLowThreshold, 50.0);
    EXPECT_DOUBLE_EQ(params_.cannyHighThreshold, 150.0);
}

TEST_F(ZernikeEdgeCPUTest, TemplateSize7IsValid) {
    params_.templateSize = 7;
    EXPECT_NO_THROW(params_.validate());
}

TEST_F(ZernikeEdgeCPUTest, InvalidTemplateSize6Rejected) {
    params_.templateSize = 6;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(ZernikeEdgeCPUTest, InvalidTemplateSize3Rejected) {
    params_.templateSize = 3;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(ZernikeEdgeCPUTest, NegativeLowThresholdRejected) {
    params_.cannyLowThreshold = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(ZernikeEdgeCPUTest, LowThresholdGeqHighRejected) {
    params_.cannyLowThreshold = 200;
    params_.cannyHighThreshold = 100;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(ZernikeEdgeCPUTest, EvenGaussianKernelRejected) {
    params_.gaussianKernelSize = 4;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(ZernikeEdgeCPUTest, NegativeSigmaRejected) {
    params_.gaussianSigma = -0.5;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(ZernikeEdgeCPUTest, InvalidSobelApertureRejected) {
    params_.sobelApertureSize = 4;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

// ============================================================
// JSON 序列化测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, JsonRoundtrip) {
    params_.templateSize = 7;
    params_.cannyLowThreshold = 30.0;
    auto j = params_.toJson();
    auto restored = ZernikeEdgeCPUParams::fromJson(j);
    EXPECT_EQ(restored.templateSize, 7);
    EXPECT_DOUBLE_EQ(restored.cannyLowThreshold, 30.0);
    EXPECT_DOUBLE_EQ(restored.cannyHighThreshold, 150.0);
    EXPECT_DOUBLE_EQ(restored.gaussianSigma, 1.0);
    EXPECT_EQ(restored.gaussianKernelSize, 3);
    EXPECT_DOUBLE_EQ(restored.edgeStrengthThreshold, 20.0);
    EXPECT_EQ(restored.sobelApertureSize, 3);
}

TEST_F(ZernikeEdgeCPUTest, JsonPartialDeserialization) {
    nlohmann::json j = {{"templateSize", 7}};
    auto restored = ZernikeEdgeCPUParams::fromJson(j);
    EXPECT_EQ(restored.templateSize, 7);
    EXPECT_DOUBLE_EQ(restored.cannyLowThreshold, 50.0);
}

TEST_F(ZernikeEdgeCPUTest, JsonEmptyObject) {
    nlohmann::json j = {};
    auto restored = ZernikeEdgeCPUParams::fromJson(j);
    EXPECT_EQ(restored.templateSize, 5);
}

// ============================================================
// Result 结构体测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, ResultDefaultValues) {
    ZernikeEdgeCPUResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_EQ(result.edgeCount, 0);
    EXPECT_TRUE(result.edgePoints.empty());
    EXPECT_TRUE(result.cannyEdgeImage.empty());
}

TEST_F(ZernikeEdgeCPUTest, ResultMoveSemantics) {
    ZernikeEdgeCPUResult result1;
    result1.success = true;
    result1.message = "test";
    result1.edgeCount = 5;
    result1.edgePoints.resize(5);

    ZernikeEdgeCPUResult result2 = std::move(result1);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.message, "test");
    EXPECT_EQ(result2.edgeCount, 5);
    EXPECT_EQ(result2.edgePoints.size(), 5u);
}

// ============================================================
// WarmupConfig 测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, WarmupConfigForImage) {
    auto config = calib::WarmupConfig::forImage(720, 1280);
    EXPECT_EQ(config.rows, 720);
    EXPECT_EQ(config.cols, 1280);
}

// ============================================================
// 构造/析构测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW({ ZernikeEdgeCPU op(params_); });
}

TEST_F(ZernikeEdgeCPUTest, ConstructWithTemplateSize7) {
    params_.templateSize = 7;
    EXPECT_NO_THROW({ ZernikeEdgeCPU op(params_); });
}

// ============================================================
// warmup 测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, WarmupAndExtract) {
    ZernikeEdgeCPU op(params_);
    op.Warmup(100, 100);

    cv::Mat img = createStepEdgeImage(100, 100, 50);
    auto result = op.Execute(img);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.edgeCount, 0);
}

TEST_F(ZernikeEdgeCPUTest, WarmupWithConfig) {
    ZernikeEdgeCPU op(params_);
    auto config = calib::WarmupConfig::forImage(100, 100);
    EXPECT_NO_THROW(op.Warmup(config));
}

// ============================================================
// extract() 正常处理测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, ExtractStepEdge) {
    ZernikeEdgeCPU op(params_);
    cv::Mat img = createStepEdgeImage(100, 100, 50);
    auto result = op.Execute(img);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.edgeCount, 0);
    EXPECT_FALSE(result.cannyEdgeImage.empty());
}

TEST_F(ZernikeEdgeCPUTest, ExtractStepEdgeSubpixelNearInteger) {
    ZernikeEdgeCPU op(params_);
    int edgeX = 50;
    cv::Mat img = createStepEdgeImage(100, 100, edgeX);
    auto result = op.Execute(img);

    ASSERT_TRUE(result.success);
    ASSERT_GT(result.edgeCount, 0);

    double sumX = 0.0;
    int count = 0;
    for (const auto& ep : result.edgePoints) {
        if (ep.pixelY > 10 && ep.pixelY < 90) {
            sumX += ep.x;
            ++count;
        }
    }

    if (count > 0) {
        double avgX = sumX / count;
        EXPECT_NEAR(avgX, static_cast<double>(edgeX), 2.0)
            << "Average sub-pixel edge position should be near the integer edge";
    }
}

TEST_F(ZernikeEdgeCPUTest, ExtractEllipse) {
    ZernikeEdgeCPU op(params_);
    cv::Mat img = createEllipseImage(200, 200, cv::Point2d(100, 100), 60, 40);
    auto result = op.Execute(img);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.edgeCount, 0);

    for (const auto& ep : result.edgePoints) {
        EXPECT_GT(ep.amplitude, 0.0);
    }
}

TEST_F(ZernikeEdgeCPUTest, ExtractEdgePointFieldsValid) {
    ZernikeEdgeCPU op(params_);
    cv::Mat img = createStepEdgeImage(100, 100, 50);
    auto result = op.Execute(img);

    ASSERT_TRUE(result.success);
    ASSERT_GT(result.edgePoints.size(), 0u);

    const auto& ep = result.edgePoints[0];
    EXPECT_GE(ep.pixelX, 0);
    EXPECT_GE(ep.pixelY, 0);
    EXPECT_LT(ep.pixelX, 100);
    EXPECT_LT(ep.pixelY, 100);
    EXPECT_GT(ep.amplitude, 0.0);
    EXPECT_GT(ep.x, -1.0);
    EXPECT_LT(ep.x, 101.0);
    EXPECT_GT(ep.y, -1.0);
    EXPECT_LT(ep.y, 101.0);
}

// ============================================================
// extract() 空输入/错误输入测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, ExtractEmptyImageReturnsError) {
    ZernikeEdgeCPU op(params_);
    cv::Mat empty;
    auto result = op.Execute(empty);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(ZernikeEdgeCPUTest, ExtractWrongTypeReturnsError) {
    ZernikeEdgeCPU op(params_);
    cv::Mat color(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = op.Execute(color);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(ZernikeEdgeCPUTest, ExtractBlackImageNoEdges) {
    ZernikeEdgeCPU op(params_);
    cv::Mat black = cv::Mat::zeros(100, 100, CV_8UC1);
    auto result = op.Execute(black);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.edgeCount, 0);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
}

TEST_F(ZernikeEdgeCPUTest, ExtractTooSmallImage) {
    params_.templateSize = 7;
    ZernikeEdgeCPU op(params_);
    cv::Mat small(5, 5, CV_8UC1, cv::Scalar(128));
    auto result = op.Execute(small);
    EXPECT_FALSE(result.success);
}

// ============================================================
// 模板切换测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, TemplateSize5And7BothWork) {
    cv::Mat img = createStepEdgeImage(100, 100, 50);

    params_.templateSize = 5;
    ZernikeEdgeCPU op5(params_);
    auto result5 = op5.Execute(img);
    EXPECT_TRUE(result5.success);
    EXPECT_GT(result5.edgeCount, 0);

    params_.templateSize = 7;
    ZernikeEdgeCPU op7(params_);
    auto result7 = op7.Execute(img);
    EXPECT_TRUE(result7.success);
    EXPECT_GT(result7.edgeCount, 0);
}

// ============================================================
// setParams / getParams 测试
// ============================================================
TEST_F(ZernikeEdgeCPUTest, SetParamsAndGetParams) {
    ZernikeEdgeCPU op(params_);

    ZernikeEdgeCPUParams newParams;
    newParams.templateSize = 7;
    newParams.cannyLowThreshold = 30.0;
    newParams.cannyHighThreshold = 100.0;

    op.SetParams(newParams);
    const auto& current = op.GetParams();
    EXPECT_EQ(current.templateSize, 7);
    EXPECT_DOUBLE_EQ(current.cannyLowThreshold, 30.0);
}

TEST_F(ZernikeEdgeCPUTest, SetParamsAffectsExtractBehavior) {
    ZernikeEdgeCPU op(params_);
    cv::Mat img = createStepEdgeImage(100, 100, 50);

    auto result1 = op.Execute(img);
    int count1 = result1.edgeCount;

    ZernikeEdgeCPUParams newParams;
    newParams.cannyLowThreshold = 10.0;
    newParams.cannyHighThreshold = 50.0;
    newParams.edgeStrengthThreshold = 5.0;
    op.SetParams(newParams);

    auto result2 = op.Execute(img);
    int count2 = result2.edgeCount;

    EXPECT_GE(count2, count1);
}

// ============================================================
// 精度测试
// ============================================================
class ZernikeEdgeCPUPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = ZernikeEdgeCPUParams{};
    }

    ZernikeEdgeCPUParams params_;
};

TEST_F(ZernikeEdgeCPUPrecisionTest, VerticalEdgeSubpixelAccuracy) {
    params_.edgeStrengthThreshold = 5.0;
    ZernikeEdgeCPU op(params_);

    int trueEdgeX = 60;
    cv::Mat img = createStepEdgeImage(200, 200, trueEdgeX);

    auto result = op.Execute(img);
    ASSERT_TRUE(result.success);
    ASSERT_GT(result.edgeCount, 0);

    double sumX = 0.0;
    int count = 0;
    for (const auto& ep : result.edgePoints) {
        if (ep.pixelX >= trueEdgeX - 3 && ep.pixelX <= trueEdgeX + 3 &&
            ep.pixelY > 10 && ep.pixelY < 190) {
            sumX += ep.x;
            ++count;
        }
    }

    if (count > 5) {
        double avgX = sumX / count;
        EXPECT_NEAR(avgX, static_cast<double>(trueEdgeX), 1.0)
            << "Sub-pixel edge position should be within 1 pixel of true edge";
    }
}

TEST_F(ZernikeEdgeCPUPrecisionTest, HorizontalEdgeSubpixelAccuracy) {
    params_.edgeStrengthThreshold = 5.0;
    ZernikeEdgeCPU op(params_);

    int trueEdgeY = 80;
    cv::Mat img(200, 200, CV_8UC1, cv::Scalar(50));
    for (int y = trueEdgeY; y < 200; ++y) {
        for (int x = 0; x < 200; ++x) {
            img.at<uchar>(y, x) = 200;
        }
    }

    auto result = op.Execute(img);
    ASSERT_TRUE(result.success);
    ASSERT_GT(result.edgeCount, 0);

    double sumY = 0.0;
    int count = 0;
    for (const auto& ep : result.edgePoints) {
        if (ep.pixelY >= trueEdgeY - 3 && ep.pixelY <= trueEdgeY + 3 &&
            ep.pixelX > 10 && ep.pixelX < 190) {
            sumY += ep.y;
            ++count;
        }
    }

    if (count > 5) {
        double avgY = sumY / count;
        EXPECT_NEAR(avgY, static_cast<double>(trueEdgeY), 1.0)
            << "Sub-pixel edge position should be within 1 pixel of true edge";
    }
}

TEST_F(ZernikeEdgeCPUPrecisionTest, EdgeAngleRange) {
    ZernikeEdgeCPU op(params_);
    cv::Mat img = createStepEdgeImage(100, 100, 50);
    auto result = op.Execute(img);

    ASSERT_TRUE(result.success);
    for (const auto& ep : result.edgePoints) {
        EXPECT_GE(ep.angle, -CV_PI);
        EXPECT_LE(ep.angle, CV_PI);
    }
}

TEST_F(ZernikeEdgeCPUPrecisionTest, TemplateSize5DetectsCorrectEdge) {
    int trueEdgeX = 75;
    cv::Mat img = createStepEdgeImage(150, 150, trueEdgeX);

    params_.templateSize = 5;
    ZernikeEdgeCPU op5(params_);
    auto result5 = op5.Execute(img);
    ASSERT_TRUE(result5.success);
    ASSERT_GT(result5.edgeCount, 10);

    double sum5 = 0.0;
    int c5 = 0;
    for (const auto& ep : result5.edgePoints) {
        if (ep.pixelY > 20 && ep.pixelY < 130) { sum5 += ep.x; ++c5; }
    }
    ASSERT_GT(c5, 0);
    double avg5 = sum5 / c5;
    EXPECT_NEAR(avg5, static_cast<double>(trueEdgeX), 2.0)
        << "5x5 template should detect edge near true position";
}

TEST_F(ZernikeEdgeCPUPrecisionTest, TemplateSize7DetectsCorrectEdge) {
    int trueEdgeX = 75;
    cv::Mat img = createStepEdgeImage(150, 150, trueEdgeX);

    params_.templateSize = 7;
    ZernikeEdgeCPU op7(params_);
    auto result7 = op7.Execute(img);
    ASSERT_TRUE(result7.success);
    ASSERT_GT(result7.edgeCount, 10);

    double sum7 = 0.0;
    int c7 = 0;
    for (const auto& ep : result7.edgePoints) {
        if (ep.pixelY > 20 && ep.pixelY < 130) { sum7 += ep.x; ++c7; }
    }
    ASSERT_GT(c7, 0);
    double avg7 = sum7 / c7;
    EXPECT_NEAR(avg7, static_cast<double>(trueEdgeX), 3.0)
        << "7x7 template should detect edge near true position";
}

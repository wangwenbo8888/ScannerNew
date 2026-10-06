/**
 * @file test_image_split_cpu.cpp
 * @brief 标记点图像分割算子 - 单元测试
 *
 * 测试覆盖：
 * - 参数校验（合法参数、JSON 序列化/反序列化）
 * - 构造/析构
 * - warmup 预热
 * - split 正常处理（单 ROI、多 ROI、边界裁剪）
 * - split 空图像/错误类型输入
 * - split 空 ROI 列表
 * - setParams 动态更新
 * - getParams 参数查询
 * - warmup(WarmupConfig) 统一配置接口
 * - Result 结构体移动语义
 * - 精度测试（与原始 ImageSplitter 结果一致性）
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../image_split_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;


class ImageSplitCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = ImageSplitCPUParams{};
    }

    ImageSplitCPUParams params_;
};

// ============================================================
// 参数校验测试
// ============================================================
TEST_F(ImageSplitCPUTest, DefaultParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
    EXPECT_FALSE(params_.enableBoundaryCheck);
}

TEST_F(ImageSplitCPUTest, EnableBoundaryCheckIsValid) {
    params_.enableBoundaryCheck = true;
    EXPECT_NO_THROW(params_.validate());
}

// ============================================================
// JSON 序列化测试
// ============================================================
TEST_F(ImageSplitCPUTest, JsonRoundtrip) {
    params_.enableBoundaryCheck = true;
    auto j = params_.toJson();
    auto restored = ImageSplitCPUParams::fromJson(j);
    EXPECT_EQ(restored.enableBoundaryCheck, true);
}

TEST_F(ImageSplitCPUTest, JsonPartialDeserialization) {
    nlohmann::json j = {{"enableBoundaryCheck", true}};
    auto restored = ImageSplitCPUParams::fromJson(j);
    EXPECT_EQ(restored.enableBoundaryCheck, true);
}

TEST_F(ImageSplitCPUTest, JsonEmptyObject) {
    nlohmann::json j = {};
    auto restored = ImageSplitCPUParams::fromJson(j);
    EXPECT_EQ(restored.enableBoundaryCheck, false);
}

// ============================================================
// Result 结构体测试
// ============================================================
TEST_F(ImageSplitCPUTest, ResultDefaultValues) {
    ImageSplitCPUResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_EQ(result.splitCount, 0);
    EXPECT_TRUE(result.splitImages.empty());
}

TEST_F(ImageSplitCPUTest, ResultMoveSemantics) {
    ImageSplitCPUResult result1;
    result1.success = true;
    result1.message = "test";
    result1.splitCount = 3;

    ImageSplitCPUResult result2 = std::move(result1);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.message, "test");
    EXPECT_EQ(result2.splitCount, 3);
}

// ============================================================
// WarmupConfig 测试
// ============================================================
TEST_F(ImageSplitCPUTest, WarmupConfigForImage) {
    auto config = calib::WarmupConfig::forImage(720, 1280);
    EXPECT_EQ(config.rows, 720);
    EXPECT_EQ(config.cols, 1280);
}

// ============================================================
// 构造/析构测试
// ============================================================
TEST_F(ImageSplitCPUTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW({
        ImageSplitCPU splitter(params_);
    });
}

TEST_F(ImageSplitCPUTest, ConstructWithBoundaryCheckEnabled) {
    params_.enableBoundaryCheck = true;
    EXPECT_NO_THROW({
        ImageSplitCPU splitter(params_);
    });
}

// ============================================================
// warmup 测试
// ============================================================
TEST_F(ImageSplitCPUTest, WarmupAndSplit) {
    ImageSplitCPU splitter(params_);
    splitter.Warmup(100, 100);

    cv::Mat gray = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(gray, cv::Point(10, 10), cv::Point(30, 30), cv::Scalar(200), -1);

    std::vector<cv::Rect> rois = { cv::Rect(5, 5, 30, 30) };
    auto result = splitter.Execute(gray, rois);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.splitCount, 1);
    EXPECT_EQ(result.splitImages.size(), 1u);
}

TEST_F(ImageSplitCPUTest, WarmupWithConfig) {
    ImageSplitCPU splitter(params_);
    auto config = calib::WarmupConfig::forImage(100, 100);
    EXPECT_NO_THROW(splitter.Warmup(config));
}

// ============================================================
// split() 正常处理测试
// ============================================================
TEST_F(ImageSplitCPUTest, SplitSingleROI) {
    ImageSplitCPU splitter(params_);

    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> rois = { cv::Rect(10, 10, 20, 20) };

    auto result = splitter.Execute(gray, rois);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.splitCount, 1);
    EXPECT_EQ(result.splitImages[0].rows, 20);
    EXPECT_EQ(result.splitImages[0].cols, 20);
}

TEST_F(ImageSplitCPUTest, SplitMultipleROIs) {
    ImageSplitCPU splitter(params_);

    cv::Mat gray(200, 200, CV_8UC1, cv::Scalar(100));
    std::vector<cv::Rect> rois = {
        cv::Rect(10, 10, 30, 30),
        cv::Rect(50, 50, 40, 40),
        cv::Rect(100, 100, 50, 50)
    };

    auto result = splitter.Execute(gray, rois);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.splitCount, 3);
    EXPECT_EQ(result.splitImages.size(), 3u);
    EXPECT_EQ(result.splitImages[0].rows, 30);
    EXPECT_EQ(result.splitImages[1].rows, 40);
    EXPECT_EQ(result.splitImages[2].rows, 50);
}

TEST_F(ImageSplitCPUTest, SplitDeepCopyIndependence) {
    ImageSplitCPU splitter(params_);

    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> rois = { cv::Rect(0, 0, 50, 50) };

    auto result = splitter.Execute(gray, rois);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.splitCount, 1);

    gray.at<uchar>(0, 0) = 0;
    EXPECT_EQ(result.splitImages[0].at<uchar>(0, 0), 128);
}

// ============================================================
// split() 边界检查测试
// ============================================================
TEST_F(ImageSplitCPUTest, SplitWithBoundaryCheckClipping) {
    params_.enableBoundaryCheck = true;
    ImageSplitCPU splitter(params_);

    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> rois = { cv::Rect(80, 80, 30, 30) };

    auto result = splitter.Execute(gray, rois);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.splitCount, 1);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Degraded);
    EXPECT_EQ(result.splitImages[0].rows, 20);
    EXPECT_EQ(result.splitImages[0].cols, 20);
}

TEST_F(ImageSplitCPUTest, SplitWithBoundaryCheckFullyOutOfBounds) {
    params_.enableBoundaryCheck = true;
    ImageSplitCPU splitter(params_);

    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> rois = { cv::Rect(200, 200, 10, 10) };

    auto result = splitter.Execute(gray, rois);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.splitCount, 0);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
}

TEST_F(ImageSplitCPUTest, SplitWithoutBoundaryCheckFullyOutOfBounds) {
    params_.enableBoundaryCheck = false;
    ImageSplitCPU splitter(params_);

    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> rois = { cv::Rect(200, 200, 10, 10) };

    auto result = splitter.Execute(gray, rois);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.splitCount, 0);
}

TEST_F(ImageSplitCPUTest, SplitNegativeSizeROIIgnored) {
    ImageSplitCPU splitter(params_);

    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> rois = { cv::Rect(10, 10, -5, 10) };

    auto result = splitter.Execute(gray, rois);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.splitCount, 0);
}

TEST_F(ImageSplitCPUTest, SplitZeroSizeROIIgnored) {
    ImageSplitCPU splitter(params_);

    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> rois = { cv::Rect(10, 10, 0, 10) };

    auto result = splitter.Execute(gray, rois);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.splitCount, 0);
}

// ============================================================
// split() 空输入测试
// ============================================================
TEST_F(ImageSplitCPUTest, SplitEmptyImageReturnsError) {
    ImageSplitCPU splitter(params_);
    cv::Mat empty;
    std::vector<cv::Rect> rois = { cv::Rect(0, 0, 10, 10) };
    auto result = splitter.Execute(empty, rois);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(ImageSplitCPUTest, SplitWrongTypeReturnsError) {
    ImageSplitCPU splitter(params_);
    cv::Mat color(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Rect> rois = { cv::Rect(0, 0, 10, 10) };
    auto result = splitter.Execute(color, rois);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(ImageSplitCPUTest, SplitEmptyROIReturnsWarning) {
    ImageSplitCPU splitter(params_);
    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> rois;
    auto result = splitter.Execute(gray, rois);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.splitCount, 0);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
}

// ============================================================
// setParams / getParams 测试
// ============================================================
TEST_F(ImageSplitCPUTest, SetParamsAndGetParams) {
    ImageSplitCPU splitter(params_);

    ImageSplitCPUParams newParams;
    newParams.enableBoundaryCheck = true;

    splitter.SetParams(newParams);
    const auto& current = splitter.GetParams();
    EXPECT_TRUE(current.enableBoundaryCheck);
}

TEST_F(ImageSplitCPUTest, SetParamsAffectsSplitBehavior) {
    ImageSplitCPU splitter(params_);

    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Rect> rois = { cv::Rect(80, 80, 30, 30) };

    auto result1 = splitter.Execute(gray, rois);
    EXPECT_EQ(result1.splitCount, 0);

    ImageSplitCPUParams newParams;
    newParams.enableBoundaryCheck = true;
    splitter.SetParams(newParams);

    auto result2 = splitter.Execute(gray, rois);
    EXPECT_EQ(result2.splitCount, 1);
    EXPECT_EQ(result2.splitImages[0].rows, 20);
    EXPECT_EQ(result2.splitImages[0].cols, 20);
}

// ============================================================
// 精度测试：与原始 ImageSplitter 逻辑一致性
// ============================================================

static int originalSplitImage(const cv::Mat& srcImg,
                               const std::vector<cv::Rect>& roiRects,
                               std::vector<cv::Mat>& dstImgs,
                               bool enableBoundaryCheck)
{
    if (srcImg.empty() || roiRects.empty()) {
        dstImgs.clear();
        return 0;
    }

    dstImgs.clear();
    dstImgs.reserve(roiRects.size());

    const cv::Rect imgBounds(0, 0, srcImg.cols, srcImg.rows);

    for (const auto& rect : roiRects) {
        cv::Rect validRect = rect;

        if (enableBoundaryCheck) {
            validRect = rect & imgBounds;
            if (validRect.width <= 0 || validRect.height <= 0) {
                continue;
            }
        } else {
            if (validRect.width <= 0 || validRect.height <= 0) {
                continue;
            }
        }

        dstImgs.emplace_back(srcImg(validRect).clone());
    }

    return static_cast<int>(dstImgs.size());
}

class ImageSplitCPUPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = ImageSplitCPUParams{};
    }

    ImageSplitCPUParams params_;
};

TEST_F(ImageSplitCPUPrecisionTest, ConsistentWithOriginalNoBoundary) {
    ImageSplitCPU splitter(params_);

    cv::Mat gray(256, 256, CV_8UC1);
    cv::randu(gray, 0, 256);

    std::vector<cv::Rect> rois = {
        cv::Rect(10, 10, 50, 50),
        cv::Rect(100, 100, 60, 60),
        cv::Rect(200, 200, 40, 40)
    };

    auto result = splitter.Execute(gray, rois);

    std::vector<cv::Mat> origResults;
    int origCount = originalSplitImage(gray, rois, origResults, false);

    EXPECT_EQ(result.splitCount, origCount);
    ASSERT_EQ(result.splitImages.size(), origResults.size());

    for (size_t i = 0; i < result.splitImages.size(); i++) {
        cv::Mat diff;
        cv::absdiff(result.splitImages[i], origResults[i], diff);
        int mismatches = cv::countNonZero(diff);
        EXPECT_EQ(mismatches, 0) << "Mismatch at ROI index " << i;
    }
}

TEST_F(ImageSplitCPUPrecisionTest, ConsistentWithOriginalWithBoundary) {
    params_.enableBoundaryCheck = true;
    ImageSplitCPU splitter(params_);

    cv::Mat gray(256, 256, CV_8UC1);
    cv::randu(gray, 0, 256);

    std::vector<cv::Rect> rois = {
        cv::Rect(10, 10, 50, 50),
        cv::Rect(200, 200, 80, 80),
        cv::Rect(300, 300, 20, 20)
    };

    auto result = splitter.Execute(gray, rois);

    std::vector<cv::Mat> origResults;
    int origCount = originalSplitImage(gray, rois, origResults, true);

    EXPECT_EQ(result.splitCount, origCount);
    ASSERT_EQ(result.splitImages.size(), origResults.size());

    for (size_t i = 0; i < result.splitImages.size(); i++) {
        cv::Mat diff;
        cv::absdiff(result.splitImages[i], origResults[i], diff);
        int mismatches = cv::countNonZero(diff);
        EXPECT_EQ(mismatches, 0) << "Mismatch at ROI index " << i;
    }
}

TEST_F(ImageSplitCPUPrecisionTest, ConsistentWithOriginalAllOutOfBounds) {
    params_.enableBoundaryCheck = true;
    ImageSplitCPU splitter(params_);

    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(128));

    std::vector<cv::Rect> rois = {
        cv::Rect(200, 200, 10, 10),
        cv::Rect(-50, -50, 10, 10)
    };

    auto result = splitter.Execute(gray, rois);

    std::vector<cv::Mat> origResults;
    int origCount = originalSplitImage(gray, rois, origResults, true);

    EXPECT_EQ(result.splitCount, origCount);
    EXPECT_EQ(origCount, 0);
}

TEST_F(ImageSplitCPUPrecisionTest, PixelExactContentMatch) {
    ImageSplitCPU splitter(params_);

    cv::Mat gray(128, 128, CV_8UC1);
    cv::randu(gray, 50, 200);

    cv::Rect roi(20, 30, 40, 50);
    std::vector<cv::Rect> rois = { roi };

    auto result = splitter.Execute(gray, rois);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.splitCount, 1);

    for (int r = 0; r < roi.height; r++) {
        for (int c = 0; c < roi.width; c++) {
            EXPECT_EQ(result.splitImages[0].at<uchar>(r, c),
                      gray.at<uchar>(roi.y + r, roi.x + c))
                << "Pixel mismatch at (" << r << ", " << c << ")";
        }
    }
}

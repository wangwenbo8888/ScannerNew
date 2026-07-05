/**
 * @file test_laser_markingpoint_mask_separation_cuda.cpp
 * @brief 激光线与标记点掩膜分离算子 - 单元测试
 *
 * 测试覆盖：
 * - 参数校验（合法/非法参数）
 * - 构造/析构
 * - warmup 预热
 * - separate 正常处理
 * - separate 空图像/错误类型输入
 * - setParams 动态更新
 * - getParams 参数查询
 * - JSON 序列化/反序列化
 * - Result 结构体移动语义
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../laser_markingpoint_mask_separation_cuda.h"
#include "common/calib_warmup_config.h"

using namespace calib;


class LaserMarkingSepTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = LaserMarkingSeparationParams{};
    }

    LaserMarkingSeparationParams params_;
};

// ============================================================
// 参数校验测试
// ============================================================
TEST_F(LaserMarkingSepTest, DefaultParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
    EXPECT_EQ(params_.gaussianSize, 5);
    EXPECT_EQ(params_.threshold, 80);
    EXPECT_EQ(params_.step2_erodeSize, 3);
    EXPECT_EQ(params_.step3_erodeSize, 5);
    EXPECT_EQ(params_.step4_dilateSize, 9);
    EXPECT_EQ(params_.step6_dilateSize, 5);
}

TEST_F(LaserMarkingSepTest, InvalidThresholdThrows) {
    params_.threshold = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
    params_.threshold = 256;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(LaserMarkingSepTest, EvenGaussianSizeThrows) {
    params_.gaussianSize = 4;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(LaserMarkingSepTest, EvenStep2ErodeSizeThrows) {
    params_.step2_erodeSize = 2;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(LaserMarkingSepTest, Step3ErodeNotLargerThanStep2Throws) {
    params_.step2_erodeSize = 5;
    params_.step3_erodeSize = 5;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(LaserMarkingSepTest, Step4DilateNotLargerThanStep3Throws) {
    params_.step3_dilateSize = 9;
    params_.step4_dilateSize = 9;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

// ============================================================
// JSON 序列化测试
// ============================================================
TEST_F(LaserMarkingSepTest, JsonRoundtrip) {
    auto j = params_.toJson();
    auto restored = LaserMarkingSeparationParams::fromJson(j);
    EXPECT_EQ(restored.gaussianSize, params_.gaussianSize);
    EXPECT_EQ(restored.threshold, params_.threshold);
    EXPECT_EQ(restored.step2_erodeSize, params_.step2_erodeSize);
    EXPECT_EQ(restored.step2_dilateSize, params_.step2_dilateSize);
    EXPECT_EQ(restored.step3_erodeSize, params_.step3_erodeSize);
    EXPECT_EQ(restored.step3_dilateSize, params_.step3_dilateSize);
    EXPECT_EQ(restored.step4_erodeSize, params_.step4_erodeSize);
    EXPECT_EQ(restored.step4_dilateSize, params_.step4_dilateSize);
    EXPECT_EQ(restored.step6_dilateSize, params_.step6_dilateSize);
}

TEST_F(LaserMarkingSepTest, JsonPartialDeserialization) {
    nlohmann::json j = {{"threshold", 120}};
    auto restored = LaserMarkingSeparationParams::fromJson(j);
    EXPECT_EQ(restored.threshold, 120);
    EXPECT_EQ(restored.gaussianSize, 5);
}

// ============================================================
// Result 结构体测试
// ============================================================
TEST_F(LaserMarkingSepTest, ResultDefaultValues) {
    LaserMarkingSeparationResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_EQ(result.d_laserMask, nullptr);
    EXPECT_EQ(result.d_markingPointMask, nullptr);
    EXPECT_EQ(result.d_combinedMask, nullptr);
}

TEST_F(LaserMarkingSepTest, ResultMoveSemantics) {
    LaserMarkingSeparationResult result1;
    result1.success = true;
    result1.message = "test";

    LaserMarkingSeparationResult result2 = std::move(result1);
    EXPECT_TRUE(result2.success);
    EXPECT_EQ(result2.message, "test");
}

// ============================================================
// GPU 测试（需要 CUDA）
// ============================================================
#ifdef WITH_CUDA_TESTS
TEST_F(LaserMarkingSepTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW({
        LaserMarkingSeparationCUDA separator(params_);
    });
}

TEST_F(LaserMarkingSepTest, ConstructWithInvalidParamsThrows) {
    LaserMarkingSeparationParams badParams;
    badParams.threshold = 999;
    EXPECT_THROW({
        LaserMarkingSeparationCUDA separator(badParams);
    }, std::invalid_argument);
}

TEST_F(LaserMarkingSepTest, WarmupAndSeparate) {
    LaserMarkingSeparationCUDA separator(params_);
    separator.Warmup(200, 200);

    cv::Mat gray = cv::Mat::zeros(200, 200, CV_8UC1);
    cv::rectangle(gray, cv::Point(10, 85), cv::Point(190, 115),
                  cv::Scalar(200), -1);
    cv::circle(gray, cv::Point(50, 50), 5, cv::Scalar(200), -1);
    cv::circle(gray, cv::Point(150, 50), 5, cv::Scalar(200), -1);

    // Run multiple iterations for stable timing
    constexpr int kIterations = 10;
    MaskSeparationTimings avgTimings{};

    for (int i = 0; i < kIterations; ++i) {
        auto result =     separator.Execute(gray);
        ASSERT_TRUE(result.success);
        ASSERT_NE(result.d_laserMask, nullptr);
        ASSERT_NE(result.d_markingPointMask, nullptr);
        ASSERT_NE(result.d_combinedMask, nullptr);

        avgTimings.upload_ms += result.timings.upload_ms;
        avgTimings.step1_gaussian_threshold_ms += result.timings.step1_gaussian_threshold_ms;
        avgTimings.step2_remove_small_noise_ms += result.timings.step2_remove_small_noise_ms;
        avgTimings.step3_remove_large_noise_ms += result.timings.step3_remove_large_noise_ms;
        avgTimings.step4_extract_laser_ms += result.timings.step4_extract_laser_ms;
        avgTimings.step5_extract_marking_ms += result.timings.step5_extract_marking_ms;
        avgTimings.step6_dilate_marking_ms += result.timings.step6_dilate_marking_ms;
        avgTimings.total_pipeline_ms += result.timings.total_pipeline_ms;

        if (i == 0) {
            std::cout << "\n=== First run (warm) ===" << std::endl;
            std::cout << "  Image size: 200x200" << std::endl;
        }
    }

    avgTimings.upload_ms /= kIterations;
    avgTimings.step1_gaussian_threshold_ms /= kIterations;
    avgTimings.step2_remove_small_noise_ms /= kIterations;
    avgTimings.step3_remove_large_noise_ms /= kIterations;
    avgTimings.step4_extract_laser_ms /= kIterations;
    avgTimings.step5_extract_marking_ms /= kIterations;
    avgTimings.step6_dilate_marking_ms /= kIterations;
    avgTimings.total_pipeline_ms /= kIterations;

    std::cout << "\n=== Average timings over " << kIterations
              << " iterations (image 200x200) ===" << std::endl;
    std::cout << "  Step 0  Host→Device upload:   "
              << avgTimings.upload_ms << " ms" << std::endl;
    std::cout << "  Step 1  Gaussian + Threshold: "
              << avgTimings.step1_gaussian_threshold_ms << " ms" << std::endl;
    std::cout << "  Step 2  Remove small noise:   "
              << avgTimings.step2_remove_small_noise_ms << " ms" << std::endl;
    std::cout << "  Step 3  Remove large noise:   "
              << avgTimings.step3_remove_large_noise_ms << " ms" << std::endl;
    std::cout << "  Step 4  Extract laser:        "
              << avgTimings.step4_extract_laser_ms << " ms" << std::endl;
    std::cout << "  Step 5  Extract marking:      "
              << avgTimings.step5_extract_marking_ms << " ms" << std::endl;
    std::cout << "  Step 6  Dilate marking:       "
              << avgTimings.step6_dilate_marking_ms << " ms" << std::endl;
    std::cout << "  ------------------------------------------" << std::endl;
    std::cout << "  TOTAL (GPU pipeline):         "
              << avgTimings.total_pipeline_ms << " ms" << std::endl;
    std::cout << std::endl;
}

TEST_F(LaserMarkingSepTest, TimingFullResolution) {
    LaserMarkingSeparationCUDA separator(params_);
    constexpr int kRows = 1080;
    constexpr int kCols = 1920;
    separator.Warmup(kRows, kCols);

    cv::Mat gray = cv::Mat::zeros(kRows, kCols, CV_8UC1);

    // 宽激光线 (宽度 30px)
    cv::rectangle(gray, cv::Point(50, 525), cv::Point(1870, 555),
                  cv::Scalar(200), -1);

    // 小标记点 (直径 ~12px)
    cv::circle(gray, cv::Point(200, 200), 6, cv::Scalar(200), -1);
    cv::circle(gray, cv::Point(500, 200), 6, cv::Scalar(200), -1);
    cv::circle(gray, cv::Point(800, 200), 6, cv::Scalar(200), -1);
    cv::circle(gray, cv::Point(1100, 200), 6, cv::Scalar(200), -1);
    cv::circle(gray, cv::Point(1400, 200), 6, cv::Scalar(200), -1);
    cv::circle(gray, cv::Point(1700, 200), 6, cv::Scalar(200), -1);

    // 添加高斯噪声
    cv::Mat noise(gray.size(), CV_8UC1);
    cv::randn(noise, 0, 15);
    cv::add(gray, noise, gray, cv::noArray(), CV_8UC1);

    constexpr int kIterations = 20;
    MaskSeparationTimings avgTimings{};

    for (int i = 0; i < kIterations; ++i) {
        auto result =     separator.Execute(gray);
        ASSERT_TRUE(result.success);

        avgTimings.upload_ms += result.timings.upload_ms;
        avgTimings.step1_gaussian_threshold_ms += result.timings.step1_gaussian_threshold_ms;
        avgTimings.step2_remove_small_noise_ms += result.timings.step2_remove_small_noise_ms;
        avgTimings.step3_remove_large_noise_ms += result.timings.step3_remove_large_noise_ms;
        avgTimings.step4_extract_laser_ms += result.timings.step4_extract_laser_ms;
        avgTimings.step5_extract_marking_ms += result.timings.step5_extract_marking_ms;
        avgTimings.step6_dilate_marking_ms += result.timings.step6_dilate_marking_ms;
        avgTimings.total_pipeline_ms += result.timings.total_pipeline_ms;
    }

    avgTimings.upload_ms /= kIterations;
    avgTimings.step1_gaussian_threshold_ms /= kIterations;
    avgTimings.step2_remove_small_noise_ms /= kIterations;
    avgTimings.step3_remove_large_noise_ms /= kIterations;
    avgTimings.step4_extract_laser_ms /= kIterations;
    avgTimings.step5_extract_marking_ms /= kIterations;
    avgTimings.step6_dilate_marking_ms /= kIterations;
    avgTimings.total_pipeline_ms /= kIterations;

    std::cout << "\n=== Average timings over " << kIterations
              << " iterations (image " << kRows << "x" << kCols << ") ===" << std::endl;
    std::cout << "  Step 0  Host→Device upload:   "
              << avgTimings.upload_ms << " ms" << std::endl;
    std::cout << "  Step 1  Gaussian + Threshold: "
              << avgTimings.step1_gaussian_threshold_ms << " ms" << std::endl;
    std::cout << "  Step 2  Remove small noise:   "
              << avgTimings.step2_remove_small_noise_ms << " ms" << std::endl;
    std::cout << "  Step 3  Remove large noise:   "
              << avgTimings.step3_remove_large_noise_ms << " ms" << std::endl;
    std::cout << "  Step 4  Extract laser:        "
              << avgTimings.step4_extract_laser_ms << " ms" << std::endl;
    std::cout << "  Step 5  Extract marking:      "
              << avgTimings.step5_extract_marking_ms << " ms" << std::endl;
    std::cout << "  Step 6  Dilate marking:       "
              << avgTimings.step6_dilate_marking_ms << " ms" << std::endl;
    std::cout << "  ------------------------------------------" << std::endl;
    std::cout << "  TOTAL (GPU pipeline):         "
              << avgTimings.total_pipeline_ms << " ms" << std::endl;
    std::cout << std::endl;
}

TEST_F(LaserMarkingSepTest, SeparateEmptyImageReturnsError) {
    LaserMarkingSeparationCUDA separator(params_);
    cv::Mat empty;
    auto result =     separator.Execute(empty);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(LaserMarkingSepTest, SeparateWrongTypeReturnsError) {
    LaserMarkingSeparationCUDA separator(params_);
    cv::Mat color(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result =     separator.Execute(color);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(LaserMarkingSepTest, WarmupWithConfig) {
    LaserMarkingSeparationCUDA separator(params_);
    auto config = calib::WarmupConfig::forImage(100, 100);
    EXPECT_NO_THROW(separator.Warmup(config));
}

TEST_F(LaserMarkingSepTest, SetParamsAndGetParams) {
    LaserMarkingSeparationCUDA separator(params_);

    LaserMarkingSeparationParams newParams;
    newParams.threshold = 120;
    newParams.step2_erodeSize = 5;
    newParams.step3_erodeSize = 7;
    newParams.step3_dilateSize = 9;
    newParams.step4_erodeSize = 7;
    newParams.step4_dilateSize = 11;
    newParams.step6_dilateSize = 7;

    separator.SetParams(newParams);
    const auto& current =     separator.GetParams();
    EXPECT_EQ(current.threshold, 120);
    EXPECT_EQ(current.step2_erodeSize, 5);
    EXPECT_EQ(current.step3_erodeSize, 7);
}

TEST_F(LaserMarkingSepTest, SetInvalidParamsThrows) {
    LaserMarkingSeparationCUDA separator(params_);
    LaserMarkingSeparationParams badParams;
    badParams.threshold = 999;
    EXPECT_THROW(separator.SetParams(badParams), std::invalid_argument);
}
#endif // WITH_CUDA_TESTS

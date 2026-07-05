/**
 * @file test_marker_match_cpu.cpp
 * @brief 标记点匹配算子 - 单元测试
 *
 * 测试覆盖：
 * 参数校验 / JSON序列化 / Result结构体 / 空输入 / 完美匹配 /
 * 含歧义点 / 偏移容差 / 预计算参考点 / setParams/getParams / warmup
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "../marker_match_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;


namespace {

void makePerfectPair(int n, float disparity,
                     std::vector<cv::Point2f>& left,
                     std::vector<cv::Point2f>& right) {
    left.clear();
    right.clear();
    left.reserve(n);
    right.reserve(n);
    for (int i = 0; i < n; ++i) {
        float y = 50.0f + static_cast<float>(i) * 10.0f;
        float xl = 200.0f + static_cast<float>(i) * 5.0f;
        left.emplace_back(xl, y);
        right.emplace_back(xl - disparity, y);
    }
}

void makeNoisyPair(int n, float disparity, float yNoise,
                   std::vector<cv::Point2f>& left,
                   std::vector<cv::Point2f>& right) {
    left.clear();
    right.clear();
    left.reserve(n);
    right.reserve(n);
    cv::RNG rng(42);
    for (int i = 0; i < n; ++i) {
        float y = 50.0f + static_cast<float>(i) * 10.0f;
        float xl = 200.0f + static_cast<float>(i) * 5.0f;
        float yn = static_cast<float>(rng.gaussian(yNoise));
        left.emplace_back(xl, y + yn);
        right.emplace_back(xl - disparity, y + yn);
    }
}

} // anonymous namespace

// ============================================================
// 参数校验
// ============================================================
class MarkerMatchCPUParamTest : public ::testing::Test {
protected:
    MarkerMatchCPUParams defaultParams_;
};

TEST_F(MarkerMatchCPUParamTest, DefaultParamsPassValidate) {
    EXPECT_NO_THROW(defaultParams_.validate());
}

TEST_F(MarkerMatchCPUParamTest, ZeroYToleranceThrows) {
    auto p = defaultParams_; p.y_tolerance = 0.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerMatchCPUParamTest, YToleranceAboveOneThrows) {
    auto p = defaultParams_; p.y_tolerance = 1.1f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerMatchCPUParamTest, NegativeNumThreadsThrows) {
    auto p = defaultParams_; p.num_threads = -1;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerMatchCPUParamTest, ZeroPreallocBufferSizeThrows) {
    auto p = defaultParams_; p.prealloc_buffer_size = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerMatchCPUParamTest, ZeroMaxPointsThrows) {
    auto p = defaultParams_; p.max_points = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerMatchCPUParamTest, MaxBufferSmallerThanMaxPointsThrows) {
    auto p = defaultParams_; p.max_buffer_size = 10; p.max_points = 100;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

// ============================================================
// JSON 序列化
// ============================================================
TEST_F(MarkerMatchCPUParamTest, JsonRoundtrip) {
    auto j = defaultParams_.toJson();
    auto restored = MarkerMatchCPUParams::fromJson(j);
    EXPECT_NO_THROW(restored.validate());
    EXPECT_FLOAT_EQ(restored.y_tolerance, defaultParams_.y_tolerance);
    EXPECT_EQ(restored.enable_parallel, defaultParams_.enable_parallel);
    EXPECT_EQ(restored.num_threads, defaultParams_.num_threads);
    EXPECT_EQ(restored.max_points, defaultParams_.max_points);
}

TEST_F(MarkerMatchCPUParamTest, JsonEmptyObjectGivesDefaults) {
    auto restored = MarkerMatchCPUParams::fromJson(nlohmann::json{});
    EXPECT_FLOAT_EQ(restored.y_tolerance, 0.15f);
    EXPECT_FALSE(restored.enable_parallel);
}

TEST_F(MarkerMatchCPUParamTest, JsonUnknownFieldsIgnored) {
    auto j = defaultParams_.toJson();
    j["unknownField"] = 42;
    auto restored = MarkerMatchCPUParams::fromJson(j);
    EXPECT_FLOAT_EQ(restored.y_tolerance, defaultParams_.y_tolerance);
}

// ============================================================
// Result 结构体
// ============================================================
TEST_F(MarkerMatchCPUParamTest, ResultDefaultValues) {
    MarkerMatchCPUResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_TRUE(result.disparities.empty());
    EXPECT_TRUE(result.valid_flags.empty());
    EXPECT_TRUE(result.confidence.empty());
}

TEST_F(MarkerMatchCPUParamTest, ResultMoveSemantics) {
    MarkerMatchCPUResult r1;
    r1.success = true;
    r1.disparities = {1.0f, 2.0f};
    r1.valid_flags = {1, 0};
    MarkerMatchCPUResult r2 = std::move(r1);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(r2.disparities.size(), 2u);
}

// ============================================================
// 构造 / warmup
// ============================================================
TEST_F(MarkerMatchCPUParamTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW(MarkerMatchCPU op(defaultParams_));
}

TEST_F(MarkerMatchCPUParamTest, WarmupWithMaxPointCount) {
    MarkerMatchCPU op(defaultParams_);
    EXPECT_NO_THROW(op.Warmup(500));
}

TEST_F(MarkerMatchCPUParamTest, WarmupWithConfig) {
    MarkerMatchCPU op(defaultParams_);
    EXPECT_NO_THROW(op.Warmup(calib::WarmupConfig::forPointCloud(500)));
}

// ============================================================
// 匹配功能
// ============================================================
class MarkerMatchCPUFitTest : public ::testing::Test {
protected:
    MarkerMatchCPUParams params_;
};

TEST_F(MarkerMatchCPUFitTest, EmptyLeftThrows) {
    MarkerMatchCPU op(params_);
    EXPECT_THROW(op.Execute({}, {cv::Point2f(100, 50)}), std::invalid_argument);
}

TEST_F(MarkerMatchCPUFitTest, EmptyRightThrows) {
    MarkerMatchCPU op(params_);
    EXPECT_THROW(op.Execute({cv::Point2f(100, 50)}, {}), std::invalid_argument);
}

TEST_F(MarkerMatchCPUFitTest, NaNInputThrows) {
    MarkerMatchCPU op(params_);
    EXPECT_THROW(op.Execute({cv::Point2f(std::nanf(""), 50)}, {cv::Point2f(100, 50)}),
                 std::invalid_argument);
}

TEST_F(MarkerMatchCPUFitTest, PerfectMatchAllMatched) {
    std::vector<cv::Point2f> left, right;
    makePerfectPair(20, 15.0f, left, right);

    MarkerMatchCPU op(params_);
    auto result = op.Execute(left, right);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.statistics.matched_points, 20u);
    EXPECT_EQ(result.statistics.total_points, 20u);
    EXPECT_FLOAT_EQ(result.statistics.match_rate, 1.0f);

    for (size_t i = 0; i < result.disparities.size(); ++i) {
        EXPECT_TRUE(result.valid_flags[i]) << "Point " << i << " should be valid";
        EXPECT_NEAR(result.disparities[i], 15.0f, 0.01f) << "Point " << i;
    }
}

TEST_F(MarkerMatchCPUFitTest, NoisyMatch) {
    std::vector<cv::Point2f> left, right;
    makeNoisyPair(30, 12.0f, 0.05f, left, right);

    MarkerMatchCPU op(params_);
    auto result = op.Execute(left, right);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.statistics.matched_points, 25u);

    for (size_t i = 0; i < result.disparities.size(); ++i) {
        if (result.valid_flags[i]) {
            EXPECT_NEAR(result.disparities[i], 12.0f, 1.0f) << "Point " << i;
        }
    }
}

TEST_F(MarkerMatchCPUFitTest, AmbiguousPoints) {
    std::vector<cv::Point2f> left = {
        cv::Point2f(200.0f, 100.0f),
        cv::Point2f(210.0f, 100.0f),
        cv::Point2f(220.0f, 200.0f)
    };
    std::vector<cv::Point2f> right = {
        cv::Point2f(185.0f, 100.0f),
        cv::Point2f(195.0f, 100.0f),
        cv::Point2f(205.0f, 200.0f)
    };

    MarkerMatchCPU op(params_);
    auto result = op.Execute(left, right);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.statistics.ambiguous_points, 0u);
}

TEST_F(MarkerMatchCPUFitTest, SinglePointPair) {
    std::vector<cv::Point2f> left = {cv::Point2f(200.0f, 100.0f)};
    std::vector<cv::Point2f> right = {cv::Point2f(180.0f, 100.0f)};

    MarkerMatchCPU op(params_);
    auto result = op.Execute(left, right);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.statistics.matched_points, 1u);
    EXPECT_NEAR(result.disparities[0], 20.0f, 0.01f);
    EXPECT_TRUE(result.valid_flags[0]);
    EXPECT_GT(result.confidence[0], 0.5f);
}

TEST_F(MarkerMatchCPUFitTest, NoMatchDifferentY) {
    std::vector<cv::Point2f> left = {cv::Point2f(200.0f, 100.0f)};
    std::vector<cv::Point2f> right = {cv::Point2f(180.0f, 200.0f)};

    MarkerMatchCPU op(params_);
    auto result = op.Execute(left, right);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.statistics.matched_points, 0u);
    EXPECT_FALSE(result.valid_flags[0]);
}

TEST_F(MarkerMatchCPUFitTest, ConfidenceDecreasesWithYDistance) {
    std::vector<cv::Point2f> left1 = {cv::Point2f(200.0f, 100.0f)};
    std::vector<cv::Point2f> right1 = {cv::Point2f(185.0f, 100.0f)};

    std::vector<cv::Point2f> left2 = {cv::Point2f(200.0f, 100.0f)};
    std::vector<cv::Point2f> right2 = {cv::Point2f(185.0f, 100.10f)};

    MarkerMatchCPU op(params_);

    auto r1 = op.Execute(left1, right1);
    auto r2 = op.Execute(left2, right2);

    EXPECT_GT(r1.confidence[0], r2.confidence[0]);
}

// ============================================================
// 预计算参考点
// ============================================================
class MarkerMatchCPUReferenceTest : public ::testing::Test {
protected:
    MarkerMatchCPUParams params_;
};

TEST_F(MarkerMatchCPUReferenceTest, SetAndClearReference) {
    MarkerMatchCPU op(params_);
    EXPECT_FALSE(op.HasReferencePoints());

    std::vector<cv::Point2f> right = {cv::Point2f(180.0f, 100.0f)};
    op.SetReferencePoints(right);
    EXPECT_TRUE(op.HasReferencePoints());

    op.ClearReferencePoints();
    EXPECT_FALSE(op.HasReferencePoints());
}

TEST_F(MarkerMatchCPUReferenceTest, MatchWithReference) {
    std::vector<cv::Point2f> left, right;
    makePerfectPair(15, 10.0f, left, right);

    MarkerMatchCPU op(params_);
    op.SetReferencePoints(right);

    auto result = op.MatchWithReference(left);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.statistics.matched_points, 15u);
    EXPECT_TRUE(result.statistics.used_precomputed);
}

TEST_F(MarkerMatchCPUReferenceTest, MatchWithReferenceWithoutSetThrows) {
    MarkerMatchCPU op(params_);
    EXPECT_THROW(op.MatchWithReference({cv::Point2f(100, 50)}), std::invalid_argument);
}

TEST_F(MarkerMatchCPUReferenceTest, EmptyReferenceThrows) {
    MarkerMatchCPU op(params_);
    EXPECT_THROW(op.SetReferencePoints({}), std::invalid_argument);
}

TEST_F(MarkerMatchCPUReferenceTest, ReuseReferenceMultipleTimes) {
    std::vector<cv::Point2f> right = {
        cv::Point2f(180.0f, 100.0f),
        cv::Point2f(170.0f, 200.0f)
    };

    MarkerMatchCPU op(params_);
    op.SetReferencePoints(right);

    for (int i = 0; i < 5; ++i) {
        std::vector<cv::Point2f> left = {
            cv::Point2f(200.0f, 100.0f),
            cv::Point2f(190.0f, 200.0f)
        };
        MarkerMatchCPUResult result;
        result = op.MatchWithReference(left);
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.statistics.matched_points, 2u);
    }
}

// ============================================================
// setParams / getParams
// ============================================================
class MarkerMatchCPUSetParamsTest : public ::testing::Test {
protected:
    MarkerMatchCPUParams params_;
};

TEST_F(MarkerMatchCPUSetParamsTest, GetParamsReflectsConstructor) {
    MarkerMatchCPU op(params_);
    EXPECT_FLOAT_EQ(op.GetParams().y_tolerance, params_.y_tolerance);
    EXPECT_EQ(op.GetParams().max_points, params_.max_points);
}

TEST_F(MarkerMatchCPUSetParamsTest, SetParamsUpdates) {
    MarkerMatchCPU op(params_);
    MarkerMatchCPUParams np;
    np.y_tolerance = 0.08f;
    np.max_points = 200;
    op.SetParams(np);
    EXPECT_FLOAT_EQ(op.GetParams().y_tolerance, 0.08f);
    EXPECT_EQ(op.GetParams().max_points, 200u);
}

TEST_F(MarkerMatchCPUSetParamsTest, SetParamsInvalidThrows) {
    MarkerMatchCPU op(params_);
    MarkerMatchCPUParams bad;
    bad.y_tolerance = 0.0f;
    EXPECT_THROW(op.SetParams(bad), std::invalid_argument);
}

// ============================================================
// 统计
// ============================================================
TEST_F(MarkerMatchCPUFitTest, StatisticsAreCollected) {
    std::vector<cv::Point2f> left, right;
    makePerfectPair(10, 20.0f, left, right);

    MarkerMatchCPU op(params_);
    auto result = op.Execute(left, right);

    EXPECT_GT(result.statistics.total_time_ms, 0.0);
    EXPECT_GE(result.statistics.sort_time_ms, 0.0);
    EXPECT_GE(result.statistics.match_time_ms, 0.0);
    EXPECT_FLOAT_EQ(result.statistics.match_rate, 1.0f);
    EXPECT_NEAR(result.statistics.avg_disparity, 20.0f, 0.01f);
    EXPECT_GT(result.statistics.avg_confidence, 0.0f);
}

TEST_F(MarkerMatchCPUFitTest, ResetStatistics) {
    MarkerMatchCPU op(params_);

    std::vector<cv::Point2f> left, right;
    makePerfectPair(5, 10.0f, left, right);
    auto result = op.Execute(left, right);

    op.ResetStatistics();
    const auto& stats = op.GetStatistics();
    EXPECT_DOUBLE_EQ(stats.total_time_ms, 0.0);
    EXPECT_EQ(stats.matched_points, 0u);
}

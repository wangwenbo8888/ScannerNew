/**
 * @file test_laser_markingpoint_mask_separation_cuda.cpp
 * @brief 激光线与标记点掩膜分离算子 - 单元测试
 *
 * 测试覆盖：
 * - 参数校验（默认值/阈值范围/奇偶/单调递增/上限 63、markerKill≤61/核形状）
 * - JSON 序列化/反序列化（往返/部分反序列化/旧 step 键忽略容错）
 * - Result 结构体默认值/移动语义
 * - 构造/析构（GPU）
 * - warmup 预热（GPU）
 * - 宽度分档落位：激光条 w=5→激光掩膜 / 标记点 φ35→标记点掩膜 / 分档噪点（GPU）
 * - separate 空图像/错误类型输入（GPU）
 * - setParams 动态更新 / getParams 参数查询（GPU）
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
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
// 参数校验测试（CPU）
// ============================================================
TEST_F(LaserMarkingSepTest, DefaultParamsAreValid) {
    EXPECT_NO_THROW(params_.validate());
    EXPECT_EQ(params_.threshold, 80);
    EXPECT_EQ(params_.noiseErodeSize, 3);
    EXPECT_EQ(params_.laserKillSize, 9);
    EXPECT_EQ(params_.markerKillSize, 37);
    EXPECT_EQ(params_.laserEdgeDilateSize, 9);
    EXPECT_EQ(params_.markingEdgeDilateSize, 7);
    EXPECT_EQ(params_.kernelShape, 1);
    EXPECT_FALSE(params_.dumpAllSteps);
}

TEST_F(LaserMarkingSepTest, InvalidThresholdThrows) {
    params_.threshold = -1;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
    params_ = LaserMarkingSeparationParams{};
    params_.threshold = 256;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
    params_ = LaserMarkingSeparationParams{};
    params_.threshold = 0;             // 合法边界
    EXPECT_NO_THROW(params_.validate());
    params_.threshold = 255;           // 合法边界
    EXPECT_NO_THROW(params_.validate());
}

TEST_F(LaserMarkingSepTest, EvenKernelSizeThrows) {
    for (auto* f : {&params_.noiseErodeSize, &params_.laserKillSize,
                    &params_.markerKillSize, &params_.laserEdgeDilateSize,
                    &params_.markingEdgeDilateSize}) {
        params_ = LaserMarkingSeparationParams{};
        *f = 4;
        EXPECT_THROW(params_.validate(), std::invalid_argument);
    }
}

TEST_F(LaserMarkingSepTest, KillSizesMustIncrease) {
    params_.markerKillSize = 9;       // == laserKillSize
    EXPECT_THROW(params_.validate(), std::invalid_argument);
    params_ = LaserMarkingSeparationParams{};
    params_.markerKillSize = 5;       // 反向：markerKill < laserKill
    EXPECT_THROW(params_.validate(), std::invalid_argument);
    // laserKill == / < noiseErode 合法（下限抬升编码：②=7/③=5＝原始带宽 [7,11)）
    params_ = LaserMarkingSeparationParams{};
    params_.noiseErodeSize = 7;
    params_.laserKillSize = 5;
    EXPECT_NO_THROW(params_.validate());
}

TEST_F(LaserMarkingSepTest, KernelSizeAboveLimitThrows) {
    // 上限 61：laserKill/markerKill 的内部还原核(+2) 不得超 CUDA 模板上限 63
    params_.markerKillSize = 63;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
    params_ = LaserMarkingSeparationParams{};
    params_.laserKillSize = 63;       // laserKill 同理（还原 65 超模板上限）
    EXPECT_THROW(params_.validate(), std::invalid_argument);
    params_ = LaserMarkingSeparationParams{};
    params_.markerKillSize = 61;
    EXPECT_NO_THROW(params_.validate());
    params_.laserEdgeDilateSize = 65;   // 膨胀核本身上限 63
    EXPECT_THROW(params_.validate(), std::invalid_argument);
}

TEST_F(LaserMarkingSepTest, InvalidKernelShapeThrows) {
    params_.kernelShape = 2;
    EXPECT_THROW(params_.validate(), std::invalid_argument);
    params_.kernelShape = 1;
    EXPECT_NO_THROW(params_.validate());
}

// ============================================================
// JSON 序列化测试（CPU）
// ============================================================
TEST_F(LaserMarkingSepTest, JsonRoundtrip) {
    params_.threshold = 120;
    params_.noiseErodeSize = 5;
    params_.laserKillSize = 11;
    params_.markerKillSize = 41;
    params_.laserEdgeDilateSize = 11;
    params_.markingEdgeDilateSize = 9;
    params_.kernelShape = 1;
    params_.dumpAllSteps = true;
    auto j = params_.toJson();
    auto restored = LaserMarkingSeparationParams::fromJson(j);
    EXPECT_EQ(restored.threshold, params_.threshold);
    EXPECT_EQ(restored.noiseErodeSize, params_.noiseErodeSize);
    EXPECT_EQ(restored.laserKillSize, params_.laserKillSize);
    EXPECT_EQ(restored.markerKillSize, params_.markerKillSize);
    EXPECT_EQ(restored.laserEdgeDilateSize, params_.laserEdgeDilateSize);
    EXPECT_EQ(restored.markingEdgeDilateSize, params_.markingEdgeDilateSize);
    EXPECT_EQ(restored.kernelShape, params_.kernelShape);
    EXPECT_EQ(restored.dumpAllSteps, params_.dumpAllSteps);
}

TEST_F(LaserMarkingSepTest, JsonPartialDeserialization) {
    nlohmann::json j = {{"threshold", 120}};
    auto restored = LaserMarkingSeparationParams::fromJson(j);
    EXPECT_EQ(restored.threshold, 120);
    EXPECT_EQ(restored.markerKillSize, 37);
}

TEST_F(LaserMarkingSepTest, JsonOldStepKeysIgnored) {
    // 旧 step2/3/4/6 键被安全忽略（config 迁移期容错）
    nlohmann::json j = {{"step2_erodeSize", 5}, {"swapOutputs", true},
                        {"gaussianSize", 7}};
    auto restored = LaserMarkingSeparationParams::fromJson(j);
    EXPECT_EQ(restored.noiseErodeSize, 3);
    EXPECT_EQ(restored.threshold, 80);   // 未被旧键污染
}

// ============================================================
// Result 结构体测试（CPU）
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
// 宽度分档落位（核心用例）：合成图 640x480
//   激光条 w=5 ×2 ｜ 标记点圆 φ35 ×2 ｜ 小噪点 φ2 ×3 ｜ 中噪点 φ15 ×1 ｜ 大噪点 φ50 ×1
namespace {
cv::Mat makeBandScene() {
    cv::Mat g = cv::Mat::zeros(480, 640, CV_8UC1);
    // 激光条（竖直，w=5——水平游程量宽）
    cv::rectangle(g, cv::Point(150, 80), cv::Point(154, 440), cv::Scalar(200), -1);
    cv::rectangle(g, cv::Point(450, 80), cv::Point(454, 440), cv::Scalar(200), -1);
    // 标记点 φ35
    cv::circle(g, cv::Point(120, 400), 17, cv::Scalar(200), -1);
    cv::circle(g, cv::Point(520, 400), 17, cv::Scalar(200), -1);
    // 中噪点 φ15（设计语义：落标记点掩膜，靠下游 ccl 面积闸过滤）
    cv::circle(g, cv::Point(320, 200), 7, cv::Scalar(200), -1);
    // 大噪点 φ50（设计语义：丢弃）
    cv::circle(g, cv::Point(320, 60), 25, cv::Scalar(200), -1);
    // 小噪点 φ2（设计语义：全灭）
    cv::circle(g, cv::Point(200, 200), 1, cv::Scalar(200), -1);
    cv::circle(g, cv::Point(240, 210), 1, cv::Scalar(200), -1);
    cv::circle(g, cv::Point(280, 190), 1, cv::Scalar(200), -1);
    return g;
}
int runLengthAt(const cv::Mat& m, cv::Point p) {   // 水平游程长（p 处）
    if (m.at<uint8_t>(p) == 0) return 0;
    int l = p.x, r = p.x;
    while (l > 0 && m.at<uint8_t>(p.y, l - 1)) --l;
    while (r < m.cols - 1 && m.at<uint8_t>(p.y, r + 1)) ++r;
    return r - l + 1;
}
} // namespace

TEST_F(LaserMarkingSepTest, WidthBandSeparation) {
    LaserMarkingSeparationCUDA sep(params_);
    sep.Warmup(480, 640);
    auto r = sep.Execute(makeBandScene());
    ASSERT_TRUE(r.success) << r.message;

    cv::Mat laser, marking;
    r.d_laserMask->download(laser);
    r.d_markingPointMask->download(marking);

    // 激光条进激光掩膜（膨胀后水平游程 ≈ 5+6=11，容差 ±2）
    EXPECT_GT(laser.at<uint8_t>(102, 152), 0);
    EXPECT_NEAR(runLengthAt(laser, {152, 102}), 11, 2);
    // 标记点进标记点掩膜（膨胀后直径 ≈ 35+6=41，容差 ±3）
    EXPECT_GT(marking.at<uint8_t>(400, 120), 0);
    EXPECT_NEAR(runLengthAt(marking, {120, 400}), 41, 3);
    // 交叉不污染
    EXPECT_EQ(marking.at<uint8_t>(102, 152), 0);   // 激光条不在标记点掩膜
    EXPECT_EQ(laser.at<uint8_t>(400, 120), 0);     // 标记点不在激光掩膜（中心）
    EXPECT_EQ(laser.at<uint8_t>(60, 320), 0);      // 大噪点不在激光掩膜（中心）
    EXPECT_EQ(marking.at<uint8_t>(60, 320), 0);    // 大噪点不在标记点掩膜（中心）
    // 中噪点 φ15 落标记点掩膜（设计语义）
    EXPECT_GT(marking.at<uint8_t>(200, 320), 0);
    // 小噪点全灭
    EXPECT_EQ(laser.at<uint8_t>(200, 200), 0);
    EXPECT_EQ(marking.at<uint8_t>(200, 200), 0);
    // 主输出齐全
    EXPECT_NE(r.d_combinedMask, nullptr);
    // 二值图常驻（无需 dumpAllSteps——harness S0 产物消费）
    EXPECT_NE(r.d_step1_binary, nullptr);
}

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

// 模板上限覆盖：61 杀核（还原 63）＋63 膨胀核全实例化
TEST_F(LaserMarkingSepTest, LargeKernelsUpToLimit) {
    LaserMarkingSeparationParams p;
    p.markerKillSize = 61;          // 还原核 63 恰达模板上限
    p.markingEdgeDilateSize = 63;
    p.kernelShape = 1;              // 圆形核路径
    p.dumpAllSteps = true;
    LaserMarkingSeparationCUDA sep(p);
    sep.Warmup(240, 240);
    cv::Mat gray = cv::Mat::zeros(240, 240, CV_8UC1);
    cv::circle(gray, cv::Point(120, 120), 45, cv::Scalar(200), -1);  // φ90 大结构
    cv::rectangle(gray, cv::Point(20, 200), cv::Point(220, 204), cv::Scalar(200), -1);
    auto r = sep.Execute(gray);
    EXPECT_TRUE(r.success) << r.message;
    EXPECT_NE(r.d_step1_binary, nullptr);
    EXPECT_NE(r.d_structuresMask, nullptr);
    EXPECT_NE(r.d_laserRaw, nullptr);
    EXPECT_NE(r.d_survivorMask, nullptr);
    EXPECT_NE(r.d_markingRaw, nullptr);
}

// 圆形核（kernelShape=1）：磁盘窗口实跑。φ35 标记点两种核形均应落入标记点掩膜
TEST_F(LaserMarkingSepTest, CircleKernelSeparate) {
    cv::Mat gray = cv::Mat::zeros(200, 200, CV_8UC1);
    cv::circle(gray, cv::Point(100, 100), 17, cv::Scalar(200), -1);  // φ35 标记点
    for (int shape : {0, 1}) {
        LaserMarkingSeparationParams p;
        p.kernelShape = shape;
        LaserMarkingSeparationCUDA sep(p);
        sep.Warmup(200, 200);
        auto r = sep.Execute(gray);
        ASSERT_TRUE(r.success) << r.message;
        cv::Mat marking;
        r.d_markingPointMask->download(marking);
        EXPECT_GT(marking.at<uint8_t>(100, 100), 0);
    }
}

TEST_F(LaserMarkingSepTest, WarmupAndTimings) {
    LaserMarkingSeparationCUDA sep(params_);
    sep.Warmup(480, 640);
    cv::Mat gray = makeBandScene();
    for (int i = 0; i < 5; ++i) {
        auto r = sep.Execute(gray);
        ASSERT_TRUE(r.success);
        // 261024 合并适配：ops 版默认走 CUDA Graph 路径（P0 优化——一次性提交
        // 整图内核链），recordEvents=false → timings 恒 0 是预期；仅
        // JMW_SEPTIME=1 诊断模式/Graph 失败回退 legacy 路径才有分段计时。
        // 断言改为条件式：有计时才校验（沙盒口径同 JMW_SEPTIME 环境下验证）
        if (r.timings.upload_ms > 0.0) {
            EXPECT_GE(r.timings.total_pipeline_ms, 0.0);
            EXPECT_GT(r.timings.step3_laser_extract_ms, 0.0);
            EXPECT_GT(r.timings.step4_marking_extract_ms, 0.0);
        }
    }
}

TEST_F(LaserMarkingSepTest, SeparateEmptyImageReturnsError) {
    LaserMarkingSeparationCUDA sep(params_);
    auto r = sep.Execute(cv::Mat());
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.message.empty());
}

TEST_F(LaserMarkingSepTest, SeparateWrongTypeReturnsError) {
    LaserMarkingSeparationCUDA sep(params_);
    cv::Mat color(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    auto r = sep.Execute(color);
    EXPECT_FALSE(r.success);
}

TEST_F(LaserMarkingSepTest, WarmupWithConfig) {
    LaserMarkingSeparationCUDA sep(params_);
    EXPECT_NO_THROW(sep.Warmup(calib::WarmupConfig::forImage(100, 100)));
}

TEST_F(LaserMarkingSepTest, SetParamsAndGetParams) {
    LaserMarkingSeparationCUDA sep(params_);
    LaserMarkingSeparationParams np;
    np.threshold = 120;
    np.noiseErodeSize = 5;
    np.laserKillSize = 11;
    np.markerKillSize = 41;
    sep.SetParams(np);
    EXPECT_EQ(sep.GetParams().threshold, 120);
    EXPECT_EQ(sep.GetParams().markerKillSize, 41);
}

TEST_F(LaserMarkingSepTest, SetInvalidParamsThrows) {
    LaserMarkingSeparationCUDA sep(params_);
    LaserMarkingSeparationParams bad;
    bad.markerKillSize = 5;   // <= laserKillSize
    EXPECT_THROW(sep.SetParams(bad), std::invalid_argument);
}
#endif // WITH_CUDA_TESTS

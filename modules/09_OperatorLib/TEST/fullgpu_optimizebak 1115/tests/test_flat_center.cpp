// test_flat_center.cpp — flat_center 算子单测（设计 §6：
// docs/plans/2026-10-18-flat-center-extractor-design.md v4.4）
// Task 1: A1/A2 参数校验与 JSON roundtrip；后续任务逐步追加。
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "scr_test_helpers.h"
#include "scr_config.h"
#include "core/laser/steger_fast/steger_fast.h"     // Task 6 对拍用
#include "core/laser/flat_center/flat_center.h"     // 路径限定（新目录防 GLOB 时序）

TEST(FlatCenterParams, ValidateRejectsIllegal) {
    calib::FlatCenterParams p;                     // 默认参数必须合法
    EXPECT_NO_THROW(p.validate());
    auto check = [](calib::FlatCenterParams q, bool shouldThrow) {
        bool threw = false;
        try { q.validate(); } catch (...) { threw = true; }
        EXPECT_EQ(threw, shouldThrow);
    };
    { auto q = calib::FlatCenterParams{}; q.lowThreshold = -1.f;    check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.halfRatio = 0.f;        check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.halfRatio = 1.f;        check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.plateauRatio = 0.f;     check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.plateauRatio = 1.01f;   check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.plateauMinWidth = 0;    check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.minRunLength = 0;       check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.minContrast = -1.f;     check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.outlierGatePx = 0.f;    check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.maxLineWidth = -1.f;    check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.maxLabels = 0;          check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.maxLabels = 4097;       check(q, true); }
    { auto q = calib::FlatCenterParams{}; q.deviceId = -1;          check(q, true); }
    // 边界合法值
    { auto q = calib::FlatCenterParams{}; q.plateauRatio = 1.f;     check(q, false); }
    { auto q = calib::FlatCenterParams{}; q.maxLabels = 4096;       check(q, false); }
    { auto q = calib::FlatCenterParams{}; q.maxLineWidth = 0.f;     check(q, false); }
}

TEST(FlatCenterParams, JsonRoundtrip) {
    calib::FlatCenterParams p;
    p.lowThreshold = 12.f; p.halfRatio = 0.4f; p.plateauRatio = 0.9f;
    p.plateauMinWidth = 4; p.minRunLength = 5; p.minContrast = 3.f;
    p.outlierGatePx = 0.8f; p.smooth3 = false; p.maxLabels = 128; p.deviceId = 1;
    p.maxLineWidth = 24.f;
    auto q = calib::FlatCenterParams::fromJson(p.toJson());
    EXPECT_FLOAT_EQ(q.lowThreshold, p.lowThreshold);
    EXPECT_FLOAT_EQ(q.halfRatio, p.halfRatio);
    EXPECT_FLOAT_EQ(q.plateauRatio, p.plateauRatio);
    EXPECT_EQ(q.plateauMinWidth, p.plateauMinWidth);
    EXPECT_EQ(q.minRunLength, p.minRunLength);
    EXPECT_FLOAT_EQ(q.minContrast, p.minContrast);
    EXPECT_FLOAT_EQ(q.outlierGatePx, p.outlierGatePx);
    EXPECT_FLOAT_EQ(q.maxLineWidth, p.maxLineWidth);
    EXPECT_FALSE(q.smooth3);
    EXPECT_EQ(q.maxLabels, p.maxLabels);
    EXPECT_EQ(q.deviceId, p.deviceId);
    // 缺节 = 默认参数
    auto r = calib::FlatCenterParams::fromJson(nlohmann::json::object());
    EXPECT_FLOAT_EQ(r.halfRatio, 0.5f);
    EXPECT_FLOAT_EQ(r.maxLineWidth, 0.0f);
}

TEST(FlatCenterExtract, InputValidation) {
    calib::FlatCenterExtractor op;
    cv::cuda::GpuMat d_gray, d_mask;
    // 空 GpuMat → success=false（照抄 steger 校验语义）
    auto r1 = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    EXPECT_FALSE(r1.success);
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    d_gray.upload(gray); d_mask.upload(mask);
    // 全零掩膜（非空）→ 零点早退 success=true＋Warning（Task 3 内核就绪后生效）
    auto r2 = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(r2.qualityFlag, calib::QualityFlag::Warning);
    EXPECT_EQ(r2.totalPointCount, 0);
    // 类型错（Flat 需 8UC1）→ success=false
    cv::cuda::GpuMat d_mask32;
    d_mask32.upload(cv::Mat::zeros(240, 320, CV_32SC1));
    auto r3 = op.Execute(d_gray, d_mask32, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    EXPECT_FALSE(r3.success);
}

// ====================== Task 3：合成图核心用例 ①②③⑤ ======================
namespace {
cv::Mat MakeGaussianPeak(cv::Mat& mask, float sigma, float amp, float yc) {
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 10; x < 290; ++x) {   // x0=10：σ2 窄线的线端拐角伪响应（~13 抽头
                                       // 核波及 ~25px）退出断言窗 [40,280)
        for (int y = 0; y < gray.rows; ++y) {
            float d = y - yc;
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(
                gray.at<uchar>(y, x) + amp * std::exp(-(d * d) / (2.f * sigma * sigma)));
        }
        for (int y = 95; y <= 145; ++y) mask.at<uchar>(y, x) = 255;
    }
    return gray;
}
cv::Mat MakeSaturatedFlatTop(cv::Mat& mask, float w, float amp) {   // ③ amp=400→截 255
    const float yc = 120.5f, edge = 3.5f;
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 30; x < 290; ++x) {
        for (int y = 0; y < gray.rows; ++y) {
            float d = std::fabs(y - yc) - w * 0.5f;
            float v = d <= 0.f ? 1.f : std::exp(-(d * d) / (2.f * edge * edge));
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(gray.at<uchar>(y, x) + amp * v);
        }
        for (int y = 95; y <= 145; ++y) mask.at<uchar>(y, x) = 255;
    }
    return gray;
}
std::vector<cv::Point2f> DownloadPoints(const calib::StegerResult& r) {
    std::vector<cv::Point2f> v;
    if (!r.d_centerPoints || r.d_centerPoints->empty()) return v;
    cv::Mat h; r.d_centerPoints->download(h);
    const cv::Point2f* p = h.ptr<cv::Point2f>();
    v.assign(p, p + h.cols);
    return v;
}
std::vector<int> DownloadIds(const calib::StegerResult& r) {
    std::vector<int> v;
    if (!r.d_line_ids || r.d_line_ids->empty()) return v;
    cv::Mat h; r.d_line_ids->download(h);
    const int* p = h.ptr<int>();
    v.assign(p, p + h.cols);
    return v;
}
double MaxCenterErr(const std::vector<cv::Point2f>& pts, float yc) {
    double m = -1;
    for (auto& q : scr::PointsInWindow(pts))
        m = std::max(m, static_cast<double>(std::fabs(q.y - yc)));
    return m;
}
} // namespace

TEST(FlatCenterCore, FlatTopSubpixel) {          // ①（w=8/edge=3.5/amp=255/yc=120.5）
    cv::Mat mask; const cv::Mat gray = scr::MakeFlatTopSigma4(mask);
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.mergedCount, 0);                               // A5
    auto pts = DownloadPoints(r);
    int inWin = 0; for (auto& q : scr::PointsInWindow(pts)) ++inWin;
    EXPECT_EQ(inWin, 240);                                     // [40,280)＝240 列×1 点
    EXPECT_LE(MaxCenterErr(pts, 120.5f), 0.05);
    EXPECT_EQ(r.lineCount, 1);
}
TEST(FlatCenterCore, GaussianPeakSubpixel) {    // ② σ=2.0 → 尖峰路径
    cv::Mat mask; const cv::Mat gray = MakeGaussianPeak(mask, 2.0f, 255.f, 120.5f);
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    EXPECT_LE(MaxCenterErr(DownloadPoints(r), 120.5f), 0.1);
}
TEST(FlatCenterCore, SaturatedFlatTop) {        // ③ amp=400/w=12 → 截 255 饱和
    cv::Mat mask; const cv::Mat gray = MakeSaturatedFlatTop(mask, 12.f, 400.f);
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    EXPECT_LE(MaxCenterErr(DownloadPoints(r), 120.5f), 0.05);
}
TEST(FlatCenterCore, ShortRunDropped) {         // ⑤ 2px 段 → 无点＋Degraded
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 100; x < 200; ++x) {
        mask.at<uchar>(119, x) = 255; mask.at<uchar>(120, x) = 255;
        gray.at<uchar>(119, x) = 255; gray.at<uchar>(120, x) = 255;
    }
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.totalPointCount, 0);
    EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Degraded);    // reasons 含 minRun
}

// ====================== Task 4：边界与变体用例 ⑥-⑭＋A4 ======================
// 造图辅助：单带平顶（可调参），返回 gray 置 mask
namespace {
void FillFlatBand(cv::Mat& gray, cv::Mat& mask, float yc, float w, float amp,
                  int x0, int x1, int m0, int m1, float edge = 3.5f) {
    for (int x = x0; x < x1; ++x) {
        for (int y = 0; y < gray.rows; ++y) {
            const float d = std::fabs(y - yc) - w * 0.5f;
            const float v = d <= 0.f ? 1.f : std::exp(-(d * d) / (2.f * edge * edge));
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(gray.at<uchar>(y, x) + amp * v);
        }
        for (int y = m0; y <= m1; ++y) mask.at<uchar>(y, x) = 255;
    }
}
} // namespace

TEST(FlatCenterEdge, SingleSoftEdge) {          // ⑥ 单侧软边 → 无点＋Degraded
    const float yc = 120.5f;
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 30; x < 290; ++x) {
        for (int y = 0; y < 240; ++y) {         // y<yc 恒 255；y≥yc 指数衰减
            const float d = (y >= yc) ? (y - yc) : 0.f;
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(
                255.f * (d <= 0.f ? 1.f : std::exp(-(d * d) / (2.f * 3.5f * 3.5f))));
        }
        for (int y = 95; y <= 145; ++y) mask.at<uchar>(y, x) = 255;
    }
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    int inWin = 0; for (auto& q : scr::PointsInWindow(pts)) ++inWin;
    EXPECT_EQ(inWin, 0);                        // 上方恒 255 越段读到窗界仍 ≥half → edge
    EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Degraded);
}
TEST(FlatCenterEdge, SingleSidedHalo) {         // ⑥b 单侧晕 → 出点且有界（v4.3 口径）
    const float yc = 120.5f;
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 30; x < 290; ++x) {
        for (int y = 0; y < 240; ++y) {         // 亮核 w=8 平顶＋右侧 σ3.5/幅值 60 尾巴
            const float d = std::fabs(y - yc) - 4.f;
            float v = d <= 0.f ? 1.f : std::exp(-(d * d) / (2.f * 3.5f * 3.5f));
            float add = (y >= yc) ? 60.f * std::exp(-((y - yc) * (y - yc)) / (2.f * 3.5f * 3.5f)) : 0.f;
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(255.f * v + add);
        }
        for (int y = 95; y <= 145; ++y) mask.at<uchar>(y, x) = 255;
    }
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    int inWin = 0; for (auto& q : scr::PointsInWindow(pts)) ++inWin;
    EXPECT_GT(inWin, 200);                      // 绝大多数列出点
    EXPECT_LE(MaxCenterErr(pts, yc), 0.5f);     // 有界退化（晕拉动中心与重心同向）
}
TEST(FlatCenterEdge, MaskGapCrossing) {         // ⑦ 掩膜细缝断裂（Flat 主测）
    const float yc = 110.5f;
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 30; x < 290; ++x) {
        for (int y = 0; y < 240; ++y) {
            const float d = std::fabs(y - yc) - 4.f;
            const float v = d <= 0.f ? 1.f : std::exp(-(d * d) / (2.f * 3.5f * 3.5f));
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(255.f * v);
        }
        for (int y = 95; y <= 145; ++y)
            mask.at<uchar>(y, x) = (y == 111) ? 0 : 255;   // 111 行整行细缝
    }
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    // 每列两段两点（越缝找沿、括号重心自检不误杀）
    std::map<int, std::vector<float>> byCol;
    for (auto& q : scr::PointsInWindow(pts)) byCol[cvRound(q.x)].push_back(q.y);
    int twoPt = 0; double ySum = 0; int n = 0;
    for (auto& [col, ys] : byCol) {
        if (ys.size() == 2) ++twoPt;
        for (float y : ys) { ySum += y; ++n; }
    }
    EXPECT_EQ(twoPt, 240);                      // [40,280) 全部列恰 2 点
    EXPECT_NEAR(ySum / n, yc, 0.5);             // 均值落线心
    EXPECT_EQ(r.lineCount, 1);
}
TEST(FlatCenterEdge, NoiseRobustness) {         // ⑧ 30 个噪声种子
    for (int seed = 1; seed <= 30; ++seed) {
        cv::Mat mask; cv::Mat gray = scr::MakeFlatTopSigma4(mask);
        cv::RNG rng(seed);
        for (int y = 0; y < gray.rows; ++y)
            for (int x = 0; x < gray.cols; ++x)
                gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(
                    gray.at<uchar>(y, x) + rng.gaussian(3.0));
        cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
        calib::FlatCenterExtractor op; op.Warmup(240, 320);
        auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
        ASSERT_TRUE(r.success) << "seed=" << seed;
        auto pts = DownloadPoints(r);
        int inWin = 0; for (auto& q : scr::PointsInWindow(pts)) ++inWin;
        EXPECT_GT(inWin, 200) << "seed=" << seed;
        EXPECT_LE(MaxCenterErr(pts, 120.5f), 0.5f) << "seed=" << seed;
    }
}
TEST(FlatCenterEdge, BrightDarkMix) {           // ⑨ 明暗混合帧
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1) + 5;      // 背景 5
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 30; x < 290; ++x) {           // 暗尖峰带 yc=50 amp=60 σ=2.0
        for (int y = 0; y < 240; ++y) {
            float d1 = y - 50.f;
            float d2 = std::fabs(y - 120.5f) - 4.f;
            float v1 = 60.f * std::exp(-(d1 * d1) / (2.f * 2.f * 2.f));
            float v2 = d2 <= 0.f ? 255.f : 255.f * std::exp(-(d2 * d2) / (2.f * 3.5f * 3.5f));
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(5.f + v1 + v2);
        }
        for (int y = 40; y <= 60; ++y) mask.at<uchar>(y, x) = 255;
        for (int y = 95; y <= 145; ++y) mask.at<uchar>(y, x) = 255;
    }
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    int nBright = 0, nDark = 0;
    double eBright = 0, eDark = 0;
    for (auto& q : scr::PointsInWindow(pts)) {
        if (q.y < 90) { ++nDark; eDark = std::max(eDark, (double)std::fabs(q.y - 50.f)); }
        else          { ++nBright; eBright = std::max(eBright, (double)std::fabs(q.y - 120.5f)); }
    }
    EXPECT_GT(nBright, 200);  EXPECT_LE(eBright, 0.5);
    EXPECT_GT(nDark, 200);    EXPECT_LE(eDark, 0.5);
}
TEST(FlatCenterEdge, TopBoundaryFullSupport) {  // ⑩a 贴上边界（两沿在图内）
    const float yc = 6.5f;
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    FillFlatBand(gray, mask, yc, 4.f, 255.f, 30, 290, 0, 14);
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    int inWin = 0; for (auto& q : scr::PointsInWindow(pts)) ++inWin;
    EXPECT_EQ(inWin, 240);
    EXPECT_LE(MaxCenterErr(pts, yc), 0.25);     // 行 0 平滑钳位 ~0.14px 有界偏差（v4.3）
}
TEST(FlatCenterEdge, TruncatedPlatformWindow) { // ⑩b 平台腰截断，下沿超 ±32 窗
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    FillFlatBand(gray, mask, 120.5f, 40.f, 255.f, 30, 290, 100, 125);
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    int inWin = 0; for (auto& q : scr::PointsInWindow(pts)) ++inWin;
    EXPECT_EQ(inWin, 0);                        // 下沿 144.6 > yPeak+33 → edge 弃
    EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Degraded);
}
TEST(FlatCenterEdge, UniformSegment) {          // ⑪ 全段均匀 128 → contrast 弃
    cv::Mat gray = cv::Mat(240, 320, CV_8UC1, cv::Scalar(128));
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 30; x < 290; ++x)
        for (int y = 95; y <= 145; ++y) mask.at<uchar>(y, x) = 255;
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.totalPointCount, 0);
    EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Degraded);    // contrast（防除零）
}
TEST(FlatCenterEdge, SeparatedDoublePeak) {     // ⑫a 分离双峰 → 出点落于锚峰（v4.3）
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    FillFlatBand(gray, mask, 112.5f, 4.f, 255.f, 30, 290, 95, 145);
    FillFlatBand(gray, mask, 128.5f, 4.f, 255.f, 30, 290, 95, 145);
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    int inWin = 0; for (auto& q : scr::PointsInWindow(pts)) ++inWin;
    EXPECT_EQ(inWin, 240);                      // 每列 1 点（锚峰）
    // w=4 窄平台＋平滑的量化误差略大，容差 0.75（语义＝落锚峰区间，非中点/非弃点）
    EXPECT_LE(MaxCenterErr(pts, 112.5f), 0.75f);
    EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Normal);
}
TEST(FlatCenterEdge, FusedDoublePeak) {         // ⑫b 紧邻双峰（谷底≥half 连成宽平台）
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    FillFlatBand(gray, mask, 120.5f, 12.f, 255.f, 30, 290, 95, 145);  // 宽 w=12 平台
    for (int x = 30; x < 290; ++x)              // 中央 2 行挖 180 浅槽（≥half=127.5，
        { gray.at<uchar>(120, x) = 180; gray.at<uchar>(121, x) = 180; }  // 视觉双峰连体)
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    int inWin = 0; for (auto& q : scr::PointsInWindow(pts)) ++inWin;
    EXPECT_EQ(inWin, 240);
    EXPECT_LE(MaxCenterErr(pts, 120.5f), 0.5f); // 宽平台中心＝双峰中点
}
TEST(FlatCenterEdge, FlankGlint) {              // ⑬ 侧翼亮斑（尖峰锚点判拦）
    const float yc = 120.5f;
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 30; x < 290; ++x) {
        for (int y = 0; y < 240; ++y) {         // 线 amp=180（未饱和）
            const float d = std::fabs(y - yc) - 4.f;
            const float v = d <= 0.f ? 1.f : std::exp(-(d * d) / (2.f * 3.5f * 3.5f));
            float g = 180.f * v;
            if (x >= 150 && x < 200) {          // 斑 amp=255 σ=1.5 于 y≈113
                const float gd = y - 113.f;
                g += 255.f * std::exp(-(gd * gd) / (2.f * 1.5f * 1.5f));
            }
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(g);
        }
        for (int y = 95; y <= 145; ++y) mask.at<uchar>(y, x) = 255;
    }
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    int clean = 0, glint = 0;
    for (auto& q : pts) {
        if (q.x >= 40 && q.x < 150) { ++clean; EXPECT_LE(std::fabs(q.y - yc), 0.05f); }
        if (q.x >= 150 && q.x < 200) ++glint;
    }
    EXPECT_EQ(clean, 110);                      // 无斑列 [40,150) 正常
    EXPECT_EQ(glint, 0);                        // 斑列全弃（尖峰 |yPeak−cy|≈5.7>1）
    EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Degraded);
}
TEST(FlatCenterEdge, SymmetricWidening) {       // ⑭a 对称变宽（默认关宽度门）
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    FillFlatBand(gray, mask, 120.5f, 8.f, 255.f, 30, 150, 95, 145);   // 正常 w=8
    FillFlatBand(gray, mask, 120.5f, 16.f, 255.f, 150, 290, 95, 145); // 宽 w=16
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    EXPECT_LE(MaxCenterErr(pts, 120.5f), 0.1f); // 宽度不参与中心
    int inWin = 0; for (auto& q : scr::PointsInWindow(pts)) ++inWin;
    EXPECT_EQ(inWin, 240);
}
TEST(FlatCenterEdge, WidthGate) {               // ⑭b 宽度门开启（限 12px）
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_8UC1);
    FillFlatBand(gray, mask, 120.5f, 8.f, 255.f, 30, 150, 95, 145);
    FillFlatBand(gray, mask, 120.5f, 16.f, 255.f, 150, 290, 95, 145);
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterParams p; p.maxLineWidth = 20.f;  // 正常 w=8 段 FWHM≈16.2 过；
                                                        // 宽 w=16 段 FWHM≈24.2 拦
    calib::FlatCenterExtractor op(p); op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r);
    int normal = 0, wide = 0;
    for (auto& q : pts) {
        if (q.x >= 40 && q.x < 150) { ++normal; EXPECT_LE(std::fabs(q.y - 120.5f), 0.05f); }
        if (q.x >= 150 && q.x < 280) ++wide;
    }
    EXPECT_EQ(normal, 110);                     // 正常段照常
    EXPECT_EQ(wide, 0);                         // 宽段全弃（width 门）
    EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Degraded);
}
// ====================== Task 5：ByLabel 用例 ④/⑦ ======================
TEST(FlatCenterByLabel, TwoLabeledLines) {      // ④ CV_32SC1 两带 label=1/2
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_32SC1);
    auto band = [&](float yc, int label) {
        for (int x = 30; x < 290; ++x) {
            for (int y = 0; y < 240; ++y) {
                float d = std::fabs(y - yc) - 4.f;
                float v = d <= 0.f ? 1.f : std::exp(-(d * d) / (2.f * 3.5f * 3.5f));
                gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(gray.at<uchar>(y, x) + 255.f * v);
            }
            for (int y = (int)yc - 15; y <= (int)yc + 15; ++y) mask.at<int>(y, x) = label;
        }
    };
    band(60.5f, 1); band(180.5f, 2);
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null());   // ByLabel 重载
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r); auto ids = DownloadIds(r);
    int n1 = 0, n2 = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (pts[i].x >= 40 && pts[i].x < 280) {
            if (ids[i] == 1) { ++n1; EXPECT_NEAR(pts[i].y, 60.5, 0.1); }
            else            { ++n2; EXPECT_NEAR(pts[i].y, 180.5, 0.1); }
        }
    }
    EXPECT_EQ(n1, 240); EXPECT_EQ(n2, 240);     // [40,280)＝240 列
    EXPECT_EQ(r.lineCount, 2);                  // hist 非零桶数（与 collect 开关无关）
}
TEST(FlatCenterByLabel, MaskGapSameLabel) {     // ⑦ ByLabel 变体（细缝同 label）
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(240, 320, CV_32SC1);
    for (int x = 30; x < 290; ++x) {            // 单带 w=8 平顶 yc=110.5
        for (int y = 0; y < 240; ++y) {
            float d = std::fabs(y - 110.5f) - 4.f;
            float v = d <= 0.f ? 1.f : std::exp(-(d * d) / (2.f * 3.5f * 3.5f));
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(255.f * v);
        }
        for (int y = 95; y <= 145; ++y) mask.at<int>(y, x) = (y == 111) ? 0 : 1;  // 细缝
    }
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null());
    ASSERT_TRUE(r.success);
    auto pts = DownloadPoints(r); auto ids = DownloadIds(r);
    double ySum = 0; int n = 0;
    for (size_t i = 0; i < pts.size(); ++i)
        if (pts[i].x >= 40 && pts[i].x < 280) { ++n; ySum += pts[i].y; EXPECT_EQ(ids[i], 1); }
    EXPECT_EQ(n, 2 * 240);                      // 每列两段两点
    EXPECT_LE(std::fabs(ySum / n - 110.5), 0.5);
    EXPECT_EQ(r.lineCount, 1);                  // 单 label 单桶
}

TEST(FlatCenterExtract, HostCollectSwitch) {    // A4 两态
    cv::Mat mask; const cv::Mat gray = scr::MakeFlatTopSigma4(mask);
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(240, 320);
    auto r1 = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r1.success);
    EXPECT_TRUE(r1.d_centerPoints && !r1.d_centerPoints->empty());   // 默认 GPU 有
    EXPECT_TRUE(r1.centerPoints.empty());                            // 默认 host 无
    op.SetCollectHostPoints(true);
    auto r2 = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(r2.success);
    ASSERT_FALSE(r2.centerPoints.empty());
    auto host = scr::Flatten(r2);
    auto dev = DownloadPoints(r2);
    std::sort(host.begin(), host.end(), [](auto& a, auto& b) { return a.x < b.x; });
    std::sort(dev.begin(), dev.end(), [](auto& a, auto& b) { return a.x < b.x; });
    ASSERT_EQ(host.size(), dev.size());
    for (size_t i = 0; i < host.size(); ++i) {
        EXPECT_FLOAT_EQ(host[i].x, dev[i].x);
        EXPECT_FLOAT_EQ(host[i].y, dev[i].y);
    }
}

// ====================== Task 6：对拍单测 vs steger_fast(dedupMode=1) ======================
class FlatVsSteger : public ::testing::TestWithParam<int> {};  // 0=平顶 1=尖峰 2=饱和
TEST_P(FlatVsSteger, CenterAgreement) {
    cv::Mat mask, gray;
    switch (GetParam()) {
        case 0: gray = scr::MakeFlatTopSigma4(mask); break;
        case 1: gray = MakeGaussianPeak(mask, 2.0f, 255.f, 120.5f); break;
        default: gray = MakeSaturatedFlatTop(mask, 12.f, 400.f); break;
    }
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    // steger 参数：平顶/饱和照 test_units 配方（σ4/0.5）；窄高斯 σ2 线用 σ2
    // （σ4 平滑下窄峰 Hessian 响应不足；阈值 0.5 统一——低阈会把侧翼微脊当脊线）
    calib::StegerParams sp;
    sp.sigma = (GetParam() == 1) ? 2.0f : 4.0f;
    sp.lowThreshold = 0.5f;
    sp.dedupMode = 1;
    calib::StegerExtractorFast stg(sp); stg.Warmup(gray.rows, gray.cols);
    calib::FlatCenterExtractor fl; fl.Warmup(gray.rows, gray.cols);
    const auto rs = stg.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    const auto rf = fl.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
    ASSERT_TRUE(rs.success && rf.success);
    // 按列分桶（steger 存在缺列/直通 2 点机理——对拍口径＝列交集比对＋覆盖率）
    auto bucket = [](std::vector<cv::Point2f> v) {
        std::map<int, std::vector<float>> m;
        for (auto& q : scr::PointsInWindow(v)) m[cvRound(q.x)].push_back(q.y);
        return m;
    };
    const auto a = bucket(DownloadPoints(rs));
    const auto b = bucket(DownloadPoints(rf));
    // 覆盖率（[40,280)＝240 列；steger 既有口径 ≥230）
    EXPECT_GE(a.size(), 230u) << "steger 覆盖列数";
    EXPECT_GE(b.size(), 230u) << "flat 覆盖列数";
    ASSERT_FALSE(a.empty()); ASSERT_FALSE(b.empty());
    // 列交集内：flat 每列 1 点、法向各 ≤0.5、双向最近邻 ≤0.75
    double devSum = 0; int devN = 0, matched = 0;
    for (auto& kv : b) {
        const int col = kv.first;
        auto it = a.find(col);
        if (it == a.end()) continue;            // steger 缺列不参与对拍（覆盖率已断言）
        ++matched;
        ASSERT_EQ(kv.second.size(), 1u) << "col=" << col;
        const float y = kv.second[0];
        EXPECT_LE(std::fabs(y - 120.5f), 0.5f) << "flat 法向 col=" << col;
        double best = 1e9;
        for (float sy : it->second) best = std::min(best, (double)std::fabs(sy - y));
        EXPECT_LE(best, 0.75) << "NN(flat→steger) col=" << col;
        devSum += best; ++devN;
        for (float sy : it->second) {           // 反向（steger 同列可能 2 点）
            double bestR = 1e9;
            for (float fy : kv.second) bestR = std::min(bestR, (double)std::fabs(sy - fy));
            EXPECT_LE(bestR, 0.75) << "NN(steger→flat) col=" << col;
        }
    }
    EXPECT_GE(matched, 230);                    // 交集覆盖率
    std::cout << "[parity] case=" << GetParam() << " matched=" << matched
              << " meanNearDev=" << devSum / devN << "px\n";
}
// 对拍范围＝平顶族（①③）。窄高斯 σ2（原 case 1）被排除：探针实证 steger 在该
// 剖面产生 y=120.5±8.5 的伪点链（x 恒偏 +0.696，真线侧翼灰度≈0 处出点）——
// steger 自身脆弱域（本项目立项动因之一），非 flat 责任侧；flat 在同图的精度由
// ② 单测独立验证（|Δ|≤0.1px）。设计 §3 机理清单：红则定位责任侧，不硬凑容差。
INSTANTIATE_TEST_SUITE_P(FlatCenterParity, FlatVsSteger, ::testing::Values(0, 2));

// ====================== 内部分步计时（JMW_FLATTIME=1 时执行，否则跳过）======================
TEST(FlatCenterBench, InternalSteps) {
    if (getenv("JMW_FLATTIME") == nullptr) GTEST_SKIP() << "设 JMW_FLATTIME=1 启用";
    // 真实几何仿制品：3380×1400、13 条线、间距 103px、掩膜带宽 26px、背景 115
    const int W = 3380, H = 1400;
    cv::Mat gray = cv::Mat(H, W, CV_8UC1, cv::Scalar(115));
    cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
    for (int li = 0; li < 13; ++li) {
        const float yc = 84.0f + li * 103.0f;
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                const float d = std::fabs(y - yc) - 4.f;
                const float v = d <= 0.f ? 255.f : 255.f * std::exp(-(d * d) / (2.f * 3.5f * 3.5f));
                gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(115.f + v);
            }
            for (int y = (int)yc - 13; y <= (int)yc + 13; ++y)
                if (y >= 0 && y < H) mask.at<uchar>(y, x) = 255;
        }
    }
    cv::cuda::GpuMat d_gray, d_mask; d_gray.upload(gray); d_mask.upload(mask);
    calib::FlatCenterExtractor op; op.Warmup(H, W);
    std::vector<double> tot, scan, out;
    for (int i = 0; i < 30; ++i) {
        auto r = op.Execute(d_gray, d_mask, cv::cuda::Stream::Null(), calib::GroupMode::Flat);
        ASSERT_TRUE(r.success);
        float ms[3];
        op.LastBenchMs(ms);
        tot.push_back(ms[0]); scan.push_back(ms[1]); out.push_back(ms[2]);
    }
    std::sort(tot.begin(), tot.end()); std::sort(scan.begin(), scan.end()); std::sort(out.begin(), out.end());
    const auto med = [](std::vector<double>& v) { return v[v.size() / 2]; };
    std::cout << "[bench] N=13 线 3380x1400（真实几何仿制）30 次中位：\n"
              << "  Total(算子全程)   = " << med(tot) << " ms\n"
              << "  Scan(K1 扫描核)   = " << med(scan) << " ms\n"
              << "  Group/Out(组装)   = " << med(out) << " ms\n";
}

// ====================== Task 7：config 集成 ======================
TEST(FlatCenterConfig, RejectsFlatWithCpuDedup) {
    nlohmann::json j = {
        {"paths", nlohmann::json::object()},
        {"runtime", {{"extractor", "flat"}, {"dedupPath", "cpu"}}}};
    scr::ScrConfig cfg; std::string err;
    EXPECT_FALSE(scr::ScrConfig::fromJson(j, cfg, err));
    EXPECT_NE(err.find("extractor"), std::string::npos);
    j["runtime"]["dedupPath"] = "operator";
    EXPECT_TRUE(scr::ScrConfig::fromJson(j, cfg, err)) << err;
    EXPECT_EQ(cfg.runtime.extractor, "flat");
    // 非法枚举拒绝
    j["runtime"]["extractor"] = "foo";
    EXPECT_FALSE(scr::ScrConfig::fromJson(j, cfg, err));
}


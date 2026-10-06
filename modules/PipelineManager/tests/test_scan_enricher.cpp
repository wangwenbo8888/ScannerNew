#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "FrameEnricher.h"
#include "TempTableTypes.h"
#include "EnhancedFrame.h"
using Scanner::data::CalibSnapshot;
using Scanner::data::EnhancedFrame;
using Scanner::data::PlaneMapTempTableRef;
using Scanner::data::PlaneMapTempTierRef;
using Scanner::data::StereoTempTable;
using Scanner::data::StereoTempTier;
using Scanner::data::enrich;

namespace {

// 立体表：tempC 从 from 起 step 间隔 n 档；R/P/Q 首元素 = 100+i*100 作档身份标记
StereoTempTable mkStereo(double from, double step, int n) {
    StereoTempTable t;
    for (int i = 0; i < n; ++i) {
        StereoTempTier tier;
        tier.tempC = from + i * step;
        tier.R1 = cv::Matx33d::eye(); tier.R1(0, 0) = 100.0 + i * 100;
        tier.R2 = cv::Matx33d::eye(); tier.R2(0, 0) = 200.0 + i * 100;
        tier.P1 = cv::Matx34d::zeros(); tier.P1(0, 0) = 300.0 + i * 100;
        tier.P2 = cv::Matx34d::zeros(); tier.P2(0, 0) = 400.0 + i * 100;
        tier.Q = cv::Matx44d::zeros(); tier.Q(0, 0) = 500.0 + i * 100;
        t.tiers.push_back(tier);
    }
    return t;
}

PlaneMapTempTableRef mkLaser(double from, double step, int n) {
    PlaneMapTempTableRef t;
    for (int i = 0; i < n; ++i)
        t.tiers.push_back(PlaneMapTempTierRef{from + i * step});
    return t;
}

cv::Mat mkGray(uint8_t v) { return cv::Mat(4, 6, CV_8UC1, cv::Scalar(v)); }

} // namespace

// 语义 1：档内选最近档——[25.0,25.2,25.4]，25.09→档0；25.10→并列 0.10 取低温档 0；
//         25.11→档1；25.4 精确命中→档2；范围内结果 quality Normal
TEST(FrameEnricher, TierSelectionMid) {
    auto stereo = mkStereo(25.0, 0.2, 3);
    auto laser = mkLaser(25.0, 0.2, 3);
    EnhancedFrame f;

    auto r = enrich(mkGray(1), mkGray(2), 25.09, stereo, laser, 7, f);
    EXPECT_TRUE(r.success);
    EXPECT_FALSE(r.hasWarning());
    EXPECT_EQ(f.snapshot.stereoTier, 0);
    EXPECT_EQ(f.snapshot.laserTier, 0);

    enrich(mkGray(1), mkGray(2), 25.10, stereo, laser, 8, f);
    EXPECT_EQ(f.snapshot.stereoTier, 0);   // |25.10-25.0|=|25.2-25.10|=0.10 并列→低温档
    EXPECT_EQ(f.snapshot.laserTier, 0);

    enrich(mkGray(1), mkGray(2), 25.11, stereo, laser, 9, f);
    EXPECT_EQ(f.snapshot.stereoTier, 1);
    EXPECT_EQ(f.snapshot.laserTier, 1);

    enrich(mkGray(1), mkGray(2), 25.4, stereo, laser, 10, f);
    EXPECT_EQ(f.snapshot.stereoTier, 2);
    EXPECT_EQ(f.snapshot.laserTier, 2);
}

// 语义 2：帧温越出表范围——clamp 首档/末档 + success + Warning
TEST(FrameEnricher, ClampOutOfRange) {
    auto stereo = mkStereo(25.0, 0.2, 3);
    auto laser = mkLaser(25.0, 0.2, 3);
    EnhancedFrame f;

    auto rLow = enrich(mkGray(1), mkGray(2), 20.0, stereo, laser, 1, f);
    EXPECT_TRUE(rLow.success);
    EXPECT_TRUE(rLow.hasWarning());
    EXPECT_EQ(f.snapshot.stereoTier, 0);
    EXPECT_EQ(f.snapshot.laserTier, 0);

    auto rHigh = enrich(mkGray(1), mkGray(2), 30.0, stereo, laser, 2, f);
    EXPECT_TRUE(rHigh.success);
    EXPECT_TRUE(rHigh.hasWarning());
    EXPECT_EQ(f.snapshot.stereoTier, 2);
    EXPECT_EQ(f.snapshot.laserTier, 2);
}

// 语义 3：两表全空 → fail
TEST(FrameEnricher, EmptyTableFails) {
    StereoTempTable stereo;
    PlaneMapTempTableRef laser;
    EnhancedFrame f;
    auto r = enrich(mkGray(1), mkGray(2), 25.0, stereo, laser, 1, f);
    EXPECT_FALSE(r.success);
    EXPECT_TRUE(r.isFault());
}

// 语义 4：单表降级——laser 表空 → success 但 laserTier=-1 + Warning
TEST(FrameEnricher, OneTableOnly) {
    auto stereo = mkStereo(25.0, 0.2, 3);
    PlaneMapTempTableRef laser;               // 激光表空
    EnhancedFrame f;
    auto r = enrich(mkGray(1), mkGray(2), 25.11, stereo, laser, 3, f);
    EXPECT_TRUE(r.success);
    EXPECT_TRUE(r.hasWarning());
    EXPECT_EQ(f.snapshot.stereoTier, 1);
    EXPECT_EQ(f.snapshot.laserTier, -1);
}

// 语义 5：字段拷贝——snapshot 与所选档一致、frameId/temperature 保留、gray 深拷贝、
//         device 指针不碰（保持 nullptr）
TEST(FrameEnricher, FieldsCopied) {
    auto stereo = mkStereo(25.0, 0.2, 3);
    auto laser = mkLaser(25.0, 0.2, 3);
    cv::Mat gl = mkGray(10), gr = mkGray(20);
    EnhancedFrame f;

    auto r = enrich(gl, gr, 25.2, stereo, laser, 42, f);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(f.frameId, 42u);
    EXPECT_DOUBLE_EQ(f.temperature, 25.2);

    EXPECT_DOUBLE_EQ(f.snapshot.R1(0, 0), 200.0);   // 档 1 身份标记
    EXPECT_DOUBLE_EQ(f.snapshot.R2(0, 0), 300.0);
    EXPECT_DOUBLE_EQ(f.snapshot.P1(0, 0), 400.0);
    EXPECT_DOUBLE_EQ(f.snapshot.P2(0, 0), 500.0);
    EXPECT_DOUBLE_EQ(f.snapshot.Q(0, 0), 600.0);

    ASSERT_FALSE(f.grayL.empty());
    ASSERT_FALSE(f.grayR.empty());
    EXPECT_NE(f.grayL.data, gl.data);               // clone 深拷贝
    EXPECT_NE(f.grayR.data, gr.data);
    EXPECT_EQ(f.grayL.at<uint8_t>(0, 0), 10);
    EXPECT_EQ(f.grayR.at<uint8_t>(0, 0), 20);
    gl.setTo(255);                                  // 改源不影响副本
    EXPECT_EQ(f.grayL.at<uint8_t>(0, 0), 10);

    EXPECT_EQ(f.d_grayL, nullptr);                  // 06 不碰 CUDA
    EXPECT_EQ(f.d_grayR, nullptr);
}

// 语义 6：乱序传入——实现内升序排序后选档（防御），档身份标记 = tempC*10
TEST(FrameEnricher, SortedAssumption) {
    StereoTempTable stereo;
    const double temps[] = {25.4, 25.0, 25.2};      // 乱序传入
    for (double tc : temps) {
        StereoTempTier tier;
        tier.tempC = tc;
        tier.R1 = cv::Matx33d::eye(); tier.R1(0, 0) = tc * 10;   // 250/252/254
        stereo.tiers.push_back(tier);
    }
    auto laser = mkLaser(25.0, 0.2, 3);
    EnhancedFrame f;

    auto r = enrich(mkGray(1), mkGray(2), 25.09, stereo, laser, 1, f);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(f.snapshot.stereoTier, 0);            // 排序后 25.0 为档 0
    EXPECT_DOUBLE_EQ(f.snapshot.R1(0, 0), 250.0);   // 命中 25.0 档数据

    enrich(mkGray(1), mkGray(2), 25.30, stereo, laser, 2, f);
    EXPECT_EQ(f.snapshot.stereoTier, 1);            // 排序后 25.2/25.4 并列 0.10→低温档
    EXPECT_DOUBLE_EQ(f.snapshot.R1(0, 0), 252.0);
}

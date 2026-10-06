// ============================================================================
// test_calib_camera.cpp — P4-T21 B 相机链 TDD 测试（3-1~3-5 + 5-1/5-2 编排）
// ============================================================================
// 1 ChainRecoversTruth   ：合成数据收敛回真值（fx/外参/Q/reprojError）
// 2 PromiseSetAfterRectify：run 返回后 promise 已兑现且与 outStereo 一致
// 3 CancelMidway         ：前置 cancel → fail 返回、outStereo 未写、promise 未兑现
// 4 ProgressMonotonic    ：percent 单调不减且 ≤50（相机半区）
// 5 TempTablesGenerated  ：5-1/5-2/3-5 三表非空（档数=温度阶梯数 101）
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <future>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "calib_synthetic.h"
#include "pipelines/calibcompute/CameraChain.h"
#include "pipelines/calibcompute/CalibComputeTypes.h"
#include "pipelines/posture/PostureTypes.h"

using Scanner::pipeline::CalibComputeOutput;
using Scanner::pipeline::CameraChain;
using Scanner::pipeline::CancelToken;
using Scanner::pipeline::InitialCalibParams;
using Scanner::pipeline::PostureSessionData;
using Scanner::pipeline::StereoParams;
using Scanner::pipeline::synthetic::SyntheticTruth;

namespace {

constexpr int kExpectedTempEntries = 101;   // ±10℃ / 0.2 步距 → 101 档

// 期望档数 = floor((max-min)/step)+1（对齐算子容差循环端点含闭）
int expectedEntries(double rangeMin, double rangeMax, double step) {
    return static_cast<int>(std::floor((rangeMax - rangeMin) / step + 1e-9)) + 1;
}

} // namespace

// —— 1. 合成数据 → 收敛回真值 ——
TEST(CameraChainTest, ChainRecoversTruth) {
    SyntheticTruth truth;
    PostureSessionData session = Scanner::pipeline::synthetic::makeSyntheticSession(25, truth);
    InitialCalibParams init = Scanner::pipeline::synthetic::makeInitialFromTruth(truth);

    CameraChain chain;
    StereoParams stereo;
    std::promise<StereoParams> toLaser;
    CalibComputeOutput out;
    auto res = chain.run(session, init, truth.boardPoints3D, stereo, toLaser, out,
                         nullptr, CancelToken{});
    ASSERT_TRUE(res.success) << res.message;

    // 内参：|Δfx| < 1.0 px
    const double fx = stereo.cameraMatrixL.at<double>(0, 0);
    const double fxTruth = truth.K1.at<double>(0, 0);
    EXPECT_NEAR(fx, fxTruth, 1.0);
    EXPECT_NEAR(stereo.cameraMatrixL.at<double>(1, 1), truth.K1.at<double>(1, 1), 1.0);
    EXPECT_NEAR(stereo.cameraMatrixL.at<double>(0, 2), truth.K1.at<double>(0, 2), 2.0);
    EXPECT_NEAR(stereo.cameraMatrixL.at<double>(1, 2), truth.K1.at<double>(1, 2), 2.0);

    // 外参：R 角差 < 0.1°，T 平移差 < 0.5mm（物方 mm 单位）
    cv::Mat rErr;
    cv::Rodrigues(stereo.R * truth.R.t(), rErr);
    const double angleDeg = cv::norm(rErr) * 180.0 / CV_PI;
    EXPECT_LT(angleDeg, 0.1) << "R angle diff " << angleDeg << " deg";
    const double transMm = cv::norm(stereo.T - truth.T);
    EXPECT_LT(transMm, 0.5) << "T diff " << transMm << " mm";

    // Q 有效（非零；(3,2) 为 -1/Tx 驱动项）
    ASSERT_EQ(stereo.Q.rows, 4);
    ASSERT_EQ(stereo.Q.cols, 4);
    EXPECT_GT(std::fabs(stereo.Q.at<double>(3, 2)), 1e-9);

    // 重投影误差 < 0.5 px
    EXPECT_LT(stereo.reprojError, 0.5) << "reprojError=" << stereo.reprojError;
}

// —— 2. promise 兑现：run 返回后 future 已 ready 且值与 outStereo 一致 ——
TEST(CameraChainTest, PromiseSetAfterRectify) {
    SyntheticTruth truth;
    PostureSessionData session = Scanner::pipeline::synthetic::makeSyntheticSession(25, truth);
    InitialCalibParams init = Scanner::pipeline::synthetic::makeInitialFromTruth(truth);

    CameraChain chain;
    StereoParams stereo;
    std::promise<StereoParams> toLaser;
    auto fut = toLaser.get_future();
    CalibComputeOutput out;
    ASSERT_TRUE(chain.run(session, init, truth.boardPoints3D, stereo, toLaser, out,
                          nullptr, CancelToken{}).success);

    EXPECT_EQ(fut.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    StereoParams fromPromise = fut.get();
    EXPECT_NEAR(fromPromise.cameraMatrixL.at<double>(0, 0),
                stereo.cameraMatrixL.at<double>(0, 0), 1e-12);
    EXPECT_LT(cv::norm(fromPromise.T - stereo.T), 1e-12);
    EXPECT_LT(cv::norm(fromPromise.Q - stereo.Q, cv::NORM_INF), 1e-12);
}

// —— 3. 前置 cancel：fail 返回、outStereo 未写、promise 未兑现 ——
TEST(CameraChainTest, CancelMidway) {
    SyntheticTruth truth;
    PostureSessionData session = Scanner::pipeline::synthetic::makeSyntheticSession(25, truth);
    InitialCalibParams init = Scanner::pipeline::synthetic::makeInitialFromTruth(truth);

    CameraChain chain;
    StereoParams stereo;
    std::promise<StereoParams> toLaser;
    auto fut = toLaser.get_future();
    CalibComputeOutput out;
    CancelToken cancel;
    cancel.cancel();                              // run 前置取消（算子间取消点粒度）

    auto res = chain.run(session, init, truth.boardPoints3D, stereo, toLaser, out,
                         nullptr, cancel);
    EXPECT_FALSE(res.success);
    EXPECT_TRUE(stereo.cameraMatrixL.empty());    // outStereo 未写（安全态）
    EXPECT_FALSE(out.quality.ok);
    EXPECT_EQ(fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
}

// —— 3b. 3-4 兑现后取消：不构成失败——run 跑完 3-5 返回 ok、promise/outStereo 有效 ——
TEST(CameraChainTest, CancelAfterStereoDefined) {
    SyntheticTruth truth;
    PostureSessionData session = Scanner::pipeline::synthetic::makeSyntheticSession(25, truth);
    InitialCalibParams init = Scanner::pipeline::synthetic::makeInitialFromTruth(truth);

    CancelToken cancel;
    Scanner::pipeline::ProgressCb cb = [&](int percent, const std::string&) {
        if (percent >= 42) cancel.cancel();       // 42% = promise 已兑现、3-5 起点
    };

    CameraChain chain;
    StereoParams stereo;
    std::promise<StereoParams> toLaser;
    auto fut = toLaser.get_future();
    CalibComputeOutput out;
    auto res = chain.run(session, init, truth.boardPoints3D, stereo, toLaser, out,
                         cb, cancel);

    EXPECT_TRUE(res.success) << res.message;      // 3-5 纯 CPU 且短：跑完返回 ok 最干净
    EXPECT_FALSE(stereo.cameraMatrixL.empty());   // outStereo 有效
    EXPECT_EQ(fut.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    EXPECT_TRUE(out.rectifyTempTable.success);    // 3-5 表仍生成
    EXPECT_EQ(static_cast<int>(out.rectifyTempTable.table.size()), kExpectedTempEntries);
    EXPECT_TRUE(out.quality.ok);
}

// —— 4. 进度：percent 单调不减、≤50（相机半区） ——
TEST(CameraChainTest, ProgressMonotonic) {
    SyntheticTruth truth;
    PostureSessionData session = Scanner::pipeline::synthetic::makeSyntheticSession(25, truth);
    InitialCalibParams init = Scanner::pipeline::synthetic::makeInitialFromTruth(truth);

    std::vector<int> pcts;
    std::vector<std::string> stages;
    Scanner::pipeline::ProgressCb cb = [&](int p, const std::string& s) {
        pcts.push_back(p); stages.push_back(s);
    };

    CameraChain chain;
    StereoParams stereo;
    std::promise<StereoParams> toLaser;
    CalibComputeOutput out;
    ASSERT_TRUE(chain.run(session, init, truth.boardPoints3D, stereo, toLaser, out,
                          cb, CancelToken{}).success);

    ASSERT_GE(pcts.size(), 3u) << "至少 3 次进度回调";
    for (size_t i = 0; i < pcts.size(); ++i) {
        EXPECT_LE(pcts[i], 50) << "idx=" << i << " stage=" << stages[i];
        if (i > 0) EXPECT_GE(pcts[i], pcts[i - 1]) << "idx=" << i << " stage=" << stages[i];
    }
    EXPECT_EQ(pcts.front(), 0);
}

// —— 5. 三温度表非空：档数 = 温度阶梯数 ——
TEST(CameraChainTest, TempTablesGenerated) {
    SyntheticTruth truth;
    PostureSessionData session = Scanner::pipeline::synthetic::makeSyntheticSession(25, truth);
    InitialCalibParams init = Scanner::pipeline::synthetic::makeInitialFromTruth(truth);

    CameraChain chain;
    StereoParams stereo;
    std::promise<StereoParams> toLaser;
    CalibComputeOutput out;
    ASSERT_TRUE(chain.run(session, init, truth.boardPoints3D, stereo, toLaser, out,
                          nullptr, CancelToken{}).success);

    // 5-1 内参补偿表（±10℃/0.2 → 101 档）
    EXPECT_TRUE(out.intrinsicTempTable.success) << out.intrinsicTempTable.message;
    EXPECT_EQ(static_cast<int>(out.intrinsicTempTable.table.size()),
              expectedEntries(-10.0, 10.0, 0.2));
    EXPECT_EQ(kExpectedTempEntries, static_cast<int>(out.intrinsicTempTable.table.size()));

    // 5-2 外参补偿表
    EXPECT_TRUE(out.extrinsicTempTable.success) << out.extrinsicTempTable.message;
    EXPECT_EQ(static_cast<int>(out.extrinsicTempTable.table.size()),
              expectedEntries(-10.0, 10.0, 0.2));

    // 3-5 矫正温度表（每档含 R1/R2/P1/P2/Q）
    EXPECT_TRUE(out.rectifyTempTable.success) << out.rectifyTempTable.message;
    EXPECT_EQ(out.rectifyTempTable.tableSize, expectedEntries(-10.0, 10.0, 0.2));
    ASSERT_FALSE(out.rectifyTempTable.table.empty());
    const auto& e0 = out.rectifyTempTable.table.front();
    EXPECT_FALSE(e0.R1.empty());
    EXPECT_FALSE(e0.P2.empty());
    EXPECT_FALSE(e0.Q.empty());
}

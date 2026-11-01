// ============================================================================
// test_calib_laser.cpp — P4-T22 B 激光链 + CalibComputePipeline TDD 测试
// ============================================================================
// 两层策略（任务书）：
//   层 1（结构编排，假算子 LaserOps）：
//     1 FrontRunsBeforeStereoReady：future 延迟兑现——前段(4-1..4-4)计数先达 25，
//       后段仅在兑现后运行
//     2 BrokenPromiseFailsFast：promise 破诺（相机链早期失败）→ fail 快速返回 <2s
//     3 ChainOrderAndAggregate：调用序 front×25 → back×25 → pjc → plane_map →
//       plane_map_temp_table → laser_extrinsic_compensate；PJC poses.size()==25
//       且按姿态分组；PJC initialT/f 注入正确；进度 50..100 单调
//     4 LaserCancelBeforeStart：前置取消 → fail、算子零调用
//     5 PipelineTwoThreadsJoin：总装双线程并行 join/输出合并/进度 0..100 单调
//   层 2（真算子冒烟，条件跑）：6 RealOpsSmoke：1 姿态小图合成竖直亮线帧
//     跑 4-1→4-8 不崩（合成条纹无真值断言；PJC/4-12/4-13/5-3 留联调）
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#ifdef JMW_BUILD_CUDA
#include <opencv2/core/cuda.hpp>
#endif

#include "calib_synthetic.h"
#include "pipelines/calibcompute/CalibComputePipeline.h"
#include "pipelines/calibcompute/CalibComputeTypes.h"
#include "pipelines/calibcompute/LaserChain.h"
#include "pipelines/posture/PostureTypes.h"

using Scanner::pipeline::CalibComputeOutput;
using Scanner::pipeline::CalibComputePipeline;
using Scanner::pipeline::CancelToken;
using Scanner::pipeline::InitialCalibParams;
using Scanner::pipeline::LaserChain;
using Scanner::pipeline::LaserOps;
using Scanner::pipeline::LaserFramePoints;
using Scanner::pipeline::PoseLaserExtractions;
using Scanner::pipeline::PostureSessionData;
using Scanner::pipeline::StereoParams;
using Scanner::pipeline::synthetic::SyntheticTruth;

namespace {

// 轮询等待谓词成立（超时 false）
template <typename Pred>
bool waitUntil(Pred pred, int timeoutMs, int stepMs = 5) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(stepMs));
    }
    return pred();
}

// 假 StereoParams（结构有效即可；P1.fx=800 供 f 派生断言）
StereoParams makeFakeStereoParams() {
    StereoParams sp;
    sp.cameraMatrixL = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
    sp.cameraMatrixR = sp.cameraMatrixL.clone();
    sp.distCoeffsL = cv::Mat::zeros(1, 5, CV_64F);
    sp.distCoeffsR = sp.distCoeffsL.clone();
    sp.R = cv::Mat::eye(3, 3, CV_64F);
    sp.T = (cv::Mat_<double>(3, 1) << -100.0, 1.5, 3.0);
    sp.R1 = cv::Mat::eye(3, 3, CV_64F);
    sp.R2 = cv::Mat::eye(3, 3, CV_64F);
    sp.P1 = (cv::Mat_<double>(3, 4) << 800, 0, 320, 0, 0, 800, 240, 0, 0, 0, 1, 0);
    sp.P2 = sp.P1.clone();
    sp.Q = cv::Mat::eye(4, 4, CV_64F);
    sp.reprojError = 0.1;
    return sp;
}

PostureSessionData makeLaserSession(int poses, SyntheticTruth& truth) {
    auto s = Scanner::pipeline::synthetic::makeSyntheticSession(poses, truth);
    Scanner::pipeline::synthetic::attachSyntheticLaserFrames(s, 1, cv::Size(64, 48));
    return s;
}

// —— 假算子组状态（层 1）——
struct FakeOpsState {
    std::atomic<int> frontCount{0};
    std::atomic<int> backCount{0};
    std::vector<std::string> log;                       // 链线程单线程写（run 内）
    std::vector<calib::PosePointSet> pjcPoses;          // PJC 收到的 poses 拷贝
    cv::Vec3d pjcInitialTSeen{0, 0, 0};
    double fSeen = 0.0;
    cv::Point2d ppSeen{0, 0};
    cv::Vec3d projectorTSeen{0, 0, 0};
};

// 假算子：front 以调用序号（姿态号）作为线号 tag → back 透传 → PJC 收到的
// poses[i].lineIds[0]==i 证明按姿态分组且保序
LaserOps makeFakeLaserOps(FakeOpsState& st) {
    LaserOps ops;
    ops.front = [&](const Scanner::data::CycleUnit&, PoseLaserExtractions& out) {
        const int poseIdx = st.frontCount.fetch_add(1);
        st.log.push_back("front");
        LaserFramePoints fp;
        fp.points = {cv::Point2f(1.5f, 2.5f), cv::Point2f(3.5f, 4.5f)};
        fp.lineIds = {poseIdx, poseIdx};
        out.left.assign(1, fp);
        out.right.assign(1, fp);
        return Scanner::Result::ok();
    };
    ops.back = [&](const PoseLaserExtractions& in, const StereoParams&,
                   calib::PosePointSet& out) {
        const int poseIdx = in.left.at(0).lineIds.at(0);
        ++st.backCount;
        st.log.push_back("back");
        out.points3d = {cv::Vec3f(1, 2, 3), cv::Vec3f(4, 5, 6)};
        out.lineIds = {poseIdx};
        return Scanner::Result::ok();
    };
    ops.pjc = [&](const std::vector<calib::PosePointSet>& poses, double f,
                  const cv::Point2d& pp, const cv::Vec3d& initialT,
                  calib::ProjectorJointCalibResult& out) {
        st.pjcPoses = poses;
        st.fSeen = f;
        st.ppSeen = pp;
        st.pjcInitialTSeen = initialT;
        st.log.push_back("pjc");
        out.success = true;
        out.projectorT = cv::Vec3d(80.0, 3.0, 3.0);
        out.poseCount = static_cast<int>(poses.size());
        return Scanner::Result::ok();
    };
    ops.planeMap = [&](const cv::Vec3d& projectorT, double, const cv::Point2d&,
                       const StereoParams&, const std::vector<int>&,
                       const cv::Size&, calib::PlaneMapResult& out) {
        st.projectorTSeen = projectorT;
        st.log.push_back("plane_map");
        out.success = true;
        out.totalPairs = 1;
        return Scanner::Result::ok();
    };
    ops.planeMapTempTable = [&](const cv::Vec3d&, double, const cv::Point2d&,
                                const StereoParams&, const std::vector<int>&,
                                const cv::Size&, calib::PlaneMapTempTableResult& out) {
        st.log.push_back("plane_map_temp_table");
        out.success = true;
        out.tableSize = 1;
        return Scanner::Result::ok();
    };
    ops.laserExtrinsicCompensate = [&](const cv::Vec3d&, const StereoParams&,
                                       calib::LaserExtrinsicCompensateCPUResult& out) {
        st.log.push_back("laser_extrinsic_compensate");
        out.success = true;
        return Scanner::Result::ok();
    };
    return ops;
}

LaserChain makeFakeChain(FakeOpsState& st) {
    LaserChain::Deps deps;
    deps.pjcInitialT = cv::Vec3d(80.0, 3.0, 3.0);
    deps.ops = makeFakeLaserOps(st);
    return LaserChain(deps);
}

} // namespace

// —— 1. 前段先行：promise 未兑现时前段计数已达 25；后段在兑现后才跑 ——
TEST(LaserChainTest, FrontRunsBeforeStereoReady) {
    SyntheticTruth truth;
    PostureSessionData session = makeLaserSession(25, truth);
    FakeOpsState st;
    LaserChain chain = makeFakeChain(st);

    std::promise<StereoParams> promise;
    auto fut = promise.get_future();
    CalibComputeOutput out;
    std::atomic<bool> runDone{false};
    Scanner::Result res = Scanner::Result::fail("unset");
    std::thread th([&] {
        res = chain.run(session, std::move(fut), out, nullptr, CancelToken{});
        runDone.store(true);
    });

    // 前段（4-1..4-4）全 25 姿态在 promise 兑现前完成
    // （join 前不用 ASSERT：断言失败提前 return 会析构 joinable 线程 → terminate）
    const bool frontReached = waitUntil([&] { return st.frontCount.load() == 25; }, 5000);
    EXPECT_TRUE(frontReached) << "front count=" << st.frontCount.load();
    if (frontReached) {
        EXPECT_EQ(st.frontCount.load(), 25);
        EXPECT_EQ(st.backCount.load(), 0);       // 后段未跑
        EXPECT_FALSE(runDone.load());            // run 仍阻塞在 future
    }

    promise.set_value(makeFakeStereoParams());   // 3-4 兑现
    th.join();

    EXPECT_TRUE(runDone.load());
    EXPECT_TRUE(res.success) << res.message;
    EXPECT_EQ(st.backCount.load(), 25);          // 后段仅在兑现后跑
    EXPECT_TRUE(out.laserValid);
}

// —— 2. 破诺快速失败（相机链早期失败 → 不产出、不挂死 <2s）——
TEST(LaserChainTest, BrokenPromiseFailsFast) {
    SyntheticTruth truth;
    PostureSessionData session = makeLaserSession(25, truth);
    FakeOpsState st;
    LaserChain chain = makeFakeChain(st);

    std::future<StereoParams> fut;
    {
        std::promise<StereoParams> p;            // 作用域结束未 set → 破诺
        fut = p.get_future();
    }

    CalibComputeOutput out;
    const auto t0 = std::chrono::steady_clock::now();
    auto res = chain.run(session, std::move(fut), out, nullptr, CancelToken{});
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0).count();

    EXPECT_FALSE(res.success);
    EXPECT_LT(elapsedMs, 2000) << "broken promise must fail fast, took " << elapsedMs << "ms";
    EXPECT_NE(res.message.find("promise"), std::string::npos) << res.message;
    EXPECT_EQ(st.backCount.load(), 0);           // 后段零调用
    EXPECT_FALSE(out.laserValid);                // 不产出
}

// —— 3. 调用序 + 跨姿态聚合：front×25 → back×25 → pjc → 4-12 → 4-13 → 5-3 ——
TEST(LaserChainTest, ChainOrderAndAggregate) {
    SyntheticTruth truth;
    PostureSessionData session = makeLaserSession(25, truth);
    FakeOpsState st;
    LaserChain chain = makeFakeChain(st);

    std::promise<StereoParams> promise;
    auto fut = promise.get_future();
    const StereoParams sp = makeFakeStereoParams();
    promise.set_value(sp);

    std::vector<int> pcts;
    std::vector<std::string> stages;
    Scanner::pipeline::ProgressCb cb = [&](int p, const std::string& s) {
        pcts.push_back(p);
        stages.push_back(s);
    };

    CalibComputeOutput out;
    auto res = chain.run(session, std::move(fut), out, cb, CancelToken{});
    ASSERT_TRUE(res.success) << res.message;

    // 调用序：25×front → 25×back → pjc → plane_map → plane_map_temp_table
    //        → laser_extrinsic_compensate（各一次）
    ASSERT_EQ(st.log.size(), 54u) << "25 front + 25 back + 4 tail calls";
    for (int i = 0; i < 25; ++i) EXPECT_EQ(st.log[i], "front") << "i=" << i;
    for (int i = 25; i < 50; ++i) EXPECT_EQ(st.log[i], "back") << "i=" << i;
    EXPECT_EQ(st.log[50], "pjc");
    EXPECT_EQ(st.log[51], "plane_map");
    EXPECT_EQ(st.log[52], "plane_map_temp_table");
    EXPECT_EQ(st.log[53], "laser_extrinsic_compensate");

    // 聚合：PJC poses 25 组、按姿态分组且保序（lineIds[0]==姿态号）
    ASSERT_EQ(st.pjcPoses.size(), 25u);
    for (size_t i = 0; i < st.pjcPoses.size(); ++i) {
        ASSERT_EQ(st.pjcPoses[i].lineIds.size(), 1u);
        EXPECT_EQ(st.pjcPoses[i].lineIds[0], static_cast<int>(i));
        EXPECT_EQ(st.pjcPoses[i].points3d.size(), 2u);
    }

    // PJC 输入派生：f/主点 ← stereoK(P1 3×3)；initialT ← Deps
    EXPECT_EQ(st.fSeen, sp.P1.at<double>(0, 0));
    EXPECT_EQ(st.ppSeen.x, sp.P1.at<double>(0, 2));
    EXPECT_EQ(st.ppSeen.y, sp.P1.at<double>(1, 2));
    EXPECT_EQ(st.pjcInitialTSeen, cv::Vec3d(80.0, 3.0, 3.0));
    // 4-12 输入：virtualT = projectorT（PJC 输出直通）
    EXPECT_EQ(st.projectorTSeen, cv::Vec3d(80.0, 3.0, 3.0));

    // 输出：激光半区产物标记有效
    EXPECT_TRUE(out.laserValid);
    EXPECT_TRUE(out.pjc.success);
    EXPECT_TRUE(out.planeMap.success);
    EXPECT_TRUE(out.planeMapTempTable.success);
    EXPECT_TRUE(out.laserExtrinsicTempTable.success);

    // 进度：50..100 单调，首 50 尾 100
    ASSERT_GE(pcts.size(), 3u);
    EXPECT_EQ(pcts.front(), 50);
    EXPECT_EQ(pcts.back(), 100);
    for (size_t i = 0; i < pcts.size(); ++i) {
        EXPECT_GE(pcts[i], 50) << "idx=" << i << " stage=" << stages[i];
        EXPECT_LE(pcts[i], 100) << "idx=" << i << " stage=" << stages[i];
        if (i > 0) EXPECT_GE(pcts[i], pcts[i - 1]) << "idx=" << i << " stage=" << stages[i];
    }
}

// —— 4. 前置取消：fail 返回、算子零调用 ——
TEST(LaserChainTest, LaserCancelBeforeStart) {
    SyntheticTruth truth;
    PostureSessionData session = makeLaserSession(25, truth);
    FakeOpsState st;
    LaserChain chain = makeFakeChain(st);

    std::promise<StereoParams> promise;
    auto fut = promise.get_future();
    promise.set_value(makeFakeStereoParams());

    CalibComputeOutput out;
    CancelToken cancel;
    cancel.cancel();
    auto res = chain.run(session, std::move(fut), out, nullptr, cancel);

    EXPECT_FALSE(res.success);
    EXPECT_EQ(st.frontCount.load(), 0);
    EXPECT_FALSE(out.laserValid);
}

// —— 5. 总装双线程：并行 join / 输出合并 / 进度 0..100 单调 ——
// （T23 起 quality 由门禁合成：假链须填全好值产物方得 ok=true）
TEST(CalibComputePipelineTest, PipelineTwoThreadsJoin) {
    SyntheticTruth truth;
    PostureSessionData session = makeLaserSession(25, truth);
    const StereoParams spSent = makeFakeStereoParams();

    CalibComputePipeline pipe;
    pipe.attachBoardPoints(truth.boardPoints3D);
    pipe.attachInitialParams(Scanner::pipeline::synthetic::makeInitialFromTruth(truth));

    std::atomic<bool> camStarted{false}, lasStarted{false}, laserGotStereo{false};

    CalibComputePipeline::TestHooks hooks;
    hooks.cameraRun = [&](const PostureSessionData&, const InitialCalibParams&,
                          const std::vector<cv::Point3f>&, StereoParams& outStereo,
                          std::promise<StereoParams>& promise, CalibComputeOutput& out,
                          const Scanner::pipeline::ProgressCb& cb, const CancelToken&) {
        camStarted.store(true);
        cb(0, "camera start");
        // 相机链慢（3-x 标定重）；等激光链启动以证明并行——串行相机先行则超时失败
        if (!waitUntil([&] { return lasStarted.load(); }, 4000))
            return Scanner::Result::fail("laser chain not started concurrently");
        outStereo = spSent;
        out.stereo = spSent;
        out.intrinsicRmsL = 0.1;
        out.intrinsicRmsR = 0.1;
        out.rectifyValidRoiL = cv::Rect(0, 0, 600, 460);
        out.rectifyValidRoiR = cv::Rect(0, 0, 600, 460);
        out.rectifyTempTable.success = true;
        out.rectifyTempTable.tableSize = 1;
        promise.set_value(spSent);
        cb(50, "camera done");
        return Scanner::Result::ok("camera fake ok");
    };
    hooks.laserRun = [&](const PostureSessionData&, std::future<StereoParams> fut,
                         CalibComputeOutput& out, const Scanner::pipeline::ProgressCb& cb,
                         const CancelToken&) {
        lasStarted.store(true);
        if (fut.wait_for(std::chrono::seconds(3)) != std::future_status::ready)
            return Scanner::Result::fail("stereo future not ready in time");
        fut.get();                                // 4-5 起消费
        laserGotStereo.store(true);
        cb(50, "laser start");
        out.laserValid = true;
        out.pjc.success = true;
        out.planeMapTempTable.success = true;
        out.planeMapTempTable.tableSize = 1;
        out.laserExtrinsicTempTable.success = true;
        out.laserExtrinsicTempTable.leftResult.table.resize(1);
        out.laserExtrinsicTempTable.rightResult.table.resize(1);
        cb(100, "laser done");
        return Scanner::Result::ok("laser fake ok");
    };
    pipe.setTestHooks(hooks);

    std::vector<int> pcts;
    std::vector<std::string> stages;
    Scanner::pipeline::ProgressCb cb = [&](int p, const std::string& s) {
        pcts.push_back(p);
        stages.push_back(s);
    };

    EXPECT_FALSE(pipe.isRunning());
    CancelToken cancel;
    auto res = pipe.run(session, cb, cancel);
    EXPECT_FALSE(pipe.isRunning());

    // 双链均成功、并行（laser 等到了 promise）且相机确认 laser 并发起跑
    ASSERT_TRUE(res.success) << res.message;
    EXPECT_TRUE(camStarted.load());
    EXPECT_TRUE(lasStarted.load());
    EXPECT_TRUE(laserGotStereo.load());

    // 输出合并：相机半区 stereo + 激光半区 laserValid 同桶
    EXPECT_NEAR(pipe.output().stereo.cameraMatrixL.at<double>(0, 0),
                spSent.cameraMatrixL.at<double>(0, 0), 1e-12);
    EXPECT_TRUE(pipe.output().laserValid);
    EXPECT_TRUE(pipe.output().quality.ok);

    // 进度合并：0..100、单调不回退（总装层单调门）
    ASSERT_GE(pcts.size(), 3u);
    EXPECT_EQ(pcts.front(), 0);
    EXPECT_EQ(pcts.back(), 100);
    for (size_t i = 0; i < pcts.size(); ++i) {
        EXPECT_GE(pcts[i], 0) << "idx=" << i << " stage=" << stages[i];
        EXPECT_LE(pcts[i], 100) << "idx=" << i << " stage=" << stages[i];
        if (i > 0) EXPECT_GE(pcts[i], pcts[i - 1]) << "idx=" << i << " stage=" << stages[i];
    }
}

// —— 6. 层 2 真算子冒烟（1 姿态小图；覆盖 4-1→4-8；条件跑）——
// 覆盖算子：4-1 mask_extract / 4-2 ccl(region_analyze) / 4-3 laser_label /
//   4-4 steger(ByLabel) / 4-5 undistort_cuda / 4-6 epipolar_interp(lineIdCheck) /
//   4-7 laser_match / 4-8 laser_reconstruct。
// 不覆盖：PJC / 4-12 / 4-13 / 5-3（需 ≥5 姿态真值点云——留联调）。
// 合成条纹非实拍：steger 中心/匹配对无真值断言；算子抛异常 → 记 warn 断言不崩。
#ifdef JMW_BUILD_CUDA
TEST(LaserChainRealOpsTest, FrontBackSmoke) {
    if (cv::cuda::getCudaEnabledDeviceCount() == 0)
        GTEST_SKIP() << "no CUDA device available";

    SyntheticTruth truth;
    PostureSessionData session =
        Scanner::pipeline::synthetic::makeSyntheticSession(1, truth);
    // 真分辨率帧（640×480）：默认 mask erode=5 需条纹核区 ≥~7px（σ=2.5 →
    // ≥80 灰度核宽约 8px，腐蚀后余 4px——过 mask/ccl/label 阈值）
    Scanner::pipeline::synthetic::attachSyntheticLaserFrames(
        session, 1, cv::Size(640, 480), 2, 2.5);
    StereoParams sp = Scanner::pipeline::synthetic::makeSyntheticStereoParams(truth);
    LaserOps ops = Scanner::pipeline::makeRealLaserOps();
    ASSERT_TRUE(ops.front && ops.back) << "real laser ops unavailable";

    // —— 前段 4-1→4-4 ——
    PoseLaserExtractions ext;
    bool frontOk = false;
    try {
        Scanner::Result rf = ops.front(session.poses[0].cycle, ext);
        frontOk = rf.success;
        if (!frontOk) std::cout << "[WARN] real front fail: " << rf.message << "\n";
    } catch (const std::exception& e) {
        std::cout << "[WARN] real front threw (联调): " << e.what() << "\n";
    } catch (...) {
        std::cout << "[WARN] real front threw unknown exception (联调)\n";
    }
    EXPECT_TRUE(frontOk) << "4-1..4-4 应在合成亮线帧上成功";
    if (frontOk) {
        ASSERT_EQ(ext.left.size(), 1u);
        ASSERT_EQ(ext.right.size(), 1u);
        std::cout << "[INFO] front points L=" << ext.left[0].points.size()
                  << " R=" << ext.right[0].points.size() << "\n";
        EXPECT_FALSE(ext.left[0].points.empty());
        EXPECT_EQ(ext.left[0].points.size(), ext.left[0].lineIds.size());

        // —— 后段 4-5→4-8（零视差合成帧：match 可能空产出——不崩即达标）——
        try {
            calib::PosePointSet ps;
            Scanner::Result rb = ops.back(ext, sp, ps);
            std::cout << "[INFO] back success=" << rb.success
                      << " pts=" << ps.points3d.size()
                      << " msg=" << rb.message << "\n";
            SUCCEED() << "4-5..4-8 completed without crash";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "real back threw: " << e.what();
        } catch (...) {
            ADD_FAILURE() << "real back threw unknown exception";
        }
    }
}
#endif // JMW_BUILD_CUDA

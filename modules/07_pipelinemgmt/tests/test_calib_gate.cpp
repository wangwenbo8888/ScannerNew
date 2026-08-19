// ============================================================================
// test_calib_gate.cpp — P4-T23 B 质量门禁 + 落盘接线 TDD 测试
// ============================================================================
// 1 AllPassNormal       ：全好值合成 output → evaluate → ok=true/Normal/items=门禁数
// 2 PjcFailFault        ：pjcRms 超阈 + PJC 产物缺 → overall=Fault
// 3 SoftFailDegraded    ：stereoReprojError 轻超 → Degraded/ok=false/summary 含项名
// 4 SerializeRoundtrip  ：serializeCalic → json 含 stereo/tempTables/pjc/quality；
//                         quality.items 数组可读回
// 5 RunEndWritesRepo    ：假 repo + 假链（TestHooks）→ run → write 恰 1 次且 json
//                         可解析；repo 空 → 不调不崩；写失败 → sink 收 Fault 1801
//                         （真 EventBus + EventBusEventSink 旁观）
// 6 ThresholdConfigurable：自定义阈值改变判定（evaluate 直评 + Config 贯穿 run）
#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "base/EventBus.h"
#include "base/types.h"
#include "pipelines/PipelineDeps.h"
#include "pipelines/PipelineEventSink.h"
#include "pipelines/calibcompute/CalibComputePipeline.h"
#include "pipelines/calibcompute/CalibComputeTypes.h"
#include "pipelines/calibcompute/CalibSerialize.h"
#include "pipelines/calibcompute/QualityGate.h"
#include "pipelines/posture/PostureTypes.h"

using Scanner::pipeline::CalibComputeOutput;
using Scanner::pipeline::CalibComputePipeline;
using Scanner::pipeline::CancelToken;
using Scanner::pipeline::InitialCalibParams;
using Scanner::pipeline::PostureSessionData;
using Scanner::pipeline::QualityReport;
using Scanner::pipeline::StereoParams;
using Scanner::pipeline::gate::Thresholds;
using Scanner::pipeline::gate::evaluate;
using Scanner::QualityFlag;

namespace {

constexpr size_t kGateItemCount = 8;   // 门禁项数（任务书清单）

// 全好值合成输出（各门禁指标均在默认阈值内）
void fillGoodStereoHalf(CalibComputeOutput& out) {
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
    sp.reprojError = 0.1;                       // < 1.0
    out.stereo = sp;
    out.intrinsicRmsL = 0.15;                   // < 0.5
    out.intrinsicRmsR = 0.18;
    out.rectifyValidRoiL = cv::Rect(0, 0, 600, 460);
    out.rectifyValidRoiR = cv::Rect(0, 0, 598, 455);
    out.intrinsicTempTable.success = true;
    out.extrinsicTempTable.success = true;
    out.rectifyTempTable.success = true;
    out.rectifyTempTable.tableSize = 101;
}

void fillGoodLaserHalf(CalibComputeOutput& out) {
    out.laserValid = true;
    out.pjc.success = true;
    out.pjc.finalSampsonRms = 0.045;            // < 0.15
    out.pjc.jacobianConditionNumber = 1e3;      // < 1e10
    out.pjc.improvementRatio = 8.0;             // >= 1
    out.planeMap.success = true;
    out.planeMap.totalPairs = 2;
    out.planeMapTempTable.success = true;
    out.planeMapTempTable.tableSize = 101;
    out.laserExtrinsicTempTable.success = true;
    out.laserExtrinsicTempTable.leftResult.table.resize(1);
    out.laserExtrinsicTempTable.rightResult.table.resize(1);
}

CalibComputeOutput makeGoodOutput() {
    CalibComputeOutput out;
    fillGoodStereoHalf(out);
    fillGoodLaserHalf(out);
    return out;
}

// 假仓库：记录 write 调用（06 侧 ICalibRepoWriter 由 01/06 接入期实现）
struct FakeRepo : Scanner::pipeline::ICalibRepoWriter {
    int calls = 0;
    std::string last;
    bool ret = true;
    bool write(const std::string& json) override {
        ++calls;
        last = json;
        return ret;
    }
};

// 假链钩子：直接填好值产物（不跑真算子）
CalibComputePipeline::TestHooks makeGoodHooks() {
    CalibComputePipeline::TestHooks hooks;
    hooks.cameraRun = [](const PostureSessionData&, const InitialCalibParams&,
                         const std::vector<cv::Point3f>&, StereoParams& outStereo,
                         std::promise<StereoParams>& promise, CalibComputeOutput& out,
                         const Scanner::pipeline::ProgressCb& cb, const CancelToken&) {
        fillGoodStereoHalf(out);
        outStereo = out.stereo;
        promise.set_value(outStereo);
        if (cb) cb(50, "camera fake done");
        return Scanner::Result::ok("camera fake ok");
    };
    hooks.laserRun = [](const PostureSessionData&, std::future<StereoParams> fut,
                        CalibComputeOutput& out, const Scanner::pipeline::ProgressCb& cb,
                        const CancelToken&) {
        fut.wait();
        fillGoodLaserHalf(out);
        if (cb) cb(100, "laser fake done");
        return Scanner::Result::ok("laser fake ok");
    };
    return hooks;
}

} // namespace

// —— 1. 全过 → Normal ——
TEST(CalibGateTest, AllPassNormal) {
    CalibComputeOutput out = makeGoodOutput();
    QualityReport qr = evaluate(out, Thresholds{});
    EXPECT_TRUE(qr.ok);
    EXPECT_EQ(qr.overall, QualityFlag::Normal);
    ASSERT_EQ(qr.items.size(), kGateItemCount);
    for (const auto& item : qr.items) EXPECT_TRUE(item.pass) << item.name;
}

// —— 2. PJC rms 超阈 + 产物缺 → Fault ——
TEST(CalibGateTest, PjcFailFault) {
    CalibComputeOutput out = makeGoodOutput();
    out.pjc.finalSampsonRms = 0.48;   // > 0.15 默认阈
    out.pjc.success = false;          // PJC 产物缺（关键项）
    out.laserValid = false;
    QualityReport qr = evaluate(out, Thresholds{});
    EXPECT_FALSE(qr.ok);
    EXPECT_EQ(qr.overall, QualityFlag::Fault);
}

// —— 3. 软失败（产物齐、指标轻超）→ Degraded，summary 含项名 ——
TEST(CalibGateTest, SoftFailDegraded) {
    CalibComputeOutput out = makeGoodOutput();
    out.stereo.reprojError = 1.2;     // > 1.0 默认阈（轻超）
    QualityReport qr = evaluate(out, Thresholds{});
    EXPECT_FALSE(qr.ok);
    EXPECT_EQ(qr.overall, QualityFlag::Degraded);
    EXPECT_NE(qr.summary.find("stereoReprojError"), std::string::npos) << qr.summary;
}

// —— 4. 序列化往返：顶层键齐 + quality.items 可读回 ——
TEST(CalibGateTest, SerializeRoundtrip) {
    CalibComputeOutput out = makeGoodOutput();
    out.quality = evaluate(out, Thresholds{});
    nlohmann::json j = Scanner::pipeline::serializeCalib(out);

    EXPECT_TRUE(j.contains("stereo"));
    EXPECT_TRUE(j.contains("tempTables"));
    EXPECT_TRUE(j.contains("pjc"));
    EXPECT_TRUE(j.contains("planeMap"));
    EXPECT_TRUE(j.contains("quality"));

    EXPECT_NEAR(j["stereo"]["reprojError"].get<double>(), 0.1, 1e-12);

    const auto& items = j["quality"]["items"];
    ASSERT_TRUE(items.is_array());
    ASSERT_EQ(items.size(), kGateItemCount);
    EXPECT_TRUE(items[0]["name"].get<std::string>().size() > 0);
    EXPECT_TRUE(items[0]["pass"].get<bool>());
    EXPECT_EQ(j["quality"]["overall"].get<std::string>(), "Normal");
}

// —— 5. run 尾自动写仓库 ——
TEST(CalibGateTest, RunEndWritesRepo) {
    PostureSessionData session;       // 假链不消费会话内容
    CancelToken cancel;

    // 5a. 假 repo → write 恰 1 次、json 可解析
    {
        FakeRepo repo;
        CalibComputePipeline pipe;
        Scanner::pipeline::PipelineDeps deps;
        deps.calibRepo = &repo;
        ASSERT_TRUE(pipe.configure(deps).success);
        pipe.setTestHooks(makeGoodHooks());

        auto res = pipe.run(session, nullptr, cancel);
        ASSERT_TRUE(res.success) << res.message;
        EXPECT_EQ(repo.calls, 1);
        nlohmann::json j = nlohmann::json::parse(repo.last);
        EXPECT_TRUE(j.contains("stereo"));
        EXPECT_TRUE(j.contains("quality"));
    }

    // 5b. repo 空 → 不调不崩
    {
        CalibComputePipeline pipe;    // 未 configure（repo/sink 均空）
        pipe.setTestHooks(makeGoodHooks());
        auto res = pipe.run(session, nullptr, cancel);
        EXPECT_TRUE(res.success) << res.message;
    }

    // 5c. 写失败 → sink 收 Fault 1801（真 EventBus 旁观 FaultOccurred）
    {
        Scanner::infra::EventBus bus;
        std::vector<int64_t> codes;
        auto sub = bus.subscribe(Scanner::EventType::FaultOccurred,
                                 [&](const Scanner::Event& e) { codes.push_back(e.param1); });
        FakeRepo repo;
        repo.ret = false;             // write 返回 false
        CalibComputePipeline pipe;
        Scanner::pipeline::PipelineDeps deps;
        deps.calibRepo = &repo;
        deps.eventBus = &bus;         // configure 内建 EventBusEventSink
        ASSERT_TRUE(pipe.configure(deps).success);
        pipe.setTestHooks(makeGoodHooks());

        auto res = pipe.run(session, nullptr, cancel);
        EXPECT_TRUE(res.success) << res.message;   // 产物完好，落盘失败经事件上报
        bool got1801 = false;
        for (int64_t c : codes)
            if (c == 1801) got1801 = true;
        EXPECT_TRUE(got1801) << "codes seen: " << codes.size();
        bus.unsubscribe(sub);
    }
}

// —— 6. 阈值参数化：自定义阈值改变判定 ——
TEST(CalibGateTest, ThresholdConfigurable) {
    CalibComputeOutput out = makeGoodOutput();    // stereoReprojError=0.1

    Thresholds loose;                             // 默认 stereoReprojMax=1.0 → 过
    EXPECT_EQ(evaluate(out, loose).overall, QualityFlag::Normal);

    Thresholds tight;
    tight.stereoReprojMax = 0.05;                 // 收紧 → stereoReprojError 项 fail
    QualityReport qr = evaluate(out, tight);
    EXPECT_EQ(qr.overall, QualityFlag::Degraded);
    EXPECT_FALSE(qr.ok);

    // Config 贯穿：紧阈 Config 的 run 结果带 Degraded 标志
    PostureSessionData session;
    CancelToken cancel;
    CalibComputePipeline::Config cfg;
    cfg.thresholds = tight;
    CalibComputePipeline pipe(cfg);
    pipe.setTestHooks(makeGoodHooks());
    auto res = pipe.run(session, nullptr, cancel);
    EXPECT_TRUE(res.success);                     // Degraded 仍成功（可复检）
    EXPECT_EQ(res.qualityFlag, QualityFlag::Degraded);
    EXPECT_FALSE(pipe.output().quality.ok);
}

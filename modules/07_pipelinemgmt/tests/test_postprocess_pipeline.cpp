// ============================================================================
// test_postprocess_pipeline.cpp — E 后处理五阶段编排壳（桩算子注入/skip 掩码/
// 进度取消）。网格四族算子 09 待建——IMeshStageOp 记录型桩注入验证编排：
// 执行序/进度单调/skip 不降级/桩 pending 降级续跑/阶段 fail-fast/阶段间取消
// （cancel=degraded，B 链惯例）/数据流转真实性/start-stop 适配。
// ============================================================================
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "base/EventBus.h"
#include "pipelines/postprocess/PostProcessPipeline.h"

using namespace Scanner::pipeline;
using Scanner::infra::EventBus;

namespace {

// —— 记录型桩：run 时名字压栈；统一钩子（数据变更/取消注入/阻塞节奏）——
struct RecordingOp : IMeshStageOp {
    std::string label;
    std::vector<std::string>* log = nullptr;
    Scanner::Result behavior = Scanner::Result::ok();
    std::function<void(MeshData&, const CancelToken&)> hook;

    std::string name() const override { return label; }
    Scanner::Result run(MeshData& io, const CancelToken& cancel) override {
        log->push_back(label);
        if (hook) hook(io, cancel);
        return behavior;
    }
};

std::unique_ptr<RecordingOp> recOp(const std::string& label,
                                   std::vector<std::string>* log) {
    auto op = std::make_unique<RecordingOp>();
    op->label = label;
    op->log = log;
    return op;
}

/// 五阶段注入齐全（法线/封装/补洞/光顺/边界）
void injectAll(PostProcessPipeline& p, std::vector<std::string>& log) {
    p.setStageOp(0, recOp("法线", &log));
    p.setStageOp(1, recOp("封装", &log));
    p.setStageOp(2, recOp("补洞", &log));
    p.setStageOp(3, recOp("光顺", &log));
    p.setStageOp(4, recOp("边界", &log));
}

/// 追加点位的桩（流转真实性：每桩给 xyz 加一点 (v,v,v)）
std::unique_ptr<RecordingOp> appendOp(const std::string& label,
                                      std::vector<std::string>* log, float v) {
    auto op = recOp(label, log);
    op->hook = [v](MeshData& io, const CancelToken&) {
        io.xyz.push_back(v);
        io.xyz.push_back(v + 1);
        io.xyz.push_back(v + 2);
    };
    return op;
}

/// 事件采集（同 test_globaloptim：订阅 FaultOccurred，记录 param1=事件码）
struct EventCollector {
    EventBus bus;
    std::vector<int64_t> codes;
    Scanner::infra::SubscriberId sub = 0;

    EventCollector() {
        sub = bus.subscribe(Scanner::EventType::FaultOccurred,
                            [this](const Scanner::Event& e) { codes.push_back(e.param1); });
    }
    ~EventCollector() { bus.unsubscribe(sub); }
    bool hasCode(int32_t code) const {
        return std::find(codes.begin(), codes.end(), static_cast<int64_t>(code)) != codes.end();
    }
};

/// 假 STL 导出（T27/接入期由 app 侧以 file_io::exportSTL 适配接线）
struct FakeExporter {
    int calls = 0;
    std::string lastPath;
    bool ok = true;
    bool operator()(const std::string& path, const MeshData&) {
        ++calls;
        lastPath = path;
        return ok;
    }
};

void wireExporter(PostProcessPipeline& p, FakeExporter& exporter) {
    p.setStlExporter([&exporter](const std::string& path, const MeshData& m) {
        return exporter(path, m);
    });
}

MeshData makeCloud(size_t n) {
    MeshData m;
    m.xyz.reserve(n * 3);
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) m.xyz.push_back(static_cast<float>(i * 3 + k));
    return m;
}

const std::vector<std::string> kFiveOrder = {"法线", "封装", "补洞", "光顺", "边界"};

} // namespace

// ============================================================================
// 1：五阶段顺序——记录型桩（名字压栈）→ 执行序 == 法线/封装/补洞/光顺/边界；
//    完成事件 1900、STL 导出经注入器恰一次（路径=配置路径）
// ============================================================================
TEST(PostProcessPipelineTest, FiveStagesInOrder) {
    std::vector<std::string> log;
    PostProcessPipeline::Config cfg;
    cfg.outputPath = "outfive.stl";
    PostProcessPipeline p(cfg);
    injectAll(p, log);

    EventCollector ev;
    PipelineDeps deps;
    deps.eventBus = &ev.bus;
    ASSERT_TRUE(p.configure(deps).success);
    FakeExporter exporter;
    wireExporter(p, exporter);

    CancelToken cancel;
    auto res = p.run(makeCloud(2), nullptr, cancel);
    ASSERT_TRUE(res.success) << res.message;
    EXPECT_EQ(res.qualityFlag, Scanner::QualityFlag::Normal);
    EXPECT_EQ(log, kFiveOrder);
    EXPECT_TRUE(ev.hasCode(1900));                       // 完成事件
    EXPECT_EQ(exporter.calls, 1);                        // STL 导出恰一次
    EXPECT_EQ(exporter.lastPath, "outfive.stl");
}

// ============================================================================
// 2：进度——cb 收 0..100 单调不减、首值 0、终值 100（跳过段亦占均分份额）
// ============================================================================
TEST(PostProcessPipelineTest, ProgressMonotonicFullRange) {
    std::vector<std::string> log;
    PostProcessPipeline p;
    injectAll(p, log);
    ASSERT_TRUE(p.configure(PipelineDeps{}).success);    // 无 bus：sink nullptr 安全
    p.setStlExporter([](const std::string&, const MeshData&) { return true; });

    std::vector<int> pcts;
    ProgressCb cb = [&](int pct, const std::string&) { pcts.push_back(pct); };
    CancelToken cancel;
    auto res = p.run(makeCloud(1), cb, cancel);
    ASSERT_TRUE(res.success) << res.message;

    ASSERT_FALSE(pcts.empty());
    EXPECT_EQ(pcts.front(), 0);
    EXPECT_EQ(pcts.back(), 100);
    EXPECT_TRUE(std::is_sorted(pcts.begin(), pcts.end())) << "进度应单调不减";
    for (int v : pcts) {
        EXPECT_GE(v, 0);
        EXPECT_LE(v, 100);
    }
}

// ============================================================================
// 3：skip 掩码——skipStages=0b01010（跳封装+光顺）→ 执行序仅 法线/补洞/边界；
//    显式跳过不算降级（quality Normal）
// ============================================================================
TEST(PostProcessPipelineTest, SkipMask) {
    std::vector<std::string> log;
    PostProcessPipeline::Config cfg;
    cfg.skipStages = 0b01010;
    PostProcessPipeline p(cfg);
    injectAll(p, log);
    ASSERT_TRUE(p.configure(PipelineDeps{}).success);
    p.setStlExporter([](const std::string&, const MeshData&) { return true; });

    CancelToken cancel;
    auto res = p.run(makeCloud(1), nullptr, cancel);
    ASSERT_TRUE(res.success) << res.message;
    EXPECT_EQ(log, (std::vector<std::string>{"法线", "补洞", "边界"}));
    EXPECT_EQ(res.qualityFlag, Scanner::QualityFlag::Normal) << "显式跳过不算降级";
}

// ============================================================================
// 4：桩 pending 降级续跑——桩 2 返回 degraded("operator pending") → 整体
//    degraded、后续段仍执行（09 落地前网格四族桩即此态）
// ============================================================================
TEST(PostProcessPipelineTest, StubPendingDegradesButContinues) {
    std::vector<std::string> log;
    PostProcessPipeline p;
    injectAll(p, log);
    auto hole = recOp("补洞", &log);
    hole->behavior = Scanner::Result::degraded("operator pending");
    p.setStageOp(2, std::move(hole));
    ASSERT_TRUE(p.configure(PipelineDeps{}).success);
    p.setStlExporter([](const std::string&, const MeshData&) { return true; });

    CancelToken cancel;
    auto res = p.run(makeCloud(1), nullptr, cancel);
    EXPECT_TRUE(res.success) << "降级非失败";
    EXPECT_TRUE(res.isDegraded());
    EXPECT_EQ(log, kFiveOrder) << "后续段仍执行";
}

// ============================================================================
// 5：阶段 fail-fast——桩 1 fail → 立即返回 fail、桩 2+ 未执行、不导出、
//    无完成事件 1900（阶段失败 Fault 1901 上报）
// ============================================================================
TEST(PostProcessPipelineTest, StageFailFast) {
    std::vector<std::string> log;
    PostProcessPipeline p;
    injectAll(p, log);
    auto wrap = recOp("封装", &log);
    wrap->behavior = Scanner::Result::fail("synthetic stage failure");
    p.setStageOp(1, std::move(wrap));

    EventCollector ev;
    PipelineDeps deps;
    deps.eventBus = &ev.bus;
    ASSERT_TRUE(p.configure(deps).success);
    FakeExporter exporter;
    wireExporter(p, exporter);

    CancelToken cancel;
    auto res = p.run(makeCloud(1), nullptr, cancel);
    EXPECT_FALSE(res.success);
    EXPECT_TRUE(res.isFault());
    EXPECT_EQ(log, (std::vector<std::string>{"法线", "封装"})) << "fail 后续段不执行";
    EXPECT_EQ(exporter.calls, 0) << "fail 不导出";
    EXPECT_TRUE(ev.hasCode(1901));                       // 阶段失败 Fault 上报
    EXPECT_FALSE(ev.hasCode(1900)) << "失败无完成事件";
}

// ============================================================================
// 6：阶段间取消——阶段 1 执行中置取消 → 阶段 2 前检查点安全退出 degraded
//    （B 链惯例：cancel=degraded——用户意图中止非数据问题，不记 Fault）、
//    后续段不执行、不导出、无完成事件
// ============================================================================
TEST(PostProcessPipelineTest, CancelBetweenStages) {
    std::vector<std::string> log;
    PostProcessPipeline p;
    injectAll(p, log);
    CancelToken cancel;
    auto wrap = recOp("封装", &log);
    wrap->hook = [&cancel](MeshData&, const CancelToken&) { cancel.cancel(); };
    p.setStageOp(1, std::move(wrap));

    EventCollector ev;
    PipelineDeps deps;
    deps.eventBus = &ev.bus;
    ASSERT_TRUE(p.configure(deps).success);
    FakeExporter exporter;
    wireExporter(p, exporter);

    auto res = p.run(makeCloud(1), nullptr, cancel);
    EXPECT_TRUE(res.success) << "取消安全退出非崩非 Fault";
    EXPECT_TRUE(res.isDegraded());
    EXPECT_EQ(log, (std::vector<std::string>{"法线", "封装"})) << "取消后不进后续段";
    EXPECT_EQ(exporter.calls, 0) << "取消不导出";
    EXPECT_FALSE(ev.hasCode(1900)) << "取消无完成事件";
}

// ============================================================================
// 7：数据流转真实性——五桩每桩给 xyz 加一点 → output 含累计变更
//    （2 初始点 → 7；末点=边界桩追加值）
// ============================================================================
TEST(PostProcessPipelineTest, OutputPassedThrough) {
    std::vector<std::string> log;
    PostProcessPipeline p;
    p.setStageOp(0, appendOp("法线", &log, 1));
    p.setStageOp(1, appendOp("封装", &log, 4));
    p.setStageOp(2, appendOp("补洞", &log, 7));
    p.setStageOp(3, appendOp("光顺", &log, 10));
    p.setStageOp(4, appendOp("边界", &log, 13));
    ASSERT_TRUE(p.configure(PipelineDeps{}).success);
    p.setStlExporter([](const std::string&, const MeshData&) { return true; });

    CancelToken cancel;
    auto res = p.run(makeCloud(2), nullptr, cancel);
    ASSERT_TRUE(res.success) << res.message;
    EXPECT_EQ(log, kFiveOrder);
    EXPECT_EQ(p.output().pointCount(), 7u) << "五桩累计变更应留存于 output";
    const auto& xyz = p.output().xyz;                    // 末点=边界桩 (13,14,15)
    ASSERT_EQ(xyz.size(), 21u);
    EXPECT_EQ(xyz[18], 13);
    EXPECT_EQ(xyz[19], 14);
    EXPECT_EQ(xyz[20], 15);
}

// ============================================================================
// 8：start/stop 适配——attachCloud 后 start() 后台 run；阶段 4 桩先追加数据、
//    通知入段、随后轮询直至取消（stop 内部 cancel）→ isRunning 翻转正确、
//    stop() join 收尾安全、导出前检查点已取消不导出、output 留存累计变更
// ============================================================================
TEST(PostProcessPipelineTest, StartStopAdapter) {
    std::vector<std::string> log;
    PostProcessPipeline p;
    injectAll(p, log);
    // 阶段 4 桩：追加数据 → 通知已入段 → 自旋等取消（stop 触发）
    std::promise<void> entered;
    std::future<void> enteredFut = entered.get_future();
    auto border = recOp("边界", &log);
    border->hook = [&entered](MeshData& io, const CancelToken& tok) {
        io.xyz.push_back(9);
        io.xyz.push_back(9);
        io.xyz.push_back(9);
        entered.set_value();
        while (!tok.cancelled()) std::this_thread::yield();
    };
    p.setStageOp(4, std::move(border));
    ASSERT_TRUE(p.configure(PipelineDeps{}).success);
    FakeExporter exporter;
    wireExporter(p, exporter);

    p.attachCloud(makeCloud(1));
    ASSERT_TRUE(p.start().success);
    ASSERT_EQ(enteredFut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(p.isRunning());

    p.stop();                                             // cancel（桩解除自旋）+ join
    EXPECT_FALSE(p.isRunning());
    EXPECT_EQ(log, kFiveOrder);
    EXPECT_EQ(p.output().pointCount(), 2u) << "1 初始点 + 边界桩 1 点";
    EXPECT_EQ(exporter.calls, 0) << "stop 的 cancel 先于 join——导出前检查点取消，不导出";
}

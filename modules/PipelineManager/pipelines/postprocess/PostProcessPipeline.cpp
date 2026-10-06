// ============================================================================
// PostProcessPipeline.cpp — E 后处理流水线实现（编排细节见头文件注）
//
// 五阶段顺序：法线→封装→补洞→光顺→边界（skipStages 位跳过，显式跳过不算
// 降级）→ STL 导出（注入器空=产物占位+Degraded）。进度 5 段均分 0..100（跳过
// 段占份额报 skipped）；取消检查点在每阶段前+导出前（cancel=degraded，B 链
// 惯例）；阶段 fail 立即中止（Fault 1901）；桩 degraded 聚合为整体 Degraded
// 续跑。完成事件 1900（reportCompletion）。
// ============================================================================
#include "pipelines/postprocess/PostProcessPipeline.h"

#include <exception>
#include <utility>

#include <spdlog/spdlog.h>

namespace Scanner::pipeline {
namespace {

// 事件码（07 E 后处理族）
constexpr int32_t kEvtDone = 1900;          // E 完成（Normal/Degraded 按整体质量）
constexpr int32_t kEvtStageFail = 1901;     // 阶段失败（Fault，fail-fast 中止）
constexpr int32_t kEvtStlPlaceholder = 1902;// STL 导出未接线（Degraded，产物占位）
constexpr int32_t kEvtStlExportFail = 1903; // STL 导出失败（Fault，不中止完成上报）

const char* const kStageNames[PostProcessPipeline::kStageCount] = {
    "法线重算", "封装", "补洞", "光顺", "边界"};

/// 内置 pending 桩：不动数据，恒 degraded（09 网格四族/法线实接后替换）
class PendingStageOp final : public IMeshStageOp {
public:
    explicit PendingStageOp(const char* name) : name_(name) {}
    std::string name() const override { return name_; }
    Result run(MeshData& /*io*/, const CancelToken& /*cancel*/) override {
        return Result::degraded("operator pending");
    }
private:
    const char* name_;
};

} // namespace

// ============================================================================
// 构造 / 装配
// ============================================================================
PostProcessPipeline::PostProcessPipeline(Config cfg) : cfg_(std::move(cfg)) {
    for (int i = 0; i < kStageCount; ++i)
        ops_[i] = std::make_unique<PendingStageOp>(kStageNames[i]);
}

PostProcessPipeline::~PostProcessPipeline() { stop(); }

void PostProcessPipeline::setStageOp(int stageIdx, std::unique_ptr<IMeshStageOp> op) {
    if (stageIdx < 0 || stageIdx >= kStageCount) return;   // 域外忽略（防御）
    ops_[stageIdx] = std::move(op);
}

void PostProcessPipeline::setStlExporter(StlExportFn fn) { stlExport_ = std::move(fn); }

Result PostProcessPipeline::configure(const PipelineDeps& deps) {
    if (running_.load())
        return Result::fail("PostProcessPipeline::configure: 运行中不可配置");
    sink_ = std::make_unique<EventBusEventSink>(deps.eventBus);   // nullptr 安全
    configured_ = true;
    return Result::ok("postprocess configured");
}

const MeshData& PostProcessPipeline::output() const { return out_; }

// ============================================================================
// run — 阻塞批算（五阶段 → STL 导出）
// ============================================================================
Result PostProcessPipeline::run(MeshData&& cloud, ProgressCb cb, CancelToken& cancel) {
    if (!configured_)
        return Result::fail("PostProcessPipeline::run: 须先 configure");
    std::lock_guard<std::mutex> lock(runMutex_);
    if (running_.load())
        return Result::fail("postprocess already running");
    return runInternal(std::move(cloud), cb, cancel);
}

Result PostProcessPipeline::runInternal(MeshData cloud, const ProgressCb& cb,
                                        CancelToken& cancel) {
    running_.store(true);
    struct RunningGuard {
        std::atomic<bool>& flag;
        ~RunningGuard() { flag.store(false); }
    } runningGuard{running_};

    out_ = std::move(cloud);
    Scanner::QualityFlag quality = Scanner::QualityFlag::Normal;
    if (cb) cb(0, "postprocess start");

    // —— 五阶段顺序执行（skipStages 位跳过；跳过段占均分进度份额报 skipped）——
    for (int i = 0; i < kStageCount; ++i) {
        // 取消检查点（阶段前）：cancel=degraded（B 链惯例——用户意图中止非
        // 数据问题，不记 Fault）；不导出、不发完成事件
        if (cancel.cancelled())
            return Result::degraded(std::string("PostProcess 已取消（") + kStageNames[i] +
                                    " 前检查点）——cancel=degraded（B 链惯例）");
        const int pctBegin = i * 100 / kStageCount;
        const int pctEnd = (i + 1) * 100 / kStageCount;

        if ((cfg_.skipStages >> i) & 1u) {          // 显式跳过：不算降级
            if (cb) cb(pctEnd, std::string(kStageNames[i]) + " skipped");
            continue;
        }

        IMeshStageOp* op = ops_[i].get();
        if (cb) cb(pctBegin, kStageNames[i]);

        Result r;
        if (!op) {                                  // 防御：未注入且内置被清——视同 pending
            r = Result::degraded("operator pending");
        } else {
            try {
                r = op->run(out_, cancel);
            } catch (const std::exception& e) {
                r = Result::fail(std::string("阶段异常: ") + e.what());
            } catch (...) {
                r = Result::fail("阶段未知异常");
            }
        }

        if (r.isFault()) {                          // fail-fast：立即中止
            if (sink_)
                sink_->report(Scanner::QualityFlag::Fault, kEvtStageFail,
                              std::string(kStageNames[i]) + " 失败: " + r.message);
            return Result::fail(std::string("PostProcess 阶段失败（") + kStageNames[i] +
                                "）: " + r.message);
        }
        if (r.isDegraded())                         // 桩 pending 等：聚合降级续跑
            quality = Scanner::QualityFlag::Degraded;

        if (cb) cb(pctEnd, std::string(kStageNames[i]) + " done");
    }

    // —— 取消检查点（导出前）——
    if (cancel.cancelled())
        return Result::degraded("PostProcess 已取消（STL 导出前检查点）——"
                                "cancel=degraded（B 链惯例）");

    // —— STL 导出（06 file_io 接线 T27/接入期；现阶段注入器空=产物占位）——
    if (stlExport_) {
        bool wrote = false;
        std::string err;
        try {
            wrote = stlExport_(cfg_.outputPath, out_);
        } catch (const std::exception& e) {
            err = e.what();
        } catch (...) {
            err = "unknown exception";
        }
        if (!wrote) {
            quality = Scanner::QualityFlag::Degraded;
            if (sink_)
                sink_->report(Scanner::QualityFlag::Fault, kEvtStlExportFail,
                              "STL 导出失败: " + cfg_.outputPath +
                                  (err.empty() ? std::string() : ": " + err));
        }
    } else {
        // file_io.cpp 现编入 app 且依赖 OSG——07 库不直链，产物占位+降级（T27 接线）
        quality = Scanner::QualityFlag::Degraded;
        if (sink_)
            sink_->report(Scanner::QualityFlag::Degraded, kEvtStlPlaceholder,
                          "STL 导出未接线（产物占位）——06 file_io exportSTL 适配"
                          " T27/接入期接入");
    }

    // —— 完成事件 + 终点进度 ——
    if (sink_)
        sink_->reportCompletion(quality, kEvtDone,
                                std::string("postprocess done | points=") +
                                    std::to_string(out_.pointCount()) +
                                    " triangles=" + std::to_string(out_.triangles.size() / 3) +
                                    " skip=0x" + std::to_string(cfg_.skipStages));
    if (cb) cb(100, "postprocess done");
    return quality == Scanner::QualityFlag::Normal
               ? Result::ok("postprocess done")
               : Result::degraded("postprocess done (degraded)");
}

// ============================================================================
// IPipelineObject 适配：start=后台 run（attachCloud 数据源）；stop=cancel+join
// ============================================================================
void PostProcessPipeline::attachCloud(MeshData cloud) { attachedCloud_ = std::move(cloud); }

Result PostProcessPipeline::start() {
    std::lock_guard<std::mutex> lock(runMutex_);
    if (running_.load())
        return Result::fail("postprocess already running");
    if (!configured_)
        return Result::fail("PostProcessPipeline::start: 须先 configure");
    if (attachedCloud_.xyz.empty())
        return Result::fail("PostProcessPipeline::start: 须先 attachCloud(非空点云)");
    if (worker_.joinable()) worker_.join();

    auto token = std::make_shared<CancelToken>();
    cancelToken_ = token;                    // 管道持有；worker 捕获共享保活
    MeshData cloud = std::move(attachedCloud_);
    running_.store(true);                    // 先置位（runInternal 的 guard 负责清位）
    try {
        worker_ = std::thread([this, token, cloud = std::move(cloud)]() mutable {
            runInternal(std::move(cloud), nullptr, *token);
        });
    } catch (...) {                          // 线程构造失败：复位标志/令牌，不留卡死态
        running_.store(false);
        cancelToken_.reset();
        return Result::fail("postprocess failed to start worker thread");
    }
    return Result::ok("postprocess started (background run)");
}

void PostProcessPipeline::stop() {
    // 锁内取令牌共享引用再 cancel：与 start() 重建令牌互斥（消 UAF 窗口，同 B/D 模式）
    std::shared_ptr<CancelToken> token;
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        token = cancelToken_;
    }
    if (token) token->cancel();
    std::thread local;
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        local = std::move(worker_);
    }
    if (local.joinable()) local.join();
}

bool PostProcessPipeline::isRunning() const { return running_.load(); }

} // namespace Scanner::pipeline

// ============================================================================
// PostProcessWorkflow.cpp — 后处理工作流实现（编排壳；五阶段移交 07 E 对象）
//
// P6-T29c：旧七阶段（GBA/重融合/法线/封装/补洞/光顺/边界）sleep 空壳删除
// （GBA/重融合 Q5 定案归 02）——本类装配 07 PostProcessPipeline 阻塞 run：
// 进度回调透传 UI、STL 导出经 StlExportFn 适配 06 file_io exportSTL
// （file_io.cpp 编入 app 且依赖 OSG——07 库不链 OSG，适配须在本 app 编译
// 单元内完成，见 exportStlViaFileIo）。
// ============================================================================

#include "PostProcessWorkflow.h"
#include "PointCloudBuffer.h"
#include "file_io.h"            // 06 STL 导出（app 编译单元内适配）

#include "pipelines/PipelineDeps.h"

#include <osg/Vec3>
#include <spdlog/spdlog.h>
#include <chrono>
#include <utility>

namespace Scanner::workflow {
namespace {

// ============================================================================
// exportStlViaFileIo — StlExportFn → 06 file_io::exportSTL 适配
//
// 07 的 MeshData（float xyz/normals + uint32 三角索引）→ file_io::MeshData
// （osg::Vec3 顶点/法线 + 索引）。真导出接线（T27 占位注释的兑现）：
// file_io.cpp 与 OSG 均属 app 侧——本函数只能在 app 编译单元内存在。
// ============================================================================
bool exportStlViaFileIo(const std::string& path, const Scanner::pipeline::MeshData& mesh) {
    file_io::MeshData out;
    out.vertices.reserve(mesh.pointCount());
    for (size_t i = 0; i + 2 < mesh.xyz.size(); i += 3) {
        out.vertices.emplace_back(mesh.xyz[i], mesh.xyz[i + 1], mesh.xyz[i + 2]);
    }
    if (mesh.normals.size() == mesh.xyz.size()) {         // 法线数量匹配才透传
        out.normals.reserve(mesh.pointCount());
        for (size_t i = 0; i + 2 < mesh.normals.size(); i += 3) {
            out.normals.emplace_back(mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2]);
        }
    }
    out.indices.assign(mesh.triangles.begin(), mesh.triangles.end());
    return file_io::exportSTL(path, out);
}

} // namespace

PostProcessWorkflow::PostProcessWorkflow(WorkflowContext* ctx)
    : ctx_(ctx), cancelToken_(std::make_unique<Scanner::pipeline::CancelToken>()) {}
PostProcessWorkflow::~PostProcessWorkflow() { stop(); }

Result PostProcessWorkflow::initialize() {
    if (!ctx_) return Result::fail("无 WorkflowContext");
    spdlog::info("[PostProcess] 初始化 (五阶段, skip=0x{:x}, 输出={})", skipStages_, outputPath_);
    return Result::ok();
}

Result PostProcessWorkflow::makeCloudData(Scanner::pipeline::MeshData& out) const {
    if (!ctx_ || !ctx_->pointCloudBuffer()) return Result::fail("无点云数据");

    uint64_t version = 0;
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colors;
    ctx_->pointCloudBuffer()->getSnapshot(version, points, colors);
    if (points.empty()) return Result::fail("点云为空");

    out.xyz.reserve(points.size() * 3);
    for (const auto& p : points) {
        out.xyz.push_back(p.x);
        out.xyz.push_back(p.y);
        out.xyz.push_back(p.z);
    }
    spdlog::info("[PostProcess] 读入 {} 点 (version={})", points.size(), version);
    // TODO(接入期): 输入改 06 点云仓库（app 存活件）内存句柄——02 GBA 修正
    // 点云写入仓库后此处直取修正结果（现 PointCloudBuffer 快照即其内存形态）
    return Result::ok();
}

Result PostProcessWorkflow::start() {
    if (state_ == WorkflowState::Running) return Result::ok();

    Scanner::pipeline::MeshData cloud;
    auto mr = makeCloudData(cloud);
    if (!mr.success) {
        result_.success = false;
        result_.message = mr.message;
        state_ = WorkflowState::Error;
        spdlog::error("[PostProcess] {}", mr.message);
        return Result::fail(mr.message);
    }

    // 装配 07 E（会话私有件：skipStages/outputPath 透传；STL 真导出接线）
    Scanner::pipeline::PostProcessPipeline::Config cfg;
    cfg.skipStages = skipStages_;
    cfg.outputPath = outputPath_;
    pipeline_ = std::make_unique<Scanner::pipeline::PostProcessPipeline>(std::move(cfg));
    pipeline_->setStlExporter(&exportStlViaFileIo);   // app 侧适配 06 file_io exportSTL
    // TODO(接入期): 网格四族算子 09 落地后经 pipeline_->setStageOp 注入

    Scanner::pipeline::PipelineDeps deps;
    deps.eventBus = ctx_ ? ctx_->eventBus() : nullptr;
    auto cr = pipeline_->configure(deps);
    if (!cr.success) {
        pipeline_.reset();
        result_.success = false;
        result_.message = "PostProcessPipeline 装配失败: " + cr.message;
        state_ = WorkflowState::Error;
        return Result::fail(result_.message);
    }

    cancelToken_ = std::make_unique<Scanner::pipeline::CancelToken>();   // 会话新令牌
    result_ = PostProcessResult{};
    state_ = WorkflowState::Running;
    running_ = true;
    postThread_ = std::thread([this, c = std::move(cloud)]() mutable {
        postProcessLoop(std::move(c));
    });
    return Result::ok();
}

void PostProcessWorkflow::postProcessLoop(Scanner::pipeline::MeshData cloud) {
    spdlog::info("[PostProcess] 后处理启动（五阶段移交 07 PostProcessPipeline）");
    const auto t0 = std::chrono::steady_clock::now();

    auto res = pipeline_->run(
        std::move(cloud),
        [this](int pct, const std::string& stage) {
            WorkflowProgress p;
            p.state = state_.load();
            p.currentStage = pct / 20;                 // 0..100 → 阶段 0..5
            p.totalStages = Scanner::pipeline::PostProcessPipeline::kStageCount;
            p.stageName = stage;
            p.progress = pct / 100.0f;
            if (callback_) callback_(p);
        },
        *cancelToken_);

    const auto t1 = std::chrono::steady_clock::now();
    result_.success = res.success;
    result_.meshTriangles = static_cast<int>(pipeline_->output().triangles.size() / 3);
    result_.smoothTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result_.outputPath = outputPath_;
    result_.message = res.message;
    state_ = res.success ? WorkflowState::Completed : WorkflowState::Error;
    running_ = false;

    spdlog::info("[PostProcess] 后处理结束: success={} degraded={} → {} (三角面 {}, {:.0f}ms)",
                 res.success, res.isDegraded(), outputPath_, result_.meshTriangles,
                 result_.smoothTimeMs);

    // 合账钩子——app 侧注入调 gate->notifyCompleted 切 S2（§9 04 行）。
    // 恰一次：线程体每 start() 会话恰执行一次（start 同步失败在 handler 内
    // 已被 gate 回滚 S2 不经此处；stop 后重复 join 不重跑线程体）。
    // 用户主动停止（令牌已取消）按 T15 口径报 ok=true——会话正常终止非故障。
    // 04 不依赖 10：经 std::function 回调反向解耦（JMW_LOG 宏头在 10，
    // 以下按其展开格式直写 spdlog——输出与 JMW_LOG_INFO 等价，§8.2 生命周期）
    const bool userCancelled = cancelToken_ && cancelToken_->cancelled();
    const bool ok = userCancelled ? true : res.success;
    spdlog::info("[04-PostProcess] 后处理会话终止 ok={}（{}）", ok,
                 userCancelled ? "用户停止" : (res.success ? "批算完成" : "阶段失败"));
    if (onFinished_) onFinished_(ok);
}

Result PostProcessWorkflow::stop() {
    running_ = false;
    if (cancelToken_) cancelToken_->cancel();   // 阶段前/导出前检查点安全退出
    if (pipeline_) pipeline_->stop();           // cancel + join（run 路径经外部令牌）
    if (postThread_.joinable()) postThread_.join();
    pipeline_.reset();
    state_ = WorkflowState::Idle;
    return Result::ok();
}

Result PostProcessWorkflow::setProgressCallback(WorkflowCallback cb) {
    callback_ = std::move(cb);
    return Result::ok();
}

} // namespace Scanner::workflow

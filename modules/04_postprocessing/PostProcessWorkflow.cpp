#include "PostProcessWorkflow.h"
#include "data/PointCloudBuffer.h"
#include <spdlog/spdlog.h>
#include <chrono>

namespace Scanner::workflow {

PostProcessWorkflow::PostProcessWorkflow(WorkflowContext* ctx) : ctx_(ctx) {}
PostProcessWorkflow::~PostProcessWorkflow() { stop(); }

Result PostProcessWorkflow::initialize() {
    if (!ctx_) return Result::fail("无 WorkflowContext");
    spdlog::info("[PostProcess] 初始化 (光顺迭代={}, 输出={})", smoothIter_, outputPath_);
    return Result::ok();
}

Result PostProcessWorkflow::start() {
    if (state_ == WorkflowState::Running) return Result::ok();
    state_ = WorkflowState::Running;
    running_ = true;
    postThread_ = std::thread(&PostProcessWorkflow::postProcessLoop, this);
    return Result::ok();
}

Result PostProcessWorkflow::stop() {
    running_ = false;
    state_ = WorkflowState::Idle;
    if (postThread_.joinable()) postThread_.join();
    return Result::ok();
}

Result PostProcessWorkflow::setProgressCallback(WorkflowCallback cb) {
    callback_ = std::move(cb);
    return Result::ok();
}

void PostProcessWorkflow::postProcessLoop() {
    spdlog::info("[PostProcess] 后处理启动 (七阶段)");

    // 读取点云快照
    if (!ctx_ || !ctx_->pointCloudBuffer()) {
        result_.success = false;
        result_.message = "无点云数据";
        state_ = WorkflowState::Error;
        return;
    }

    uint64_t version = 0;
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colors;
    ctx_->pointCloudBuffer()->getSnapshot(version, points, colors);

    if (points.empty()) {
        result_.success = false;
        result_.message = "点云为空";
        state_ = WorkflowState::Error;
        return;
    }

    spdlog::info("[PostProcess] 读入 {} 点 (version={})", points.size(), version);

    // 七阶段流水线
    const char* stages[] = {
        "全局标记点优化",  // 0: GBA
        "重融合",          // 1: laser_cloud_fuse
        "法线计算",        // 2: laser_cloud_normal
        "封装(网格化)",    // 3: Poisson/贪心三角化
        "补洞",            // 4: 网格补洞
        "光顺(HC)",        // 5: HC 平滑
        "边界优化"         // 6: 边界优化
    };

    for (int i = 0; i < 7 && running_; ++i) {
        notifyProgress(i, stages[i]);

        // TODO: 调用对应算子
        // Stage 0: global_ba_cpu (GBA)
        // Stage 1: laser_cloud_fuse_cpu (重融合)
        // Stage 2: laser_cloud_normal_cpu (法线)
        // Stage 3: 网格封装 (待建)
        // Stage 4: 补洞 (待建)
        // Stage 5: HC 平滑 (待建)
        // Stage 6: 边界优化 (待建)

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 导出 STL (TODO: 调用 FileService)
    result_.success = true;
    result_.meshTriangles = 0;  // 占位
    result_.outputPath = outputPath_;
    result_.message = "后处理完成";
    state_ = WorkflowState::Completed;

    spdlog::info("[PostProcess] 后处理完成 → {}", outputPath_);
}

void PostProcessWorkflow::notifyProgress(int stage, const std::string& name) {
    spdlog::info("[PostProcess] [{}/7] {}", stage + 1, name);
    if (!callback_) return;
    WorkflowProgress p;
    p.state = state_.load();
    p.currentStage = stage;
    p.totalStages = 7;
    p.stageName = name;
    p.progress = static_cast<float>(stage + 1) / 7.0f;
    callback_(p);
}

} // namespace Scanner::workflow

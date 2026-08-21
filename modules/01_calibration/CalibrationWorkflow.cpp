// ============================================================================
// CalibrationWorkflow.cpp — 标定工作流实现（编排壳；A/B 移交 07 对象）
//
// P6-T29b：旧棋盘格 6 步链（findChessboardCorners/内参/外参/矫正/温度表）
// 整体删除——A 姿态采集（PosturePipeline）集齐 25 后 completionHook 异步
// 移交本类，B 标定批算（CalibComputePipeline：相机链‖激光链 PJC+质量门禁）
// 阻塞 run，进度回调透传 UI；run 尾经 ICalibRepoWriter 适配（RepositoryWriter）
// 直写 06 标定仓库。
// ============================================================================

#include "CalibrationWorkflow.h"
#include "CalibrationRepository.h"

#include "pipelines/PipelineDeps.h"
#include "pipelines/calibcompute/CalibComputePipeline.h"
#include "pipelines/calibcompute/CalibSerialize.h"

#include <nlohmann/json.hpp>
#include "core/common/json_utils.h"     // calib::jsonToMatAuto

#include <spdlog/spdlog.h>
#include <utility>

namespace Scanner::workflow {
namespace {

constexpr const char* kCalibSessionPath = "calib_session.json";   // 06 会话档三键

// ============================================================================
// RepositoryWriter — ICalibRepoWriter 适配（B run 尾自动写 06 标定仓库）
//
// 07 serializeCalib 载荷原文 + imageSize 一次入仓：解析→校验→填内存→
// 临时文件+原子改名落盘（calibration.json）；app 存活件，扫描门禁/转接同源读。
// ============================================================================
class RepositoryWriter final : public Scanner::pipeline::ICalibRepoWriter {
public:
    RepositoryWriter(WorkflowContext* ctx, cv::Size imageSize)
        : ctx_(ctx), imageSize_(imageSize) {}

    bool write(const std::string& json) override {
        auto* repo = ctx_ ? ctx_->calibRepo() : nullptr;
        if (!repo) return false;
        const auto r = repo->write(json, imageSize_);
        if (!r.success) {
            spdlog::error("[CalibWorkflow] 标定仓库写入失败: {}", r.message);
            return false;
        }
        return true;
    }

private:
    WorkflowContext* ctx_;
    cv::Size imageSize_;
};

// ============================================================================
// applyInitialParamsJson — 会话档 initialParams 原文→PostureInitialParams
// （字段照 PosturePipeline.h：九 Mat + imageWidth/imageHeight/maskRatioThreshold；
//   缺键/坏值跳过该字段保默认——paramsReady 兜底拦截）
// ============================================================================
void applyInitialParamsJson(const nlohmann::json& j,
                            Scanner::pipeline::PostureInitialParams& p) {
    if (!j.is_object()) return;
    auto mat = [&j](const char* key, cv::Mat& dst) {
        const auto it = j.find(key);
        if (it == j.end() || !it->is_array() || it->empty()) return;
        try { dst = calib::jsonToMatAuto(*it); } catch (...) {}   // 坏值保默认
    };
    mat("K1", p.K1); mat("D1", p.D1); mat("K2", p.K2); mat("D2", p.D2);
    mat("R1", p.R1); mat("R2", p.R2); mat("P1", p.P1); mat("P2", p.P2); mat("Q", p.Q);
    if (j.contains("imageWidth") && j["imageWidth"].is_number_integer())
        p.imageWidth = j["imageWidth"].get<int>();
    if (j.contains("imageHeight") && j["imageHeight"].is_number_integer())
        p.imageHeight = j["imageHeight"].get<int>();
    if (j.contains("maskRatioThreshold") && j["maskRatioThreshold"].is_number())
        p.maskRatioThreshold = j["maskRatioThreshold"].get<double>();
}

} // namespace

CalibrationWorkflow::CalibrationWorkflow(WorkflowContext* ctx)
    : ctx_(ctx), cancelToken_(std::make_unique<Scanner::pipeline::CancelToken>()) {}
CalibrationWorkflow::~CalibrationWorkflow() { stop(); }

// ============================================================================
// 参数就绪检查（真实来源 TODO 接入期；缺参防御 fail 不崩）
// ============================================================================
bool CalibrationWorkflow::paramsReady() const {
    const auto& p = initialParams_;
    const bool initOk = !p.K1.empty() && !p.K2.empty() && !p.D1.empty() && !p.D2.empty()
        && !p.R1.empty() && !p.R2.empty() && !p.P1.empty() && !p.P2.empty() && !p.Q.empty()
        && p.imageWidth > 0 && p.imageHeight > 0 && p.maskRatioThreshold > 0.0;
    return initOk && !targets_.empty() && !boardPoints_.empty();
}

Result CalibrationWorkflow::initialize() {
    if (!ctx_) return Result::fail("无 WorkflowContext");

    // 配置装载（06 会话件）：缺档/坏档不崩——保持空参防言语义（paramsReady 拦截）
    Scanner::data::CalibSessionConfig cfg;
    const auto lr = calibSession_.load(kCalibSessionPath, cfg);
    if (lr.success) {
        setTargets(std::move(cfg.targets));
        setBoardPoints(std::move(cfg.boardPoints));
        applyInitialParamsJson(cfg.initialParams, initialParams_);
        spdlog::info("[CalibWorkflow] 会话档已装载: 目标 {} / 板点 {}",
                     targets_.size(), boardPoints_.size());
    } else {
        spdlog::info("[CalibWorkflow] 未装载会话档（{}）——空参防言语义", lr.message);
    }

    if (!paramsReady()) {
        state_ = WorkflowState::Error;
        const std::string msg =
            "标定装配参数不全（初始参数组/25 目标表/板点）——TODO 接入期：真实参数来源装载";
        spdlog::error("[CalibWorkflow] {}", msg);
        notifyError(msg);
        return Result::fail(msg);
    }
    spdlog::info("[CalibWorkflow] 初始化（A/B 参数齐备：目标 {} / 板点 {} / 分辨率 {}x{}）",
                 targets_.size(), boardPoints_.size(),
                 initialParams_.imageWidth, initialParams_.imageHeight);
    return Result::ok();
}

Result CalibrationWorkflow::start() {
    if (state_ == WorkflowState::Running) return Result::ok();
    if (!paramsReady()) {                     // UI 未检 initialize 返回值亦不崩
        state_ = WorkflowState::Error;
        const std::string msg = "标定装配参数不全——先 initialize 校验（TODO 接入期：真实参数来源）";
        spdlog::error("[CalibWorkflow] {}", msg);
        notifyError(msg);
        return Result::fail(msg);
    }

    auto ar = startPosture();
    if (!ar.success) {
        state_ = WorkflowState::Error;
        return ar;
    }

    if (calibThread_.joinable()) calibThread_.join();   // 上一会话 B 线程收尾
    cancelToken_ = std::make_unique<Scanner::pipeline::CancelToken>();   // 会话新令牌
    result_ = CalibrationResult{};
    state_ = WorkflowState::Running;
    running_ = true;
    notifyProgress(0, Scanner::pipeline::PostureSessionData::kTargetCount, "姿态采集");
    spdlog::info("[CalibWorkflow] 已启动（A 姿态采集移交 07 PosturePipeline）");
    return Result::ok();
}

Result CalibrationWorkflow::startPosture() {
    namespace sp = Scanner::pipeline;

    posture_ = std::make_unique<sp::PosturePipeline>();   // 确认表阈值默认（联调标定）
    // TODO(接入期): 08 采集侧写 CycleUnit 进姿态环（Backpressure 反压）；
    //   当前无真帧源——A 空转等周期（07 防御路径保留）
    posture_->attachRing(calibSession_.cycleRing());
    using TargetRow = const double[16];                    // std::array<double,16> 布局对齐 double[16]
    posture_->attachTargets(reinterpret_cast<TargetRow*>(targets_.data()),
                             static_cast<int>(targets_.size()));
    posture_->attachInitialParams(initialParams_);

    // 收口钩子：A 集齐 25 自动收口后携 SessionData 叫醒 B。
    // 钩子契约（只做异步移交即返）：B 批算在专属线程，长阻塞会占住 A 收口
    // watcher 线程并传导到 stop()/析构。
    posture_->setCompletionHook([this](sp::PostureSessionData&& session) {
        try {
            calibThread_ = std::thread(
                [this, s = std::move(session)]() mutable { runCompute(std::move(s)); });
        } catch (const std::exception& e) {
            spdlog::error("[CalibWorkflow] B 批算线程创建失败: {}", e.what());
            state_ = WorkflowState::Error;
            running_ = false;
            notifyError(std::string("标定计算线程创建失败: ") + e.what());
            // 异步失败同样合账（§3.3 不许悬死 S3——会话已终，回报 false）
            spdlog::info("[01-CalibWorkflow] 标定完成 ok=false（B 线程创建失败）");
            if (onFinished_) onFinished_(false);
        }
    });

    sp::PipelineDeps deps;
    deps.eventBus = ctx_ ? ctx_->eventBus() : nullptr;
    // TODO(接入期): deps.acquisition（08 采集门面——集齐收口自动停采）
    // TODO(接入期): deps.sceneFeed（03 渲染——实时姿态/标志点检出推送）
    auto cr = posture_->configure(deps);
    if (!cr.success) {
        posture_.reset();
        const std::string msg = "PosturePipeline 装配失败: " + cr.message;
        spdlog::error("[CalibWorkflow] {}", msg);
        notifyError(msg);
        return Result::fail(msg);
    }

    auto sr = posture_->start();
    if (!sr.success) {
        posture_.reset();
        const std::string msg = "PosturePipeline 启动失败: " + sr.message;
        spdlog::error("[CalibWorkflow] {}", msg);
        notifyError(msg);
        return Result::fail(msg);
    }
    return Result::ok();
}

// ============================================================================
// B 批算（completionHook 异步线程内；阻塞 run→进度透传→结果记账）
// ============================================================================
void CalibrationWorkflow::runCompute(Scanner::pipeline::PostureSessionData session) {
    namespace sp = Scanner::pipeline;

    spdlog::info("[CalibWorkflow] A 集齐 {} 姿态——B 标定批算启动", session.collectedCount);
    notifyProgress(0, 100, "标定计算");

    compute_ = std::make_unique<sp::CalibComputePipeline>();   // PJC 初值/门禁阈值默认（T23 联调）

    // B 初始参数与 A 严格同组（装配层单源=initialParams_）
    sp::InitialCalibParams init;
    init.K1 = initialParams_.K1;  init.D1 = initialParams_.D1;
    init.K2 = initialParams_.K2;  init.D2 = initialParams_.D2;
    init.R1 = initialParams_.R1;  init.P1 = initialParams_.P1;
    init.R2 = initialParams_.R2;  init.P2 = initialParams_.P2;
    init.imageSize = cv::Size(initialParams_.imageWidth, initialParams_.imageHeight);
    compute_->attachInitialParams(std::move(init));
    compute_->attachBoardPoints(boardPoints_);

    calibRepo_ = std::make_unique<RepositoryWriter>(
        ctx_, cv::Size(initialParams_.imageWidth, initialParams_.imageHeight));

    sp::PipelineDeps deps;
    deps.eventBus = ctx_ ? ctx_->eventBus() : nullptr;
    deps.calibRepo = calibRepo_.get();          // run 尾自动写 06 标定仓库
    compute_->configure(deps);

    auto res = compute_->run(
        session,
        [this](int pct, const std::string& stage) { notifyProgress(pct, 100, stage); },
        *cancelToken_);

    const auto& out = compute_->output();
    result_.success = res.success;
    result_.reprojErrorLeft = out.intrinsicRmsL;
    result_.reprojErrorRight = out.intrinsicRmsR;
    result_.stereoError = out.stereo.reprojError;
    result_.message = res.message;

    state_ = res.success ? WorkflowState::Completed : WorkflowState::Error;
    running_ = false;
    notifyProgress(100, 100, res.success ? "标定完成" : "标定失败");
    spdlog::info("[CalibWorkflow] B 批算结束: success={} rmsL={:.4f} rmsR={:.4f} stereo={:.4f}",
                 res.success, result_.reprojErrorLeft, result_.reprojErrorRight,
                 result_.stereoError);

    // 完成回报（P5-T14）：合账钩子——app 侧注入调 gate->notifyCompleted 切 S2。
    // 01 不依赖 10：此处经 std::function 回调反向解耦（JMW_LOG 宏头在 10，
    // 以下按其展开格式直写 spdlog——输出与 JMW_LOG_INFO 等价，§8.2 生命周期）
    spdlog::info("[01-CalibWorkflow] 标定完成 ok={}", res.success);
    if (onFinished_) onFinished_(res.success);
}

Result CalibrationWorkflow::stop() {
    running_ = false;
    if (posture_) posture_->stop();          // 同步停（不触发 completion、不停采集）
    if (cancelToken_) cancelToken_->cancel();  // B 链取消检查点贯穿
    if (compute_) compute_->stop();          // cancel+join（幂等；run 路径经外部令牌）
    if (calibThread_.joinable()) calibThread_.join();
    posture_.reset();
    compute_.reset();
    calibRepo_.reset();
    state_ = WorkflowState::Idle;
    return Result::ok();
}

Result CalibrationWorkflow::setProgressCallback(WorkflowCallback cb) {
    callback_ = std::move(cb);
    return Result::ok();
}

void CalibrationWorkflow::notifyProgress(int current, int total, const std::string& stage) {
    spdlog::debug("[CalibWorkflow] {} ({}/{})", stage, current, total);
    if (!callback_) return;
    WorkflowProgress p;
    p.state = state_.load();
    p.currentStage = current;
    p.totalStages = total;
    p.stageName = stage;
    p.progress = total > 0 ? static_cast<float>(current) / total : 0.0f;
    callback_(p);
}

void CalibrationWorkflow::notifyError(const std::string& msg) {
    if (!callback_) return;
    WorkflowProgress p;
    p.state = WorkflowState::Error;
    p.stageName = "错误";
    p.message = msg;
    p.progress = 0.0f;
    callback_(p);
}

} // namespace Scanner::workflow

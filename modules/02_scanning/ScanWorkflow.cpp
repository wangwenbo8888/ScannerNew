// ============================================================================
// ScanWorkflow.cpp — 扫描工作流实现（CPU 标记点链 + 体素融合）
//
// 管线: Capture → Preprocess(阈值掩码) → Marker(CCL→Zernike→Ellipse→Match→Reconstruct)
//       → Fuse(laser_cloud_fuse_cpu → PointCloudBuffer)
// ============================================================================

#include "ScanWorkflow.h"
#include "FrameBuffer.h"
#include "PointCloudBuffer.h"
#include "SessionService.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>
#include <chrono>

// 标记点链算子
#include "zernike_edge_cpu.h"
#include "ellipse_fit_cpu.h"
#include "marker_match_cpu.h"
#include "point_reconstruct_cpu.h"

// 融合算子
#include "laser_cloud_fuse_cpu.h"

namespace Scanner::workflow {

// ============================================================================
// CaptureStage
// ============================================================================
CaptureStage::CaptureStage(WorkflowContext* ctx)
    : Stage("Capture"), ctx_(ctx) {}

Result CaptureStage::process() {
    if (!ctx_ || !ctx_->frameBuffer()) return Result::fail("无 FrameBuffer");
    auto frame = ctx_->frameBuffer()->popFrame(std::chrono::milliseconds(100));
    if (!frame) return Result::degraded("Capture: 帧超时");
    std::lock_guard lock(frameMutex_);
    latestFrame_ = *frame;
    return Result::ok();
}

data::FrameData CaptureStage::getLatestFrame() const {
    std::lock_guard lock(frameMutex_);
    return latestFrame_;
}

// ============================================================================
// PreprocessStage — OpenCV 阈值 + 形态学生成标记点掩码
// ============================================================================
PreprocessStage::PreprocessStage(WorkflowContext* ctx)
    : Stage("Preprocess"), ctx_(ctx) {}

cv::Mat PreprocessStage::createMarkerMask(const cv::Mat& gray) {
    cv::Mat binary;
    // 自适应阈值检测亮标记点
    cv::adaptiveThreshold(gray, binary, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 51, -5);
    // 形态学开运算去噪
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    return binary;
}

Result PreprocessStage::process() {
    if (!hasInput_) return Result::degraded("Preprocess: 无输入");

    leftMarkerMask = createMarkerMask(inputFrame_.leftGray);
    rightMarkerMask = createMarkerMask(inputFrame_.rightGray);
    return Result::ok();
}

// ============================================================================
// MarkerStage — 标记点检测链
// ============================================================================
MarkerStage::MarkerStage(WorkflowContext* ctx)
    : Stage("Marker"), ctx_(ctx) {
    // 创建算子一次，复用
    zernikeOp_ = std::make_unique<calib::ZernikeEdgeCPU>();
    ellipseOp_ = std::make_unique<calib::EllipseFitCPU>();
    matchOp_ = std::make_unique<calib::MarkerMatchCPU>();
}

MarkerStage::~MarkerStage() = default;

void MarkerStage::setInput(const data::FrameData& frame,
                            const cv::Mat& leftMask, const cv::Mat& rightMask) {
    inputFrame_ = frame;
    leftMask_ = leftMask;
    rightMask_ = rightMask;
}

std::vector<cv::Point2f> MarkerStage::detectCenters(const cv::Mat& gray, const cv::Mat& mask) {
    std::vector<cv::Point2f> centers;
    if (gray.empty() || mask.empty()) return centers;

    try {
    // CCL: 找连通域（替代 CUDA region_analyze）
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    for (int i = 1; i < numLabels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < 15 || area > 5000) continue;

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (w < 3 || h < 3) continue;

        // 扩展 ROI 边界（安全裁剪）
        int pad = 5;
        x = std::max(0, x - pad);
        y = std::max(0, y - pad);
        w = std::min(gray.cols - x, w + 2 * pad);
        h = std::min(gray.rows - y, h + 2 * pad);
        if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > gray.cols || y + h > gray.rows) continue;

        cv::Rect roi(x, y, w, h);
        cv::Mat subImage = gray(roi).clone();  // clone 确保连续内存
        if (subImage.empty() || subImage.cols < 5 || subImage.rows < 5) continue;

        // Zernike 边缘提取（复用算子对象）
        auto zr = zernikeOp_->Execute(subImage);
        if (!zr.success || zr.edgePoints.empty()) continue;

        // 椭圆拟合（复用算子对象）
        auto er = ellipseOp_->Execute(zr.edgePoints);
        if (!er.success) continue;

        auto center = er.centerPoint2f();
        centers.emplace_back(center.x + x, center.y + y);
    }
    } catch (...) {
        spdlog::warn("[MarkerStage] detectCenters 异常，跳过本帧");
    }

    return centers;
}

Result MarkerStage::process() {
    result = ScanFrameResult();

    try {
    // 检测左右图像标记点中心
    auto centersL = detectCenters(inputFrame_.leftGray, leftMask_);
    auto centersR = detectCenters(inputFrame_.rightGray, rightMask_);

    if (centersL.size() < 3 || centersR.size() < 3) {
        spdlog::debug("[MarkerStage] 标记点不足: L={} R={}", centersL.size(), centersR.size());
        return Result::degraded("标记点不足");
    }

    // 立体匹配（复用算子）
    auto matchR = matchOp_->Execute(centersL, centersR);
    if (!matchR.success || matchR.centerMatches.empty()) {
        spdlog::debug("[MarkerStage] 匹配失败");
        return Result::degraded("匹配失败");
    }

    // 3D 重建（需要标定参数）
    if (calib_.valid) {
        calib::PointReconstructCPU reconOp;
        reconOp.SetProjectionMatrices(calib_.P1, calib_.P2, calib_.Q);

        // 使用简化重建接口
        std::vector<int> leftIds, rightIds;
        for (size_t i = 0; i < matchR.centerMatches.size(); ++i) {
            leftIds.push_back(static_cast<int>(i));
            rightIds.push_back(matchR.centerMatches[i]);
        }

        auto reconR = reconOp.Execute(centersL, centersR, leftIds, rightIds, matchR.centerMatches);
        if (reconR.success) {
            for (auto& mr : reconR.markerResults) {
                result.markerPoints3d.emplace_back(
                    static_cast<float>(mr.centerX),
                    static_cast<float>(mr.centerY),
                    static_cast<float>(mr.centerZ));
                result.markerNormals.emplace_back(
                    static_cast<float>(mr.normalX),
                    static_cast<float>(mr.normalY),
                    static_cast<float>(mr.normalZ));
            }
        }
    } else {
        // 无标定参数：仅输出 2D 匹配结果（调试用）
        spdlog::debug("[MarkerStage] 无标定参数，跳过 3D 重建");
    }

    result.markerCount = static_cast<int>(matchR.centerMatches.size());
    result.success = true;

    spdlog::debug("[MarkerStage] 检测 {} 个标记点，匹配 {}，重建 {} 个3D点",
                  centersL.size(), matchR.centerMatches.size(), result.markerPoints3d.size());

    } catch (const std::exception& e) {
        spdlog::error("[MarkerStage] 异常: {}", e.what());
        return Result::fail(e.what());
    } catch (...) {
        spdlog::error("[MarkerStage] 未知异常");
        return Result::fail("未知异常");
    }

    return Result::ok();
}

// ============================================================================
// LaserStage — CUDA only，CPU 模式跳过
// ============================================================================
LaserStage::LaserStage(WorkflowContext* ctx, ScanMode mode)
    : Stage("Laser"), ctx_(ctx), mode_(mode) {}

Result LaserStage::process() {
    if (mode_ == ScanMode::MarkerOnly) return Result::ok("纯标记点模式");
    spdlog::debug("[LaserStage] 激光链需要 CUDA（BUILD_CUDA=ON），当前跳过");
    return Result::ok("Laser: CUDA required, skipped");
}

// ============================================================================
// FuseStage — 体素融合 → PointCloudBuffer
// ============================================================================
FuseStage::FuseStage(WorkflowContext* ctx)
    : Stage("Fuse"), ctx_(ctx) {
    fuseOp_ = std::make_unique<calib::LaserCloudFuseCPU>();
}

FuseStage::~FuseStage() = default;

void FuseStage::addPoints(const std::vector<cv::Point3f>& points,
                           const cv::Matx33d& R, const cv::Vec3d& T) {
    std::lock_guard lock(pointsMutex_);
    pendingPoints_.insert(pendingPoints_.end(), points.begin(), points.end());
    pendingR_ = R;
    pendingT_ = T;
}

Result FuseStage::process() {
    std::lock_guard lock(pointsMutex_);
    if (pendingPoints_.empty()) return Result::ok("无待融合点");

    try {
    // 体素融合（复用算子）
    auto fuseR = fuseOp_->Execute(pendingPoints_, pendingR_, pendingT_);

    if (fuseR.success && ctx_ && ctx_->pointCloudBuffer()) {
        data::PointCloudFrame cloud;
        cloud.points.assign(fuseR.survivingPoints.begin(), fuseR.survivingPoints.end());
        cloud.pointCount = static_cast<int>(fuseR.survivingPoints.size());
        ctx_->pointCloudBuffer()->pushPointCloud(cloud);
    }
    } catch (const std::exception& e) {
        spdlog::error("[FuseStage] 异常: {}", e.what());
    } catch (...) {
        spdlog::error("[FuseStage] 未知异常");
    }

    pendingPoints_.clear();
    return Result::ok();
}

// ============================================================================
// ScanWorkflow
// ============================================================================
ScanWorkflow::ScanWorkflow(WorkflowContext* ctx) : ctx_(ctx) {
    capture_    = std::make_unique<CaptureStage>(ctx_);
    preprocess_ = std::make_unique<PreprocessStage>(ctx_);
    marker_     = std::make_unique<MarkerStage>(ctx_);
    laser_      = std::make_unique<LaserStage>(ctx_, scanMode_);
    fuse_       = std::make_unique<FuseStage>(ctx_);
}

ScanWorkflow::~ScanWorkflow() { stop(); }

void ScanWorkflow::setCalibration(const ScanCalibration& calib) {
    calib_ = calib;
    marker_->setCalibration(calib);
}

Result ScanWorkflow::initialize() {
    if (!ctx_) return Result::fail("无 WorkflowContext");
    if (!ctx_->frameBuffer()) return Result::fail("无 FrameBuffer");
    spdlog::info("[ScanWorkflow] 初始化 (模式={}, 标定={})",
                 scanMode_ == ScanMode::MarkerOnly ? "纯标记点" : "标记点+激光",
                 calib_.valid ? "已加载" : "未加载");
    return Result::ok();
}

Result ScanWorkflow::start() {
    if (state_ == WorkflowState::Running) return Result::ok("已在运行");
    state_ = WorkflowState::Running;
    running_ = true;
    if (ctx_ && ctx_->session()) ctx_->session()->startSession();
    ctx_->publishEvent(EventType::ScanStarted);
    scanThread_ = std::thread(&ScanWorkflow::scanLoop, this);
    spdlog::info("[ScanWorkflow] 已启动");
    return Result::ok();
}

Result ScanWorkflow::pause() {
    if (state_ != WorkflowState::Running) return Result::fail("非运行状态");
    state_ = WorkflowState::Paused;
    if (ctx_ && ctx_->session()) ctx_->session()->pauseSession();
    return Result::ok();
}

Result ScanWorkflow::resume() {
    if (state_ != WorkflowState::Paused) return Result::fail("非暂停状态");
    state_ = WorkflowState::Running;
    if (ctx_ && ctx_->session()) ctx_->session()->resumeSession();
    return Result::ok();
}

Result ScanWorkflow::stop() {
    if (state_ == WorkflowState::Idle) return Result::ok();
    state_ = WorkflowState::Stopping;
    running_ = false;
    if (scanThread_.joinable()) scanThread_.join();
    state_ = WorkflowState::Completed;
    if (ctx_) {
        if (ctx_->session()) ctx_->session()->stopSession();
        ctx_->publishEvent(EventType::ScanStopped);
    }
    spdlog::info("[ScanWorkflow] 已停止");
    return Result::ok();
}

Result ScanWorkflow::setProgressCallback(WorkflowCallback cb) {
    callback_ = std::move(cb);
    return Result::ok();
}

void ScanWorkflow::scanLoop() {
    spdlog::info("[ScanWorkflow] 扫描循环启动");
    int frameCount = 0;

    while (running_.load()) {
        if (state_ == WorkflowState::Paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        try {
        // Stage 0: Capture
        auto r0 = capture_->process();
        if (!r0.success || r0.isDegraded()) continue;
        auto frame = capture_->getLatestFrame();
        if (frame.leftGray.empty() || frame.rightGray.empty()) continue;
        ++frameCount;

        // Stage 1: Preprocess（OpenCV 阈值掩码）
        preprocess_->setInput(frame);
        preprocess_->process();

        // Stage 2: Marker chain（复用算子对象）
        marker_->setInput(frame, preprocess_->leftMarkerMask, preprocess_->rightMarkerMask);
        marker_->process();

        // Stage 3: Laser (CUDA only, 跳过)
        // laser_->process();

        // Stage 4: Fuse
        if (marker_->result.success && !marker_->result.markerPoints3d.empty()) {
            fuse_->addPoints(marker_->result.markerPoints3d,
                             marker_->result.R, marker_->result.T);
        }
        fuse_->process();

        if (frameCount % 100 == 0) {
            spdlog::info("[ScanWorkflow] 已处理 {} 帧", frameCount);
        }

        } catch (const std::exception& e) {
            spdlog::error("[ScanWorkflow] 异常: {}", e.what());
        } catch (...) {
            spdlog::error("[ScanWorkflow] 未知异常");
        }
    }

    spdlog::info("[ScanWorkflow] 扫描循环结束 (处理 {} 帧)", frameCount);
}

void ScanWorkflow::notifyProgress(const std::string& stage, float progress) {
    if (!callback_) return;
    WorkflowProgress p;
    p.state = state_.load();
    p.currentStage = 4;
    p.totalStages = 5;
    p.stageName = stage;
    p.progress = progress;
    callback_(p);
}

} // namespace Scanner::workflow

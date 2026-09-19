// ============================================================================
// SimScanSource.cpp — 实现见头文件注释（260917 中段模拟提取版）
// ============================================================================
#include "pipelines/scan/SimScanSource.h"

#include "file_io.h"   // Scanner::data::fileio::importPLY（06；07 已链 mod_fileio）

#include <algorithm>
#include <cmath>
#include <utility>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

namespace Scanner::pipeline {

namespace {

constexpr double kPi = 3.14159265358979323846;

/// 轴角 → 3×3 旋转矩阵（行主序；轴内部归一化；避免 calib3d 依赖）
cv::Matx33d axisAngleRot(const cv::Vec3d& axis, double deg) {
    const double n = cv::norm(axis);
    if (n <= 0.0) return cv::Matx33d::eye();
    const double x = axis[0] / n, y = axis[1] / n, z = axis[2] / n;
    const double a = deg * kPi / 180.0;
    const double c = std::cos(a), s = std::sin(a), C = 1.0 - c;
    return {
        c + x * x * C,     x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C,     y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C};
}

} // namespace

// ============================================================================
// SimScene
// ============================================================================
SimScene SimScene::builtin(uint32_t seed) {
    SimScene s;
    // 标志点：6×5 网格 @30mm，中心原点——坐标全部钉在体素中心（+2.5 偏移：
    // marker 融合适配器 voxelSize=5mm，网格 30mm 恰对齐 5mm 边界，浮点舍入会在
    // 边界两侧翻 voxel 致点数翻倍——2026-09-13 实测 30→60 教训）
    constexpr int kCols = 6, kRows = 5;
    constexpr double kStep = 30.0;
    constexpr double kMid = 2.5;        // 5mm 体素的中心偏移
    s.markers.reserve(kCols * kRows);
    for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kCols; ++c) {
            calib::MarkerCloudPoint p;
            p.x = static_cast<float>((c - (kCols - 1) / 2.0) * kStep + kMid);
            p.y = static_cast<float>((r - (kRows - 1) / 2.0) * kStep + kMid);
            p.z = -402.5f;              // -400 恰为 5mm 边界——钉中心
            p.nx = 0.0f; p.ny = 0.0f; p.nz = 1.0f;
            p.whiteRadius = 1.5f;
            s.markers.push_back(p);
        }
    // 激光：200×200mm 平面片 z=-400.25——2.5mm 网格＋0.25 偏移，全坐标钉在
    // 0.5mm 体素中心（激光融合默认 voxelSize=0.5mm；σ=0 时可精确断言 M 点）
    constexpr int kGrid = 81;           // 81×81=6561（80 间隔 ×2.5mm=200mm）
    constexpr double kExtent = 100.0;   // 半宽 mm
    constexpr double kHalf = 2.5;       // 步距
    s.laser.reserve(kGrid * kGrid);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> jitter(-0.02, 0.02);  // << 0.25 体素半宽
    for (int i = 0; i < kGrid; ++i)
        for (int j = 0; j < kGrid; ++j) {
            const double x = -kExtent + 0.25 + kHalf * i;
            const double y = -kExtent + 0.25 + kHalf * j;
            const double z = -400.25;
            s.laser.emplace_back(
                static_cast<float>(x + jitter(rng)),
                static_cast<float>(y + jitter(rng)),
                static_cast<float>(z + jitter(rng)));
        }
    return s;
}

Scanner::Result SimScene::loadMarkersFromPly(const std::string& path, SimScene& out) {
    std::vector<cv::Point3f> pts;
    if (!Scanner::data::fileio::importPLY(path, pts)) {
        return Scanner::Result::fail("SimScene: 标志点 PLY 读入失败: " + path);
    }
    if (pts.empty())
        return Scanner::Result::fail("SimScene: 标志点 PLY 为空: " + path);
    out.markers.clear();
    out.markers.reserve(pts.size());
    for (const auto& p : pts) {
        calib::MarkerCloudPoint m;
        m.x = p.x; m.y = p.y; m.z = p.z;
        m.nx = 0.0f; m.ny = 0.0f; m.nz = 1.0f;   // PLY 无法线——缺省朝 +Z
        m.whiteRadius = 1.5f;
        out.markers.push_back(m);
    }
    JMW_LOG_INFO("07-SimScan", "[SimScene] 标志点 PLY 装载: {} 点 ← {}", pts.size(), path);
    return Scanner::Result::ok();
}

// ============================================================================
// SimScanSource
// ============================================================================
SimScanSource::SimScanSource(SimScene scene, SimTrajParams params)
    : scene_(std::move(scene)), params_(params), rng_(20260913) {
    if (params_.laserPerFrame == 0) params_.laserPerFrame = 1;
    laserIdx_.resize(scene_.laser.size());
    for (uint32_t i = 0; i < laserIdx_.size(); ++i) laserIdx_[i] = i;
    for (size_t i = laserIdx_.size(); i > 1; --i) {  // 全池洗牌（Fisher–Yates）：
        std::uniform_int_distribution<size_t> pick(0, i - 1);   // 游标环扫的池序
        std::swap(laserIdx_[i - 1], laserIdx_[pick(rng_)]);     // 均匀无周期
    }
    // 场景 y 包围盒＋平均深度（sweepY 行程与 FOV 窗宽估算）
    {
        bool first = true;
        double zSum = 0.0;
        size_t zn = 0;
        auto acc = [&](double y, double z) {
            if (first) { sceneYMin_ = sceneYMax_ = y; first = false; }
            else { sceneYMin_ = std::min(sceneYMin_, y); sceneYMax_ = std::max(sceneYMax_, y); }
            zSum += z; ++zn;
        };
        for (const auto& m : scene_.markers) acc(m.y, m.z);
        for (const auto& p : scene_.laser) acc(p.y, p.z);
        if (zn > 0) sceneMeanDepth_ = std::max(1.0, -(zSum / static_cast<double>(zn)));
    }
    // 递增调度用：标志点按 y 升序下标表（滑动窗沿扫描方向推进）
    markerOrder_.resize(scene_.markers.size());
    for (size_t i = 0; i < markerOrder_.size(); ++i) markerOrder_[i] = i;
    std::stable_sort(markerOrder_.begin(), markerOrder_.end(),
                     [this](size_t a, size_t b) {
                         return scene_.markers[a].y < scene_.markers[b].y;
                     });
    JMW_LOG_INFO("07-SimScan",
        "[SimScan] 场景: 标志点={} 激光={}（每帧≤{}）；轨迹: {:.2f}°/帧 +{:.2f}mm/帧, "
        "标志点σ={:.3f}mm 激光σ={:.3f}mm；R/T=生产链配准（观测直供）；"
        "视场 H{:.0f}°/V{:.0f}° 深[{:.0f},{:.0f}]mm 纵扫={}；上限 {} 帧",
        scene_.markers.size(), scene_.laser.size(), params_.laserPerFrame,
        params_.rotDegPerFrame, params_.transMmPerFrame, params_.markerNoiseSigmaMm,
        params_.jitterSigmaMm, params_.fovHalfHDeg, params_.fovHalfVDeg,
        params_.depthMinMm, params_.depthMaxMm, params_.sweepY, params_.maxFrames);
}

void SimScanSource::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    frame_ = 0;
    rng_.seed(20260913);
    for (uint32_t i = 0; i < laserIdx_.size(); ++i) laserIdx_[i] = i;
    for (size_t i = laserIdx_.size(); i > 1; --i) {  // 与 ctor 同序（可复现）
        std::uniform_int_distribution<size_t> pick(0, i - 1);
        std::swap(laserIdx_[i - 1], laserIdx_[pick(rng_)]);
    }
    laserCursor_ = 0;
}

void SimScanSource::lastTruePose(double R[9], double T[3]) const {
    std::lock_guard<std::mutex> lock(mtx_);
    for (int i = 0; i < 9; ++i) R[i] = trueR_[i];
    for (int i = 0; i < 3; ++i) T[i] = trueT_[i];
}

bool SimScanSource::next(SimFrameObs& out) {
    std::lock_guard<std::mutex> lock(mtx_);
    out.markerPositions.clear();
    out.markerNormals.clear();
    out.laser.clear();
    if (frame_ >= params_.maxFrames) return false;
    if (scene_.markers.empty() && scene_.laser.empty()) return false;
    const uint64_t k = frame_++;

    // G_k = T(k·t)·R(k·θ)：设备在全局系的真值位姿——仅用于生成观测与精度对照
    //（R/T 由生产链对标志点观测配准估计，本源不配准）
    const cv::Matx33d Rg = axisAngleRot(params_.rotAxis,
                                        params_.rotDegPerFrame * static_cast<double>(k));
    // 纵向扫描行程：视场窗中心沿 y 匀速扫过场景 y 包围盒——两端各留
    // 1/4 窗宽余量（首帧/末帧即见足量标志点；余量打破网格场景与体素边界
    // 的对齐——σ=0 精确断言不翻格）
    double ty = 0.0;
    if (params_.sweepY && params_.fovHalfVDeg > 0.0 && params_.maxFrames > 1) {
        const double margin = std::tan(params_.fovHalfVDeg * kPi / 180.0) *
                              sceneMeanDepth_ * 0.25;
        const double lo = sceneYMin_ - margin;
        const double hi = sceneYMax_ + margin;
        ty = lo + (hi - lo) * static_cast<double>(k) /
                      static_cast<double>(params_.maxFrames - 1);
    }
    const cv::Vec3d Tg(params_.transMmPerFrame * static_cast<double>(k), ty, 0.0);
    const cv::Matx33d RgT = Rg.t();               // G_k⁻¹ 的旋转部
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) trueR_[r * 3 + c] = Rg(r, c);
    trueT_[0] = Tg[0]; trueT_[1] = Tg[1]; trueT_[2] = Tg[2];

    // 视场观测窗判定（设备系：深度=−z；FOV=|x|,|y|≤深度·tan(半角)；0=不限）
    const double tanH = params_.fovHalfHDeg > 0.0
                            ? std::tan(params_.fovHalfHDeg * kPi / 180.0) : -1.0;
    const double tanV = params_.fovHalfVDeg > 0.0
                            ? std::tan(params_.fovHalfVDeg * kPi / 180.0) : -1.0;
    auto visible = [&](const cv::Vec3d& d) {
        const double depth = -d[2];
        if (params_.depthMinMm > 0.0 && depth < params_.depthMinMm) return false;
        if (params_.depthMaxMm > 0.0 && depth > params_.depthMaxMm) return false;
        if (tanH > 0.0 && std::abs(d[0]) > depth * tanH) return false;
        if (tanV > 0.0 && std::abs(d[1]) > depth * tanV) return false;
        return true;
    };

    // —— 标志点观测（递增调度/FOV 窗双模式）：m_dev = Rᵀ(m − T) + N(0,σ) ——
    // 递增调度（用户定版 260919）：滑动窗沿 y 序推进——帧0 观测 W0 个；每步
    // 前移 advance（与上一步重合 W-advance 个）、窗口缓增（8→9→10…逐帧增加）。
    // 步内窗口不动（设备微动 ~0.2mm << 配准匹配阈 2mm → 重合标志点全匹配），
    // 跨步进新标志点——由生产链光流配准用重合子集估 R/T 链入全局锚。
    std::normal_distribution<double> mNoise(0.0, params_.markerNoiseSigmaMm);
    const bool subsetMode = params_.markerStepFrames > 0 && !scene_.markers.empty();
    size_t winStart = 0, winSize = scene_.markers.size();
    uint64_t step = 0;
    if (subsetMode) {
        step = k / params_.markerStepFrames;
        winSize = std::min(params_.markerWindowStart +
                               static_cast<size_t>(step / std::max<size_t>(1, params_.markerGrowEvery)),
                           scene_.markers.size());
        const size_t maxStart = scene_.markers.size() - winSize;
        winStart = std::min(static_cast<size_t>(step) * params_.markerAdvancePerStep, maxStart);
    }
    out.markerPositions.reserve(winSize);
    out.markerNormals.reserve(winSize);
    for (size_t j = 0; j < scene_.markers.size(); ++j) {
        const auto& m = scene_.markers[subsetMode ? markerOrder_[winStart + j] : j];
        if (subsetMode && j >= winSize) break;          // 滑动窗出口
        const cv::Vec3d d = RgT * (cv::Vec3d(m.x, m.y, m.z) - Tg);
        if (!subsetMode && !visible(d)) continue;       // FOV 窗口径（旧模式）
        const cv::Vec3d n = RgT * cv::Vec3d(m.nx, m.ny, m.nz);
        out.markerPositions.emplace_back(d[0] + mNoise(rng_), d[1] + mNoise(rng_),
                                         d[2] + mNoise(rng_));
        out.markerNormals.push_back(n);
    }

    // —— 激光观测（区域门控＝**当帧可见标志点包围盒**±边距 ∩ 深度窗）：逐帧
    // 部分观测口径（260919 用户定版）：帧 k 采「当帧 N 个标志点覆盖区域」的点云
    // ——下一帧标志点子集部分重合（窗口缓移），经重合标志点配准入全局云，模拟
    // 真实手持扫描「扫到哪、哪块表面长出来」。无标志点帧不采激光（配准无凭据
    // ——与真机「配准失败帧不发点」语义一致）。真值变换＋逐点抖动：
    // p_dev = Rᵀ(p − T) + N(0,σ)
    std::normal_distribution<double> gauss(0.0, params_.jitterSigmaMm);
    if (!scene_.laser.empty() && !out.markerPositions.empty()) {
        // 可见标志点包围盒（设备系）±固定边距——**激光区域＝标志点区域**（260919
        // 用户口径：真实扫描中激光只打标志点锚定的区域，点云区域与标志点一致；
        // 持续增长由条带相位扫掠承担，不扩边距）。z 以盒中心 ±固定边距
        constexpr double kMarkerRegionMargin = 120.0;        // mm（覆盖区域外扩）
        constexpr double kMarkerRegionMarginZ = 250.0;       // mm（260919：100 时
                                                              // 深 relief 工件被裁
                                                              //=域受限原因之一）
        const double marginXY = kMarkerRegionMargin;
        cv::Vec3d bmin(out.markerPositions[0].x, out.markerPositions[0].y,
                       out.markerPositions[0].z);
        cv::Vec3d bmax = bmin;
        for (const auto& m : out.markerPositions) {
            bmin[0] = std::min(bmin[0], static_cast<double>(m.x));
            bmin[1] = std::min(bmin[1], static_cast<double>(m.y));
            bmin[2] = std::min(bmin[2], static_cast<double>(m.z));
            bmax[0] = std::max(bmax[0], static_cast<double>(m.x));
            bmax[1] = std::max(bmax[1], static_cast<double>(m.y));
            bmax[2] = std::max(bmax[2], static_cast<double>(m.z));
        }
        bmin[0] -= marginXY; bmax[0] += marginXY;
        bmin[1] -= marginXY; bmax[1] += marginXY;
        const double zMid = (bmin[2] + bmax[2]) / 2.0;
        bmin[2] = zMid - kMarkerRegionMarginZ;
        bmax[2] = zMid + kMarkerRegionMarginZ;

        const size_t M = scene_.laser.size();
        const size_t n = std::min(params_.laserPerFrame, M);
        // —— 激光线条带观测（260919 用户定版：点云呈一条条线状态增长＋**每帧
        //    3~4 万点**）：y 按行厚分桶取整行（行＝一条线），**条带相位随步扫掠**
        //    （行选偏移 +1/步——线逐条推进，区域全行覆盖后循环）。行数按密度
        //    定（本数据集 ~280 点/行 → 125 行×280≈3.5 万/帧；真机 25 线×~1.4k
        //    点/线同量级——观感同为条带群）。行厚 2.5mm 对齐 builtin 池网格
        constexpr int kLaserLinesPerFrame = 125;
        constexpr double kRowThickness = 2.5;   // mm
        // 相位按**帧**推进（260919：按步推进时步内 15 帧重采同 125 行、全为
        // 重复点去重=融合只涨几百。逐帧换行——每帧 ~3.5 万新点，线一条条扫过）
        const long stripePhase = subsetMode ? static_cast<long>(k % 4096) : 0;
        const long rowsInRegion =
            std::max(1L, static_cast<long>((bmax[1] - bmin[1]) / kRowThickness));
        const long stride = std::max(1L, rowsInRegion / kLaserLinesPerFrame);
        // 每帧扫描预算：须覆盖目标点量（行线筛选后命中率为区域占比——预算
        // = 15×目标，保证 3~4 万实收；小场景全池兜底）
        const size_t scanBudget =
            std::min(M, std::max(n * 15, static_cast<size_t>(65536)));
        out.laser.reserve(n);
        size_t got = 0;
        for (size_t t = 0; t < scanBudget && got < n; ++t) {
            const auto& p = scene_.laser[laserIdx_[(laserCursor_ + t) % M]];
            const cv::Vec3d d = RgT * (cv::Vec3d(p.x, p.y, p.z) - Tg);
            if (d[0] < bmin[0] || d[0] > bmax[0] ||
                d[1] < bmin[1] || d[1] > bmax[1] ||
                d[2] < bmin[2] || d[2] > bmax[2])
                continue;                          // 标志点覆盖区域外不入观测
            const long row = static_cast<long>(
                std::floor((d[1] - bmin[1]) / kRowThickness));
            if (((row + stripePhase) % stride) != 0) continue;   // 相位扫掠后的当帧线行
            if (subsetMode) {
                // 递增调度：观测判据=覆盖区域＋深度窗（设备近静，覆盖区可出
                // FOV 锥——真实口径是「扫到的区域即观测」）
                const double depth = -d[2];
                if (params_.depthMinMm > 0.0 && depth < params_.depthMinMm) continue;
                if (params_.depthMaxMm > 0.0 && depth > params_.depthMaxMm) continue;
            } else if (!visible(d)) {
                continue;                          // FOV+深度窗（旧口径）
            }
            out.laser.emplace_back(
                static_cast<float>(d[0] + gauss(rng_)),
                static_cast<float>(d[1] + gauss(rng_)),
                static_cast<float>(d[2] + gauss(rng_)));
            ++got;
        }
        laserCursor_ = (laserCursor_ + scanBudget) % M;
    }

    out.frameId = k;
    if (k % 100 == 0) {
        JMW_LOG_INFO("07-SimScan",
            "[SimScan] 帧 {}: 观测 markers={} laser={}（真值 Tg=({:.1f},{:.1f},{:.1f})）",
            k, out.markerPositions.size(), out.laser.size(),
            Tg[0], Tg[1], Tg[2]);
    }
    return true;
}

} // namespace Scanner::pipeline

// harness.cpp — 链装配＋执行＋计时＋dump 实现
//
// 接线依据：docs/plans/2026-10-04-scan-chain-replay-impl.md 算子接线速查表
// （全部 Execute 签名经算子头文件逐一核验；结果字段以 zitai_result_types.h 等为准）。
// 计时：GPU 步骤在计时块内 stream_ 同步（单流串行；非流水线模式，README 注明）；
// mask_separation 用算子内置 timings 覆盖（source=internal）。
#include "harness.h"

#include "fc_executor.h"   // 【五链 P2】fc_ 完整类型（析构/透传定义处）
	#include "gpu_accum.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "common/calib_logging.h"   // CALIB_LOG_WARN（批1 写库体检/配置告警）

CALIB_DEFINE_LOG_TAG(90, ScrHarness)   // 本翻译单元日志标签
#include "steger_dedup_merge.h"   // 去重变体差异件：dedupPath=cpu 时的外挂位移式去重（对照 oracle；默认 operator 路径算子内去重不走此件）

#include "ccl_cpu.h"              // 【150FPS·方案 A】CPU 连通域路径
#include "pipeline.h"             // 【150FPS·方案 D】跨帧流水线执行器

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

#include "artifacts.h"

namespace scr {
namespace fs = std::filesystem;

namespace {

double medianOf(std::vector<double> v) {
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double matMaxAbsDiff(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.size() != b.size() || a.type() != b.type())
        return std::numeric_limits<double>::infinity();
    cv::Mat d;
    cv::absdiff(a, b, d);
    double mx = 0.0;
    cv::minMaxIdx(d, nullptr, &mx);
    return mx;
}

cv::Rect componentRect(const calib::ComponentStats& cs) {
    return cv::Rect(cs.boundingBoxX, cs.boundingBoxY, cs.boundingBoxWidth,
                    cs.boundingBoxHeight);
}

// ConicFailReason 枚举名：单源 ConicFailReasonName（算子头导出；summary.json
// conicHistogram 键名用——禁本地复制名表）

// 5 参数 D → 12 参数 rational 形态（k4-k6=0）
void fillMarkerUndistortParams(calib::MarkerUndistortCPUParams& p, const cv::Mat& K1,
                               const cv::Mat& D1, const cv::Mat& K2, const cv::Mat& D2,
                               const cv::Mat& Rext, const cv::Mat& Text, int w, int h) {
    p.fx1 = K1.at<double>(0, 0);  p.fy1 = K1.at<double>(1, 1);
    p.cx1 = K1.at<double>(0, 2);  p.cy1 = K1.at<double>(1, 2);
    p.fx2 = K2.at<double>(0, 0);  p.fy2 = K2.at<double>(1, 1);
    p.cx2 = K2.at<double>(0, 2);  p.cy2 = K2.at<double>(1, 2);
    const double* d1 = D1.ptr<double>(0);
    const double* d2 = D2.ptr<double>(0);
    p.k1_1 = d1[0]; p.k2_1 = d1[1]; p.p1_1 = d1[2]; p.p2_1 = d1[3]; p.k3_1 = d1[4];
    p.k1_2 = d2[0]; p.k2_2 = d2[1]; p.p1_2 = d2[2]; p.p2_2 = d2[3]; p.k3_2 = d2[4];
    for (int i = 0; i < 9; ++i) p.R[i] = Rext.at<double>(i / 3, i % 3);
    for (int i = 0; i < 3; ++i) p.T[i] = Text.at<double>(i, 0);
    p.imageWidth = w;
    p.imageHeight = h;
}

// ==== 可视化绘图助手（dump 专用） ============================================

/// GpuMat 点集(CV_32FC2)安全下载（ROI 视图先物化）
std::vector<cv::Point2f> downloadP2(const std::shared_ptr<cv::cuda::GpuMat>& m,
                                    cv::cuda::Stream& s) {
    std::vector<cv::Point2f> out;
    if (!m) return out;
    try {
        cv::cuda::GpuMat c;
        m->copyTo(c, s);
        cv::Mat h;
        c.download(h, s);
        s.waitForCompletion();
        if (!h.empty() && h.type() == CV_32FC2)
            out.assign(h.begin<cv::Point2f>(), h.end<cv::Point2f>());
    } catch (...) {
    }
    return out;
}

/// GpuMat 整型数组(CV_32SC1)安全下载（线号/段 ID 等）
std::vector<int> downloadIds(const std::shared_ptr<cv::cuda::GpuMat>& m,
                             cv::cuda::Stream& s) {
    std::vector<int> out;
    if (!m) return out;
    try {
        cv::cuda::GpuMat c;
        m->copyTo(c, s);
        cv::Mat h;
        c.download(h, s);
        s.waitForCompletion();
        if (!h.empty() && h.type() == CV_32SC1)
            out.assign(h.begin<int>(), h.end<int>());
    } catch (...) {
    }
    return out;
}

/// 左右图并合为一张（中间 4px 白隔条；右图按左图尺寸对齐缩放）
cv::Mat mergeLR(const cv::Mat& lIn, const cv::Mat& rIn) {
    if (lIn.empty() && rIn.empty()) return cv::Mat();
    cv::Mat l = lIn, r = rIn;
    if (l.channels() == 1) cv::cvtColor(l, l, cv::COLOR_GRAY2BGR);
    if (r.channels() == 1) cv::cvtColor(r, r, cv::COLOR_GRAY2BGR);
    if (l.empty()) l = cv::Mat::zeros(r.size(), CV_8UC3);
    if (r.empty()) r = cv::Mat::zeros(l.size(), CV_8UC3);
    if (l.size() != r.size()) cv::resize(r, r, l.size());
    cv::Mat out(l.rows, l.cols + r.cols + 4, CV_8UC3, cv::Scalar(255, 255, 255));
    l.copyTo(out(cv::Rect(0, 0, l.cols, l.rows)));
    r.copyTo(out(cv::Rect(l.cols + 4, 0, r.cols, r.rows)));
    return out;
}

/// 按 id 生成稳定颜色（HSV 轮转，同 id 同色——配对/分组可视化用）
cv::Scalar paletteColor(int id) {
    if (id < 0) return cv::Scalar(40, 40, 40);
    const int hue = (id * 47) % 180;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Vec3b(static_cast<uchar>(hue), 220, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return cv::Scalar(bgr.at<cv::Vec3b>(0, 0)[0], bgr.at<cv::Vec3b>(0, 0)[1],
                      bgr.at<cv::Vec3b>(0, 0)[2]);
}

cv::Mat toBgr(const cv::Mat& gray) {
    cv::Mat bgr;
    if (gray.channels() == 1)
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    else
        bgr = gray.clone();
    return bgr;
}

void drawPts(cv::Mat& canvas, const std::vector<cv::Point2f>& pts,
             const cv::Scalar& color, int radius, int thickness = -1) {
    for (const auto& pt : pts) {
        const int x = cvRound(pt.x), y = cvRound(pt.y);
        if (x < -radius || y < -radius || x >= canvas.cols + radius
            || y >= canvas.rows + radius)
            continue;
        cv::circle(canvas, cv::Point(x, y), radius, color, thickness);
    }
}

// 8 连通栅格连通域：点集 → 每点段号（0 起始；compSize 回传各段点数）。
// cell 取值须小于相邻激光线间距（本数据线距 ~34px，取 8px 安全）。
std::vector<int> labelConnected(const std::vector<cv::Point2f>& pts, float cell,
                                std::vector<int>& compSize) {
    compSize.clear();
    const int n = static_cast<int>(pts.size());
    std::vector<int> label(n, -1);
    if (n == 0) return label;
    std::unordered_map<long long, std::vector<int>> cellPts;
    cellPts.reserve(static_cast<size_t>(n) * 2);
    for (int i = 0; i < n; ++i)
        cellPts[((long long)(pts[i].y / cell) << 24) | (long long)(pts[i].x / cell)]
            .push_back(i);
    for (const auto& kv : cellPts) {
        if (label[kv.second.front()] >= 0) continue;
        const int comp = static_cast<int>(compSize.size());
        compSize.push_back(0);
        std::vector<long long> stack{kv.first};
        while (!stack.empty()) {
            const long long k = stack.back();
            stack.pop_back();
            auto it = cellPts.find(k);
            if (it == cellPts.end() || label[it->second.front()] >= 0) continue;
            for (int idx : it->second) {
                label[idx] = comp;
                ++compSize[comp];
            }
            const long long cy = k >> 24, cx = k & 0xFFFFFF;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dy && !dx) continue;
                    stack.push_back(((cy + dy) << 24) | (cx + dx));
                }
        }
    }
    return label;
}

cv::Mat makeCollage(const std::vector<cv::Mat>& tiles, int cell, int maxTiles) {
    cv::Mat canvas;
    const size_t n = std::min(tiles.size(), static_cast<size_t>(maxTiles));
    if (n == 0) return canvas;
    const int cols = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))));
    const int rows = static_cast<int>((n + cols - 1) / cols);
    canvas.create(rows * cell, cols * cell, CV_8UC3);
    canvas.setTo(cv::Scalar(30, 30, 30));
    for (size_t i = 0; i < n; ++i) {
        cv::Mat t;
        if (tiles[i].channels() == 1) cv::cvtColor(tiles[i], t, cv::COLOR_GRAY2BGR);
        else t = tiles[i];
        cv::Mat sq;
        const int side = std::max(t.cols, t.rows);
        // 等比缩放进 cell，居中贴黑底方块
        const double scale = static_cast<double>(cell) / side;
        cv::resize(t, sq, cv::Size(), scale, scale);
        cv::Mat blk = cv::Mat::zeros(cell, cell, CV_8UC3);
        sq.copyTo(blk(cv::Rect((cell - sq.cols) / 2, (cell - sq.rows) / 2,
                               sq.cols, sq.rows)));
        blk.copyTo(canvas(cv::Rect(static_cast<int>(i) % cols * cell,
                                   static_cast<int>(i) / cols * cell, cell, cell)));
    }
    return canvas;
}

cv::Mat makeTrajectory(const std::vector<std::array<double, 3>>& traj, int w, int h) {
    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(20, 20, 20));
    if (traj.size() < 2) {
        cv::putText(canvas, "trajectory needs >=2 frames", cv::Point(10, h / 2), 0,
                    0.6, cv::Scalar(128, 128, 128));
        return canvas;
    }
    double xMn = 1e30, xMx = -1e30, zMn = 1e30, zMx = -1e30;
    for (const auto& t : traj) {
        xMn = std::min(xMn, t[0]); xMx = std::max(xMx, t[0]);
        zMn = std::min(zMn, t[2]); zMx = std::max(zMx, t[2]);
    }
    if (xMx - xMn < 1e-9) xMx = xMn + 1;
    if (zMx - zMn < 1e-9) zMx = zMn + 1;
    const int pad = 40;
    auto mapPt = [&](const std::array<double, 3>& t) {
        const double fx = (t[0] - xMn) / (xMx - xMn);
        const double fz = (t[2] - zMn) / (zMx - zMn);
        return cv::Point(pad + static_cast<int>(fx * (w - 2 * pad)),
                         h - pad - static_cast<int>(fz * (h - 2 * pad)));
    };
    for (size_t i = 1; i < traj.size(); ++i)
        cv::line(canvas, mapPt(traj[i - 1]), mapPt(traj[i]), cv::Scalar(0, 255, 0), 2);
    cv::circle(canvas, mapPt(traj.front()), 5, cv::Scalar(0, 0, 255), -1);
    cv::circle(canvas, mapPt(traj.back()), 5, cv::Scalar(255, 0, 0), -1);
    cv::putText(canvas, cv::format("X[%.0f..%.0f] Z[%.0f..%.0f] n=%zu",
                                   xMn, xMx, zMn, zMx, traj.size()),
                cv::Point(pad, 20), 0, 0.5, cv::Scalar(0, 255, 255));
    return canvas;
}

} // namespace

// ============================================================================
// detectRegularGrid — 规则网格板检测（2026-10-15 裁决：宁拒勿错）
// 判据：各点最近邻距离的变异系数（std/mean）< cvTol。规则阵列的最近邻间距恒等
// 于格距（噪声 ~0.5mm/100mm 格距 → CV≈0.005）；随机布点间距连续分布（本批实测
// CV≈0.2+）。阈值 0.04 与两类分布相距一个数量级，稳健。
// ============================================================================
bool detectRegularGrid(const std::vector<cv::Point3d>& pts, double cvTol) {
    const size_t n = pts.size();
    if (n < 8) return false;   // 样本不足不判（小子集易偶发整齐，实测唯一 CV<0.02 帧为 6 点）
    std::vector<double> nn(n);
    for (size_t i = 0; i < n; ++i) {
        double best = std::numeric_limits<double>::max();
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const double dx = pts[i].x - pts[j].x;
            const double dy = pts[i].y - pts[j].y;
            const double dz = pts[i].z - pts[j].z;
            const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d < best) best = d;
        }
        if (best <= 0.0) return true;   // 重合点（零距离）视作病态网格
        nn[i] = best;
    }
    double mean = 0.0;
    for (double v : nn) mean += v;
    mean /= static_cast<double>(n);
    if (mean <= 0.0) return false;
    double var = 0.0;
    for (double v : nn) var += (v - mean) * (v - mean);
    var /= static_cast<double>(n);
    const double cv = std::sqrt(var) / mean;
    return cv < cvTol;
}

// ============================================================================
// 构造/析构
// ============================================================================

// ============================================================================
// makeMarkerOps — 标记点链算子组装配（串行组与流水线 lane 组同参各建一份；
// 09 CPU 算子非线程安全，跨线程并行须独立实例）
// ============================================================================
Harness::MarkerOps Harness::makeMarkerOps() {
    MarkerOps ops;
    ops.splitOp_ = std::make_unique<calib::ImageSplitCPU>(
        calib::ImageSplitCPUParams::fromJson(cfg_.opParams("image_split")));
    ops.zernOp_ = std::make_unique<calib::ZernikeEdgeCPU>(
        calib::ZernikeEdgeCPUParams::fromJson(cfg_.opParams("zernike_edge")));
    ops.mergeOp_ = std::make_unique<calib::ImageMergeCPU>(
        calib::ImageMergeCPUParams::fromJson(cfg_.opParams("image_merge")));
    // undistort_cpu：Params 构造校验需合法 K/D/R/T；矫正矩阵注入温表档
    // R1/R2/P1/P2/Q（与激光链同源同温 → 双链矫正系一致，设计 §3.3）
    calib::MarkerUndistortCPUParams up;
    fillMarkerUndistortParams(up, calib_.K1c, calib_.D1, calib_.K2c, calib_.D2,
                              calib_.Rext, calib_.Tcomp, calib_.imageWidth,
                              calib_.imageHeight);
    ops.undistCpuOp_ = std::make_unique<calib::MarkerUndistortCPU>(up);
    ops.undistCpuOp_->SetRectifyMatrices(calib_.R1, calib_.R2, calib_.P1, calib_.P2,
                                         calib_.Q);
    ops.ellOp_ = std::make_unique<calib::EllipseFitCPU>(
        calib::EllipseFitCPUParams::fromJson(cfg_.opParams("ellipse_fit")));
    ops.mmOp_ = std::make_unique<calib::MarkerMatchCPU>(
        calib::MarkerMatchCPUParams::fromJson(cfg_.opParams("marker_match")));
    // conic_reconstruct：矫正系 K/C2 由标定档派生：K1=P1 左 3×3；K2=P2 左 3×3；
    // t2=P2 第 4 列；C2=−K2⁻¹·t2（口径与 conic_offline.cpp 同段一致）
    ops.conicOp_ = std::make_unique<calib::ConicReconstructCPU>(
        calib::ConicReconstructCPUParams::fromJson(cfg_.opParams("conic_reconstruct")));
    cv::Matx33d cK1 = cv::Matx33d::eye(), cK2 = cv::Matx33d::eye();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            cK1(r, c) = calib_.P1.at<double>(r, c);
            cK2(r, c) = calib_.P2.at<double>(r, c);
        }
    const cv::Vec3d cT2(calib_.P2.at<double>(0, 3), calib_.P2.at<double>(1, 3),
                        calib_.P2.at<double>(2, 3));
    const cv::Vec3d cC2 = -(cK2.inv() * cT2);   // P2=K2[I|−C2] → C2=−K2⁻¹·t2
    ops.conicOp_->SetProjectionMatrices(cK1, cK2, cC2);
    return ops;
}

void Harness::LaserOps::destroy() {
#define SCR_LO_DESTROY(op) if (op) (op)->Destroy()
    SCR_LO_DESTROY(stegerL_);     SCR_LO_DESTROY(stegerR_);
    SCR_LO_DESTROY(flatL_);       SCR_LO_DESTROY(flatR_);
    SCR_LO_DESTROY(undistCudaL_); SCR_LO_DESTROY(undistCudaR_);
    SCR_LO_DESTROY(interpL_);     SCR_LO_DESTROY(interpR_);
    SCR_LO_DESTROY(matchOp_);     SCR_LO_DESTROY(reconOp_);
#undef SCR_LO_DESTROY
}

// 【利用率专项】激光链算子组装配（双收尾 lane 双实例）；laneIdx=0 打印注入
// 日志（stdout 口径与原单实例一致），>0 静默
Harness::LaserOps Harness::makeLaserOps(int laneIdx, std::string& err) {
    LaserOps lo;
    auto stgParams = calib::StegerParams::fromJson(cfg_.opParams("steger_fast"));
    if (dedupPath_ == "cpu") {
        stgParams.dedupMode = 0;
        if (laneIdx == 0)
            std::cout << "[dedup:path] cpu（外挂对照 oracle；算子 dedupMode 已强制 0）"
                      << std::endl;
    } else if (extractor_ != "flat" && laneIdx == 0) {
        std::cout << "[dedup:path] operator（算子内 GPU 去重，dedupMode="
                  << stgParams.dedupMode << "）" << std::endl;
    }
    lo.stegerL_ = std::make_unique<calib::StegerExtractorFast>(stgParams);
    lo.stegerR_ = std::make_unique<calib::StegerExtractorFast>(stgParams);
    if (extractor_ == "flat") {
        auto fparams = calib::FlatCenterParams::fromJson(cfg_.opParams("flat_center"));
        lo.flatL_ = std::make_unique<calib::FlatCenterExtractor>(fparams);
        lo.flatR_ = std::make_unique<calib::FlatCenterExtractor>(fparams);
    }

    calib::UndistortPointsParams ul, ur;
    ul.cameraMatrix = calib_.K1c; ul.distCoeffs = calib_.D1;
    ul.R = calib_.R1; ul.P = calib_.P1;
    ur.cameraMatrix = calib_.K2c; ur.distCoeffs = calib_.D2;
    ur.R = calib_.R2;
    // 右目 P 的 (0,3)=-B·f 平移项必须置零：激光链（interp/v3 查表/recon 的
    // d=fB/Z）消费“右目自身列坐标”约定（CMTT 表 uR 域 [0..1850] 实证；Q14=1/B
    // 自洽验证）。保留平移项会使右目 x 偏移 -17.1 万像素 → v3 全排除。
    ur.P = calib_.P2.clone();
    ur.P.at<double>(0, 3) = 0.0;
    lo.undistCudaL_ = std::make_unique<calib::UndistortPointsCuda>(ul);
    lo.undistCudaR_ = std::make_unique<calib::UndistortPointsCuda>(ur);

    auto ip = calib::EpipolarInterpDualParams::fromJson(
        cfg_.opParams("epipolar_interp_dual"));
    ip.mode = calib::InterpMode::Scan;   // 强制扫描模式（设计 §4 模式显式化）
    lo.interpL_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);
    lo.interpR_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);

    lo.matchOp_ = std::make_unique<calib::LaserMatchScanCudaV3>(
        calib::LaserMatchScanParamsV3::fromJson(
            cfg_.opParams("laser_match_scan_v3")));
    // 双管温表（奇偶＝管号；缺省两管共表）——表指针共享（只读），注备上传
    // 为实例私有（第二 lane 首注入时各自上传 140MB×2，一次性）
    if (!lo.matchOp_->SetTempTable(tableOdd_)) {
        err = "laser_match_scan_v3 SetTempTable 失败";
        return lo;
    }
    // uR 偏差补偿（urBiasModel 4 参数优先；否则回退 urBias 逐线数组）
    {
        const auto& v3node = cfg_.opParams("laser_match_scan_v3");
        bool modelSet = false;
        if (v3node.contains("urBiasModel") && v3node.at("urBiasModel").is_object()) {
            const auto& m = v3node.at("urBiasModel");
            const float wx = m.value("wx", 0.0f);
            const float wy = m.value("wy", 0.0f);
            const float wz = m.value("wz", 0.0f);
            const float dtx = m.value("dtx", 0.0f);
            lo.matchOp_->SetuRBiasModel(wx, wy, wz, dtx);
            if (laneIdx == 0)
                std::cout << "[calib] uR 4参数偏差模型已注入: wx=" << wx
                          << " wy=" << wy << " wz=" << wz << " dtx=" << dtx
                          << "（dtx 为占位，一阶无贡献）" << std::endl;
            modelSet = true;
        }
        if (!modelSet && v3node.contains("urBias") && v3node.at("urBias").is_array()
            && !v3node.at("urBias").empty()) {
            std::vector<float> bias;
            for (const auto& b : v3node.at("urBias"))
                bias.push_back(b.get<float>());
            lo.matchOp_->SetuRBias(bias);
            if (laneIdx == 0)
                std::cout << "[calib] uR 逐线补偿已注入: " << bias.size()
                          << " lids" << std::endl;
        }
    }

    lo.reconOp_ = std::make_unique<calib::LaserReconstructCuda>(
        calib::LaserReconstructParams::fromJson(cfg_.opParams("laser_reconstruct")));
    return lo;
}

Harness::Harness(const ScrConfig& cfg, const CalibData& calib, const FrameSource& src,
                 std::string& err)
    : cfg_(cfg), calib_(calib), src_(&src) {
    // 【Z1·2026-10-23 二次修正】OpenCV 全局线程池：默认 setNumThreads(0)＝
    // 彻底顺序执行（零 parallel_for 派发税）。语义勘误：OpenCV 文档口径
    // 0＝禁用线程优化全顺序（旧注释「'0'＝恢复默认核数」系误读——不调用
    // setNumThreads 才是默认多线程池，实测最差：每 ROI Gaussian/Canny 全场
    // 派工与绑核流水线线程抢核，wallMk 4.8→7.6ms/帧）。三档实测（ABBA）：
    // 0 顺序＝最优 4.78ms｜1 池单工＝5.31ms（旧默认）｜不设帽多线程＝7.6ms。
    // ch2 路径无 cv 并行调用（间接收益机制见 测量记录-绑核核查与环节耗时-
    // 2026-10-23.md）。主线程单点调用（非线程安全），构造先于一切 Execute。
    // 逃生门：JMW_OCV_THREADS=N（N≥1 池线程数；未设＝0 顺序）。
    if (const char* z1n = std::getenv("JMW_OCV_THREADS")) {
        cv::setNumThreads(std::max(0, std::atoi(z1n)));
    } else {
        cv::setNumThreads(0);
    }
    ensureGrayLanes(1);   // 【利用率专项】串行/单 lane d_gray [0] 就位
    outDir_ = cfg_.paths.outDir;
    try {
        // —— 同帧掩膜分离（一次执行，双链分叉消费）——
        // 2026-10-04 算子细激光形态重设计：腐蚀级联 3/9/37＋双 top-hat 减法
        // 原生出双掩膜（宽圆斑直接进 marking）——历史 swapOutputs 组合掩膜补丁退役
        auto sepParams = calib::LaserMarkingSeparationParams::fromJson(
            cfg_.opParams("mask_separation"));
        sepLaserL_ = std::make_unique<calib::LaserMarkingSeparationCUDA>(sepParams);
        sepLaserR_ = std::make_unique<calib::LaserMarkingSeparationCUDA>(sepParams);

        const auto cclParams =
            calib::RegionAnalyzerParams::fromJson(cfg_.opParams("ccl"));
        cclL_ = std::make_unique<calib::RegionAnalyzerCUDA>(cclParams);
        cclR_ = std::make_unique<calib::RegionAnalyzerCUDA>(cclParams);
        // 【150FPS·方案 A】CCL 路径开关（cpu=默认；流水线模式强制 cpu）
        cclPath_ = cfg_.runtime.cclPath;
        cclMinArea_ = cclParams.minArea;
        cclMaxArea_ = cclParams.maxArea;
        cclTopX_ = cclParams.topXCount;
        cudaEventCreateWithFlags(&evSepDone_, cudaEventDisableTiming);
        std::cout << "[ccl:path] " << cclPath_
                  << (cclPath_ == "cpu" ? "（RLE 并查集 CPU 路径）" : "（GPU 路径）")
                  << std::endl;

        // —— 标记点链算子组（串行组；流水线 lane 组同参另建）——
        serialOps_ = makeMarkerOps();

        flowOp_ = std::make_unique<calib::MarkerOpticalFlowFuseCPU>(
            calib::MarkerOpticalFlowFuseCPUParams::fromJson(
                cfg_.opParams("optical_flow_fuse")));
        // 配置可见性（批1 P0）：frame_fuse 节缺失＝全默认（matchMode 回 0＝PFH）
        // 静默降级——必须告警；生效口径启动即打印
        if (!cfg_.operators.contains("frame_fuse") && cfg_.runtime.markerOnly == false)
            std::cout << "[fuse:WARN] config 缺 operators.frame_fuse 节——算子全默认参数"
                         "（matchMode=0 PFH），共面稀疏形态大概率失败！请补配置"
                      << std::endl;
        fuseParams_ = calib::FrameFuseCPUParams::fromJson(cfg_.opParams("frame_fuse"));
        fuseOp_ = std::make_unique<calib::FrameFuseCPU>(fuseParams_);
        std::cout << "[fuse] matchMode=" << fuseParams_.matchMode
                  << (fuseParams_.matchMode == 1 ? "(三角形)" : "(PFH)")
                  << " 子集上限=" << std::min(fuseParams_.triangleMaxPoints,
                                              cfg_.runtime.fuseSubsetMax)
                  << " 验收门 matched>=" << cfg_.runtime.fuseMinMatched
                  << " rmse<" << cfg_.runtime.fuseMaxRmseMm << "mm"
                  << std::endl;

        dedupPath_ = cfg_.runtime.dedupPath;
        extractor_ = cfg_.runtime.extractor;   // steger（默认）｜flat（flat_center）
        // 双管温表（奇偶＝管号；不同管不同标定参数；缺省两管共表）——先赋表
        // 指针再建算子组（组内 SetTempTable 注备上传）
        tableOdd_ = calib_.laserTable;
        tableEven_ = calib_.laserTableEven ? calib_.laserTableEven : calib_.laserTable;
        // —— 激光链算子组（串行组；流水线收尾 lane 组同参另建）——
        laserOps_ = makeLaserOps(0, err);
        if (!err.empty()) return;

        // rowStep 一致性检查（温表档 vs 插值算子参数）
        {
            const auto ip = calib::EpipolarInterpDualParams::fromJson(
                cfg_.opParams("epipolar_interp_dual"));
            std::string rsMsg;
            const int rc = checkRowStep(calib_.laserRowStep, ip.epipolar_row_step, rsMsg);
            std::cout << "[rowStep] " << rsMsg << std::endl;
            if (rc < 0) {
                err = rsMsg;
                return;
            }
        }
        err.clear();
        // 【五链 P1】产线体素融合算子装配（0.25mm 去重；替代 gpuTransformAppend 原始累积）
        {
            calib::LaserCloudFuseCUDAParams fp;
            fp.voxelSize = static_cast<float>(cfg_.fivechain.voxelSizeMm);
            fp.reserveVoxelCount = static_cast<size_t>(cfg_.fivechain.voxelReserve);
            fp.growSlotCount = static_cast<size_t>(cfg_.fivechain.voxelGrowSlots);
            fp.collectStatistics = true;
            fp.validate();
            voxelFuseOp_ = std::make_unique<calib::LaserCloudFuseCuda>(fp);
            voxelFuseOp_->Clear();   // A/B 重跑保险：断言净空起步
            if (voxelFuseOp_->GetVoxelCount() != 0) {
                err = "体素融合算子初始状态非空";
                return;
            }
            std::cout << "[fuse:voxel] size=" << cfg_.fivechain.voxelSizeMm << "mm"
                      << " reserve=" << cfg_.fivechain.voxelReserve
                      << " grow=" << cfg_.fivechain.voxelGrowSlots << std::endl;
        }
    } catch (const std::exception& e) {
        err = std::string("Harness 构造异常: ") + e.what();
    }
}

Harness::~Harness() {
    fc_.reset();     // 【五链 P2】先停五链线程（join）再 Destroy 算子
    pipe_.reset();   // 【150FPS】先停流水线线程（join）再 Destroy 算子——防在飞帧访问已毁实例
    laserOps_.destroy();
    for (auto& lo : laserLanes_) lo.destroy();   // 【利用率专项】收尾 lane 组
#define SCR_DESTROY(op) if (op) (op)->Destroy()
    SCR_DESTROY(sepLaserL_);   SCR_DESTROY(sepLaserR_);
    SCR_DESTROY(cclL_);        SCR_DESTROY(cclR_);
    SCR_DESTROY(voxelFuseOp_);   // 【五链 P1】体素融合算子
#undef SCR_DESTROY
    if (evSepDone_) { cudaEventDestroy(evSepDone_); evSepDone_ = nullptr; }
    // CPU 算子 Destroy 可选（析构自清理），不强制
}

// ============================================================================
// 计时
// ============================================================================

// 【双链并行】计时基点线程感知：主线程（激光链）用成员 t0_；标记点链工作线程
// 用 thread_local 基点——两链并行计时互不踩踏。
// 【事件计时制】gpu=true 的步骤：不同步，只在流上记事件（帧尾统一算）。
static thread_local std::chrono::steady_clock::time_point g_t0Worker_{};

cudaEvent_t Harness::acquireEvent() {
    cudaEvent_t e = nullptr;
    cudaEventCreate(&e);
    return e;
}

void Harness::timeBegin() {
    if (std::this_thread::get_id() == mainTid_) {
        t0_ = std::chrono::steady_clock::now();
        // GPU 步骤起始事件（主流；gpu=false 步骤不用它也无害——事件轻量）
        if (evPendingStart_ == nullptr) evPendingStart_ = acquireEvent();
        cudaEventRecord(evPendingStart_,
                        cv::cuda::StreamAccessor::getStream(stream_));
    } else {
        g_t0Worker_ = std::chrono::steady_clock::now();
    }
}

double Harness::timeEnd(const char* step, CycleProducts& p, bool record, bool gpu) {
    const bool isMain = (std::this_thread::get_id() == mainTid_);
    if (gpu && isMain) {
        // 【事件计时制】只记结束事件——不同步；帧尾 flushStepEvents 统一算时长。
        cudaEvent_t evEnd = acquireEvent();
        cudaEventRecord(evEnd, cv::cuda::StreamAccessor::getStream(stream_));
        stepEvents_.push_back({step, evPendingStart_, evEnd});
        evPendingStart_ = nullptr;   // 下次 timeBegin 再起
        return 0.0;                  // 时长帧尾补记（此处仅占位）
    }
    const auto base = isMain ? t0_ : g_t0Worker_;
    const auto d = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - base).count();
    if (record) {
        // 【利用率专项】护栏：多线程并发写同一 p 的容器（标记 lane/激光收尾/
        // GPU lane 并行计时）——无锁并发 emplace_back＝堆损坏（实测 cycle~95 AV）
        std::lock_guard<std::mutex> lk(stepRecMtx_);
        p.stepMs.emplace_back(step, d);
        p.stepSource[step] = "external";
    }
    return d;
}

void Harness::flushStepEvents(CycleProducts& p, bool record) {
    if (!stepEvents_.empty()) {
        stream_.waitForCompletion();   // 帧尾一次同步
        for (auto& se : stepEvents_) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, se.evStart, se.evEnd);
            cudaEventDestroy(se.evStart);
            cudaEventDestroy(se.evEnd);
            if (record) {
                p.stepMs.emplace_back(se.step, static_cast<double>(ms));
                p.stepSource[se.step] = "event";
            }
        }
        stepEvents_.clear();
    }
    if (evPendingStart_) { cudaEventDestroy(evPendingStart_); evPendingStart_ = nullptr; }
}

void Harness::fail(const char* step, CycleProducts& p, const std::string& msg) const {
    {
        std::lock_guard<std::mutex> lk(stepRecMtx_);   // 【利用率专项】跨线程写护栏
        p.stepError[step] = msg;
        p.stepSuccess[step] = false;
    }
    std::cerr << "[harness] step " << step << " FAIL: " << msg << std::endl;
}

void Harness::setStepOk(CycleProducts& p, const char* step, bool ok) {
    std::lock_guard<std::mutex> lk(stepRecMtx_);
    p.stepSuccess[step] = ok;
}

bool Harness::stepOk(const CycleProducts& p, const char* step) const {
    std::lock_guard<std::mutex> lk(stepRecMtx_);
    const auto it = p.stepSuccess.find(step);
    return it != p.stepSuccess.end() && it->second;
}

// ============================================================================
// 同帧分离（一次执行，双链分叉消费）
// ============================================================================

void Harness::ensureGrayLanes(int lanes) {
    if (d_grayLanesL_.size() < static_cast<size_t>(lanes)) {
        d_grayLanesL_.resize(static_cast<size_t>(lanes));
        d_grayLanesR_.resize(static_cast<size_t>(lanes));
    }
}

void Harness::runSeparation(const cv::Mat& grayL, const cv::Mat& grayR,
                            CycleProducts& p, bool record, int laneIdx) {
    try {
        timeBegin();
        // 串行 L→R；主机侧计时诊断（定位提交开销）
        const auto hL0 = std::chrono::steady_clock::now();
        // 【回退双流】WDDM 下双流掩膜分离实测严重劣化（驱动流切换开销 >> 重叠收益）
        // 【速度优化】帧上传一次、双消费（掩膜＋steger 共享 d_gray）——
        // 免掩膜算子内部二次 H2D（原每帧每眼多传 2.36MB）
        // 【利用率专项】上传目标＝本 lane 独立 d_gray（跨 lane 解耦）
        ensureGrayLanes(laneIdx + 1);
        auto& gL = d_grayLanesL_[static_cast<size_t>(laneIdx)];
        auto& gR = d_grayLanesR_[static_cast<size_t>(laneIdx)];
        gL.upload(grayL, stream_);
        gR.upload(grayR, stream_);
        p.sepL = sepLaserL_->Execute(gL, stream_);
        p.sepR = sepLaserR_->Execute(gR, stream_);
        // 【150FPS·方案 A】分离完成事件：CPU CCL 下载掩膜只拦到这里（不等激光链）
        cudaEventRecord(evSepDone_, cv::cuda::StreamAccessor::getStream(stream_));
        const auto hL1 = std::chrono::steady_clock::now();
        if (getenv("JMW_SEPTIME"))
            fprintf(stderr, "[sepH] wall: %.2f ms\n",
                    std::chrono::duration<double, std::milli>(hL1 - hL0).count());
        setStepOk(p, "mask_separation", p.sepL.success && p.sepR.success);
        timeEnd("mask_separation", p, record, true);
        if (!stepOk(p, "mask_separation")) {
            stream_.waitForCompletion();   // 失败路径同步后再取消息
            fail("mask_separation", p, p.sepL.message + " | " + p.sepR.message);
            return;
        }
    } catch (const std::exception& e) {
        fail("mask_separation", p, e.what());
        return;
    }
}

// ============================================================================
// 激光链（吃同帧 d_laserMask）
// ============================================================================

void Harness::runLaserChain(CycleProducts& p, bool record, bool tubeOdd) {
    runLaserSteger(laserOps_, p, record, tubeOdd, 0, stream_);
    runLaserTail(laserOps_, p, record, tubeOdd, stream_);
}

// ============================================================================
// 激光链·steger 段（【利用率专项】收尾线程拆分·前半）：steger＋去重＋中心点检查
// ============================================================================

void Harness::runLaserSteger(LaserOps& L, CycleProducts& p, bool record, bool tubeOdd, int laneIdx, cv::cuda::Stream& ls) {
    (void)tubeOdd;
    ensureGrayLanes(laneIdx + 1);
    auto& gL = d_grayLanesL_[static_cast<size_t>(laneIdx)];
    auto& gR = d_grayLanesR_[static_cast<size_t>(laneIdx)];
    if (!p.sepL.d_laserMask || p.sepL.d_laserMask->empty()) {
        fail("steger_fast", p, "d_laserMask 为空（本帧激光线未分离出）");
        return;
    }

    const cv::Mat& grayL = p.frameGrayL;
    const cv::Mat& grayR = p.frameGrayR;
    (void)grayL; (void)grayR;   // 已在 runSeparation 上传（d_grayL_/d_grayR_ 共享）
    try {
        timeBegin();
        // 【速度优化·三轮结论】双流 steger 实测 WDDM 下严重劣化（帧 wall 从 116→470ms，
        // 驱动流切换开销远超重叠收益）——保持单流串行，steger 提速走算子内核优化
        // 【速度优化·四轮】非导出帧跳过 steger host 输出（省 0.5MB D2H＋同步/眼）
        // 【flat_center 顺带修复】cpu oracle 需 host 点（steger_dedup::apply 读
        // r.centerPoints）；operator 路径维持 false 保四轮优化——按帧条件置位
        const bool hostPts = (dedupPath_ == "cpu");
        L.stegerL_->SetCollectHostPoints(hostPts);
        L.stegerR_->SetCollectHostPoints(hostPts);
        if (extractor_ == "flat") {
            if (hostPts) {
                // 对照 oracle 需 host 点——原单眼同步路径（延迟版不支持 host 输出）
                p.stegerL = L.flatL_->Execute(gL, *p.sepL.d_laserMask, ls,
                                               calib::GroupMode::Flat);
                p.stegerR = L.flatR_->Execute(gR, *p.sepR.d_laserMask, ls,
                                               calib::GroupMode::Flat);
            } else {
                // 【flat 优化·合并唤醒】双眼连续提交、共用一次流同步——省 1 次
                // 计数回读往返（唤醒）＋右眼内核在宿主处理左眼结果时已在跑
                p.stegerL = L.flatL_->ExecuteDeferred(gL, *p.sepL.d_laserMask, ls);
                p.stegerR = L.flatR_->ExecuteDeferred(gR, *p.sepR.d_laserMask, ls);
                ls.waitForCompletion();
                p.gpuDrained = true;   // 【E3】五链拷出免事件同步（pipe 不消费——见字段注释）
                L.flatL_->FinishDeferred(p.stegerL);
                L.flatR_->FinishDeferred(p.stegerR);
            }
        } else {
            p.stegerL = L.stegerL_->Execute(gL, *p.sepL.d_laserMask, ls,
                                          calib::GroupMode::Flat);
            p.stegerR = L.stegerR_->Execute(gR, *p.sepR.d_laserMask, ls,
                                          calib::GroupMode::Flat);
        }
        setStepOk(p, "steger_fast", p.stegerL.success && p.stegerR.success);
        timeEnd("steger_fast", p, record, false);
        if (!stepOk(p, "steger_fast")) {
            fail("steger_fast", p, p.stegerL.message + " | " + p.stegerR.message);
            return;
        }
        // —— 去重路径开关（runtime.dedupPath）：
        //    operator → 算子内已完成（打印 mergedCount，dedupMode 照 config）；
        //    cpu → 外挂 CPU 位移式合并（对照 oracle；算子 dedupMode 已被构造期置 0）——
        if (dedupPath_ == "cpu") {
            timeBegin();
            std::string dmsgL, dmsgR;
            bool okDedup = steger_dedup::apply(p.stegerL, ls, dmsgL);
            okDedup = steger_dedup::apply(p.stegerR, ls, dmsgR) && okDedup;
            (void)timeEnd("steger_dedup", p, record, false);
            // [dedup:cpu] 日志已降噪
            setStepOk(p, "steger_dedup", okDedup);
            if (!okDedup) {
                fail("steger_dedup", p, dmsgL + " | " + dmsgR);
                return;
            }
        } else {
            // [dedup:operator] 日志已降噪（速度优化）
        }
        if (!p.stegerL.d_centerPoints || !p.stegerR.d_centerPoints
            || p.stegerL.d_centerPoints->empty() || p.stegerR.d_centerPoints->empty()) {
            fail("undistort_cuda", p,
                 "steger 0 中心点（激光掩膜为空或非线形态——检查 mask_separation "
                 "参数：laserKill/markKill 选档）L="
                     + std::to_string(p.stegerL.totalPointCount) + " R="
                     + std::to_string(p.stegerR.totalPointCount));
            return;
        }
    } catch (const std::exception& e) {
        fail("steger_fast", p, e.what());
        return;
    }
}

// ============================================================================
// 激光链·收尾段（【利用率专项】收尾线程拆分·后半）：undist→interp→v3→recon
// ＋行残差统计——算子组仅收尾线程触碰（天然串行），与 GPU lane 上传/分离经
// laserFed 闸门＋单流 FIFO 保序
// ============================================================================

// ============================================================================
// 【E4/E5·五链】激光前段提交/完成对——flat(延迟)→undist(容量)→interp(延迟)
// 提交段零宿主同步；完成段在调用方排空流后调用（读 pinned 计数建精确 ROI）
// ============================================================================

int Harness::submitLaserFront(LaserOps& L, CycleProducts& p, bool record, bool tubeOdd,
                              int laneIdx, cv::cuda::Stream& ls) {
    (void)tubeOdd;
    ensureGrayLanes(laneIdx + 1);
    auto& gL = d_grayLanesL_[static_cast<size_t>(laneIdx)];
    auto& gR = d_grayLanesR_[static_cast<size_t>(laneIdx)];
    if (!p.sepL.d_laserMask || p.sepL.d_laserMask->empty()) {
        fail("steger_fast", p, "d_laserMask 为空（本帧激光线未分离出）");
        return 0;
    }
    try {
        timeBegin();
        const bool hostPts = (dedupPath_ == "cpu");
        bool deferred = false;
        if (extractor_ == "flat" && !hostPts) {
            // 【E5】延迟提交：双眼连续入队，stats 28B D2H 在飞——不同步
            p.stegerL = L.flatL_->ExecuteDeferred(gL, *p.sepL.d_laserMask, ls);
            p.stegerR = L.flatR_->ExecuteDeferred(gR, *p.sepR.d_laserMask, ls);
            deferred = true;
            timeEnd("steger_fast", p, record, false);   // 提交墙钟（语义注记）
            if (!p.stegerL.success || !p.stegerR.success) {
                setStepOk(p, "steger_fast", false);
                fail("steger_fast", p, p.stegerL.message + " | " + p.stegerR.message);
                return 0;
            }
            setStepOk(p, "steger_fast", true);   // 提交级；N/零点在完成段定
        } else {
            // legacy：对照 oracle（flat+cpu）或 steger 提取器——内部自带同步
            if (extractor_ == "flat") {
                p.stegerL = L.flatL_->Execute(gL, *p.sepL.d_laserMask, ls,
                                               calib::GroupMode::Flat);
                p.stegerR = L.flatR_->Execute(gR, *p.sepR.d_laserMask, ls,
                                               calib::GroupMode::Flat);
            } else {
                L.stegerL_->SetCollectHostPoints(hostPts);
                L.stegerR_->SetCollectHostPoints(hostPts);
                p.stegerL = L.stegerL_->Execute(gL, *p.sepL.d_laserMask, ls,
                                                calib::GroupMode::Flat);
                p.stegerR = L.stegerR_->Execute(gR, *p.sepR.d_laserMask, ls,
                                                calib::GroupMode::Flat);
            }
            setStepOk(p, "steger_fast", p.stegerL.success && p.stegerR.success);
            timeEnd("steger_fast", p, record, false);
            if (!stepOk(p, "steger_fast")) {
                fail("steger_fast", p, p.stegerL.message + " | " + p.stegerR.message);
                return 0;
            }
            // 【R3-F9】dedupPath=cpu（steger+cpu 对照口径）——补外挂 CPU 去重
            //（原 runLaserSteger 逻辑，fivechain 提交段此前遗漏）
            if (dedupPath_ == "cpu") {
                timeBegin();
                std::string dmsgL, dmsgR;
                bool okDedup = steger_dedup::apply(p.stegerL, ls, dmsgL);
                okDedup = steger_dedup::apply(p.stegerR, ls, dmsgR) && okDedup;
                (void)timeEnd("steger_dedup", p, record, false);
                setStepOk(p, "steger_dedup", okDedup);
                if (!okDedup) {
                    fail("steger_dedup", p, dmsgL + " | " + dmsgR);
                    return 0;
                }
            }
            // 【R3-F2】legacy 零点护栏（原 runLaserSteger 尾段口径——防 undistort
            // 空解引用：steger/flat 零点帧 success=true 但视图 null/空）
            if (!p.stegerL.d_centerPoints || !p.stegerR.d_centerPoints
                || p.stegerL.d_centerPoints->empty() || p.stegerR.d_centerPoints->empty()) {
                fail("undistort_cuda", p,
                     "steger 0 中心点（激光掩膜为空或非线形态——检查 mask_separation "
                     "参数：laserKill/markKill 选档）L="
                         + std::to_string(p.stegerL.totalPointCount) + " R="
                         + std::to_string(p.stegerR.totalPointCount));
                return 0;   // legacy 内部已同步——无在飞工作
            }
        }

        // —— undistort：容量宽视图（延迟；1:1 映射——尾区陈旧点多算无害 ~μs 级，
        //    下游消费界＝设备计数）或精确 [0,N) 视图（legacy）——
        timeBegin();
        p.undistCudaL = L.undistCudaL_->Execute(*p.stegerL.d_centerPoints,
                                                *p.stegerL.d_line_ids, ls);
        p.undistCudaR = L.undistCudaR_->Execute(*p.stegerR.d_centerPoints,
                                                *p.stegerR.d_line_ids, ls);
        setStepOk(p, "undistort_cuda",
                  p.undistCudaL.success && p.undistCudaR.success);
        timeEnd("undistort_cuda", p, record, false);
        if (!stepOk(p, "undistort_cuda")) {
            fail("undistort_cuda", p,
                 p.undistCudaL.message + " | " + p.undistCudaR.message);
            return deferred ? 1 : 0;   // 延迟：flat D2H 在飞——仍须排水
        }

        // —— interp：延迟（flat 设备计数＝undistort 有效输出数）或 legacy ——
        timeBegin();
        if (deferred) {
            p.interpL = L.interpL_->ExecuteDeferred(
                *p.undistCudaL.d_rectifiedPoints, *p.undistCudaL.d_line_ids,
                L.flatL_->GetDeviceCountPtr(), ls);
            p.interpR = L.interpR_->ExecuteDeferred(
                *p.undistCudaR.d_rectifiedPoints, *p.undistCudaR.d_line_ids,
                L.flatR_->GetDeviceCountPtr(), ls);
            timeEnd("epipolar_interp_dual", p, record, false);
            if (!p.interpL.success || !p.interpR.success) {
                setStepOk(p, "epipolar_interp_dual", false);
                fail("epipolar_interp_dual", p,
                     p.interpL.message + " | " + p.interpR.message);
                return 1;   // 已提交——排水（完成段跳过 FinishInterp）
            }
        } else {
            p.interpL = L.interpL_->Execute(*p.undistCudaL.d_rectifiedPoints,
                                            *p.undistCudaL.d_line_ids, ls);
            p.interpR = L.interpR_->Execute(*p.undistCudaR.d_rectifiedPoints,
                                            *p.undistCudaR.d_line_ids, ls);
            timeEnd("epipolar_interp_dual", p, record, false);
            setStepOk(p, "epipolar_interp_dual",
                      p.interpL.success && p.interpR.success);
            if (!stepOk(p, "epipolar_interp_dual")) {
                fail("epipolar_interp_dual", p,
                     p.interpL.message + " | " + p.interpR.message);
                return 0;
            }
        }
        return deferred ? 1 : 2;
    } catch (const std::exception& e) {
        fail("steger_fast", p, e.what());
        return 0;
    }
}

void Harness::finishLaserFront(LaserOps& L, CycleProducts& p, bool record, int kind) {
    (void)record;
    if (kind == 0) return;
    const bool deferred = (kind == 1);
    if (deferred) {
        L.flatL_->FinishDeferred(p.stegerL);
        L.flatR_->FinishDeferred(p.stegerR);
        setStepOk(p, "steger_fast", p.stegerL.success && p.stegerR.success);
        if (!stepOk(p, "steger_fast"))
            fail("steger_fast", p, p.stegerL.message + " | " + p.stegerR.message);
    }
    // —— 零点检查（原 runLaserSteger 尾段口径逐字保留；失败槽位 undistort_cuda）——
    if (stepOk(p, "steger_fast") &&
        (!p.stegerL.d_centerPoints || !p.stegerR.d_centerPoints
         || p.stegerL.d_centerPoints->empty() || p.stegerR.d_centerPoints->empty())) {
        fail("undistort_cuda", p,
             "steger 0 中心点（激光掩膜为空或非线形态——检查 mask_separation "
             "参数：laserKill/markKill 选档）L="
                 + std::to_string(p.stegerL.totalPointCount) + " R="
                 + std::to_string(p.stegerR.totalPointCount));
        return;
    }
    if (deferred && stepOk(p, "undistort_cuda")
        && p.interpL.success && p.interpR.success) {
        L.interpL_->FinishInterp(p.interpL);
        L.interpR_->FinishInterp(p.interpR);
        setStepOk(p, "epipolar_interp_dual",
                  p.interpL.success && p.interpR.success);
        if (!stepOk(p, "epipolar_interp_dual"))
            fail("epipolar_interp_dual", p,
                 p.interpL.message + " | " + p.interpR.message);
    }
}

void Harness::runLaserTail(LaserOps& L, CycleProducts& p, bool record, bool tubeOdd, cv::cuda::Stream& ls) {
    // 前段失败护栏（拆分后由收尾线程调用——复刻原串行早退语义；锁内读：标记
    // lane/配准线程并发写同 p 容器，裸遍历＝数据竞争）
    if (!stepOk(p, "steger_fast")) return;
    if (dedupPath_ == "cpu" && !stepOk(p, "steger_dedup")) return;

    try {
        timeBegin();
        p.undistCudaL = L.undistCudaL_->Execute(*p.stegerL.d_centerPoints,
                                              *p.stegerL.d_line_ids, ls);
        p.undistCudaR = L.undistCudaR_->Execute(*p.stegerR.d_centerPoints,
                                              *p.stegerR.d_line_ids, ls);
        setStepOk(p, "undistort_cuda", p.undistCudaL.success && p.undistCudaR.success);
        timeEnd("undistort_cuda", p, record, true);
        if (!stepOk(p, "undistort_cuda")) return;
    } catch (const std::exception& e) {
        fail("undistort_cuda", p, e.what());
        return;
    }

    try {
        timeBegin();
        p.interpL = L.interpL_->Execute(*p.undistCudaL.d_rectifiedPoints,
                                      *p.undistCudaL.d_line_ids, ls);
        p.interpR = L.interpR_->Execute(*p.undistCudaR.d_rectifiedPoints,
                                      *p.undistCudaR.d_line_ids, ls);
        setStepOk(p, "epipolar_interp_dual", p.interpL.success && p.interpR.success);
        timeEnd("epipolar_interp_dual", p, record, true);
        if (!stepOk(p, "epipolar_interp_dual")) return;
    } catch (const std::exception& e) {
        fail("epipolar_interp_dual", p, e.what());
        return;
    }

    try {
        timeBegin();
        // 按管注表（奇偶＝激光管号，不同管不同标定参数；缺省两管共表）＋逐帧温度
        L.matchOp_->SetTempTable(tubeOdd ? tableOdd_ : tableEven_);
        L.matchOp_->SetCurrentTemperature(cfg_.runtime.nominalTemperature);
        p.matchV3 = L.matchOp_->Execute(*p.interpL.d_interpPoints,
                                      *p.interpL.d_interp_line_ids,
                                       *p.interpR.d_interpPoints,
                                       *p.interpR.d_interp_line_ids, ls);
        // 【速度优化】v3 Execute 内部尾部已有一次流同步（计数回传后）——
        // 此处免重复等待（原注释：ROI 视图先同步再读计数；该同步已由算子完成）
        setStepOk(p, "laser_match_scan_v3", p.matchV3.success);
        p.laserMatched = p.matchV3.matchedCount;
        timeEnd("laser_match_scan_v3", p, record, false);
        if (!stepOk(p, "laser_match_scan_v3")) return;
    } catch (const std::exception& e) {
        fail("laser_match_scan_v3", p, e.what());
        return;
    }

    try {
        timeBegin();
        p.recon = L.reconOp_->Execute(*p.matchV3.d_matched_left, *p.matchV3.d_matched_right,
                                    *p.matchV3.d_matched_line_ids, calib_.Q, ls);
        // 【速度优化】重建 Execute 内部计数回传后已同步——免重复等待
        setStepOk(p, "laser_reconstruct", p.recon.success);
        p.laserPoints3D = p.recon.validCount;
        timeEnd("laser_reconstruct", p, record, false);
    } catch (const std::exception& e) {
        fail("laser_reconstruct", p, e.what());
        return;
    }

    // 行对齐残差/视差统计（矫正系极线自洽——断言5 口径；本底 0.2246px 对照）
    // 【产线口径 2026-10-19·用户裁定】此项与逐帧控制台/报表均为回放测试专用，
    // 产线不消费——JMW_PROD=1 跳过（省每帧 2×N 点下载＋同步＋中位统计）
    if (prodProfile_) return;   // 产线口径：跳过测试专用统计
    if (p.matchV3.matchedCount > 0 && p.matchV3.d_matched_left
        && p.matchV3.d_matched_right) {
        try {
            cv::Mat ml, mr;
            p.matchV3.d_matched_left->download(ml, ls);
            p.matchV3.d_matched_right->download(mr, ls);
            ls.waitForCompletion();
            std::vector<double> dRow, dDisp;
            const int n = ml.rows * ml.cols;
            for (int i = 0; i < n && i < mr.rows * mr.cols; ++i) {
                const cv::Point2f& l = ml.at<cv::Point2f>(i);
                const cv::Point2f& r = mr.at<cv::Point2f>(i);
                dRow.push_back(std::abs(l.y - r.y));
                dDisp.push_back(l.x - r.x);
            }
            p.rowResidMedPx = medianOf(dRow);
            p.dispMedPx = medianOf(dDisp);
        } catch (...) {
        }
    }
}

// ============================================================================
// 标记点链（吃同帧 d_markingPointMask——与激光链同一位姿）
// ============================================================================

void Harness::runMarkerChain(CycleProducts& p, bool record, MarkerOps& ops,
                             const std::shared_ptr<cv::cuda::GpuMat>& maskL,
                             const std::shared_ptr<cv::cuda::GpuMat>& maskR,
                             cudaEvent_t evMaskReady, cv::cuda::Stream& dlStream) {
    calib::RegionAnalysisResult cclL, cclR;
    try {
        timeBegin();
        if (cclPath_ == "cpu") {
            // 【150FPS·方案 A】CPU 路径：下载标记点掩膜（事件只拦到分离/快照完成，
            // 不等激光链）→ RLE 并查集单线程统计（免 GPU 三次全设备同步）
            if (!maskL || !maskR) {
                fail("ccl", p, "标记点掩膜为空（分离失败或快照缺失）");
                return;
            }
            if (evMaskReady)   // null＝数据已同步（回退路径），免等
                cudaStreamWaitEvent(cv::cuda::StreamAccessor::getStream(dlStream),
                                    evMaskReady, 0);
            cv::Mat hmL, hmR;
            maskL->download(hmL, dlStream);
            maskR->download(hmR, dlStream);
            dlStream.waitForCompletion();
            cclL = runCclCpu(hmL, cclMinArea_, cclMaxArea_, cclTopX_);
            cclR = runCclCpu(hmR, cclMinArea_, cclMaxArea_, cclTopX_);
        } else {
            // 【双链并行】ccl 走标记点链专用流——与激光链 GPU 内核并行执行
            cclL = cclL_->Execute(p.sepL.d_markingPointMask, streamMarker_);
            cclR = cclR_->Execute(p.sepR.d_markingPointMask, streamMarker_);
            streamMarker_.waitForCompletion();
        }
        p.cclL = std::move(cclL);
        p.cclR = std::move(cclR);
        setStepOk(p, "ccl", cclL.success && cclR.success);
        timeEnd("ccl", p, record, false);
        if (!stepOk(p, "ccl")) return;
    } catch (const std::exception& e) {
        fail("ccl", p, e.what());
        return;
    }
    runMarkerChainCore(p, record, ops);
}

// 【150FPS】行程直入版：GPU-RLE 产物（GPU lane 同步后回填）——标记点 lane 零
// CUDA 调用（免 2×2.36MB/帧 D2H 与 lane 侧下载流）
void Harness::runMarkerChainRuns(CycleProducts& p, bool record, MarkerOps& ops,
                                 const std::vector<int>& runsL, int nL,
                                 const std::vector<int>& runsR, int nR,
                                 int W, int H) {
    try {
        timeBegin();
        p.cclL = runCclCpuFromRuns(runsL, nL, W, H, cclMinArea_, cclMaxArea_, cclTopX_);
        p.cclR = runCclCpuFromRuns(runsR, nR, W, H, cclMinArea_, cclMaxArea_, cclTopX_);
        setStepOk(p, "ccl", p.cclL.success && p.cclR.success);
        timeEnd("ccl", p, record, false);
        if (!stepOk(p, "ccl")) return;
    } catch (const std::exception& e) {
        fail("ccl", p, e.what());
        return;
    }
    runMarkerChainCore(p, record, ops);
}

void Harness::runMarkerChainCore(CycleProducts& p, bool record, MarkerOps& ops) {
    std::vector<cv::Rect> roisL, roisR;
    for (const auto& c : p.cclL.components) roisL.push_back(componentRect(c));
    for (const auto& c : p.cclR.components) roisR.push_back(componentRect(c));
    if (roisL.empty() || roisR.empty()) {
        fail("image_split", p, "ccl 无组件（rois 空）L=" + std::to_string(roisL.size())
                                   + " R=" + std::to_string(roisR.size()));
        return;
    }

    try {
        timeBegin();
        p.splitL = ops.splitOp_->Execute(p.frameGrayL, roisL);
        p.splitR = ops.splitOp_->Execute(p.frameGrayR, roisR);
        setStepOk(p, "image_split", p.splitL.success && p.splitR.success);
        timeEnd("image_split", p, record, false);
        if (!stepOk(p, "image_split")) return;
    } catch (const std::exception& e) {
        fail("image_split", p, e.what());
        return;
    }

    try {
        timeBegin();
        p.zernL.clear();
        for (const auto& img : p.splitL.splitImages) p.zernL.push_back(ops.zernOp_->Execute(img));
        p.zernR.clear();
        for (const auto& img : p.splitR.splitImages) p.zernR.push_back(ops.zernOp_->Execute(img));
        bool ok = !p.zernL.empty() && !p.zernR.empty();
        for (const auto& z : p.zernL) ok = ok && z.success;
        for (const auto& z : p.zernR) ok = ok && z.success;
        setStepOk(p, "zernike_edge", ok);
        timeEnd("zernike_edge", p, record, false);
        if (!ok) return;
    } catch (const std::exception& e) {
        fail("zernike_edge", p, e.what());
        return;
    }

    try {
        timeBegin();
        std::vector<std::vector<calib::EdgePoint>> epL, epR;
        epL.reserve(p.zernL.size());
        epR.reserve(p.zernR.size());
        for (const auto& z : p.zernL) epL.push_back(z.edgePoints);
        for (const auto& z : p.zernR) epR.push_back(z.edgePoints);
        p.mergeL = ops.mergeOp_->Execute(epL, roisL);
        p.mergeR = ops.mergeOp_->Execute(epR, roisR);
        setStepOk(p, "image_merge", p.mergeL.success && p.mergeR.success);
        timeEnd("image_merge", p, record, false);
        if (!stepOk(p, "image_merge")) return;
    } catch (const std::exception& e) {
        fail("image_merge", p, e.what());
        return;
    }

    try {
        timeBegin();
        p.undistCpu = ops.undistCpuOp_->Execute(p.mergeL.mergedEdgePoints,
                                            p.mergeR.mergedEdgePoints,
                                            p.mergeL.groupIds, p.mergeR.groupIds);
        setStepOk(p, "undistort_cpu", p.undistCpu.success);
        timeEnd("undistort_cpu", p, record, false);
        if (!stepOk(p, "undistort_cpu")) return;
    } catch (const std::exception& e) {
        fail("undistort_cpu", p, e.what());
        return;
    }

    try {
        timeBegin();
        const auto g1 = p.undistCpu.splitRectifiedPoints1ByGroup();
        const auto g2 = p.undistCpu.splitRectifiedPoints2ByGroup();
        p.ellL.clear();
        p.ellR.clear();
        for (size_t i = 0; i < g1.size(); ++i) {
            if (g1[i].size() < 5) continue;   // min inliers (default 5)
            auto r = ops.ellOp_->Execute(g1[i]);
            r.groupId = static_cast<int>(i);
            p.ellL.push_back(std::move(r));
        }
        for (size_t i = 0; i < g2.size(); ++i) {
            if (g2[i].size() < 5) continue;
            auto r = ops.ellOp_->Execute(g2[i]);
            r.groupId = static_cast<int>(i);
            p.ellR.push_back(std::move(r));
        }
        int okL = 0, okR = 0;
        for (const auto& e : p.ellL) okL += e.success ? 1 : 0;
        for (const auto& e : p.ellR) okR += e.success ? 1 : 0;
        p.markersDetected = std::min(okL, okR);
        setStepOk(p, "ellipse_fit", p.markersDetected > 0);
        timeEnd("ellipse_fit", p, record, false);
        if (!stepOk(p, "ellipse_fit")) {
            fail("ellipse_fit", p, "椭圆拟合成功数不足 L=" + std::to_string(okL)
                                       + " R=" + std::to_string(okR));
            return;
        }
    } catch (const std::exception& e) {
        fail("ellipse_fit", p, e.what());
        return;
    }

    try {
        timeBegin();
        p.mm = ops.mmOp_->Execute(calib::extractEllipseCenters(p.ellL),
                              calib::extractEllipseCenters(p.ellR));
        setStepOk(p, "marker_match", p.mm.success);
        if (getenv("JMW_VERBOSE")) {   // 【150FPS】逐帧 stderr 打印管道下数十 ms——默认关
            int pairs = 0;
            for (int m : p.mm.centerMatches) pairs += (m >= 0) ? 1 : 0;
            std::cerr << "[marker_match] L=" << p.ellL.size() << " R=" << p.ellR.size()
                      << " 有效对=" << pairs << std::endl;
        }
        timeEnd("marker_match", p, record, false);
        if (!stepOk(p, "marker_match")) return;
    } catch (const std::exception& e) {
        fail("marker_match", p, e.what());
        return;
    }

    try {
        timeBegin();
        // 一步重建：椭圆整体参数＋marker_match 种子对 → 空间圆（三步旧链退役）
        p.conic = ops.conicOp_->Execute(p.ellL, p.ellR, p.mm.centerMatches);
        setStepOk(p, "conic_reconstruct", p.conic.success);
        p.markersValid = static_cast<int>(p.conic.statistics.validCount);
        timeEnd("conic_reconstruct", p, record, false);
    } catch (const std::exception& e) {
        fail("conic_reconstruct", p, e.what());
        return;
    }
}

// ============================================================================
// 配准（三级降级：光流 → frame_fuse 兜底【测试侧门 matched≥3 且 rmse<50mm】→ 沿用）
// ============================================================================

void Harness::runRegistration(CycleProducts& p, bool record) {
    // 基线：先置上帧 R/T（hold 路径即保留此值）
    if (!prevState_.empty()) {
        for (int i = 0; i < 9; ++i) p.R[i] = prevState_.R.val[i];
        for (int i = 0; i < 3; ++i) p.T[i] = prevState_.T.val[i];
    }

    if (!stepOk(p, "conic_reconstruct") || p.markersValid == 0) {
        p.regSource = "skip";
        return;
    }

    // 规则网格板检测（2026-10-15 裁决，硬拒锁存）：网格平移对称＝原理性不可分辨，
    // 检测到即拒绝配准（每帧落 stepError，summary 可见），直至会话结束
    if (!gridBoardRejected_) {
        std::vector<cv::Point3d> probe;
        for (const auto& m : p.conic.markers) {
            if (m.validCircle) probe.emplace_back(m.centerX, m.centerY, m.centerZ);
        }
        if (detectRegularGrid(probe)) {
            gridBoardRejected_ = true;
            std::cerr << "[registration] 规则阵列标记点板：平移对称不可分辨，配准拒绝（改用随机布点板）"
                      << std::endl;
        }
    }
    if (gridBoardRejected_) {
        fail("registration", p, "规则阵列标记点板：平移对称不可分辨，配准拒绝（改用随机布点板）");
        p.regSource = "hold";
        return;
    }

    timeBegin();
    try {
        // 有效圆（validCircle）center/normal → 光流 vector 重载（旧 PointReconstruct
        // 重载随三步链退役；成功语义不变：matched≥3）
        std::vector<cv::Point3d> centers;
        std::vector<cv::Vec3d> normals;
        for (const auto& m : p.conic.markers) {
            if (!m.validCircle) continue;
            centers.emplace_back(m.centerX, m.centerY, m.centerZ);
            normals.emplace_back(m.normalX, m.normalY, m.normalZ);
        }
        p.regFlow = flowOp_->Execute(centers, normals, prevState_);
        // 失败原因落盘（summary.json flowDiag 节——断链可诊断：差几对/误差多少/算子原因）
        p.flowMatchedDiag = static_cast<int>(p.regFlow.getMatchedCount());
        p.flowRmseDiag = p.regFlow.statistics.rmse;
        p.flowMessageDiag = p.regFlow.message;
        // 误差趋势门（2026-10-15 加固）：误差超"开局基线中位×倍率"判失败——专抓
        // 温漂/斜视角等全体一起慢变（绝对门只拦单帧尖刺，拦不住整体爬升；帧间差分
        // 结构性看不见同向漂移）。基线＝前 50 个成功帧中位，冻结后全会话不变。
        const double driftRatio = cfg_.runtime.flowRmseDriftRatio;
        bool driftRejected = false;
        if (driftRatio > 0.0 && p.regFlow.success) {
            if (flowRmseBaselineMed_ < 0.0) {
                flowRmseBaseline_.push_back(p.regFlow.statistics.rmse);
                if (flowRmseBaseline_.size() >= 50) {
                    auto s = flowRmseBaseline_;
                    std::sort(s.begin(), s.end());
                    flowRmseBaselineMed_ = s[s.size() / 2];
                }
            } else if (p.regFlow.statistics.rmse
                       > driftRatio * flowRmseBaselineMed_) {
                driftRejected = true;
                p.flowMessageDiag += " [drift-gate: rmse>"
                                     + std::to_string(driftRatio * flowRmseBaselineMed_)
                                     + " = " + std::to_string(driftRatio)
                                       + "x baseline]";
            }
        }
        const bool flowOk = !driftRejected && p.regFlow.success
                            && static_cast<int>(p.regFlow.getMatchedCount()) >= 3;
        setStepOk(p, "optical_flow_fuse", p.regFlow.success);
        if (flowOk) {
            p.regSource = prevState_.empty() ? "first" : "flow";
            p.regMatched = static_cast<int>(p.regFlow.getMatchedCount());
            adoptRegistration(p.regFlow, p);
        } else if (!prevState_.empty() && !globalSet_.empty()) {
                calib::MarkerPointSet cur;
                for (const auto& m : p.conic.markers) {
                    if (!m.validCircle) continue;
                    cur.positions.emplace_back(m.centerX, m.centerY, m.centerZ);
                    cur.normals.emplace_back(m.normalX, m.normalY, m.normalZ);
                }
                // 批2 ①：法线合理性门——双峰法线（正反混合/大面积翻转）时拒走
                // 兜底（治已知边界：孪生布点＋法线翻转→折中混合解）
                lastNormalOpposeRatio_ = checkNormalPlausibility(cur.normals);
                if (lastNormalOpposeRatio_ > 0.25) {
                    CALIB_LOG_WARN("fuse: 法线双峰（反对占比 {:.1f}%>25%）——"
                                   "数据异常（贴反/重建翻转/双面混合），拒走兜底",
                                   lastNormalOpposeRatio_ * 100.0);
                    p.regSource = "hold";
                } else {
                // —— 步2 子集契约（2026-10-15）：绝不把全库直接喂算子——库超
                //    triangleMaxPoints 会静默回退 PFH（共面形态＝必死路径）。
                //    快路径＝视野捞点（上帧 R/T 预测，半径随 hold 连击放大）；
                //    失败→慢路径＝步5 三角索引全库投票（不依赖预测——光流刚丢
                //    时 R/T 恰恰最可疑）。慢路径节流（批1 F4）：连败 15 帧内
                //    每帧可试，之后每 3 帧一试——不停手（回环唯一生路）但不交
                //    无谓学费。
                p.usedFrameFuse = true;
                // 子集尺寸与算子 triangleMaxPoints 解绑（批1 P0：防误配——
                // 调大算子上限想"大库直配"时不再连带放大子集装配）
                const int libMax = std::max(6, std::min(64,
                    std::min(fuseParams_.triangleMaxPoints,
                             cfg_.runtime.fuseSubsetMax)));
                std::vector<int> sel;
                if (holdStreak_ <= 15) {
                    sel = cullByView(cur, libMax);
                    if (sel.size() >= 3)
                        p.regFuse = fuseExecute(cur, sel);
                }
                bool fuseOk = acceptFuse(p.regFuse);
                bool fromSlow = false;   // 本帧胜出解是否来自慢路径（跳变门依据）
                std::vector<int> sel2;   // 慢路径子集（胜出时用它译下标——F3）
                const bool slowPathTurn = holdStreak_ <= 15 || (holdStreak_ % 3) == 0;
                if (!fuseOk && slowPathTurn) {
                    // 批3：索引重建节流——写库即脏，但连续扫描时每帧都脏→每帧
                    // 全量重建（N=500 约 1.7ms）＝浪费。距上次重建 ≥5 帧才重建；
                    // 索引从未建过时立即建。旧索引仍可查询（数据略陈旧但比没有好）。
                    if (libIndexVersion_ != libVersion_
                        && (lastIndexRebuildCycle_ < 0
                            || cycleCount_ - lastIndexRebuildCycle_ >= 5)) {
                        if (!libIndex_) libIndex_ = std::make_unique<TriangleLibraryIndex>();
                        libIndex_->rebuild(globalSet_.positions, globalSet_.normals);
                        libIndexVersion_ = libVersion_;
                        lastIndexRebuildCycle_ = cycleCount_;
                    }
                    if (libIndex_ && !libIndex_->empty()) {
                        sel2 = libIndex_->query(cur.positions, libMax);
                        if (sel2.size() >= 3 && sel2 != sel) {
                            p.regFuse = fuseExecute(cur, sel2);
                            fromSlow = true;   // 先记账：无论成败，regFuse 已属 sel2
                            fuseOk = acceptFuse(p.regFuse);
                        }
                    }
                }
                p.fuseSlowPath = fromSlow && fuseOk;   // 慢路径扳回＝预测失效证据
                // 跳变门（批1 F7 收紧）：大跳要么真回环要么错解——支持点不足 7
                // 时【且未经慢路径通道复核】拒（快路径预测失效时恰最易产翻转
                // 解；慢路径全库投票不依赖预测，是唯一可信的大跳证据来源）
                if (fuseOk) {
                    const cv::Matx33d dR = cv::Matx33d(p.regFuse.R) * cv::Matx33d(prevState_.R).t();
                    const double tr = dR(0, 0) + dR(1, 1) + dR(2, 2);
                    const double dAng = std::acos(std::min(1.0,
                        std::max(-1.0, (tr - 1.0) * 0.5))) * 180.0 / CV_PI;
                    const double dTmm = cv::norm(cv::Vec3d(p.regFuse.T) - cv::Vec3d(prevState_.T));
                    if ((dTmm > 500.0 || dAng > 60.0) && p.regFuse.matchedCount < 7
                        && !fromSlow) {
                        fuseOk = false;
                        p.regFuse.message = "jump gate: 大跳且支持不足且未经慢路径复核，判 hold";
                    }
                }
                setStepOk(p, "frame_fuse", p.regFuse.success && fuseOk);
                if (fuseOk) {
                    p.regSource = "fuse";
                    p.regMatched = static_cast<int>(p.regFuse.matchedCount);
                    for (int i = 0; i < 9; ++i) p.R[i] = p.regFuse.R.val[i];
                    for (int i = 0; i < 3; ++i) p.T[i] = p.regFuse.T.val[i];
                    // correspondences.second 是【实际喂算子的子集】局部下标——
                    // 批1 F3 修复：快/慢路径各自记 sel，按胜出解来源映射回库下标
                    // （旧版恒用快路径 sel 译慢路径下标→回环恢复即污染库；旧版
                    // 空 sel 时还整帧重发新号）
                    const bool slowWon = fromSlow && !sel2.empty();
                    const std::vector<int>& usedSel = slowWon ? sel2 : sel;
                    std::vector<int> libIdxOfPair(usedSel.size(), -1);
                    for (size_t j = 0; j < usedSel.size(); ++j) libIdxOfPair[j] = usedSel[j];
                    std::vector<int> gidByIndex(globalSet_.positions.size(), -1);
                    for (const auto& kv : globalIndexById_)
                        if (kv.second < gidByIndex.size()) gidByIndex[kv.second] = kv.first;
                    prevState_.rawPositions.clear();
                    prevState_.rawNormals.clear();
                    prevState_.globalIds.clear();
                    prevState_.R = p.regFuse.R;
                    prevState_.T = p.regFuse.T;
                    constexpr double kMergeRadiusMm = 2.0;   // 与 verifyGate 同口径
                    int gi = 0;
                    for (const auto& m : p.conic.markers) {
                        if (!m.validCircle) continue;
                        int gid = -1;
                        for (const auto& co : p.regFuse.correspondences)
                            if (co.first == gi
                                && co.second >= 0
                                && co.second < static_cast<int>(libIdxOfPair.size())) {
                                const int libIdx = libIdxOfPair[static_cast<size_t>(co.second)];
                                if (libIdx >= 0 && libIdx < static_cast<int>(gidByIndex.size())
                                    && gidByIndex[libIdx] >= 0)
                                    gid = gidByIndex[libIdx];
                                break;
                            }
                        const cv::Point3d c(m.centerX, m.centerY, m.centerZ);
                        const cv::Vec3d n(m.normalX, m.normalY, m.normalZ);
                        const auto& Rf = p.regFuse.R;
                        const auto& Tf = p.regFuse.T;
                        const cv::Point3d gpos(Rf(0,0)*c.x + Rf(0,1)*c.y + Rf(0,2)*c.z + Tf(0),
                                               Rf(1,0)*c.x + Rf(1,1)*c.y + Rf(1,2)*c.z + Tf(1),
                                               Rf(2,0)*c.x + Rf(2,1)*c.y + Rf(2,2)*c.z + Tf(2));
                        const cv::Vec3d gnorm(Rf(0,0)*n(0) + Rf(0,1)*n(1) + Rf(0,2)*n(2),
                                              Rf(1,0)*n(0) + Rf(1,1)*n(1) + Rf(1,2)*n(2),
                                              Rf(2,0)*n(0) + Rf(2,1)*n(1) + Rf(2,2)*n(2));
                        if (gid < 0) {
                            // 未匹配点：合并半径内归并旧号（防重复副本假候选），
                            // 否则会话计数器发新号入图。
                            // 批1 F5：归并须过有向法线门（dot≥cos60°）——防薄壁
                            // 双面背面点在板厚<2mm 时冒名顶替正面库点
                            double bestD = 1e30;
                            int nearLib = -1;
                            for (size_t li = 0; li < globalSet_.positions.size(); ++li) {
                                const double dx = gpos.x - globalSet_.positions[li].x;
                                const double dy = gpos.y - globalSet_.positions[li].y;
                                const double dz = gpos.z - globalSet_.positions[li].z;
                                const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
                                if (d < bestD) { bestD = d; nearLib = static_cast<int>(li); }
                            }
                            const double mergeNormalMinCos = std::cos(
                                fuseParams_.triangleNormalMaxAngleDeg * CV_PI / 180.0);
                            if (nearLib >= 0 && bestD < kMergeRadiusMm && gidByIndex[nearLib] >= 0
                                && gnorm.dot(globalSet_.normals[static_cast<size_t>(nearLib)])
                                       >= mergeNormalMinCos)
                                gid = gidByIndex[nearLib];
                            else
                                gid = nextGid_++;
                        }
                        // 写库体检（批1 P0）：非有限值不入库——一个 NaN 毒点会让
                        // 索引重建静默弃组、算子整批拒绝，兜底永久瘫痪且无日志
                        const bool finiteOk = std::isfinite(gpos.x) && std::isfinite(gpos.y)
                            && std::isfinite(gpos.z) && std::isfinite(gnorm(0))
                            && std::isfinite(gnorm(1)) && std::isfinite(gnorm(2));
                        if (!finiteOk) {
                            CALIB_LOG_WARN("frame_fuse: 写库拒绝非有限值点（gid={} 跳过）",
                                           gid);
                            prevState_.rawPositions.push_back(c);
                            prevState_.rawNormals.push_back(n);
                            prevState_.globalIds.push_back(gid);
                            ++gi;
                            continue;
                        }
                        const auto it = globalIndexById_.find(gid);
                        if (it == globalIndexById_.end()) {
                            globalIndexById_[gid] = globalSet_.positions.size();
                            globalSet_.positions.push_back(gpos);
                            globalSet_.normals.push_back(gnorm);
                        } else {
                            globalSet_.positions[it->second] = gpos;
                            globalSet_.normals[it->second] = gnorm;
                        }
                        prevState_.rawPositions.push_back(c);
                        prevState_.rawNormals.push_back(n);
                        prevState_.globalIds.push_back(gid);
                        ++gi;
                    }
                    ++libVersion_;   // 库已写 → 索引脏
                } else {
                    p.regSource = "hold";
                }
                }   // else（法线合理性通过——批2 ①的闭合）
            } else {
                p.regSource = "hold";
            }
        // hold 连击记账（捞点半径自适应＋停手门）；非 hold 即清零
        if (p.regSource == "hold") ++holdStreak_;
        else holdStreak_ = 0;
    } catch (const std::exception& e) {
        fail("registration", p, e.what());
        p.regSource = "hold";
    }
    timeEnd("registration", p, record, false);
    // hold/skip：保留基线上帧 R/T（prevState 停在上一次成功帧）
}

// ============================================================================
// adoptRegistration — 收编一次成功的光流/重锁结果（2026-10-14 优化项）：
// 置 R/T、重建 prevState（含新点发号入图）、维护全局地图。
// 新点入图对齐生产 ScanChains 全局发号做法——地图随扫描增长，设备进新区域
// 也有锚点（历史缺陷：地图冻结在首帧，断链后兜底无锚可对）。
// ============================================================================
void Harness::adoptRegistration(calib::MarkerOpticalFlowFuseCPUResult& res,
                                CycleProducts& p) {
    for (int i = 0; i < 9; ++i) p.R[i] = res.R.val[i];
    for (int i = 0; i < 3; ++i) p.T[i] = res.T.val[i];
    prevState_.rawPositions.clear();
    prevState_.rawNormals.clear();
    prevState_.globalIds.clear();
    prevState_.R = res.R;
    prevState_.T = res.T;
    // 发号查重用：地图下标 → 编号反查表（2026-10-15 裁决项）
    std::vector<int> gidByIndex(globalSet_.positions.size(), -1);
    for (const auto& kv : globalIndexById_)
        if (kv.second < gidByIndex.size()) gidByIndex[kv.second] = kv.first;
    for (auto& m : res.markers) {
        // 新点发号前查重（2026-10-15 裁决，对齐兜底路径做法）：先在全局系找最近
        // 地图条目，命中查重半径内沿用旧号（防同点重复领号永久污染地图）；
        // 查重量的是累积漂移，半径独立于 matchDistThresh
        if (m.globalId < 0) {
            int nearGid = -1;
            double best = cfg_.runtime.mintDedupRadiusMm;
            for (size_t k = 0; k < globalSet_.positions.size(); ++k) {
                const double dx = m.transformedPosition.x - globalSet_.positions[k].x;
                const double dy = m.transformedPosition.y - globalSet_.positions[k].y;
                const double dz = m.transformedPosition.z - globalSet_.positions[k].z;
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d < best && gidByIndex[k] >= 0) {
                    // 批2 ②：归并须过法线同向门——防薄壁双面背面点冒名正面库点
                    if (!mergeNormalCompatible(m.transformedNormal,
                                               globalSet_.normals[k]))
                        continue;
                    best = d;
                    nearGid = gidByIndex[k];
                }
            }
            if (nearGid >= 0) {
                m.globalId = nearGid;
            } else {
                // 批2 ③：候选期——新点先过候选池，连续 3 次重观测一致才转正入图
                // （防光晕假点/瞬态误检一帧入图即成永久毒点）
                bool promoted = false;
                for (auto& cand : candidates_) {
                    const double dx = m.transformedPosition.x - cand.position.x;
                    const double dy = m.transformedPosition.y - cand.position.y;
                    const double dz = m.transformedPosition.z - cand.position.z;
                    const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (d < kCandidateMergeMm) {
                        ++cand.count;
                        cand.position = m.transformedPosition;   // 最新坐标
                        cand.normal = m.transformedNormal;
                        if (cand.count >= kPromoteThreshold) {
                            m.globalId = nextGid_++;
                            promoted = true;
                            // 转正：从候选池移除，入正式库（下方统一处理）
                        }
                        break;
                    }
                }
                if (!promoted && m.globalId < 0) {
                    // 首次出现：入候选池，暂不发号入图
                    candidates_.push_back({m.transformedPosition,
                                           m.transformedNormal, 1});
                    // prevState 仍记录（下帧光流可用），但 globalId 置 -1 标记候选中
                    m.globalId = -1;
                    // 注：此点不入 globalSet_（候选不入图），但 prevState 记录
                    // rawPosition/rawNormal 供下帧光流匹配。globalId=-1 意味着
                    // 光流下帧会重新尝试匹配并走候选/归并流程。
                }
            }
        } else {
            nextGid_ = std::max(nextGid_, m.globalId + 1);
        }
        prevState_.rawPositions.push_back(m.rawPosition);
        prevState_.rawNormals.push_back(m.rawNormal);
        prevState_.globalIds.push_back(m.globalId);
        // 写库体检（批1 P0）：非有限值不入图——防一个 NaN 毒点让索引静默弃组、
        // 算子整批拒绝（兜底永久瘫痪且无日志，不可自愈）
        const auto& tp = m.transformedPosition;
        const auto& tn = m.transformedNormal;
        const bool finiteOk = std::isfinite(tp.x) && std::isfinite(tp.y)
            && std::isfinite(tp.z) && std::isfinite(tn(0)) && std::isfinite(tn(1))
            && std::isfinite(tn(2));
        if (!finiteOk) {
            CALIB_LOG_WARN("adoptRegistration: 写库拒绝非有限值点（gid={} 跳过入图）",
                           m.globalId);
            continue;
        }
        // 地图维护：新号追加、旧号最新覆盖（last-write-wins 历史口径）
        // 批2 ③：globalId == -1 表示"仍在候选期"——不入图（防瞬态假点）
        if (m.globalId < 0) {
            continue;   // 候选点：prevState 已记录，跳过入图
        }
        const auto it = globalIndexById_.find(m.globalId);
        if (it == globalIndexById_.end()) {
            globalIndexById_[m.globalId] = globalSet_.positions.size();
            globalSet_.positions.push_back(m.transformedPosition);
            globalSet_.normals.push_back(m.transformedNormal);
        } else {
            globalSet_.positions[it->second] = m.transformedPosition;
            globalSet_.normals[it->second] = m.transformedNormal;
        }
    }
    ++libVersion_;   // 库已写 → 三角索引脏（光流/重锁路径同样要触发重建）
    // 批2 ③：候选池过期清理——超过 30 帧未达转正门槛的候选视为瞬态误检，丢弃
    //   （防候选池无限增长；正常扫描 3 帧即转正，30 帧远超合理等待）
    if (candidates_.size() > 4) {
        candidates_.erase(
            std::remove_if(candidates_.begin(), candidates_.end(),
                           [](const CandidateEntry& c) { return c.count < 2; }),
            candidates_.end());
    }
}

// ============================================================================
// frame_fuse 兜底增强三助手（2026-10-15 步2/步5）
// ============================================================================

std::vector<int> Harness::cullByView(const calib::MarkerPointSet& cur,
                                     int maxCount) const {
    std::vector<int> out;
    if (cur.empty() || globalSet_.positions.empty() || maxCount < 3) return out;
    // 当前帧标记质心（当前帧坐标系）
    cv::Vec3d c(0, 0, 0);
    for (const auto& q : cur.positions)
        c += cv::Vec3d(q.x, q.y, q.z);
    c /= static_cast<double>(cur.size());
    // 库点经上帧 R/T 逆变换到当前帧系：p_cur = Rᵀ(p_glob − T)
    // 半径自适应：基值 150mm ＋ 30mm×hold 连击（长跟丢→预测越不可信→圈越大），封顶 600
    const double radius = std::min(600.0, 150.0 + 30.0 * static_cast<double>(holdStreak_));
    std::vector<std::pair<double, int>> byDist;   // (到质心距离, 库下标)
    byDist.reserve(globalSet_.positions.size());
    const auto& Rp = prevState_.R;
    const auto& Tp = prevState_.T;
    for (size_t i = 0; i < globalSet_.positions.size(); ++i) {
        const auto& g = globalSet_.positions[i];
        const double gx = g.x - Tp(0), gy = g.y - Tp(1), gz = g.z - Tp(2);
        const cv::Vec3d pc(Rp(0, 0) * gx + Rp(1, 0) * gy + Rp(2, 0) * gz,
                           Rp(0, 1) * gx + Rp(1, 1) * gy + Rp(2, 1) * gz,
                           Rp(0, 2) * gx + Rp(1, 2) * gy + Rp(2, 2) * gz);
        byDist.emplace_back(cv::norm(pc - c), static_cast<int>(i));
    }
    std::sort(byDist.begin(), byDist.end());
    for (const auto& kv : byDist) {          // 先收半径内的
        if (static_cast<int>(out.size()) >= maxCount) break;
        if (kv.first <= radius) out.push_back(kv.second);
    }
    if (out.size() < 3) {                     // 半径内不足：退取全库最近（终审兜底）
        out.clear();
        for (size_t i = 0; i < byDist.size() && static_cast<int>(out.size()) < maxCount; ++i)
            out.push_back(byDist[i].second);
    }
    std::sort(out.begin(), out.end());        // 库下标序＝确定性
    return out;
}

calib::FrameFuseCPUResult Harness::fuseExecute(
    const calib::MarkerPointSet& cur, const std::vector<int>& libIdx) const {
    calib::MarkerPointSet gms;   // 子集装配（契约：绝不传全库）
    gms.positions.reserve(libIdx.size());
    gms.normals.reserve(libIdx.size());
    for (int li : libIdx) {
        gms.positions.push_back(globalSet_.positions[static_cast<size_t>(li)]);
        gms.normals.push_back(globalSet_.normals[static_cast<size_t>(li)]);
    }
    return fuseOp_->Execute(cur, gms);
}

bool Harness::acceptFuse(const calib::FrameFuseCPUResult& r) const {
    return r.success
           && static_cast<int>(r.matchedCount) >= cfg_.runtime.fuseMinMatched
           && r.rmse < cfg_.runtime.fuseMaxRmseMm;
}

// ============================================================================
// 批2 ①：帧级法线合理性检查——治已知边界"孪生布点＋法线翻转→折中混合解"
// 大白话：看这帧圆片朝向是不是"一部分朝上一部分朝下"——如果是，说明数据
// 本身有问题（贴反/重建翻转/双面混合），兜底拿这种数据只会产折中解（有界
// 但不精确）；不如老实说配不上。判据：与"主方向"反对的占比 > 25% → 拒。
// ============================================================================
double Harness::checkNormalPlausibility(
    const std::vector<cv::Vec3d>& normals) const {
    if (normals.size() < 4) return 0.0;   // 点太少不判
    // 中位法线方向：取归一化均值（法线已归一化）
    cv::Vec3d median(0, 0, 0);
    for (const auto& n : normals) median += n;
    const double len = std::sqrt(median.dot(median));
    if (len < 1e-12) return 1.0;          // 全互相抵消＝极端双峰
    median /= len;
    int oppose = 0;
    for (const auto& n : normals)
        if (n.dot(median) < 0.0) ++oppose;
    return static_cast<double>(oppose) / static_cast<double>(normals.size());
}

// ============================================================================
// 批2 ②：光流归并法线同向门（对齐兜底路径批1 F5 做法）——防薄壁双面冒名
// ============================================================================
bool Harness::mergeNormalCompatible(const cv::Vec3d& a, const cv::Vec3d& b) const {
    const double minCos = std::cos(
        fuseParams_.triangleNormalMaxAngleDeg * CV_PI / 180.0);
    return a.dot(b) >= minCos;
}

// ============================================================================
// runCycle
// ============================================================================

bool Harness::runCycle(int cycleIdx, CycleProducts& p, bool recordTiming,
                       std::string& err) {
    if (src_ == nullptr || cycleIdx < 0
        || cycleIdx >= static_cast<int>(src_->cycles.size())) {
        err = "cycleIdx 越界: " + std::to_string(cycleIdx);
        return false;
    }
    const Cycle& cyc = src_->cycles[cycleIdx];
    mainTid_ = std::this_thread::get_id();   // 计时基点线程感知锚
    prodProfile_ = (getenv("JMW_PROD") != nullptr);   // 【产线口径】测试记账旁路定
    // 【五链 P2】fivechain 执行器（config fivechain.executor=fivechain 或 env
    // JMW_FIVECHAIN=1）：串行头后切换——8 线程五链（读帧/预上传/CUDA链1/CPU链1×1/
    // CPU链3/CPU链2/CUDA链2/主线程收割）。标记链 CPU 化与 pipe 同因。
    if (!cfg_.runtime.markerOnly
        && (cfg_.fivechain.executor == "fivechain"
            || getenv("JMW_FIVECHAIN") != nullptr)) {
        if (cclPath_ != "cpu") {
            cclPath_ = "cpu";
            std::cout << "[fivechain] cclPath 强制 cpu（标记点链 CPU 化）" << std::endl;
        }
        const int serialHead =
            std::min<int>(cfg_.runtime.warmupCycles + cfg_.runtime.dumpFullFrames,
                          static_cast<int>(src_->cycles.size()));
        if (cycleIdx >= serialHead) {
            if (!fc_) {
                const int total = cfg_.runtime.maxCycles > 0
                    ? std::min<int>(cfg_.runtime.maxCycles,
                                    static_cast<int>(src_->cycles.size()))
                    : static_cast<int>(src_->cycles.size());
                fc_ = std::make_unique<fc::FiveChainExecutor>(*this, cycleIdx, total);
            }
            return fc_->collect(cycleIdx, p, err);
        }
    }
    // 【150FPS·方案 D】JMW_PIPELINE=1 启用跨帧流水线（GPU lane∥标记点 lane×3∥
    // 配准线程）。串行头（warmup＋dumpFullFrames 帧）仍走原串行路径——逐帧全量
    // dump 与金样口径逐字节一致；流水线帧只落 summary（report/PLY 口径不变）。
    // 流水线标记点链在 CPU lane——强制 CPU ccl（GPU ccl 的 streamMarker_ 单流
    // 不可跨 lane 并发）。
    if (!cfg_.runtime.markerOnly && getenv("JMW_PIPELINE") != nullptr) {
        if (cclPath_ != "cpu") {
            cclPath_ = "cpu";
            std::cout << "[pipeline] cclPath 强制 cpu（标记点链 CPU lane 化）"
                      << std::endl;
        }
        const int serialHead =
            std::min<int>(cfg_.runtime.warmupCycles + cfg_.runtime.dumpFullFrames,
                          static_cast<int>(src_->cycles.size()));
        if (cycleIdx >= serialHead) {
            if (!pipe_) {
                int lanes = 3;
                if (const char* e = getenv("JMW_MLANES")) lanes = std::max(1, atoi(e));
                // 总帧数＝主循环将跑的帧数（maxCycles 截断口径一致——防越过
                // 截断点继续铺帧）
                const int total = cfg_.runtime.maxCycles > 0
                    ? std::min<int>(cfg_.runtime.maxCycles,
                                    static_cast<int>(src_->cycles.size()))
                    : static_cast<int>(src_->cycles.size());
                pipe_ = std::make_unique<PipelineExecutor>(*this, lanes, cycleIdx,
                                                           total);
            }
            // 【150FPS】提前铺帧（在飞窗口 8）＋按序收割——runCycle 返回该帧产物
            pipe_->submitAhead(cycleIdx);
            bool recUsed = false;   // 帧内记账已按提交时点 warmup 边界执行
            return pipe_->collect(cycleIdx, p, recUsed, err);
        }
    }
    // 【执行模式·默认全串行】帧内顺序执行＋读图在本帧内完成（严格
    // "第一帧彻底结束再开第二帧"）。JMW_PARALLEL=1 开帧内双链并行；
    // JMW_PREFETCH=1 开后台图片预读（二者均为诊断/对照用）。
    cv::Mat grayL, grayR;
    if (getenv("JMW_PREFETCH") != nullptr) {
        if (!prefetch_) prefetch_ = std::make_unique<FramePrefetcher>(*src_);
        std::string perr;
        prefetch_->get(cyc.laserIdx, grayL, grayR, perr);
        if (grayL.empty() || grayR.empty()) {
            err = perr.empty() ? "帧读取失败" : perr;
            return false;
        }
    } else {
        if (!src_->readFrame(cyc.laserIdx, true, grayL, err)
            || !src_->readFrame(cyc.laserIdx, false, grayR, err)) {
            return false;
        }
    }
    p.frameGrayL = grayL;
    p.frameGrayR = grayR;
    runSeparation(grayL, grayR, p, recordTiming);          // 一次分离
    // 【执行模式·默认全串行】激光链→标记点链→配准，顺序执行；
    // JMW_PARALLEL=1 才走双链并行（标记点链工作线程∥激光链主线程）
    if (!cfg_.runtime.markerOnly && getenv("JMW_PARALLEL") != nullptr) {
        // —— 并行路径（默认）：标记点链工作线程 ∥ 激光链主线程 ——
        CycleProducts pm;   // 标记点链独立产物
        pm.sepL.d_markingPointMask = p.sepL.d_markingPointMask;   // 共享视图（只读）
        pm.sepR.d_markingPointMask = p.sepR.d_markingPointMask;
        pm.frameGrayL = p.frameGrayL;   // Mat 浅拷贝（引用计数安全）
        pm.frameGrayR = p.frameGrayR;
        std::thread markerTh([this, &pm, recordTiming]() {
            runMarkerChain(pm, recordTiming, serialOps_,
                           pm.sepL.d_markingPointMask, pm.sepR.d_markingPointMask,
                           evSepDone_, streamMarker_);
        });
        runLaserChain(p, recordTiming, cyc.tubeOdd);       // 吃 d_laserMask（同帧）
        markerTh.join();
        // —— 合并标记点链产物（串行，无竞争）——
        p.cclL = std::move(pm.cclL);   p.cclR = std::move(pm.cclR);
        p.splitL = std::move(pm.splitL); p.splitR = std::move(pm.splitR);
        p.zernL = std::move(pm.zernL);   p.zernR = std::move(pm.zernR);
        p.mergeL = std::move(pm.mergeL); p.mergeR = std::move(pm.mergeR);
        p.undistCpu = std::move(pm.undistCpu);
        p.ellL = std::move(pm.ellL);   p.ellR = std::move(pm.ellR);
        p.mm = std::move(pm.mm);       p.conic = std::move(pm.conic);
        p.markersDetected = pm.markersDetected;
        p.markersValid = pm.markersValid;
        for (auto& kv : pm.stepMs) p.stepMs.push_back(std::move(kv));
        for (auto& kv : pm.stepSource) p.stepSource[kv.first] = std::move(kv.second);
        for (auto& kv : pm.stepSuccess) p.stepSuccess[kv.first] = kv.second;
        for (auto& kv : pm.stepError) p.stepError[kv.first] = std::move(kv.second);
        runRegistration(p, recordTiming);
        // 轨迹只记正式帧（预热帧不计时亦不进轨迹，防统计污染）
        if (recordTiming) trajT_.push_back({p.T[0], p.T[1], p.T[2]});    // 轨迹记账（M11_trajectory.png）
        ++cycleCount_;   // 帧：帧计数（索引重建节流计时用）
    } else {
        // —— 串行路径（JMW_SERIAL=1 / markerOnly）：原执行序（激光→标记点）——
        if (!cfg_.runtime.markerOnly)
            runLaserChain(p, recordTiming, cyc.tubeOdd);
        runMarkerChain(p, recordTiming, serialOps_,
                       p.sepL.d_markingPointMask, p.sepR.d_markingPointMask,
                       evSepDone_, streamMarker_);
        if (!cfg_.runtime.markerOnly) {
            runRegistration(p, recordTiming);
            if (recordTiming) trajT_.push_back({p.T[0], p.T[1], p.T[2]});
            ++cycleCount_;
        }
    }
    // 【事件计时制】帧尾：一次同步＋补记全部 GPU 步骤时长
    flushStepEvents(p, recordTiming);
    return true;
}

// ============================================================================
// 一致性检查 / dump / lastRT
// ============================================================================

double Harness::checkConsistency(const CycleProducts& p, std::string& detail) const {
    if (!stepOk(p, "undistort_cpu")) {
        detail = "undistort_cpu 未成功，跳过";
        return -1.0;
    }
    const double dR1 = matMaxAbsDiff(p.undistCpu.R1, calib_.R1);
    const double dR2 = matMaxAbsDiff(p.undistCpu.R2, calib_.R2);
    const double dP1 = matMaxAbsDiff(p.undistCpu.P1, calib_.P1);
    const double dP2 = matMaxAbsDiff(p.undistCpu.P2, calib_.P2);
    const double dQ = matMaxAbsDiff(p.undistCpu.Q, calib_.Q);
    const double mx = std::max({dR1, dR2, dP1, dP2, dQ});
    std::ostringstream os;
    os << std::fixed << std::setprecision(8)
       << "max|Δ|=" << mx << " (dR1=" << dR1 << " dR2=" << dR2
       << " dP1=" << dP1 << " dP2=" << dP2 << " dQ=" << dQ << ")";
    detail = os.str();
    return mx;
}

void Harness::dumpCycle(int cycleIdx, const CycleProducts& p, bool full) const {
    dumpCycleFiltered(cycleIdx, p, full, nullptr);
}

void Harness::dumpCycleFiltered(int cycleIdx, const CycleProducts& p, bool full,
                                const std::vector<std::string>* onlySteps) const {
    char cycBuf[32];
    std::snprintf(cycBuf, sizeof(cycBuf), "cycle_%03d", cycleIdx);
    const fs::path dir = fs::path(outDir_) / cycBuf;
    std::error_code ec;
    fs::create_directories(dir, ec);

    const auto want = [onlySteps](const char* step) -> bool {
        if (onlySteps == nullptr) return true;
        return std::find(onlySteps->begin(), onlySteps->end(), step) != onlySteps->end();
    };

    // summary.json（恒写）
    nlohmann::json sum;
    sum["cycle"] = cycleIdx;
    nlohmann::json steps = nlohmann::json::object();
    for (const auto& kv : p.stepSuccess) steps[kv.first]["success"] = kv.second;
    for (const auto& kv : p.stepError) steps[kv.first]["error"] = kv.second;
    for (const auto& kv : p.stepMs) steps[kv.first]["ms"] = kv.second;
    for (const auto& kv : p.stepSource) steps[kv.first]["timingSource"] = kv.second;
    sum["steps"] = steps;
 if (extractor_ == "flat") sum["extractor"] = "flat_center";   // 仅 flat 时写——默认路径产物逐字节不变
    sum["markersDetected"] = p.markersDetected;
    sum["markersValid"] = p.markersValid;
    sum["regSource"] = p.regSource;
    sum["regMatched"] = p.regMatched;
    // 配准诊断（2026-10-14 优化项）：光流失败原因/误差——断链可诊断
    sum["flowDiag"]["matched"] = p.flowMatchedDiag;
    sum["flowDiag"]["rmse"] = p.flowRmseDiag;
    if (!p.flowMessageDiag.empty()) sum["flowDiag"]["message"] = p.flowMessageDiag;
    nlohmann::json rt = nlohmann::json::array({});
    for (int i = 0; i < 9; ++i) rt.push_back(p.R[i]);
    sum["R"] = rt;
    sum["T"] = {p.T[0], p.T[1], p.T[2]};
    sum["laserMatched"] = p.laserMatched;
    sum["laserPoints3D"] = p.laserPoints3D;
    sum["rowResidMedPx"] = p.rowResidMedPx;
    sum["dispMedPx"] = p.dispMedPx;
    {   // conic_reconstruct 拒因直方图（枚举名→计数；全零＝步骤未达/未跑）
        nlohmann::json hist = nlohmann::json::object();
        for (size_t k = 0; k < calib::kFailReasonCount; ++k)
            hist[calib::ConicFailReasonName(static_cast<calib::ConicFailReason>(k))] =
            p.conic.statistics.failReasonHistogram[k];
        sum["conicHistogram"] = hist;
    }
    if (p.usedFrameFuse) {
        sum["frameFuse"]["matched"] = p.regFuse.matchedCount;
        sum["frameFuse"]["rmse"] = p.regFuse.rmse;
        sum["frameFuse"]["slowPath"] = p.fuseSlowPath;   // 索引慢路径扳回＝预测失效证据
    }
    saveJson((dir / "summary.json").string(), sum);

    if (!full) return;

    // ============================================================
    // 全量可视化产物（全部左右并合一张图 L|4px|R）
    // 命名：0 原图 ｜ L* 激光链（L0 掩膜→L5 3D）｜ M* 标记点链（M0 掩膜→M11 配准）
    // ============================================================
    cv::cuda::Stream cs;   // dump 专用流（ROI 视图先物化后下载）

    // 0- 原图（L|R 并合底版，后续叠加类可视化的同款画布基准）
    {
        cv::Mat img = mergeLR(p.frameGrayL, p.frameGrayR);
        if (!img.empty()) savePng((dir / "0-frames.png").string(), img);
    }

    // 1- 掩膜分离（纯掩膜二值图，白=掩膜；L|R 并合）
    if (want("mask_separation")) {
        auto maskLR = [&](const char* name,
                          const std::shared_ptr<cv::cuda::GpuMat>& mL,
                          const std::shared_ptr<cv::cuda::GpuMat>& mR) {
            if (!mL && !mR) return;
            cv::Mat hl, hr;
            try {
                if (mL) {
                    cv::cuda::GpuMat c; mL->copyTo(c, cs); c.download(hl, cs);
                }
                if (mR) {
                    cv::cuda::GpuMat c; mR->copyTo(c, cs); c.download(hr, cs);
                }
                cs.waitForCompletion();
            } catch (...) {
            }
            cv::Mat img = mergeLR(hl, hr);
            if (!img.empty()) savePng((dir / name).string(), img);
        };
        maskLR("L0-laser_mask.png", p.sepL.d_laserMask, p.sepR.d_laserMask);
        maskLR("M0-marking_mask.png",
               p.sepL.d_markingPointMask, p.sepR.d_markingPointMask);
        // 0-binary：高斯+二值化后、未分割的全量前景（分离前端证据图——调 threshold 看这张）
        maskLR("0-binary.png", p.sepL.d_step1_binary, p.sepR.d_step1_binary);
    }

    // 2- steger：激光中心点（绿）叠原帧，L|R 并合
    if (want("steger_fast") && p.stegerL.d_centerPoints) {
        saveGpuMatP2((dir / "L1-steger_L.csv").string(), *p.stegerL.d_centerPoints, cs);
        auto stgL = downloadP2(p.stegerL.d_centerPoints, cs);
        auto stgR = downloadP2(p.stegerR.d_centerPoints, cs);
        cv::Mat cl = toBgr(p.frameGrayL), cr = toBgr(p.frameGrayR);
        drawPts(cl, stgL, cv::Scalar(0, 255, 0), 1);
        drawPts(cr, stgR, cv::Scalar(0, 255, 0), 1);
        cv::Mat img = mergeLR(cl, cr);
        if (!img.empty()) savePng((dir / "L1-steger.png").string(), img);
    }

    // 2b- 连通域编号（人工标注基准图）：左右图各自独立把激光点按空间连通
    //     性分段（不含任何匹配语义），每段一个编号＋专属色，编号沿段多处
    //     标注；小于 minPts 的碎段画暗灰点不编号。左右编号独立（1.. 各自
    //     起算），供人工标注"左 N ↔ 右 M"匹配真值；CSV 落盘供真值回映到点
    if (want("steger_fast") && p.stegerL.d_centerPoints) {
        const float kCell = 8.0f;    // 栅格粒度 px（< 线间距 ~34px）
        const int kMinPts = 30;      // 编号段最小点数（碎渣不编号）
        const int kBand = 220;       // 编号标注 y 带宽 px（长段多处标号）
        auto stgL = downloadP2(p.stegerL.d_centerPoints, cs);
        auto stgR = downloadP2(p.stegerR.d_centerPoints, cs);
        // 矫正系同源点（undistort 消费同一 post-dedup 点集、同序）——段↔lid 精确映射用
        auto rectL = p.undistCudaL.success && p.undistCudaL.d_rectifiedPoints
                         ? downloadP2(p.undistCudaL.d_rectifiedPoints, cs)
                         : std::vector<cv::Point2f>();
        auto rectR = p.undistCudaR.success && p.undistCudaR.d_rectifiedPoints
                         ? downloadP2(p.undistCudaR.d_rectifiedPoints, cs)
                         : std::vector<cv::Point2f>();
        cv::Mat cl = toBgr(p.frameGrayL), cr = toBgr(p.frameGrayR);
        auto drawSide = [&](cv::Mat& canvas, const std::vector<cv::Point2f>& pts,
                            const std::vector<cv::Point2f>& ptsRect, const std::string& csvPath,
                            const std::string& csvRectPath) {
            std::vector<int> compSize;
            const std::vector<int> label = labelConnected(pts, kCell, compSize);
            // 有效段按质心 x（左右顺序）排序 → 显示编号 1..K（左→右递增，
            // 与扇面线序一致，左右图编号可按位置对读；顺序稳定可复现）
            struct Comp { int id, size; cv::Point2f c; };
            std::vector<cv::Point2f> sum(compSize.size(), cv::Point2f(0, 0));
            for (int i = 0; i < (int)pts.size(); ++i)
                if (label[i] >= 0) {
                    sum[label[i]].x += pts[i].x;
                    sum[label[i]].y += pts[i].y;
                }
            std::vector<Comp> keep;
            for (size_t c = 0; c < compSize.size(); ++c)
                if (compSize[c] >= kMinPts)
                    keep.push_back({(int)c, compSize[c], sum[c] / (float)compSize[c]});
            std::sort(keep.begin(), keep.end(), [](const Comp& a, const Comp& b) {
                return a.c.x != b.c.x ? a.c.x < b.c.x : a.c.y < b.c.y;
            });
            std::vector<int> display(compSize.size(), 0);
            for (size_t k = 0; k < keep.size(); ++k) display[keep[k].id] = (int)k + 1;

            std::ofstream csv(csvPath);
            csv << "x,y,comp\n" << std::fixed << std::setprecision(2);
            std::ofstream csvR(csvRectPath);
            csvR << "xr,yr,comp\n" << std::fixed << std::setprecision(2);
            std::map<int, std::map<int, std::vector<float>>> bandX;   // 编号→y带→x 列表
            for (int i = 0; i < (int)pts.size(); ++i) {
                const int d = label[i] >= 0 ? display[label[i]] : 0;
                csv << pts[i].x << ',' << pts[i].y << ',' << d << '\n';
                if (i < (int)ptsRect.size())
                    csvR << ptsRect[i].x << ',' << ptsRect[i].y << ',' << d << '\n';
                const int x = cvRound(pts[i].x), y = cvRound(pts[i].y);
                if (d == 0) {
                    if (x >= 0 && y >= 0 && x < canvas.cols && y < canvas.rows)
                        cv::circle(canvas, cv::Point(x, y), 1, cv::Scalar(80, 80, 80), -1);
                    continue;
                }
                cv::circle(canvas, cv::Point(x, y), 2, paletteColor(d), -1);
                bandX[d][y / kBand].push_back(pts[i].x);
            }
            // 编号标注：每段每 y 带取带内中位 x 点位（黑底描边保证可读）
            for (const auto& bv : bandX) {
                for (const auto& band : bv.second) {
                    std::vector<float> xs = band.second;
                    std::sort(xs.begin(), xs.end());
                    const float mx = xs[xs.size() / 2];
                    const float my = (band.first + 0.5f) * kBand;
                    const std::string t = std::to_string(bv.first);
                    cv::putText(canvas, t, cv::Point(cvRound(mx) + 6, cvRound(my)),
                                0, 0.9, cv::Scalar(0, 0, 0), 4);
                    cv::putText(canvas, t, cv::Point(cvRound(mx) + 6, cvRound(my)),
                                0, 0.9, paletteColor(bv.first), 2);
                }
            }
            std::cout << "[dump] L1c 连通段（编号段 " << keep.size()
                      << " / 总段 " << compSize.size() << "）" << std::endl;
        };
        drawSide(cl, stgL, rectL, (dir / "L1c-lines_L.csv").string(),
                 (dir / "L1r-lines_L.csv").string());
        drawSide(cr, stgR, rectR, (dir / "L1c-lines_R.csv").string(),
                 (dir / "L1r-lines_R.csv").string());
        cv::Mat img = mergeLR(cl, cr);
        if (!img.empty()) savePng((dir / "L1c-lines.png").string(), img);
    }

    // 3- undistort_cuda：矫正后激光点（绿）画暗画布（矫正系坐标，无法叠原帧），L|R 并合
    if (want("undistort_cuda") && p.undistCudaL.success) {
        auto rl = downloadP2(p.undistCudaL.d_rectifiedPoints, cs);
        auto rr = downloadP2(p.undistCudaR.d_rectifiedPoints, cs);
        if (!rl.empty() || !rr.empty()) {
            cv::Mat cl(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3,
                       cv::Scalar(20, 20, 20));
            cv::Mat cr = cl.clone();
            drawPts(cl, rl, cv::Scalar(0, 255, 0), 1);
            drawPts(cr, rr, cv::Scalar(0, 255, 0), 1);
            savePng((dir / "L2-undistort_cuda.png").string(), mergeLR(cl, cr));
        }
    }

    // 4- 极线插值：插值点（黄）叠原帧，L|R 并合
    if (want("epipolar_interp_dual") && p.interpL.d_interpPoints) {
        saveGpuMatP2((dir / "L3-interp_L.csv").string(), *p.interpL.d_interpPoints, cs);
        if (p.interpR.d_interpPoints)
            saveGpuMatP2((dir / "L3-interp_R.csv").string(), *p.interpR.d_interpPoints, cs);
        auto il = downloadP2(p.interpL.d_interpPoints, cs);
        auto ir = downloadP2(p.interpR.d_interpPoints, cs);
        {   // L3t：左点段 ID（v3 投票的段来源，诊断用）
            auto trk = downloadIds(p.interpL.d_interp_line_ids, cs);
            const size_t tn = std::min(il.size(), trk.size());
            std::ofstream f((dir / "L3t-tracks_L.csv").string());
            f << "x,y,track\n" << std::fixed << std::setprecision(2);
            for (size_t i = 0; i < tn; ++i)
                f << il[i].x << ',' << il[i].y << ',' << trk[i] << '\n';
        }
        cv::Mat cl = toBgr(p.frameGrayL), cr = toBgr(p.frameGrayR);
        drawPts(cl, il, cv::Scalar(0, 255, 255), 1);
        drawPts(cr, ir, cv::Scalar(0, 255, 255), 1);
        cv::Mat img = mergeLR(cl, cr);
        if (!img.empty()) savePng((dir / "L3-interp.png").string(), img);
    }

    // 5- v3 匹配：白=未匹配（左右各自插值点），彩色=匹配对（按激光线号同色，
    //    左点旁标注线号编号、编号与线同色；同线号左右同色），L|R 并合
    if (want("laser_match_scan_v3") && p.matchV3.d_matched_left) {
        saveGpuMatP2((dir / "L4-match_L.csv").string(), *p.matchV3.d_matched_left, cs);
        if (p.matchV3.d_matched_right)
            saveGpuMatP2((dir / "L4-match_R.csv").string(), *p.matchV3.d_matched_right, cs);
        auto ml = downloadP2(p.matchV3.d_matched_left, cs);
        auto mr = downloadP2(p.matchV3.d_matched_right, cs);
        auto ids = downloadIds(p.matchV3.d_matched_line_ids, cs);
        auto il = downloadP2(p.interpL.d_interpPoints, cs);
        auto ir = downloadP2(p.interpR.d_interpPoints, cs);
        cv::Mat cl(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::Mat cr = cl.clone();
        drawPts(cl, il, cv::Scalar(255, 255, 255), 2);      // 白=未匹配左点
        drawPts(cr, ir, cv::Scalar(255, 255, 255), 2);      // 白=未匹配右点
        const size_t n = std::min({ml.size(), mr.size(), ids.size()});
        // 编号稀疏化：每条线号首点＋每隔 60 点标一次（万级点全标会糊成一片）
        std::map<int, int> lidSeen;
        for (size_t i = 0; i < n; ++i) {
            const cv::Scalar c = paletteColor(ids[i]);   // 同线号同色
            cv::circle(cl, cv::Point(cvRound(ml[i].x), cvRound(ml[i].y)), 3, c, -1);
            cv::circle(cr, cv::Point(cvRound(mr[i].x), cvRound(mr[i].y)), 3, c, -1);
            const int seen = lidSeen[ids[i]]++;
            if (seen == 0 || seen % 60 == 0) {
                cv::putText(cl, std::to_string(ids[i]),
                            cv::Point(cvRound(ml[i].x) + 5, cvRound(ml[i].y) - 5), 0, 0.7,
                            c, 2);   // 编号=激光线号（放大加粗便于缩放查看）
                cv::putText(cr, std::to_string(ids[i]),
                            cv::Point(cvRound(mr[i].x) + 5, cvRound(mr[i].y) - 5), 0, 0.7,
                            c, 2);   // 右目同号（左右同色同编号对应）
            }
        }
        savePng((dir / "L4-match_v3.png").string(), mergeLR(cl, cr));
        {   // L4t：匹配对段 ID（含 lid，诊断投票/丢段用）
            auto mtrk = downloadIds(p.matchV3.d_matched_track_ids, cs);
            const size_t tn = std::min({n, mtrk.size()});
            std::ofstream f((dir / "L4t-match-tracks.csv").string());
            f << "x,y,lid,track\n" << std::fixed << std::setprecision(2);
            for (size_t i = 0; i < tn; ++i)
                f << ml[i].x << ',' << ml[i].y << ',' << ids[i] << ',' << mtrk[i] << '\n';
        }
    }

    // 5b- 带内真值候选图：匹配点（矫正系坐标）按连通段聚合，每段标注
    //     主导线号＋纯度（段内该线号占比）。带内单卡独占工况下，若每段
    //     纯度 100% 且左右同号线几何对应，则线号可作人工核验后的匹配
    //     基准；混色段/低纯度段＝带内匹配自身可疑处，人工重点核查。
    //     段编号按质心 x 排序（与 L1c 同惯例）；CSV 落段级真值表
    if (want("laser_match_scan_v3") && p.matchV3.d_matched_left) {
        const float kCellTruth = 8.0f;
        const int kMinTruth = 30;
        const int kBandTruth = 220;
        auto ml = downloadP2(p.matchV3.d_matched_left, cs);
        auto mr = downloadP2(p.matchV3.d_matched_right, cs);
        auto ids = downloadIds(p.matchV3.d_matched_line_ids, cs);
        const size_t n = std::min({ml.size(), mr.size(), ids.size()});
        cv::Mat cl(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::Mat cr = cl.clone();
        std::ofstream segCsv((dir / "L4c-truth_segments.csv").string());
        segCsv << "side,comp,n,dominantLid,purityPct,x0,x1,y0,y1\n";
        auto drawTruthSide = [&](cv::Mat& canvas, const std::vector<cv::Point2f>& pts,
                                 const char* side) {
            std::vector<int> compSize;
            const std::vector<int> label = labelConnected(pts, kCellTruth, compSize);
            const int nCompAll = (int)compSize.size();
            // 段 lid 直方图＋质心＋包围盒
            std::vector<std::map<int, int>> lidHist(nCompAll);
            std::vector<cv::Point2f> sum(nCompAll, cv::Point2f(0, 0));
            std::vector<cv::Vec4f> bbox(nCompAll,
                                        cv::Vec4f(1e30f, -1e30f, 1e30f, -1e30f));  // x0,x1,y0,y1
            for (size_t i = 0; i < n && i < pts.size(); ++i) {
                if (label[i] < 0) continue;
                ++lidHist[label[i]][ids[i]];
                sum[label[i]].x += pts[i].x;
                sum[label[i]].y += pts[i].y;
                auto& bb = bbox[label[i]];
                bb[0] = std::min(bb[0], pts[i].x);
                bb[1] = std::max(bb[1], pts[i].x);
                bb[2] = std::min(bb[2], pts[i].y);
                bb[3] = std::max(bb[3], pts[i].y);
            }
            struct Seg { int id, n, dom; double purity; cv::Point2f c; };
            std::vector<Seg> keep;
            for (int c = 0; c < nCompAll; ++c) {
                if (compSize[c] < kMinTruth) continue;
                int dom = -1, domN = 0;
                for (auto& kv : lidHist[c])
                    if (kv.second > domN) { dom = kv.first; domN = kv.second; }
                keep.push_back({c, compSize[c], dom, 100.0 * domN / compSize[c],
                                sum[c] / (float)compSize[c]});
            }
            std::sort(keep.begin(), keep.end(), [](const Seg& a, const Seg& b) {
                return a.c.x != b.c.x ? a.c.x < b.c.x : a.c.y < b.c.y;
            });
            std::vector<int> display(nCompAll, 0);
            for (size_t k = 0; k < keep.size(); ++k) display[keep[k].id] = (int)k + 1;
            // 点按各自 lid 着色（混段直接显色）；编号按段 x 序
            std::map<int, std::map<int, std::vector<float>>> bandX;
            for (size_t i = 0; i < n && i < pts.size(); ++i) {
                if (label[i] < 0) continue;
                const int d = display[label[i]];
                const cv::Scalar c = paletteColor(ids[i]);
                cv::circle(canvas, cv::Point(cvRound(pts[i].x), cvRound(pts[i].y)), 2, c, -1);
                if (d > 0) bandX[d][(int)(pts[i].y / kBandTruth)].push_back(pts[i].x);
            }
            for (const auto& seg : keep) {
                const int d = display[seg.id];
                const std::string t = seg.purity < 99.5
                    ? std::to_string(seg.dom) + "(" + std::to_string((int)std::round(seg.purity)) + "%)"
                    : std::to_string(seg.dom);
                for (const auto& band : bandX[d]) {
                    std::vector<float> xs = band.second;
                    std::sort(xs.begin(), xs.end());
                    const float mx = xs[xs.size() / 2];
                    const float my = (band.first + 0.5f) * kBandTruth;
                    cv::putText(canvas, t, cv::Point(cvRound(mx) + 6, cvRound(my)),
                                0, 0.9, cv::Scalar(0, 0, 0), 4);
                    cv::putText(canvas, t, cv::Point(cvRound(mx) + 6, cvRound(my)),
                                0, 0.9, paletteColor(seg.dom), 2);
                }
                const auto& bb = bbox[seg.id];
                segCsv << side << ',' << d << ',' << seg.n << ',' << seg.dom << ','
                       << std::fixed << std::setprecision(1) << seg.purity << ','
                       << bb[0] << ',' << bb[1] << ',' << bb[2] << ',' << bb[3] << '\n';
            }
            int impure = 0;
            for (const auto& s : keep)
                if (s.purity < 99.5) ++impure;
            std::cout << "[dump] L4c 真值段（" << side << "）：编号段 " << keep.size()
                      << "，纯度<99.5% 段 " << impure << std::endl;
        };
        drawTruthSide(cl, ml, "L");
        drawTruthSide(cr, mr, "R");
        savePng((dir / "L4c-truth.png").string(), mergeLR(cl, cr));
    }

    // 6- 激光 3D 点云（PLY，3D 查看器）
    if (want("laser_reconstruct") && p.recon.d_points3d) {
        try {
            cv::cuda::GpuMat contiguous;
            p.recon.d_points3d->copyTo(contiguous, cs);
            cv::Mat h;
            contiguous.download(h, cs);
            cs.waitForCompletion();
            std::vector<cv::Point3f> pts;
            if (!h.empty() && h.type() == CV_32FC3)
                pts.assign(h.begin<cv::Point3f>(), h.end<cv::Point3f>());
            savePlyXYZf((dir / "L5-laser3d.ply").string(), pts);
        } catch (...) {
        }
    }

    // 7- ccl：组件伪彩标签＋白框＋编号，L|R 并合
    if (want("ccl")) {
        auto cclCanvas = [&](const calib::RegionAnalysisResult& r) {
            cv::Mat canvas;
            if (!r.d_labeledMask) return canvas;
            try {
                cv::cuda::GpuMat c;
                r.d_labeledMask->copyTo(c, cs);
                cv::Mat lab;
                c.download(lab, cs);
                cs.waitForCompletion();
                if (lab.empty() || lab.type() != CV_32SC1) return canvas;
                double mx = 0;
                cv::minMaxIdx(lab, nullptr, &mx);
                if (mx <= 0) {
                    canvas = cv::Mat::zeros(lab.size(), CV_8UC3);
                    return canvas;
                }
                cv::Mat norm;
                lab.convertTo(norm, CV_8UC1, 255.0 / mx);
                cv::applyColorMap(norm, canvas, cv::COLORMAP_HSV);
                canvas.setTo(0, lab == 0);
                for (const auto& comp : r.components) {
                    cv::rectangle(canvas,
                                  cv::Rect(comp.boundingBoxX, comp.boundingBoxY,
                                           comp.boundingBoxWidth, comp.boundingBoxHeight),
                                  cv::Scalar(255, 255, 255), 2);
                    cv::putText(canvas, std::to_string(comp.label),
                                cv::Point(comp.boundingBoxX, comp.boundingBoxY - 4), 0,
                                0.5, cv::Scalar(255, 255, 255));
                }
            } catch (...) {
                canvas.release();
            }
            return canvas;
        };
        cv::Mat img = mergeLR(cclCanvas(p.cclL), cclCanvas(p.cclR));
        if (!img.empty()) savePng((dir / "M1-ccl.png").string(), img);
    }

    // 8- image_split：ROI 子图网格拼贴（L|R 各一拼贴再并合）
    if (want("image_split")) {
        const cv::Mat colL = makeCollage(p.splitL.splitImages, 160, 36);
        const cv::Mat colR = makeCollage(p.splitR.splitImages, 160, 36);
        cv::Mat img = mergeLR(colL, colR);
        if (!img.empty()) savePng((dir / "M2-split.png").string(), img);
    }

    // 9- zernike：ROI 拼贴上叠边缘点（红，ROI 局部系），L|R 并合
    if (want("zernike_edge")) {
        std::vector<EdgePointCsv> eps;
        for (const auto& z : p.zernL)
            for (const auto& e : z.edgePoints)
                eps.push_back({e.x, e.y, e.angle, e.amplitude, e.pixelX, e.pixelY});
        saveCsvEdgePoints((dir / "M3-zern_L.csv").string(), eps);
        auto zernCollage = [&](const std::vector<calib::ZernikeEdgeCPUResult>& zs,
                               const std::vector<cv::Mat>& tiles) {
            std::vector<cv::Mat> drawn;
            const size_t n = std::min(zs.size(), tiles.size());
            for (size_t i = 0; i < n; ++i) {
                cv::Mat t = toBgr(tiles[i]);
                for (const auto& e : zs[i].edgePoints) {
                    const int x = cvRound(e.x), y = cvRound(e.y);
                    if (x >= 0 && y >= 0 && x < t.cols && y < t.rows)
                        cv::circle(t, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
                }
                drawn.push_back(t);
            }
            return makeCollage(drawn, 160, 36);
        };
        cv::Mat img = mergeLR(zernCollage(p.zernL, p.splitL.splitImages),
                              zernCollage(p.zernR, p.splitR.splitImages));
        if (!img.empty()) savePng((dir / "M3-zernike.png").string(), img);
    }

    // 10- image_merge：合并后边缘点按组着色叠原帧（大图系），L|R 并合
    if (want("image_merge") && p.mergeL.success) {
        auto mergeCanvas = [&](const calib::ImageMergeCPUResult& m,
                               const cv::Mat& gray) {
            cv::Mat canvas = toBgr(gray);
            for (size_t i = 0; i < m.mergedEdgePoints.size(); ++i) {
                const auto& e = m.mergedEdgePoints[i];
                const int g = (i < m.groupIds.size()) ? m.groupIds[i] : 0;
                const int x = cvRound(e.x), y = cvRound(e.y);
                if (x >= 0 && y >= 0 && x < canvas.cols && y < canvas.rows)
                    cv::circle(canvas, cv::Point(x, y), 1, paletteColor(g), -1);
            }
            return canvas;
        };
        cv::Mat img = mergeLR(mergeCanvas(p.mergeL, p.frameGrayL),
                              mergeCanvas(p.mergeR, p.frameGrayR));
        if (!img.empty()) savePng((dir / "M4-merge.png").string(), img);
    }

    // 11- undistort_cpu：L/R 矫正后点同图散点（蓝=左/红=右）——行对齐证据（本身即合并对照）
    if (want("undistort_cpu") && p.undistCpu.success
        && !p.undistCpu.rectifiedPoints1.empty()) {
        cv::Mat canvas(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3,
                       cv::Scalar(20, 20, 20));
        std::vector<cv::Point2f> lpts, rpts;
        lpts.reserve(p.undistCpu.rectifiedPoints1.size());
        rpts.reserve(p.undistCpu.rectifiedPoints2.size());
        for (const auto& q : p.undistCpu.rectifiedPoints1) lpts.emplace_back(float(q.x), float(q.y));
        for (const auto& q : p.undistCpu.rectifiedPoints2) rpts.emplace_back(float(q.x), float(q.y));
        drawPts(canvas, lpts, cv::Scalar(255, 0, 0), 1);
        drawPts(canvas, rpts, cv::Scalar(0, 0, 255), 1);
        savePng((dir / "M5-undistort_cpu.png").string(), canvas);
    }

    // 12- ellipse_fit：椭圆轮廓＋中心十字＋组号（矫正系画布），L|R 并合
    if (want("ellipse_fit")) {
        nlohmann::json ellArr = nlohmann::json::array();
        for (const auto& e : p.ellL)
            if (e.success)
                ellArr.push_back({{"groupId", e.groupId}, {"cx", e.centerX}, {"cy", e.centerY},
                                  {"major", e.majorAxis}, {"minor", e.minorAxis},
                                  {"angle", e.angle}, {"inliers", e.inlierCount}});
        saveJson((dir / "M6-ellipse_L.json").string(), ellArr);
        nlohmann::json ellArrR = nlohmann::json::array();
        for (const auto& e : p.ellR)
            if (e.success)
                ellArrR.push_back({{"groupId", e.groupId}, {"cx", e.centerX}, {"cy", e.centerY},
                                   {"major", e.majorAxis}, {"minor", e.minorAxis},
                                   {"angle", e.angle}, {"inliers", e.inlierCount}});
        saveJson((dir / "M6-ellipse_R.json").string(), ellArrR);
        auto ellipseCanvas = [&](const std::vector<calib::EllipseFitCPUResult>& es) {
            cv::Mat canvas(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3,
                           cv::Scalar(20, 20, 20));
            for (const auto& e : es) {
                if (!e.success) continue;
                cv::ellipse(canvas,
                            cv::RotatedRect(cv::Point2f(float(e.centerX), float(e.centerY)),
                                            cv::Size2f(float(e.majorAxis), float(e.minorAxis)),
                                            float(e.angle)),
                            cv::Scalar(255, 200, 0), 1);
                cv::drawMarker(canvas, cv::Point(cvRound(e.centerX), cvRound(e.centerY)),
                               cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 12, 1);
                cv::putText(canvas, std::to_string(e.groupId),
                            cv::Point(cvRound(e.centerX) + 6, cvRound(e.centerY) - 6), 0,
                            0.5, cv::Scalar(0, 255, 255));
            }
            return canvas;
        };
        cv::Mat img = mergeLR(ellipseCanvas(p.ellL), ellipseCanvas(p.ellR));
        if (!img.empty()) savePng((dir / "M6-ellipse.png").string(), img);
    }

    // 13- marker_match：中心点配对——匹配对同号同色（编号=左索引），
    //     未匹配灰点；左右并合同一张（同编号跨左右对应）
    if (want("marker_match") && p.mm.success) {
        auto centersOf = [](const std::vector<calib::EllipseFitCPUResult>& es) {
            std::vector<cv::Point2f> out;
            for (const auto& e : es)
                if (e.success) out.emplace_back(float(e.centerX), float(e.centerY));
            return out;
        };
        const auto cl = centersOf(p.ellL), cr = centersOf(p.ellR);
        cv::Mat canvasL(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3,
                        cv::Scalar(20, 20, 20));
        cv::Mat canvasR = canvasL.clone();
        // 未匹配：灰
        for (size_t i = 0; i < cl.size(); ++i)
            if (i >= p.mm.centerMatches.size() || p.mm.centerMatches[i] < 0)
                cv::circle(canvasL, cv::Point(cvRound(cl[i].x), cvRound(cl[i].y)), 4,
                           cv::Scalar(90, 90, 90), -1);
        for (size_t j = 0; j < cr.size(); ++j) {
            bool matched = false;
            for (int m : p.mm.centerMatches) matched |= (m == static_cast<int>(j));
            if (!matched)
                cv::circle(canvasR, cv::Point(cvRound(cr[j].x), cvRound(cr[j].y)), 4,
                           cv::Scalar(90, 90, 90), -1);
        }
        // 匹配对：同色＋同编号（编号=左索引）
        for (size_t i = 0; i < cl.size() && i < p.mm.centerMatches.size(); ++i) {
            const int j = p.mm.centerMatches[i];
            if (j < 0 || j >= static_cast<int>(cr.size())) continue;
            const cv::Scalar c = paletteColor(static_cast<int>(i));
            cv::circle(canvasL, cv::Point(cvRound(cl[i].x), cvRound(cl[i].y)), 5, c, -1);
            cv::circle(canvasR, cv::Point(cvRound(cr[j].x), cvRound(cr[j].y)), 5, c, -1);
            cv::putText(canvasL, std::to_string(i),
                        cv::Point(cvRound(cl[i].x) + 6, cvRound(cl[i].y) - 6), 0, 0.7, c, 2);
            cv::putText(canvasR, std::to_string(i),
                        cv::Point(cvRound(cr[j].x) + 6, cvRound(cr[j].y) - 6), 0, 0.7, c, 2);
            // 右图逐对标注：dy=yL-yR（离极线行偏差）＋ d=xL-xR（视差），与编号同色
            cv::putText(canvasR, cv::format("dy=%+.2f", cl[i].y - cr[j].y),
                        cv::Point(cvRound(cr[j].x) + 6, cvRound(cr[j].y) + 16), 0, 0.45, c, 1);
            cv::putText(canvasR, cv::format("d=%.1f", cl[i].x - cr[j].x),
                        cv::Point(cvRound(cr[j].x) + 6, cvRound(cr[j].y) + 30), 0, 0.45, c, 1);
        }
        savePng((dir / "M7-marker_match.png").string(), mergeLR(canvasL, canvasR));
    }

    // 14- 标记点 3D（PLY，3D 查看器；conic 空间圆合成圆环点=reconstructedPoints，
    //     每 validMarker ringPointsForViz 点；文件名沿用 M10-markers3d.ply）
    if (want("conic_reconstruct")) {
        std::vector<cv::Point3d> pts;
        for (const auto& m : p.conic.markers) {
            if (!m.validCircle) continue;
            for (const auto& q : m.reconstructedPoints)
                pts.emplace_back(q.x, q.y, q.z);
        }
        savePlyXYZd((dir / "M10-markers3d.ply").string(), pts);
    }

    // 15- 配准详情＋轨迹图（轨迹为全局累积，落 outDir 根）
    if (want("registration")) {
        nlohmann::json reg;
        reg["source"] = p.regSource;
        reg["matched"] = p.regMatched;
        reg["globalIds"] = prevState_.globalIds;
        saveJson((dir / "M11-registration.json").string(), reg);
        if (trajT_.size() >= 2)
            savePng((dir.parent_path() / "M11-trajectory.png").string(),
                    makeTrajectory(trajT_, 640, 640));
    }
}

// ============================================================================
// 端到端 P0：全局激光点云累积＋导出
// ============================================================================
int Harness::fcChCount(int i) const {
    return fc_ ? fc_->chCount(i) : -1;
}

int Harness::fcTombCount() const {
    return fc_ ? fc_->tombCount() : 0;
}

void Harness::accumulateGlobalLaser(const CycleProducts& p) {
    // 【五链 P2】fivechain 激活时融合已由 CUDA链2 完成（F 汇聚点）——main 免重复
    if (fcActive_) return;
    // 只融合配准成功帧的激光点——hold/skip 帧 R/T 是陈旧沿用值，
    // 设备已移动，此时用旧 R/T 变换激光点会放在错误位置（＝重影来源）
    if (p.regSource != "first" && p.regSource != "flow"
        && p.regSource != "fuse")
        return;
    if (!p.recon.d_points3d) return;
    // 【五链 P1】产线体素融合（算子内部完成 R/T 变换＋0.25mm 体素去重插入）；
    // host 口径计时（GPU 段随流异步完成——事件口径会跨步骤计入排队等待而虚高）
    timeBegin();
    auto res = voxelFuseOp_->Execute(*p.recon.d_points3d,
                                     cv::Matx33d(p.R),
                                     cv::Vec3d(p.T[0], p.T[1], p.T[2]),
                                     stream_);
    if (!res.success) {
        std::cout << "[fuse:voxel] Execute 失败: " << res.message << std::endl;
    } else {
        fuseInputTotal_ += static_cast<size_t>(res.inputCount);
        ++fuseFrameCount_;
        if (res.rehashHappened) {
            ++fuseRehashCount_;
            std::cout << "[fuse:voxel] rehash → " << res.hashCapacity
                      << " 槽 耗时 " << res.rehashTimeMs << "ms" << std::endl;
        }
        // 负载断言（设计 §七.6）：算子 >70% 自动扩容在前；扩容后仍越 0.75 线＝计数告警
        if (static_cast<double>(res.totalVoxelCount) >
            cfg_.fivechain.voxelLoadFactor * static_cast<double>(res.hashCapacity)) {
            ++fuseOverloadCount_;
            std::cout << "[fuse:voxel] 负载越限告警: voxels=" << res.totalVoxelCount
                      << " capacity=" << res.hashCapacity << std::endl;
        }
    }
    timeEnd("accumulate_global_laser", const_cast<CycleProducts&>(p), true, false);
}

size_t Harness::saveGlobalLaserPly(const std::string& path) const {
    // 【五链 P1】体素融合点云导出：设备侧 fusedXyz 一次性下载（downloadFusedXyz 模式）
    const auto ctx = voxelFuseOp_->GetDeviceContext();
    if (ctx.fusedPointCount == 0) return 0;
    std::vector<float> xyz(3 * ctx.fusedPointCount);
    try {
        // 【五链 P2】融合流可能非主流（fivechain=fS_）——按指针同步正确流
        cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(
            fcFuseStreamPtr_ ? *fcFuseStreamPtr_ : stream_));
        if (cudaMemcpy(xyz.data(), ctx.d_fusedXyz,
                       xyz.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return 0;
    } catch (...) { return 0; }
    std::ofstream ply(path);
    if (!ply) return 0;
    ply << "ply\nformat ascii 1.0\n";
    ply << "element vertex " << ctx.fusedPointCount << "\n";
    ply << "property float x\nproperty float y\nproperty float z\n";
    ply << "end_header\n";
    ply << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < ctx.fusedPointCount; ++i) {
        ply << xyz[3 * i] << " " << xyz[3 * i + 1] << " " << xyz[3 * i + 2] << "\n";
    }
    return ctx.fusedPointCount;
}

bool Harness::lastRT(double R[9], double T[3]) const {
    if (prevState_.empty()) return false;
    for (int i = 0; i < 9; ++i) R[i] = prevState_.R.val[i];
    for (int i = 0; i < 3; ++i) T[i] = prevState_.T.val[i];
    return true;
}

} // namespace scr


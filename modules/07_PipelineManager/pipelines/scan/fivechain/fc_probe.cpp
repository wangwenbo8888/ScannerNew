// fc_probe.cpp — 五链中间结果抽检转储实现（design: docs/plans/2026-11-03-五链中间结果抽检转储-design.md）
//
// 画法与字段遍历照抄沙箱 harness.cpp dumpCycle 段（modules/09_OperatorLib/
// TEST/speed_optimize/src/harness.cpp L1925-2524——同仓库移植，不是重写）：
// 产物命名对齐 scan_replay README §5 产物对照表，在线/离线可直接对拍。
// 两处 PNG 名按设计文档 §4 口径：L2-undistort.png（沙箱原名 L2-undistort_cuda.png）、
// L4-match.png（沙箱原名 L4-match_v3.png）——CSV 名与沙箱完全一致。
//
// 全部落盘经 scr::artifacts 工具箱；整体 try-catch，失败 JMW_LOG_WARN 不打断
// 主链（诊断件语义）。默认关＝maybeDump/appendRt 首行短路＝零开销。
#include "pipelines/scan/fivechain/fc_probe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

#include "pipelines/scan/fivechain/artifacts.h"
#include "jmw_logging.h"

namespace jmw::probe {

// ============================================================================
// 可视化绘图助手（自沙箱 harness.cpp L92-235 逐段搬用——dump 专用）
// ============================================================================
namespace {

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

/// ROI 子图网格拼贴（M2/M3 用；等比缩进 cell 方格，深底）
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
        const double scale = static_cast<double>(cell) / side;   // 等比缩放进 cell，居中贴黑底方块
        cv::resize(t, sq, cv::Size(), scale, scale);
        cv::Mat blk = cv::Mat::zeros(cell, cell, CV_8UC3);
        sq.copyTo(blk(cv::Rect((cell - sq.cols) / 2, (cell - sq.rows) / 2,
                               sq.cols, sq.rows)));
        blk.copyTo(canvas(cv::Rect(static_cast<int>(i) % cols * cell,
                                   static_cast<int>(i) / cols * cell, cell, cell)));
    }
    return canvas;
}

/// steger 点集：GPU 视图优先（沙箱同源），为空时回退摊平 host map
/// （StegerResult.centerPoints——07 侧 steger_fast 默认收集 host 点）
std::vector<cv::Point2f> stegerPoints(const calib::StegerResult& s,
                                      cv::cuda::Stream& cs) {
    if (s.d_centerPoints) {
        auto v = downloadP2(s.d_centerPoints, cs);
        if (!v.empty()) return v;
    }
    std::vector<cv::Point2f> out;
    for (const auto& kv : s.centerPoints)
        out.insert(out.end(), kv.second.begin(), kv.second.end());
    return out;
}

} // namespace

// ============================================================================
// ProbeConfig
// ============================================================================

ProbeConfig ProbeConfig::load() {
    ProbeConfig c;
    try {
        std::error_code ec;
        const std::filesystem::path p = std::filesystem::path("config") / "fc_dump.json";
        if (!std::filesystem::exists(p, ec)) return c;
        std::ifstream f(p);
        if (!f) return c;
        nlohmann::json j;
        f >> j;
        if (!j.contains("probe") || !j.at("probe").is_object()) return c;
        const auto& pr = j.at("probe");
        c.enabled = pr.value("enabled", false);
        c.maxFrames = pr.value("maxFrames", c.maxFrames);
        if (pr.contains("frames") && pr.at("frames").is_array())
            for (const auto& e : pr.at("frames"))
                if (e.is_number_integer()) c.frames.insert(e.get<long>());
    } catch (...) {
        // 开关/配置件失败静默回默认（关）——诊断件不得影响主链装配
        return ProbeConfig{};
    }
    return c;
}

// ============================================================================
// FrameProbe::maybeDump — 挑帧全套落盘（沙箱 dumpCycle full 段逐项移植）
// ============================================================================

void FrameProbe::maybeDump(long idx, const scr::CycleProductsCompat& p) {
    if (!cfg_.enabled || cfg_.frames.empty()) return;
    const long frameNo = idx + 1;                     // 帧号 1 起（与 probe.frames 口径一致）
    if (cfg_.frames.find(frameNo) == cfg_.frames.end()) return;

    try {
        char cycBuf[32];
        std::snprintf(cycBuf, sizeof(cycBuf), "cycle_%04ld", frameNo);
        const std::filesystem::path dir =
            std::filesystem::path("diagnostics") / "fc_probe" / cycBuf;
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);

        // 段级守护：单类产物失败只告警一次，不牵连其余产物
        auto step = [&, frameNo](const char* name, auto&& fn) {
            try {
                fn();
            } catch (const std::exception& e) {
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} {} 落盘失败: {}",
                             frameNo, name, e.what());
            } catch (...) {
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} {} 落盘未知异常",
                             frameNo, name);
            }
        };

        // dump 专用流（ROI 视图先物化后下载——沙箱同法）
        cv::cuda::Stream cs;

        // ------------------------------------------------------------
        // summary.json（恒写——沙箱口径；07 侧无 stepMs/stepSource 字段，略）
        // ------------------------------------------------------------
        step("summary.json", [&] {
            nlohmann::json sum;
            sum["cycle"] = frameNo;
            nlohmann::json steps = nlohmann::json::object();
            for (const auto& kv : p.stepSuccess_) steps[kv.first]["success"] = kv.second;
            for (const auto& kv : p.stepError_) steps[kv.first]["error"] = kv.second;
            sum["steps"] = steps;
            sum["markersDetected"] = p.markersDetected;
            sum["markersValid"] = p.markersValid;
            sum["regSource"] = p.regSource;
            sum["regMatched"] = p.regMatched;
            // 配准诊断：光流失败原因/误差——断链可诊断
            sum["flowDiag"]["matched"] = p.flowMatchedDiag;
            sum["flowDiag"]["rmse"] = p.flowRmseDiag;
            if (!p.flowMessageDiag.empty()) sum["flowDiag"]["message"] = p.flowMessageDiag;
            nlohmann::json rt = nlohmann::json::array();
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
            // 07 专属增列：真实帧号对照（fc idx ≠ 采集帧号——在线对拍定位用）
            sum["metaFrameId"] = p.metaFrameId;
            scr::saveJson((dir / "summary.json").string(), sum);
        });

        // ------------------------------------------------------------
        // 0- 原图（L|R 并合底版，后续叠加类可视化的同款画布基准）
        // ------------------------------------------------------------
        step("0-frames.png", [&] {
            cv::Mat img = mergeLR(p.frameGrayL, p.frameGrayR);
            if (!img.empty()) scr::savePng((dir / "0-frames.png").string(), img);
        });

        // ------------------------------------------------------------
        // 1- 掩膜分离（纯掩膜二值图，白=掩膜；L|R 并合）。GpuMat 先物化再下载
        //     （沙箱 maskLR 同法）。0-binary 用 sep.d_step1_binary（生产常驻字段：
        //     ① 高斯+阈值二值图，未分割——分离前端证据图）
        // ------------------------------------------------------------
        step("masks", [&] {
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
                if (!img.empty()) scr::savePng((dir / name).string(), img);
            };
            maskLR("L0-laser_mask.png", p.sepL.d_laserMask, p.sepR.d_laserMask);
            maskLR("M0-marking_mask.png", p.sepL.d_markingPointMask, p.sepR.d_markingPointMask);
            maskLR("0-binary.png", p.sepL.d_step1_binary, p.sepR.d_step1_binary);
        });

        // ------------------------------------------------------------
        // 2- steger：激光中心点（绿）叠原帧，L|R 并合。
        //     CSV 沙箱走 d_centerPoints（GPU）；07 侧 GPU 视图为空时回退摊平
        //     host centerPoints map（同点集同序）
        // ------------------------------------------------------------
        step("L1-steger", [&] {
            if (!p.stegerL.d_centerPoints && p.stegerL.centerPoints.empty()) return;
            if (p.stegerL.d_centerPoints) {
                scr::saveGpuMatP2((dir / "L1-steger_L.csv").string(),
                                  *p.stegerL.d_centerPoints, cs);
            } else {
                std::vector<cv::Point2f> flat;
                for (const auto& kv : p.stegerL.centerPoints)
                    flat.insert(flat.end(), kv.second.begin(), kv.second.end());
                scr::saveCsvP2f((dir / "L1-steger_L.csv").string(), flat);
            }
            auto stgL = stegerPoints(p.stegerL, cs);
            auto stgR = stegerPoints(p.stegerR, cs);
            cv::Mat cl = toBgr(p.frameGrayL), cr = toBgr(p.frameGrayR);
            drawPts(cl, stgL, cv::Scalar(0, 255, 0), 1);
            drawPts(cr, stgR, cv::Scalar(0, 255, 0), 1);
            cv::Mat img = mergeLR(cl, cr);
            if (!img.empty()) scr::savePng((dir / "L1-steger.png").string(), img);
        });

        // ------------------------------------------------------------
        // 3- undistort_cuda：矫正后激光点（绿）画暗画布（矫正系坐标，无法叠
        //     原帧），L|R 并合（沙箱原名 L2-undistort_cuda.png）
        // ------------------------------------------------------------
        step("L2-undistort", [&] {
            if (!p.undistCudaL.success) return;
            auto rl = downloadP2(p.undistCudaL.d_rectifiedPoints, cs);
            auto rr = downloadP2(p.undistCudaR.d_rectifiedPoints, cs);
            if (rl.empty() && rr.empty()) return;
            cv::Mat cl(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3,
                       cv::Scalar(20, 20, 20));
            cv::Mat cr = cl.clone();
            drawPts(cl, rl, cv::Scalar(0, 255, 0), 1);
            drawPts(cr, rr, cv::Scalar(0, 255, 0), 1);
            scr::savePng((dir / "L2-undistort.png").string(), mergeLR(cl, cr));
        });

        // ------------------------------------------------------------
        // 4- 极线插值：插值点（黄）叠原帧，L|R 并合
        // ------------------------------------------------------------
        step("L3-interp", [&] {
            if (!p.interpL.d_interpPoints) return;
            scr::saveGpuMatP2((dir / "L3-interp_L.csv").string(),
                              *p.interpL.d_interpPoints, cs);
            if (p.interpR.d_interpPoints)
                scr::saveGpuMatP2((dir / "L3-interp_R.csv").string(),
                                  *p.interpR.d_interpPoints, cs);
            auto il = downloadP2(p.interpL.d_interpPoints, cs);
            auto ir = downloadP2(p.interpR.d_interpPoints, cs);
            cv::Mat cl = toBgr(p.frameGrayL), cr = toBgr(p.frameGrayR);
            drawPts(cl, il, cv::Scalar(0, 255, 255), 1);
            drawPts(cr, ir, cv::Scalar(0, 255, 255), 1);
            cv::Mat img = mergeLR(cl, cr);
            if (!img.empty()) scr::savePng((dir / "L3-interp.png").string(), img);
        });

        // ------------------------------------------------------------
        // 5- v3 匹配：白=未匹配（左右各自插值点），彩色=匹配对（按激光线号
        //     同色，左点旁标注线号、编号与线同色；同线号左右同色），L|R 并合
        //     （沙箱原名 L4-match_v3.png）
        // ------------------------------------------------------------
        step("L4-match", [&] {
            if (!p.matchV3.d_matched_left) return;
            scr::saveGpuMatP2((dir / "L4-match_L.csv").string(),
                              *p.matchV3.d_matched_left, cs);
            if (p.matchV3.d_matched_right)
                scr::saveGpuMatP2((dir / "L4-match_R.csv").string(),
                                  *p.matchV3.d_matched_right, cs);
            auto ml = downloadP2(p.matchV3.d_matched_left, cs);
            auto mr = downloadP2(p.matchV3.d_matched_right, cs);
            auto ids = downloadIds(p.matchV3.d_matched_line_ids, cs);
            auto il = downloadP2(p.interpL.d_interpPoints, cs);
            auto ir = downloadP2(p.interpR.d_interpPoints, cs);
            cv::Mat cl(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3,
                       cv::Scalar(20, 20, 20));
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
            scr::savePng((dir / "L4-match.png").string(), mergeLR(cl, cr));
        });

        // ------------------------------------------------------------
        // 6- 激光 3D 点云（PLY，3D 查看器）：recon.d_points3d（CV_32FC3）
        // ------------------------------------------------------------
        step("L5-laser3d", [&] {
            if (!p.recon.d_points3d) return;
            cv::cuda::GpuMat contiguous;
            p.recon.d_points3d->copyTo(contiguous, cs);
            cv::Mat h;
            contiguous.download(h, cs);
            cs.waitForCompletion();
            std::vector<cv::Point3f> pts;
            if (!h.empty() && h.type() == CV_32FC3)
                pts.assign(h.begin<cv::Point3f>(), h.end<cv::Point3f>());
            scr::savePlyXYZf((dir / "L5-laser3d.ply").string(), pts);
        });

        // ------------------------------------------------------------
        // 7- ccl：组件伪彩标签＋白框＋编号，L|R 并合
        // ------------------------------------------------------------
        step("M1-ccl", [&] {
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
            if (!img.empty()) scr::savePng((dir / "M1-ccl.png").string(), img);
        });

        // ------------------------------------------------------------
        // 8- image_split：ROI 子图网格拼贴（L|R 各一拼贴再并合）
        // ------------------------------------------------------------
        step("M2-split", [&] {
            const cv::Mat colL = makeCollage(p.splitL.splitImages, 160, 36);
            const cv::Mat colR = makeCollage(p.splitR.splitImages, 160, 36);
            cv::Mat img = mergeLR(colL, colR);
            if (!img.empty()) scr::savePng((dir / "M2-split.png").string(), img);
        });

        // ------------------------------------------------------------
        // 9- zernike：ROI 拼贴上叠边缘点（红，ROI 局部系），L|R 并合
        //     （CSV 沙箱口径仅左：M3-zern_L.csv）
        // ------------------------------------------------------------
        step("M3-zernike", [&] {
            std::vector<scr::EdgePointCsv> eps;
            for (const auto& z : p.zernL)
                for (const auto& e : z.edgePoints)
                    eps.push_back({e.x, e.y, e.angle, e.amplitude, e.pixelX, e.pixelY});
            scr::saveCsvEdgePoints((dir / "M3-zern_L.csv").string(), eps);
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
            if (!img.empty()) scr::savePng((dir / "M3-zernike.png").string(), img);
        });

        // ------------------------------------------------------------
        // 10- image_merge：合并后边缘点按组着色叠原帧（大图系），L|R 并合
        // ------------------------------------------------------------
        step("M4-merge", [&] {
            if (!p.mergeL.success) return;
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
            if (!img.empty()) scr::savePng((dir / "M4-merge.png").string(), img);
        });

        // ------------------------------------------------------------
        // 11- undistort_cpu：L/R 矫正后点同图散点（蓝=左/红=右）
        //     ——行对齐证据（本身即合并对照）
        // ------------------------------------------------------------
        step("M5-undistort_cpu", [&] {
            if (!p.undistCpu.success || p.undistCpu.rectifiedPoints1.empty()) return;
            cv::Mat canvas(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3,
                           cv::Scalar(20, 20, 20));
            std::vector<cv::Point2f> lpts, rpts;
            lpts.reserve(p.undistCpu.rectifiedPoints1.size());
            rpts.reserve(p.undistCpu.rectifiedPoints2.size());
            for (const auto& q : p.undistCpu.rectifiedPoints1) lpts.emplace_back(float(q.x), float(q.y));
            for (const auto& q : p.undistCpu.rectifiedPoints2) rpts.emplace_back(float(q.x), float(q.y));
            drawPts(canvas, lpts, cv::Scalar(255, 0, 0), 1);
            drawPts(canvas, rpts, cv::Scalar(0, 0, 255), 1);
            scr::savePng((dir / "M5-undistort_cpu.png").string(), canvas);
        });

        // ------------------------------------------------------------
        // 12- ellipse_fit：椭圆轮廓＋中心十字＋组号（矫正系画布），L|R 并合
        // ------------------------------------------------------------
        step("M6-ellipse", [&] {
            auto dumpSide = [&](const std::vector<calib::EllipseFitCPUResult>& es,
                                const char* jsonName) {
                nlohmann::json arr = nlohmann::json::array();
                for (const auto& e : es)
                    if (e.success)
                        arr.push_back({{"groupId", e.groupId}, {"cx", e.centerX}, {"cy", e.centerY},
                                       {"major", e.majorAxis}, {"minor", e.minorAxis},
                                       {"angle", e.angle}, {"inliers", e.inlierCount}});
                scr::saveJson((dir / jsonName).string(), arr);
            };
            dumpSide(p.ellL, "M6-ellipse_L.json");
            dumpSide(p.ellR, "M6-ellipse_R.json");
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
            if (!img.empty()) scr::savePng((dir / "M6-ellipse.png").string(), img);
        });

        // ------------------------------------------------------------
        // 13- marker_match：中心点配对——匹配对同号同色（编号=左索引），
        //     未匹配灰点；右图逐对标注 dy（离极线行偏差）/d（视差）；左右并合
        // ------------------------------------------------------------
        step("M7-marker_match", [&] {
            if (!p.mm.success) return;
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
            scr::savePng((dir / "M7-marker_match.png").string(), mergeLR(canvasL, canvasR));
        });

        // ------------------------------------------------------------
        // 14- 标记点 3D（PLY，3D 查看器；conic 空间圆合成圆环点=
        //     reconstructedPoints，每 validMarker ringPointsForViz 点）
        // ------------------------------------------------------------
        step("M10-markers3d", [&] {
            std::vector<cv::Point3d> pts;
            for (const auto& m : p.conic.markers) {
                if (!m.validCircle) continue;
                for (const auto& q : m.reconstructedPoints)
                    pts.emplace_back(q.x, q.y, q.z);
            }
            scr::savePlyXYZd((dir / "M10-markers3d.ply").string(), pts);
        });

        // ------------------------------------------------------------
        // 15- 配准详情：R/T＋配准方式＋fuse/光流诊断（沙箱 M11 口径＋07 增列）
        // ------------------------------------------------------------
        step("M11-registration", [&] {
            nlohmann::json reg;
            reg["source"] = p.regSource;
            reg["matched"] = p.regMatched;
            nlohmann::json rArr = nlohmann::json::array();
            for (int i = 0; i < 9; ++i) rArr.push_back(p.R[i]);
            reg["R"] = rArr;
            reg["T"] = {p.T[0], p.T[1], p.T[2]};
            reg["usedFrameFuse"] = p.usedFrameFuse;
            reg["fuseSlowPath"] = p.fuseSlowPath;
            reg["flowDiag"]["matched"] = p.flowMatchedDiag;
            reg["flowDiag"]["rmse"] = p.flowRmseDiag;
            if (!p.flowMessageDiag.empty()) reg["flowDiag"]["message"] = p.flowMessageDiag;
            if (p.regFlow.success)
                reg["globalIds"] = p.regFlow.getGlobalIds();   // 沙箱 prevState_.globalIds 同源（光流帧）
            if (!p.fuseMarkerGids.empty())
                reg["fuseMarkerGids"] = p.fuseMarkerGids;      // 07 增列：fuse 帧逐点 gid（-1=未关联）
            scr::saveJson((dir / "M11-registration.json").string(), reg);
        });

        JMW_LOG_INFO("07-FiveChain", "[fc_probe] cycle {} 全链中间产物已落盘 → {}",
                     frameNo, dir.string());
    } catch (const std::exception& e) {
        JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} 落盘异常: {}", frameNo, e.what());
    } catch (...) {
        JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} 落盘未知异常", frameNo);
    }
}

// ============================================================================
// FrameProbe::appendRt — probe 开时逐帧位姿表（列格式同 JMW_FC_FRAME_DUMPS
// 的 rt.csv：帧,regSource,R0..R8,T0..T2，无表头；帧列＝idx+1 与 cycle 目录号一致）
// ============================================================================

void FrameProbe::appendRt(long idx, const scr::CycleProductsCompat& p) {
    if (!cfg_.enabled) return;
    try {
        const std::filesystem::path root =
            std::filesystem::path("diagnostics") / "fc_probe";
        std::error_code ec;
        std::filesystem::create_directories(root, ec);
        std::ofstream rt((root / "rt.csv").string(), std::ios::app);
        if (!rt) return;
        rt << (idx + 1) << ',' << p.regSource << ','
           << p.R[0] << ',' << p.R[1] << ',' << p.R[2] << ','
           << p.R[3] << ',' << p.R[4] << ',' << p.R[5] << ','
           << p.R[6] << ',' << p.R[7] << ',' << p.R[8] << ','
           << p.T[0] << ',' << p.T[1] << ',' << p.T[2] << '\n';
    } catch (...) {
        // 静默——诊断件不得打断主链
    }
}

} // namespace jmw::probe

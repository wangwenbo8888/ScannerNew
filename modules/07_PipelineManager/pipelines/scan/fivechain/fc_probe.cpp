// fc_probe.cpp — 五链中间结果抽检转储实现（design: docs/plans/2026-11-03-五链中间结果抽检转储-design.md）
//
// 画法与字段遍历照抄沙箱 harness.cpp dumpCycle 段（modules/09_OperatorLib/
// TEST/speed_optimize/src/harness.cpp L1925-2524——同仓库移植，不是重写）：
// 产物命名对齐 scan_replay README §5 产物对照表，在线/离线可直接对拍。
// 两处 PNG 名按设计文档 §4 口径：L2-undistort.png（沙箱原名 L2-undistort_cuda.png）、
// L4-match.png（沙箱原名 L4-match_v3.png）——CSV 名与沙箱完全一致。
//
// 【v2·快照制】maybeDump 全程零 CUDA：GPU 产物改由产出线程写入 ProbeSnapshot
//（fc_executor 1a/1b/chain2 段——该帧数据未被后续帧覆写的时机），此处只读快照
// 宿主数据＋prod 宿主字段；快照缺段＝该产物跳过＋一次 WARN（宁缺勿错）。
// 全部落盘经 scr::artifacts 工具箱；整体 try-catch，失败 JMW_LOG_WARN 不打断
// 主链（诊断件语义）。默认关＝maybeDump/appendRt 首行短路＝零开销。
#include "pipelines/scan/fivechain/fc_probe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

#include "pipelines/scan/fivechain/harness.h"   // CycleProductsCompat/算子结果真身
#include "pipelines/scan/fivechain/artifacts.h"
#include "jmw_logging.h"

namespace jmw::probe {

// ============================================================================
// 可视化绘图助手（自沙箱 harness.cpp L92-235 逐段搬用——dump 专用）
// ============================================================================
namespace {

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

/// 路径串取文件名（落盘回执日志用；兼容 / 与 \）
std::string fileNameOf(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
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
        if (tiles[i].empty()) continue;   // 空 tile 跳过（防 side=0 → scale=inf）
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

/// 【v2】快照截断：容量宽下载按精确计数 n 取前缀（n<=0 或快照空 → 空）
template <typename T>
std::vector<T> snapTrim(const std::vector<T>& v, long n) {
    std::vector<T> out;
    if (v.empty() || n <= 0) return out;
    const size_t k = std::min(v.size(), static_cast<size_t>(n));
    out.assign(v.begin(), v.begin() + static_cast<long>(k));
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
            for (const auto& e : pr.at("frames")) {
                // 单个坏值只跳过本值，不弃整配置（审查#9）；帧号合法域 [1, 1e7]
                if (!e.is_number_integer()) continue;
                long v = 0;
                try {
                    v = e.get<long>();
                } catch (...) {
                    continue;   // 超出 long 的无符号整数等——按非法值跳过
                }
                if (v < 1 || v > 10000000L) continue;
                c.frames.insert(v);
            }
    } catch (...) {
        // 开关/配置件失败静默回默认（关）——诊断件不得影响主链装配
        return ProbeConfig{};
    }
    return c;
}

// ============================================================================
// FrameProbe::maybeDump — 挑帧全套落盘（快照制 v2：纯宿主读写，零 CUDA）
// ============================================================================

void FrameProbe::maybeDump(long idx, const scr::CycleProductsCompat& p,
                           const ProbeSnapshot& snap) {
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

        // 落盘回执（审查#1）：全部 save* 检查返回值，失败 WARN（含产物文件名）
        // 并计入 failCount——末尾"已落盘"INFO 仅全数成功时打
        unsigned failCount = 0;
        auto savePngW = [&](const std::string& path, const cv::Mat& img) {
            if (!scr::savePng(path, img)) {
                ++failCount;
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} 写 {} 失败",
                             frameNo, fileNameOf(path));
            }
        };
        auto saveJsonW = [&](const std::string& path, const nlohmann::json& j) {
            if (!scr::saveJson(path, j)) {
                ++failCount;
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} 写 {} 失败",
                             frameNo, fileNameOf(path));
            }
        };
        auto saveCsvP2fW = [&](const std::string& path,
                               const std::vector<cv::Point2f>& pts) {
            if (!scr::saveCsvP2f(path, pts)) {
                ++failCount;
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} 写 {} 失败",
                             frameNo, fileNameOf(path));
            }
        };
        auto saveCsvEdgeW = [&](const std::string& path,
                                const std::vector<scr::EdgePointCsv>& pts) {
            if (!scr::saveCsvEdgePoints(path, pts)) {
                ++failCount;
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} 写 {} 失败",
                             frameNo, fileNameOf(path));
            }
        };
        auto savePlyFW = [&](const std::string& path,
                             const std::vector<cv::Point3f>& pts) {
            if (!scr::savePlyXYZf(path, pts)) {
                ++failCount;
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} 写 {} 失败",
                             frameNo, fileNameOf(path));
            }
        };
        auto savePlyDW = [&](const std::string& path,
                             const std::vector<cv::Point3d>& pts) {
            if (!scr::savePlyXYZd(path, pts)) {
                ++failCount;
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} 写 {} 失败",
                             frameNo, fileNameOf(path));
            }
        };

        // 段级守护：单类产物失败只告警并计数，不牵连其余产物
        auto step = [&, frameNo](const char* name, auto&& fn) {
            try {
                fn();
            } catch (const std::exception& e) {
                ++failCount;
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} {} 落盘失败: {}",
                             frameNo, name, e.what());
            } catch (...) {
                ++failCount;
                JMW_LOG_WARN("07-FiveChain", "[fc_probe] cycle {} {} 落盘未知异常",
                             frameNo, name);
            }
        };
        // 【v2】快照缺段：该产物跳过＋一次 WARN（宁缺勿错——GPU 视图已过期，
        // 绝不回退直读；缺段原因＝产出线程下载失败/该帧该步未成功）
        auto warnMiss = [frameNo](const char* what) {
            JMW_LOG_WARN("07-FiveChain",
                         "[fc_probe] cycle {} 快照缺 {}——该产物跳过（宁缺勿错）",
                         frameNo, what);
        };

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
            saveJsonW((dir / "summary.json").string(), sum);
        });

        // ------------------------------------------------------------
        // 0- 原图（L|R 并合底版，后续叠加类可视化的同款画布基准）
        // ------------------------------------------------------------
        step("0-frames.png", [&] {
            cv::Mat img = mergeLR(p.frameGrayL, p.frameGrayR);
            if (!img.empty()) savePngW((dir / "0-frames.png").string(), img);
        });

        // ------------------------------------------------------------
        // 1- 掩膜分离（纯掩膜二值图，白=掩膜；L|R 并合）。【v2】快照宿主 Mat
        //     直拼（1a 段产出线程下载）。0-binary 用分离前端二值图（生产常驻
        //     证据图：高斯+阈值二值化，未分割）。
        // ------------------------------------------------------------
        step("masks", [&] {
            auto maskPair = [&](const char* name, const cv::Mat& l, const cv::Mat& r,
                                const char* what) {
                if (l.empty() && r.empty()) {
                    warnMiss(what);
                    return;
                }
                // 【S-5】半对：单侧缺——另一侧有效仍出图（mergeLR 黑屏侧补齐），
                // 但左右不对称本身是分离/下载异常证据，显式告警防误读为算法问题
                if (l.empty() != r.empty())
                    JMW_LOG_WARN("07-FiveChain",
                                 "[fc_probe] cycle {} {} 半对缺失（L={},R={}）",
                                 frameNo, what,
                                 l.empty() ? "缺" : "有", r.empty() ? "缺" : "有");
                cv::Mat img = mergeLR(l, r);
                if (!img.empty()) savePngW((dir / name).string(), img);
            };
            maskPair("L0-laser_mask.png", snap.laserMaskL, snap.laserMaskR, "laserMask");
            maskPair("M0-marking_mask.png", snap.markingMaskL, snap.markingMaskR, "markingMask");
            maskPair("0-binary.png", snap.binaryL, snap.binaryR, "binary");
        });

        // ------------------------------------------------------------
        // 2- steger：激光中心点（绿）叠原帧，L|R 并合。【v2】快照容量宽点集按
        //     totalPointCount 截断；快照空时回退摊平 host centerPoints map
        //     （SetCollectHostPoints=true 的收集模式——在线默认空，仅离线口径）
        // ------------------------------------------------------------
        step("L1-steger", [&] {
            auto take = [&](const std::vector<cv::Point2f>& snapPts,
                            const calib::StegerResult& s) {
                std::vector<cv::Point2f> out = snapTrim(snapPts, s.totalPointCount);
                if (!out.empty()) return out;
                for (const auto& kv : s.centerPoints)   // 回退：宿主 map（收集模式）
                    out.insert(out.end(), kv.second.begin(), kv.second.end());
                return out;
            };
            const auto stgL = take(snap.stegerL, p.stegerL);
            const auto stgR = take(snap.stegerR, p.stegerR);
            if (stgL.empty() && stgR.empty()) {
                warnMiss("steger");
                return;
            }
            // CSV（沙箱口径仅左）：左目有源即写
            if (!stgL.empty())
                saveCsvP2fW((dir / "L1-steger_L.csv").string(), stgL);
            cv::Mat cl = toBgr(p.frameGrayL), cr = toBgr(p.frameGrayR);
            drawPts(cl, stgL, cv::Scalar(0, 255, 0), 1);
            drawPts(cr, stgR, cv::Scalar(0, 255, 0), 1);
            cv::Mat img = mergeLR(cl, cr);
            if (!img.empty()) savePngW((dir / "L1-steger.png").string(), img);
        });

        // ------------------------------------------------------------
        // 3- undistort_cuda：矫正后激光点（绿）画暗画布（矫正系坐标，无法叠
        //     原帧），L|R 并合（沙箱原名 L2-undistort_cuda.png）。【v2】快照
        //     容量宽点集按 steger totalPointCount 截断（矫正点与中心点一一对应）
        // ------------------------------------------------------------
        step("L2-undistort", [&] {
            // 左右独立判断（审查#6——失败侧不取点，另一侧有效仍出图）
            auto rl = p.undistCudaL.success
                          ? snapTrim(snap.undistL, p.stegerL.totalPointCount)
                          : std::vector<cv::Point2f>();
            auto rr = p.undistCudaR.success
                          ? snapTrim(snap.undistR, p.stegerR.totalPointCount)
                          : std::vector<cv::Point2f>();
            if (rl.empty() && rr.empty()) {
                warnMiss("undistort");
                return;
            }
            cv::Mat cl(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3,
                       cv::Scalar(20, 20, 20));
            cv::Mat cr = cl.clone();
            drawPts(cl, rl, cv::Scalar(0, 255, 0), 1);
            drawPts(cr, rr, cv::Scalar(0, 255, 0), 1);
            savePngW((dir / "L2-undistort.png").string(), mergeLR(cl, cr));
        });

        // ------------------------------------------------------------
        // 4- 极线插值：插值点（黄）叠原帧，L|R 并合。【v2】快照容量宽点集按
        //     interpCount 截断（精确计数在 1b 排水后 FinishInterp 才知）
        // ------------------------------------------------------------
        step("L3-interp", [&] {
            const auto il = snapTrim(snap.interpL, p.interpL.interpCount);
            const auto ir = snapTrim(snap.interpR, p.interpR.interpCount);
            if (il.empty() && ir.empty()) {
                warnMiss("interp");
                return;
            }
            if (!il.empty())
                saveCsvP2fW((dir / "L3-interp_L.csv").string(), il);
            if (!ir.empty())
                saveCsvP2fW((dir / "L3-interp_R.csv").string(), ir);
            cv::Mat cl = toBgr(p.frameGrayL), cr = toBgr(p.frameGrayR);
            drawPts(cl, il, cv::Scalar(0, 255, 255), 1);
            drawPts(cr, ir, cv::Scalar(0, 255, 255), 1);
            cv::Mat img = mergeLR(cl, cr);
            if (!img.empty()) savePngW((dir / "L3-interp.png").string(), img);
        });

        // ------------------------------------------------------------
        // 5- v3 匹配：白=未匹配（左右各自插值点），彩色=匹配对（按激光线号
        //     同色，左点旁标注线号、编号与线同色；同线号左右同色），L|R 并合
        //     （沙箱原名 L4-match_v3.png）。【v2】匹配对来自 chain2 段快照
        //     （在线 prod.matchV3 从不赋值——E 段唯一来源即快照）
        // ------------------------------------------------------------
        step("L4-match", [&] {
            if (snap.matchL.empty() && snap.matchR.empty()) {
                warnMiss("match_v3");
                return;
            }
            const auto& ml = snap.matchL;
            const auto& mr = snap.matchR;
            const auto& ids = snap.matchLine;
            saveCsvP2fW((dir / "L4-match_L.csv").string(), ml);
            saveCsvP2fW((dir / "L4-match_R.csv").string(), mr);
            const auto il = snapTrim(snap.interpL, p.interpL.interpCount);
            const auto ir = snapTrim(snap.interpR, p.interpR.interpCount);
            cv::Mat cl(p.frameGrayL.rows, p.frameGrayL.cols, CV_8UC3,
                       cv::Scalar(20, 20, 20));
            cv::Mat cr = cl.clone();
            drawPts(cl, il, cv::Scalar(255, 255, 255), 2);      // 白=未匹配左点
            drawPts(cr, ir, cv::Scalar(255, 255, 255), 2);      // 白=未匹配右点
            const size_t n = std::min({ml.size(), mr.size(), ids.size()});
            // 颜色缓存（审查#4）：万级点逐点 paletteColor＝逐点 1×1 Mat 分配——
            // 按 line id 缓存后每线只算一次
            std::map<int, cv::Scalar> colorOf;
            auto colorById = [&colorOf](int id) -> const cv::Scalar& {
                auto it = colorOf.find(id);
                if (it != colorOf.end()) return it->second;
                return colorOf.emplace(id, paletteColor(id)).first->second;
            };
            // 编号稀疏化：每条线号首点＋每隔 60 点标一次（万级点全标会糊成一片）
            std::map<int, int> lidSeen;
            for (size_t i = 0; i < n; ++i) {
                const cv::Scalar& c = colorById(ids[i]);   // 同线号同色
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
            savePngW((dir / "L4-match.png").string(), mergeLR(cl, cr));
        });

        // ------------------------------------------------------------
        // 6- 激光 3D 点云（PLY，3D 查看器）。【v2】chain2 段 E 块宿主拷出快照
        //     （在线 prod.recon 从不赋值——E 段唯一来源即快照）
        // ------------------------------------------------------------
        step("L5-laser3d", [&] {
            if (snap.laser3d.empty()) {
                warnMiss("laser3d");
                return;
            }
            savePlyFW((dir / "L5-laser3d.ply").string(), snap.laser3d);
        });

        // ------------------------------------------------------------
        // 7- ccl：组件伪彩＋白框＋编号，L|R 并合。【v2·M1 修正】在线 CCL＝CPU
        //     版（runCclCpuFromRuns——RLE 行程驱动）不产 d_labeledMask 像素级
        //     标签——改宿主 components 列表画 bbox 框＋编号伪彩（编号色=
        //     paletteColor(label)，与沙箱标签伪彩同语义），底图=灰帧
        // ------------------------------------------------------------
        step("M1-ccl", [&] {
            auto cclCanvas = [&](const calib::RegionAnalysisResult& r,
                                 const cv::Mat& gray) {
                cv::Mat canvas;
                if (r.components.empty() || gray.empty()) return canvas;
                canvas = toBgr(gray);
                for (const auto& comp : r.components) {
                    const cv::Scalar c = paletteColor(comp.label);
                    cv::rectangle(canvas,
                                  cv::Rect(comp.boundingBoxX, comp.boundingBoxY,
                                           comp.boundingBoxWidth, comp.boundingBoxHeight),
                                  c, 2);
                    cv::putText(canvas, std::to_string(comp.label),
                                cv::Point(comp.boundingBoxX,
                                          std::max(12, comp.boundingBoxY - 4)),
                                0, 0.6, c, 2);
                }
                return canvas;
            };
            cv::Mat img = mergeLR(cclCanvas(p.cclL, p.frameGrayL),
                                  cclCanvas(p.cclR, p.frameGrayR));
            if (!img.empty()) savePngW((dir / "M1-ccl.png").string(), img);
        });

        // ------------------------------------------------------------
        // 8- image_split：ROI 子图网格拼贴（L|R 各一拼贴再并合）
        // ------------------------------------------------------------
        step("M2-split", [&] {
            const cv::Mat colL = makeCollage(p.splitL.splitImages, 160, 36);
            const cv::Mat colR = makeCollage(p.splitR.splitImages, 160, 36);
            cv::Mat img = mergeLR(colL, colR);
            if (!img.empty()) savePngW((dir / "M2-split.png").string(), img);
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
            saveCsvEdgeW((dir / "M3-zern_L.csv").string(), eps);
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
            if (!img.empty()) savePngW((dir / "M3-zernike.png").string(), img);
        });

        // ------------------------------------------------------------
        // 10- image_merge：合并后边缘点按组着色叠原帧（大图系），L|R 并合
        // ------------------------------------------------------------
        step("M4-merge", [&] {
            if (!p.mergeL.success && !p.mergeR.success) return;   // 左右独立（同 #6 口径）
            // 颜色缓存：mergedEdgePoints 亦可达千级点——按 groupId 缓存（同 #4）
            std::map<int, cv::Scalar> colorOf;
            auto colorByGroup = [&colorOf](int id) -> const cv::Scalar& {
                auto it = colorOf.find(id);
                if (it != colorOf.end()) return it->second;
                return colorOf.emplace(id, paletteColor(id)).first->second;
            };
            auto mergeCanvas = [&](const calib::ImageMergeCPUResult& m,
                                   const cv::Mat& gray) {
                cv::Mat canvas = toBgr(gray);
                for (size_t i = 0; i < m.mergedEdgePoints.size(); ++i) {
                    const auto& e = m.mergedEdgePoints[i];
                    const int g = (i < m.groupIds.size()) ? m.groupIds[i] : 0;
                    const int x = cvRound(e.x), y = cvRound(e.y);
                    if (x >= 0 && y >= 0 && x < canvas.cols && y < canvas.rows)
                        cv::circle(canvas, cv::Point(x, y), 1, colorByGroup(g), -1);
                }
                return canvas;
            };
            cv::Mat img = mergeLR(mergeCanvas(p.mergeL, p.frameGrayL),
                                  mergeCanvas(p.mergeR, p.frameGrayR));
            if (!img.empty()) savePngW((dir / "M4-merge.png").string(), img);
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
            savePngW((dir / "M5-undistort_cpu.png").string(), canvas);
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
                saveJsonW((dir / jsonName).string(), arr);
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
            if (!img.empty()) savePngW((dir / "M6-ellipse.png").string(), img);
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
            savePngW((dir / "M7-marker_match.png").string(), mergeLR(canvasL, canvasR));
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
            savePlyDW((dir / "M10-markers3d.ply").string(), pts);
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
            saveJsonW((dir / "M11-registration.json").string(), reg);
        });

        // 回执汇总（审查#1）：全部 step 且全部写盘成功才报"已落盘"；
        // 有失败则只报一条汇总（逐文件/逐段告警已在上方打）
        if (failCount == 0) {
            JMW_LOG_INFO("07-FiveChain", "[fc_probe] cycle {} 全链中间产物已落盘 → {}",
                         frameNo, dir.string());
        } else {
            JMW_LOG_WARN("07-FiveChain",
                         "[fc_probe] cycle {} 落盘完成（{} 类失败，详见上方告警）→ {}",
                         frameNo, failCount, dir.string());
        }
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
        if (!dirEnsured_) {   // 目录只建一次（审查#3——原每帧建目录）
            std::error_code ec;
            std::filesystem::create_directories(root, ec);
            dirEnsured_ = true;
        }
        if (!rtStarted_) {    // 首帧 trunc 覆盖旧会话残留表（审查#2——原只 append
            std::ofstream trunc((root / "rt.csv").string(), std::ios::trunc);
            rtStarted_ = true;   // 永不截断＝跨会话脏数据混入）；此后只 append
        }
        std::ofstream rt((root / "rt.csv").string(), std::ios::app);
        if (!rt) return;
        // 口径注（审查#8）：帧列＝idx+1（与 cycle_NNNN 目录号/probe.frames 同基）；
        // 旧 JMW_FC_FRAME_DUMPS 的 rt.csv 写 j->idx（0 基）——两表帧号相差 1，对拍勿混
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

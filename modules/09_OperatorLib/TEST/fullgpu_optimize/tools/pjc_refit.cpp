// pjc_refit.cpp — "fcscan 平替"：照片→多线重拟合→CMTT 表再生＋对拍（离线工具）
// ============================================================================
// 背景：温度补偿-专题-CMTT表uR随线号偏差成因调查-2026-11-08.md §10.1——根治＝
//   用修好 F8 雅可比缺口的 ExecuteMultiLine 重拟合 20 姿态→重出表。
//   生产工具 fcscan 源码不在仓，但全部算法零件在仓（09 沙箱 ops/），
//   本工具把散着的零件按 LaserChain.cpp / laser_calib_cli.cpp / 金标准测试
//   里已验证的配方拧起来。factory_calib 只读（人工授权），输出全落本目录 out\。
//
// 三种模式：
//   identity — 不碰照片：date_in 生产 json → CurveMapTempTableGenerator 多档再生
//              → 与现产 188MB bin 按档温逐字节对拍（锁死"表生成口径"）
//   refit    — factory_calib\data_in\laser 20 姿态×2管照片 → 4-1..4-8 链 →
//              lines[25][pose] → ExecuteMultiLine（--f8 1 开右目残差）→
//              laser_calib_refit.json ＋ pre/post 逐线诊断（pre 应复现实测斜坡形状）
//   regen    — refit 产物（或 --laser-json 指定）→ 参考档表再生 → bin＋对拍报告
//
// 用法示例（fullgpu_optimize 目录）：
//   pjc_refit --mode identity \
//     --date-in date_in/laser-mark1688-1006
//   pjc_refit --mode refit --f8 0 \
//     --factory-data ../../../../../factory_calib/data_in/laser \
//     --date-in date_in/laser-mark1688-1006
//   pjc_refit --mode regen --date-in date_in/laser-mark1688-1006
//
// 2026-11-08 创建（根治方案第 1-4 步的仓内执行体）。
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <set>
#include <string>
#include <vector>

// —— 09 沙箱算子（speed_ops 公开 include 各算子自目录）——
#include "mask_extract_cuda.h"          // 4-1 MaskExtractCUDA
#include "region_analyze_cuda.h"        // 4-2 RegionAnalyzerCUDA
#include "laser_label_cuda.h"           // 4-3 LaserLabelerCUDA
#include "steger_extract_cuda.h"        // 4-4 StegerExtractorCUDA（StegerParams 公共）
#include "undistort_points_cuda.h"      // 4-5 UndistortPointsCuda
#include "epipolar_interp_cuda.h"       // 4-6 EpipolarInterpCuda
#include "laser_match_cuda.h"           // 4-7 LaserMatchCuda
#include "laser_reconstruct_cuda.h"     // 4-8 LaserReconstructCuda
#include "projector_joint_calib.h"      // 多线联合拟合（含 2026-11-08 F8 补丁）
#include "curve_map_temp_table.h"       // CMTT 批量生成器
#include "cmtt_container.h"             // CmttBuilder/Reader
#include "laser_extrinsic_compensate_cpu.h"  // LaserExtrinsicCompensateCPUResult

namespace fs = std::filesystem;
using nlohmann::json;
using namespace calib;

namespace {

// ============================================================================
// 小工具
// ============================================================================
json readJson(const fs::path& p, std::string& err) {
    std::ifstream f(p);
    if (!f) { err = "cannot open " + p.string(); return {}; }
    try {
        return json::parse(f);
    } catch (const std::exception& e) {
        err = std::string("parse ") + p.string() + ": " + e.what();
        return {};
    }
}

cv::Mat parseMat64F(const json& j) {
    const int rows = static_cast<int>(j.size());
    const int cols = static_cast<int>(j.at(0).size());
    cv::Mat m(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m.at<double>(r, c) = j.at(r).at(c).get<double>();
    return m;
}

std::vector<uint8_t> readBin(const fs::path& p, std::string& err) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { err = "cannot open " + p.string(); return {}; }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
}

/// 逐线向量对 lid 的线性回归斜率（px/线；NaN 项跳过——usedLineIdx 可能缺线）
double slopeOf(const std::vector<double>& v) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (std::isnan(v[i])) continue;
        const double x = static_cast<double>(i + 1);
        sx += x; sy += v[i]; sxx += x * x; sxy += x * v[i]; n += 1.0;
    }
    if (n < 2) return 0.0;
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
/// 表口径红线：rowStep 随标定走（现产 0.67；重拟合后由 --row-step 显式传）
double rowStepForJson() { return 0.67; }

// ============================================================================
// 相机/激光 JSON 装载
// ============================================================================

struct FactoryCamera {
    cv::Mat K1, D1, K2, D2, R1, R2, P1, P2, Q, Rst, Tst;
    cv::Size imageSize{0, 0};
};

FactoryCamera loadFactoryCamera(const fs::path& p, std::string& err) {
    const json j = readJson(p, err);
    if (!err.empty()) return {};
    FactoryCamera c;
    c.imageSize = cv::Size(j.at("imageSize")[0].get<int>(),
                           j.at("imageSize")[1].get<int>());
    const json& intr = j.at("intrinsic");
    c.K1 = parseMat64F(intr.at("left").at("camera_matrix"));
    c.D1 = parseMat64F(intr.at("left").at("dist_coeffs"));
    c.K2 = parseMat64F(intr.at("right").at("camera_matrix"));
    c.D2 = parseMat64F(intr.at("right").at("dist_coeffs"));
    const json& ext = j.at("extrinsic");
    c.Rst = parseMat64F(ext.at("R"));
    c.Tst = parseMat64F(ext.at("T"));
    const json& rec = j.at("rectify");
    c.R1 = parseMat64F(rec.at("R1"));
    c.R2 = parseMat64F(rec.at("R2"));
    c.P1 = parseMat64F(rec.at("P1"));
    c.P2 = parseMat64F(rec.at("P2"));
    c.Q = parseMat64F(rec.at("Q"));
    return c;
}

/// date_in 生产激光 json → 25 条曲线 + projectorT + rowStep
struct LaserFit {
    std::vector<ImplicitCurve> curves;   // coeffs 已是去归一化像素域
    cv::Vec3d projectorT{0, 0, 0};
    cv::Point2d principalPoint{0, 0};    // 【eval-prod】pjc.principalPoint
    double rowStep = 0.67;
    double f = 0, baseline = 0;
    double refTemp = 22.5;
    std::vector<double> urPrePerLine, urPostPerLine;
};

LaserFit loadLaserFit(const fs::path& p, std::string& err) {
    const json j = readJson(p, err);
    if (!err.empty()) return {};
    LaserFit s;
    const json& pjc = j.at("pjc");
    for (const auto& cj : pjc.at("emissionCurves")) {
        ImplicitCurve c;
        for (int k = 0; k < 6; ++k) c.coeffs[k] = cj.at("coeffs").at(k).get<double>();
        s.curves.push_back(c);
    }
    for (int i = 0; i < 3; ++i) s.projectorT[i] = pjc.at("projectorT").at(i).get<double>();
    if (pjc.contains("principalPoint") && pjc.at("principalPoint").size() >= 2) {
        s.principalPoint.x = pjc.at("principalPoint").at(0).get<double>();
        s.principalPoint.y = pjc.at("principalPoint").at(1).get<double>();
    }
    s.rowStep = pjc.value("epipolarRowStep", 0.67);
    s.f = pjc.value("f", 0.0);
    s.baseline = pjc.value("baseline", 0.0);
    if (pjc.contains("urBiasPrePerLine"))
        for (const auto& v : pjc.at("urBiasPrePerLine")) s.urPrePerLine.push_back(v.get<double>());
    if (pjc.contains("urBiasPostPerLine"))
        for (const auto& v : pjc.at("urBiasPostPerLine")) s.urPostPerLine.push_back(v.get<double>());
    return s;
}

/// date_in 相机 json 的 61 档 stereoRectifyTempTable → Generate 的 rectifyTable
StereoRectifyTempTableResult loadRectifyTable(const fs::path& p, std::string& err) {
    const json j = readJson(p, err);
    if (!err.empty()) return {};
    StereoRectifyTempTableResult r;
    r.success = true;
    r.referenceTemp = j.value("referenceTemp", 22.5);
    const json& tab = j.at("stereoRectifyTempTable");
    if (tab.contains("referenceTemperature") && tab.at("referenceTemperature").is_number())
        r.referenceTemp = tab.at("referenceTemperature").get<double>();
    for (const auto& e : tab.at("table")) {
        StereoRectifyTempEntry t;
        t.temperature = e.at("temperature").get<double>();
        t.P1 = parseMat64F(e.at("P1"));
        t.P2 = parseMat64F(e.at("P2"));
        const json& vr = e.at("validRoiLeft");
        t.validRoiLeft = cv::Rect(vr.at("x").get<int>(), vr.at("y").get<int>(),
                                  vr.at("w").get<int>(), vr.at("h").get<int>());
        r.table.push_back(std::move(t));
    }
    r.tableSize = static_cast<int>(r.table.size());
    return r;
}

/// date_in 激光 json 的 laserExtrinsicTempTable.virtualToLeft → Generate 的激光外参表
/// ⚠ 实测格式：virtualToLeft 是对象（含 table 数组），逐条 T 为空格分隔字符串
LaserExtrinsicCompensateCPUResult loadLaserExtrinTable(const fs::path& p,
                                                       std::string& err) {
    const json j = readJson(p, err);
    if (!err.empty()) return {};
    LaserExtrinsicCompensateCPUResult l;
    l.success = true;
    const json& vl = j.at("laserExtrinsicTempTable").at("virtualToLeft");
    l.referenceTemp = vl.value("referenceTemperature", 22.5);
    l.leftResult.success = true;
    l.leftResult.referenceTemp = l.referenceTemp;
    for (const auto& e : vl.at("table")) {
        ExtrinsicCompensatedEntry t;
        t.temperature = e.at("temperature").get<double>();
        const json& Tj = e.at("T");
        if (Tj.is_array()) {
            for (int i = 0; i < 3; ++i) t.T[i] = Tj.at(i).get<double>();
        } else if (Tj.is_string()) {   // 兼容空格分隔串格式
            std::istringstream ss(Tj.get<std::string>());
            ss >> t.T[0] >> t.T[1] >> t.T[2];
            if (!ss) { err = "virtualToLeft T 解析失败"; return {}; }
        } else { err = "virtualToLeft T 格式未知"; return {}; }
        l.leftResult.table.push_back(std::move(t));
    }
    return l;
}

// ============================================================================
// identity / regen 共用：Generate + 按档温逐字节对拍
// ============================================================================

struct RegenArgs {
    fs::path dateIn;          // date_in/laser-mark1688-1006
    fs::path laserJson;       // 曲线来源（identity=现产 json；regen=refit 产物）
    fs::path outBin;          // 再生表落盘（空=不落）
    float halfRange = 28.0f;  // ±28℃@4℃步 → 15 档（覆盖现产 10 档）
    float tempStep = 4.0f;
    float depthMin = 200.0f, depthMax = 500.0f;
    bool compare = true;
};

int runGenerateAndCompare(const RegenArgs& a) {
    std::string err;
    const fs::path camJson = a.dateIn / "camera_calib.json";
    std::cerr << "[gen] 装载 json...\n";
    const LaserFit fit = loadLaserFit(a.laserJson, err);
    if (!err.empty()) { std::cerr << "[refit] " << err << "\n"; return 1; }
    std::cerr << "[gen] laser json OK (curves=" << fit.curves.size() << ")\n";
    const StereoRectifyTempTableResult rect = loadRectifyTable(camJson, err);
    if (!err.empty()) { std::cerr << "[refit] " << err << "\n"; return 1; }
    std::cerr << "[gen] camera rectify OK (tiers=" << rect.table.size() << ")\n";
    LaserExtrinsicCompensateCPUResult extr =
        loadLaserExtrinTable(a.dateIn / "laser_calib.json", err);
    if (!err.empty()) { std::cerr << "[refit] " << err << "\n"; return 1; }
    std::cerr << "[gen] laser extrin OK (tiers=" << extr.leftResult.table.size() << ")\n";

    std::cout << "[gen] curves=" << fit.curves.size()
              << " projectorT=" << fit.projectorT[0] << "," << fit.projectorT[1]
              << "," << fit.projectorT[2] << " rowStep=" << fit.rowStep
              << " rectTiers=" << rect.table.size()
              << " extrTiers=" << extr.leftResult.table.size() << "\n";

    CurveMapTempTableGenParams gp;
    gp.referenceTemp = 22.5;
    gp.tempHalfRange = a.halfRange;
    gp.tempStep = a.tempStep;
    gp.rowStep = static_cast<float>(fit.rowStep);
    gp.depthMin = a.depthMin;
    gp.depthMax = a.depthMax;
    std::cerr << "[gen] Generate 开始（15 档 × 8.7M 条目，分钟级）...\n";
    CurveMapTempTableGenerator gen(gp);
    const auto r = gen.Generate(fit.curves, rect, extr, cv::Size(1688, 1400));
    if (!r.success) {
        std::cerr << "[gen] Generate 失败: " << r.message << "\n";
        return 1;
    }
    std::cout << "[gen] 再生 " << r.tiers.size() << " 档，sidecar "
              << r.sidecarBytes.size() << " B\n";

    if (!a.outBin.empty()) {
        std::ofstream f(a.outBin, std::ios::binary);
        f.write(reinterpret_cast<const char*>(r.sidecarBytes.data()),
                static_cast<std::streamsize>(r.sidecarBytes.size()));
        std::cout << "[gen] 落盘 " << a.outBin << "\n";
    }

    if (!a.compare) return 0;
    // —— 与现产 bin 按档温逐字节对拍 ——
    CmttReader mine;
    if (!mine.load(r.sidecarBytes.data(), r.sidecarBytes.size())) {
        std::cerr << "[cmp] 再生 sidecar 自校验失败\n";
        return 1;
    }
    const std::vector<uint8_t> prod =
        readBin(a.dateIn / "curve_map_temp_table.bin", err);
    if (!err.empty()) { std::cerr << "[cmp] " << err << "\n"; return 1; }
    CmttReader prd;
    if (!prd.load(prod.data(), prod.size())) {
        std::cerr << "[cmp] 现产 bin 校验失败\n";
        return 1;
    }
    int pass = 0, fail = 0, skip = 0;
    for (size_t i = 0; i < prd.tierCount(); ++i) {
        const double tC = prd.tierTemperature(i);
        const auto mineIdx = mine.nearestTier(tC);
        if (!mineIdx || std::fabs(mine.tierTemperature(*mineIdx) - tC) > 0.01) {
            std::cout << "[cmp] " << tC << "C: 再生无对应档 SKIP\n";
            ++skip;
            continue;
        }
        const auto a1 = prd.tierBytes(i);
        const auto a2 = mine.tierBytes(*mineIdx);
        if (!a1 || !a2) { std::cout << "[cmp] " << tC << "C: 载荷缺失 FAIL\n"; ++fail; continue; }
        if (a1->size() == a2->size() &&
            std::memcmp(a1->data(), a2->data(), a1->size()) == 0) {
            std::cout << "[cmp] " << tC << "C: 逐字节一致 PASS ("
                      << a1->size() << " B"
                      << (prd.isClamped(i) ? ", clamped" : "") << ")\n";
            ++pass;
        } else {
            std::cout << "[cmp] " << tC << "C: 不一致 FAIL (prod " << a1->size()
                      << "B vs regen " << a2->size() << "B)\n";
            ++fail;
        }
    }
    std::cout << "[cmp] 合计 PASS=" << pass << " FAIL=" << fail
              << " SKIP=" << skip << "\n";
    return fail == 0 ? 0 : 2;
}

// ============================================================================
// refit：照片 → 4-1..4-8 → MultiLineInput → ExecuteMultiLine
// ============================================================================

struct RefitArgs {
    fs::path factoryData;     // factory_calib/data_in/laser
    fs::path dateIn;          // date_in/laser-mark1688-1006（initialT 来源）
    fs::path outJson;         // 产物 json
    int f8 = 1;               // 右目残差项开关
    double rightWeight = 1.0;
    int expectedLines = 25;
    int mergeTubes = 1;       // 1=两管合并进同一 pose 组
    int dumpDiag = 0;         // 1=打印逐环节诊断
    int maskThreshOverride = -999;   // ≥0 覆盖 config 的 mask threshold
    int scanDirOverride = -1;        // ≥0 覆盖 laser_label scanDirection
    int idOffsetR = 0;               // R 侧线号统一偏移（编号错位修正）
    double minDisp = -999, maxDisp = -999;   // 视差窗覆盖（含负视差探测）
    int p2Zero = 0;                  // 1=右目 P(0,3) 置零（扫描链约定）
    std::string onlyPose;     // 非空=只跑该姿态（调试）
    std::string initT;        // "x,y,z"（空=date_in projectorT）
    // —— 【uR 根因终裁探针 2026-11-09】——
    float interpStep = -1.0f;   // >0＝4-6 插值接入并设行距（生产=0.7；≤0＝旧行为不插值）
    int prodPreset = 0;         // 1＝生产口径预设（run_log.txt 2026-10-13 实测参数）
    std::string dumpObsDir;     // 非空＝逐姿态落盘配对观测 (lid,uL,vL,uR,vR)
    int evalProd = 0;           // 1＝累积 3D 点对生产曲线逐点评测 ΔuR（判别跨采集/链路）
    int maxRunWidthOverride = -1;  // ≥0＝覆盖 laser_label.maxRunWidth（探针放宽用）
};

struct FramePts {
    std::vector<cv::Point2f> points;
    std::vector<int> lineIds;
};

/// 【eval-prod】复刻 PJC evalRightResidual 数学（2601001 projector_joint_calib.cpp
/// :698-751 同式）：给定 3D 点 P（左矫正系）＋生产 (t,C_raw) 求 ΔuR＝预测−实测。
/// 返回 false＝门控（深度窗/判别式），true＝dUrOut 有效（未 clamp）。
static bool evalProdDur(const cv::Vec3d& P, const cv::Vec3d& t, const double Craw[6],
                        double f, const cv::Point2d& pp, double B, double& dUrOut) {
    const double X = P[0], Y = P[1], Z = P[2];
    if (Z < 1e-6) return false;
    const double uL = f * X / Z + pp.x;
    const double v  = f * Y / Z + pp.y;
    if (std::fabs(t[2]) < 1e-9) return false;
    const double Ex = f * t[0] / t[2] + pp.x;
    const double Ey = f * t[1] / t[2] + pp.y;
    const double du = Ex - uL, dv = Ey - v;
    const double A = Craw[0], Bq = Craw[1], Cq = Craw[2];
    const double D = Craw[3], Eq = Craw[4], F0 = Craw[5];
    const double a = A*du*du + Bq*du*dv + Cq*dv*dv;
    const double b = 2*A*uL*du + Bq*(uL*dv + v*du) + 2*Cq*v*dv + D*du + Eq*dv;
    const double c = A*uL*uL + Bq*uL*v + Cq*v*v + D*uL + Eq*v + F0;
    double cand[2]; int nCand = 0;
    if (std::fabs(a) < 1e-18) {
        if (std::fabs(b) > 1e-18) cand[nCand++] = -c / b;
    } else {
        const double disc = b*b - 4.0*a*c;
        if (disc > 0.0) {
            const double sq = std::sqrt(disc);
            const double q = -0.5 * (b + (b >= 0.0 ? sq : -sq));
            cand[nCand++] = q / a;
            if (std::fabs(q) > 1e-300) cand[nCand++] = c / q;
        }
    }
    const double uc = uL - pp.x;
    double best = 0.0; bool have = false;
    for (int i = 0; i < nCand; ++i) {
        const double qu = uL + cand[i] * du;
        const double qc = qu - pp.x;
        const double den = qc - uc;
        if (std::fabs(den) < 1e-9) continue;
        const double Zm = (qc * t[2] - f * t[0]) / den;
        if (Zm < 50.0 || Zm > 3000.0) continue;
        const double dur = f * B * (Zm - Z) / (Zm * Z);
        if (!have || std::fabs(dur) < std::fabs(best)) { best = dur; have = true; }
    }
    if (!have) return false;
    dUrOut = best;
    return true;
}

int runRefit(const RefitArgs& a) {
    std::string err;

    // —— 装载 config.json（mask/ccl/steger/laser_label/match/pjc 调参节）——
    const json cfg = readJson(a.factoryData / "config.json", err);
    if (!err.empty()) { std::cerr << "[refit] " << err << "\n"; return 1; }
    const FactoryCamera cam =
        loadFactoryCamera(a.factoryData / "camera_calib.json", err);
    if (!err.empty()) { std::cerr << "[refit] " << err << "\n"; return 1; }

    // —— 前段参数（factory config 逐键接线——仓内 CLI 忽略这些节，0 点事故根因）——
    MaskExtractParams mp;
    if (cfg.contains("mask")) {
        const json& m = cfg.at("mask");
        mp.threshold = m.value("threshold", 80);
        mp.erodeSize = m.value("erodeSize", 5);
        mp.laserDilateSize = m.value("laserDilateSize", 3);
        mp.postErodeSize = m.value("postErodeSize", 0);
    }
    if (a.maskThreshOverride > -900) mp.threshold = a.maskThreshOverride;
    RegionAnalyzerParams cp;
    if (cfg.contains("ccl")) {
        const json& c = cfg.at("ccl");
        cp.minArea = c.value("minArea", 100);
        cp.topXCount = c.value("topXCount", 0);
    }
    LaserLabelParams lp;
    if (cfg.contains("laser_label")) {
        const json& l = cfg.at("laser_label");
        lp.scanDirection = l.value("scanDirection", 0);
        lp.centerRowOffset = l.value("centerRowOffset", 0);
        lp.realLineTolerance = l.value("realLineTolerance", 0);
    }
    if (a.scanDirOverride >= 0) lp.scanDirection = a.scanDirOverride;
    if (a.maxRunWidthOverride >= 0) lp.maxRunWidth = a.maxRunWidthOverride;
    lp.expectedLineCount = a.expectedLines;   // F2 建议：缺线帧整帧弃用防错位（0=不设门）
    if (a.expectedLines <= 0) lp.expectedLineCount = 0;
    StegerParams sp;
    if (cfg.contains("steger")) {
        const json& s = cfg.at("steger");
        sp.sigma = s.value("sigma", 1.5f);
        sp.kernelSize = s.value("kernelSize", 0);
        sp.lowThreshold = s.value("lowThreshold", 2.0f);
        sp.highThreshold = s.value("highThreshold", 0.0f);
        sp.maxLabels = s.value("maxLabels", 256);
    }
    // —— 深窗与视差窗（max_disparity 默认 500 会静默丢 Z≲333mm 的点）——
    double depthMin = cfg.value(
        "depthMin", cfg.contains("plane_map") ? cfg.at("plane_map").value("depthMin", 100.0) : 100.0);
    double depthMax = cfg.value(
        "depthMax", cfg.contains("plane_map") ? cfg.at("plane_map").value("depthMax", 5000.0) : 5000.0);
    const double marginF = cfg.contains("match")
        ? cfg.at("match").value("disparityMarginFactor", 1.05) : 1.05;
    // —— 生产口径预设（2601001 run_log.txt 2026-10-13 18:28 实测参数）——
    if (a.prodPreset) {
        cp.minArea = 0;              // 生产 ccl minArea=0
        cp.topXCount = 27;           // 生产 topX=27
        lp.realLineTolerance = 10;   // 生产 label tol=10
        if (a.maxRunWidthOverride < 0) lp.maxRunWidth = 30;  // 生产门2（CLI 显式值优先）
        sp.maxLineWidth = 6.0f;      // 生产逐点宽度门 6.0
        depthMin = 200.0; depthMax = 500.0;   // 生产深窗 [200,500]
    }
    const double f = cam.P1.at<double>(0, 0);
    const double B = std::fabs(cam.P2.at<double>(0, 3)) / f;
    LaserMatchParams mtp;
    mtp.epipolar_row_step = cfg.contains("plane_map")
        ? cfg.at("plane_map").value("epipolarStep", 0.5f) : 0.5f;
    mtp.min_disparity = static_cast<float>(std::floor(f * B / depthMax * 0.95));
    mtp.max_disparity = static_cast<float>(std::ceil(f * B / depthMin * marginF));
    if (a.minDisp > -900) mtp.min_disparity = static_cast<float>(a.minDisp);
    if (a.maxDisp > -900) mtp.max_disparity = static_cast<float>(a.maxDisp);
    std::cout << "[refit] f=" << f << " B=" << B << " 深窗[" << depthMin << ","
              << depthMax << "] 视差窗[" << mtp.min_disparity << ","
              << mtp.max_disparity << "]\n";

    // —— 算子构造（左右目独立前段；后段一次）——
    MaskExtractCUDA maskL(mp), maskR(mp);
    RegionAnalyzerCUDA cclL(cp), cclR(cp);
    LaserLabelerCUDA labL(lp), labR(lp);
    StegerExtractorCUDA stgL(sp), stgR(sp);
    UndistortPointsParams upL;
    upL.cameraMatrix = cam.K1;  upL.distCoeffs = cam.D1;
    upL.R = cam.R1;             upL.P = cam.P1;
    UndistortPointsParams upR;
    upR.cameraMatrix = cam.K2;  upR.distCoeffs = cam.D2;
    upR.R = cam.R2;             upR.P = cam.P2;
    if (a.p2Zero || a.prodPreset) {   // prodPreset：生产 4-5 无条件置零 P2(0,3)
        // 【扫描链约定探针】README §9-3：右目 P(0,3) 置零→右目自身列坐标。实测
        // 用原 P2 时右目矫正 x 呈现 −f·B 双移（disp≈164893≈fB）——置零判定。
        upR.P = cam.P2.clone();
        upR.P.at<double>(0, 3) = 0.0;
    }
    UndistortPointsCuda undL(upL), undR(upR);
    EpipolarInterpParams eip;
    eip.lineIdCheck = true;
    if (a.interpStep > 0) eip.epipolar_row_step = a.interpStep;   // 生产=0.7（4-6 Labeled）
    EpipolarInterpCuda interpOp(eip);
    LaserMatchCuda matchOp(mtp);
    LaserReconstructCuda reconOp;

    cv::cuda::Stream stream;

    // 单幅图前段（4-1..4-4）
    auto front = [&](const cv::Mat& gray, MaskExtractCUDA& mo,
                     RegionAnalyzerCUDA& co, LaserLabelerCUDA& lo,
                     StegerExtractorCUDA& so, FramePts& out) -> bool {
        auto m = mo.Execute(gray, stream);
        if (!m.success) { std::cerr << "  4-1: " << m.message << "\n"; return false; }
        auto c = co.Execute(*m.d_cleanedMask, stream);
        if (!c.success) { std::cerr << "  4-2: " << c.message << "\n"; return false; }
        if (a.dumpDiag) {
            cv::Mat hm;
            m.d_cleanedMask->download(hm, stream);
            stream.waitForCompletion();
            std::cerr << "  [diag] mask nonzero="
                      << (hm.empty() ? -1 : cv::countNonZero(hm))
                      << " ccl comps=" << c.componentCount << "\n";
        }
        auto l = lo.Execute(*c.d_labeledMask, stream);
        if (!l.success) { std::cerr << "  4-3: " << l.message << "\n"; return false; }
        if (a.dumpDiag)
            std::cerr << "  [diag] label lines=" << l.lineCount
                      << " rejected=" << l.frameRejected << "\n";
        if (l.frameRejected) return false;
        auto s = so.Execute(*m.d_grayImage, *l.d_labeledMask, stream,
                            GroupMode::ByLabel);
        if (!s.success) { std::cerr << "  4-4: " << s.message << "\n"; return false; }
        cv::Mat hPts, hIds;
        if (s.d_centerPoints && s.d_line_ids) {
            s.d_centerPoints->download(hPts, stream);
            s.d_line_ids->download(hIds, stream);
            stream.waitForCompletion();
        }
        if (!hPts.empty())
            out.points.assign(hPts.begin<cv::Point2f>(), hPts.end<cv::Point2f>());
        if (!hIds.empty())
            out.lineIds.assign(hIds.begin<int>(), hIds.end<int>());
        if (a.dumpDiag && !out.lineIds.empty()) {
            int mn = out.lineIds[0], mx = out.lineIds[0];
            for (int v : out.lineIds) { mn = std::min(mn, v); mx = std::max(mx, v); }
            std::set<int> uniq(out.lineIds.begin(), out.lineIds.end());
            std::cerr << "  [diag] steger pts=" << out.points.size()
                      << " ids " << mn << ".." << mx
                      << " unique=" << uniq.size() << "\n";
            // 线 1 与中线的外接框（判朝向：x 宽=横线 / y 高=竖线）
            for (int probe : {1, static_cast<int>(*std::next(uniq.begin(), uniq.size() / 2))}) {
                float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
                int n = 0;
                for (size_t q = 0; q < out.points.size(); ++q) {
                    if (out.lineIds[q] != probe) continue;
                    const auto& p = out.points[q];
                    x0 = std::min(x0, p.x); x1 = std::max(x1, p.x);
                    y0 = std::min(y0, p.y); y1 = std::max(y1, p.y);
                    ++n;
                }
                if (n > 0)
                    std::cerr << "  [diag] line " << probe << ": n=" << n
                              << " x[" << (int)x0 << "," << (int)x1 << "]"
                              << " y[" << (int)y0 << "," << (int)y1 << "]"
                              << ((x1 - x0) > (y1 - y0) ? " 横线" : " 竖线") << "\n";
            }
        }
        return !out.points.empty();
    };

    // —— 后段替代（4-6'/4-7'）：CPU 直接匹配——
    // 本批照片激光线近乎竖直，标定链 4-6（行插值）与 4-7（行键匹配）假设横线
    // 结构（行行有对应点），竖线场景左右插值点行号不相交→match 恒 0。此处按
    // 几何本义直接配对：同线号＋矫正后同高度（|Δy|≤1.5px，就近年配对）＋
    // 视差在窗内 → 配对喂 4-8（Q 三角化）。横线场景此逻辑同样成立（更宽松）。
    // 【2026-11-09 终裁探针】a.interpStep>0 时在配对前真实接入 4-6 插值
    // （lineIdCheck=true＝生产 Labeled 口径），并可逐姿态落盘配对观测。
    auto back = [&](const FramePts& fl, const FramePts& fr,
                    std::vector<cv::Vec3f>& pts3d,
                    std::vector<int>& lids, const std::string& tag = "") -> bool {
        cv::Mat hL(1, static_cast<int>(fl.points.size()), CV_32FC2,
                   const_cast<cv::Point2f*>(fl.points.data()));
        cv::Mat hLIds(1, static_cast<int>(fl.lineIds.size()), CV_32SC1,
                      const_cast<int*>(fl.lineIds.data()));
        cv::Mat hR(1, static_cast<int>(fr.points.size()), CV_32FC2,
                   const_cast<cv::Point2f*>(fr.points.data()));
        cv::Mat hRIds(1, static_cast<int>(fr.lineIds.size()), CV_32SC1,
                      const_cast<int*>(fr.lineIds.data()));
        cv::cuda::GpuMat dL, dLIds;
        dL.upload(hL, stream);  dLIds.upload(hLIds, stream);
        auto uL = undL.Execute(dL, dLIds, stream);       // 4-5 L（矫正）
        if (!uL.success || !uL.d_rectifiedPoints || !uL.d_line_ids) {
            std::cerr << "  4-5L: " << uL.message << "\n";
            return false;
        }
        cv::Mat hR2(1, static_cast<int>(fr.points.size()), CV_32FC2,
                    const_cast<cv::Point2f*>(fr.points.data()));
        cv::Mat hRIds2(1, static_cast<int>(fr.lineIds.size()), CV_32SC1,
                       const_cast<int*>(fr.lineIds.data()));
        cv::cuda::GpuMat dR2, dRIds2;
        dR2.upload(hR2, stream);  dRIds2.upload(hRIds2, stream);
        auto uR = undR.Execute(dR2, dRIds2, stream);     // 4-5 R（矫正）
        if (!uR.success || !uR.d_rectifiedPoints || !uR.d_line_ids) {
            std::cerr << "  4-5R: " << uR.message << "\n";
            return false;
        }
        // —— 【终裁探针】4-6 插值（可选，生产 Labeled 口径）——
        std::shared_ptr<cv::cuda::GpuMat> dPtsL, dPtsR, dIdsL, dIdsR;
        if (a.interpStep > 0) {
            auto iL = interpOp.Execute(*uL.d_rectifiedPoints, *uL.d_line_ids, stream);
            if (!iL.success || !iL.d_interpPoints || !iL.d_interp_line_ids) {
                std::cerr << "  4-6L: " << iL.message << "\n"; return false;
            }
            auto iR = interpOp.Execute(*uR.d_rectifiedPoints, *uR.d_line_ids, stream);
            if (!iR.success || !iR.d_interpPoints || !iR.d_interp_line_ids) {
                std::cerr << "  4-6R: " << iR.message << "\n"; return false;
            }
            dPtsL = iL.d_interpPoints; dIdsL = iL.d_interp_line_ids;
            dPtsR = iR.d_interpPoints; dIdsR = iR.d_interp_line_ids;
        } else {
            dPtsL = uL.d_rectifiedPoints; dIdsL = uL.d_line_ids;
            dPtsR = uR.d_rectifiedPoints; dIdsR = uR.d_line_ids;
        }
        // CPU 匹配（下载矫正点后）
        cv::Mat hl, hidl, hr, hidr;
        dPtsL->download(hl, stream);
        dIdsL->download(hidl, stream);
        dPtsR->download(hr, stream);
        dIdsR->download(hidr, stream);
        stream.waitForCompletion();
        const int nL = std::min(hl.cols, hidl.cols);
        const int nR = std::min(hr.cols, hidr.cols);
        // R 侧按 (lid, 行桶) 建索引：桶高 3px
        constexpr float kBucket = 3.0f;
        std::multimap<long long, int> rIdx;
        for (int j = 0; j < nR; ++j) {
            const float y = hr.at<cv::Point2f>(j).y;
            const int id = hidr.at<int>(j);
            rIdx.emplace((static_cast<long long>(id) << 24)
                         | static_cast<long long>(y / kBucket), j);
        }
        std::vector<cv::Point2f> mL, mR;
        std::vector<int> mIds;
        for (int i = 0; i < nL; ++i) {
            const cv::Point2f pl = hl.at<cv::Point2f>(i);
            const int id = hidl.at<int>(i);
            const long long b0 = (static_cast<long long>(id) << 24)
                                 | static_cast<long long>(pl.y / kBucket);
            int bestJ = -1;
            float bestDy = 1.6f;   // 行对齐容差（矫正后应 <1px，留余量）
            for (int db = -1; db <= 1; ++db) {
                const auto er = rIdx.equal_range(b0 + db);
                for (auto it = er.first; it != er.second; ++it) {
                    const int j = it->second;
                    const cv::Point2f pr = hr.at<cv::Point2f>(j);
                    const float dy = std::fabs(pr.y - pl.y);
                    if (dy < bestDy) {
                        const float disp = pl.x - pr.x;
                        if (disp >= mtp.min_disparity && disp <= mtp.max_disparity) {
                            bestDy = dy;
                            bestJ = j;
                        }
                    }
                }
            }
            if (bestJ >= 0) {
                mL.push_back(pl);
                mR.push_back(hr.at<cv::Point2f>(bestJ));
                mIds.push_back(id);
            }
        }
        // —— 【终裁探针】配对观测落盘（lid,uL,vL,uR,vR；重建/抽样之前）——
        if (!a.dumpObsDir.empty() && !tag.empty()) {
            std::ofstream f(a.dumpObsDir + "/" + tag + "_obs.csv");
            if (f) {
                f << "lid,uL,vL,uR,vR\n" << std::fixed << std::setprecision(4);
                for (size_t q = 0; q < mL.size(); ++q)
                    f << mIds[q] << ',' << mL[q].x << ',' << mL[q].y
                      << ',' << mR[q].x << ',' << mR[q].y << '\n';
            }
        }
        if (a.dumpDiag && nL > 0 && nR > 0) {
            // 纯几何探查：对 L 前 5 个点找同线号最近 y 的 R 点，报 Δy/视差
            for (int probe = 0; probe < 5 && probe < nL; ++probe) {
                const cv::Point2f pl = hl.at<cv::Point2f>(probe);
                const int id = hidl.at<int>(probe);
                float bdy = 1e9f, bdisp = 0;
                for (int j = 0; j < nR; ++j) {
                    if (hidr.at<int>(j) != id) continue;
                    const float dy = std::fabs(hr.at<cv::Point2f>(j).y - pl.y);
                    if (dy < bdy) {
                        bdy = dy;
                        bdisp = pl.x - hr.at<cv::Point2f>(j).x;
                    }
                }
                std::cerr << "  [diag] L" << probe << "(x=" << pl.x << ",y="
                          << pl.y << ",id=" << id << ") 最近同号R: dy=" << bdy
                          << " disp=" << bdisp << "\n";
            }
        }
        if (a.dumpDiag)
            std::cerr << "  [diag] CPU匹配 nL=" << nL << " nR=" << nR
                      << " 对=" << mL.size() << "\n";
        if (mL.empty()) return false;
        cv::Mat hmL(1, static_cast<int>(mL.size()), CV_32FC2, mL.data());
        cv::Mat hmR(1, static_cast<int>(mR.size()), CV_32FC2, mR.data());
        cv::Mat hmI(1, static_cast<int>(mIds.size()), CV_32SC1, mIds.data());
        cv::cuda::GpuMat dML, dMR, dMI;
        dML.upload(hmL, stream);  dMR.upload(hmR, stream);  dMI.upload(hmI, stream);
        auto r = reconOp.Execute(dML, dMR, dMI, cam.Q, stream);   // 4-8
        if (!r.success) { std::cerr << "  4-8: " << r.message << "\n"; return false; }
        if (r.d_points3d && r.d_valid_line_ids && r.validCount > 0) {
            cv::Mat hP, hI;
            r.d_points3d->download(hP, stream);
            r.d_valid_line_ids->download(hI, stream);
            stream.waitForCompletion();
            if (!hP.empty())
                pts3d.assign(hP.begin<cv::Vec3f>(), hP.end<cv::Vec3f>());
            if (!hI.empty())
                lids.assign(hI.begin<int>(), hI.end<int>());
        }
        return !pts3d.empty();
    };

    // —— 姿态循环（每姿态两管；a.mergeTubes=1 时两管点合并进同一 pose 组）——
    // 【编号错位修正】实测本批照片 L 恒少 1 条线（L=24/21 vs R=25/22）——若 L 丢失
    // 的是首线，则 L#k == R#(k+1)，同号配对整体错一档 → match=0。--id-offset-r
    // 对 R 侧线号统一加偏移后再进 4-7（经验证定值）。
    // 【2026-11-09】无 pose_* 目录时回退 left/+right/ 同名顺序帧布局
    // （test_data/laser261006-1668 生产数据集形态；每对＝1 姿态单管）。
    struct PoseJob { fs::path l, r; std::string tag; };
    std::vector<PoseJob> jobs;
    bool hasPoseDirs = false;
    for (const auto& e : fs::directory_iterator(a.factoryData))
        if (e.is_directory() && e.path().filename().string().rfind("pose_", 0) == 0)
            { hasPoseDirs = true; break; }
    if (hasPoseDirs) {
        for (const auto& poseDir : fs::directory_iterator(a.factoryData)) {
            if (!poseDir.is_directory()) continue;
            const std::string name = poseDir.path().filename().string();
            if (name.rfind("pose_", 0) != 0) continue;
            if (!a.onlyPose.empty() && name != a.onlyPose) continue;
            for (int tube = 0; tube < 2; ++tube)
                jobs.push_back({poseDir.path() / ("L_tube" + std::to_string(tube) + ".png"),
                                poseDir.path() / ("R_tube" + std::to_string(tube) + ".png"),
                                name + "_tube" + std::to_string(tube)});
        }
    } else {
        fs::path ldir = a.factoryData / "left", rdir = a.factoryData / "right";
        std::vector<fs::path> lfiles;
        if (fs::exists(ldir))
            for (const auto& e : fs::directory_iterator(ldir))
                if (e.is_regular_file() && e.path().extension() == ".png")
                    lfiles.push_back(e.path());
        std::sort(lfiles.begin(), lfiles.end(),
                  [](const fs::path& x, const fs::path& y) { return x.filename() < y.filename(); });
        for (const auto& lp : lfiles) {
            const fs::path rp = rdir / lp.filename();
            if (!fs::exists(rp)) continue;
            const std::string stem = lp.stem().string();
            if (!a.onlyPose.empty() && stem != a.onlyPose) continue;
            jobs.push_back({lp, rp, stem});
        }
        std::cout << "[refit] 布局回退 left/right 顺序帧：jobs=" << jobs.size() << "\n";
    }
    std::vector<std::vector<PosePointSet>> perPoseTubes;   // [pose][tube]
    std::vector<std::string> poseNames;
    for (const auto& job : jobs) {
        if (!fs::exists(job.l) || !fs::exists(job.r)) continue;
        const std::string name = hasPoseDirs
            ? job.tag.substr(0, job.tag.rfind("_tube")) : job.tag;
        const int tube = hasPoseDirs
            ? std::atoi(job.tag.substr(job.tag.rfind("_tube") + 5).c_str()) : 0;
        const cv::Mat gl = cv::imread(job.l.string(), cv::IMREAD_GRAYSCALE);
        const cv::Mat gr = cv::imread(job.r.string(), cv::IMREAD_GRAYSCALE);
        if (gl.empty() || gr.empty()) continue;
        FramePts fl, fr;
        if (!front(gl, maskL, cclL, labL, stgL, fl)) { std::cout << "[pose] " << job.tag << " L 前段空/拒\n"; continue; }
        if (!front(gr, maskR, cclR, labR, stgR, fr)) { std::cout << "[pose] " << job.tag << " R 前段空/拒\n"; continue; }
        if (a.idOffsetR != 0)
            for (int& id : fr.lineIds) id += a.idOffsetR;
        std::vector<cv::Vec3f> pts3d;
        std::vector<int> lids;
        if (!back(fl, fr, pts3d, lids, job.tag)) { std::cout << "[pose] " << job.tag << " 后段空\n"; continue; }
        // 聚合：同名姿态一组；left/right 模式每 job 即一组
        size_t pi = SIZE_MAX;
        for (size_t q = 0; q < poseNames.size(); ++q)
            if (poseNames[q] == name) { pi = q; break; }
        if (pi == SIZE_MAX) {
            perPoseTubes.push_back(std::vector<PosePointSet>(2));
            poseNames.push_back(name);
            pi = poseNames.size() - 1;
        }
        perPoseTubes[pi][static_cast<size_t>(tube)].points3d = pts3d;
        perPoseTubes[pi][static_cast<size_t>(tube)].lineIds = lids;
        std::cout << "[pose] " << job.tag << " 3D点=" << pts3d.size() << "\n";
    }
    const size_t nPoses = perPoseTubes.size();
    std::cout << "[refit] 有效姿态 " << nPoses << " / " << jobs.size() << "\n";
    if (nPoses < 5) {
        std::cerr << "[refit] 有效姿态不足 5（minPoses），中止——参数接线可疑\n";
        return 1;
    }

    // —— 【eval-prod】累积 3D 点对生产曲线逐点评测（跨采集 vs 链路口径判别器）——
    // ΔuR＝生产曲线预测−本链实测（PJC 口径，与 E2 res＝实测−预测 反号）。
    // 平坦→本链观测与生产曲线一致（偏差在另一链）；斜坡→本链自带偏差。
    if (a.evalProd) {
        std::string e2;
        const LaserFit prod = loadLaserFit(a.dateIn / "laser_calib.json", e2);
        if (!e2.empty()) { std::cerr << "[eval-prod] " << e2 << "\n"; return 1; }
        if (prod.curves.empty() || prod.f <= 0 || prod.baseline <= 0) {
            std::cerr << "[eval-prod] 生产 json 缺曲线/f/baseline\n"; return 1;
        }
        const size_t L = prod.curves.size();
        // —— 线号偏移自动扫描：factory 批 L 侧少 1 条线（编号错一档），正确 δ 下
        //    总 rms 塌到亚像素级；错误 δ 下逐线错配 → rms 数十像素量级。
        for (int delta : {-1, 0, 1}) {
            std::vector<double> sum(L, 0.0), sumSq(L, 0.0);
            std::vector<long long> cnt(L, 0);
            long long gated = 0, total = 0;
            for (size_t p = 0; p < nPoses; ++p)
                for (int tube = 0; tube < 2; ++tube) {
                    const auto& pp = perPoseTubes[p][static_cast<size_t>(tube)];
                    for (size_t k = 0; k < pp.points3d.size(); ++k) {
                        const int lidEff = pp.lineIds[k] + delta;
                        if (lidEff < 1 || static_cast<size_t>(lidEff) > L) continue;
                        ++total;
                        const cv::Vec3d P(pp.points3d[k][0], pp.points3d[k][1],
                                          pp.points3d[k][2]);
                        double dur;
                        if (!evalProdDur(P, prod.projectorT,
                                         prod.curves[static_cast<size_t>(lidEff - 1)].coeffs,
                                         prod.f, prod.principalPoint, prod.baseline, dur)) {
                            ++gated; continue;
                        }
                        const size_t l = static_cast<size_t>(lidEff - 1);
                        sum[l] += dur; sumSq[l] += dur * dur; ++cnt[l];
                    }
                }
            double totSq = 0.0; long long totN = 0;
            for (size_t l = 0; l < L; ++l) { totSq += sumSq[l]; totN += cnt[l]; }
            const double totRms = totN > 0 ? std::sqrt(totSq / static_cast<double>(totN)) : -1;
            std::cout << "\n[eval-prod] delta=" << delta << " (local lid -> prod lid)"
                      << "  totRms=" << totRms << "  gated=" << gated << "/" << total << "\n";
            if (delta != 0 && totRms > 30.0) {   // 错位档：只报总览不逐线刷屏
                std::cout << "  (line-index mismatch artifact, skip per-line)\n";
                continue;
            }
            std::cout << " lid |      mean |     rms   | 点数\n"
                         " --- | --------- | --------- | -----\n";
            std::vector<double> means;
            for (size_t l = 0; l < L; ++l) {
                const double m = cnt[l] > 0 ? sum[l] / static_cast<double>(cnt[l]) : kNaN;
                const double r = cnt[l] > 0
                    ? std::sqrt(std::max(0.0, sumSq[l] / static_cast<double>(cnt[l]) - m * m))
                    : kNaN;
                means.push_back(cnt[l] > 0 ? m : kNaN);
                char buf[96];
                std::snprintf(buf, sizeof(buf), " %3zu | %9.4f | %9.4f | %lld",
                              l + 1, m, r, cnt[l]);
                std::cout << buf << "\n";
            }
            std::cout << "[eval-prod] delta=" << delta
                      << "  slope(px/line)=" << slopeOf(means)
                      << " (dur=pred-obs; cf E2 measured 0.139, opposite sign)\n";
        }
        std::cout << "\n";
    }

    // —— 按线分桶：lines[l][poseIdx]（mergeTubes=1 两管合并；每线只装自己的点，
    //    按线按姿态均匀抽样到 cap——生产拟合 227k 点口径，全量 12M 会 OOM）——
    int maxLid = a.expectedLines;
    for (size_t p = 0; p < nPoses; ++p)
        for (int tube = 0; tube < 2; ++tube)
            for (int lid : perPoseTubes[p][static_cast<size_t>(tube)].lineIds)
                maxLid = std::max(maxLid, lid);
    MultiLineInput in;
    in.lines.resize(static_cast<size_t>(std::max(1, maxLid)));
    const size_t kCap = 2000;   // 每线每姿态抽样上限（25×20×2000=1M 点）
    for (size_t p = 0; p < nPoses; ++p) {
        // 合并两管后按线分组下标
        std::vector<std::vector<size_t>> byLine(
            static_cast<size_t>(std::max(1, maxLid)));
        std::vector<cv::Vec3f> allPts;
        std::vector<int> allIds;
        for (int tube = 0; tube < 2; ++tube) {
            const auto& pp = perPoseTubes[p][static_cast<size_t>(tube)];
            allPts.insert(allPts.end(), pp.points3d.begin(), pp.points3d.end());
            allIds.insert(allIds.end(), pp.lineIds.begin(), pp.lineIds.end());
        }
        for (size_t i = 0; i < allIds.size(); ++i) {
            const int lid = allIds[i];
            if (lid >= 1 && lid <= maxLid)
                byLine[static_cast<size_t>(lid - 1)].push_back(i);
        }
        for (size_t l = 0; l < byLine.size(); ++l) {
            if (byLine[l].empty()) continue;
            PosePointSet pps;
            const size_t n = byLine[l].size();
            const size_t stride = std::max<size_t>(1, n / kCap);
            for (size_t k = 0; k < n; k += stride) {
                pps.points3d.push_back(allPts[byLine[l][k]]);
                pps.lineIds.push_back(static_cast<int>(l) + 1);
            }
            in.lines[l].push_back(std::move(pps));
        }
    }

    in.f = f;
    in.principalPoint = cv::Point2d(cam.P1.at<double>(0, 2),
                                    cam.P1.at<double>(1, 2));
    in.baseline = B;
    in.enableRightTerm = a.f8 != 0;
    in.rightTermWeight = a.rightWeight;
    if (!a.initT.empty()) {
        std::sscanf(a.initT.c_str(), "%lf,%lf,%lf", &in.initialT[0],
                    &in.initialT[1], &in.initialT[2]);
    } else {
        const LaserFit prodFit = loadLaserFit(a.dateIn / "laser_calib.json", err);
        if (!err.empty()) { std::cerr << "[refit] " << err << "\n"; return 1; }
        in.initialT = prodFit.projectorT;   // 现产 projectorT 热启动（吸引盆小）
    }
    size_t totalPts = 0;
    for (const auto& l : in.lines) {
        for (const auto& p : l) totalPts += p.points3d.size();
    }
    std::cout << "[refit] 拟合输入：totalPts=" << totalPts
              << " f=" << in.f << " baseline=" << in.baseline
              << " initialT=" << in.initialT[0] << "," << in.initialT[1]
              << "," << in.initialT[2] << " F8=" << a.f8 << "\n";

    // —— 拟合参数（config pjc 节直灌 fromJson）——
    ProjectorJointCalibParams pp;
    if (cfg.contains("pjc")) {
        try {
            pp = ProjectorJointCalibParams::fromJson(cfg.at("pjc"));
        } catch (...) { /* 保留默认 */ }
    }
    ProjectorJointCalib op(pp);
    const MultiLineResult r = op.ExecuteMultiLine(in);
    if (!r.success) {
        std::cerr << "[refit] ExecuteMultiLine 失败: " << r.message << "\n";
        return 1;
    }

    // —— 诊断打印（pre 形状应复现 E2 实测斜坡＝治本证据链的关键互证）——
    std::cout << "\n[fit] success=" << r.success
              << " Sampson rms " << r.initialSampsonRms << " -> "
              << r.finalSampsonRms
              << " cond=" << r.jacobianConditionNumber
              << " rightTermUsed=" << r.rightTermUsed
              << " phaseBIters=" << r.phaseBIters << "\n";
    if (r.rightTermUsed) {
        std::cout << "[fit] urBiasRms pre=" << r.urBiasPreRms
                  << " post=" << r.urBiasPostRms
                  << " gate=" << r.urGateCount << "/" << r.urRowCount
                  << " crossAngleMed=" << r.urCrossAngleMedianDeg << "deg\n";
    }
    std::cout << "\n lid |      pre |     post | perLineRms | 点数\n";
    std::cout <<   " --- | -------- | -------- | ---------- | -----\n";
    std::vector<double> pre, post;
    for (size_t l = 0; l < r.emissionCurves.size(); ++l) {
        const double pv = l < r.urBiasPrePerLine.size()
            ? r.urBiasPrePerLine[l] : kNaN;
        const double qv = l < r.urBiasPostPerLine.size()
            ? r.urBiasPostPerLine[l] : kNaN;
        pre.push_back(pv);
        post.push_back(qv);
        size_t npts = 0;
        if (l < r.perLine.size()) npts = r.perLine[l].pointCount;
        double lrms = 0;
        if (l < r.perLine.size()) lrms = r.perLine[l].rms;
        char buf[160];
        std::snprintf(buf, sizeof(buf), " %3zu | %8.4f | %8.4f | %10.5f | %zu",
                      l + 1, pv, qv, lrms, npts);
        std::cout << buf << "\n";
    }
    std::cout << "\n[fit] 斜率(px/线)：pre=" << slopeOf(pre)
              << "  post=" << slopeOf(post) << "\n";
    std::cout << "[fit] ⚠ 符号约定：PJC 的 ΔuR＝预测−实测；E2 实测 res＝实测−预测\n"
                 "      ⇒ pre 斜率应 ≈ −(E2 斜率 0.139) ≈ −0.14（形状互证）\n";

    // —— 产物 JSON（含现产缺的逐线诊断向量）——
    json out;
    out["schema"] = "factory_calib.laser_calib.v1";
    out["build"] = "pjc_refit_2026-11-08";
    out["status"] = "ok";
    json pj;
    pj["f"] = in.f;
    pj["principalPoint"] = {in.principalPoint.x, in.principalPoint.y};
    pj["baseline"] = in.baseline;
    pj["epipolarRowStep"] = rowStepForJson();
    pj["projectorT"] = {r.projectorT[0], r.projectorT[1], r.projectorT[2]};
    pj["initialSampsonRms"] = r.initialSampsonRms;
    pj["finalSampsonRms"] = r.finalSampsonRms;
    pj["jacobianConditionNumber"] = r.jacobianConditionNumber;
    pj["rightTermUsed"] = r.rightTermUsed;
    pj["phaseBIters"] = r.phaseBIters;
    pj["urBiasPreRms"] = r.urBiasPreRms;
    pj["urBiasPostRms"] = r.urBiasPostRms;
    pj["urBiasPrePerLine"] = r.urBiasPrePerLine;
    pj["urBiasPostPerLine"] = r.urBiasPostPerLine;
    pj["urGateCount"] = r.urGateCount;
    pj["urRowCount"] = r.urRowCount;
    pj["urCrossAngleMedianDeg"] = r.urCrossAngleMedianDeg;
    pj["totalPointCount"] = r.totalPointCount;
    pj["lineCount"] = r.emissionCurves.size();
    pj["poseCount"] = nPoses;
    json curves = json::array();
    for (size_t l = 0; l < r.emissionCurves.size(); ++l) {
        json c;
        c["lineId"] = l + 1;
        c["coeffs"] = {
            r.emissionCurves[l].coeffs[0], r.emissionCurves[l].coeffs[1],
            r.emissionCurves[l].coeffs[2], r.emissionCurves[l].coeffs[3],
            r.emissionCurves[l].coeffs[4], r.emissionCurves[l].coeffs[5]};
        c["pointCount"] = (l < r.perLine.size()) ? r.perLine[l].pointCount : 0;
        curves.push_back(std::move(c));
    }
    pj["emissionCurves"] = std::move(curves);
    out["pjc"] = std::move(pj);
    out["posesProcessed"] = nPoses;
    out["poseNames"] = poseNames;
    std::ofstream of(a.outJson);
    of << out.dump(2) << "\n";
    std::cout << "\n[refit] 产物 " << a.outJson << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = "refit";
    RefitArgs ra;
    RegenArgs ga;
    std::string outDir = "out/pjc_refit";
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (k == "--mode") mode = next();
        else if (k == "--factory-data") ra.factoryData = next();
        else if (k == "--date-in") { ra.dateIn = next(); ga.dateIn = ra.dateIn; }
        else if (k == "--f8") ra.f8 = std::atoi(next().c_str());
        else if (k == "--diag") ra.dumpDiag = 1;
        else if (k == "--mask-threshold") ra.maskThreshOverride = std::atoi(next().c_str());
        else if (k == "--scan-dir") ra.scanDirOverride = std::atoi(next().c_str());
        else if (k == "--id-offset-r") ra.idOffsetR = std::atoi(next().c_str());
        else if (k == "--min-disp") ra.minDisp = std::atof(next().c_str());
        else if (k == "--max-disp") ra.maxDisp = std::atof(next().c_str());
        else if (k == "--p2-zero") ra.p2Zero = 1;
        else if (k == "--interp-step") ra.interpStep = static_cast<float>(std::atof(next().c_str()));
        else if (k == "--dump-obs") ra.dumpObsDir = next();
        else if (k == "--prod-preset") ra.prodPreset = 1;
        else if (k == "--eval-prod") ra.evalProd = 1;
        else if (k == "--max-run-width") ra.maxRunWidthOverride = std::atoi(next().c_str());
        else if (k == "--pose") ra.onlyPose = next();
        else if (k == "--expected-lines") ra.expectedLines = std::atoi(next().c_str());
        else if (k == "--probe-img") { ra.onlyPose = next(); mode = "probe"; }
        else if (k == "--right-weight") ra.rightWeight = std::atof(next().c_str());
        else if (k == "--init-t") ra.initT = next();
        else if (k == "--out-dir") outDir = next();
        else if (k == "--laser-json") ga.laserJson = next();
        else if (k == "--half-range") ga.halfRange = static_cast<float>(std::atof(next().c_str()));
        else if (k == "--no-compare") ga.compare = false;
        else { std::cerr << "未知参数 " << k << "\n"; return 1; }
    }
    fs::create_directories(outDir);
    if (mode == "selftest-match") {
        // 匹配算子合成自测：10 点同键同视差——matchCount 应=10
        LaserMatchParams p;
        p.epipolar_row_step = 0.5f;
        p.min_disparity = 0.0f;
        p.max_disparity = 1000.0f;
        LaserMatchCuda op(p);
        std::vector<cv::Point2f> L, R;
        std::vector<int> ids;
        for (int i = 0; i < 10; ++i) {
            const float y = 100.0f + i * 5.0f;      // 行对齐
            L.push_back({600.0f + i, y});
            R.push_back({100.0f + i, y});           // 视差 500
            ids.push_back(3);
        }
        cv::Mat hL(1, 10, CV_32FC2, L.data());
        cv::Mat hR(1, 10, CV_32FC2, R.data());
        cv::Mat hI(1, 10, CV_32SC1, ids.data());
        cv::cuda::GpuMat dL, dR, dI;
        cv::cuda::Stream s;
        dL.upload(hL, s); dR.upload(hR, s); dI.upload(hI, s);
        auto r = op.Execute(dL, dI, dR, dI, s);
        std::cout << "[selftest-match] success=" << r.success
                  << " matchCount=" << r.matchCount
                  << " (期望 10) msg=" << r.message << "\n";
        return r.matchCount == 10 ? 0 : 1;
    }
    if (mode == "probe") {
        // 探针：OpenCV 读图统计（调色板 PNG 转换是否正常的独立验证）
        cv::Mat g = cv::imread(ra.onlyPose, cv::IMREAD_GRAYSCALE);
        if (g.empty()) { std::cerr << "[probe] 读图失败 " << ra.onlyPose << "\n"; return 1; }
        long long hist[256] = {0};
        for (int y = 0; y < g.rows; ++y)
            for (int x = 0; x < g.cols; ++x) ++hist[g.at<uchar>(y, x)];
        double mn = 255, mx = 0;
        for (int i = 0; i < 256; ++i) {
            if (hist[i]) { if (i < mn) mn = i; if (i > mx) mx = i; }
        }
        std::cout << "[probe] " << ra.onlyPose << " " << g.cols << "x" << g.rows
                  << " min=" << mn << " max=" << mx << "\n";
        // ASCII 缩略图（40 行×100 列，亮度→字符）——快速人眼判断场景结构
        const char* ramp = " .:-=+*#%@";
        for (int ry = 0; ry < 40; ++ry) {
            std::string line;
            for (int rx = 0; rx < 100; ++rx) {
                const int y = ry * g.rows / 40, x = rx * g.cols / 100;
                const int v = g.at<uchar>(y, x);
                line += ramp[std::min(9, v / 26)];
            }
            std::cout << line << "\n";
        }
        for (int b = 0; b < 16; ++b) {
            long long cnt = 0;
            for (int i = b * 16; i < b * 16 + 16 && i < 256; ++i)
                cnt += hist[i];
            std::cout << "  [" << b * 16 << "," << b * 16 + 15 << "] "
                      << cnt << "\n";
        }
        return 0;
    }
    try {
        if (mode == "identity") {
            if (ga.laserJson.empty()) ga.laserJson = ga.dateIn / "laser_calib.json";
            ga.outBin = fs::path(outDir) / "regen_identity.bin";
            return runGenerateAndCompare(ga);
        }
        if (mode == "refit") {
            ra.outJson = fs::path(outDir) / "laser_calib_refit.json";
            return runRefit(ra);
        }
        if (mode == "regen") {
            if (ga.laserJson.empty())
                ga.laserJson = fs::path(outDir) / "laser_calib_refit.json";
            ga.outBin = fs::path(outDir) / "regen_refit.bin";
            return runGenerateAndCompare(ga);
        }
    } catch (const std::exception& e) {
        std::cerr << "[refit] 异常: " << e.what() << "\n";
        return 3;
    }
    std::cerr << "mode 须为 identity|refit|regen\n";
    return 1;
}

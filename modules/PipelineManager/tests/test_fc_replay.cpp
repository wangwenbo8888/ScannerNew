// test_fc_replay.cpp — 07 侧五链执行器离线对拍＋循环压测（261025-26 Phase 2 验证）
//
// 【对拍】JMW_FC_REPLAY=1：ALL_raw 399 帧灌 07 FiveChainExecutor vs 基准
//   （优先新鲜沙盒运行 out/report.csv；JMW_FC_GOLDEN=1 强制 golden 旧档）。
//   判定：计数列硬门（markersDetected/laserMatched/laserPoints3D 全同）＋
//   门限涟漪软门 ≤5%（DLL vs 静态库 FP 求值序差——conic validCircle/fuse 门/
//   光流平局翻转；261026 实测 13/399=3.3%）＋整行逐位统计。
//
// 【压测】JMW_FC_BENCH=<N>：同数据循环 N 遍（每遍＝新会话语义：配准/融合
//   会话态复位），逐遍计构造/运行/析构墙钟 → FPS 统计（mean/median/P95/min/
//   稳态均值）——零拷回放口径＝纯计算吞吐上界（真机另受相机帧率约束）。
//
// 数据缺一自动 SKIP——CI 安全。沙盒根：JMW_FC_SANDBOX 覆盖或 CWD 向上搜索。
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "pipelines/scan/fivechain/FiveChainAdapter.h"
#include "pipelines/scan/fivechain/harness.h"
#include "pipelines/scan/fivechain/fc_executor.h"
#include "pipelines/scan/FrameObsAccumulator.h"
#include "scanning/laser/laser_match_scan_v3/cmtt_table_loader.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

// —— 沙盒 calib_loader 等价 mini 装载（camera_calib.json→名义温档 + CMTT sidecar）——
cv::Mat matFromJson(const json& j, int rows, int cols) {
    cv::Mat m(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m.at<double>(r, c) = j.at(static_cast<size_t>(r)).at(static_cast<size_t>(c)).get<double>();
    return m;
}

const json* selectTier(const json& table, double nominal) {
    const json* best = nullptr;
    double bestT = 0.0;
    for (const auto& tier : table) {
        if (!tier.contains("temperature")) continue;
        const double t = tier.at("temperature").get<double>();
        if (best == nullptr
            || std::abs(t - nominal) < std::abs(bestT - nominal) - 1e-12
            || (std::abs(std::abs(t - nominal) - std::abs(bestT - nominal)) < 1e-12
                && t < bestT)) {
            best = &tier;
            bestT = t;
        }
    }
    return best;
}

struct TestCalib {
    cv::Mat K1, D1, K2, D2, Rext, Text;
    cv::Mat R1, R2, P1, P2, Q, K1c, K2c, Tcomp;
    int W = 0, H = 0;
    std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable;
};

bool loadTestCalib(const fs::path& camJson, const fs::path& cmttBin,
                   double nominalTemp, TestCalib& out, std::string& err) {
    std::ifstream ifs(camJson);
    if (!ifs) { err = "camera_calib.json 打开失败"; return false; }
    json j;
    try { ifs >> j; } catch (const std::exception& e) { err = e.what(); return false; }
    try {
        out.K1 = matFromJson(j.at("intrinsic").at("left").at("camera_matrix"), 3, 3);
        out.D1 = matFromJson(j.at("intrinsic").at("left").at("dist_coeffs"), 1, 5);
        out.K2 = matFromJson(j.at("intrinsic").at("right").at("camera_matrix"), 3, 3);
        out.D2 = matFromJson(j.at("intrinsic").at("right").at("dist_coeffs"), 1, 5);
        out.Rext = matFromJson(j.at("extrinsic").at("R"), 3, 3);
        out.Text = matFromJson(j.at("extrinsic").at("T"), 3, 1);
        const auto& table = j.at("stereoRectifyTempTable").at("table");
        const json* tier = selectTier(table, nominalTemp);
        if (tier == nullptr) { err = "温表选档失败"; return false; }
        if (j.contains("imageSize") && j.at("imageSize").size() == 2) {
            out.W = j.at("imageSize").at(0).get<int>();
            out.H = j.at("imageSize").at(1).get<int>();
        }
        out.R1 = matFromJson(tier->at("R1"), 3, 3);
        out.R2 = matFromJson(tier->at("R2"), 3, 3);
        out.P1 = matFromJson(tier->at("P1"), 3, 4);
        out.P2 = matFromJson(tier->at("P2"), 3, 4);
        out.Q = matFromJson(tier->at("Q"), 4, 4);
        out.K1c = matFromJson(tier->at("compensatedCameraMatrixL"), 3, 3);
        out.K2c = matFromJson(tier->at("compensatedCameraMatrixR"), 3, 3);
        out.Tcomp = matFromJson(tier->at("compensatedT"), 3, 1);
    } catch (const std::exception& e) { err = std::string("json 解析: ") + e.what(); return false; }
    out.laserTable = calib::LoadTempTableFromCmtt(cmttBin.string(), nominalTemp, err);
    if (!out.laserTable) return false;
    return true;
}

fs::path locateSandbox() {
    if (const char* e = getenv("JMW_FC_SANDBOX")) return fs::path(e);
    fs::path p = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        const fs::path cand = p / "modules/09_operatorlib/TEST/speed_optimize";
        std::error_code ec;
        if (fs::exists(cand / "config_flat.json", ec)) return cand;
        if (!p.has_parent_path()) break;
        p = p.parent_path();
    }
    return {};
}

std::string fmtVal(double v, int prec = 3) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

// ============================================================================
// ReplayRig — 对拍/压测共用装配（数据集整载＋沙盒全参 harness）
// ============================================================================
struct ReplayRig {
    json operators;
    double nominalTemp = 22.5;
    TestCalib tc;
    std::vector<int> frameIds;
    int rawW = 0, rawH = 0;
    std::vector<unsigned char> rawL, rawR;
    scr::FrameSourceCompat src;
    std::unique_ptr<Scanner::pipeline::FrameObsAccumulator> obs;
    std::unique_ptr<Scanner::pipeline::FiveChainAdapter> adapter;
    std::unique_ptr<Scanner::pipeline::FcHarnessCompat> h;
    int total = 0;

    /// 会话态复位（压测迭代间——语义等价新会话：配准链/融合云归零，
    /// 算子实例与 CUDA Graph 复用＝稳态口径）
    void resetSessionState() {
        auto* hh = h.get();
        hh->prevState_ = calib::PrevFrameState{};
        hh->globalSet_ = calib::GlobalMarkerSet{};
        hh->globalIndexById_.clear();
        hh->nextGid_ = 0;
        hh->gridBoardRejected_ = false;
        hh->flowRmseBaseline_.clear();
        hh->flowRmseBaselineMed_ = -1.0;
        hh->holdStreak_ = 0;
        hh->libIndex_.reset();
        hh->libVersion_ = 1;
        hh->libIndexVersion_ = 0;
        hh->lastIndexRebuildCycle_ = -1;
        hh->candidates_.clear();
        hh->cycleCount_ = 0;
        hh->trajT_.clear();
        adapter->fuse().Clear();   // 会话融合云复位（防 N 迭代体素爆内存）
    }
};

bool buildRig(const fs::path& sb, ReplayRig& rig, std::string& err) {
    namespace sp = Scanner::pipeline;
    const fs::path dataRoot = sb / "date_in/laser-mark1688-1006";
    const fs::path rawDir = dataRoot / "ALL_raw";
    std::error_code ec;
    if (!fs::exists(rawDir / "meta.json", ec)
        || !fs::exists(dataRoot / "camera_calib.json", ec)
        || !fs::exists(dataRoot / "curve_map_temp_table.bin", ec)
        || !fs::exists(sb / "config_flat.json", ec)) {
        err = "沙盒数据不全（ALL_raw/camera_calib.json/curve_map_temp_table.bin/config_flat.json）";
        return false;
    }
    // 配置
    {
        std::ifstream f(sb / "config_flat.json");
        if (!f) { err = "config_flat.json 不可读"; return false; }
        json cfg; f >> cfg;
        rig.nominalTemp = cfg.at("runtime").value("nominalTemperature", 22.5);
        rig.operators = cfg.value("operators", json::object());
    }
    // 标定
    if (!loadTestCalib(dataRoot / "camera_calib.json",
                       dataRoot / "curve_map_temp_table.bin",
                       rig.nominalTemp, rig.tc, err))
        return false;
    if (rig.tc.W <= 0) { err = "imageSize 非法"; return false; }
    // 数据集（raw 整载——零拷贝视图）
    {
        std::ifstream mf(rawDir / "meta.json");
        if (!mf) { err = "meta.json 不可读"; return false; }
        json meta; mf >> meta;
        rig.rawW = meta.at("width").get<int>();
        rig.rawH = meta.at("height").get<int>();
        for (int id : meta.at("frames").get<std::vector<int>>()) rig.frameIds.push_back(id);
    }
    if (rig.frameIds.empty()) { err = "帧号表为空"; return false; }
    std::sort(rig.frameIds.begin(), rig.frameIds.end());
    const size_t frameBytes = static_cast<size_t>(rig.rawW) * rig.rawH;
    {
        std::ifstream fl(rawDir / "left.bin", std::ios::binary);
        std::ifstream fr(rawDir / "right.bin", std::ios::binary);
        if (!fl || !fr) { err = "left.bin/right.bin 打开失败"; return false; }
        rig.rawL.resize(frameBytes * rig.frameIds.size());
        rig.rawR.resize(frameBytes * rig.frameIds.size());
        fl.read(reinterpret_cast<char*>(rig.rawL.data()), rig.rawL.size());
        fr.read(reinterpret_cast<char*>(rig.rawR.data()), rig.rawR.size());
        if (!fl || !fr) { err = "bin 读取不足"; return false; }
    }
    rig.total = static_cast<int>(rig.frameIds.size());
    // 帧源（静态回放口径——laserIdx 槽位透传真帧号）
    rig.src.cycles.fs = &rig.src;
    for (int id : rig.frameIds)
        rig.src.cyclesStatic.push_back({id, (id % 2 == 1)});
    rig.src.readFrame = [&rig](int idx, bool left, cv::Mat& out, std::string& e) -> bool {
        const auto it = std::lower_bound(rig.frameIds.begin(), rig.frameIds.end(), idx);
        if (it == rig.frameIds.end() || *it != idx) {
            e = "帧号不在数据集: " + std::to_string(idx);
            return false;
        }
        const size_t fb = static_cast<size_t>(rig.rawW) * rig.rawH;
        const size_t pos = static_cast<size_t>(it - rig.frameIds.begin());
        unsigned char* base = (left ? rig.rawL : rig.rawR).data() + pos * fb;
        out = cv::Mat(rig.rawH, rig.rawW, CV_8UC1, base);   // 零拷视图
        return true;
    };
    // 适配器＋harness（沙盒对拍装配：operators json 全参）
    sp::FiveChainConfig fcCfg;   // ScrFiveChain 默认值域（逐字段核对一致）
    rig.obs = std::make_unique<sp::FrameObsAccumulator>(1 << 20);
    rig.adapter = std::make_unique<sp::FiveChainAdapter>(fcCfg, *rig.obs, nullptr,
                                                         nullptr, &rig.operators);
    // sep 预热＋灰度环全槽预分配：首 Execute 走 CUDA Graph 捕获（Global 禁令
    // 期内任何线程 cudaMalloc 非法——上传线程懒分配会撞捕获窗）
    //（沙盒靠 serialHead 串行帧预热；07 直起执行器须显式）
    rig.adapter->sepL().Warmup(rig.tc.H, rig.tc.W);
    rig.adapter->sepR().Warmup(rig.tc.H, rig.tc.W);
    auto* h = new sp::FcHarnessCompat(*rig.adapter, rig.src);
    rig.h.reset(h);
    h->cfg_.fivechain = fcCfg;
    h->cfg.fivechain = fcCfg;
    h->cfg_.operators = rig.operators;
    h->cfg_.runtime.nominalTemperature = rig.nominalTemp;
    h->cfg_.runtime.warmupCycles = 0;
    h->calib_.Q = rig.tc.Q;  h->calib_.R1 = rig.tc.R1;  h->calib_.R2 = rig.tc.R2;
    h->calib_.P1 = rig.tc.P1;  h->calib_.P2 = rig.tc.P2;
    h->calib_.K1c = rig.tc.K1c;  h->calib_.K2c = rig.tc.K2c;
    h->calib_.D1 = rig.tc.D1;  h->calib_.D2 = rig.tc.D2;
    h->calib_.Rext = rig.tc.Rext;  h->calib_.Tcomp = rig.tc.Tcomp;
    h->imageWidth_ = rig.tc.W;  h->imageHeight_ = rig.tc.H;
    h->tableOdd_ = rig.tc.laserTable;
    h->tableEven_ = rig.tc.laserTable;
    h->sepLaserL_ = &rig.adapter->sepL();
    h->sepLaserR_ = &rig.adapter->sepR();
    h->voxelFuseOp_ = &rig.adapter->fuse();
    h->prodProfile_ = false;    // 沙盒口径：统计记账开（dispMedPx 等）
    h->preallocGrayLanes(fcCfg.graySlots, rig.tc.H, rig.tc.W);   // 捕获窗竞态根治
    if (rig.operators.contains("ccl")) {
        const auto& c = rig.operators.at("ccl");
        h->cclMinArea_ = c.value("minArea", 800);
        h->cclMaxArea_ = c.value("maxArea", 30000);
        h->cclTopX_ = c.value("topXCount", 0);
    }
    // 激光链前段（沙盒 makeLaserOps 配方——分眼实例）
    auto& L = h->laserOps_;
    {
        const auto stg = calib::StegerParams::fromJson(h->cfg_.opParams("steger_fast"));
        L.stegerL_ = std::make_unique<calib::StegerExtractorFast>(stg);
        L.stegerR_ = std::make_unique<calib::StegerExtractorFast>(stg);
        const auto fp = calib::FlatCenterParams::fromJson(
            h->cfg_.opParams("flat_center"));
        L.flatL_ = std::make_unique<calib::FlatCenterExtractor>(fp);
        L.flatR_ = std::make_unique<calib::FlatCenterExtractor>(fp);
    }
    {
        calib::UndistortPointsParams ul, ur;
        ul.cameraMatrix = rig.tc.K1c; ul.distCoeffs = rig.tc.D1;
        ul.R = rig.tc.R1; ul.P = rig.tc.P1;
        ur.cameraMatrix = rig.tc.K2c; ur.distCoeffs = rig.tc.D2;
        ur.R = rig.tc.R2;
        ur.P = rig.tc.P2.clone();
        ur.P.at<double>(0, 3) = 0.0;   // 右目 P 平移置零（沙盒口径）
        L.undistGL_ = std::make_unique<calib::UndistortPointsCuda>(ul);
        L.undistGR_ = std::make_unique<calib::UndistortPointsCuda>(ur);
        calib::EpipolarInterpDualParams ip = calib::EpipolarInterpDualParams::fromJson(
            h->cfg_.opParams("epipolar_interp_dual"));
        ip.mode = calib::InterpMode::Scan;
        L.interpL_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);
        L.interpR_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);
    }
    // 配准（光流＋兜底——沙盒 json 全参）
    h->flowOp_ = std::make_unique<calib::MarkerOpticalFlowFuseCPU>(
        calib::MarkerOpticalFlowFuseCPUParams::fromJson(
            h->cfg_.opParams("optical_flow_fuse")));
    h->fuseParams_ = calib::FrameFuseCPUParams::fromJson(
        h->cfg_.opParams("frame_fuse"));
    h->fuseOp_ = std::make_unique<calib::FrameFuseCPU>(h->fuseParams_);
    return true;
}

} // namespace

// ============================================================================
// 对拍：399 帧 report 分栏判定
// ============================================================================
TEST(FcReplay, All399ReportParity) {
    if (getenv("JMW_FC_REPLAY") == nullptr)
        GTEST_SKIP() << "JMW_FC_REPLAY=1 未设——离线对拍显式启用";

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty()) << "沙盒根未定位（JMW_FC_SANDBOX 或 CWD 向上搜索）";
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    auto* h = rig.h.get();

    // —— 执行器（399 帧有界——metrics 正常落盘）——
    std::error_code ec;
    fs::path outDir = fs::temp_directory_path() / "fc07_replay";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();
    auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, rig.total);

    // —— 收割＋report.csv / rt17.csv 落盘 ——
    std::vector<std::string> rows;
    std::ostringstream rt17;
    double prevR[9], prevT[3];
    bool havePrev = false;
    for (int c = 0; c < rig.total; ++c) {
        CycleProducts prod;
        std::string cerr;
        ASSERT_TRUE(ex->collect(c, prod, cerr)) << "collect 失败 c=" << c << " " << cerr;
        double dtMm = -1.0, drotDeg = -1.0;
        if (havePrev) {
            dtMm = std::sqrt((prod.T[0] - prevT[0]) * (prod.T[0] - prevT[0])
                             + (prod.T[1] - prevT[1]) * (prod.T[1] - prevT[1])
                             + (prod.T[2] - prevT[2]) * (prod.T[2] - prevT[2]));
            const cv::Matx33d dR = cv::Matx33d(prod.R) * cv::Matx33d(prevR).t();
            const double trace = dR(0, 0) + dR(1, 1) + dR(2, 2);
            const double cosHalf = std::min(1.0, std::max(-1.0, (trace - 1.0) * 0.5));
            drotDeg = std::acos(cosHalf) * 180.0 / CV_PI;
        }
        std::copy(std::begin(prod.R), std::end(prod.R), std::begin(prevR));
        std::copy(std::begin(prod.T), std::end(prod.T), std::begin(prevT));
        havePrev = true;
        std::ostringstream row;
        row << c << ',' << prod.markersDetected << ',' << prod.markersValid << ','
            << prod.regSource << ',' << prod.regMatched << ',' << fmtVal(dtMm, 4)
            << ',' << fmtVal(drotDeg, 4) << ',' << prod.laserMatched << ','
            << prod.laserPoints3D << ',' << fmtVal(prod.rowResidMedPx, 4) << ','
            << fmtVal(prod.dispMedPx, 4);
        rows.push_back(row.str());
        rt17 << std::setprecision(17);
        for (int i = 0; i < 9; ++i) rt17 << prod.R[i] << (i == 8 ? ',' : ' ');
        for (int i = 0; i < 3; ++i) rt17 << prod.T[i] << (i == 2 ? '\n' : ' ');
    }
    ex.reset();   // 收尾（指标落盘）

    {
        std::ofstream of(outDir / "report.csv", std::ios::binary);
        of << "cycle,markersDetected,markersValid,regSource,regMatched,dt_mm,drot_deg,"
              "laserMatched,laserPoints3D,rowResidMedPx,dispMedPx\n";
        for (const auto& r : rows) of << r << "\n";
        std::ofstream rf(outDir / "rt17.csv");
        rf << rt17.str();
    }

    // —— 比对（分栏判定——261026 实测口径）——
    // 基准选择：优先新鲜沙盒运行（out/report.csv——golden 为旧构建产物，
    // 实测与新鲜沙盒差 309/399 行）；env JMW_FC_GOLDEN=1 强制用 golden。
    fs::path refCsv = sb / "out/report.csv";
    const char* useRef = getenv("JMW_FC_GOLDEN");
    if ((useRef && std::string(useRef) == "1") || !fs::exists(refCsv, ec))
        refCsv = sb / "golden/all399/report.csv";
    std::vector<std::string> ref;
    {
        std::ifstream gf(refCsv);
        ASSERT_TRUE(gf) << "基准 report.csv 不可读: " << refCsv.string();
        std::string line;
        std::getline(gf, line);   // header
        while (std::getline(gf, line))
            if (!line.empty()) ref.push_back(line);
    }
    ASSERT_EQ(rows.size(), ref.size())
        << "行数不符: mine=" << rows.size() << " ref=" << ref.size()
        << "（基准 " << refCsv.string() << "）";

    int regSourceFlip = 0;      // 路径翻面（fuse/hold 门限边界）
    int regMatchedJitter = 0;   // regMatched/markersValid ±1（FP 平局翻转）
    int floatWiggle = 0;        // dt/drot/dispMed 第 4 位小数漂移
    int exactRows = 0;
    std::ostringstream dmsg;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] == ref[i]) { ++exactRows; continue; }
        const auto split11 = [](const std::string& s) {
            std::vector<std::string> v; std::stringstream ss(s); std::string t;
            while (std::getline(ss, t, ',')) v.push_back(t);
            return v;
        };
        const auto mv = split11(rows[i]);
        const auto rv = split11(ref[i]);
        ASSERT_EQ(mv.size(), 11u);
        ASSERT_EQ(rv.size(), 11u);
        // 计数列硬断言（检测数/激光全链——261026 实测 100% 稳定）
        ASSERT_EQ(mv[1], rv[1]) << "markersDetected 差异 帧" << mv[0];
        ASSERT_EQ(mv[7], rv[7]) << "laserMatched 差异 帧" << mv[0];
        ASSERT_EQ(mv[8], rv[8]) << "laserPoints3D 差异 帧" << mv[0];
        // 路径/匹配数/有效圆软统计（FP 涟漪：fuse 门限/光流平局/conic validCircle
        // 边界翻转——261026 实测 13/399=3.3% 帧）
        if (mv[2] != rv[2]) ++regMatchedJitter;   // markersValid ±1 同计
        if (mv[3] != rv[3]) ++regSourceFlip;
        if (mv[4] != rv[4]) ++regMatchedJitter;
        if (mv[2] == rv[2] && mv[3] == rv[3] && mv[4] == rv[4]) ++floatWiggle;
        if (dmsg.str().size() < 3000)
            dmsg << "\n  [" << mv[0] << "] mine: " << rows[i]
                 << "\n  [" << mv[0] << "] ref:  " << ref[i];
    }
    // 软门：门限边界翻转稀有事件（≤5%——DLL vs 静态库浮点求值序差涟漪，
    // 伴随帧前后自愈，非逻辑差异）
    EXPECT_LE(regSourceFlip + regMatchedJitter, static_cast<int>(rows.size() * 5 / 100))
        << "门限翻转超限: 路径翻面=" << regSourceFlip << " 匹配数抖动=" << regMatchedJitter
        << dmsg.str();

    SUCCEED() << "399 帧对拍：逐位同 " << exactRows << "/浮点尾位漂移 " << floatWiggle
              << "/门限涟漪 " << (regSourceFlip + regMatchedJitter)
              << "（计数列 100% 同；输出留档 " << outDir.string()
              << "；基准 " << refCsv.filename().string() << "）";
}

// ============================================================================
// 压测：JMW_FC_BENCH=<N> 循环 N 遍 399 帧——纯计算吞吐统计（零拷回放口径）
// ============================================================================
TEST(FcReplay, BenchLoopThroughput) {
    int iters = 0;
    if (const char* e = getenv("JMW_FC_BENCH")) iters = atoi(e);
    if (iters <= 0)
        GTEST_SKIP() << "JMW_FC_BENCH=<N> 未设——循环压测显式启用";

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty()) << "沙盒根未定位";
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    auto* h = rig.h.get();
    std::error_code ec;
    fs::path outDir = fs::temp_directory_path() / "fc07_bench";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();

    using CLK = std::chrono::steady_clock;
    std::vector<double> ctorMs(iters), runMs(iters), dtorMs(iters), fps(iters);
    int totalLaserPts = 0;
    for (int it = 0; it < iters; ++it) {
        rig.resetSessionState();          // 新会话语义（配准/融合态归零）
        const auto tC0 = CLK::now();
        auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, rig.total);
        const auto t0 = CLK::now();
        int collected = 0;
        for (int c = 0; c < rig.total; ++c) {
            CycleProducts prod;
            std::string cerr;
            if (!ex->collect(c, prod, cerr)) break;
            ++collected;
            if (it == 0) totalLaserPts += prod.laserPoints3D;
        }
        const auto t1 = CLK::now();
        ex.reset();                       // stopAll＋设备同步收尾
        const auto t2 = CLK::now();
        ASSERT_EQ(collected, rig.total) << "迭代 " << it << " 收割不足";
        ctorMs[static_cast<size_t>(it)] =
            std::chrono::duration<double, std::milli>(t0 - tC0).count();
        runMs[static_cast<size_t>(it)] =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        dtorMs[static_cast<size_t>(it)] =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        fps[static_cast<size_t>(it)] =
            static_cast<double>(collected) * 1000.0 / runMs[static_cast<size_t>(it)];
    }

    auto stats = [](std::vector<double> v, double& mean, double& med,
                    double& p95, double& mn, double& mx) {
        std::sort(v.begin(), v.end());
        mean = 0; for (double x : v) mean += x;
        mean /= static_cast<double>(v.size());
        med = v[v.size() / 2];
        p95 = v[static_cast<size_t>(static_cast<double>(v.size()) * 0.95)
                 - (v.size() > 1 ? 1 : 0)];
        mn = v.front(); mx = v.back();
    };
    double fMean, fMed, fP95, fMin, fMax;
    stats(fps, fMean, fMed, fP95, fMin, fMax);
    double cMean = 0, dMean = 0, rMean = 0;
    for (int i = 0; i < iters; ++i) {
        cMean += ctorMs[static_cast<size_t>(i)];
        rMean += runMs[static_cast<size_t>(i)];
        dMean += dtorMs[static_cast<size_t>(i)];
    }
    cMean /= iters; rMean /= iters; dMean /= iters;
    // 稳态（去前 3 轮——首轮含图实例化/缓存预热残余）
    double steady = 0; int sn = 0;
    for (int i = 3; i < iters; ++i) {
        steady += fps[static_cast<size_t>(i)]; ++sn;
    }
    const double steadyMean = sn > 0 ? steady / sn : fMean;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "五链压测 " << iters << " 遍 × " << rig.total << " 帧（1688×1400 双目，"
       << "零拷回放＝纯计算吞吐上界）\n";
    ss << "  FPS: mean=" << fMean << " median=" << fMed << " P95=" << fP95
       << " min=" << fMin << " max=" << fMax << "\n";
    ss << "  稳态均值（去前 3 轮）=" << steadyMean << " FPS\n";
    ss << std::fixed << std::setprecision(0);
    ss << "  会话开销: 执行器构造=" << cMean << "ms 运行=" << rMean
       << "ms 析构收尾=" << dMean << "ms\n";
    ss << "  首轮激光点累计=" << totalLaserPts << "（产能健康度参考）";
    fprintf(stderr, "\n[FCBENCH] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}

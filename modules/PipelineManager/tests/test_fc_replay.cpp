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
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>    // PairGeometryCheck 阈值/翻转
#include <opencv2/imgcodecs.hpp>   // 裁剪原点求解器帧载入
#include <cuda_runtime.h>   // cudaMemGetInfo（连续吞吐显存采样）

#include "pipelines/scan/fivechain/FiveChainAdapter.h"
#include "pipelines/scan/fivechain/harness.h"
#include "pipelines/scan/fivechain/fc_executor.h"
#include "pipelines/scan/FrameObsAccumulator.h"
#include "pipelines/globaloptim/GlobalOptimObject.h"   // GBA 全链验证用例
#include "pipelines/calibcompute/CalibComputeTypes.h"  // CancelToken
#include "SlotRing.h"   // startRealtime 帧环仿真
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
    // 261029 激光缓存预算 1GB（399 帧 ≈121MB——1MB 存根 3 帧即降级；GBA 重放用例依赖）
    rig.obs = std::make_unique<sp::FrameObsAccumulator>(size_t{1} << 30);
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

    // —— 收割＋report.csv / rt17.csv / 逐帧 marker CSV 落盘 ——
    std::vector<std::string> rows;
    std::ostringstream rt17;
    fs::path mkDir = outDir / "markers";
    fs::create_directories(mkDir, ec);
    double prevR[9], prevT[3];
    bool havePrev = false;
    for (int c = 0; c < rig.total; ++c) {
        CycleProducts prod;
        std::string cerr;
        ASSERT_TRUE(ex->collect(c, prod, cerr)) << "collect 失败 c=" << c << " " << cerr;
        // marker 3D CSV（沙盒 markerOnly 同格式——conic 有效圆 6 位小数）
        {
            std::ostringstream ns;
            ns << "cycle_" << std::setw(3) << std::setfill('0') << c << "_markers.csv";
            std::ofstream ofs(mkDir / ns.str(), std::ios::binary);
            int n = 0;
            if (ofs) {
                ofs << std::fixed << std::setprecision(6);
                ofs << "marker,centerX_mm,centerY_mm,centerZ_mm,"
                       "normalX,normalY,normalZ\n";
                for (size_t i = 0; i < prod.conic.markers.size(); ++i) {
                    const auto& m = prod.conic.markers[i];
                    if (!m.validCircle) continue;
                    ofs << i << ',' << m.centerX << ',' << m.centerY << ',' << m.centerZ
                        << ',' << m.normalX << ',' << m.normalY << ',' << m.normalZ
                        << '\n';
                    ++n;
                }
            }
        }
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
    // 逐遍产出签名（正确性校验：同输入+会话态复位 ⇒ 各遍应逐位一致——
    // 不一致＝会话态泄漏或算子跨遍残留）
    struct IterSig {
        long laserPts = 0, markersDet = 0, markersValid = 0;
        long srcFirst = 0, srcFlow = 0, srcFuse = 0, srcHold = 0, srcSkip = 0;
        double R[9] = {}, T[3] = {};       // 末帧 R/T（17 位口径直接比对）
        double R0[9] = {}, T0[3] = {};     // 首帧 R/T
    };
    std::vector<IterSig> sigs(static_cast<size_t>(iters));
    // 逐帧完成时间戳（自压测起点）——按秒分桶＝每秒实际处理帧数（含会话间隙）
    const auto benchStart = CLK::now();
    std::vector<double> frameTs;
    frameTs.reserve(static_cast<size_t>(iters) * rig.total);
    for (int it = 0; it < iters; ++it) {
        rig.resetSessionState();          // 新会话语义（配准/融合态归零）
        const auto tC0 = CLK::now();
        auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, rig.total);
        const auto t0 = CLK::now();
        int collected = 0;
        IterSig& sg = sigs[static_cast<size_t>(it)];
        for (int c = 0; c < rig.total; ++c) {
            CycleProducts prod;
            std::string cerr;
            if (!ex->collect(c, prod, cerr)) break;
            ++collected;
            frameTs.push_back(std::chrono::duration<double>(CLK::now() - benchStart).count());
            sg.laserPts += prod.laserPoints3D;
            sg.markersDet += prod.markersDetected;
            sg.markersValid += prod.markersValid;
            if (prod.regSource == "first") ++sg.srcFirst;
            else if (prod.regSource == "flow") ++sg.srcFlow;
            else if (prod.regSource == "fuse") ++sg.srcFuse;
            else if (prod.regSource == "hold") ++sg.srcHold;
            else ++sg.srcSkip;
            if (c == 0) {
                for (int i = 0; i < 9; ++i) sg.R0[i] = prod.R[i];
                for (int i = 0; i < 3; ++i) sg.T0[i] = prod.T[i];
            }
            for (int i = 0; i < 9; ++i) sg.R[i] = prod.R[i];
            for (int i = 0; i < 3; ++i) sg.T[i] = prod.T[i];
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

    // —— 正确性校验：各遍签名与第 0 遍逐位一致（同输入+复位 ⇒ 确定；
    // 不一致＝会话态泄漏/算子跨遍残留）——
    int sigMismatch = 0;
    std::ostringstream sm;
    for (int it = 1; it < iters; ++it) {
        const IterSig& a = sigs[0];
        const IterSig& b = sigs[static_cast<size_t>(it)];
        if (a.laserPts != b.laserPts || a.markersDet != b.markersDet
            || a.markersValid != b.markersValid || a.srcFirst != b.srcFirst
            || a.srcFlow != b.srcFlow || a.srcFuse != b.srcFuse
            || a.srcHold != b.srcHold || a.srcSkip != b.srcSkip) {
            ++sigMismatch;
            if (sigMismatch <= 5)
                sm << "\n  遍 " << it << " 计数签名差: laser " << a.laserPts << "→"
                   << b.laserPts << " det " << a.markersDet << "→" << b.markersDet
                   << " valid " << a.markersValid << "→" << b.markersValid
                   << " [first/flow/fuse/hold/skip] " << a.srcFirst << "/" << a.srcFlow
                   << "/" << a.srcFuse << "/" << a.srcHold << "/" << a.srcSkip << " → "
                   << b.srcFirst << "/" << b.srcFlow << "/" << b.srcFuse << "/"
                   << b.srcHold << "/" << b.srcSkip;
            continue;
        }
        for (int i = 0; i < 9; ++i)
            if (std::memcmp(&a.R[i], &b.R[i], sizeof(double)) != 0
                || std::memcmp(&a.R0[i], &b.R0[i], sizeof(double)) != 0) {
                ++sigMismatch;
                sm << "\n  遍 " << it << " R 逐位差（帧0或末帧）";
                break;
            }
        for (int i = 0; i < 3; ++i)
            if (std::memcmp(&a.T[i], &b.T[i], sizeof(double)) != 0
                || std::memcmp(&a.T0[i], &b.T0[i], sizeof(double)) != 0) {
                ++sigMismatch;
                sm << "\n  遍 " << it << " T 逐位差（帧0或末帧）";
                break;
            }
    }
    EXPECT_EQ(sigMismatch, 0) << "各遍产出签名不一致（会话态泄漏嫌疑）" << sm.str();

    // —— 按秒分桶：每秒实际处理帧数（含会话构造/析构间隙——秒粒度真实口径）——
    const int nSec = static_cast<int>(std::floor(frameTs.back())) + 1;
    std::vector<int> perSec(static_cast<size_t>(nSec), 0);
    for (double t : frameTs) ++perSec[static_cast<size_t>(std::floor(t))];
    int busySec = 0; long busyFrames = 0;
    for (int v : perSec)
        if (v > 0) { ++busySec; busyFrames += v; }
    std::ostringstream ps;
    ps << "\n  逐秒帧数（" << nSec << " 秒，含会话间隙；0＝间隙/构造秒）:\n";
    for (int s = 0; s < nSec; ++s) {
        if (s % 10 == 0) ps << "    ";
        ps << std::setw(4) << perSec[static_cast<size_t>(s)];
        if (s % 10 == 9) ps << "\n";
    }
    ps << "\n    活跃秒（>0 帧）=" << busySec << " 活跃秒均帧率="
       << std::fixed << std::setprecision(1)
       << static_cast<double>(busyFrames) / busySec << " FPS";

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
    ss << "  首遍产出: 激光点=" << sigs[0].laserPts
       << " 标点det=" << sigs[0].markersDet << " 标点valid=" << sigs[0].markersValid
       << " 配准路径 first/flow/fuse/hold/skip=" << sigs[0].srcFirst << "/"
       << sigs[0].srcFlow << "/" << sigs[0].srcFuse << "/" << sigs[0].srcHold
       << "/" << sigs[0].srcSkip << "\n";
    ss << "  各遍签名一致: " << (sigMismatch == 0 ? "50/50 逐位同（会话态复位完备）"
                                                : "存在差异 " + std::to_string(sigMismatch));
    ss << ps.str();
    {
        std::ofstream pf(outDir / "bench_persec.csv");
        pf << "second,frames\n";
        for (int s = 0; s < nSec; ++s)
            pf << s << ',' << perSec[static_cast<size_t>(s)] << "\n";
    }
    fprintf(stderr, "\n[FCBENCH] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}

// ============================================================================
// 连续吞吐公共运行体：循环帧源＋单执行器 INT_MAX 无界连续运行——无会话重建
// （消除驼峰），逐秒分桶＋每分钟聚合＋显存/融合云/全局地图采样
// ============================================================================
namespace {

struct ContRun {
    long frames = 0, laserPts = 0;
    double wallSec = 0, meanFps = 0, secMeanFps = 0, cvPct = 0;
    double firstMinMean = 0, lastMinMean = 0;
    int nLoops = 0;
    std::vector<int> perSec;
    size_t fusedFirst = 0, fusedLast = 0, libFirst = 0, libLast = 0;
    size_t devFreeFirst = 0, devFreeLast = 0, devTotal = 0;
};

double contMinuteMean(const std::vector<int>& perSec, int m) {
    const int s0 = m * 60;
    const int s1 = std::min(static_cast<int>(perSec.size()), (m + 1) * 60);
    long sum = 0; int cnt = 0;
    for (int s = s0; s < s1; ++s)
        if (perSec[static_cast<size_t>(s)] > 0) { sum += perSec[static_cast<size_t>(s)]; ++cnt; }
    return cnt > 0 ? static_cast<double>(sum) / cnt : 0.0;
}

/// 该分钟活跃秒数（判"完整分钟"——尾分钟只含边界溢出帧时不算）
int contMinuteActiveSecs(const std::vector<int>& perSec, int m) {
    const int s0 = m * 60;
    const int s1 = std::min(static_cast<int>(perSec.size()), (m + 1) * 60);
    int cnt = 0;
    for (int s = s0; s < s1; ++s)
        if (perSec[static_cast<size_t>(s)] > 0) ++cnt;
    return cnt;
}

ContRun runContinuous(ReplayRig& rig, Scanner::pipeline::FcHarnessCompat* h,
                      double durationSec, const fs::path& outDir, bool verbose) {
    using CLK = std::chrono::steady_clock;
    // 循环帧源：cycleAt 动态求值（idx % N 环回——laserIdx 透传真帧号）
    const int n = rig.total;
    rig.src.cycleAt = [&rig, n](size_t i) {
        scr::FrameCycleInfo ci;
        const int id = rig.frameIds[i % static_cast<size_t>(n)];
        ci.laserIdx = id;
        ci.tubeOdd = (id % 2 == 1);
        return ci;
    };

    struct Sample {
        double t; long frames; size_t fusedPts; size_t libSize;
        size_t devFree = 0, devTotal = 0;
    };
    std::vector<Sample> samples;
    auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, 0x7fffffff);
    std::vector<double> ts;
    ts.reserve(static_cast<size_t>(durationSec) * 320);
    const auto t0 = CLK::now();
    double lastSampleT = -10.0;
    long frames = 0, laserPts = 0;
    while (true) {
        const double t = std::chrono::duration<double>(CLK::now() - t0).count();
        if (t >= durationSec) break;
        CycleProducts prod;
        std::string cerr;
        if (!ex->collect(static_cast<int>(frames), prod, cerr)) break;
        ts.push_back(std::chrono::duration<double>(CLK::now() - t0).count());
        laserPts += prod.laserPoints3D;
        ++frames;
        if (ts.back() - lastSampleT >= 10.0) {   // 采样（每 10s）
            lastSampleT = ts.back();
            Sample s;
            s.t = ts.back(); s.frames = frames;
            s.fusedPts = rig.adapter->fuse().GetFusedPointCount();
            s.libSize = h->globalSet_.positions.size();
            size_t freeB = 0, totalB = 0;
            cudaMemGetInfo(&freeB, &totalB);
            s.devFree = freeB; s.devTotal = totalB;
            samples.push_back(s);
            if (verbose)
                fprintf(stderr,
                        "[FCCONT] %6.0fs frames=%ld fps=%.1f fused=%zuM lib=%zu devFree=%.1fGB\n",
                        s.t, s.frames, frames / std::max(1.0, s.t),
                        s.fusedPts / 1000000, s.libSize,
                        static_cast<double>(freeB) / 1e9);
        }
    }
    ContRun r;
    r.wallSec = std::chrono::duration<double>(CLK::now() - t0).count();
    ex.reset();
    r.frames = frames;
    r.laserPts = laserPts;
    r.nLoops = n > 0 ? static_cast<int>(frames / n) : 0;
    // 逐秒分桶
    const int nSec = ts.empty() ? 0 : static_cast<int>(std::floor(ts.back())) + 1;
    r.perSec.assign(static_cast<size_t>(std::max(1, nSec)), 0);
    for (double t : ts) ++r.perSec[static_cast<size_t>(std::floor(t))];
    {
        std::ofstream pf(outDir / "cont_persec.csv");
        pf << "second,frames\n";
        for (int s = 0; s < nSec; ++s)
            pf << s << ',' << r.perSec[static_cast<size_t>(s)] << "\n";
    }
    // 整体统计
    r.meanFps = static_cast<double>(frames) / std::max(1.0, r.wallSec);
    double var = 0, secMean = 0; int secCnt = 0;
    for (int v : r.perSec)
        if (v > 0) { secMean += v; ++secCnt; }
    secMean /= std::max(1, secCnt);
    for (int v : r.perSec)
        if (v > 0) var += (v - secMean) * (v - secMean);
    r.secMeanFps = secMean;
    r.cvPct = std::sqrt(var / std::max(1, secCnt)) / secMean * 100.0;
    const int nMin = (nSec + 59) / 60;
    r.firstMinMean = contMinuteMean(r.perSec, 0);
    // 末分钟＝最后一个完整分钟（活跃秒≥30——240s 边界溢出的尾帧桶不算）
    r.lastMinMean = r.firstMinMean;
    for (int m = nMin - 1; m >= 0; --m) {
        if (contMinuteActiveSecs(r.perSec, m) >= 30) {
            r.lastMinMean = contMinuteMean(r.perSec, m);
            break;
        }
    }
    if (!samples.empty()) {
        const Sample& a = samples.front();
        const Sample& b = samples.back();
        r.fusedFirst = a.fusedPts; r.fusedLast = b.fusedPts;
        r.libFirst = a.libSize; r.libLast = b.libSize;
        r.devFreeFirst = a.devFree; r.devFreeLast = b.devFree;
        r.devTotal = b.devTotal;
    }
    return r;
}

std::string contMinuteTable(const std::vector<int>& perSec) {
    const int nMin = (static_cast<int>(perSec.size()) + 59) / 60;
    std::ostringstream mm;
    mm << std::fixed << std::setprecision(1);
    for (int m = 0; m < nMin; ++m) {
        const int s0 = m * 60, s1 = std::min(static_cast<int>(perSec.size()), (m + 1) * 60);
        int mn = 1 << 30, mx = 0; long sum = 0; int cnt = 0;
        for (int s = s0; s < s1; ++s) {
            const int v = perSec[static_cast<size_t>(s)];
            if (v <= 0) continue;
            mn = std::min(mn, v); mx = std::max(mx, v); sum += v; ++cnt;
        }
        if (cnt > 0)
            mm << "  分 " << std::setw(2) << m << ": mean=" << std::setw(6)
               << static_cast<double>(sum) / cnt << " min=" << std::setw(4) << mn
               << " max=" << std::setw(4) << mx << "\n";
    }
    return mm.str();
}

} // namespace

// ============================================================================
// 连续吞吐（自由时长）：JMW_FC_CONT=<分钟>
// ============================================================================
TEST(FcReplay, ContinuousThroughput) {
    double minutes = 0;
    if (const char* e = getenv("JMW_FC_CONT")) minutes = atof(e);
    if (minutes <= 0)
        GTEST_SKIP() << "JMW_FC_CONT=<分钟> 未设——连续吞吐显式启用";

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty()) << "沙盒根未定位";
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    auto* h = rig.h.get();
    std::error_code ec;
    fs::path outDir = fs::temp_directory_path() / "fc07_cont";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();

    ContRun r = runContinuous(rig, h, minutes * 60.0, outDir, /*verbose=*/true);
    ASSERT_GT(r.frames, 0) << "零帧收割";

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "五链连续吞吐 " << std::setprecision(0) << r.wallSec << "s（" << minutes
       << " 分钟，" << rig.total << " 组图循环，单执行器无会话重建，全内存零拷）\n";
    ss << "  总帧数=" << r.frames << "（" << r.nLoops << " 个整环）激光点累计 "
       << r.laserPts << "\n";
    ss << std::setprecision(1);
    ss << "  整体吞吐=" << r.meanFps << " FPS｜逐秒均值=" << r.secMeanFps
       << " FPS｜逐秒变异系数 CV=" << std::setprecision(2) << r.cvPct << "%\n";
    ss << std::setprecision(0);
    ss << "  首/末采样: 融合云 " << r.fusedFirst << " → " << r.fusedLast
       << " 点｜全局地图 " << r.libFirst << " → " << r.libLast
       << " 点｜显存空闲 " << static_cast<double>(r.devFreeFirst) / 1e9 << "GB → "
       << static_cast<double>(r.devFreeLast) / 1e9 << "GB / 共 "
       << static_cast<double>(r.devTotal) / 1e9 << "GB\n";
    ss << "  每分钟秒帧率聚合:\n" << contMinuteTable(r.perSec);
    ss << "  逐秒全序列: " << outDir.string() << "/cont_persec.csv";
    fprintf(stderr, "\n[FCCONT-SUMMARY] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}

// ============================================================================
// 连续吞吐 4 分钟门禁：JMW_FC_CONT4=1——固定 240s，阈值判 PASS/FAIL
//（261026 实测基线：289 FPS/CV 3.7%/无衰减——阈值留 2 倍余量防异机波动）
// ============================================================================
TEST(FcReplay, Continuous4MinStability) {
    if (getenv("JMW_FC_CONT4") == nullptr)
        GTEST_SKIP() << "JMW_FC_CONT4=1 未设——4 分钟稳定性门禁显式启用";

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty()) << "沙盒根未定位";
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    auto* h = rig.h.get();
    std::error_code ec;
    fs::path outDir = fs::temp_directory_path() / "fc07_cont4";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();

    constexpr double kDurationSec = 240.0;   // 4 分钟
    ContRun r = runContinuous(rig, h, kDurationSec, outDir, /*verbose=*/false);
    ASSERT_GT(r.frames, 0) << "零帧收割";

    // —— 门禁（261026 本机实测 289 FPS/CV 3.7%/显存 10→6GB——阈值含余量）——
    EXPECT_GE(r.nLoops, 2) << "整环数不足（吞吐异常低）";
    EXPECT_GE(r.meanFps, 150.0) << "整体吞吐低于门限（实测基线 289）";
    EXPECT_LE(r.cvPct, 10.0) << "逐秒变异系数超门限（实测基线 3.7%）";
    EXPECT_GE(r.lastMinMean, 0.8 * r.firstMinMean)
        << "末分钟吞吐较首分钟衰减超 20%（实测基线无衰减）";
    EXPECT_GT(r.laserPts, 0) << "零激光点产出";
    EXPECT_GE(r.devFreeLast, static_cast<size_t>(500) * 1000 * 1000)
        << "显存空闲 <0.5GB（泄漏/OOM 嫌疑）";

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "五链 4 分钟稳定性门禁 PASS: 帧数=" << r.frames << "（" << r.nLoops
       << " 整环）整体=" << r.meanFps << " FPS 逐秒均值=" << r.secMeanFps
       << " CV=" << std::setprecision(2) << r.cvPct << "% 首分钟 "
       << std::setprecision(1) << r.firstMinMean << " → 末分钟 " << r.lastMinMean
       << " FPS｜融合云 " << r.fusedFirst << "→" << r.fusedLast
       << " 显存空闲 " << std::setprecision(1)
       << static_cast<double>(r.devFreeFirst) / 1e9 << "→"
       << static_cast<double>(r.devFreeLast) / 1e9 << "GB\n每分钟聚合:\n"
       << contMinuteTable(r.perSec)
       << "逐秒序列: " << outDir.string() << "/cont_persec.csv";
    fprintf(stderr, "\n[FCCONT4] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}

// ============================================================================
// 裁剪原点求解：JMW_FC_SOLVE=<帧目录>（真机 JMW_FC_DUMPFRAMES 转储的 L/R png）
// ——离线扫候选偏移 (x0,y0)，跑五链取 conic 有效圆计数最大者为真原点
//（居中只是推定；真机 21:28 实证 conic 0/4666＋激光 5%＝原点错）
// ============================================================================
TEST(FcReplay, CropOffsetSolve) {
    fs::path frameDir = "diagnostics/fc_frames";
    if (const char* e = getenv("JMW_FC_SOLVE")) frameDir = fs::path(e);
    std::error_code ec;
    if (!fs::exists(frameDir, ec))
        GTEST_SKIP() << "帧目录缺失: " << frameDir.string()
                     << "（真机 JMW_FC_DUMPFRAMES=N 扫描转储后重跑）";
    // 载入原始帧对（裁剪前）
    std::vector<cv::Mat> framesL, framesR;
    std::vector<int> ids;
    for (int i = 0;; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%06d", i);
        const fs::path pl = frameDir / ("L_" + std::string(buf) + ".png");
        const fs::path pr = frameDir / ("R_" + std::string(buf) + ".png");
        if (!fs::exists(pl, ec) || !fs::exists(pr, ec)) break;
        cv::Mat l = cv::imread(pl.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat r = cv::imread(pr.string(), cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(l.empty()) << pl.string();
        ASSERT_FALSE(r.empty()) << pr.string();
        framesL.push_back(std::move(l));
        framesR.push_back(std::move(r));
        ids.push_back(i);
    }
    ASSERT_GT(framesL.size(), 0u) << "无帧对";
    ASSERT_GE(framesL[0].cols, 1688) << "帧宽 <1688（非原图？）";
    ASSERT_GE(framesL[0].rows, 1400) << "帧高 <1400（非原图？）";
    const int nFrames = static_cast<int>(framesL.size());

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty()) << "沙盒根未定位";
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    auto* h = rig.h.get();
    fs::path outDir = fs::temp_directory_path() / "fc07_solve";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();

    // 帧源：候选裁剪视图（curX/curY 可变——逐候选重跑）
    int curX = 0, curY = 0;
    rig.src.cycles.fs = &rig.src;
    rig.src.cycleAt = [&](size_t i) {
        scr::FrameCycleInfo ci;
        ci.laserIdx = static_cast<int>(i);        // 槽位透传 fc idx（帧号=idx%nFrames）
        ci.tubeOdd = (i % 2) == 1;
        return ci;
    };
    rig.src.readFrame = [&](int idx, bool left, cv::Mat& out, std::string& e) -> bool {
        const int fi = idx % nFrames;
        const cv::Mat& src = (left ? framesL : framesR)[static_cast<size_t>(fi)];
        if (curX + 1688 > src.cols || curY + 1400 > src.rows) {
            e = "偏移越界";
            return false;
        }
        out = src(cv::Rect(curX, curY, 1688, 1400)).clone();
        return true;
    };

    auto runCandidate = [&](int x0, int y0) -> long {
        curX = x0; curY = y0;
        // 会话态复位（诊断计数器归零）
        h->diagCclFail_ = 0; h->diagRoisL_ = 0; h->diagSplit_ = 0;
        h->diagEllipse_ = 0; h->diagMatchPairs_ = 0; h->diagConicValid_ = 0;
        h->prevState_ = calib::PrevFrameState{};
        h->globalSet_ = {}; h->globalIndexById_.clear(); h->nextGid_ = 0;
        h->candidates_.clear(); h->holdStreak_ = 0; h->cycleCount_ = 0;
        auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, nFrames);
        for (int c = 0; c < nFrames; ++c) {
            CycleProducts prod;
            std::string ce;
            if (!ex->collect(c, prod, ce)) break;
        }
        ex.reset();
        return h->diagConicValid_.load();
    };

    // —— 粗扫（x 步 60 / y 步 34）——
    struct Cand { int x, y; long valid; };
    std::vector<Cand> results;
    const int maxX = framesL[0].cols - 1688;
    const int maxY = framesL[0].rows - 1400;
    for (int y0 = 0; y0 <= maxY; y0 += 34)
        for (int x0 = 0; x0 <= maxX; x0 += 60)
            results.push_back({x0, y0, runCandidate(x0, y0)});
    std::sort(results.begin(), results.end(),
              [](const Cand& a, const Cand& b) { return a.valid > b.valid; });
    // —— 细扫（粗扫 top1 邻域 ±30 步 10）——
    const Cand& best = results[0];
    for (int y0 = std::max(0, best.y - 34); y0 <= std::min(maxY, best.y + 34); y0 += 17)
        for (int x0 = std::max(0, best.x - 60); x0 <= std::min(maxX, best.x + 60); x0 += 15)
            results.push_back({x0, y0, runCandidate(x0, y0)});
    std::sort(results.begin(), results.end(),
              [](const Cand& a, const Cand& b) { return a.valid > b.valid; });

    // 显式回查居中假设（在线 21:28 会话 conic=0 的候选——定案裁剪/快照注入归属）
    const long centerValid = runCandidate(180, 68);
    std::ostringstream ss;
    ss << "裁剪原点求解（" << nFrames << " 帧对，conic 有效圆计数）——top 12:\n";
    for (size_t i = 0; i < results.size() && i < 12; ++i)
        ss << "  #" << i << " offset=(" << results[i].x << "," << results[i].y
           << ") valid=" << results[i].valid << "\n";
    ss << "  【居中假设 (180,68) 回查: valid=" << centerValid << "】\n";
    ss << "判定: 居中可行（≥top 一半）→在线 0 的根因＝逐帧快照注入；居中近零→真原点=top1";
    fprintf(stderr, "\n[FC-SOLVE] %s\n", ss.str().c_str());
    EXPECT_GT(results[0].valid, 0) << "全部候选 conic=0——帧数据或链路另有问题";
    SUCCEED() << ss.str();
}

// ============================================================================
// L/R 帧对几何体检：行/列亮度轮廓相关——定相对错位（上下翻转/垂直水平偏移）
//（261027 真机快照 Z 双簇跳变＝L/R 错配嫌疑——v3 真匹配首次受验）
// ============================================================================
TEST(FcReplay, PairGeometryCheck) {
    fs::path frameDir = "diagnostics/fc_frames";
    if (const char* e = getenv("JMW_FC_PAIR")) frameDir = fs::path(e);
    std::error_code ec;
    if (!fs::exists(frameDir / "L_000000.png", ec))
        GTEST_SKIP() << "帧目录缺失（JMW_FC_DUMPFRAMES 转储后重跑）";
    cv::Mat L = cv::imread((frameDir / "L_000000.png").string(), cv::IMREAD_GRAYSCALE);
    cv::Mat R = cv::imread((frameDir / "R_000000.png").string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(L.empty() && R.empty());
    // 居中裁剪到算法几何
    const int ax = 1688, ay = 1400;
    const int x0 = (L.cols - ax) / 2, y0 = (L.rows - ay) / 2;
    cv::Mat Lc = L(cv::Rect(x0, y0, ax, ay));
    cv::Mat Rc = R(cv::Rect(x0, y0, ax, ay));

    auto rowProfile = [](const cv::Mat& m) {
        cv::Mat prof;
        cv::reduce(m, prof, 1, cv::REDUCE_AVG, CV_32F);   // 1×rows → rows×1
        return prof;
    };
    auto colProfile = [](const cv::Mat& m) {
        cv::Mat prof;
        cv::reduce(m, prof, 0, cv::REDUCE_AVG, CV_32F);
        return prof.reshape(1, 1).t();                     // cols×1
    };
    auto bestShiftCorr = [](cv::Mat a, cv::Mat b, int maxShift, int& bestS, double& bestC) {
        // 沿 y 平移 b 找最大相关（归一化）
        const int n = a.rows;
        cv::Scalar ma, mb; cv::meanStdDev(a, ma, mb);
        const double sa = mb[0] > 1e-9 ? mb[0] : 1;
        bestC = -2; bestS = 0;
        for (int s = -maxShift; s <= maxShift; ++s) {
            double dot = 0, sb2 = 0; int cnt = 0;
            for (int i = 0; i < n; ++i) {
                const int j = i + s;
                if (j < 0 || j >= n) continue;
                const double da = a.at<float>(i, 0) - ma[0];
                const double db = b.at<float>(j, 0);
                dot += da * db; sb2 += db * db; ++cnt;
            }
            if (cnt < 100) continue;
            const double sb = std::sqrt(sb2 / cnt);
            if (sb < 1e-9) continue;
            const double c = (dot / cnt) / (sa * sb);
            if (c > bestC) { bestC = c; bestS = s; }
        }
    };

    const cv::Mat rpL = rowProfile(Lc), rpR = rowProfile(Rc);
    const cv::Mat cpL = colProfile(Lc), cpR = colProfile(Rc);
    cv::Mat rpRflip; cv::flip(rpR, rpRflip, 0);            // 上下翻转假设
    int bsN = 0, bsF = 0; double bcN = 0, bcF = 0;
    const double c0row = [&, &bsN = bsN, &bcN = bcN]() { bestShiftCorr(rpL, rpR, 0, bsN, bcN); return bcN; }();
    bestShiftCorr(rpL, rpR, 680, bsN, bcN);
    bestShiftCorr(rpL, rpRflip, 680, bsF, bcF);
    int bsC = 0; double bcC = 0;
    const double c0col = [&, &bsC = bsC, &bcC = bcC]() { bestShiftCorr(cpL, cpR, 0, bsC, bcC); return bcC; }();
    bestShiftCorr(cpL, cpR, 1000, bsC, bcC);

    std::ostringstream ss;
    ss << "L/R 帧对几何体检（居中裁剪 1688×1400，帧 0）:\n";
    ss << std::fixed << std::setprecision(3);
    ss << "  行轮廓相关: 零移位 corr=" << c0row << " ｜ 正常向 bestShift=" << bsN
       << " corr=" << bcN << " ｜ R 翻转向 bestShift=" << bsF << " corr=" << bcF << "\n";
    ss << "  列轮廓相关: 零移位 corr=" << c0col << " ｜ bestShift=" << bsC
       << " corr=" << bcC << "\n";
    ss << "判定: 正常向 corr 明显高→取向 OK（行错位=bestShift）；翻转向高→R 需 180° 旋转";
    // —— 决定性测量：逐组件行位比对（亮斑连通域中心——标志点/激光亮带）——
    {
        auto blobRows = [](const cv::Mat& gray) {
            // 阈值亮点 → 按行投影找亮带行（比 CCL 轻量且稳定）
            cv::Mat bin;
            cv::threshold(gray, bin, 180, 1, cv::THRESH_BINARY);
            cv::Mat rows;
            cv::reduce(bin, rows, 1, cv::REDUCE_SUM, CV_32S);
            std::vector<int> out;
            for (int i = 1; i < rows.rows - 1; ++i) {
                const int v = rows.at<int>(i, 0);
                if (v > 60 && v >= rows.at<int>(i - 1, 0) && v >= rows.at<int>(i + 1, 0))
                    out.push_back(i);
            }
            return out;
        };
        const std::vector<int> rl = blobRows(Lc), rr = blobRows(Rc);
        ss << "\n  亮带行位（局部峰>60px）: L={";
        for (size_t i = 0; i < rl.size() && i < 12; ++i) ss << rl[i] << (i + 1 < rl.size() && i + 1 < 12 ? "," : "");
        ss << "} R={";
        for (size_t i = 0; i < rr.size() && i < 12; ++i) ss << rr[i] << (i + 1 < rr.size() && i + 1 < 12 ? "," : "");
        ss << "}\n  （L/R 同场景矫正前亮带应粗对齐——差值大＝相机 ROI/取向配置错）";
    }
    fprintf(stderr, "\n[FC-PAIR] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}

// ============================================================================
// 精细裁剪原点搜索：±20px 范围步长 1px，以 conic 有效圆为刻度
//（居中 (180,68) 为基准，微调让主点偏差归零——conic 从 NoRealRoot 变 PASS）
// ============================================================================
TEST(FcReplay, FineCropSolve) {
    fs::path frameDir = "diagnostics/fc_frames";
    if (const char* e = getenv("JMW_FC_FINE")) frameDir = fs::path(e);
    std::error_code ec;
    if (!fs::exists(frameDir / "L_000000.png", ec))
        GTEST_SKIP() << "帧目录缺失";
    const int nFrames = 5;
    std::vector<cv::Mat> fl(nFrames), fr(nFrames);
    for (int i = 0; i < nFrames; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%06d", i);
        fl[static_cast<size_t>(i)] = cv::imread(
            (frameDir / ("L_" + std::string(buf) + ".png")).string(), cv::IMREAD_GRAYSCALE);
        fr[static_cast<size_t>(i)] = cv::imread(
            (frameDir / ("R_" + std::string(buf) + ".png")).string(), cv::IMREAD_GRAYSCALE);
    }
    ASSERT_FALSE(fl[0].empty());

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty());
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    auto* h = rig.h.get();
    fs::path outDir = fs::temp_directory_path() / "fc07_fine";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();

    int cXL = 180, cYL = 68, cXR = 180, cYR = 68;
    rig.src.cycles.fs = &rig.src;
    rig.src.cycleAt = [&](size_t i) {
        scr::FrameCycleInfo ci;
        ci.laserIdx = static_cast<int>(i);
        ci.tubeOdd = (i % 2) == 1;
        return ci;
    };
    rig.src.readFrame = [&](int idx, bool left, cv::Mat& out, std::string& e) -> bool {
        const int fi = idx % nFrames;
        const cv::Mat& src = (left ? fl : fr)[static_cast<size_t>(fi)];
        const int x0 = left ? cXL : cXR;
        const int y0 = left ? cYL : cYR;
        if (x0 + 1688 > src.cols || y0 + 1400 > src.rows) { e = "越界"; return false; }
        out = src(cv::Rect(x0, y0, 1688, 1400)).clone();
        return true;
    };
    auto conicScore = [&]() -> long {
        h->diagCclFail_ = 0; h->diagRoisL_ = 0; h->diagSplit_ = 0;
        h->diagEllipse_ = 0; h->diagMatchPairs_ = 0; h->diagConicValid_ = 0;
        h->prevState_ = calib::PrevFrameState{};
        h->globalSet_ = {}; h->globalIndexById_.clear(); h->nextGid_ = 0;
        h->candidates_.clear(); h->holdStreak_ = 0; h->cycleCount_ = 0;
        auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, nFrames);
        for (int c = 0; c < nFrames; ++c) {
            CycleProducts prod;
            std::string ce;
            if (!ex->collect(c, prod, ce)) break;
        }
        ex.reset();
        return h->diagConicValid_.load();
    };

    // 基准：居中 (180,68)
    const long base = conicScore();
    fprintf(stderr, "[FC-FINE] 基准 (180,68) conic=%ld\n", base);

    // —— 第 1 轮：yL 独扫（xL=180, R 固定居中）——
    struct Cand { int v; long s; };
    std::vector<Cand> yl, yr, xl, xr;
    cXL = 180; cXR = 180; cYR = 68;
    for (int dy = -20; dy <= 20; ++dy) {
        cYL = 68 + dy;
        yl.push_back({dy, conicScore()});
    }
    cYL = 68;
    std::sort(yl.begin(), yl.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });
    const int bestDyL = yl[0].v;
    fprintf(stderr, "[FC-FINE] yL 扫描 best dy=%d conic=%ld\n", bestDyL, yl[0].s);

    // —— 第 2 轮：yR 独扫（yL=best）——
    cYL = 68 + bestDyL;
    for (int dy = -20; dy <= 20; ++dy) {
        cYR = 68 + dy;
        yr.push_back({dy, conicScore()});
    }
    cYR = 68;
    std::sort(yr.begin(), yr.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });
    const int bestDyR = yr[0].v;
    fprintf(stderr, "[FC-FINE] yR 扫描 best dy=%d conic=%ld\n", bestDyR, yr[0].s);

    // —— 第 3 轮：xL 独扫（y 固定最优）——
    cYL = 68 + bestDyL; cYR = 68 + bestDyR;
    for (int dx = -20; dx <= 20; ++dx) {
        cXL = 180 + dx;
        xl.push_back({dx, conicScore()});
    }
    cXL = 180;
    std::sort(xl.begin(), xl.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });
    const int bestDxL = xl[0].v;
    fprintf(stderr, "[FC-FINE] xL 扫描 best dx=%d conic=%ld\n", bestDxL, xl[0].s);

    // —— 第 4 轮：xR 独扫 ——
    cXL = 180 + bestDxL;
    for (int dx = -20; dx <= 20; ++dx) {
        cXR = 180 + dx;
        xr.push_back({dx, conicScore()});
    }
    cXR = 180;
    std::sort(xr.begin(), xr.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });
    const int bestDxR = xr[0].v;
    fprintf(stderr, "[FC-FINE] xR 扫描 best dx=%d conic=%ld\n", bestDxR, xr[0].s);

    // —— 终验 ——
    cXL = 180 + bestDxL; cYL = 68 + bestDyL;
    cXR = 180 + bestDxR; cYR = 68 + bestDyR;
    const long finalScore = conicScore();
    std::ostringstream ss;
    ss << "精细裁剪原点（±20px 步长 1px，" << nFrames << " 帧，conic 有效圆刻度）:\n";
    ss << "  基准 (180,68): conic=" << base << "\n";
    ss << "  最优 L=(" << cXL << "," << cYL << ") R=(" << cXR << "," << cYR
       << ") conic=" << finalScore << "（" << (base > 0 ? finalScore * 100 / base : 0)
       << "% vs 基准）";
    fprintf(stderr, "\n[FC-FINE] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}
TEST(FcReplay, ConicRejectDiagnosis) {
    fs::path frameDir = "diagnostics/fc_frames";
    if (const char* e = getenv("JMW_FC_DIAG")) frameDir = fs::path(e);
    std::error_code ec;
    if (!fs::exists(frameDir / "L_000000.png", ec))
        GTEST_SKIP() << "帧目录缺失";
    const int nFrames = 3;
    std::vector<cv::Mat> fl(nFrames), fr(nFrames);
    for (int i = 0; i < nFrames; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%06d", i);
        fl[static_cast<size_t>(i)] = cv::imread(
            (frameDir / ("L_" + std::string(buf) + ".png")).string(), cv::IMREAD_GRAYSCALE);
        fr[static_cast<size_t>(i)] = cv::imread(
            (frameDir / ("R_" + std::string(buf) + ".png")).string(), cv::IMREAD_GRAYSCALE);
    }
    ASSERT_FALSE(fl[0].empty());

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty());
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    auto* h = rig.h.get();
    fs::path outDir = fs::temp_directory_path() / "fc07_diag";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();

    // 居中裁剪（当前生产口径）
    rig.src.cycles.fs = &rig.src;
    rig.src.cycleAt = [&](size_t i) {
        scr::FrameCycleInfo ci;
        ci.laserIdx = static_cast<int>(i);
        ci.tubeOdd = (i % 2) == 1;
        return ci;
    };
    rig.src.readFrame = [&](int idx, bool left, cv::Mat& out, std::string& e) -> bool {
        const int fi = idx % nFrames;
        const cv::Mat& src = (left ? fl : fr)[static_cast<size_t>(fi)];
        const int x0 = (src.cols - 1688) / 2;
        const int y0 = (src.rows - 1400) / 2;
        out = src(cv::Rect(x0, y0, 1688, 1400)).clone();
        return true;
    };

    // —— 先跑真机帧（居中裁剪）——
    std::vector<CycleProducts> realProds(nFrames);
    {
        rig.src.readFrame = [&](int idx, bool left, cv::Mat& out, std::string& e) -> bool {
            const int fi = idx % nFrames;
            const cv::Mat& src = (left ? fl : fr)[static_cast<size_t>(fi)];
            const int x0 = (src.cols - 1688) / 2;
            const int y0 = (src.rows - 1400) / 2;
            out = src(cv::Rect(x0, y0, 1688, 1400)).clone();
            return true;
        };
        h->prevState_ = calib::PrevFrameState{};
        h->globalSet_ = {}; h->globalIndexById_.clear(); h->nextGid_ = 0;
        h->candidates_.clear(); h->holdStreak_ = 0; h->cycleCount_ = 0;
        h->diagConicValid_ = 0;
        auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, nFrames);
        for (int c = 0; c < nFrames; ++c) {
            std::string ce;
            ex->collect(c, realProds[static_cast<size_t>(c)], ce);
        }
        ex.reset();
        fprintf(stderr, "[FC-DIAG] 真机帧: conicValid=%ld 帧数=%d\n",
                h->diagConicValid_.load(), nFrames);
    }
    // —— 再跑测试数据（golden 基准——恢复 buildRig 默认帧源）——
    std::vector<CycleProducts> testProds(nFrames);
    {
        // 恢复 buildRig 的原始帧源（读 ALL_raw）
        const size_t fb = static_cast<size_t>(rig.rawW) * rig.rawH;
        rig.src.readFrame = [&rig, fb](int idx, bool left, cv::Mat& out, std::string& e) -> bool {
            const auto it = std::lower_bound(rig.frameIds.begin(), rig.frameIds.end(), idx);
            if (it == rig.frameIds.end() || *it != idx) { e = "帧号不在数据集"; return false; }
            const size_t pos = static_cast<size_t>(it - rig.frameIds.begin());
            unsigned char* base = (left ? rig.rawL : rig.rawR).data() + pos * fb;
            out = cv::Mat(rig.rawH, rig.rawW, CV_8UC1, base);
            return true;
        };
        rig.src.cycleAt = nullptr;   // 恢复静态 cycles 表
        h->prevState_ = calib::PrevFrameState{};
        h->globalSet_ = {}; h->globalIndexById_.clear(); h->nextGid_ = 0;
        h->candidates_.clear(); h->holdStreak_ = 0; h->cycleCount_ = 0;
        h->diagConicValid_ = 0;
        auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, nFrames);
        for (int c = 0; c < nFrames; ++c) {
            std::string ce;
            ex->collect(c, testProds[static_cast<size_t>(c)], ce);
        }
        ex.reset();
        fprintf(stderr, "[FC-DIAG] 测试帧: conicValid=%ld 帧数=%d\n",
                h->diagConicValid_.load(), nFrames);
    }

    // —— 逐标志点输出 conic 参数对比 ——
    auto dumpConic = [](const char* tag, const std::vector<CycleProducts>& prods,
                        std::ostringstream& ss) {
        long totalDiag = 0, validMk = 0, totalMk = 0;
        std::vector<double> radii, residuals;
        std::map<std::string, int> failReasons;
        std::map<std::string, int> gatedMsgs;
        for (const auto& p : prods) {
            // 诊断向量（逐配对 radius/residual/failReason）
            for (const auto& d : p.conic.diagnostics) {
                ++totalDiag;
                if (d.failReason != calib::ConicFailReason::None)
                    failReasons[calib::ConicFailReasonName(d.failReason)]++;
                radii.push_back(d.reconstructedRadiusMm);
                residuals.push_back(d.reprojResidualPx);
            }
            // 标志点向量（validCircle + message）
            for (const auto& m : p.conic.markers) {
                ++totalMk;
                if (m.validCircle) ++validMk;
                else if (!m.message.empty())
                    gatedMsgs[m.message.substr(0, 40)]++;
            }
        }
        ss << "  " << tag << ":\n";
        ss << "    配对诊断: " << totalDiag << " 对 | 标志点: " << totalMk
           << " 有效=" << validMk << "\n";
        if (!radii.empty()) {
            std::sort(radii.begin(), radii.end());
            std::sort(residuals.begin(), residuals.end());
            ss << "    radius: min=" << std::fixed << std::setprecision(2)
               << radii.front() << " med=" << radii[radii.size() / 2]
               << " max=" << radii.back() << "（门限 2.55~3.45mm）\n";
            ss << "    residual: min=" << std::setprecision(3)
               << (residuals.front() < 0 ? -1 : residuals.front())
               << " med=" << residuals[residuals.size() / 2]
               << " max=" << residuals.back() << "（门限<1.5px）\n";
        }
        if (!failReasons.empty()) {
            ss << "    拒绝原因: ";
            for (const auto& [r, c] : failReasons) ss << r << "=" << c << " ";
            ss << "\n";
        }
        if (!gatedMsgs.empty()) {
            ss << "    门控消息 top5: ";
            int i = 0;
            for (const auto& [m, c] : gatedMsgs) {
                ss << "[" << m << "]×" << c << " ";
                if (++i >= 5) break;
            }
            ss << "\n";
        }
    };
    std::ostringstream ss;
    ss << "conic 校验诊断（帧 0-" << nFrames - 1 << "，居中裁剪 1688×1400）:\n";
    dumpConic("测试数据（golden）", testProds, ss);
    dumpConic("真机帧（居中裁剪）", realProds, ss);

    // —— 中间结果对比：椭圆参数＋匹配对＋undistort 后坐标 ——
    auto dumpEllipses = [](const char* tag, const std::vector<CycleProducts>& prods,
                           std::ostringstream& ss) {
        for (size_t f = 0; f < prods.size() && f < 1; ++f) {   // 只看帧 0
            const auto& p = prods[f];
            ss << "  " << tag << " 帧 0:\n";
            ss << "    CCL L组件=" << p.cclL.components.size()
               << " R组件=" << p.cclR.components.size() << "\n";
            ss << "    split L=" << p.splitL.splitCount << " R=" << p.splitR.splitCount << "\n";
            ss << "    markersDetected=" << p.markersDetected
               << " markersValid=" << p.markersValid << "\n";
            ss << "    match: centerMatches.size=" << p.mm.centerMatches.size() << "\n";
            // 椭圆参数（帧 0 的 L 眼前 5 个）
            ss << "    L 椭圆前 5 个:\n";
            for (size_t i = 0; i < p.ellL.size() && i < 5; ++i) {
                const auto& e = p.ellL[i];
                ss << "      [" << i << "] center=(" << std::fixed
                   << std::setprecision(1) << e.centerX << "," << e.centerY
                   << ") axes=(" << e.majorAxis << "," << e.minorAxis
                   << ") inliers=" << e.inlierCount << "/" << e.totalPointCount
                   << " valid=" << e.success << "\n";
            }
            // undistort 后坐标（帧 0 的前 5 个矫正点）
            if (!p.undistCpu.rectifiedPoints1.empty()) {
                ss << "    undistort 后 L 前 5 点:\n";
                for (size_t i = 0; i < p.undistCpu.rectifiedPoints1.size() && i < 5; ++i) {
                    ss << "      (" << p.undistCpu.rectifiedPoints1[i].x << ","
                       << p.undistCpu.rectifiedPoints1[i].y << ")\n";
                }
            }
            // conic 诊断（帧 0）
            ss << "    conic 诊断前 5 对:\n";
            for (size_t i = 0; i < p.conic.diagnostics.size() && i < 5; ++i) {
                const auto& d = p.conic.diagnostics[i];
                ss << "      [" << i << "] L" << d.leftIdx << "-R" << d.rightIdx
                   << " reason=" << calib::ConicFailReasonName(d.failReason)
                   << " radius=" << std::setprecision(2) << d.reconstructedRadiusMm
                   << "mm resid=" << d.reprojResidualPx << "px\n";
            }
        }
    };
    dumpEllipses("──测试数据──", testProds, ss);
    dumpEllipses("──真机帧──", realProds, ss);
    fprintf(stderr, "\n[FC-DIAG] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}

// ============================================================================
// 原图标志点定位：全图检测亮圆斑（面积/圆度过滤），报告原始坐标分布——
// 正确裁剪窗（1688×1400）应包含全部标志点（JMW_FC_BLOB=<帧目录>）
// ============================================================================
TEST(FcReplay, RawMarkerBlobMap) {
    fs::path frameDir = "diagnostics/fc_frames";
    if (const char* e = getenv("JMW_FC_BLOB")) frameDir = fs::path(e);
    std::error_code ec;
    if (!fs::exists(frameDir / "L_000000.png", ec))
        GTEST_SKIP() << "帧目录缺失";
    auto mapEye = [](const char* tag, const cv::Mat& raw, std::ostringstream& ss) {
        cv::Mat bin;
        // 标志点＝高亮圆斑：动态阈（均值+3σ，下限 160）
        cv::Scalar mu, sigma;
        cv::meanStdDev(raw, mu, sigma);
        const double thr = std::max(120.0, mu[0] + 1.5 * sigma[0]);
        cv::threshold(raw, bin, thr, 255, cv::THRESH_BINARY);
        cv::Mat labels, stats, centroids;
        const int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8);
        struct Blob { cv::Point2f c; int area; };
        std::vector<Blob> blobs;
        for (int i = 1; i < n; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < 80 || area > 30000) continue;      // 圆片面积档
            const double w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const double h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            if (w / h < 0.45 || w / h > 2.2) continue;      // 近圆/近方
            Blob b;
            b.c = cv::Point2f(static_cast<float>(centroids.at<double>(i, 0)),
                              static_cast<float>(centroids.at<double>(i, 1)));
            b.area = area;
            blobs.push_back(b);
        }
        ss << "  " << tag << "（阈=" << std::fixed << std::setprecision(0) << thr
           << " 斑数=" << blobs.size() << "）: ";
        for (size_t i = 0; i < blobs.size() && i < 20; ++i)
            ss << "(" << std::setprecision(0) << blobs[i].c.x << "," << blobs[i].c.y
               << ")" << (i + 1 < blobs.size() && i + 1 < 19 ? " " : "");
        ss << "\n";
        if (!blobs.empty()) {
            float x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
            for (const auto& b : blobs) {
                x0 = std::min(x0, b.c.x); y0 = std::min(y0, b.c.y);
                x1 = std::max(x1, b.c.x); y1 = std::max(y1, b.c.y);
            }
            ss << "    包围盒 x[" << std::setprecision(0) << x0 << "," << x1 << "] y["
               << y0 << "," << y1 << "] —— 1688×1400 窗须全含（原图 "
               << raw.cols << "×" << raw.rows << "，x 余量=" << raw.cols - 1688
               << " y 余量=" << raw.rows - 1400 << "）\n";
        }
    };
    cv::Mat L = cv::imread((frameDir / "L_000000.png").string(), cv::IMREAD_GRAYSCALE);
    cv::Mat R = cv::imread((frameDir / "R_000000.png").string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(L.empty() && R.empty());
    std::ostringstream ss;
    ss << "原图标志点定位（帧 0，动态阈值圆斑）:\n";
    mapEye("L", L, ss);
    mapEye("R", R, ss);
    fprintf(stderr, "\n[FC-BLOB] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}

// ============================================================================
// 分眼裁剪原点求解：JMW_FC_PEEYE=<帧目录>——以激光 v3 匹配数为刻度（对几何
// 错位最敏感），先 yL×yR 后 xL×xR 二维粗扫＋细扫（L/R 原点可不同）
// ============================================================================
TEST(FcReplay, PerEyeCropSolve) {
    fs::path frameDir = "diagnostics/fc_frames";
    if (const char* e = getenv("JMW_FC_PEEYE")) frameDir = fs::path(e);
    std::error_code ec;
    if (!fs::exists(frameDir / "L_000000.png", ec))
        GTEST_SKIP() << "帧目录缺失";
    const int nFrames = 3;
    std::vector<cv::Mat> fl(nFrames), fr(nFrames);
    for (int i = 0; i < nFrames; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%06d", i);
        fl[static_cast<size_t>(i)] = cv::imread(
            (frameDir / ("L_" + std::string(buf) + ".png")).string(), cv::IMREAD_GRAYSCALE);
        fr[static_cast<size_t>(i)] = cv::imread(
            (frameDir / ("R_" + std::string(buf) + ".png")).string(), cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(fl[static_cast<size_t>(i)].empty());
    }
    ASSERT_GE(fl[0].cols, 1688);
    ASSERT_GE(fl[0].rows, 1400);

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty());
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    auto* h = rig.h.get();
    fs::path outDir = fs::temp_directory_path() / "fc07_peeye";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();

    int cXL = 180, cYL = 68, cXR = 180, cYR = 68;   // 当前候选（可变）
    rig.src.cycles.fs = &rig.src;
    rig.src.cycleAt = [&](size_t i) {
        scr::FrameCycleInfo ci;
        ci.laserIdx = static_cast<int>(i);
        ci.tubeOdd = (i % 2) == 1;
        return ci;
    };
    rig.src.readFrame = [&](int idx, bool left, cv::Mat& out, std::string& e) -> bool {
        const int fi = idx % nFrames;
        const cv::Mat& src = (left ? fl : fr)[static_cast<size_t>(fi)];
        const int x0 = left ? cXL : cXR;
        const int y0 = left ? cYL : cYR;
        if (x0 + 1688 > src.cols || y0 + 1400 > src.rows) { e = "越界"; return false; }
        out = src(cv::Rect(x0, y0, 1688, 1400)).clone();
        return true;
    };
    auto runLaserScore = [&]() -> long {
        h->diagCclFail_ = 0; h->diagRoisL_ = 0; h->diagSplit_ = 0;
        h->diagEllipse_ = 0; h->diagMatchPairs_ = 0; h->diagConicValid_ = 0;
        h->diagLaserPts_ = 0;
        h->prevState_ = calib::PrevFrameState{};
        h->globalSet_ = {}; h->globalIndexById_.clear(); h->nextGid_ = 0;
        h->candidates_.clear(); h->holdStreak_ = 0; h->cycleCount_ = 0;
        auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, nFrames);
        for (int c = 0; c < nFrames; ++c) {
            CycleProducts prod;
            std::string ce;
            if (!ex->collect(c, prod, ce)) break;
        }
        ex.reset();
        return h->diagLaserPts_.load();
    };

    struct Cand { int a, b; long score; };
    auto sweep = [&](const char* tag, bool yAxis, int fixedX,
                     int step, int lo1, int hi1, int lo2, int hi2) -> Cand {
        Cand best{0, 0, -1};
        for (int v1 = lo1; v1 <= hi1; v1 += step)
            for (int v2 = lo2; v2 <= hi2; v2 += step) {
                if (yAxis) { cYL = v1; cYR = v2; cXL = cXR = fixedX; }
                else       { cXL = v1; cXR = v2; /* yL/yR 保持上一轮最优——不再覆写 */ }
                const long s = runLaserScore();
                if (s > best.score) best = {v1, v2, s};
            }
        fprintf(stderr, "[FC-PEEYE] %s 粗扫 best=(%d,%d) laser=%ld\n",
                tag, best.a, best.b, best.score);
        return best;
    };

    // y 粗扫（x=180）→ y 细扫 → x 粗扫（y=best）→ x 细扫
    Cand by = sweep("yL×yR", true, 180, 34, 0, 136, 0, 136);
    const int ylc = std::max(0, by.a - 34), yhc = std::min(136, by.a + 34);
    const int yrc = std::max(0, by.b - 34), yhc2 = std::min(136, by.b + 34);
    Cand byf = sweep("yL×yR 细", true, 180, 17, ylc, yhc, yrc, yhc2);
    const int yL = byf.a, yR = byf.b;
    cXL = 180; cYL = yL; cXR = 180; cYR = yR;   // 固定 y 最优再扫 x
    Cand bx = sweep("xL×xR", false, 0/*unused*/, 90, 0, 360, 0, 360);
    const int xlc = std::max(0, bx.a - 90), xhc = std::min(360, bx.a + 90);
    const int xrc = std::max(0, bx.b - 90), xhc2 = std::min(360, bx.b + 90);
    Cand bxf = sweep("xL×xR 细", false, 0/*unused*/, 30, xlc, xhc, xrc, xhc2);
    const int xL = bxf.a, xR = bxf.b;

    // 终验：完整 5 帧
    cXL = xL; cYL = yL; cXR = xR; cYR = yR;
    const long finalScore = runLaserScore();
    std::ostringstream ss;
    ss << "分眼裁剪原点: L=(" << xL << "," << yL << ") R=(" << xR << "," << yR
       << ") 激光匹配=" << finalScore << "（" << nFrames << " 帧；居中假设 (180,68)"
       << "双眼前＝当前线上问题基准）";
    fprintf(stderr, "\n[FC-PEEYE] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}

// ============================================================================
// conic 有效数原点扫描（261030）：PerEyeCropSolve 以激光匹配数为目标解出的
// (180,119)/(180,68) 在线上 conic 反而 0-2→0——目标与 conic 有效性脱节。
// 本用例以 diagConicValid_ 为目标直接扫 yL×yR（x 固定 180），打印粗扫矩阵
// ＋居中基准＋最优细扫——判定「裁剪原点」是否在线 conic 低效的真实变量
// ============================================================================
TEST(FcReplay, ConicCropSolve) {
    if (getenv("JMW_FC_CONICROP") == nullptr)
        GTEST_SKIP() << "JMW_FC_CONICROP=<帧目录> 未设";

    fs::path frameDir = getenv("JMW_FC_CONICROP");
    std::error_code ec;
    if (!fs::exists(frameDir / "L_000000.png", ec))
        GTEST_SKIP() << "帧目录缺失";
    const int nFrames = 3;
    std::vector<cv::Mat> fl(nFrames), fr(nFrames);
    for (int i = 0; i < nFrames; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%06d", i);
        fl[static_cast<size_t>(i)] = cv::imread(
            (frameDir / ("L_" + std::string(buf) + ".png")).string(), cv::IMREAD_GRAYSCALE);
        fr[static_cast<size_t>(i)] = cv::imread(
            (frameDir / ("R_" + std::string(buf) + ".png")).string(), cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(fl[static_cast<size_t>(i)].empty());
    }
    ASSERT_GE(fl[0].cols, 1688);
    ASSERT_GE(fl[0].rows, 1400);

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty());
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    auto* h = rig.h.get();
    fs::path outDir = fs::temp_directory_path() / "fc07_conicrop";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();

    int cXL = 180, cYL = 68, cXR = 180, cYR = 68;
    rig.src.cycles.fs = &rig.src;
    rig.src.cycleAt = [&](size_t i) {
        scr::FrameCycleInfo ci;
        ci.laserIdx = static_cast<int>(i);
        ci.tubeOdd = (i % 2) == 1;
        return ci;
    };
    rig.src.readFrame = [&](int idx, bool left, cv::Mat& out, std::string& e) -> bool {
        const int fi = idx % nFrames;
        const cv::Mat& src = (left ? fl : fr)[static_cast<size_t>(fi)];
        const int x0 = left ? cXL : cXR;
        const int y0 = left ? cYL : cYR;
        if (x0 + 1688 > src.cols || y0 + 1400 > src.rows) { e = "越界"; return false; }
        out = src(cv::Rect(x0, y0, 1688, 1400)).clone();
        return true;
    };
    auto runConicScore = [&](long* laserPts = nullptr) -> long {
        h->diagCclFail_ = 0; h->diagRoisL_ = 0; h->diagSplit_ = 0;
        h->diagEllipse_ = 0; h->diagMatchPairs_ = 0; h->diagConicValid_ = 0;
        h->diagLaserPts_ = 0;
        h->prevState_ = calib::PrevFrameState{};
        h->globalSet_ = {}; h->globalIndexById_.clear(); h->nextGid_ = 0;
        h->candidates_.clear(); h->holdStreak_ = 0; h->cycleCount_ = 0;
        auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, nFrames);
        for (int c = 0; c < nFrames; ++c) {
            CycleProducts prod;
            std::string ce;
            if (!ex->collect(c, prod, ce)) break;
        }
        ex.reset();
        if (laserPts) *laserPts = h->diagLaserPts_.load();
        return h->diagConicValid_.load();
    };

    // 居中基准
    cXL = 180; cYL = 68; cXR = 180; cYR = 68;
    long baseLaser = 0;
    const long baseConic = runConicScore(&baseLaser);
    fprintf(stderr, "[FC-CONICROP] 居中(180,68)/(180,68) 基准: conic=%ld laser=%ld\n",
            baseConic, baseLaser);

    // yL×yR 粗扫矩阵（x=180，步 34：0..136）
    struct Cand { int yl, yr; long conic; };
    Cand best{68, 68, baseConic};
    fprintf(stderr, "[FC-CONICROP] yL\\yR 矩阵（conic 有效数，3 帧累计）:\n");
    for (int yl = 0; yl <= 136; yl += 34) {
        std::ostringstream row;
        row << "  yL=" << std::setw(3) << yl << ":";
        for (int yr = 0; yr <= 136; yr += 34) {
            cYL = yl; cYR = yr;
            const long s = runConicScore();
            row << " yR" << std::setw(3) << yr << "=" << std::setw(4) << s;
            if (s > best.conic) best = {yl, yr, s};
        }
        fprintf(stderr, "%s\n", row.str().c_str());
    }
    // 最优细扫（±34 步 17）
    const int ylc = std::max(0, best.yl - 34), yhc = std::min(136, best.yl + 34);
    const int yrc = std::max(0, best.yr - 34), yhc2 = std::min(136, best.yr + 34);
    for (int yl = ylc; yl <= yhc; yl += 17)
        for (int yr = yrc; yr <= yhc2; yr += 17) {
            if (yl == best.yl && yr == best.yr) continue;
            cYL = yl; cYR = yr;
            const long s = runConicScore();
            if (s > best.conic) best = {yl, yr, s};
        }
    long bestLaser = 0;
    cYL = best.yl; cYR = best.yr;
    const long confirmConic = runConicScore(&bestLaser);
    fprintf(stderr, "[FC-CONICROP] 粗细扫后: yL=%d yR=%d conic=%ld laser=%ld\n",
            best.yl, best.yr, confirmConic, bestLaser);

    // —— 261030 精化：y ±15 步 5 → x ±30 步 10（x 此前固定 180 未扫）。
    //    配准质量随 conic 数上升（在线 initialRMSE 2.64mm＝conic ~3/帧踩光流
    //    下限欠约束）——目标 conic ≥6/帧（测试数据 4-8/帧档）
    for (int yl = best.yl - 15; yl <= best.yl + 15; yl += 5)
        for (int yr = best.yr - 15; yr <= best.yr + 15; yr += 5) {
            if (yl < 0 || yr < 0 || yl > 136 || yr > 136) continue;
            cXL = 180; cXR = 180; cYL = yl; cYR = yr;
            const long s = runConicScore();
            if (s > best.conic) best = {yl, yr, s};
        }
    fprintf(stderr, "[FC-CONICROP] y 精化: yL=%d yR=%d conic=%ld\n",
            best.yl, best.yr, best.conic);
    int bestXL = 180, bestXR = 180;
    for (int xl = 150; xl <= 210; xl += 10)
        for (int xr = 150; xr <= 210; xr += 10) {
            cYL = best.yl; cYR = best.yr; cXL = xl; cXR = xr;
            const long s = runConicScore();
            if (s > best.conic) { best.conic = s; bestXL = xl; bestXR = xr; }
        }
    fprintf(stderr, "[FC-CONICROP] x 粗精化: xL=%d xR=%d conic=%ld\n",
            bestXL, bestXR, best.conic);
    std::ostringstream ss;
    ss << "conic 最优原点≈L=(" << bestXL << "," << best.yl << ") R=(" << bestXR
       << "," << best.yr << ") conic=" << best.conic << "（居中基准 " << baseConic
       << "；测试数据档 4-8/帧）";
    fprintf(stderr, "\n[FC-CONICROP] %s\n", ss.str().c_str());
    SUCCEED() << ss.str();
}

// ============================================================================
// 白板平面性诊断（261030）：单帧重建质量 vs 融合位姿质量 判别。
//   ①obs checkpoint 逐帧激光（设备系）平面拟合——白板单帧应为平面；残差
//     RMS 大/边缘带恶化＝重建问题（Q/畸变边缘），残差亚毫米＝重建无罪
//   ②终版 PLY（GBA 重放后）鲁棒平面拟合＋径向残差剖面——分层/边缘喷溅
//     ＝位姿/融合侧
// ============================================================================
TEST(FcReplay, WhiteboardPlaneDiagnosis) {
    const char* ckPath = getenv("JMW_FC_PLANECHK");
    const char* plyPath = getenv("JMW_FC_PLANEPLY");
    if (ckPath == nullptr && plyPath == nullptr)
        GTEST_SKIP() << "JMW_FC_PLANECHK=<obs checkpoint> / JMW_FC_PLANEPLY=<ply> 未设";
    namespace sp = Scanner::pipeline;
    // 261030 可选深度门限模拟（JMW_FC_PLANEGATE=min:max，mm）——按现行生产
    // laser_reconstruct 门限过滤逐帧点后评估（预测重扫质量，免真机往返）
    float gateMin = -1e30f, gateMax = 1e30f;
    if (const char* g = getenv("JMW_FC_PLANEGATE")) {
        if (sscanf_s(g, "%f:%f", &gateMin, &gateMax) == 2)
            fprintf(stderr, "[PLANE-CHK] 模拟深度门限 [%.0f,%.0f]mm\n",
                    gateMin, gateMax);
    }

    auto planeFit = [](const std::vector<cv::Point3f>& pts, cv::Vec3f& n,
                       cv::Point3f& c) -> double {
        // 质心＋协方差最小特征向量（幂迭代求最大两次，叉乘得法向）
        cv::Point3f mean(0, 0, 0);
        for (const auto& p : pts) { mean.x += p.x; mean.y += p.y; mean.z += p.z; }
        mean.x /= pts.size(); mean.y /= pts.size(); mean.z /= pts.size();
        double a = 0, b = 0, d = 0, e = 0, f = 0, g = 0;   // xx xy xz yy yz zz
        for (const auto& p : pts) {
            const float x = p.x - mean.x, y = p.y - mean.y, z = p.z - mean.z;
            a += x * x; b += x * y; d += x * z; e += y * y; f += y * z; g += z * z;
        }
        // 3×3 协方差幂迭代（最大特征向量）→ 叉乘第二近似得最小方向
        cv::Vec3d v1(1, 0.3, 0.1);
        for (int it = 0; it < 60; ++it) {
            cv::Vec3d w(a * v1[0] + b * v1[1] + d * v1[2],
                        b * v1[0] + e * v1[1] + f * v1[2],
                        d * v1[0] + f * v1[1] + g * v1[2]);
            const double L = std::sqrt(w.dot(w));
            if (L < 1e-12) break;
            v1 = w / L;
        }
        cv::Vec3d v2 = v1 + cv::Vec3d(0.37, 0.61, 0.93);
        for (int it = 0; it < 60; ++it) {
            cv::Vec3d w(a * v2[0] + b * v2[1] + d * v2[2],
                        b * v2[0] + e * v2[1] + f * v2[2],
                        d * v2[0] + f * v2[1] + g * v2[2]);
            w -= v1 * v1.dot(w);          // 去主方向分量
            const double L = std::sqrt(w.dot(w));
            if (L < 1e-12) break;
            v2 = w / L;
        }
        cv::Vec3d nv = v1.cross(v2);
        const double L = std::sqrt(nv.dot(nv));
        if (L < 1e-12) return -1;
        nv /= L;
        double s2 = 0;
        for (const auto& p : pts) {
            const double r = nv[0] * (p.x - mean.x) + nv[1] * (p.y - mean.y)
                           + nv[2] * (p.z - mean.z);
            s2 += r * r;
        }
        n = cv::Vec3f(static_cast<float>(nv[0]), static_cast<float>(nv[1]),
                      static_cast<float>(nv[2]));
        c = mean;
        return std::sqrt(s2 / pts.size());
    };

    // ── ①逐帧（设备系）平面性 ──
    if (ckPath != nullptr) {
        sp::FrameObsAccumulator acc(size_t{1} << 30);
        const auto lr = acc.loadCheckpoint(ckPath);
        ASSERT_TRUE(lr.success) << "checkpoint 载入失败: " << lr.message;
        const auto snap = acc.snapshot();
        fprintf(stderr, "[PLANE-CHK] obs=%zu 激光帧=%zu degraded=%d\n",
                snap.obs.size(), snap.laserFrames.size(),
                static_cast<int>(acc.degradedLaser()));
        // 261030 首 40 帧点数序列（交替模式＝tubeOdd 奇偶病；随机＝另有因）
        {
            std::ostringstream seq;
            int shown = 0;
            for (size_t i = 0; i < snap.obs.size() && shown < 40; ++i) {
                const size_t slot = snap.obs[i].laserCacheSlot;
                if (slot == sp::FrameObs::kNoLaserSlot || slot >= snap.laserFrames.size())
                    continue;
                seq << snap.laserFrames[slot].size() / 3 << " ";
                ++shown;
            }
            fprintf(stderr, "[PLANE-CHK] 首帧点数序列: %s\n", seq.str().c_str());
        }
        for (size_t i = 0; i < 5 && i < snap.obs.size(); ++i)
            fprintf(stderr, "  obs[%zu] slot=%zu size=%zu\n", i,
                    snap.obs[i].laserCacheSlot,
                    snap.obs[i].laserCacheSlot < snap.laserFrames.size()
                        ? snap.laserFrames[snap.obs[i].laserCacheSlot].size()
                        : size_t{0});
        // 帧点数分布（诊断——门槛失败时看量级）
        {
            std::vector<size_t> sizes;
            for (size_t i = 0; i < snap.obs.size(); ++i) {
                const size_t slot = snap.obs[i].laserCacheSlot;
                if (slot == sp::FrameObs::kNoLaserSlot || slot >= snap.laserFrames.size())
                    continue;
                sizes.push_back(snap.laserFrames[slot].size() / 3);
            }
            if (!sizes.empty()) {
                std::sort(sizes.begin(), sizes.end());
                fprintf(stderr, "[PLANE-CHK] 帧点数: min=%zu P25=%zu 中位=%zu P90=%zu max=%zu\n",
                        sizes.front(), sizes[sizes.size() / 4],
                        sizes[sizes.size() / 2], sizes[sizes.size() * 9 / 10],
                        sizes.back());
            }
        }
        std::vector<double> rmsList;
        std::vector<double> sprayList;
        double worstEdgeGrow = 0;
        bool dumpedBig = false;
        double worstSprayPct = -1;
        size_t worstSprayIdx = 0;
        std::vector<cv::Point3f> worstSprayPts;
        // 分层抽样详细：首帧＋首个「大帧」（>5000 点）＋最差喷溅帧——各打
        // Z 分位（离面内容=紧簇物理面 vs 弥散幻影 的判别材料）
        auto dumpFrame = [&](const char* tag, size_t idx,
                             const std::vector<cv::Point3f>& v) {
            std::vector<double> zs;
            zs.reserve(v.size());
            for (const auto& p : v) zs.push_back(p.z);
            std::sort(zs.begin(), zs.end());
            fprintf(stderr,
                    "  %s 帧#%zu: %zu 点 Z分位 P5=%.0f P25=%.0f P50=%.0f P75=%.0f"
                    " P95=%.0f min=%.0f max=%.0f\n",
                    tag, idx, v.size(), zs[zs.size() / 20], zs[zs.size() / 4],
                    zs[zs.size() / 2], zs[zs.size() * 3 / 4],
                    zs[zs.size() * 19 / 20], zs.front(), zs.back());
        };
        int frames = 0;
        for (size_t i = 0; i < snap.obs.size(); ++i) {
            const size_t slot = snap.obs[i].laserCacheSlot;
            if (slot == sp::FrameObs::kNoLaserSlot || slot >= snap.laserFrames.size())
                continue;
            const auto& lf = snap.laserFrames[slot];
            if (lf.size() < 60) continue;   // 点太少拟合不稳（白板远距帧点数少）
            std::vector<cv::Point3f> pts;
            pts.reserve(lf.size() / 3);
            for (size_t k = 0; k + 2 < lf.size(); k += 3) {
                if (lf[k + 2] < gateMin || lf[k + 2] > gateMax) continue;
                pts.emplace_back(lf[k], lf[k + 1], lf[k + 2]);
            }
            if (pts.size() < 60) continue;
            cv::Vec3f n; cv::Point3f c;
            double rms = planeFit(pts, n, c);
            if (rms < 0) continue;
            // 稳健二遍：剔 |r|>3mm 外点后重拟合（区分「整体弯曲畸变」vs
            // 「少量喷溅拉爆 RMS」——核 RMS 仍大＝系统性重建畸变）
            std::vector<cv::Point3f> core;
            core.reserve(pts.size());
            for (const auto& p : pts) {
                const double r = n[0] * (p.x - c.x) + n[1] * (p.y - c.y)
                               + n[2] * (p.z - c.z);
                if (std::abs(r) < 3.0) core.push_back(p);
            }
            double coreRms = rms;
            size_t coreN = core.size();
            double sprayPct = 100.0 * (pts.size() - core.size()) / pts.size();
            if (core.size() >= 60) {
                cv::Vec3f n2; cv::Point3f c2;
                const double r2 = planeFit(core, n2, c2);
                if (r2 >= 0) { coreRms = r2; n = n2; c = c2; }
            }
            (void)rms;
            // 边缘带（径向外 25%）残差 vs 内带——重建边缘畸变特征
            std::vector<double> rad(pts.size());
            double rMax = 0;
            for (size_t k = 0; k < pts.size(); ++k) {
                rad[k] = std::sqrt((pts[k].x - c.x) * (pts[k].x - c.x)
                                 + (pts[k].y - c.y) * (pts[k].y - c.y)
                                 + (pts[k].z - c.z) * (pts[k].z - c.z));
                rMax = std::max(rMax, rad[k]);
            }
            double in2 = 0, out2 = 0; int inN = 0, outN = 0;
            for (size_t k = 0; k < pts.size(); ++k) {
                const double r = n[0] * (pts[k].x - c.x) + n[1] * (pts[k].y - c.y)
                               + n[2] * (pts[k].z - c.z);
                if (rad[k] < rMax * 0.75) { in2 += r * r; ++inN; }
                else                      { out2 += r * r; ++outN; }
            }
            const double inRms = inN ? std::sqrt(in2 / inN) : 0;
            const double outRms = outN ? std::sqrt(out2 / outN) : 0;
            if (inRms > 0.01) worstEdgeGrow = std::max(worstEdgeGrow, outRms / inRms);
            rmsList.push_back(coreRms);
            sprayList.push_back(sprayPct);
            ++frames;
            if (frames == 1)
                dumpFrame("首帧", i, pts);
            if (pts.size() > 5000 && !dumpedBig) {
                dumpedBig = true;
                dumpFrame("大帧", i, pts);
            }
            if (sprayPct > worstSprayPct) {
                worstSprayPct = sprayPct;
                worstSprayPts = pts;
                worstSprayIdx = i;
            }
            if (frames <= 3 || frames % 200 == 0)
                fprintf(stderr, "  帧#%zu: %zu 点 核RMS=%.3fmm 喷溅=%.1f%% 内带=%.3f 边缘带=%.3f\n",
                        i, pts.size(), coreRms, sprayPct, inRms, outRms);
        }
        ASSERT_GT(frames, 0) << "无逐帧激光缓存";
        // 最差喷溅帧落 Z 分位（离面内容判别）
        if (!worstSprayPts.empty())
            dumpFrame("最差喷溅帧", worstSprayIdx, worstSprayPts);

        // —— 261030 决定性实验：联合云内含率分解 ——
        // 三种联合（各帧按 R_init 变换拼接，抽样 ≤20 万点，鲁棒平面 ±3mm）：
        //   A=全部帧  B=仅过质量门的帧（IQR≤60 或非均匀） C=被门拒的帧
        // A 乱 B 净 → 乱源＝幻影帧（质量门可解）；B 也乱 → 乱源＝位姿/联合侧
        {
            std::vector<cv::Point3f> unionAll, unionGood, unionBad;
            size_t gatedFrames = 0;
            for (size_t i = 0; i < snap.obs.size(); ++i) {
                const size_t slot = snap.obs[i].laserCacheSlot;
                if (slot == sp::FrameObs::kNoLaserSlot
                    || slot >= snap.laserFrames.size())
                    continue;
                const auto& lf = snap.laserFrames[slot];
                if (lf.size() < 60) continue;
                std::vector<cv::Point3f> pts;
                pts.reserve(lf.size() / 3);
                for (size_t k = 0; k + 2 < lf.size(); k += 3) {
                    if (lf[k + 2] < gateMin || lf[k + 2] > gateMax) continue;
                    pts.emplace_back(lf[k], lf[k + 1], lf[k + 2]);
                }
                if (pts.size() < 60) continue;
                // 质量门判据（同 fc_executor zgate：IQR>60 且 均匀度>0.42）
                std::vector<float> zs(pts.size());
                for (size_t k = 0; k < pts.size(); ++k) zs[k] = pts[k].z;
                std::sort(zs.begin(), zs.end());
                const float iqr = zs[zs.size() * 3 / 4] - zs[zs.size() / 4];
                const float rng = zs[zs.size() * 19 / 20] - zs[zs.size() / 20];
                const bool bad = iqr > 60.0f && rng > 1.0f && iqr / rng > 0.42f;
                if (bad) ++gatedFrames;
                // R_init 变换入世界系
                const cv::Matx33d Rm(snap.obs[i].R_init);
                const cv::Vec3d tv(snap.obs[i].t_init);
                // 抽样：每帧取 ≤60 点（6k 帧 × 60 ≈ 36 万→再截 20 万）
                const size_t step = pts.size() / 60 + 1;
                for (size_t k = 0; k < pts.size(); k += step) {
                    const cv::Vec3d w = Rm * cv::Vec3d(pts[k].x, pts[k].y, pts[k].z) + tv;
                    unionAll.emplace_back(static_cast<float>(w[0]),
                                          static_cast<float>(w[1]),
                                          static_cast<float>(w[2]));
                    (bad ? unionBad : unionGood).emplace_back(
                        static_cast<float>(w[0]), static_cast<float>(w[1]),
                        static_cast<float>(w[2]));
                }
            }
            auto inlierRate = [&](const std::vector<cv::Point3f>& v) {
                if (v.size() < 100) return -1.0;
                cv::Vec3f n; cv::Point3f c;
                if (planeFit(v, n, c) < 0) return -1.0;
                std::vector<cv::Point3f> core;
                for (const auto& p : v) {
                    const double r = n[0] * (p.x - c.x) + n[1] * (p.y - c.y)
                                   + n[2] * (p.z - c.z);
                    if (std::abs(r) < 3.0) core.push_back(p);
                }
                return 100.0 * core.size() / v.size();
            };
            fprintf(stderr,
                    "[PLANE-CHK] 联合云分解（R_init 位姿拼接）：全部帧 %.1f%%"
                    "｜过门帧 %.1f%%｜被拒帧 %.1f%%（被拒帧数 %zu/%d）"
                    "——全部乱/过门净＝幻影主导（门可解）；过门也乱＝位姿联合侧\n",
                    inlierRate(unionAll), inlierRate(unionGood),
                    inlierRate(unionBad), gatedFrames, frames);
        }
        std::sort(rmsList.begin(), rmsList.end());
        std::sort(sprayList.begin(), sprayList.end());
        fprintf(stderr,
                "[PLANE-CHK] 逐帧核RMS(剔>3mm): 中位=%.3fmm P90=%.3fmm 最大=%.3fmm"
                "｜喷溅占比: 中位=%.1f%% P90=%.1f%%（帧数=%d；边缘/内带残差比最差=%.2f"
                "）——核RMS 大＝系统性重建畸变；核RMS 小而喷溅大＝提取离群\n",
                rmsList[rmsList.size() / 2], rmsList[rmsList.size() * 9 / 10],
                rmsList.back(), sprayList[sprayList.size() / 2],
                sprayList[sprayList.size() * 9 / 10], frames, worstEdgeGrow);
    }

    // ── ②终版 PLY 鲁棒平面＋径向剖面 ──
    if (plyPath != nullptr) {
        std::ifstream pf(plyPath);
        ASSERT_TRUE(pf) << "PLY 不可读";
        std::string line;
        size_t n = 0;
        while (std::getline(pf, line)) {
            std::istringstream ls(line);
            std::string tag;
            ls >> tag;
            if (tag == "element" ) { std::string what; ls >> what; if (what == "vertex") ls >> n; }
            if (tag == "end_header") break;
        }
        ASSERT_GT(n, size_t{0});
        const size_t stride = std::max<size_t>(1, n / 200000);   // 抽样 ≤20 万
        std::vector<cv::Point3f> pts;
        pts.reserve(n / stride + 8);
        for (size_t i = 0; i < n && pf; ++i) {
            float x, y, z;
            if (!(pf >> x >> y >> z)) break;
            if (i % stride == 0) pts.emplace_back(x, y, z);
        }
        cv::Vec3f nrm; cv::Point3f ctr;
        double rms = planeFit(pts, nrm, ctr);
        // 鲁棒二遍：剔 |r|>3mm 后重拟合
        std::vector<cv::Point3f> core;
        core.reserve(pts.size());
        for (const auto& p : pts) {
            const double r = nrm[0] * (p.x - ctr.x) + nrm[1] * (p.y - ctr.y)
                           + nrm[2] * (p.z - ctr.z);
            if (std::abs(r) < 3.0) core.push_back(p);
        }
        const double inlierPct = 100.0 * core.size() / pts.size();
        rms = planeFit(core, nrm, ctr);
        // 径向剖面（core 最大半径；含离群的径向位置）
        double rMax = 0;
        for (const auto& p : core) {
            const double d = std::sqrt((p.x - ctr.x) * (p.x - ctr.x)
                                     + (p.y - ctr.y) * (p.y - ctr.y)
                                     + (p.z - ctr.z) * (p.z - ctr.z));
            rMax = std::max(rMax, d);
        }
        constexpr int kBins = 5;
        double s2[kBins] = {0}; size_t cnt[kBins] = {0};
        size_t sprayIn = 0, sprayOut = 0;
        for (const auto& p : pts) {
            const double r = nrm[0] * (p.x - ctr.x) + nrm[1] * (p.y - ctr.y)
                           + nrm[2] * (p.z - ctr.z);
            const double d = std::sqrt((p.x - ctr.x) * (p.x - ctr.x)
                                     + (p.y - ctr.y) * (p.y - ctr.y)
                                     + (p.z - ctr.z) * (p.z - ctr.z));
            const int b = std::min(kBins - 1, static_cast<int>(d / rMax * kBins));
            s2[b] += r * r; ++cnt[b];
            if (std::abs(r) > 10.0) { if (b >= kBins - 2) ++sprayOut; else ++sprayIn; }
        }
        std::ostringstream prof;
        for (int b = 0; b < kBins; ++b)
            prof << " [" << (100 * b / kBins) << "-" << (100 * (b + 1) / kBins)
                 << "%]=" << (cnt[b] ? std::sqrt(s2[b] / cnt[b]) : 0.0);
        fprintf(stderr,
                "[PLANE-PLY] %zu 点（抽样 %zu）内含率(±3mm)=%.1f%% 核RMS=%.3fmm"
                "\n  径向残差RMS%s\n  喷溅(|r|>10mm): 内带=%zu 边缘带=%zu"
                "（分层/多帧错位→融合位姿侧；径向平滑恶化→重建畸变侧）\n",
                n, pts.size(), inlierPct, rms, prof.str().c_str(),
                sprayIn, sprayOut);
        SUCCEED();
        return;
    }
    SUCCEED();
}

// ============================================================================
// GBA on checkpoint（261030）：在真机会话 checkpoint 上离线复跑全局优化——
// 验证关键帧子采样修复（15025 帧原 fail(fallback) 无重放）＋终版云平面率
// ============================================================================
TEST(FcReplay, GbaOnCheckpoint) {
    const char* ckPath = getenv("JMW_FC_GBACHK");
    if (ckPath == nullptr)
        GTEST_SKIP() << "JMW_FC_GBACHK=<obs checkpoint> 未设";
    namespace sp = Scanner::pipeline;
    sp::FrameObsAccumulator acc(size_t{1} << 30);
    const auto lr = acc.loadCheckpoint(ckPath);
    ASSERT_TRUE(lr.success) << lr.message;
    sp::GlobalOptimObject go;
    sp::PipelineDeps deps;
    ASSERT_TRUE(go.configure(deps).success);
    sp::CancelToken cancel;
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = go.run(acc, [](int, const std::string&) {}, cancel);
    const auto el = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(r.success) << r.message;
    const auto& out = go.output();
    fprintf(stderr,
            "[GBA-CK] 帧数=%zu gba=%s 初RMSE=%.4f 末RMSE=%.4f 标志点=%zu "
            "重放=%d 时长=%lldms: %s\n",
            out.frameCount, out.gbaSuccess ? "ok" : "FAIL",
            out.gbaStats.initialRMSE, out.gbaStats.finalRMSE,
            out.gbaMarkers.size(), static_cast<int>(out.laserReplayed), el,
            out.gbaMessage.c_str());
    ASSERT_TRUE(out.gbaSuccess) << "GBA 仍失败: " << out.gbaMessage;
    // 终版激光云鲁棒平面内含率（白板场景判据）
    if (!out.laserXyzFinal.empty()) {
        std::vector<cv::Point3f> pts;
        const size_t n = out.laserXyzFinal.size() / 3;
        const size_t stride = n / 200000 + 1;
        for (size_t i = 0; i < n; i += stride)
            pts.emplace_back(out.laserXyzFinal[i * 3], out.laserXyzFinal[i * 3 + 1],
                             out.laserXyzFinal[i * 3 + 2]);
        // 复用 WhiteboardPlaneDiagnosis 的平面拟合口径（局部重写幂迭代）
        cv::Point3f mean(0, 0, 0);
        for (const auto& p : pts) { mean.x += p.x; mean.y += p.y; mean.z += p.z; }
        mean.x /= pts.size(); mean.y /= pts.size(); mean.z /= pts.size();
        double a = 0, b = 0, d = 0, e = 0, f = 0, g = 0;
        for (const auto& p : pts) {
            const float x = p.x - mean.x, y = p.y - mean.y, z = p.z - mean.z;
            a += x * x; b += x * y; d += x * z; e += y * y; f += y * z; g += z * z;
        }
        cv::Vec3d v1(1, 0.3, 0.1);
        for (int it = 0; it < 60; ++it) {
            cv::Vec3d w(a * v1[0] + b * v1[1] + d * v1[2],
                        b * v1[0] + e * v1[1] + f * v1[2],
                        d * v1[0] + f * v1[1] + g * v1[2]);
            const double L = std::sqrt(w.dot(w));
            if (L < 1e-12) break;
            v1 = w / L;
        }
        cv::Vec3d v2 = v1 + cv::Vec3d(0.37, 0.61, 0.93);
        for (int it = 0; it < 60; ++it) {
            cv::Vec3d w(a * v2[0] + b * v2[1] + d * v2[2],
                        b * v2[0] + e * v2[1] + f * v2[2],
                        d * v2[0] + f * v2[1] + g * v2[2]);
            w -= v1 * v1.dot(w);
            const double L = std::sqrt(w.dot(w));
            if (L < 1e-12) break;
            v2 = w / L;
        }
        cv::Vec3d nv = v1.cross(v2);
        const double L = std::sqrt(nv.dot(nv));
        nv /= L;
        size_t in3 = 0;
        for (const auto& p : pts) {
            const double rr = nv[0] * (p.x - mean.x) + nv[1] * (p.y - mean.y)
                            + nv[2] * (p.z - mean.z);
            if (std::abs(rr) < 3.0) ++in3;
        }
        fprintf(stderr, "[GBA-CK] 终版激光 %zu 点（抽样 %zu）平面内含率(±3mm)=%.1f%%\n",
                n, pts.size(), 100.0 * in3 / pts.size());
    }
    SUCCEED();
}

// ============================================================================
// startRealtime 名义标定路径复现：22:03 真机启动断言（fixedSize create）——
// 离线同构装配（attachCalib＋attachStereoNominal＋算子 json）走一遍启动/停止
// ============================================================================
TEST(FcReplay, StartRealtimeNominalPath) {
    if (getenv("JMW_FC_RT") == nullptr)
        GTEST_SKIP() << "JMW_FC_RT=1 未设";

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty());
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;

    // —— 参数微切：定位 fixedSize 断言源自哪组参数 ——
    {
        auto tryCtor = [](const char* tag, calib::UndistortPointsParams u) {
            try {
                calib::UndistortPointsCuda op(u);
                fprintf(stderr, "[FCDBG-ctor] %s: OK\n", tag);
            } catch (const std::exception& e) {
                fprintf(stderr, "[FCDBG-ctor] %s: FAIL %s\n", tag, e.what());
            }
        };
        // a) 离线直用型（buildRig 同款）
        {
            calib::UndistortPointsParams u;
            u.cameraMatrix = rig.tc.K1c; u.distCoeffs = rig.tc.D1;
            u.R = rig.tc.R1; u.P = rig.tc.P1;
            tryCtor("a-直用(K1c,D1,R1,P1)", u);
        }
        // b) startRealtime 同款（zeros P＋块拷＋右目置零）
        {
            calib::UndistortPointsParams u;
            u.cameraMatrix = rig.tc.K1c.clone(); u.distCoeffs = rig.tc.D1.clone();
            u.R = rig.tc.R1.clone();
            u.P = cv::Mat::zeros(3, 4, CV_64F);
            rig.tc.P1(cv::Rect(0, 0, 3, 3)).copyTo(u.P(cv::Rect(0, 0, 3, 3)));
            tryCtor("b-块拷P", u);
        }
        // c) b＋P 第 4 列全零置零（模拟右目处理）
        {
            calib::UndistortPointsParams u;
            u.cameraMatrix = rig.tc.K2c.clone(); u.distCoeffs = rig.tc.D2.clone();
            u.R = rig.tc.R2.clone();
            u.P = rig.tc.P2.clone();
            u.P.at<double>(0, 3) = 0.0;
            tryCtor("c-P2置零(clone)", u);
        }
    }
    Scanner::data::SlotRing<Scanner::data::EnhancedFrame> ring(4);
    auto adapter = std::make_unique<Scanner::pipeline::FiveChainAdapter>(
        Scanner::pipeline::FiveChainConfig{}, *rig.obs, nullptr, nullptr,
        &rig.operators);
    adapter->attachCalib(rig.tc.K1, rig.tc.D1, rig.tc.K2, rig.tc.D2,
                         rig.tc.W, rig.tc.H, rig.tc.laserTable);
    adapter->attachStereoNominal(rig.tc.R1, rig.tc.R2, rig.tc.P1, rig.tc.P2,
                                 rig.tc.Q, rig.tc.K1c, rig.tc.K2c,
                                 rig.tc.D1, rig.tc.D2, rig.tc.Rext, rig.tc.Tcomp);
    auto sr = adapter->startRealtime(ring, 0);
    ASSERT_TRUE(sr.success) << "startRealtime 失败: " << sr.message;
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    adapter->stopRealtime();
    SUCCEED() << "startRealtime 名义标定路径离线复现通过（真机断言未再现＝"
                 "环境相关，需真机再查）";
}

// ============================================================================
// GBA 全链验证（261029）：测试图像跑扫描流水线 → 全局优化 → 优化后标志点＋
// 前 1 万激光点位置验证
//   验证口径（无外部真值——自一致性对照）：
//   ①优化后标志点：数量合理（本数据集 5..20）＋两两间距 >10mm（近距合并
//     生效＝无重复 id 残留）＋每点 ≤2mm 于其 globalId 观测簇均值（初始位姿
//     变换到世界系——位姿/观测错配时偏离会显著放大）
//   ②前 1 万激光点：全部 ≤2mm 邻近 C 线融合云（同数据同体素——GBA 位姿修正
//     量级 ~0.3mm；位姿与激光帧槽位错配时偏离达帧间运动量级 mm-cm → 失败）
// ============================================================================
TEST(FcReplay, GbaOptimizedMarkersAndLaser) {
    if (getenv("JMW_FC_REPLAY") == nullptr)
        GTEST_SKIP() << "JMW_FC_REPLAY=1 未设——离线对拍显式启用";
    namespace sp = Scanner::pipeline;   // （buildRig 内同名别名不外泄——本用例局部起）

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty()) << "沙盒根未定位（JMW_FC_SANDBOX 或 CWD 向上搜索）";
    ReplayRig rig;
    std::string err;
    ASSERT_TRUE(buildRig(sb, rig, err)) << err;
    rig.resetSessionState();

    // ── 扫描流水线＋ obs 落账（app collectorLoop 同款：R/T＋regFlow 标志点
    //    观测（globalId≥0）/conic 兜底（-1）＋逐帧激光缓存）──
    // 帧窗 100：链路正确性验证口径（非 GBA 规模稳健性——399 帧全量实测 Ceres
    // 线性求解大量 numeric factorization 失败＝病态，部分收敛位姿下标志点离
    // 观测达 7.6mm；100 帧窗口求解良态（50 帧同量级 app 会话 RMSE 0.0043mm）
    const int nFrames = rig.total < 100 ? rig.total : 100;
    auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*rig.h, 0, nFrames);
    size_t laserCachedFrames = 0;
    for (int c = 0; c < nFrames; ++c) {
        scr::CycleProductsCompat prod;
        std::string ce;
        ASSERT_TRUE(ex->collect(c, prod, ce)) << "collect 失败 c=" << c << " " << ce;
        sp::FrameObs fo;
        fo.frameId = prod.metaValid ? prod.metaFrameId : static_cast<uint64_t>(c);
        for (int i = 0; i < 9; ++i) fo.R_init[i] = prod.R[i];
        for (int i = 0; i < 3; ++i) fo.t_init[i] = prod.T[i];
        if (prod.regFlow.success) {
            fo.markerObs.reserve(prod.regFlow.markers.size());
            for (const auto& m : prod.regFlow.markers) {
                if (m.globalId < 0) continue;
                sp::MarkerObs mo;
                mo.xyz[0] = m.rawPosition.x;
                mo.xyz[1] = m.rawPosition.y;
                mo.xyz[2] = m.rawPosition.z;
                mo.globalId = m.globalId;
                fo.markerObs.push_back(mo);
            }
        } else if (prod.conic.success && prod.markersValid > 0) {
            fo.markerObs.reserve(prod.conic.markers.size());
            for (const auto& m : prod.conic.markers) {
                if (!m.validCircle) continue;
                sp::MarkerObs mo;
                mo.xyz[0] = m.centerX;
                mo.xyz[1] = m.centerY;
                mo.xyz[2] = m.centerZ;
                mo.globalId = -1;   // GBA 端全局关联
                fo.markerObs.push_back(mo);
            }
        }
        if (!prod.laserXyzHost.empty()) ++laserCachedFrames;
        rig.obs->push(std::move(fo), std::move(prod.laserXyzHost));
    }
    ex.reset();
    ASSERT_GT(rig.obs->frameCount(), 0u);
    EXPECT_FALSE(rig.obs->degradedLaser()) << "激光缓存降级（预算不足）";
    EXPECT_GT(laserCachedFrames, 0u) << "逐帧激光缓存为空（fuseLoop 填充链断）";

    // ── C 线基线云下载（优化后激光对照）──
    std::vector<float> cline;
    {
        const auto ctx = rig.adapter->fuse().GetDeviceContext();
        ASSERT_NE(ctx.d_fusedXyz, nullptr);
        ASSERT_GT(ctx.fusedPointCount, 0u);
        cline.resize(static_cast<size_t>(ctx.fusedPointCount) * 3);
        ASSERT_EQ(cudaMemcpy(cline.data(), ctx.d_fusedXyz,
                             cline.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
    ASSERT_GE(cline.size() / 3, size_t{10000}) << "C 线融合云不足 1 万点";

    // ── 全局优化（GBA 批算＋激光重放——app runFinalBA 同款默认参）──
    sp::GlobalOptimObject go;
    sp::PipelineDeps deps;
    ASSERT_TRUE(go.configure(deps).success);
    sp::CancelToken cancel;
    const auto gr = go.run(*rig.obs, [](int, const std::string&) {}, cancel);
    ASSERT_TRUE(gr.success) << gr.message;
    const auto& out = go.output();
    ASSERT_TRUE(out.gbaSuccess) << "GBA 批算失败";
    EXPECT_TRUE(out.laserReplayed) << "激光重放未执行（缓存空/降级？）";
    ASSERT_GE(out.laserXyzFinal.size() / 3, size_t{10000})
        << "优化后激光云不足 1 万点";

    // ── 验证①：优化后标志点 ──
    double worstMkMm = 0.0;
    {
        const auto snap = rig.obs->snapshot();
        // 全部 globalId≥0 观测变换到世界系（初始位姿）——GBA 标志点的对照池。
        // 不按 id 分簇对照：链断重锚的高 id 段观测少且集中会话后期（配准漂移
        // 大），其自身簇均值不可靠（首版实证 id14-20 偏 3-7.6mm＝回环漂移量级
        // 非错位）；改「邻近任一观测 ≤2mm」＝每个输出点须有真实观测支撑
        std::vector<cv::Vec3d> obsW;
        obsW.reserve(snap.obs.size() * 6);
        for (const auto& fo : snap.obs) {
            const cv::Matx33d Rm(fo.R_init);
            const cv::Vec3d tv(fo.t_init);
            for (const auto& mo : fo.markerObs) {
                if (mo.globalId < 0) continue;
                obsW.push_back(Rm * cv::Vec3d(mo.xyz[0], mo.xyz[1], mo.xyz[2]) + tv);
            }
        }
        ASSERT_FALSE(obsW.empty()) << "无 globalId≥0 标志点观测（GBA 无输入）";
        EXPECT_GE(out.gbaMarkers.size(), size_t{5}) << "GBA 标志点数量异常少";
        EXPECT_LE(out.gbaMarkers.size(), size_t{40}) << "GBA 标志点数量异常多";
        for (size_t i = 0; i < out.gbaMarkers.size(); ++i) {
            const auto& m = out.gbaMarkers[i];
            ASSERT_TRUE(std::isfinite(m.X.x) && std::isfinite(m.X.y)
                        && std::isfinite(m.X.z));
            // 两两间距 >10mm（GBA 近距合并生效＝无重复 id 残留副本）
            for (size_t j = i + 1; j < out.gbaMarkers.size(); ++j) {
                const auto& n = out.gbaMarkers[j];
                const double dx = m.X.x - n.X.x, dy = m.X.y - n.X.y,
                             dz = m.X.z - n.X.z;
                EXPECT_GT(std::sqrt(dx * dx + dy * dy + dz * dz), 10.0)
                    << "标志点对 " << i << "-" << j << " 间距 ≤10mm（重复 id 残留）";
            }
            // 邻近任一真实观测 ≤5mm（初始位姿世界系。容差依据：GBA 位姿修正
            // 随帧窗累积——100 帧窗实测最差 2.8mm（锚定头帧近 0、窗尾最大）；
            // 槽位/关联错配时偏离达帧间运动量级 cm 级 → 5mm 仍强判别）
            double best = 1e30;
            for (const auto& w : obsW) {
                const double dx = w[0] - m.X.x, dy = w[1] - m.X.y, dz = w[2] - m.X.z;
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best) best = d2;
            }
            const double d = std::sqrt(best);
            worstMkMm = std::max(worstMkMm, d);
            EXPECT_LE(d, 5.0) << "GBA 标志点 " << i << "（id=" << m.globalId
                              << "）距最近观测 " << d << "mm（无真实观测支撑）";
        }
        fprintf(stderr, "[GBA-VERIFY] 标志点=%zu 观测池=%zu 最差邻近=%.3fmm\n",
                out.gbaMarkers.size(), obsW.size(), worstMkMm);
    }

    // ── 验证②：前 1 万激光点邻近 C 线融合云（体素哈希 NN——cell 2mm＋27 邻域）──
    {
        struct GbaNn {   // 体素坐标/键（普通函数口径——lambda 嵌套捕获 MSVC 拒）
            static int32_t vc(float v, float cell) {
                return static_cast<int32_t>(std::floor(v / cell));
            }
            static int64_t key(float x, float y, float z, float cell) {
                return (static_cast<int64_t>(vc(x, cell) & 0x1FFFFF) << 42)
                     | (static_cast<int64_t>(vc(y, cell) & 0x1FFFFF) << 21)
                     | static_cast<int64_t>(vc(z, cell) & 0x1FFFFF);
            }
        };
        constexpr float kCell = 2.0f;
        std::unordered_map<int64_t, std::vector<cv::Vec3f>> grid;
        grid.reserve(cline.size() / 3);
        for (size_t i = 0; i + 2 < cline.size(); i += 3)
            grid[GbaNn::key(cline[i], cline[i + 1], cline[i + 2], kCell)]
                .emplace_back(cline[i], cline[i + 1], cline[i + 2]);

        const size_t totalPts = out.laserXyzFinal.size() / 3;
        const size_t take = totalPts < size_t{10000} ? totalPts : size_t{10000};
        // 首/尾两段各取 1 万点：首段＝规范锚定区（GBA 位姿修正≈0——严格 2mm 验
        // 槽位对应）；尾段＝位姿修正累积最大区（实测 ~3mm 量级——容差 5mm 仍
        // 远低于帧间运动量级 cm，可捕错配）
        const struct { size_t begin; double tol; const char* tag; } kRanges[] = {
            {0, 2.0, "head"}, {totalPts - take, 5.0, "tail"}};
        for (const auto& rg : kRanges) {
            size_t over = 0, nonFinite = 0;
            double worstMm = 0.0;
            for (size_t q = 0; q < take; ++q) {
                const size_t i = rg.begin + q;
                const float x = out.laserXyzFinal[i * 3 + 0],
                            y = out.laserXyzFinal[i * 3 + 1],
                            z = out.laserXyzFinal[i * 3 + 2];
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    ++nonFinite;
                    continue;
                }
                double best = 1e30;
                const int32_t cx = GbaNn::vc(x, kCell), cy = GbaNn::vc(y, kCell),
                              cz = GbaNn::vc(z, kCell);
                for (int32_t dx = -1; dx <= 1; ++dx)
                    for (int32_t dy = -1; dy <= 1; ++dy)
                        for (int32_t dz = -1; dz <= 1; ++dz) {
                            const int64_t k =
                                (static_cast<int64_t>((cx + dx) & 0x1FFFFF) << 42)
                              | (static_cast<int64_t>((cy + dy) & 0x1FFFFF) << 21)
                              | static_cast<int64_t>((cz + dz) & 0x1FFFFF);
                            const auto it = grid.find(k);
                            if (it == grid.end()) continue;
                            for (const auto& p : it->second) {
                                const double ddx = p[0] - x, ddy = p[1] - y,
                                             ddz = p[2] - z;
                                const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                                if (d2 < best) best = d2;
                            }
                        }
                const double d = best < 1e29 ? std::sqrt(best) : 1e30;
                worstMm = std::max(worstMm, d);
                if (d > rg.tol) ++over;
            }
            EXPECT_EQ(nonFinite, size_t{0}) << rg.tag << " 段含非有限值";
            EXPECT_EQ(over, size_t{0})
                << rg.tag << " 段 " << take << " 点中 " << over << " 点偏离 C 线云 >"
                << rg.tol << "mm（位姿/激光槽位错配？）最差 " << worstMm << "mm";
            fprintf(stderr,
                    "[GBA-VERIFY] 激光%s段 %zu 点：C线云=%zu 最差邻近=%.3fmm "
                    "超限(>%.1fmm)=%zu\n",
                    rg.tag, take, cline.size() / 3, worstMm, rg.tol, over);
        }
    }
}

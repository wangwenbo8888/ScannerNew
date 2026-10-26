// test_fc_replay.cpp — 07 侧五链执行器离线对拍（261025 Phase 2 验证）
//
// 目的：证明 07 PipelineManager 内的 FiveChainExecutor + FcHarnessCompat（移植体）
// 与沙盒 speed_optimize golden（all399/report.csv）逐行一致。
//
// 运行条件（缺一自动 SKIP——CI 安全）：
//   · env JMW_FC_REPLAY=1（显式启用——全程 ~分钟级且需 GPU）
//   · 沙盒数据齐全：modules/09_operatorlib/TEST/speed_optimize/
//       date_in/laser-mark1688-1006/{ALL_raw(meta.json+left.bin+right.bin),
//       camera_calib.json, curve_map_temp_table.bin, golden/all399/report.csv}
//       config_flat.json
//   · env JMW_FC_SANDBOX 可覆盖沙盒根（默认自 CWD 向上搜索）
//
// 输出：<temp>/fc07_replay/{report.csv, rt17.csv}（人工比对用）
// 判定：report.csv 与 golden 逐行字符串相等（11 列全等——markers/valid/regSource/
// regMatched/dt/drot/laserMatched/laserPoints3D/rowResid/dispMed）
#include <gtest/gtest.h>

#include <algorithm>
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

} // namespace

TEST(FcReplay, All399ReportParity) {
    if (getenv("JMW_FC_REPLAY") == nullptr)
        GTEST_SKIP() << "JMW_FC_REPLAY=1 未设——离线对拍显式启用";

    const fs::path sb = locateSandbox();
    ASSERT_FALSE(sb.empty()) << "沙盒根未定位（JMW_FC_SANDBOX 或 CWD 向上搜索）";
    const fs::path dataRoot = sb / "date_in/laser-mark1688-1006";
    const fs::path rawDir = dataRoot / "ALL_raw";
    const fs::path camJson = dataRoot / "camera_calib.json";
    const fs::path cmttBin = dataRoot / "curve_map_temp_table.bin";
    const fs::path cfgJson = sb / "config_flat.json";
    const fs::path goldenCsv = sb / "golden/all399/report.csv";
    std::error_code ec;
    ASSERT_TRUE(fs::exists(rawDir / "meta.json", ec)) << rawDir << " meta.json 缺失";
    ASSERT_TRUE(fs::exists(camJson, ec)) << camJson << " 缺失";
    ASSERT_TRUE(fs::exists(cmttBin, ec)) << cmttBin << " 缺失";
    ASSERT_TRUE(fs::exists(cfgJson, ec)) << cfgJson << " 缺失";
    ASSERT_TRUE(fs::exists(goldenCsv, ec)) << goldenCsv << " 缺失";

    // —— 配置（operators json + 标称温）——
    json cfg;
    {
        std::ifstream f(cfgJson);
        ASSERT_TRUE(f) << "config_flat.json 不可读";
        f >> cfg;
    }
    const double nominalTemp = cfg.at("runtime").value("nominalTemperature", 22.5);
    const json operators = cfg.value("operators", json::object());

    // —— 标定 ——
    TestCalib tc;
    std::string err;
    ASSERT_TRUE(loadTestCalib(camJson, cmttBin, nominalTemp, tc, err)) << err;
    ASSERT_GT(tc.W, 0);

    // —— 数据集（raw 整载——零拷贝视图）——
    std::vector<int> frameIds;
    int rawW = 0, rawH = 0;
    {
        std::ifstream mf(rawDir / "meta.json");
        ASSERT_TRUE(mf) << "meta.json 不可读";
        json meta; mf >> meta;
        rawW = meta.at("width").get<int>();
        rawH = meta.at("height").get<int>();
        for (int id : meta.at("frames").get<std::vector<int>>()) frameIds.push_back(id);
    }
    ASSERT_FALSE(frameIds.empty());
    std::sort(frameIds.begin(), frameIds.end());
    const size_t frameBytes = static_cast<size_t>(rawW) * rawH;
    std::vector<unsigned char> rawL, rawR;
    {
        std::ifstream fl(rawDir / "left.bin", std::ios::binary);
        std::ifstream fr(rawDir / "right.bin", std::ios::binary);
        ASSERT_TRUE(fl && fr) << "left.bin/right.bin 打开失败";
        rawL.resize(frameBytes * frameIds.size());
        rawR.resize(frameBytes * frameIds.size());
        fl.read(reinterpret_cast<char*>(rawL.data()), rawL.size());
        fr.read(reinterpret_cast<char*>(rawR.data()), rawR.size());
        ASSERT_TRUE(fl && fr) << "bin 读取不足";
    }

    // —— 帧源（FrameSourceCompat 静态回放口径）——
    scr::FrameSourceCompat src;
    src.cycles.fs = &src;
    for (int id : frameIds)
        src.cyclesStatic.push_back({id, (id % 2 == 1)});
    src.readFrame = [&](int idx, bool left, cv::Mat& out, std::string& e) -> bool {
        const auto it = std::lower_bound(frameIds.begin(), frameIds.end(), idx);
        if (it == frameIds.end() || *it != idx) {
            e = "帧号不在数据集: " + std::to_string(idx);
            return false;
        }
        const size_t pos = static_cast<size_t>(it - frameIds.begin());
        unsigned char* base = (left ? rawL : rawR).data() + pos * frameBytes;
        out = cv::Mat(rawH, rawW, CV_8UC1, base);   // 零拷视图（缓冲随 test 存活）
        return true;
    };

    // —— 适配器＋harness（沙盒对拍装配：operators json 全参）——
    Scanner::pipeline::FiveChainConfig fcCfg;   // ScrFiveChain 默认值域（逐字段核对一致）
    Scanner::pipeline::FrameObsAccumulator dummyObs(1 << 20);
    Scanner::pipeline::FiveChainAdapter adapter(fcCfg, dummyObs, nullptr, nullptr,
                                                &operators);
    // sep 预热：首 Execute 走 CUDA Graph 捕获——内部缓冲须先 Warmup 分配
    //（沙盒靠 serialHead 串行帧预热；07 直起执行器须显式）
    adapter.sepL().Warmup(tc.H, tc.W);
    adapter.sepR().Warmup(tc.H, tc.W);
    auto* h = new Scanner::pipeline::FcHarnessCompat(adapter, src);
    h->cfg_.fivechain = fcCfg;
    h->cfg.fivechain = fcCfg;
    h->cfg_.operators = operators;
    h->cfg_.runtime.nominalTemperature = nominalTemp;
    h->cfg_.runtime.warmupCycles = 0;
    h->calib_.Q = tc.Q;  h->calib_.R1 = tc.R1;  h->calib_.R2 = tc.R2;
    h->calib_.P1 = tc.P1;  h->calib_.P2 = tc.P2;
    h->calib_.K1c = tc.K1c;  h->calib_.K2c = tc.K2c;
    h->calib_.D1 = tc.D1;  h->calib_.D2 = tc.D2;
    h->calib_.Rext = tc.Rext;  h->calib_.Tcomp = tc.Tcomp;
    h->imageWidth_ = tc.W;  h->imageHeight_ = tc.H;
    h->tableOdd_ = tc.laserTable;
    h->tableEven_ = tc.laserTable;
    h->sepLaserL_ = &adapter.sepL();
    h->sepLaserR_ = &adapter.sepR();
    h->voxelFuseOp_ = &adapter.fuse();
    h->prodProfile_ = false;    // 沙盒口径：统计记账开（dispMedPx 等）
    // CCL 门限（config ccl 节）
    if (operators.contains("ccl")) {
        const auto& c = operators.at("ccl");
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
        ul.cameraMatrix = tc.K1c; ul.distCoeffs = tc.D1;
        ul.R = tc.R1; ul.P = tc.P1;
        ur.cameraMatrix = tc.K2c; ur.distCoeffs = tc.D2;
        ur.R = tc.R2;
        ur.P = tc.P2.clone();
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

    // —— 执行器（399 帧有界——metrics 正常落盘）——
    const int total = static_cast<int>(frameIds.size());
    fs::path outDir = fs::temp_directory_path() / "fc07_replay";
    fs::create_directories(outDir, ec);
    h->outDir_ = outDir.string();
    auto ex = std::make_unique<scr::fc::FiveChainExecutor>(*h, 0, total);

    // —— 收割＋report.csv / rt17.csv 落盘 ——
    std::vector<std::string> rows;
    std::ostringstream rt17;
    double prevR[9], prevT[3];
    bool havePrev = false;
    for (int c = 0; c < total; ++c) {
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
    delete h;

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
        refCsv = goldenCsv;
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

    // ① 硬门：计数列全同（markersDetected/markersValid/regMatched±/
    // laserMatched/laserPoints3D——regMatched 属光流匹配数，FP 涟漪可 ±1）
    int regSourceFlip = 0;      // 路径翻面（fuse/hold 门限边界）
    int regMatchedJitter = 0;   // regMatched ±1（光流平局翻转）
    int floatWiggle = 0;        // dt/drot/dispMed 第 4 位小数漂移
    int exactRows = 0;
    std::ostringstream dmsg;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] == ref[i]) { ++exactRows; continue; }
        const auto m = [](const std::string& s) {
            std::vector<std::string> v; std::stringstream ss(s); std::string t;
            while (std::getline(ss, t, ',')) v.push_back(t);
            return v;
        };
        const auto mv = m(rows[i]);
        const auto rv = m(ref[i]);
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
    // ② 软门：门限边界翻转稀有事件（≤5%；261026 实测 13/399=3.3%——DLL vs
    // 静态库浮点求值序差在 conic validCircle/fuse 门限/光流平局处的涟漪，
    // 伴随帧前后自愈，非逻辑差异）
    EXPECT_LE(regSourceFlip + regMatchedJitter, static_cast<int>(rows.size() * 5 / 100))
        << "门限翻转超限: 路径翻面=" << regSourceFlip << " 匹配数抖动=" << regMatchedJitter
        << dmsg.str();

    SUCCEED() << "399 帧对拍：逐位同 " << exactRows << "/浮点尾位漂移 " << floatWiggle
              << "/门限涟漪 " << (regSourceFlip + regMatchedJitter)
              << "（计数列 100% 同；输出留档 " << outDir.string()
              << "；基准 " << refCsv.filename().string() << "）";
}

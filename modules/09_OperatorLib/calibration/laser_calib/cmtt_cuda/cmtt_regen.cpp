// ============================================================================
// cmtt_regen.cpp — CMTT 温表再生入口工具（261117 W1）
//
// 用途：从装机位/产物的标定 JSON（激光 laser_calib.json + 相机 camera_calib.json）
//       三源（PJC 发射曲线 / 立体矫正温表 / 激光外参温表）重新生成 CMTT sidecar。
//       补上产线 09 侧"生成器零调用方"的缺口（此前生成只在 factory_calib CLI 4-14）。
//
// 用法：
//   cmtt_regen <laser_calib.json> <camera_calib.json> <out.bin>
//       [--below N] [--above N] [--step C] [--rowstep F] [--dmin F] [--dmax F]
//       [--engine cpu|cuda]
//   默认档位口径＝2026-10-13 产线终态定稿：10 档 4.0C（below=2/above=7）、
//   深度窗 [200,500]、rowStep 取激光 JSON pjc.epipolarRowStep——全部可覆盖
//   （吸取 fc --cmtt-gen 硬编码对称 61 档/过时深度窗的教训）。
//
// 设计先例：gba_dataset_runner（本目录树内独立 exe）；三源解析逻辑对齐
//       fc fcstepdump runCmttGen（纯 nlohmann 实现，不引 fc calib_io 依赖）。
// ============================================================================

#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "curve_map.h"
#include "curve_map_temp_table.h"
#include "cmtt_cuda.h"
#include "stereo_rectify_temp_table_cpu.h"
#include "laser_extrinsic_compensate_cpu.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

// ---- 小工具 ---------------------------------------------------------------

json loadJson(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("打不开 JSON: " + path);
    json j = json::parse(f, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded()) throw std::runtime_error("JSON 解析失败: " + path);
    return j;
}

cv::Mat matFromJson(const json& j, int rows, int cols, const char* what) {
    cv::Mat m(rows, cols, CV_64F);
    if (!j.is_array() || (int)j.size() != rows)
        throw std::runtime_error(std::string(what) + ": 期望 " +
                                 std::to_string(rows) + "x" + std::to_string(cols) + " 行列数组");
    for (int r = 0; r < rows; ++r) {
        const json& row = j.at(r);
        if (!row.is_array() || (int)row.size() < cols)
            throw std::runtime_error(std::string(what) + ": 第 " + std::to_string(r) + " 行不足 " +
                                     std::to_string(cols) + " 列");
        for (int c = 0; c < cols; ++c) m.at<double>(r, c) = row.at(c).get<double>();
    }
    return m;
}

// ---- 三源解析 --------------------------------------------------------------

// 源①：PJC 发射曲线（laser JSON → pjc.emissionCurves[]）
std::vector<calib::ImplicitCurve> parseCurves(const json& laser) {
    if (!laser.contains("pjc") || !laser.at("pjc").contains("emissionCurves"))
        throw std::runtime_error("laser JSON 缺 pjc.emissionCurves");
    std::vector<calib::ImplicitCurve> curves;
    for (const auto& c : laser.at("pjc").at("emissionCurves")) {
        calib::ImplicitCurve ic{};
        const auto& co = c.at("coeffs");
        if (!co.is_array() || co.size() != 6)
            throw std::runtime_error("emissionCurves.coeffs 期望 6 元");
        for (int k = 0; k < 6; ++k) ic.coeffs[k] = co.at(k).get<double>();
        curves.push_back(ic);
    }
    if (curves.empty()) throw std::runtime_error("emissionCurves 为空");
    return curves;
}

// 源②：立体矫正温表（camera JSON → stereoRectifyTempTable）
// 仅填 TempParamInterpolator 消费面：temperature / P1 / P2 / validRoiLeft（硬规则 #1）。
calib::StereoRectifyTempTableResult parseRectTable(const json& camera) {
    if (!camera.contains("stereoRectifyTempTable"))
        throw std::runtime_error("camera JSON 缺 stereoRectifyTempTable");
    const json& st = camera.at("stereoRectifyTempTable");
    if (!st.contains("table") || st.at("table").empty())
        throw std::runtime_error("stereoRectifyTempTable.table 为空");
    calib::StereoRectifyTempTableResult r;
    r.success = true;
    r.referenceTemp = st.at("referenceTemp").get<double>();
    for (const auto& e : st.at("table")) {
        calib::StereoRectifyTempEntry en{};
        en.temperature = e.at("temperature").get<double>();
        en.P1 = matFromJson(e.at("P1"), 3, 4, "P1");
        en.P2 = matFromJson(e.at("P2"), 3, 4, "P2");
        const json& roi = e.at("validRoiLeft");
        en.validRoiLeft = cv::Rect(roi.at("x").get<int>(), roi.at("y").get<int>(),
                                   roi.at("w").get<int>(), roi.at("h").get<int>());
        r.table.push_back(std::move(en));
    }
    return r;
}

// 源③：激光外参温表（laser JSON → laserExtrinsicTempTable.virtualToLeft）
// 仅填插值器消费面：referenceTemperature + virtualToLeft.table[].{temperature, T}
calib::LaserExtrinsicCompensateCPUResult parseLaserTable(const json& laser) {
    if (!laser.contains("laserExtrinsicTempTable"))
        throw std::runtime_error("laser JSON 缺 laserExtrinsicTempTable");
    const json& lt = laser.at("laserExtrinsicTempTable");
    if (!lt.contains("virtualToLeft") || !lt.at("virtualToLeft").contains("table"))
        throw std::runtime_error("laserExtrinsicTempTable.virtualToLeft.table 缺失");
    calib::LaserExtrinsicCompensateCPUResult r;
    r.success = true;
    r.referenceTemp = lt.at("referenceTemperature").get<double>();
    for (const auto& e : lt.at("virtualToLeft").at("table")) {
        calib::ExtrinsicCompensatedEntry en{};
        en.temperature = e.at("temperature").get<double>();
        const auto& t = e.at("T");
        for (int k = 0; k < 3; ++k) en.T[k] = t.at(k).get<double>();
        r.leftResult.table.push_back(en);
    }
    if (r.leftResult.table.empty()) throw std::runtime_error("virtualToLeft.table 为空");
    return r;
}

cv::Size parseImageSize(const json& camera) {
    const json& s = camera.at("imageSize");
    if (s.is_array() && s.size() >= 2)            // 模块1 输出 schema: [width, height]
        return cv::Size(s.at(0).get<int>(), s.at(1).get<int>());
    if (s.is_object())                             // 容错: 对象形态
        return cv::Size(s.at("width").get<int>(), s.at("height").get<int>());
    throw std::runtime_error("imageSize 格式不识别（期望 [w,h] 数组或对象）");
}

void usage() {
    std::cout <<
        "cmtt_regen <laser_calib.json> <camera_calib.json> <out.bin>\n"
        "  [--below N(2)] [--above N(7)] [--step C(4.0)] [--rowstep F(pjc.epipolarRowStep)]\n"
        "  [--dmin F(200)] [--dmax F(500)] [--engine cpu|cpu(default)|cuda]\n"
        "默认档位＝2026-10-13 产线终态定稿（10 档 4.0C，below 2 / above 7，深度窗 [200,500]）\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) { usage(); return 1; }
    std::string laserPath = argv[1], cameraPath = argv[2], outPath = argv[3];

    calib::CurveMapTempTableGenParams gp;   // 默认＝产线终态定稿口径
    gp.tiersBelow = 2;
    gp.tiersAbove = 7;
    gp.tempStep = 4.0f;
    gp.depthMin = 200.0f;
    gp.depthMax = 500.0f;
    bool useCuda = false;

    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " 缺参数值");
            return argv[++i];
        };
        if (a == "--below") gp.tiersBelow = std::stoi(next("--below"));
        else if (a == "--above") gp.tiersAbove = std::stoi(next("--above"));
        else if (a == "--step") gp.tempStep = std::stof(next("--step"));
        else if (a == "--rowstep") gp.rowStep = std::stof(next("--rowstep"));
        else if (a == "--dmin") gp.depthMin = std::stof(next("--dmin"));
        else if (a == "--dmax") gp.depthMax = std::stof(next("--dmax"));
        else if (a == "--engine") useCuda = (next("--engine") == "cuda");
        else { std::cerr << "未知参数: " << a << "\n"; usage(); return 1; }
    }

    try {
        const json laser = loadJson(laserPath);
        const json camera = loadJson(cameraPath);

        auto curves = parseCurves(laser);
        auto rectTable = parseRectTable(camera);
        auto laserTable = parseLaserTable(laser);
        const cv::Size imageSize = parseImageSize(camera);

        // rowStep 红线：调用方显式给定（默认取激光 JSON pjc.epipolarRowStep，无猜默认）
        if (gp.rowStep <= 0.0f) {
            if (laser.contains("pjc") && laser.at("pjc").contains("epipolarRowStep"))
                gp.rowStep = laser.at("pjc").at("epipolarRowStep").get<float>();
            else
                throw std::runtime_error("rowStep 未指定且激光 JSON 无 pjc.epipolarRowStep——须 --rowstep");
        }
        gp.referenceTemp = rectTable.referenceTemp;   // 参考温锚＝矫正温表（两表一致性由生成器校验）

        std::cout << "源: curves=" << curves.size()
                  << " rectT=" << rectTable.table.size() << " 档(ref=" << rectTable.referenceTemp << "C)"
                  << " laserT=" << laserTable.leftResult.table.size() << " 档"
                  << " imageSize=" << imageSize.width << "x" << imageSize.height << "\n";
        std::cout << "参数: below=" << gp.tiersBelow << " above=" << gp.tiersAbove
                  << " step=" << gp.tempStep << "C rowStep=" << gp.rowStep
                  << " depth=[" << gp.depthMin << "," << gp.depthMax << "]"
                  << " engine=" << (useCuda ? "cuda" : "cpu") << "\n";

        calib::CurveMapTempTableResult r;
        if (useCuda) {
            calib::CmttCudaGenerator gen(gp);
            r = gen.Generate(curves, rectTable, laserTable, imageSize);
        } else {
            calib::CurveMapTempTableGenerator gen(gp);
            r = gen.Generate(curves, rectTable, laserTable, imageSize);
        }
        if (!r.success) { std::cerr << "生成失败: " << r.message << "\n"; return 2; }

        std::ofstream out(outPath, std::ios::binary);
        if (!out) { std::cerr << "写不出: " << outPath << "\n"; return 3; }
        out.write(reinterpret_cast<const char*>(r.sidecarBytes.data()),
                  (std::streamsize)r.sidecarBytes.size());
        out.close();

        std::cout << "完成: " << outPath << "\n"
                  << "  tiers=" << r.tiers.size() << " ok=" << r.okCount()
                  << " refIdx=" << r.referenceIndex() << "\n"
                  << "  温度范围 [" << r.tiers.front().temperature << "C, "
                  << r.tiers.back().temperature << "C]\n"
                  << "  bytes=" << r.sidecarBytes.size() << "\n"
                  << "  sha256 校验: Get-FileHash " << outPath << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 4;
    }
}

// calib_loader.cpp — 标定件装载实现
#include "calib_loader.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <nlohmann/json.hpp>

// 09 算子库内 cmtt_table_loader.h（scanning/laser/laser_match_scan_v3/ 经
// mod_operatorlib 的 OP_INC_DIRS 公共 include 可达，此处用相对算子根路径）
#include "scanning/laser/laser_match_scan_v3/cmtt_table_loader.h"

namespace scr {

cv::Mat matFromJson(const nlohmann::json& j, int rows, int cols) {
    if (!j.is_array() || j.empty()) return cv::Mat();
    // 形态判定：首元素是数组 → 嵌套 rows×cols；否则 → 扁平 rows*cols
    const bool nested = j.at(0).is_array();
    if (nested) {
        if (static_cast<int>(j.size()) != rows) return cv::Mat();
        cv::Mat m(rows, cols, CV_64F);
        for (int r = 0; r < rows; ++r) {
            const auto& row = j.at(r);
            if (!row.is_array() || static_cast<int>(row.size()) != cols)
                return cv::Mat();
            for (int c = 0; c < cols; ++c) m.at<double>(r, c) = row.at(c).get<double>();
        }
        return m;
    }
    // 扁平（如 T: [-131, 2.08, 22.5] → 3×1；dist_coeffs → 1×5）
    if (static_cast<int>(j.size()) != rows * cols) return cv::Mat();
    cv::Mat m(rows, cols, CV_64F);
    for (int i = 0; i < rows * cols; ++i) m.at<double>(i / cols, i % cols) = j.at(i).get<double>();
    return m;
}

const nlohmann::json* selectTier(const nlohmann::json& table, double nominal) {
    const nlohmann::json* best = nullptr;
    double bestT = 0.0;
    for (const auto& tier : table) {
        if (!tier.contains("temperature")) continue;
        const double t = tier.at("temperature").get<double>();
        if (best == nullptr || std::abs(t - nominal) < std::abs(bestT - nominal) - 1e-12 ||
            (std::abs(std::abs(t - nominal) - std::abs(bestT - nominal)) < 1e-12 && t < bestT)) {
            best = &tier;
            bestT = t;
        }
    }
    return best;
}

int checkRowStep(float rowStep, float epipolarRowStep, std::string& msg) {
    if (rowStep <= 0.0f || epipolarRowStep <= 0.0f) {
        msg = "rowStep 非法: table=" + std::to_string(rowStep)
              + " interp=" + std::to_string(epipolarRowStep);
        return -1;
    }
    if (std::abs(rowStep - epipolarRowStep) < 1e-4f) {
        msg = "rowStep 一致 PASS (table=" + std::to_string(rowStep) + ")";
        return 0;
    }
    const float ratio = rowStep > epipolarRowStep ? rowStep / epipolarRowStep
                                                  : epipolarRowStep / rowStep;
    if (std::abs(ratio - std::round(ratio)) < 1e-3f) {
        msg = "rowStep 为倍数关系 WARN (table=" + std::to_string(rowStep) + " interp="
              + std::to_string(epipolarRowStep) + " ratio≈" + std::to_string(ratio)
              + ")——行桶对齐关系待实现期确认";
        return 1;
    }
    msg = "rowStep 无整数倍关系 ERROR (table=" + std::to_string(rowStep) + " interp="
          + std::to_string(epipolarRowStep) + ")";
    return -1;
}

bool loadCalib(const ScrConfig& cfg, CalibData& out, std::string& err) {
    try {
        std::ifstream ifs(cfg.paths.cameraCalib);
        if (!ifs) {
            err = "camera_calib.json 打开失败: " + cfg.paths.cameraCalib;
            return false;
        }
        const nlohmann::json j = nlohmann::json::parse(ifs);

        if (!j.contains("intrinsic") || !j.contains("extrinsic")
            || !j.contains("stereoRectifyTempTable")) {
            err = "camera_calib.json 缺 intrinsic/extrinsic/stereoRectifyTempTable 节";
            return false;
        }

        // —— 参考内参（键名 snake_case，实测确认）——
        out.K1 = matFromJson(j.at("intrinsic").at("left").at("camera_matrix"), 3, 3);
        out.D1 = matFromJson(j.at("intrinsic").at("left").at("dist_coeffs"), 1, 5);
        out.K2 = matFromJson(j.at("intrinsic").at("right").at("camera_matrix"), 3, 3);
        out.D2 = matFromJson(j.at("intrinsic").at("right").at("dist_coeffs"), 1, 5);
        out.Rext = matFromJson(j.at("extrinsic").at("R"), 3, 3);
        out.Text = matFromJson(j.at("extrinsic").at("T"), 3, 1);
        if (out.K1.empty() || out.K2.empty() || out.D1.empty() || out.D2.empty()
            || out.Rext.empty() || out.Text.empty()) {
            err = "camera_calib.json 内参/外参矩阵布局解析失败";
            return false;
        }

        // —— 名义温就近档 ——
        const auto& table = j.at("stereoRectifyTempTable").at("table");
        if (!table.is_array() || table.empty()) {
            err = "stereoRectifyTempTable.table 为空";
            return false;
        }
        const nlohmann::json* tier = selectTier(table, cfg.runtime.nominalTemperature);
        if (tier == nullptr) {
            err = "温表选档失败（无含 temperature 的档）";
            return false;
        }
        out.tierTemperature = tier->at("temperature").get<double>();
        if (j.contains("imageSize") && j.at("imageSize").is_array()
            && j.at("imageSize").size() == 2) {
            out.imageWidth = j.at("imageSize").at(0).get<int>();
            out.imageHeight = j.at("imageSize").at(1).get<int>();
        }
        out.R1 = matFromJson(tier->at("R1"), 3, 3);
        out.R2 = matFromJson(tier->at("R2"), 3, 3);
        out.P1 = matFromJson(tier->at("P1"), 3, 4);
        out.P2 = matFromJson(tier->at("P2"), 3, 4);
        out.Q = matFromJson(tier->at("Q"), 4, 4);
        out.K1c = matFromJson(tier->at("compensatedCameraMatrixL"), 3, 3);
        out.K2c = matFromJson(tier->at("compensatedCameraMatrixR"), 3, 3);
        out.Tcomp = matFromJson(tier->at("compensatedT"), 3, 1);
        if (out.R1.empty() || out.R2.empty() || out.P1.empty() || out.P2.empty()
            || out.Q.empty() || out.K1c.empty() || out.K2c.empty() || out.Tcomp.empty()) {
            err = "温表档内矩阵布局解析失败 (tier temp="
                  + std::to_string(out.tierTemperature) + ")";
            return false;
        }

        // —— CMTT 激光温表（就近单档；双管分立可选）——
        std::string cmttErr;
        out.laserTable = calib::LoadTempTableFromCmtt(
            cfg.paths.cmttSidecar, cfg.runtime.nominalTemperature, cmttErr,
            &out.laserTierIdx, &out.laserTierTemp, &out.laserRowStep);
        if (!out.laserTable) {
            err = "CMTT 温表装载失败: " + cmttErr;
            return false;
        }
        // 【已废弃 · 人工裁决 2026-10-11】按工作视差带/距离裁剪表（E3'）方向废弃：
        // 实际扫描工件多种多样，可能镂空、存在剧烈高度差——帧内深度不连续，
        // "单帧落在窄 disp 带"的前提不成立。不要再在此方向提出新方案或展开讨论
        // （含动态选带/帧级视差定带等一切变体）。
        // 本块仅为历史三帧基线（match_baseline_3frames 带内口径）保留复现能力：
        // 默认关闭，仅环境变量 JMW_DISP_BAND="lo,hi" 显式设置才生效
        // （空/非法＝不过滤，保持全表基线）。
        if (const char* bandEnv = std::getenv("JMW_DISP_BAND")) {
            double b0 = 0, b1 = 0;
            if (std::sscanf(bandEnv, "%lf,%lf", &b0, &b1) == 2 && b1 > b0) {
                const auto* src = out.laserTable->lookup(cfg.runtime.nominalTemperature);
                if (src && !src->leftToRightMap.empty()) {
                    const int srcRows = src->leftToRightMap.rows;
                    std::vector<cv::Vec4f> keep;
                    keep.reserve((size_t)srcRows);
                    for (int i = 0; i < srcRows; ++i) {
                        const cv::Vec4f& e = src->leftToRightMap.at<cv::Vec4f>(i);
                        double disp = (double)e[0] - (double)e[2];
                        if (disp >= b0 && disp <= b1) keep.push_back(e);
                    }
                    auto filt = std::make_shared<calib::LaserPlaneMapTempTable>();
                    calib::LaserPlaneMap dst;
                    dst.temperature = src->temperature;
                    dst.leftToRightMap =
                        cv::Mat((int)keep.size(), 1, CV_32FC4, keep.data()).clone();
                    dst.totalPairs = (int)keep.size();
                    std::cout << "[calib][E3'] disp 带裁剪 [" << b0 << "," << b1
                              << "]：表条目 " << srcRows << " -> " << (int)keep.size()
                              << "（保留 " << std::fixed << std::setprecision(1)
                              << (100.0 * (double)keep.size() / srcRows) << "%）" << std::endl;
                    filt->table[cfg.runtime.nominalTemperature] = std::move(dst);
                    out.laserTable = filt;
                }
            }
        }
        if (!cfg.paths.cmttSidecarEven.empty()) {
            int idx2 = -1; double t2 = 0; float rs2 = 0;
            out.laserTableEven = calib::LoadTempTableFromCmtt(
                cfg.paths.cmttSidecarEven, cfg.runtime.nominalTemperature, cmttErr,
                &idx2, &t2, &rs2);
            if (!out.laserTableEven) {
                err = "偶号管 CMTT 温表装载失败: " + cmttErr;
                return false;
            }
            std::cout << "[calib] 偶号管 CMTT 档 idx=" << idx2 << " temp=" << t2
                      << "°C rowStep=" << rs2 << std::endl;
        } else {
            out.laserTableEven = out.laserTable;   // 缺省两管共表
        }

        // ——【uR 烘表止血 2026-11-08】装载层逐线补偿（专题 §10.2 方案 C）——
        // 实测（E2）：表 uR 预测随线号偏差 0→+2.9px（逐线常数、随批次变化）。
        // 烘在装载层＝v3 整版/v4 前段/v4 后段残差口径全体一致，无算子改动。
        // sidecar 绑表校验（字节数＋rowStep），失配＝报错拒载（防错拷案复发）。
        // 与 2026-10-11 停用的两代算子内补偿的差异：偏差已由 399 帧残差实测钉死
        // ＋绑表校验＋过 E5 验证闸（后 199 帧每线中位 |res|≤0.3px、斜率≤0.02）。
        if (!cfg.paths.cmttBiasSidecar.empty()) {
            std::ifstream bf(cfg.paths.cmttBiasSidecar);
            if (!bf) {
                err = "uR 烘表 sidecar 不可读: " + cfg.paths.cmttBiasSidecar;
                return false;
            }
            nlohmann::json sj;
            try {
                bf >> sj;
            } catch (const std::exception& e) {
                err = std::string("uR 烘表 sidecar JSON 解析失败: ") + e.what();
                return false;
            }
            std::vector<float> bias;
            if (sj.contains("biasPx") && sj.at("biasPx").is_array())
                for (const auto& v : sj.at("biasPx")) bias.push_back(v.get<float>());
            if (bias.size() != 25) {
                err = "uR 烘表 sidecar biasPx 须恰 25 个（实际 "
                      + std::to_string(bias.size()) + "）";
                return false;
            }
            // 绑表校验①：表文件字节数
            const long long expectBytes =
                sj.value("expectTableBytes", (long long)-1);
            std::error_code ec;
            const auto actualBytes = static_cast<long long>(
                std::filesystem::file_size(cfg.paths.cmttSidecar, ec));
            if (ec || actualBytes != expectBytes) {
                err = "uR 烘表 sidecar 与表失配（字节 " +
                      std::to_string(actualBytes) + " ≠ 期望 " +
                      std::to_string(expectBytes) + "）——换表必须重测并更新 sidecar，"
                      "拒载防静默错拷（专题 §10.2 红线）";
                return false;
            }
            // 绑表校验②：rowStep（装载回传值）
            const float expectRowStep = sj.value("expectRowStep", 0.0f);
            if (std::fabs(out.laserRowStep - expectRowStep) > 1e-6f) {
                err = "uR 烘表 sidecar 与表失配（rowStep " +
                      std::to_string(out.laserRowStep) + " ≠ 期望 " +
                      std::to_string(expectRowStep) + "）";
                return false;
            }
            // 烘表：激活温档逐条目 uR += bias[lid-1]（uR 列不参与行内 xL 有序性，
            // 排序不变；两管共表时指针相等只烘一次；CalibData 表为 const 共享
            // 指针——按 DISP_BAND 先例产出新表替换）
            const auto bake =
                [&](const std::shared_ptr<const calib::LaserPlaneMapTempTable>& t)
                -> std::shared_ptr<const calib::LaserPlaneMapTempTable> {
                    const calib::LaserPlaneMap* src =
                        t->lookup(cfg.runtime.nominalTemperature);
                    auto baked = std::make_shared<calib::LaserPlaneMapTempTable>();
                    calib::LaserPlaneMap dst;
                    dst.temperature = src->temperature;
                    cv::Mat m = src->leftToRightMap.clone();
                    long long touched = 0;
                    for (int i = 0; i < m.rows; ++i) {
                        cv::Vec4f& e = m.at<cv::Vec4f>(i);
                        const int lid = cvRound(e[3]);
                        if (lid >= 1 && lid <= 25) {
                            e[2] += bias[static_cast<size_t>(lid - 1)];
                            ++touched;
                        }
                    }
                    dst.leftToRightMap = m;
                    dst.totalPairs = src->totalPairs;
                    baked->table[cfg.runtime.nominalTemperature] = std::move(dst);
                    return baked;
                };
            const auto oldOdd = out.laserTable;
            const auto oldEven = out.laserTableEven;
            out.laserTable = bake(oldOdd);
            out.laserTableEven =
                (oldEven == oldOdd) ? out.laserTable : bake(oldEven);
            std::cout << "[calib][uRBias] 烘表生效：bias L1=" << bias[0]
                      << " L25=" << bias[24] << "px（sidecar="
                      << cfg.paths.cmttBiasSidecar << "）" << std::endl;
        }
        std::cout << "[calib] 立体温表选档 " << out.tierTemperature << "°C | CMTT 档 idx="
                  << out.laserTierIdx << " temp=" << out.laserTierTemp
                  << "°C rowStep=" << out.laserRowStep << std::endl;
        // 温表覆盖域诊断（与 interp 点域对照）
        if (const auto* m = out.laserTable->lookup(cfg.runtime.nominalTemperature);
            m && !m->leftToRightMap.empty()) {
            float xMin = 1e30f, xMax = -1e30f, yMin = 1e30f, yMax = -1e30f;
            float uMin = 1e30f, uMax = -1e30f;
            for (int i = 0; i < m->leftToRightMap.rows; ++i) {
                const cv::Vec4f& e = m->leftToRightMap.at<cv::Vec4f>(i, 0);
                xMin = std::min(xMin, e[0]); xMax = std::max(xMax, e[0]);
                yMin = std::min(yMin, e[1]); yMax = std::max(yMax, e[1]);
                uMin = std::min(uMin, e[2]); uMax = std::max(uMax, e[2]);
            }
            std::cout << "[calib] 温表域: N=" << m->leftToRightMap.rows
                      << " xL[" << xMin << ".." << xMax << "] yL[" << yMin << ".."
                      << yMax << "] uR[" << uMin << ".." << uMax << "]" << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("标定装载异常: ") + e.what();
        return false;
    }
}

} // namespace scr

#pragma once
// ============================================================================
// calib_loader.h — 标定件装载（camera_calib.json 解析＋CMTT 温表＋一致性检查）
// ============================================================================
// 设计：docs/plans/2026-10-04-scan-chain-replay-test-design.md §3.3
// - K/D 取参考档 intrinsic（键名 snake_case：camera_matrix/dist_coeffs，实测确认）
// - R1/R2/P1/P2/Q/compensatedCameraMatrixL/R/compensatedT 取 stereoRectifyTempTable
//   名义温就近档（61 档 7.5~37.5/0.5）
// - 激光温表经 LoadTempTableFromCmtt 就近取单档（2.36GB sidecar，单档 ~38MB）
// - rowStep 软检查：与 epipolar_interp_dual 的 epipolar_row_step（0.7）比对
// 全容错：失败返 false＋err，不抛异常。
#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "common/calib_result_types.h"  // calib::LaserPlaneMapTempTable
#include "scr_config.h"

namespace scr {

struct CalibData {
    // 参考内参（intrinsic 节，5 参数 D）
    cv::Mat K1, D1, K2, D2;
    // 双目外参（extrinsic 节；Rext 供 undistort_cpu 兜底口径用。Text 原消费者 point_reconstruct
    // 已退役——由 conic_reconstruct 顶替（K1/K2/C2 派生自温表档 P1/P2），Text 现仅装载保留）
    cv::Mat Rext, Text;
    // 名义温就近档立体矫正参数（stereoRectifyTempTable）
    cv::Mat R1, R2, P1, P2, Q;
    cv::Mat K1c, K2c, Tcomp;     // 档内 compensated（compensatedCameraMatrixL/R + compensatedT）
    double tierTemperature = 0.0;
    int imageWidth = 0, imageHeight = 0;   // imageSize 节（undistort_cpu 构造校验需）
    // 激光温表（CMTT 单档）——双激光管分立（奇偶＝管号；缺省两管共表）
    std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable;       // 奇号管（兼默认）
    std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTableEven;   // 偶号管（空=与奇号管同表）
    int laserTierIdx = -1;
    double laserTierTemp = 0.0;
    float laserRowStep = 0.0f;
};

/// 解析嵌套 JSON 数组 → cv::Mat（rows×cols CV_64F）
cv::Mat matFromJson(const nlohmann::json& j, int rows, int cols);

/// 就近选档（|t - nominal| 最小；并列取低温侧，与 LaserPlaneMapTempTable::lookup 一致）
const nlohmann::json* selectTier(const nlohmann::json& table, double nominal);

/// 装载全部标定件；err 填失败原因。成功时日志打印关键档位信息。
bool loadCalib(const ScrConfig& cfg, CalibData& out, std::string& err);

/// rowStep 与 interp_dual epipolar_row_step 一致性：相等 PASS；倍数 warn（返回 1）；
/// 无关系 error（返回 -1）。返回 0=PASS。
int checkRowStep(float rowStep, float epipolarRowStep, std::string& msg);

} // namespace scr

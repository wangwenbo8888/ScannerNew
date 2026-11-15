/**
 * @file cmtt_table_loader.h
 * @brief CMTT sidecar（curve_map 温度表 .bin）→ 单档 LaserPlaneMapTempTable 装载器
 *
 * 产线标定产出 CMTT sidecar（96B 头＋32B×档索引＋逐档 CMPU blob，cmtt_container.h）；
 * 扫描算子 laser_match_scan_v3 经 SetTempTable 消费 LaserPlaneMapTempTable
 * （calib_result_types.h）。本装载器补齐二者之间的缺口（此前仓库无 .bin 装载路径，
 * 02 ScanWorkflow 现状 laserTable 恒 nullptr）。
 *
 * 语义：
 *   - tempC 就近选档（CmttReader::nearestTier：等距取低温侧），只取该单档，不做档间插值
 *   - rowStep 用 CMPU 表内值（CurveMapTable::rowStep）
 *   - leftToRightMap：N×1 CV_32FC4，每行 [xL, yL, uR, lineId]
 *     = [uL·rowStep, row·rowStep, uR·rowStep, lid]（栅格号×rowStep＝像素坐标），
 *     内存布局与契约注释的 N×4 CV_32FC1 逐字节等价
 *   - tierIdx/tierTemp 非空时回传选中档下标与档温（供 CalibSnapshot.laserTier 等）；
 *     rowStep 非空时回传 CMPU 表内 rowStep（供消费方与 ScanChainDeps.laserRowStep
 *     一致性检查——行桶错位风险）
 *   - 失败共 8 类 → err 填原因并返回 nullptr：文件打不开 / 文件为空 / 定长读取失败 /
 *     容器 load 拒绝（magic/版本/索引边界）/ 无档 / missing 档或 CRC 不符 /
 *     CMPU Load 失败 / 条目解码数不一致
 *   - 顶层兜底捕获一切异常（bad_alloc/cv::Exception 等）→ err 填原因返回 nullptr，
 *     对外不抛异常
 */

#pragma once

#include <memory>
#include <string>
#include "common/calib_result_types.h"
#include "common/scanner_api.h"

namespace calib {

std::shared_ptr<const LaserPlaneMapTempTable>
LoadTempTableFromCmtt(const std::string& cmttPath, double tempC, std::string& err,
                      int* tierIdx = nullptr, double* tierTemp = nullptr,
                      float* rowStep = nullptr);

} // namespace calib

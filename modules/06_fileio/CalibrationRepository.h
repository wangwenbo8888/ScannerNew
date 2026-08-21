#pragma once
// CalibrationRepository.h — 标定结果仓库（app 存活件；设计 §2）
// 一格仓库两副面孔：结果仓库(01-⑦落盘持久)+参数仓库(02-①门禁查询)。
// JSON＝07 serializeCalib 载荷原样＋外层 meta{version,createdAt,calibId,imageSize}。
// GpuMat 设备数据不落盘(07 已定)；激光档表=温补相机参数原文(无网格数据，§8-1)。
#include "base/types.h"
#include "TempTableTypes.h"
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <string>
#include <vector>

namespace Scanner::data {

struct StereoData {                       // stereo 键 + meta.imageSize 汇总
    cv::Mat cameraMatrixL, cameraMatrixR, distCoeffsL, distCoeffsR;
    cv::Mat R, T, R1, R2, P1, P2, Q;      // R/T＝外参
    cv::Size imageSize;
};

struct ReadyReport {
    bool ready = false;
    std::vector<std::string> missing;     // 缺哪样报哪样
};

class CalibrationRepository {
public:
    // 写入端（01 适配 ICalibRepoWriter）：解析→校验→填内存→临时文件+原子改名落盘
    Scanner::Result write(const std::string& payloadJson, cv::Size imageSize,
                          const std::string& path = "calibration.json");
    Scanner::Result load(const std::string& path);
    // typed getters（T7/T8 实现）
    StereoData stereo() const;
    StereoTempTable stereoTempTable() const;         // tempTables.rectify.table → 06 档类型
    PlaneMapTempTableRef planeMapTiers() const;      // planeMap.tempTable.table 档温索引
    nlohmann::json planeMapTempTableRaw() const;     // 激光档表参数原文(温补相机参数)
    nlohmann::json tempTablesRaw() const;            // 相机四表原文
    Scanner::Result readyForScan(ReadyReport& out) const;
    void clear();
    std::string lastPath() const;            // 锁内返回拷贝（write/load 并发改写安全）
private:
    mutable std::mutex mtx_;
    nlohmann::json doc_;                   // 载荷+meta 全量内存态
    bool hasData_ = false;
    std::string lastPath_;
};

} // namespace Scanner::data

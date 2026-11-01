#pragma once
// ============================================================================
// artifacts.h — 中间产物落盘/再载入（PNG/CSV/PLY/JSON）
// ============================================================================
// harness 层统一负责（算子不动）。3D 用 ASCII PLY；点表 CSV；掩膜叠加 PNG。
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace scr {

// —— PNG（imgcodecs）——
bool savePng(const std::string& path, const cv::Mat& img);
/// 掩膜叠加可视化：灰度底转彩＋掩膜像素着色
bool saveMaskOverlayPng(const std::string& path, const cv::Mat& gray,
                        const cv::Mat& mask, cv::Scalar color);

// —— CSV 点表 ——
bool saveCsvP2f(const std::string& path, const std::vector<cv::Point2f>& pts);
bool saveCsvP2d(const std::string& path, const std::vector<cv::Point2d>& pts);
bool loadCsvP2f(const std::string& path, std::vector<cv::Point2f>& pts);

/// EdgePoint CSV：x,y,angle,amplitude,pixelX,pixelY
struct EdgePointCsv { double x,y,angle,amplitude; int pixelX,pixelY; };
bool saveCsvEdgePoints(const std::string& path, const std::vector<EdgePointCsv>& pts);

// —— GpuMat 点集 CSV（下载→写；读→上传）——
bool saveGpuMatP2(const std::string& path, const cv::cuda::GpuMat& d_pts,
                  cv::cuda::Stream& stream);
bool loadGpuMatP2(const std::string& path, cv::cuda::GpuMat& d_pts,
                  cv::cuda::Stream& stream);

// —— PLY（ASCII，仅 x y z；可导入点云工具人工核查）——
bool savePlyXYZf(const std::string& path, const std::vector<cv::Point3f>& pts);
bool savePlyXYZd(const std::string& path, const std::vector<cv::Point3d>& pts);

// —— 表格追加（timing/report）——
void appendCsvRow(const std::string& path, const std::string& headerOnce,
                  const std::string& row);
/// 首行表头（文件不存在时写）
void ensureCsvHeader(const std::string& path, const std::string& header);

// —— JSON ——
bool saveJson(const std::string& path, const nlohmann::json& j);

} // namespace scr

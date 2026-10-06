#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace Scanner::data::fileio {

// ============================================================================
// 点云导入/导出
// ============================================================================

// 导入点云（自动识别格式：.ply .xyz .txt .pcd）
bool importPointCloud(const std::string& filepath,
                      std::vector<cv::Point3f>& points,
                      std::vector<cv::Point3f>* normals = nullptr);
bool exportPointCloud(const std::string& filepath,
                      const std::vector<cv::Point3f>& points,
                      const std::vector<cv::Point3f>* normals = nullptr);

// 分格式函数
bool importXYZ(const std::string& filepath, std::vector<cv::Point3f>& points, std::vector<cv::Point3f>* normals = nullptr);
bool exportXYZ(const std::string& filepath, const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>* normals = nullptr);
bool importPLY(const std::string& filepath, std::vector<cv::Point3f>& points, std::vector<cv::Point3f>* normals = nullptr);
bool exportPLY(const std::string& filepath, const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>* normals = nullptr);
bool importPCD(const std::string& filepath, std::vector<cv::Point3f>& points, std::vector<cv::Point3f>* normals = nullptr);
bool exportPCD(const std::string& filepath, const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>* normals = nullptr);

// ============================================================================
// 网格导入/导出
// ============================================================================

struct MeshData {
    std::vector<cv::Point3f> vertices;
    std::vector<unsigned int> indices;  // 三角形索引，每3个一组
    std::vector<cv::Point3f> normals;     // 逐顶点法线（可选）
};

bool importMesh(std::string filepath, MeshData& mesh);
bool exportMesh(const std::string& filepath, const MeshData& mesh);
bool importSTL(const std::string& filepath, MeshData& mesh);
bool exportSTL(const std::string& filepath, const MeshData& mesh);
bool importOBJ(const std::string& filepath, MeshData& mesh);
bool exportOBJ(const std::string& filepath, const MeshData& mesh);

// ============================================================================
// 标志点导入/导出
// ============================================================================

bool importMarkers(const std::string& filepath, std::vector<cv::Point3f>& markers);
bool exportMarkers(const std::string& filepath, const std::vector<cv::Point3f>& markers);

} // namespace Scanner::data::fileio

#pragma once

#include <osg/Vec3>
#include <string>
#include <vector>

namespace file_io {

// ============================================================================
// 点云导入/导出
// ============================================================================

// 导入点云（自动识别格式：.ply .xyz .txt .pcd）
bool importPointCloud(const std::string& filepath,
                     std::vector<osg::Vec3>& points,
                     std::vector<osg::Vec3>* normals = nullptr);
bool exportPointCloud(const std::string& filepath,
                      const std::vector<osg::Vec3>& points,
                      const std::vector<osg::Vec3>* normals = nullptr);

// 分格式函数
bool importXYZ(const std::string& filepath, std::vector<osg::Vec3>& points, std::vector<osg::Vec3>* normals = nullptr);
bool exportXYZ(const std::string& filepath, const std::vector<osg::Vec3>& points, const std::vector<osg::Vec3>* normals = nullptr);
bool importPLY(const std::string& filepath, std::vector<osg::Vec3>& points, std::vector<osg::Vec3>* normals = nullptr);
bool exportPLY(const std::string& filepath, const std::vector<osg::Vec3>& points, const std::vector<osg::Vec3>* normals = nullptr);
bool importPCD(const std::string& filepath, std::vector<osg::Vec3>& points, std::vector<osg::Vec3>* normals = nullptr);
bool exportPCD(const std::string& filepath, const std::vector<osg::Vec3>& points, const std::vector<osg::Vec3>* normals = nullptr);

// ============================================================================
// 网格导入/导出
// ============================================================================

struct MeshData {
    std::vector<osg::Vec3> vertices;
    std::vector<unsigned int> indices;  // 三角形索引，每3个一组
    std::vector<osg::Vec3> normals;     // 逐顶点法线（可选）
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

bool importMarkers(const std::string& filepath, std::vector<osg::Vec3>& markers);
bool exportMarkers(const std::string& filepath, const std::vector<osg::Vec3>& markers);

} // namespace file_io

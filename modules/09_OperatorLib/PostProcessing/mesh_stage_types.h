#pragma once
// ============================================================================
// mesh_stage_types.h — 网格四族算子统一数据形态（R1「统一索引网格」落地）
//
// 07 PostProcessPipeline 的 MeshData 是编排侧透传句柄，真形态由 09 网格算子
// 定义（07 头注原话）；本头即该形态：
//   · IndexedMesh   —— 索引网格：xyz + normals + triangles 三个 host 数组，
//                      四族算子间唯一交接物（阶段间零文件落盘、零格式转换）
//   · MarkerHoleRef —— 标志点三元组（位置/法向/半径），补洞算子切割标志点
//                      区域用；来源＝core/marker marker_cloud_fuse_cpu 输出直取
//
// 法线权威源（R2）：stage0（mesh_normal_est）出口起，normals 即唯一权威，
// 随网格流转至下游各阶段，禁止旁路重算或外置翻转。
// ============================================================================
#include <cstdint>
#include <cstddef>
#include <vector>

namespace calib {

/// 索引网格（host 内存，单位 mm）——网格四族算子的输入输出统一形态
struct IndexedMesh {
    std::vector<float> xyz;           // 顶点坐标 (3N)
    std::vector<float> normals;       // 顶点法线 (3N)；stage0 出口起为权威源
    std::vector<uint32_t> triangles;  // 索引三角形 (3M)；网格化阶段前为空

    size_t vertexCount() const { return xyz.size() / 3; }
    size_t triangleCount() const { return triangles.size() / 3; }
    bool hasNormals() const { return normals.size() == xyz.size() && !xyz.empty(); }

    void clear() {
        xyz.clear();
        normals.clear();
        triangles.clear();
    }
};

/// 标志点参考（切割回填用）——与 marker_cloud_fuse_cpu 输出同构
struct MarkerHoleRef {
    float x = 0.0f, y = 0.0f, z = 0.0f;        // 标志点中心（mm）
    float nx = 0.0f, ny = 0.0f, nz = 1.0f;     // 表面法向（单位向量）
    float radius = 0.0f;                        // 标志点物理半径（mm）
};

} // namespace calib

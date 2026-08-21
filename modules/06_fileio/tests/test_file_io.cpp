// ============================================================================
// test_file_io.cpp — file_io 收库契约测试（B-T3）
//
// 用例 = 六格式往返（PLY ASCII / XYZ / PCD / STL / OBJ / Markers JSON）+
// importPointCloud 扩展名自动分派 + 坏文件返回 false 不崩。
// 点类型契约：cv::Point3f（收库解 OSG）；命名空间 Scanner::data::fileio。
//
// 已知格式缺陷（按实际能力断言，不修格式逻辑）：
// - PCD：importPCD 头循环 break 条件 data_type 初值 "ascii" 非空——首行后即
//   跳出，POINTS 行未读到即 false，exportPCD 全头格式读不回；仅首行
//   "POINTS n" 的极简格式可导入。
// - OBJ：exportOBJ 不写法线、importOBJ 不读法线（可选字段）——往返只断言
//   顶点+索引。
// ============================================================================

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "file_io.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fio = Scanner::data::fileio;
namespace fs = std::filesystem;

namespace {

fs::path tempFile(const char* name) {
    return fs::temp_directory_path() / name;
}

void expectPoint3fNear(const cv::Point3f& got, const cv::Point3f& want, float tol = 1e-4f) {
    EXPECT_NEAR(got.x, want.x, tol);
    EXPECT_NEAR(got.y, want.y, tol);
    EXPECT_NEAR(got.z, want.z, tol);
}

} // namespace

// —— 1. PLY ASCII 往返（点+法线）——
TEST(FileIo, PlyAsciiRoundTrip) {
    auto path = tempFile("jmw_t3.ply").string();
    std::vector<cv::Point3f> pts = {{1.5f, 2.5f, 3.5f}, {-4.25f, 5.75f, -6.125f}, {0.f, 100.f, 0.25f}};
    std::vector<cv::Point3f> nrm = {{0.f, 0.f, 1.f}, {0.f, 1.f, 0.f}, {1.f, 0.f, 0.f}};
    ASSERT_TRUE(fio::exportPLY(path, pts, &nrm));

    std::vector<cv::Point3f> out, outN;
    ASSERT_TRUE(fio::importPLY(path, out, &outN));
    ASSERT_EQ(out.size(), pts.size());
    ASSERT_EQ(outN.size(), nrm.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        expectPoint3fNear(out[i], pts[i]);
        expectPoint3fNear(outN[i], nrm[i]);
    }
    fs::remove(path);
}

// —— 2. XYZ 往返（点+法线）——
TEST(FileIo, XyzRoundTrip) {
    auto path = tempFile("jmw_t3.xyz").string();
    std::vector<cv::Point3f> pts = {{0.5f, -1.5f, 2.5f}, {10.f, 20.f, 30.f}, {-0.125f, 0.25f, 99.5f}};
    std::vector<cv::Point3f> nrm = {{0.1f, 0.2f, 0.3f}, {-1.f, 0.f, 0.f}, {0.f, 0.5f, -0.5f}};
    ASSERT_TRUE(fio::exportXYZ(path, pts, &nrm));

    std::vector<cv::Point3f> out, outN;
    ASSERT_TRUE(fio::importXYZ(path, out, &outN));
    ASSERT_EQ(out.size(), pts.size());
    ASSERT_EQ(outN.size(), nrm.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        expectPoint3fNear(out[i], pts[i]);
        expectPoint3fNear(outN[i], nrm[i]);
    }
    fs::remove(path);
}

// —— 3. PCD：按实际能力断言（全头格式读不回，见文件头注释缺陷说明）——
TEST(FileIo, PcdActualCapability) {
    auto path = tempFile("jmw_t3.pcd").string();
    std::vector<cv::Point3f> pts = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}, {7.f, 8.f, 9.f}};
    ASSERT_TRUE(fio::exportPCD(path, pts));

    std::vector<cv::Point3f> out;
    EXPECT_FALSE(fio::importPCD(path, out));   // exportPCD 全头格式读不回（缺陷）

    auto minimal = tempFile("jmw_t3_min.pcd").string();   // 实际可读的极简格式
    {
        std::ofstream f(minimal);
        f << "POINTS 2\n0.5 1.5 2.5\n-3.0 -4.0 -5.0\n";
    }
    ASSERT_TRUE(fio::importPCD(minimal, out));
    ASSERT_EQ(out.size(), 2u);
    expectPoint3fNear(out[0], {0.5f, 1.5f, 2.5f}, 1e-5f);
    expectPoint3fNear(out[1], {-3.0f, -4.0f, -5.0f}, 1e-5f);
    fs::remove(path);
    fs::remove(minimal);
}

// —— 4. STL 二进制往返（MeshData）——
TEST(FileIo, StlBinaryRoundTrip) {
    auto path = tempFile("jmw_t3.stl").string();
    fio::MeshData mesh;
    mesh.vertices = {{0.f, 0.f, 0.f}, {10.f, 0.f, 0.f}, {0.f, 10.f, 0.f}, {10.f, 10.f, 0.f}};
    mesh.indices = {0, 1, 2, 1, 3, 2};
    ASSERT_TRUE(fio::exportSTL(path, mesh));

    fio::MeshData out;
    ASSERT_TRUE(fio::importSTL(path, out));
    // STL 无共享顶点：导入按三角形展开，顶点数=索引数、索引顺序 0..N-1
    ASSERT_EQ(out.vertices.size(), mesh.indices.size());
    ASSERT_EQ(out.indices.size(), mesh.indices.size());
    ASSERT_EQ(out.normals.size(), mesh.indices.size());
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        for (int k = 0; k < 3; ++k) {
            expectPoint3fNear(out.vertices[t + k], mesh.vertices[mesh.indices[t + k]], 1e-6f);
            EXPECT_EQ(out.indices[t + k], t + k);
        }
    }
    fs::remove(path);
}

// —— 5. OBJ 往返（MeshData；法线可选不往返，见文件头注释）——
TEST(FileIo, ObjRoundTrip) {
    auto path = tempFile("jmw_t3.obj").string();
    fio::MeshData mesh;
    mesh.vertices = {{0.f, 0.f, 0.f}, {5.f, 0.f, 0.f}, {0.f, 5.f, 0.f}, {5.f, 5.f, 1.f}};
    mesh.indices = {0, 1, 2, 1, 3, 2};
    ASSERT_TRUE(fio::exportOBJ(path, mesh));

    fio::MeshData out;
    ASSERT_TRUE(fio::importOBJ(path, out));
    ASSERT_EQ(out.vertices.size(), mesh.vertices.size());
    ASSERT_EQ(out.indices.size(), mesh.indices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i)
        expectPoint3fNear(out.vertices[i], mesh.vertices[i]);
    for (size_t i = 0; i < mesh.indices.size(); ++i)
        EXPECT_EQ(out.indices[i], mesh.indices[i]);
    EXPECT_TRUE(out.normals.empty());
    fs::remove(path);
}

// —— 6. Markers JSON 往返 ——
TEST(FileIo, MarkersJsonRoundTrip) {
    auto path = tempFile("jmw_t3_markers.json").string();
    std::vector<cv::Point3f> markers = {{1.25f, -2.5f, 3.75f}, {-10.f, 0.f, 99.5f}};
    ASSERT_TRUE(fio::exportMarkers(path, markers));

    std::vector<cv::Point3f> out;
    ASSERT_TRUE(fio::importMarkers(path, out));
    ASSERT_EQ(out.size(), markers.size());
    for (size_t i = 0; i < markers.size(); ++i)
        expectPoint3fNear(out[i], markers[i]);
    fs::remove(path);
}

// —— 7. importPointCloud 扩展名自动分派（.xyz → importXYZ）——
TEST(FileIo, AutoDetectImport) {
    auto path = tempFile("jmw_t3_auto.xyz").string();
    std::vector<cv::Point3f> pts = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    ASSERT_TRUE(fio::exportXYZ(path, pts));

    std::vector<cv::Point3f> out;
    ASSERT_TRUE(fio::importPointCloud(path, out));
    ASSERT_EQ(out.size(), 2u);
    expectPoint3fNear(out[1], {4.f, 5.f, 6.f});
    fs::remove(path);
}

// —— 8. 坏文件：不存在路径 false；垃圾内容 .ply false 不崩 ——
TEST(FileIo, BadFileReturnsFalse) {
    auto missing = (fs::temp_directory_path() / "jmw_t3_no_such_dir" / "f.bin").string();
    std::vector<cv::Point3f> pts;
    EXPECT_FALSE(fio::importPLY(missing, pts));
    EXPECT_FALSE(fio::importXYZ(missing, pts));
    EXPECT_FALSE(fio::importPCD(missing, pts));
    EXPECT_FALSE(fio::importPointCloud(missing, pts));
    fio::MeshData mesh;
    EXPECT_FALSE(fio::importSTL(missing, mesh));
    EXPECT_FALSE(fio::importOBJ(missing, mesh));
    std::vector<cv::Point3f> markers;
    EXPECT_FALSE(fio::importMarkers(missing, markers));

    auto garbage = tempFile("jmw_t3_garbage.ply").string();
    {
        std::ofstream f(garbage);
        f << "this is not a ply file\njust garbage\n";
    }
    EXPECT_FALSE(fio::importPLY(garbage, pts));
    fs::remove(garbage);
}

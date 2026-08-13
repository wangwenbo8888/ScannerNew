#include "file_io.h"

#include <osg/Vec3>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cmath>

namespace file_io {

// ============================================================================
// 点云
// ============================================================================

bool importPointCloud(const std::string& filepath,
                     std::vector<osg::Vec3>& points,
                     std::vector<osg::Vec3>* normals)
{
    if (filepath.empty()) return false;
    // 根据扩展名分派
    size_t dotPos = filepath.find_last_of('.');
    std::string ext = (dotPos != std::string::npos) ? filepath.substr(dotPos + 1) : "";
    if (ext == "ply") return importPLY(filepath, points, normals);
    if (ext == "xyz" || ext == "txt") return importXYZ(filepath, points, normals);
    if (ext == "pcd") return importPCD(filepath, points, normals);
    // 默认按 xyz 处理
    return importXYZ(filepath, points, normals);
}

bool exportPointCloud(const std::string& filepath,
                      const std::vector<osg::Vec3>& points,
                      const std::vector<osg::Vec3>* normals)
{
    size_t dotPos = filepath.find_last_of('.');
    auto ext = (dotPos != std::string::npos) ? filepath.substr(dotPos + 1) : std::string();
    if (ext == "ply") return exportPLY(filepath, points, normals);
    if (ext == "pcd") return exportPCD(filepath, points, normals);
    // 默认 xyz
    return exportXYZ(filepath, points, normals);
}

// --- XYZ/TXT ---
bool importXYZ(const std::string& filepath,
               std::vector<osg::Vec3>& points,
               std::vector<osg::Vec3>* normals)
{
    FILE* f = fopen(filepath.c_str(), "r");
    if (!f) return false;
    points.clear();
    if (normals) normals->clear();
    points.reserve(100000);
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        float x, y, z;
        int n = sscanf(line, "%f %f %f", &x, &y, &z);
        if (n >= 3) {
            points.push_back(osg::Vec3(x, y, z));
            if (normals) {
                float nx, ny, nz;
                if (sscanf(line, "%*f %*f %*f %f %f %f", &nx, &ny, &nz) == 3)
                    normals->push_back(osg::Vec3(nx, ny, nz));
            }
        }
    }
    fclose(f);
    return !points.empty();
}

bool exportXYZ(const std::string& filepath,
               const std::vector<osg::Vec3>& points,
               const std::vector<osg::Vec3>* normals)
{
    std::ofstream f(filepath);
    if (!f.is_open()) return false;
    f << std::fixed;
    f.precision(5);
    for (size_t i = 0; i < points.size(); ++i) {
        f << points[i].x() << " " << points[i].y() << " " << points[i].z();
        if (normals && i < normals->size())
            f << " " << (*normals)[i].x() << " " << (*normals)[i].y() << " " << (*normals)[i].z();
        f << "\n";
    }
    return true;
}

// --- PLY ---
bool importPLY(const std::string& filepath,
               std::vector<osg::Vec3>& points,
               std::vector<osg::Vec3>* normals)
{
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) return false;
    std::string line;
    int vertexCount = 0;
    bool hasNormal = false;
    bool binary = false;
    int vertexStride = 0;
    bool inVertexSection = false;

    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.find("format binary") != std::string::npos) binary = true;
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &vertexCount);
            inVertexSection = true;
        } else if (line.find("element ") == 0) {
            inVertexSection = false;
        }
        if (line.find("property float nx") != std::string::npos) hasNormal = true;
        if (inVertexSection && line.find("property ") == 0 && line.find("list") == std::string::npos) {
            if (line.find("char") != std::string::npos)       vertexStride += 1;
            else if (line.find("short") != std::string::npos)  vertexStride += 2;
            else if (line.find("float") != std::string::npos)  vertexStride += 4;
            else if (line.find("double") != std::string::npos) vertexStride += 8;
            else if (line.find("int") != std::string::npos)    vertexStride += 4;
        }
        if (line == "end_header") break;
    }

    if (vertexCount <= 0) return false;
    points.clear();
    if (normals) normals->clear();
    points.reserve(vertexCount);
    if (normals) normals->reserve(vertexCount);

    if (binary) {
        int skipBytes = vertexStride - 12;
        if (hasNormal) skipBytes -= 12;
        if (skipBytes < 0) skipBytes = 0;
        for (int i = 0; i < vertexCount; ++i) {
            float xyz[3];
            f.read(reinterpret_cast<char*>(xyz), 12);
            points.emplace_back(xyz[0], xyz[1], xyz[2]);
            if (hasNormal) {
                float n[3];
                f.read(reinterpret_cast<char*>(n), 12);
                if (normals) normals->emplace_back(n[0], n[1], n[2]);
            }
            if (skipBytes > 0) f.ignore(skipBytes);
        }
    } else {
        for (int i = 0; i < vertexCount; ++i) {
            float x, y, z;
            if (!(f >> x >> y >> z)) break;
            points.emplace_back(x, y, z);
            if (hasNormal && normals) {
                float nx, ny, nz;
                f >> nx >> ny >> nz;
                normals->emplace_back(nx, ny, nz);
            }
            f.ignore(256, '\n');
        }
    }
    return !points.empty();
}

bool exportPLY(const std::string& filepath,
               const std::vector<osg::Vec3>& points,
               const std::vector<osg::Vec3>* normals)
{
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) return false;
    bool hasN = normals && normals->size() == points.size();
    // header
    f << "ply\n";
    f << "format ascii 1.0\n";
    f << "element vertex " << points.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    if (hasN)
        f << "property float nx\nproperty float ny\nproperty float nz\n";
    f << "end_header\n";
    // data
    f << std::fixed;
    f.precision(5);
    for (size_t i = 0; i < points.size(); ++i) {
        f << points[i].x() << " " << points[i].y() << " " << points[i].z();
        if (hasN)
            f << " " << (*normals)[i].x() << " " << (*normals)[i].y() << " " << (*normals)[i].z();
        f << "\n";
    }
    return true;
}

// --- PCD ---
bool importPCD(const std::string& filepath,
               std::vector<osg::Vec3>& points,
               std::vector<osg::Vec3>* normals)
{
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) return false;
    std::string line;
    int points_count = 0;
    std::string data_type = "ascii";

    while (std::getline(f, line)) {
        if (line.find("POINTS") != std::string::npos)
            sscanf(line.c_str(), "POINTS %d", &points_count);
        if (line.find("DATA") != std::string::npos)
            data_type = line.substr(5);
        if (line == "DATA" || data_type != "") break;
    }

    if (points_count <= 0) return false;
    points.clear();
    points.reserve(points_count);

    if (data_type == "binary") {
        for (int i = 0; i < points_count; ++i) {
            float xyz[3];
            f.read(reinterpret_cast<char*>(xyz), 12);
            points.emplace_back(xyz[0], xyz[1], xyz[2]);
        }
    } else {
        for (int i = 0; i < points_count; ++i) {
            float x, y, z;
            if (!(f >> x >> y >> z)) break;
            points.emplace_back(x, y, z);
        }
    }
    return !points.empty();
}

bool exportPCD(const std::string& filepath,
               const std::vector<osg::Vec3>& points,
               const std::vector<osg::Vec3>* normals)
{
    std::ofstream f(filepath);
    if (!f.is_open()) return false;
    f << "# .PCD v0.7 - Point Cloud Data file format\n";
    f << "VERSION 0.7\n";
    f << "FIELDS x y z\n";
    f << "SIZE 4 4 4\n";
    f << "TYPE F F F\n";
    f << "COUNT 1 1 1\n";
    f << "WIDTH " << points.size() << "\n";
    f << "HEIGHT 1\n";
    f << "VIEWPOINT 0 0 0 1 0 0 0\n";
    f << "POINTS " << points.size() << "\n";
    f << "DATA ascii\n";
    f << std::fixed;
    f.precision(5);
    for (const auto& p : points)
        f << p.x() << " " << p.y() << " " << p.z() << "\n";
    return true;
}

// ============================================================================
// 网格
// ============================================================================

bool importMesh(std::string filepath, MeshData& mesh)
{
    size_t dotPos = filepath.find_last_of('.');
    auto ext = (dotPos != std::string::npos) ? filepath.substr(dotPos + 1) : std::string();
    if (ext == "stl") return importSTL(filepath, mesh);
    if (ext == "obj") return importOBJ(filepath, mesh);
    return false;
}

bool exportMesh(const std::string& filepath, const MeshData& mesh)
{
    size_t dotPos = filepath.find_last_of('.');
    auto ext = (dotPos != std::string::npos) ? filepath.substr(dotPos + 1) : std::string();
    if (ext == "stl") return exportSTL(filepath, mesh);
    if (ext == "obj") return exportOBJ(filepath, mesh);
    return false;
}

// --- STL (auto-detect ASCII or binary) ---
bool importSTL(const std::string& filepath, MeshData& mesh)
{
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) return false;

    // 读前 512 字节判断 ASCII/Binary
    char buf[512] = {0};
    size_t nread = fread(buf, 1, 512, f);
    fclose(f);

    bool isAscii = false;
    // ASCII STL 以 "solid" 开头
    if (nread >= 5 && (strncmp(buf, "solid", 5) == 0)) {
        // 进一步检查：ASCII STL 通常含 "facet" 关键字
        if (strstr(buf, "facet") || strstr(buf, "normal"))
            isAscii = true;
    }

    if (isAscii)
    {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) return false;
        mesh.vertices.clear();
        mesh.indices.clear();
        mesh.normals.clear();
        osg::Vec3 curN(0, 0, 1);
        std::string tok;
        while (ifs >> tok) {
            if (tok == "facet") {
                std::string nkw; float nx, ny, nz;
                ifs >> nkw >> nx >> ny >> nz;
                curN.set(nx, ny, nz);
            } else if (tok == "vertex") {
                float x, y, z;
                ifs >> x >> y >> z;
                unsigned int idx = (unsigned int)mesh.vertices.size();
                mesh.vertices.emplace_back(x, y, z);
                mesh.indices.push_back(idx);
                mesh.normals.push_back(curN);
            }
        }
        return !mesh.vertices.empty();
    }

    // Binary STL
    f = fopen(filepath.c_str(), "rb");
    if (!f) return false;

    char header[80];
    if (fread(header, 1, 80, f) != 80) { fclose(f); return false; }

    unsigned int numTris = 0;
    if (fread(&numTris, 4, 1, f) != 1) { fclose(f); return false; }
    if (numTris == 0 || numTris > 100000000) { fclose(f); return false; }

    mesh.vertices.clear();
    mesh.indices.clear();
    mesh.normals.clear();
    mesh.vertices.reserve(numTris * 3);
    mesh.indices.reserve(numTris * 3);
    mesh.normals.reserve(numTris * 3);

    for (unsigned int i = 0; i < numTris; ++i) {
        float n[3], v[9];
        unsigned short attr;
        if (fread(n, 4, 3, f) != 3) break;
        if (fread(v, 4, 9, f) != 9) break;
        if (fread(&attr, 2, 1, f) != 1) break;

        unsigned int base = (unsigned int)mesh.vertices.size();
        mesh.vertices.emplace_back(v[0], v[1], v[2]);
        mesh.vertices.emplace_back(v[3], v[4], v[5]);
        mesh.vertices.emplace_back(v[6], v[7], v[8]);
        mesh.indices.push_back(base);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.normals.emplace_back(n[0], n[1], n[2]);
        mesh.normals.emplace_back(n[0], n[1], n[2]);
        mesh.normals.emplace_back(n[0], n[1], n[2]);
    }
    fclose(f);
    return !mesh.vertices.empty();
}

bool exportSTL(const std::string& filepath, const MeshData& mesh)
{
    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) return false;

    char header[80] = {0};
    memcpy(header, "binary STL", 10);
    fwrite(header, 1, 80, f);

    unsigned int numTris = (unsigned int)(mesh.indices.size() / 3);
    fwrite(&numTris, 4, 1, f);

    for (unsigned int i = 0; i < numTris; ++i) {
        const auto& v0 = mesh.vertices[mesh.indices[i * 3]];
        const auto& v1 = mesh.vertices[mesh.indices[i * 3 + 1]];
        const auto& v2 = mesh.vertices[mesh.indices[i * 3 + 2]];

        // 面法线
        osg::Vec3 e1 = v1 - v0;
        osg::Vec3 e2 = v2 - v0;
        osg::Vec3 n = e1 ^ e2;
        n.normalize();
        float fn[3] = { n.x(), n.y(), n.z() };
        fwrite(fn, 4, 3, f);

        float v[9] = {
            v0.x(), v0.y(), v0.z(),
            v1.x(), v1.y(), v1.z(),
            v2.x(), v2.y(), v2.z()
        };
        fwrite(v, 4, 9, f);

        unsigned short attr = 0;
        fwrite(&attr, 2, 1, f);
    }
    fclose(f);
    return true;
}

// --- OBJ ---
bool importOBJ(const std::string& filepath, MeshData& mesh)
{
    std::ifstream f(filepath);
    if (!f.is_open()) return false;
    mesh.vertices.clear();
    mesh.indices.clear();

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string type;
        ss >> type;
        if (type == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            mesh.vertices.emplace_back(x, y, z);
        } else if (type == "f") {
            int idx[3];
            for (int k = 0; k < 3; ++k) {
                std::string token;
                ss >> token;
                idx[k] = std::stoi(token) - 1;  // OBJ 从1开始
                if (idx[k] < 0) idx[k] += (int)mesh.vertices.size() + 1;  // 负索引
            }
            mesh.indices.push_back((unsigned int)idx[0]);
            mesh.indices.push_back((unsigned int)idx[1]);
            mesh.indices.push_back((unsigned int)idx[2]);
        }
    }
    return !mesh.vertices.empty();
}

bool exportOBJ(const std::string& filepath, const MeshData& mesh)
{
    std::ofstream f(filepath);
    if (!f.is_open()) return false;
    f << std::fixed;
    f.precision(5);
    for (const auto& v : mesh.vertices)
        f << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    for (size_t i = 0; i < mesh.indices.size(); i += 3)
        f << "f " << mesh.indices[i] + 1 << " " << mesh.indices[i + 1] + 1 << " " << mesh.indices[i + 2] + 1 << "\n";
    return true;
}

// ============================================================================
// 标志点 (JSON)
// ============================================================================

bool importMarkers(const std::string& filepath, std::vector<osg::Vec3>& markers)
{
    std::ifstream f(filepath);
    if (!f.is_open()) return false;
    markers.clear();
    std::string line;
    while (std::getline(f, line)) {
        // 兼容简单 JSON 数组格式: [x, y, z] 或 x y z
        float x, y, z;
        // 去掉 [ ] ,
        std::string clean;
        for (char c : line) {
            if (c != '[' && c != ']' && c != ',') clean += c;
            else clean += ' ';
        }
        std::istringstream ss(clean);
        if (ss >> x >> y >> z)
            markers.emplace_back(x, y, z);
    }
    return !markers.empty();
}

bool exportMarkers(const std::string& filepath, const std::vector<osg::Vec3>& markers)
{
    std::ofstream f(filepath);
    if (!f.is_open()) return false;
    f << "[\n";
    f << std::fixed;
    f.precision(5);
    for (size_t i = 0; i < markers.size(); ++i) {
        f << "  [" << markers[i].x() << ", " << markers[i].y() << ", " << markers[i].z() << "]";
        if (i + 1 < markers.size()) f << ",";
        f << "\n";
    }
    f << "]\n";
    return true;
}

} // namespace file_io

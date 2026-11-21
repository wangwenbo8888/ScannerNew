#pragma once
// [port-261121] stl.hpp trimmed: Mesh data layout only (IO decls removed -
// the operator adapter feeds Mesh in memory; no file layer).
#include <cstdint>
#include <vector>

namespace sr {

struct Mesh {
    std::vector<double> verts;
    std::vector<uint32_t> triangles;
    size_t numVerts = 0;
    size_t numTris = 0;
    // Optional per-vertex normals carried natively (index-aligned reference).
    // Empty when absent. Used as the FIRST-choice orientation reference.
    std::vector<double> vnormals;
};

} // namespace sr
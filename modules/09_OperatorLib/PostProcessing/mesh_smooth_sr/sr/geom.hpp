#pragma once
#include <cstdint>
#include <vector>

namespace sr {


struct Topology {
    std::vector<std::vector<uint32_t>> adj;
    std::vector<std::vector<std::vector<uint32_t>>> rings;
};

Topology buildTopology(const std::vector<uint32_t>& triangles, size_t numVerts, int maxRing);
std::vector<double> vertexNormals(const std::vector<double>& verts,
                                  const std::vector<uint32_t>& triangles, size_t numVerts);
void umbrellaNormalPass(std::vector<double>& nrm, const Topology& topo, double lambda);
void tangentFrame(double nx, double ny, double nz, double out[9]);
bool fitParaboloid(const std::vector<double>& pts2d, double coeff[6]);
double fitResidual(const double coeff[6], const std::vector<double>& pts2d);

} // namespace sr

#pragma once
// [port-261121] pipeline.hpp trimmed: smooth3-chain declarations only.
#include <vector>
#include <cstdint>

#include "geom.hpp"
#include "stl.hpp"
#include "edge_table.hpp"

namespace sr {

std::vector<double> runSmoothV3(const Mesh& mesh, double tolDeg1, double tolDeg2, double sigma,
                                int maxRing, int outerIters, bool parallel = false);
std::vector<uint32_t> optimizeEdges(const std::vector<double>& pos, const std::vector<uint32_t>& triIn,
                                    const std::vector<double>& origVerts,
                                    const std::vector<double>& origNrm, long long* flipCount,
                                    double coplanarDeg, int nm_guard, EdgeTable* tab = nullptr);

} // namespace sr
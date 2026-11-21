#pragma once
#include "mesh_types.h"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace hole_filler_pro {
NeighborhoodInfo extractNeighborhood(const BoundaryLoop& loop,
                                     const Eigen::MatrixXd& V,
                                     const Eigen::MatrixXi& F,
                                     int rings,
                                     const std::unordered_map<int, double>& global_mean_edge,
                                     const std::unordered_map<int, std::vector<int>>& global_v2t);
}

#pragma once
#include "mesh_types.h"
#include "mesh_entry_cleaner.h"
#include "utils.h"
#include <Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hole_filler_pro {

struct EntryCleanReport;   // mesh_entry_cleaner.h

// [port-261121] fillHolesFile（Open3D 文件壳层）已删——内存路径见 fillHoles。

// `clean_report` (optional): receives the S1 entry-clean counters of this run
// (passthrough of EntryCleanResult.report). Untouched when entry_clean==false.
// `watertight_fail` (optional): set to 1 when the S3 gate fires (boundary
// loops INCREASED after filling: post > pre). The gate is a unified-exit
// wrapper, so it covers every return path including the no-holes fast path
// (pre=post=0, passes trivially).
bool fillHoles(const Eigen::MatrixXd& vertices,
               const Eigen::MatrixXi& triangles,
               Eigen::MatrixXd& out_vertices,
               Eigen::MatrixXi& out_triangles,
               const FillParams& params = {},
               EntryCleanReport* clean_report = nullptr,
               int* watertight_fail = nullptr);

std::vector<BoundaryLoop> detectBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

FillResult fillSingleHole(const BoundaryLoop& loop,
                           const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F,
                           const FillParams& params,
                           const std::unordered_map<int, double>& global_mean_edge,
                           const std::unordered_map<int, std::vector<int>>& global_v2t,
                           const std::function<bool(int, int)>& edge_occupied = {},
                           const std::unordered_set<int>* guide_boundary_verts = nullptr);

} // namespace hole_filler_pro

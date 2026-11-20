// ============================================================================
// tsdf_shell_extract.cpp — TSDF+MC 抽薄壳 CLI（备用融合算子的命令行入口）
// ============================================================================
// 用法:
//   tsdf_shell_extract --in <cloud.ply> [--out <shell.ply>] [--stats <stats.json>]
//                      [--grid 0.5] [--trunc 2.0] [--step 2] [--knn 16]
// 输入: 点云 PLY（xyz; 可选 nx/ny/nz——有则用已知法线, 无则 kNN-PCA 自估）
// 输出: 薄壳网格 PLY + 统计 JSON（含"输入点→壳"距离: 量化抽壳效果）
// 全 CPU, 无显卡依赖。参数缺省 = TsdfShellFuseParams 默认（ZG 证据常数）。
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "tsdf_shell_fuse.h"
#include "tsdf_shell_ply.h"

static void usage() {
    std::printf(
        "usage: tsdf_shell_extract --in <cloud.ply> [--out <shell.ply>] "
        "[--stats <stats.json>] [--grid 0.5] [--trunc 2.0] [--step 2] [--knn 16]\n");
}

int main(int argc, char** argv) {
    std::string inPath, outPath, statsPath;
    calib::TsdfShellFuseParams p;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--in") inPath = next("--in");
        else if (a == "--out") outPath = next("--out");
        else if (a == "--stats") statsPath = next("--stats");
        else if (a == "--grid") p.voxelSize = (float)atof(next("--grid").c_str());
        else if (a == "--trunc") p.truncFactor = (float)atof(next("--trunc").c_str());
        else if (a == "--step") p.mcCornerStep = atoi(next("--step").c_str());
        else if (a == "--knn") p.normalKnn = atoi(next("--knn").c_str());
        else { usage(); return 2; }
    }
    if (inPath.empty()) { usage(); return 2; }
    if (outPath.empty()) outPath = inPath + ".shell.ply";
    if (statsPath.empty()) statsPath = inPath + ".shell_stats.json";

    std::vector<cv::Point3f> pts, nrm;
    bool hasN = false;
    if (!calib::tsdfply::loadPlyPoints(inPath, pts, nrm, hasN)) {
        std::fprintf(stderr, "failed to load PLY: %s\n", inPath.c_str());
        return 1;
    }
    std::printf("loaded %zu points (normals: %s)\n", pts.size(), hasN ? "yes" : "no, estimating");
    p.validate();

    calib::TsdfShellFuseCpu fuse(p);
    const bool ok = hasN ? fuse.addCloudWithNormals(pts, nrm)
                         : fuse.addCloudAutoNormals(pts);
    if (!ok) {
        std::fprintf(stderr, "fusion failed\n");
        return 1;
    }
    const auto mesh = fuse.extractMesh();
    if (mesh.triangleCount == 0) {
        std::fprintf(stderr, "no triangles extracted (grid too coarse / cloud too sparse?)\n");
        return 1;
    }
    auto st0 = fuse.stats();
    st0.cubesCut = mesh.cubesCut;
    st0.triangleCount = mesh.triangleCount;
    st0.vertexCount = mesh.vertexCount;
    st0.voxelCount = mesh.voxelCount;
    const auto st = calib::computeCloudToMeshStats(mesh, pts, st0, p.voxelSize);
    if (!calib::tsdfply::savePlyMesh(outPath, mesh)) {
        std::fprintf(stderr, "failed to write %s\n", outPath.c_str());
        return 1;
    }
    FILE* sf = std::fopen(statsPath.c_str(), "wb");
    if (sf) {
        std::fputs(st.toJson().dump(2).c_str(), sf);
        std::fclose(sf);
    }
    std::printf("shell: %d tris / %d verts / %d voxels (cubes cut %d)\n",
                mesh.triangleCount, mesh.vertexCount, mesh.voxelCount, mesh.cubesCut);
    std::printf("point->mesh: mean=%.3f p95=%.3f max=%.3f mm  -> %s\n",
                st.meanPointToMesh, st.p95PointToMesh, st.maxPointToMesh, outPath.c_str());
    return 0;
}

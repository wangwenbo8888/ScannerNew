// gba_dataset_runner: 把 marker_sim 生成的数据集喂给 GlobalBundleAdjustmentCPU 运算。
//
// 数据集契约(由 ceshi/标记点案例模拟器 产出):
//   <dir>/poses_ground_truth.txt  每帧一行 16 浮点(行优先 4x4, 相机->世界, mm, 无漂移真值)
//   <dir>/frames/frame_NNNNNN.txt 每行 "id x y z"(相机系观测 + 全局ID)
//
// 初值位姿 R_init/t_init 用 GT 位姿 + 可选扰动(--init_noise_deg/--init_noise_mm),
// 帧 0 恒为 identity 作锚点。扰动模式验证 GBA 的位姿恢复能力。
#include "global_ba_cpu.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace calib;

static bool LoadPose(const std::string& line, cv::Matx33d& R, cv::Vec3d& t) {
    std::istringstream ss(line);
    double v[16];
    for (int i = 0; i < 16; ++i)
        if (!(ss >> v[i])) return false;
    // 行优先 4x4: R = 左上 3x3, t = 右上 3x1(列 3/7/11)
    R = cv::Matx33d(v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]);
    t = cv::Vec3d(v[3], v[7], v[11]);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: gba_dataset_runner <dataset_dir> [--init_noise_deg d] "
                     "[--init_noise_mm t] [--seed s] [--pgo 0|1] [--out <corrected_txt>]\n";
        return 2;
    }
    std::string dir = argv[1];
    double noiseDeg = 0.0, noiseMm = 0.0;
    uint64_t seed = 42;
    int pgo = 1;
    std::string outPath;   // 修正后合并点云输出路径; 空则写 <dir>/global_map_corrected.txt
    bool excludeOutliers = false;   // --exclude_outliers: 剔除 GBA 标记的外点
    int centerOrigin = 1;           // --center_origin 0|1(默认开)

    auto next = [&](int& i, const char* tag) -> std::string {
        if (i + 1 >= argc) { std::cerr << "missing value for " << tag << "\n"; std::exit(2); }
        return std::string(argv[++i]);
    };
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--init_noise_deg") noiseDeg = std::stod(next(i, "deg"));
        else if (a == "--init_noise_mm") noiseMm = std::stod(next(i, "mm"));
        else if (a == "--seed") seed = std::stoull(next(i, "seed"));
        else if (a == "--pgo") pgo = std::stoi(next(i, "pgo"));
        else if (a == "--out") outPath = next(i, "out");
        else if (a == "--exclude_outliers") excludeOutliers = true;
        else if (a == "--center_origin") centerOrigin = std::stoi(next(i, "center_origin"));
        else { std::cerr << "unknown arg: " << a << "\n"; return 2; }
    }

    // 1) 读 poses_ground_truth.txt → GT 绝对位姿
    std::vector<cv::Matx33d> Rgt;
    std::vector<cv::Vec3d> tgt;
    {
        std::ifstream f(dir + "/poses_ground_truth.txt");
        if (!f) { std::cerr << "cannot open poses_ground_truth.txt in " << dir << "\n"; return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            cv::Matx33d R;
            cv::Vec3d t;
            if (LoadPose(line, R, t)) { Rgt.push_back(R); tgt.push_back(t); }
        }
    }
    const int nFrames = static_cast<int>(Rgt.size());
    if (nFrames == 0) { std::cerr << "no poses loaded\n"; return 1; }

    // 2) 读 frames/frame_NNNNNN.txt → markerObs; 构造初值位姿(GT + 可选扰动)
    GlobalBAInput in;
    in.frames.resize(nFrames);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> angN(0.0, noiseDeg * CV_PI / 180.0);
    std::normal_distribution<double> tN(0.0, noiseMm);

    for (int i = 0; i < nFrames; ++i) {
        char buf[80];
        std::snprintf(buf, sizeof(buf), "/frames/frame_%06d.txt", i);
        std::ifstream f(dir + buf);
        if (!f) { std::cerr << "cannot open " << dir << buf << "\n"; return 1; }
        GlobalBAFrame& fr = in.frames[i];
        fr.frameId = static_cast<uint64_t>(i);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int id;
            double x, y, z;
            if (ss >> id >> x >> y >> z) fr.markerObs.push_back({cv::Point3d(x, y, z), id});
        }
        // 初值: GT 位姿(含 orbit 等非 identity 起点); 帧0 不加扰动作锚(GBA 会把它设为常量),
        // 其余帧 = GT + 可选扰动。walk 模式 GT frame0 本就是 identity, 行为不变。
        cv::Matx33d R = Rgt[i];
        cv::Vec3d t = tgt[i];
        if (i != 0 && noiseDeg > 0.0) {
            double da = angN(rng), db = angN(rng), dc = angN(rng);
            cv::Matx33d Rx(1, 0, 0, 0, cos(da), -sin(da), 0, sin(da), cos(da));
            cv::Matx33d Ry(cos(db), 0, sin(db), 0, 1, 0, -sin(db), 0, cos(db));
            cv::Matx33d Rz(cos(dc), -sin(dc), 0, sin(dc), cos(dc), 0, 0, 0, 1);
            R = Rz * Ry * Rx * R;
        }
        if (i != 0 && noiseMm > 0.0) t = t + cv::Vec3d(tN(rng), tN(rng), tN(rng));
        fr.R_init = R;
        fr.t_init = t;
    }

    std::set<int> uniqIds;
    long long totalObs = 0;
    for (const auto& fr : in.frames) {
        for (const auto& o : fr.markerObs) uniqIds.insert(o.globalId);
        totalObs += static_cast<long long>(fr.markerObs.size());
    }

    // 3) 运行 GBA(sigmaObserved 与 marker_sim 噪声 0.02mm 一致)
    GlobalBAParams p;
    p.enablePoseGraphPreopt = (pgo != 0);
    p.sigmaObserved = 0.02;
    p.tukeyC = 3.0 * 0.02;
    p.centerOrigin = (centerOrigin != 0);
    GlobalBundleAdjustmentCPU op(p);
    auto r = op.Execute(in);

    // 4) 报告
    std::cout << "=== GBA dataset runner ===\n";
    std::cout << "dataset:        " << dir << "\n";
    std::cout << "frames:         " << nFrames
              << "   observations: " << totalObs
              << "   unique markers: " << uniqIds.size() << "\n";
    std::cout << "init noise:     " << noiseDeg << " deg, " << noiseMm << " mm   (seed " << seed << ")\n";
    std::cout << "pgo:            " << (p.enablePoseGraphPreopt ? "on" : "off") << "\n";
    std::cout << "success:        " << (r.success ? "true" : "false") << "   " << r.message << "\n";
    std::cout << "loopDetected:   " << (r.statistics.loopDetected ? "true" : "false") << "\n";
    std::cout << "initialRMSE:    " << r.statistics.initialRMSE << " mm\n";
    std::cout << "finalRMSE:      " << r.statistics.finalRMSE << " mm\n";
    std::cout << "ceresIter:      " << r.statistics.ceresIterations << "\n";
    std::cout << "loopResidual:   " << r.statistics.loopClosureResidual << " mm\n";
    std::cout << "outliers:       " << r.statistics.outlierObsIds.size() << "\n";
    std::cout << "optMarkers:     " << r.optimizedMarkers.size()
              << "   optPoses: " << r.optimizedPoses.size() << "\n";
    if (r.success && r.statistics.initialRMSE > 1e-12)
        std::cout << "recovery ratio: " << (r.statistics.finalRMSE / r.statistics.initialRMSE) << "\n";

    // 5) 全局修正后合并点云: 用优化位姿把每帧观测重投影到世界系, 合并成一个 TXT。
    //    world = R_opt * local + t_opt; 与 global_map_initial.ply(用漂移位姿)对照。
    if (r.success && !r.optimizedPoses.empty()) {
        // frameId -> (R_opt, t_opt)
        std::unordered_map<uint64_t, std::pair<cv::Matx33d, cv::Vec3d>> poseMap;
        for (const auto& fp : r.optimizedPoses)
            poseMap.emplace(fp.frameId, std::make_pair(fp.R, fp.t));

        if (outPath.empty()) outPath = dir + "/global_map_corrected.txt";
        // 外点全局索引集合(GBA outlierObsIds = 输入顺序的帧主序/帧内次序累计编号)
        std::set<int> outlierSet(r.statistics.outlierObsIds.begin(),
                                 r.statistics.outlierObsIds.end());
        // 先收集修正点(world), 再同时写 TXT 与 PLY(便于 MeshLab 查看, 与漂移的 global_map_initial.ply 对照)
        std::vector<cv::Vec3d> corr;
        int obsGlobal = 0;   // 与 GBA outlierObsIds 对齐的全局观测编号
        long long skipped = 0;
        for (const auto& fr : in.frames) {
            auto it = poseMap.find(fr.frameId);
            const cv::Matx33d R = (it != poseMap.end()) ? it->second.first : cv::Matx33d::eye();
            const cv::Vec3d t = (it != poseMap.end()) ? it->second.second : cv::Vec3d(0, 0, 0);
            for (const auto& o : fr.markerObs) {
                bool isOutlier = outlierSet.count(obsGlobal) > 0;
                ++obsGlobal;   // 每个观测都推进编号(无论是否写出), 保持与 GBA 对齐
                if (excludeOutliers && isOutlier) { ++skipped; continue; }
                corr.push_back(R * cv::Vec3d(o.local.x, o.local.y, o.local.z) + t);
            }
        }
        // TXT: id x y z
        std::ofstream of(outPath);
        if (!of) { std::cerr << "cannot write " << outPath << "\n"; return 1; }
        of << "# 全局修正后合并点云(GBA 优化位姿重投影, 单位 mm)\n";
        of << "# 格式: id x y z   (id=globalId, 同一物理点多次观测各占一行, 不去重)\n";
        if (excludeOutliers)
            of << "# (已剔除 GBA 卡方外点 " << outlierSet.size() << " 个)\n";
        of << std::fixed << std::setprecision(6);
        for (size_t k = 0; k < corr.size(); ++k)
            of << k << " " << corr[k](0) << " " << corr[k](1) << " " << corr[k](2) << "\n";
        // PLY: 同目录同名 .ply(便于 MeshLab 直接查看, 应见干净圆柱面)
        std::string plyPath = outPath;
        auto dot = plyPath.rfind('.');
        if (dot != std::string::npos) plyPath = plyPath.substr(0, dot);
        plyPath += ".ply";
        std::ofstream pf(plyPath);
        if (pf) {
            pf << "ply\nformat ascii 1.0\ncomment unit: mm\ncomment GBA-corrected (drift removed)\n";
            pf << "element vertex " << corr.size() << "\n";
            pf << "property float x\nproperty float y\nproperty float z\nend_header\n";
            pf << std::fixed << std::setprecision(6);
            for (const auto& w : corr) pf << w(0) << " " << w(1) << " " << w(2) << "\n";
            std::cout << "correctedCloud: " << outPath << "  + " << plyPath
                      << "   (" << corr.size() << " points";
        } else {
            std::cout << "correctedCloud: " << outPath << "   (" << corr.size() << " points";
        }
        if (excludeOutliers) std::cout << ", excluded " << skipped << " outliers";
        std::cout << ")\n";
    }
    return r.success ? 0 : 1;
}

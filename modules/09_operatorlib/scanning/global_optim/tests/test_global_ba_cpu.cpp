#include "global_ba_cpu.h"
#include "ba_residuals.h"
#include <gtest/gtest.h>
#include <utility>
#include <random>
#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>
#include <ceres/autodiff_cost_function.h>
#include <ceres/numeric_diff_cost_function.h>
#include <ceres/numeric_diff_options.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace calib;

// 辅助:构造 nFrames 帧,每帧 idsPerFrame 个 marker,相邻帧 ID 连续重叠(纯链式)
static GlobalBAInput makeChainInput(int nFrames, int idsPerFrame) {
    GlobalBAInput in;
    for (int i = 0; i < nFrames; ++i) {
        GlobalBAFrame f; f.frameId = static_cast<uint64_t>(i);
        for (int k = 0; k < idsPerFrame; ++k) {
            // globalId = i + k → 相邻帧 ID 段重叠,远端帧无共视
            f.markerObs.push_back({cv::Point3d(static_cast<double>(k), 0.0, static_cast<double>(i)), i + k});
        }
        in.frames.push_back(std::move(f));
    }
    return in;
}

TEST(GlobalBAInputValidation, EmptyFramesReturnsFailure) {
    GlobalBundleAdjustmentCPU op;
    GlobalBAInput in;  // empty
    auto r = op.Execute(in);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.message.find("empty"), std::string::npos);
}

TEST(GlobalBAInputValidation, TooFewPointsReturnsFailure) {
    GlobalBundleAdjustmentCPU op;
    GlobalBAInput in;
    GlobalBAFrame f;
    f.frameId = 0;
    f.markerObs.resize(2);  // < minPointsPerFrame (default 3)
    in.frames.push_back(f);
    auto r = op.Execute(in);
    EXPECT_FALSE(r.success);
}

TEST(GlobalBATopology, PureChainNoLoop) {
    GlobalBundleAdjustmentCPU op;
    auto in = makeChainInput(50, 10);  // 相邻帧重叠,无远端共视
    auto r = op.Execute(in);
    ASSERT_TRUE(r.success);
    EXPECT_FALSE(r.statistics.loopDetected);
}

TEST(GlobalBATopology, LoopDetectedWhenFarFramesCovis) {
    GlobalBundleAdjustmentCPU op;
    auto in = makeChainInput(100, 10);
    // 让首帧(0)与末帧(99)共享 6 个 globalId(>minCovisForLoopEdge=5, |0-99|>30)
    for (int k = 0; k < 6; ++k) {
        in.frames.front().markerObs.push_back({cv::Point3d(100.0 + k, 0.0, 0.0), 9000 + k});
        in.frames.back().markerObs.push_back({cv::Point3d(100.0 + k, 0.0, 99.0), 9000 + k});
    }
    auto r = op.Execute(in);
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.statistics.loopDetected);
}

TEST(GlobalBAResidual, GradientCheckAutoDiff) {
    // 构造已知量,并让残差非零(注入小偏差)
    // 用固定 axis-angle 构造非平凡旋转(确定性,避免 CI 抖动)
    Eigen::Quaterniond q(Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.3, 1.0, -0.2).normalized()));
    Eigen::Vector3d t(1.0, -2.0, 0.5);
    Eigen::Vector3d X(3.0, 1.0, -1.0);
    Eigen::Vector3d z;
    {
        Eigen::Vector3d pred = q.conjugate() * (X - t);
        z = pred + Eigen::Vector3d(0.002, -0.001, 0.0005);  // 注入残差
    }

    double q4[4] = {q.w(), q.x(), q.y(), q.z()};
    double t3[3] = {t(0), t(1), t(2)};
    double X3[3] = {X(0), X(1), X(2)};
    // Ceres CostFunction::Evaluate 采用指针数组形式的参数块/雅可比块
    const double* params[3] = {q4, t3, X3};

    // AutoDiff
    ceres::AutoDiffCostFunction<calib::PointPairResidual, 3, 4, 3, 3>
        cost(new calib::PointPairResidual(z, 0.01));
    double res_ad[3] = {0,0,0};
    double J_ad[3 * 10] = {0};  // 块 0(3x4)=12, 块 1(3x3)=9, 块 2(3x3)=9
    double* J_ad_ptrs[3] = {J_ad, J_ad + 12, J_ad + 21};
    ASSERT_TRUE(cost.Evaluate(params, res_ad, J_ad_ptrs));

    // 数值导数(参照对象,同 functor):用 RIDDERS 自适应法。残差经 1/sigma=100 缩放,
    // 雅可比某些项量级达 ~10^2;定步长 CENTRAL 的截断/舍入误差会随机种子在个别项
    // 上略超 1e-5。RIDDERS 自适应外推可稳定达到 ~1e-9,更适合做梯度校验基准。
    ceres::NumericDiffCostFunction<calib::PointPairResidual, ceres::RIDDERS, 3, 4, 3, 3>
        num(new calib::PointPairResidual(z, 0.01),
            ceres::TAKE_OWNERSHIP,
            3,
            ceres::NumericDiffOptions());
    double res_num[3] = {0,0,0};
    double J_num[3 * 10] = {0};
    double* J_num_ptrs[3] = {J_num, J_num + 12, J_num + 21};
    ASSERT_TRUE(num.Evaluate(params, res_num, J_num_ptrs));

    // 残差值一致(autodiff 与数值差分应几乎相同)
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(res_ad[i], res_num[i], 1e-9);
    // 雅可比一致(中央差分精度 ~1e-6,autodiff 精确;容差 1e-5)
    for (int i = 0; i < 3 * 10; ++i)
        EXPECT_NEAR(J_ad[i], J_num[i], 1e-5);
}

// 合成场景:nPoints 个全局点 + nFrames 帧。帧0为 identity(锚点),其余帧绕小转角运动。
// 每帧可见约 idsPerFrame 个点(高重叠)。返回 (input, GT点, GT位姿) 供校验。
struct SyntheticGT {
    std::vector<cv::Point3d> points;                       // GT 全局点(frame-0 系)
    std::vector<std::pair<cv::Matx33d, cv::Vec3d>> poses;  // GT 每帧位姿 (R_i, t_i)
};

static GlobalBAInput makeSyntheticInput(int nPoints, int nFrames, int idsPerFrame,
                                        SyntheticGT* gtOut = nullptr,
                                        uint64_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> posDist(-100.0, 100.0);  // 200mm 立方
    // GT 全局点
    std::vector<cv::Point3d> Xgt(nPoints);
    for (auto& p : Xgt) p = cv::Point3d(posDist(rng), posDist(rng), posDist(rng));

    GlobalBAInput in;
    std::vector<std::pair<cv::Matx33d, cv::Vec3d>> gtPoses;
    // 扰动分布(给位姿初值加噪)
    std::normal_distribution<double> angNoise(0.0, 0.5 * CV_PI / 180.0);  // 0.5 deg
    std::normal_distribution<double> tNoise(0.0, 0.1);                     // 0.1 mm

    for (int i = 0; i < nFrames; ++i) {
        // GT 位姿:帧0 identity,其余绕 z 轴小转角 + 平移
        cv::Matx33d R = cv::Matx33d::eye();
        cv::Vec3d t(0, 0, 0);
        if (i > 0) {
            double ang = (i * 0.3) * CV_PI / 180.0;  // 每帧 0.3 deg
            cv::Matx33d Rz(std::cos(ang), -std::sin(ang), 0,
                           std::sin(ang), std::cos(ang), 0,
                           0, 0, 1);
            R = Rz;
            t = cv::Vec3d(i * 0.5, 0.0, 0.0);  // 平移
        }
        gtPoses.push_back({R, t});

        GlobalBAFrame f;
        f.frameId = static_cast<uint64_t>(i);
        // 每帧可见连续 idsPerFrame 个点(滑动窗口,高重叠)
        for (int k = 0; k < idsPerFrame; ++k) {
            int pid = (i + k) % nPoints;
            // local = Rᵀ (X - t)
            cv::Vec3d Xv(Xgt[pid].x, Xgt[pid].y, Xgt[pid].z);
            cv::Vec3d local = R.t() * (Xv - t);
            f.markerObs.push_back({cv::Point3d(local(0), local(1), local(2)), pid});
        }
        // 初值 = GT + 扰动(帧0不加扰动,保持 identity 作为锚)
        if (i == 0) {
            f.R_init = cv::Matx33d::eye();
            f.t_init = cv::Vec3d(0, 0, 0);
        } else {
            // 小角度扰动旋转
            double da = angNoise(rng), db = angNoise(rng), dc = angNoise(rng);
            cv::Matx33d Rx(1, 0, 0, 0, std::cos(da), -std::sin(da), 0, std::sin(da), std::cos(da));
            cv::Matx33d Ry(std::cos(db), 0, std::sin(db), 0, 1, 0, -std::sin(db), 0, std::cos(db));
            cv::Matx33d Rz2(std::cos(dc), -std::sin(dc), 0, std::sin(dc), std::cos(dc), 0, 0, 0, 1);
            cv::Matx33d dR = Rz2 * Ry * Rx;
            f.R_init = dR * R;
            f.t_init = cv::Vec3d(t(0) + tNoise(rng), t(1) + tNoise(rng), t(2) + tNoise(rng));
        }
        in.frames.push_back(std::move(f));
    }
    if (gtOut) {
        gtOut->points = Xgt;
        gtOut->poses = gtPoses;
    }
    return in;
}

// 构造闭环场景: 末帧与首帧共视 loopSharedIds 个点(显式闭环)。可选在闭合处给末帧 t_init 注入突变,
// 模拟增量链闭合处的位姿不连续(漂移)。
// 关键: nPoints 取足够大(> nFrames + idsPerFrame), 避免滑动窗口 %nPoints 在帧域内自然循环,
//       从而保证"首末共视"完全由显式注入产生(无其它远端共视干扰闭环判定与 PGO)。
static GlobalBAInput makeLoopSyntheticInput(int nFrames, int idsPerFrame, int loopSharedIds,
                                            SyntheticGT* gtOut = nullptr,
                                            uint64_t seed = 11,
                                            bool injectClosureJump = true) {
    const int nPoints = nFrames + idsPerFrame + 16;  // 充足, 避免窗口循环产生意外共视
    SyntheticGT gt;
    auto in = makeSyntheticInput(nPoints, nFrames, idsPerFrame, &gt, seed);
    // 末帧: 前 loopSharedIds 个观测改为指向首帧看过的点 globalId 0..loopSharedIds-1,
    //       local 用末帧 GT 位姿投影 GT 点(与首帧观测几何一致 → 真闭环, 非错配)
    auto& last = in.frames.back();
    const cv::Matx33d& R = gt.poses.back().first;
    const cv::Vec3d&   t = gt.poses.back().second;
    for (int k = 0; k < loopSharedIds && k < static_cast<int>(last.markerObs.size()); ++k) {
        int pid = k;
        cv::Vec3d Xv(gt.points[pid].x, gt.points[pid].y, gt.points[pid].z);
        cv::Vec3d local = R.t() * (Xv - t);
        last.markerObs[k].local = cv::Point3d(local(0), local(1), local(2));
        last.markerObs[k].globalId = pid;
    }
    // 可选: 末帧位姿初值注入闭合突变(模拟增量链闭合处不连续, 2mm)
    if (injectClosureJump) {
        in.frames.back().t_init += cv::Vec3d(2.0, 0.0, 0.0);
    }
    if (gtOut) *gtOut = gt;
    return in;
}

TEST(GlobalBACore, RecoversSyntheticSceneBelow002mm) {
    SyntheticGT gt;
    auto in = makeSyntheticInput(20, 30, 15, &gt);  // 20点,30帧,每帧15点
    GlobalBAParams p;
    p.enablePoseGraphPreopt = false;  // 纯链式,直奔GBA(PGO尚未实现)
    GlobalBundleAdjustmentCPU op(p);
    auto r = op.Execute(in);
    ASSERT_TRUE(r.success) << r.message;
    // finalRMSE < 0.02mm,且优于初值
    EXPECT_LT(r.statistics.finalRMSE, 0.02);
    EXPECT_LT(r.statistics.finalRMSE, r.statistics.initialRMSE);
    // 无外点场景不应误杀任何观测
    EXPECT_TRUE(r.statistics.outlierObsIds.empty());
    // 优化后全局点应与 GT 匹配(用 globalId 索引)
    // 帧0锚定在 identity(=GT帧0),故优化点直接在 GT 坐标系,可逐点比较
    for (const auto& m : r.optimizedMarkers) {
        const auto& g = gt.points[m.globalId];
        double dx = m.X.x - g.x, dy = m.X.y - g.y, dz = m.X.z - g.z;
        double err = std::sqrt(dx * dx + dy * dy + dz * dz);
        EXPECT_LT(err, 0.05) << "point " << m.globalId << " err=" << err;
    }
}

TEST(GlobalBACenterOrigin, LargeOffsetRecoversCorrectly) {
    // 场景:仅把地标整体平移 offset(5000,0,0),扫描仪轨迹(位姿)不变。
    // 故 local = Rᵀ((X+offset) - t)(用原 GT 位姿重算);帧0仍 identity。
    // t_init 保持 base 的 GT+噪声(与未平移位姿一致),不再额外平移。
    SyntheticGT gt;
    auto base = makeSyntheticInput(20, 30, 15, &gt);
    cv::Vec3d offset(5000.0, 0.0, 0.0);
    GlobalBAInput inOff = base;
    std::vector<cv::Point3d> gtOff = gt.points;
    for (auto& p : gtOff) { p.x += offset(0); p.y += offset(1); p.z += offset(2); }
    // 重算每帧 local(平移后的点 + 原 GT 位姿);t_init 不动(沿用 base 的 GT+噪声)
    for (size_t i = 0; i < inOff.frames.size(); ++i) {
        const cv::Matx33d& R = gt.poses[i].first;
        const cv::Vec3d&   t = gt.poses[i].second;
        for (auto& obs : inOff.frames[i].markerObs) {
            int pid = obs.globalId;
            cv::Vec3d Xv(gtOff[pid].x, gtOff[pid].y, gtOff[pid].z);
            cv::Vec3d local = R.t() * (Xv - t);
            obs.local = cv::Point3d(local(0), local(1), local(2));
        }
    }

    // centerOrigin=true:恢复后应匹配平移后的 GT
    GlobalBAParams pOn; pOn.enablePoseGraphPreopt = false; pOn.centerOrigin = true;
    GlobalBundleAdjustmentCPU opOn(pOn);
    auto rOn = opOn.Execute(inOff);
    ASSERT_TRUE(rOn.success) << rOn.message;
    EXPECT_LT(rOn.statistics.finalRMSE, 0.02);
    for (const auto& m : rOn.optimizedMarkers) {
        const auto& g = gtOff[m.globalId];
        double dx = m.X.x - g.x, dy = m.X.y - g.y, dz = m.X.z - g.z;
        EXPECT_LT(std::sqrt(dx * dx + dy * dy + dz * dz), 0.05);
    }

    // centerOrigin=false 对照:centerOrigin 是纯重参数化,两种模式结果应一致
    // (若实现有单侧平移或漏反平移,两模式输出会相差 offset 而失败)
    GlobalBAParams pOff; pOff.enablePoseGraphPreopt = false; pOff.centerOrigin = false;
    GlobalBundleAdjustmentCPU opOff(pOff);
    auto rOff = opOff.Execute(inOff);
    ASSERT_TRUE(rOff.success) << rOff.message;
    EXPECT_LT(rOff.statistics.finalRMSE, 0.02);
    ASSERT_EQ(rOn.optimizedMarkers.size(), rOff.optimizedMarkers.size());
    for (size_t i = 0; i < rOn.optimizedMarkers.size(); ++i) {
        const auto& a = rOn.optimizedMarkers[i];
        const auto& b = rOff.optimizedMarkers[i];
        ASSERT_EQ(a.globalId, b.globalId);
        double dx = a.X.x - b.X.x, dy = a.X.y - b.X.y, dz = a.X.z - b.X.z;
        EXPECT_LT(std::sqrt(dx * dx + dy * dy + dz * dz), 1e-6)
            << "marker " << a.globalId << " differs between centerOrigin on/off";
    }
}

// 注入比例飞点:把 frac 的观测 local 加 mag mm 偏移
static void injectOutliers(GlobalBAInput& in, double frac, double mag, uint64_t seed = 7) {
    std::mt19937 rng(seed);
    // 收集所有 (帧idx, 帧内obs idx) 的全局列表
    struct Pair { size_t fi; size_t oi; };
    std::vector<Pair> all;
    for (size_t i = 0; i < in.frames.size(); ++i)
        for (size_t k = 0; k < in.frames[i].markerObs.size(); ++k)
            all.push_back({i, k});
    std::shuffle(all.begin(), all.end(), rng);
    size_t nOut = static_cast<size_t>(all.size() * frac);
    std::uniform_real_distribution<double> dir(-1.0, 1.0);
    for (size_t i = 0; i < nOut; ++i) {
        auto& obs = in.frames[all[i].fi].markerObs[all[i].oi];
        cv::Vec3d d(dir(rng), dir(rng), dir(rng));
        double len = std::sqrt(d.dot(d)); if (len < 1e-9) len = 1.0;
        d = d * (mag / len);
        obs.local = cv::Point3d(obs.local.x + d(0), obs.local.y + d(1), obs.local.z + d(2));
    }
}

TEST(GlobalBAOutlier, GrossFlyersCulledAndNotDegraded) {
    SyntheticGT gt;
    auto in = makeSyntheticInput(20, 30, 15, &gt);
    injectOutliers(in, 0.10, 1.0);  // 10% 飞点, 1mm 偏移(=100σ)
    GlobalBAParams p; p.enablePoseGraphPreopt = false;
    GlobalBundleAdjustmentCPU op(p);
    auto r = op.Execute(in);
    ASSERT_TRUE(r.success) << r.message;
    // 恰好剔除注入的 45 个飞点(30帧×15obs×10%=45),无漏剔/过杀
    EXPECT_EQ(r.statistics.outlierObsIds.size(), 45u);
    EXPECT_LT(r.statistics.finalRMSE, 0.05);            // 未被飞点拖偏
}

// 闭环场景: 首末帧共视 + 末帧 t_init 注入 2mm 闭合突变。PGO 预优化应平滑突变,
// 随后 GBA 收敛到低残差。验证 PGO+GBA 闭环分支端到端可用。
// I1: 增加 PGO-off 对照臂, 证明 PGO-on 至少不劣于 PGO-off(防 PGO 回归)。
TEST(GlobalBALoop, LoopClosurePGOLowResidual) {
    SyntheticGT gt;
    auto in = makeLoopSyntheticInput(60 /*nFrames*/, 15 /*idsPerFrame*/, 8 /*loopSharedIds*/,
                                     &gt, 11 /*seed*/, true /*closure jump*/);

    // 注: 在此合成场景中, 闭合突变(2mm)落在 GBA plain-LM 的收敛盆内,
    // 故 PGO-on 与 PGO-off 都能收敛到低残差(均 ~0)。PGO 的可度量优势体现在真实数据
    // 的大闭合不连续上(难以确定性构造)。该 PGO-off 对照确保能捕捉到 PGO 回归
    // (PGO-on 不应比 PGO-off 差), 而非仅证明 PGO 不破坏流水线。
    GlobalBAParams pOn; pOn.enablePoseGraphPreopt = true;
    GlobalBundleAdjustmentCPU opOn(pOn);
    auto r_on = opOn.Execute(in);
    ASSERT_TRUE(r_on.success) << r_on.message;
    EXPECT_TRUE(r_on.statistics.loopDetected);
    // PGO+GBA 应收敛, 观测无噪声 → 最终残差应很低
    EXPECT_LT(r_on.statistics.finalRMSE, 0.05);
    // loopClosureResidual 统计应被填充(诊断量, 非负)
    EXPECT_GE(r_on.statistics.loopClosureResidual, 0.0);

    // PGO-off 对照: 同一输入, 仅关闭 PGO 再跑一遍。
    GlobalBAParams pOff; pOff.enablePoseGraphPreopt = false;
    GlobalBundleAdjustmentCPU opOff(pOff);
    auto r_off = opOff.Execute(in);
    ASSERT_TRUE(r_off.success) << r_off.message;
    // PGO-on 不应比 PGO-off 差(容 1e-9 数值容差)。
    EXPECT_LE(r_on.statistics.finalRMSE, r_off.statistics.finalRMSE + 1e-9)
        << "PGO-on worse than PGO-off: r_on=" << r_on.statistics.finalRMSE
        << " r_off=" << r_off.statistics.finalRMSE;
}

// Task 11: 开环链式自适应回归。
// 即便 enablePoseGraphPreopt=true,纯链式(loopDetected=false)也应跳过 PGO 直奔 GBA。
// 验证自适应分支:链式 → 无 PGO 副作用 → GBA 正常恢复。
TEST(GlobalBAChain, OpenChainSkipsPGOAndRecovers) {
    SyntheticGT gt;
    // makeSyntheticInput(20,30,15):帧0与帧29共视点多,但 |29-0|=29 < loopFrameGap=30 → 非闭环边
    auto in = makeSyntheticInput(20, 30, 15, &gt);
    GlobalBAParams p; p.enablePoseGraphPreopt = true;  // 故意开启,验证链式下被正确跳过
    GlobalBundleAdjustmentCPU op(p);
    auto r = op.Execute(in);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_FALSE(r.statistics.loopDetected);          // 纯链式:无闭环
    EXPECT_LT(r.statistics.finalRMSE, r.statistics.initialRMSE);  // GBA 有改进
    EXPECT_LT(r.statistics.finalRMSE, 0.02);          // 恢复到亚 0.02mm
    EXPECT_TRUE(r.statistics.outlierObsIds.empty());  // 无外点,无误杀
}

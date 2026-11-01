#pragma once
// ============================================================================
// calib_synthetic.h — B 相机链合成标定数据工厂（T21 建，T25 激光链复用）
// ============================================================================
// 真值模型：
//   K1=K2=[fx=800, fy=800, cx=320, cy=240]（640×480）；D=0（五参数）
//   双目外参 R（小角度装配误差）/T=(-100, 1.5, 3.0)mm；恒等矫正 R1=R2=I、
//   P1=P2=[K|0] → 矫正系 ≡ 原始系（初始参数给真值时 3-1 近似恒等映射）。
// 生成（makeSyntheticSession）：每姿态随机板位姿（三轴 ±35° / x∈±45 /
//   y∈±30 / z∈150..350 mm，出界重掷保证 88 角点全在画幅内）投影 L/R，
//   加 0.02px 高斯噪声 → 直接填 PostureSessionData.ellipseCentersL/R
//   （已是矫正系=原始系等价，因初始参数给真值）+ PostureData.cycle 最小占位。
// 板点序：r 外层 / c 内层（同 09 intrinsic_calib generateObjectPoints）。
// 确定性：固定 seed（默认 42），同 seed 同数据。
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "pipelines/calibcompute/CalibComputeTypes.h"
#include "pipelines/posture/PostureTypes.h"

namespace Scanner::pipeline::synthetic {

struct SyntheticTruth {
    cv::Mat K1, D1, K2, D2;                    // 真内参/畸变（L/R）
    cv::Mat R, T;                              // 双目外参（p_R = R·p_L + T，mm）
    cv::Mat R1, R2, P1, P2;                    // 恒等矫正（R=I，P=[K|0]）
    cv::Size imageSize{640, 480};
    std::vector<cv::Point3f> boardPoints3D;    // 11×8@5mm，Z=0 网格
};

inline std::vector<cv::Point3f> makeBoardPoints3D(int cols = 11, int rows = 8,
                                                  double squareMm = 5.0) {
    std::vector<cv::Point3f> pts;
    pts.reserve(static_cast<size_t>(cols) * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            pts.emplace_back(static_cast<float>(c * squareMm),
                             static_cast<float>(r * squareMm), 0.0f);
    return pts;
}

// 绕 X/Y/Z 各转 deg 度的合成旋转（R = Rz·Ry·Rx）
inline cv::Mat rotXYZ(double degX, double degY, double degZ) {
    const double a = CV_PI / 180.0;
    cv::Mat rx = (cv::Mat_<double>(3, 3) <<
        1, 0, 0, 0, std::cos(degX * a), -std::sin(degX * a), 0, std::sin(degX * a), std::cos(degX * a));
    cv::Mat ry = (cv::Mat_<double>(3, 3) <<
        std::cos(degY * a), 0, std::sin(degY * a), 0, 1, 0, -std::sin(degY * a), 0, std::cos(degY * a));
    cv::Mat rz = (cv::Mat_<double>(3, 3) <<
        std::cos(degZ * a), -std::sin(degZ * a), 0, std::sin(degZ * a), std::cos(degZ * a), 0, 0, 0, 1);
    return rz * ry * rx;
}

inline SyntheticTruth makeTruth() {
    SyntheticTruth t;
    t.K1 = (cv::Mat_<double>(3, 3) << 800.0, 0, 320.0, 0, 800.0, 240.0, 0, 0, 1);
    t.K2 = t.K1.clone();
    t.D1 = cv::Mat::zeros(1, 5, CV_64F);        // 真值零畸变（五参数）
    t.D2 = t.D1.clone();
    t.R = rotXYZ(0.3, -0.5, 0.8);               // 小角度装配误差
    t.T = (cv::Mat_<double>(3, 1) << -100.0, 1.5, 3.0);
    t.R1 = cv::Mat::eye(3, 3, CV_64F);          // 恒等矫正 → 矫正系≡原始系
    t.R2 = t.R1.clone();
    t.P1 = (cv::Mat_<double>(3, 4) << 800.0, 0, 320.0, 0, 0, 800.0, 240.0, 0, 0, 0, 1, 0);
    t.P2 = t.P1.clone();
    t.boardPoints3D = makeBoardPoints3D();
    return t;
}

// 生成 poses 个已确认姿态的会话数据；truth 出参带回真值（供断言/初始参数）。
inline PostureSessionData makeSyntheticSession(int poses, SyntheticTruth& truth,
                                               unsigned seed = 42) {
    truth = makeTruth();
    if (poses <= 0 || poses > PostureSessionData::kTargetCount)
        poses = PostureSessionData::kTargetCount;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> angDeg(-35.0, 35.0);
    std::uniform_real_distribution<double> tx(-45.0, 45.0);
    std::uniform_real_distribution<double> ty(-30.0, 30.0);
    std::uniform_real_distribution<double> tz(150.0, 350.0);
    std::normal_distribution<double> noise(0.0, 0.02);   // px
    const float margin = 8.0f;
    const int W = truth.imageSize.width, H = truth.imageSize.height;

    PostureSessionData s;
    int made = 0;
    while (made < poses) {
        cv::Mat Rb = rotXYZ(angDeg(rng), angDeg(rng), angDeg(rng));   // 板→L 系
        cv::Mat tb = (cv::Mat_<double>(3, 1) << tx(rng), ty(rng), tz(rng));
        cv::Mat Rb2 = truth.R * Rb;                                   // 板→R 系
        cv::Mat tb2 = truth.R * tb + truth.T;
        cv::Mat rL, rR;
        cv::Rodrigues(Rb, rL);
        cv::Rodrigues(Rb2, rR);

        std::vector<cv::Point2f> pL, pR;
        cv::projectPoints(truth.boardPoints3D, rL, tb, truth.K1, truth.D1, pL);
        cv::projectPoints(truth.boardPoints3D, rR, tb2, truth.K2, truth.D2, pR);

        bool inBounds = true;
        for (const auto& p : pL)
            if (p.x < margin || p.x > W - margin || p.y < margin || p.y > H - margin) { inBounds = false; break; }
        if (inBounds)
            for (const auto& p : pR)
                if (p.x < margin || p.x > W - margin || p.y < margin || p.y > H - margin) { inBounds = false; break; }
        if (!inBounds) continue;                                      // 出界重掷（确定性）

        for (auto& p : pL) { p.x += static_cast<float>(noise(rng)); p.y += static_cast<float>(noise(rng)); }
        for (auto& p : pR) { p.x += static_cast<float>(noise(rng)); p.y += static_cast<float>(noise(rng)); }

        auto& pd = s.poses[made];
        pd.cycleId = static_cast<uint64_t>(made) + 1;
        pd.cycle.id = pd.cycleId;                 // 最小占位（B 相机链不消费帧数据）
        pd.cycle.temperature = 25.0;
        pd.cycle.timestamp = 1000 + made;
        for (int i = 0; i < 9; ++i) pd.R[i] = Rb.at<double>(i / 3, i % 3);
        for (int i = 0; i < 3; ++i) pd.T[i] = tb.at<double>(i);
        pd.ellipseCentersL = std::move(pL);
        pd.ellipseCentersR = std::move(pR);
        s.collected[made] = true;
        ++made;
    }
    s.collectedCount = poses;
    return s;
}

// 初始参数 = 真值（2-6 与 3-1 严格同组保证）
inline InitialCalibParams makeInitialFromTruth(const SyntheticTruth& t) {
    InitialCalibParams p;
    p.K1 = t.K1.clone();  p.D1 = t.D1.clone();
    p.K2 = t.K2.clone();  p.D2 = t.D2.clone();
    p.R1 = t.R1.clone();  p.P1 = t.P1.clone();
    p.R2 = t.R2.clone();  p.P2 = t.P2.clone();
    p.imageSize = t.imageSize;
    return p;
}

// ============================================================================
// T22 激光链合成数据（层 2 真算子冒烟用）
// ============================================================================
// makeSyntheticLaserFrame：黑底 + nLines 条水平激光条纹（高斯剖面
//   255·exp(-d²/2σ²)）。水平走向是 laser_label 的硬约束：其只保留与图像
//   中心列相交的连通域（按中心列最小 Y 重编号 1..N，上→下）——竖直条纹
//   不交中心列会被整域丢弃（T22 实测定版）。steger 需平滑剖面（实心条
//   内部二阶导为零提不出中心）；同 09 steger 测试的合成条纹形态。
inline cv::Mat makeSyntheticLaserFrame(const cv::Size& sz, int nLines = 2,
                                       double sigma = 2.5) {
    cv::Mat img(sz, CV_8UC1, cv::Scalar(8));           // 黑底（含少量底噪）
    std::vector<double> centers;
    for (int k = 0; k < nLines; ++k)
        centers.push_back((k + 1.0) * sz.height / (nLines + 1.0));
    const double denom = 2.0 * sigma * sigma;
    for (int y = 0; y < sz.height; ++y) {
        double v = 0.0;
        for (double c : centers) {
            const double d = y - c;
            v = std::max(v, 255.0 * std::exp(-d * d / denom));
        }
        const uchar rowVal = static_cast<uchar>(std::min(v, 255.0));
        for (int x = 0; x < sz.width; ++x) img.at<uchar>(y, x) = rowVal;
    }
    return img;
}

// attachSyntheticLaserFrames：为每个已确认姿态填 tubes 路激光管帧
//   （laserFrames 扁平 N*2：偶=L/奇=R；L/R 同图 → 零视差，仅结构冒烟用）
inline void attachSyntheticLaserFrames(PostureSessionData& s, int tubes,
                                       const cv::Size& sz, int nLines = 2,
                                       double sigma = 1.2) {
    for (int i = 0; i < PostureSessionData::kTargetCount; ++i) {
        if (!s.collected[i]) continue;
        auto& frames = s.poses[i].cycle.laserFrames;
        frames.clear();
        frames.reserve(static_cast<size_t>(tubes) * 2);
        for (int t = 0; t < tubes; ++t) {
            frames.push_back(makeSyntheticLaserFrame(sz, nLines, sigma));  // L
            frames.push_back(makeSyntheticLaserFrame(sz, nLines, sigma));  // R
        }
    }
}

// makeSyntheticStereoParams：由真值经 cv::stereoRectify 生成冒烟用 StereoParams
//   （4-5 的矫正组 R1/P1/R2/P2 与 4-8 的 Q 均需有效矩阵；合成帧非真实投影，
//   重建坐标无物理意义——只验证算子链不崩）
inline StereoParams makeSyntheticStereoParams(const SyntheticTruth& t) {
    StereoParams sp;
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(t.K1, t.D1, t.K2, t.D2, t.imageSize, t.R, t.T,
                      R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY);
    sp.cameraMatrixL = t.K1.clone();  sp.distCoeffsL = t.D1.clone();
    sp.cameraMatrixR = t.K2.clone();  sp.distCoeffsR = t.D2.clone();
    sp.R = t.R.clone();               sp.T = t.T.clone();
    sp.R1 = R1;  sp.R2 = R2;  sp.P1 = P1;  sp.P2 = P2;  sp.Q = Q;
    sp.reprojError = 0.0;
    return sp;
}

} // namespace Scanner::pipeline::synthetic

#pragma once
// testutil：圆锥相交算子合成数据源（BUILD_TESTS 专用，禁入产线）
#include <opencv2/core.hpp>
#include <vector>
#include <cstdint>

namespace calib::synth {

struct Rig {
    cv::Matx33d K1, K2;      // 左右内参（像素系）
    cv::Vec3d C2;            // 右光心（左世界系，mm）
    static Rig scanRig();    // f=1838 主点(1052,763) B=131mm
};

struct CircleSpec {
    cv::Vec3d C;             // 圆心 mm
    cv::Vec3d n;             // 单位法向
    double r;                // 半径 mm
    static CircleSpec pose(double tiltDeg, double azDeg, double Z, double rMm);
};

struct EllipseObs {
    double cx, cy, a, b, angleDeg;   // cv::fitEllipse 同构参数（a,b=全轴长）
};

struct SyntheticMarker {
    CircleSpec truth;
    EllipseObs left, right;
};

class MarkerSynth {
public:
    MarkerSynth(Rig rig, uint64_t seed);
    SyntheticMarker make(const CircleSpec& spec, int nEdges = 48, double noisePx = 0.0) const;
    // 拒枝旋钮：右目轴整体缩放（同中心）
    static SyntheticMarker axesScaleRight(const SyntheticMarker& m, double k);
    // 拒枝旋钮：左右异源（右目换成另一个圆的投影）
    static SyntheticMarker mismatched(const SyntheticMarker& m, const SyntheticMarker& rightSrc);
private:
    Rig rig_;
    uint64_t seed_;
    EllipseObs project_(const CircleSpec& s, bool rightEye, int nEdges, double noisePx) const;
};

} // namespace calib::synth

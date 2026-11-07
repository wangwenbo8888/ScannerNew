#include "marker_synth.h"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace calib::synth {

Rig Rig::scanRig() {
    cv::Matx33d K(1838.0, 0, 1052.0, 0, 1838.0, 763.0, 0, 0, 1);
    return Rig{K, K, cv::Vec3d(131.0, 0.0, 0.0)};
}

CircleSpec CircleSpec::pose(double tiltDeg, double azDeg, double Z, double rMm) {
    const double t = tiltDeg * CV_PI / 180.0, a = azDeg * CV_PI / 180.0;
    cv::Vec3d n(std::sin(t) * std::cos(a), std::sin(t) * std::sin(a), std::cos(t));
    return CircleSpec{cv::Vec3d(0.0, 0.0, Z), n, rMm};
}

MarkerSynth::MarkerSynth(Rig rig, uint64_t seed) : rig_(rig), seed_(seed) {}

EllipseObs MarkerSynth::project_(const CircleSpec& s, bool rightEye, int n, double noisePx) const {
    cv::Vec3d ax(1, 0, 0);
    if (std::abs(s.n(0)) > 0.9) ax = cv::Vec3d(0, 1, 0);
    cv::Vec3d u = s.n.cross(ax); u /= cv::norm(u);
    cv::Vec3d v = s.n.cross(u);
    cv::RNG rng(static_cast<uint64_t>(seed_) + (rightEye ? 7919u : 0u));
    std::vector<cv::Point2f> pts;
    pts.reserve(n);
    const cv::Vec3d eye = rightEye ? rig_.C2 : cv::Vec3d(0, 0, 0);
    const cv::Matx33d& K = rightEye ? rig_.K2 : rig_.K1;
    for (int i = 0; i < n; ++i) {
        const double th = 2 * CV_PI * i / n + rng.uniform(0.0, 0.01);
        cv::Vec3d X = s.C + s.r * (std::cos(th) * u + std::sin(th) * v);
        cv::Vec3d Xc = X - eye;
        cv::Vec3d x = K * Xc;
        double px = x(0) / x(2), py = x(1) / x(2);
        if (noisePx > 0) {
            px += rng.gaussian(noisePx);
            py += rng.gaussian(noisePx);
        }
        pts.emplace_back(static_cast<float>(px), static_cast<float>(py));
    }
    cv::RotatedRect e = cv::fitEllipse(pts);
    return EllipseObs{e.center.x, e.center.y, e.size.width, e.size.height, e.angle};
}

SyntheticMarker MarkerSynth::make(const CircleSpec& s, int n, double noisePx) const {
    return SyntheticMarker{s, project_(s, false, n, noisePx), project_(s, true, n, noisePx)};
}

SyntheticMarker MarkerSynth::axesScaleRight(const SyntheticMarker& m, double k) {
    SyntheticMarker r = m;
    r.right.a *= k;
    r.right.b *= k;
    return r;
}

SyntheticMarker MarkerSynth::mismatched(const SyntheticMarker& m, const SyntheticMarker& src) {
    SyntheticMarker r = m;
    r.right = src.right;
    return r;
}

} // namespace calib::synth

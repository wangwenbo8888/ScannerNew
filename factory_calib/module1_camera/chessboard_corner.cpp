#include "chessboard_corner.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <cmath>

namespace fc {

bool extractChessboardCorners(const cv::Mat& gray,
                              const ChessboardCornerParams& params,
                              ChessboardCornerResult& result) {
    result.found = false;
    result.corners.clear();
    result.meanSubpixDelta = 0.0;

    if (gray.empty()) {
        spdlog::warn("[chessboard_corner] empty image");
        return false;
    }
    cv::Mat gray8;
    if (gray.type() != CV_8U) {
        gray.convertTo(gray8, CV_8U);
    } else {
        gray8 = gray;
    }

    std::vector<cv::Point2f> coarse;
    bool found = cv::findChessboardCornersSB(gray8, params.patternSize, coarse, params.sbFlags);
    if (!found || static_cast<int>(coarse.size()) != params.patternSize.area()) {
        spdlog::debug("[chessboard_corner] not found or count mismatch");
        return false;
    }

    // 亚像素精化前快照，用于估算修正量
    std::vector<cv::Point2f> before = coarse;
    cv::cornerSubPix(gray8, coarse, params.subpixWin, params.subpixZeroZone, params.subpixTerm);

    double acc = 0.0;
    for (size_t i = 0; i < coarse.size(); ++i) {
        acc += std::hypot(coarse[i].x - before[i].x, coarse[i].y - before[i].y);
    }
    result.meanSubpixDelta = acc / coarse.size();

    result.corners = std::move(coarse);
    result.found = true;
    return true;
}

} // namespace fc

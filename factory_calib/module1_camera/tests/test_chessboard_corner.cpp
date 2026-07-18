#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>

#include "chessboard_corner.h"

using namespace fc;

namespace {
// 合成一张带已知棋盘角点的灰度图
cv::Mat makeSyntheticChessboard(int winW, int winH, int squaresX, int squaresY,
                                int squarePx, int margin) {
    int W = margin * 2 + squaresX * squarePx;
    int H = margin * 2 + squaresY * squarePx;
    cv::Mat img(H, W, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r < squaresY; ++r)
        for (int c = 0; c < squaresX; ++c) {
            if ((r + c) % 2 == 0) {
                cv::Rect roi(margin + c * squarePx, margin + r * squarePx, squarePx, squarePx);
                img(roi) = 0;
            }
        }
    return img;
}
} // namespace

TEST(ChessboardCorner, ExtractsAllCorners) {
    int squaresX = 12, squaresY = 9;     // 内角点 = 11x8
    int squarePx = 30, margin = 40;
    cv::Mat gray = makeSyntheticChessboard(0, 0, squaresX, squaresY, squarePx, margin);

    ChessboardCornerParams p;
    p.patternSize = cv::Size(squaresX - 1, squaresY - 1);  // 11x8
    ChessboardCornerResult r;
    ASSERT_TRUE(extractChessboardCorners(gray, p, r));
    EXPECT_TRUE(r.found);
    EXPECT_EQ(r.corners.size(), static_cast<size_t>(p.patternSize.area()));
}

TEST(ChessboardCorner, FailsOnBlankImage) {
    cv::Mat blank(480, 640, CV_8UC1, cv::Scalar(255));
    ChessboardCornerParams p;
    ChessboardCornerResult r;
    EXPECT_FALSE(extractChessboardCorners(blank, p, r));
    EXPECT_FALSE(r.found);
}

TEST(ChessboardCorner, SubpixBetterThanPixel) {
    int squaresX = 8, squaresY = 6;
    int squarePx = 40, margin = 30;
    cv::Mat gray = makeSyntheticChessboard(0, 0, squaresX, squaresY, squarePx, margin);
    ChessboardCornerParams p;
    p.patternSize = cv::Size(squaresX - 1, squaresY - 1);
    ChessboardCornerResult r;
    ASSERT_TRUE(extractChessboardCorners(gray, p, r));
    // OpenCV cornerSubPix 用像素中心约定：cv::Rect 整数坐标绘制的棋盘，其 4 格交汇
    // 内角点的亚像素坐标为 (margin + squarePx - 0.5, ...)（相对板外角 margin 的 -0.5 偏移）。
    const float expectedX = static_cast<float>(margin + squarePx) - 0.5f;
    const float expectedY = static_cast<float>(margin + squarePx) - 0.5f;
    // SB 的起始角不保证左上先出 → 在集合里找 min(x+y) 的点（即左上内角点）
    auto best = std::min_element(r.corners.begin(), r.corners.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) { return (a.x + a.y) < (b.x + b.y); });
    ASSERT_NE(best, r.corners.end());
    EXPECT_NEAR(best->x, expectedX, 0.5);
    EXPECT_NEAR(best->y, expectedY, 0.5);
}

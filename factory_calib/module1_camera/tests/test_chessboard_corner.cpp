#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

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
    // 第一个角点应在 (margin, margin)
    EXPECT_NEAR(r.corners[0].x, margin, 0.5);
    EXPECT_NEAR(r.corners[0].y, margin, 0.5);
}

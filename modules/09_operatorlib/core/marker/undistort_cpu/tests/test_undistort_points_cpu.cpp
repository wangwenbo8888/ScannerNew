/**
 * @file test_undistort_points_cpu.cpp
 * @brief 双目立体去畸变+矫正算子 - 单元测试
 *
 * 测试覆盖：
 * [Params] 参数校验 / JSON序列化 / WarmupConfig
 * [Stereo] 空输入 / 无畸变恒等 / 精度对比 / Y对齐 / EdgePoint重载 / 大量点 / R1R2P1P2输出
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "../undistort_points_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;


class MarkerUndistortCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.distortionModel = "brown_conrady";

        params_.fx1 = 2500.0; params_.fy1 = 2500.0;
        params_.cx1 = 640.0;  params_.cy1 = 512.0;
        params_.k1_1 = -0.1;  params_.k2_1 = 0.01;
        params_.p1_1 = 0.001; params_.p2_1 = -0.001;

        params_.fx2 = 2550.0; params_.fy2 = 2550.0;
        params_.cx2 = 645.0;  params_.cy2 = 515.0;
        params_.k1_2 = -0.12; params_.k2_2 = 0.015;
        params_.p1_2 = 0.0008; params_.p2_2 = -0.0009;

        params_.R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        params_.T = {-120.0, 0.5, 0.2};
        params_.imageWidth = 1280;
        params_.imageHeight = 1024;
    }

    MarkerUndistortCPUParams params_;

    cv::Mat buildK1() const {
        return (cv::Mat_<double>(3,3) <<
            params_.fx1, 0, params_.cx1,
            0, params_.fy1, params_.cy1,
            0, 0, 1);
    }
    cv::Mat buildD1() const {
        return (cv::Mat_<double>(5,1) <<
            params_.k1_1, params_.k2_1,
            params_.p1_1, params_.p2_1, 0.0);
    }
    cv::Mat buildK2() const {
        return (cv::Mat_<double>(3,3) <<
            params_.fx2, 0, params_.cx2,
            0, params_.fy2, params_.cy2,
            0, 0, 1);
    }
    cv::Mat buildD2() const {
        return (cv::Mat_<double>(5,1) <<
            params_.k1_2, params_.k2_2,
            params_.p1_2, params_.p2_2, 0.0);
    }
    cv::Mat buildR() const {
        return (cv::Mat_<double>(3,3) <<
            params_.R[0], params_.R[1], params_.R[2],
            params_.R[3], params_.R[4], params_.R[5],
            params_.R[6], params_.R[7], params_.R[8]);
    }
    cv::Mat buildT() const {
        return (cv::Mat_<double>(3,1) <<
            params_.T[0], params_.T[1], params_.T[2]);
    }
};

// ---- 参数校验 ----
TEST_F(MarkerUndistortCPUTest, DefaultParamsValidateThrowsDueToFx1) {
    MarkerUndistortCPUParams defaultParams;
    EXPECT_THROW(defaultParams.validate(), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, ValidParamsPassValidate) {
    EXPECT_NO_THROW(params_.validate());
}

TEST_F(MarkerUndistortCPUTest, ZeroFx1Throws) {
    auto p = params_; p.fx1 = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, NegativeFx1Throws) {
    auto p = params_; p.fx1 = -100.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, ZeroFy1Throws) {
    auto p = params_; p.fy1 = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, ZeroFx2Throws) {
    auto p = params_; p.fx2 = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, ZeroFy2Throws) {
    auto p = params_; p.fy2 = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, ZeroImageWidthThrows) {
    auto p = params_; p.imageWidth = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, ZeroImageHeightThrows) {
    auto p = params_; p.imageHeight = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, InvalidDistortionModelThrows) {
    auto p = params_; p.distortionModel = "fisheye";
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, RationalPolynomialModelIsValid) {
    auto p = params_; p.distortionModel = "rational_polynomial";
    EXPECT_NO_THROW(p.validate());
}

// ---- JSON 序列化 ----
TEST_F(MarkerUndistortCPUTest, JsonRoundtrip) {
    auto j = params_.toJson();
    auto restored = MarkerUndistortCPUParams::fromJson(j);
    EXPECT_NO_THROW(restored.validate());
    EXPECT_DOUBLE_EQ(restored.fx1, params_.fx1);
    EXPECT_DOUBLE_EQ(restored.fy1, params_.fy1);
    EXPECT_DOUBLE_EQ(restored.cx1, params_.cx1);
    EXPECT_DOUBLE_EQ(restored.cy1, params_.cy1);
    EXPECT_DOUBLE_EQ(restored.fx2, params_.fx2);
    EXPECT_EQ(restored.R, params_.R);
    EXPECT_EQ(restored.T, params_.T);
    EXPECT_EQ(restored.imageWidth, 1280);
    EXPECT_EQ(restored.distortionModel, params_.distortionModel);
}

TEST_F(MarkerUndistortCPUTest, JsonEmptyObjectGivesDefaults) {
    auto restored = MarkerUndistortCPUParams::fromJson(nlohmann::json{});
    EXPECT_DOUBLE_EQ(restored.fx1, 0.0);
    EXPECT_EQ(restored.distortionModel, "brown_conrady");
}

TEST_F(MarkerUndistortCPUTest, JsonUnknownFieldsIgnored) {
    auto j = params_.toJson();
    j["unknownField"] = 42;
    auto restored = MarkerUndistortCPUParams::fromJson(j);
    EXPECT_DOUBLE_EQ(restored.fx1, params_.fx1);
}

// ---- StereoResult 结构体 ----
TEST_F(MarkerUndistortCPUTest, StereoResultDefaultValues) {
    StereoUndistortResult result;
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.pointCount1, 0);
    EXPECT_EQ(result.pointCount2, 0);
    EXPECT_TRUE(result.rectifiedPoints1.empty());
    EXPECT_TRUE(result.rectifiedPoints2.empty());
}

TEST_F(MarkerUndistortCPUTest, StereoResultMoveSemantics) {
    StereoUndistortResult r1;
    r1.success = true; r1.pointCount1 = 3; r1.pointCount2 = 3;
    r1.rectifiedPoints1 = {cv::Point2d(1,2)};
    r1.rectifiedPoints2 = {cv::Point2d(3,4)};
    StereoUndistortResult r2 = std::move(r1);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(r2.pointCount1, 3);
}

// ---- WarmupConfig / 构造析构 ----
TEST_F(MarkerUndistortCPUTest, WarmupConfigForPointCloud) {
    auto config = calib::WarmupConfig::forPointCloud(1000);
    EXPECT_EQ(config.maxPointCount, 1000);
}

TEST_F(MarkerUndistortCPUTest, ConstructWithValidParams) {
    EXPECT_NO_THROW(MarkerUndistortCPU op(params_));
}

TEST_F(MarkerUndistortCPUTest, ConstructWithDefaultParamsThrows) {
    EXPECT_THROW(MarkerUndistortCPU op(MarkerUndistortCPUParams{}), std::invalid_argument);
}

TEST_F(MarkerUndistortCPUTest, WarmupWithMaxPointCount) {
    MarkerUndistortCPU op(params_);
    EXPECT_NO_THROW(op.Warmup(1000));
}

TEST_F(MarkerUndistortCPUTest, WarmupWithConfig) {
    MarkerUndistortCPU op(params_);
    EXPECT_NO_THROW(op.Warmup(calib::WarmupConfig::forPointCloud(500)));
}

// ---- 空输入 ----
TEST_F(MarkerUndistortCPUTest, EmptyPointsReturnsWarning) {
    MarkerUndistortCPU op(params_);
    auto result = op.Execute(
        std::vector<cv::Point2d>{}, std::vector<cv::Point2d>{});
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
}

TEST_F(MarkerUndistortCPUTest, EmptyLeftReturnsWarning) {
    MarkerUndistortCPU op(params_);
    auto result = op.Execute(
        std::vector<cv::Point2d>{}, {cv::Point2d(100,200)});
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
}

// ---- 无畸变无旋转 (R=I, T=[-baseline,0,0]) ----
TEST_F(MarkerUndistortCPUTest, NoDistortionIdentityR) {
    MarkerUndistortCPUParams p;
    p.distortionModel = "brown_conrady";
    p.fx1 = 2500; p.fy1 = 2500; p.cx1 = 640; p.cy1 = 512;
    p.fx2 = 2500; p.fy2 = 2500; p.cx2 = 640; p.cy2 = 512;
    p.R = {1,0,0, 0,1,0, 0,0,1};
    p.T = {-120.0, 0, 0};
    p.imageWidth = 1280; p.imageHeight = 1024;

    MarkerUndistortCPU op(p);
    std::vector<cv::Point2d> pts1 = {{640,512},{300,256},{1000,768}};
    std::vector<cv::Point2d> pts2 = {{560,512},{220,256},{920,768}};

    auto result = op.Execute(pts1, pts2);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.pointCount1, 3);
    EXPECT_EQ(result.pointCount2, 3);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(result.rectifiedPoints1[i].y, result.rectifiedPoints2[i].y, 1e-6)
            << "Stereo rectified y-coordinates should be aligned at point " << i;
    }
}

// ---- 精度：与 OpenCV 直接调用对比 ----
TEST_F(MarkerUndistortCPUTest, MatchOpenCVReference) {
    MarkerUndistortCPU op(params_);
    std::vector<cv::Point2d> pts1 = {{640,512},{100,100},{1180,924},{320,256}};
    std::vector<cv::Point2d> pts2 = {{600,500},{80,90},{1150,900},{300,240}};

    cv::Size imgSize(params_.imageWidth, params_.imageHeight);
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(buildK1(), buildD1(), buildK2(), buildD2(),
                      imgSize, buildR(), buildT(), R1, R2, P1, P2, Q);

    auto toMat = [](const std::vector<cv::Point2d>& pts) {
        cv::Mat m(1, static_cast<int>(pts.size()), CV_64FC2);
        for (size_t i = 0; i < pts.size(); ++i)
            m.at<cv::Vec2d>(0, static_cast<int>(i)) = cv::Vec2d(pts[i].x, pts[i].y);
        return m;
    };

    cv::Mat dst1, dst2;
    cv::undistortPoints(toMat(pts1), dst1, buildK1(), buildD1(), R1, P1);
    cv::undistortPoints(toMat(pts2), dst2, buildK2(), buildD2(), R2, P2);

    auto result = op.Execute(pts1, pts2);
    ASSERT_TRUE(result.success);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.rectifiedPoints1[i].x, dst1.at<cv::Vec2d>(0,i)[0], 1e-10)
            << "Cam1 point " << i << " x";
        EXPECT_NEAR(result.rectifiedPoints1[i].y, dst1.at<cv::Vec2d>(0,i)[1], 1e-10)
            << "Cam1 point " << i << " y";
        EXPECT_NEAR(result.rectifiedPoints2[i].x, dst2.at<cv::Vec2d>(0,i)[0], 1e-10)
            << "Cam2 point " << i << " x";
        EXPECT_NEAR(result.rectifiedPoints2[i].y, dst2.at<cv::Vec2d>(0,i)[1], 1e-10)
            << "Cam2 point " << i << " y";
    }
}

// ---- 矫正后 y 对齐验证（对应点） ----
TEST_F(MarkerUndistortCPUTest, RectifiedYCoordinatesAlignedWithCorrespondingPoints) {
    MarkerUndistortCPU op(params_);

    cv::Size imgSize(params_.imageWidth, params_.imageHeight);
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(buildK1(), buildD1(), buildK2(), buildD2(),
                      imgSize, buildR(), buildT(), R1, R2, P1, P2, Q);

    cv::Mat Tvec = buildT();
    cv::Mat Rmat = buildR();
    std::vector<cv::Point3d> pts3d = {
        {0, 0, 2000}, {100, -50, 1500}, {-80, 30, 2500}, {50, 50, 3000}
    };
    std::vector<cv::Point2d> pts1, pts2;
    cv::Mat rvec = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat tvec1 = cv::Mat::zeros(3,1,CV_64F);
    cv::projectPoints(cv::Mat(pts3d).reshape(3), rvec, tvec1, buildK1(), buildD1(), pts1);
    cv::projectPoints(cv::Mat(pts3d).reshape(3), Rmat, Tvec, buildK2(), buildD2(), pts2);

    auto result = op.Execute(pts1, pts2);
    ASSERT_TRUE(result.success);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.rectifiedPoints1[i].y, result.rectifiedPoints2[i].y, 0.5)
            << "Corresponding point " << i << " y should align after rectification";
    }
}

// ---- EdgePoint 重载 ----
TEST_F(MarkerUndistortCPUTest, EdgePointOverload) {
    MarkerUndistortCPU op(params_);
    std::vector<EdgePoint> ep1 = {EdgePoint{640,512,0,100,640,512}};
    std::vector<EdgePoint> ep2 = {EdgePoint{600,500,0,100,600,500}};

    auto result = op.Execute(ep1, ep2);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.pointCount1, 1);
    EXPECT_EQ(result.pointCount2, 1);
}

// ---- 大量点 ----
TEST_F(MarkerUndistortCPUTest, LargeNumberOfStereoPoints) {
    MarkerUndistortCPUParams p = params_;
    p.k1_1 = 0; p.k2_1 = 0; p.k1_2 = 0; p.k2_2 = 0;

    MarkerUndistortCPU op(p);
    const int N = 5000;
    std::vector<cv::Point2d> pts1, pts2;
    pts1.reserve(N); pts2.reserve(N);
    for (int i = 0; i < N; ++i) {
        pts1.emplace_back(static_cast<double>(i % 1280), static_cast<double>(i % 1024));
        pts2.emplace_back(static_cast<double>((i + 50) % 1280), static_cast<double>(i % 1024));
    }
    auto result = op.Execute(pts1, pts2);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.pointCount1, N);
    EXPECT_EQ(result.pointCount2, N);
}

// ---- R1/R2/P1/P2 输出 ----
TEST_F(MarkerUndistortCPUTest, OutputsR1R2P1P2) {
    MarkerUndistortCPU op(params_);
    std::vector<cv::Point2d> pts1 = {{640,512}};
    std::vector<cv::Point2d> pts2 = {{600,500}};

    auto result = op.Execute(pts1, pts2);
    ASSERT_TRUE(result.success);
    EXPECT_FALSE(result.R1.empty());
    EXPECT_FALSE(result.R2.empty());
    EXPECT_FALSE(result.P1.empty());
    EXPECT_FALSE(result.P2.empty());
    EXPECT_EQ(result.R1.rows, 3);
    EXPECT_EQ(result.R1.cols, 3);
    EXPECT_EQ(result.P1.rows, 3);
    EXPECT_EQ(result.P1.cols, 4);
}

// ---- setParams / getParams ----
TEST_F(MarkerUndistortCPUTest, SetParamsAndGetParams) {
    MarkerUndistortCPU op(params_);
    EXPECT_DOUBLE_EQ(op.GetParams().fx1, params_.fx1);
}

TEST_F(MarkerUndistortCPUTest, SetParamsUpdatesAndReflected) {
    MarkerUndistortCPU op(params_);
    MarkerUndistortCPUParams np;
    np.fx1 = 3000; np.fy1 = 3000; np.cx1 = 800; np.cy1 = 600;
    np.fx2 = 3100; np.fy2 = 3100; np.cx2 = 810; np.cy2 = 610;
    np.R = {1,0,0, 0,1,0, 0,0,1};
    np.T = {-100.0, 0, 0};
    np.imageWidth = 1280; np.imageHeight = 1024;
    op.SetParams(np);
    EXPECT_DOUBLE_EQ(op.GetParams().fx1, 3000.0);
}

TEST_F(MarkerUndistortCPUTest, SetParamsInvalidThrows) {
    MarkerUndistortCPU op(params_);
    EXPECT_THROW(op.SetParams(MarkerUndistortCPUParams{}), std::invalid_argument);
}

// ---- 精度：有畸变有旋转 ----
class StereoUndistortPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.distortionModel = "brown_conrady";
        params_.fx1 = 2500; params_.fy1 = 2500; params_.cx1 = 640; params_.cy1 = 512;
        params_.fx2 = 2520; params_.fy2 = 2510; params_.cx2 = 645; params_.cy2 = 508;
        params_.k1_1 = -0.15; params_.k2_1 = 0.03; params_.p1_1 = 0.002; params_.p2_1 = -0.001;
        params_.k1_2 = -0.12; params_.k2_2 = 0.025; params_.p1_2 = 0.0015; params_.p2_2 = -0.0008;
        double angle = 0.02;
        params_.R = {cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1};
        params_.T = {-100.0, 0.3, 0.1};
        params_.imageWidth = 1280; params_.imageHeight = 1024;
    }

    MarkerUndistortCPUParams params_;
};

TEST_F(StereoUndistortPrecisionTest, WithRotationAndDistortion) {
    MarkerUndistortCPU op(params_);
    std::vector<cv::Point2d> pts1 = {{640,512},{100,100},{1180,924},{320,768}};
    std::vector<cv::Point2d> pts2 = {{610,500},{80,90},{1150,900},{300,750}};

    cv::Mat K1 = (cv::Mat_<double>(3,3) << params_.fx1,0,params_.cx1, 0,params_.fy1,params_.cy1, 0,0,1);
    cv::Mat D1 = (cv::Mat_<double>(5,1) << params_.k1_1,params_.k2_1,params_.p1_1,params_.p2_1,0);
    cv::Mat K2 = (cv::Mat_<double>(3,3) << params_.fx2,0,params_.cx2, 0,params_.fy2,params_.cy2, 0,0,1);
    cv::Mat D2 = (cv::Mat_<double>(5,1) << params_.k1_2,params_.k2_2,params_.p1_2,params_.p2_2,0);
    cv::Mat Rmat = (cv::Mat_<double>(3,3) << params_.R[0],params_.R[1],params_.R[2],
        params_.R[3],params_.R[4],params_.R[5], params_.R[6],params_.R[7],params_.R[8]);
    cv::Mat Tvec = (cv::Mat_<double>(3,1) << params_.T[0],params_.T[1],params_.T[2]);

    cv::Mat R1,R2,P1,P2,Q;
    cv::stereoRectify(K1,D1,K2,D2,cv::Size(params_.imageWidth,params_.imageHeight),
                      Rmat,Tvec,R1,R2,P1,P2,Q);

    auto toMat = [](const std::vector<cv::Point2d>& pts) {
        cv::Mat m(1,static_cast<int>(pts.size()),CV_64FC2);
        for(size_t i=0;i<pts.size();++i) m.at<cv::Vec2d>(0,static_cast<int>(i))=cv::Vec2d(pts[i].x,pts[i].y);
        return m;
    };

    cv::Mat dst1,dst2;
    cv::undistortPoints(toMat(pts1),dst1,K1,D1,R1,P1);
    cv::undistortPoints(toMat(pts2),dst2,K2,D2,R2,P2);

    auto result = op.Execute(pts1,pts2);
    ASSERT_TRUE(result.success);

    for(int i=0;i<4;++i) {
        EXPECT_NEAR(result.rectifiedPoints1[i].x, dst1.at<cv::Vec2d>(0,i)[0], 1e-10) << "Cam1 pt" << i << "x";
        EXPECT_NEAR(result.rectifiedPoints1[i].y, dst1.at<cv::Vec2d>(0,i)[1], 1e-10) << "Cam1 pt" << i << "y";
        EXPECT_NEAR(result.rectifiedPoints2[i].x, dst2.at<cv::Vec2d>(0,i)[0], 1e-10) << "Cam2 pt" << i << "x";
        EXPECT_NEAR(result.rectifiedPoints2[i].y, dst2.at<cv::Vec2d>(0,i)[1], 1e-10) << "Cam2 pt" << i << "y";
    }
}

TEST_F(StereoUndistortPrecisionTest, RectifiedYAlignedWithRotationWithCorrespondingPoints) {
    MarkerUndistortCPU op(params_);

    cv::Mat K1 = (cv::Mat_<double>(3,3) << params_.fx1,0,params_.cx1, 0,params_.fy1,params_.cy1, 0,0,1);
    cv::Mat D1 = (cv::Mat_<double>(5,1) << params_.k1_1,params_.k2_1,params_.p1_1,params_.p2_1,0);
    cv::Mat K2 = (cv::Mat_<double>(3,3) << params_.fx2,0,params_.cx2, 0,params_.fy2,params_.cy2, 0,0,1);
    cv::Mat D2 = (cv::Mat_<double>(5,1) << params_.k1_2,params_.k2_2,params_.p1_2,params_.p2_2,0);
    cv::Mat Rmat = (cv::Mat_<double>(3,3) << params_.R[0],params_.R[1],params_.R[2],
        params_.R[3],params_.R[4],params_.R[5], params_.R[6],params_.R[7],params_.R[8]);
    cv::Mat Tvec = (cv::Mat_<double>(3,1) << params_.T[0],params_.T[1],params_.T[2]);

    std::vector<cv::Point3d> pts3d = {{0,0,2000}, {100,-50,1500}, {-80,30,2500}};
    std::vector<cv::Point2d> pts1, pts2;
    cv::Mat rvec0 = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat tvec0 = cv::Mat::zeros(3,1,CV_64F);
    cv::projectPoints(cv::Mat(pts3d).reshape(3), rvec0, tvec0, K1, D1, pts1);
    cv::projectPoints(cv::Mat(pts3d).reshape(3), Rmat, Tvec, K2, D2, pts2);

    auto result = op.Execute(pts1, pts2);
    ASSERT_TRUE(result.success);
    for(int i=0;i<3;++i) {
        EXPECT_NEAR(result.rectifiedPoints1[i].y, result.rectifiedPoints2[i].y, 0.5)
            << "Corresponding point " << i << " y after rectification";
    }
}

// ---- groupIds 透传 + splitByGroup ----
TEST_F(MarkerUndistortCPUTest, GroupIdsPassthroughWithPoint2d) {
    MarkerUndistortCPU op(params_);
    std::vector<cv::Point2d> pts1 = {{640,512},{300,256},{1000,768},{500,400}};
    std::vector<cv::Point2d> pts2 = {{600,500},{250,240},{960,760},{480,380}};
    std::vector<int> gids1 = {0, 0, 1, 1};
    std::vector<int> gids2 = {0, 1, 0, 1};

    auto result = op.Execute(pts1, pts2, gids1, gids2);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.groupIds1, gids1);
    EXPECT_EQ(result.groupIds2, gids2);
    EXPECT_EQ(result.groupCount1, 2);
    EXPECT_EQ(result.groupCount2, 2);
}

TEST_F(MarkerUndistortCPUTest, GroupIdsDefaultEmptyWhenNotProvided) {
    MarkerUndistortCPU op(params_);
    std::vector<cv::Point2d> pts1 = {{640,512}};
    std::vector<cv::Point2d> pts2 = {{600,500}};

    auto result = op.Execute(pts1, pts2);
    ASSERT_TRUE(result.success);
    EXPECT_TRUE(result.groupIds1.empty());
    EXPECT_TRUE(result.groupIds2.empty());
    EXPECT_EQ(result.groupCount1, 0);
    EXPECT_EQ(result.groupCount2, 0);
}

TEST_F(MarkerUndistortCPUTest, SplitRectifiedPointsByGroup) {
    MarkerUndistortCPU op(params_);
    std::vector<cv::Point2d> pts1 = {{640,512},{300,256},{1000,768},{500,400}};
    std::vector<cv::Point2d> pts2 = {{600,500},{250,240},{960,760},{480,380}};
    std::vector<int> gids1 = {0, 0, 1, 1};
    std::vector<int> gids2 = {0, 1, 0, 1};

    auto result = op.Execute(pts1, pts2, gids1, gids2);
    ASSERT_TRUE(result.success);

    auto leftGroups = result.splitRectifiedPoints1ByGroup();
    ASSERT_EQ(leftGroups.size(), 2u);
    EXPECT_EQ(leftGroups[0].size(), 2u);
    EXPECT_EQ(leftGroups[1].size(), 2u);

    auto rightGroups = result.splitRectifiedPoints2ByGroup();
    ASSERT_EQ(rightGroups.size(), 2u);
    EXPECT_EQ(rightGroups[0].size(), 2u);
    EXPECT_EQ(rightGroups[1].size(), 2u);
}

TEST_F(MarkerUndistortCPUTest, SplitReturnsEmptyWhenNoGroupIds) {
    MarkerUndistortCPU op(params_);
    std::vector<cv::Point2d> pts1 = {{640,512}};
    std::vector<cv::Point2d> pts2 = {{600,500}};

    auto result = op.Execute(pts1, pts2);
    EXPECT_TRUE(result.splitRectifiedPoints1ByGroup().empty());
    EXPECT_TRUE(result.splitRectifiedPoints2ByGroup().empty());
}

TEST_F(MarkerUndistortCPUTest, EdgePointOverloadWithGroupIds) {
    MarkerUndistortCPU op(params_);
    std::vector<EdgePoint> ep1 = {
        EdgePoint{640,512,0,100,640,512},
        EdgePoint{300,256,0,100,300,256}
    };
    std::vector<EdgePoint> ep2 = {
        EdgePoint{600,500,0,100,600,500},
        EdgePoint{250,240,0,100,250,240}
    };
    std::vector<int> gids = {0, 1};

    auto result = op.Execute(ep1, ep2, gids, gids);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.groupIds1, gids);
    EXPECT_EQ(result.groupCount1, 2);

    auto groups = result.splitRectifiedPoints1ByGroup();
    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].size(), 1u);
    EXPECT_EQ(groups[1].size(), 1u);
}

// ---- 外部矫正矩阵（温度补偿表路径）----

TEST_F(MarkerUndistortCPUTest, ExternalRectifyMatrices_MatchesInternal) {
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(buildK1(), buildD1(), buildK2(), buildD2(),
                      cv::Size(params_.imageWidth, params_.imageHeight),
                      buildR(), buildT(), R1, R2, P1, P2, Q);

    std::vector<cv::Point2d> pts1 = {{640, 512}, {700, 400}, {500, 600}};
    std::vector<cv::Point2d> pts2 = {{600, 512}, {660, 400}, {460, 600}};

    MarkerUndistortCPU opA(params_);
    auto rA = opA.Execute(pts1, pts2);

    MarkerUndistortCPU opB(params_);
    opB.SetRectifyMatrices(R1, R2, P1, P2, Q);
    auto rB = opB.Execute(pts1, pts2);

    ASSERT_TRUE(rA.success && rB.success);

    // Q passthrough
    ASSERT_FALSE(rB.Q.empty());
    EXPECT_EQ(cv::countNonZero(rB.Q.reshape(1) != Q.reshape(1)), 0);

    // Identical rectified points (same matrices → same result)
    ASSERT_EQ(rA.rectifiedPoints1.size(), rB.rectifiedPoints1.size());
    for (size_t i = 0; i < rA.rectifiedPoints1.size(); ++i) {
        EXPECT_NEAR(rA.rectifiedPoints1[i].x, rB.rectifiedPoints1[i].x, 1e-10);
        EXPECT_NEAR(rA.rectifiedPoints1[i].y, rB.rectifiedPoints1[i].y, 1e-10);
        EXPECT_NEAR(rA.rectifiedPoints2[i].x, rB.rectifiedPoints2[i].x, 1e-10);
    }
}

TEST_F(MarkerUndistortCPUTest, ClearRectifyMatrices_RevertsToInternal) {
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(buildK1(), buildD1(), buildK2(), buildD2(),
                      cv::Size(params_.imageWidth, params_.imageHeight),
                      buildR(), buildT(), R1, R2, P1, P2, Q);

    std::vector<cv::Point2d> pts1 = {{640, 512}}, pts2 = {{600, 512}};

    MarkerUndistortCPU op(params_);
    op.SetRectifyMatrices(R1, R2, P1, P2, Q);
    auto rExt = op.Execute(pts1, pts2);

    op.ClearRectifyMatrices();
    auto rInt = op.Execute(pts1, pts2);

    ASSERT_TRUE(rExt.success && rInt.success);
    EXPECT_NEAR(rExt.rectifiedPoints1[0].x, rInt.rectifiedPoints1[0].x, 1e-10);
    EXPECT_NEAR(rExt.rectifiedPoints1[0].y, rInt.rectifiedPoints1[0].y, 1e-10);
}

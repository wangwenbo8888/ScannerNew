#pragma once
#include <opencv2/core.hpp>
namespace calib {
// 【E3】cv::fitEllipse(5 点 CV_32F) 的 vendor 逐位一致实现（OpenCV 4.13
// shapedescr.cpp fitEllipseDirect 转录；稳态零堆分配）。含奇异降级路径
// fitEllipseNoDirect 的 vendor 版（file-static 非公开 API，一并转录）。
// RNG 契约：getOfs 每点调用 theRNG()（含 eps=0 轮）——调用模式逐字保留。
cv::RotatedRect fitEllipse5Direct(const cv::Point2f* pts5);
}

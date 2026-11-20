#pragma once
// ============================================================================
// ccl_cpu.h — 【150FPS·方案 A】连通域分析 CPU 单线程路径（RLE＋并查集）
// ============================================================================
// 背景：GPU 版 RegionAnalyzerCUDA 串行参与每帧 6.8ms（自身 3.9-5.1ms＋三次
// cudaDeviceSynchronize 把衔接空档撑大 2.6ms）。标记点掩膜前景稀疏（~10-25
// 个圆斑），CPU 行程编码＋并查集单遍即可 0.4-1.2ms 完成同等统计。
//
// 语义对齐（保证金样逐字节一致）：
// - 8 连通；面积过滤 [minArea, maxArea]；topXCount>0 时按面积取前 X（保序）
// - 输出组件顺序＝标签升序＝光栅首遇顺序（对齐 GPU 版 buildRemap 升序遍历）
// - d_labeledMask 留空（流水线不消费；dump 有空判保护）
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/vision/ccl/region_analyze_cuda.h"

namespace scr {

/// CPU 连通域分析：mask(CV_8UC1, 非0=前景) → RegionAnalysisResult（仅 components
/// 统计有效；d_labeledMask 为空）。失败返 success=false 填 message。
calib::RegionAnalysisResult runCclCpu(const cv::Mat& mask, int minArea, int maxArea,
                                      int topXCount);

/// 【150FPS】行程直入变体：runs 三元组 [row,x0,x1] ×nRuns（行主序＝GPU-RLE 或
/// 逐像素 RLE 的同一序）→ 同一并查集/过滤/输出序（与 runCclCpu 语义逐位一致）
calib::RegionAnalysisResult runCclCpuFromRuns(const std::vector<int>& runsTriple,
                                              int nRuns, int W, int H,
                                              int minArea, int maxArea, int topXCount);

} // namespace scr

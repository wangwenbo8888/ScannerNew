// laser_calib_cli.cpp — 模块2 激光标定 CLI（Task 6.2 分阶段实现）
//
// 当前进度: 6.2-a (4-1 mask_extract + 4-2 region_analyze + 4-3 laser_label + 主循环骨架)
//   后续: 6.2-b (4-4~4-6) / 6.2-c (4-7~4-8 + host 累积) / 6.2-d (4-9~4-11) / 6.2-e (5-3 + 4-13)
//
// 设计依据: docs/plans/2026-07-18-factory-calib-impl.md Task 6.2 Step 0
// 算子签名以 Step 0.1 速查表为准；原 Step 1 伪代码禁止照抄。

#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"

#include <opencv2/core/cuda.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

using namespace fc;
using namespace calib;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: laser_calib <input_dir> [output_json]\n"
                  << "  input_dir 含 config.json + camera_calib.json + pose_*/L_tube*.png + R_tube*.png\n";
        return 2;
    }
    std::string inDir = argv[1];
    std::string outPath = argc >= 3 ? argv[2] : "laser_calib.json";

    spdlog::info("=== laser_calib (build 6.2-a) ===");

    // ------------------------------------------------------------------
    // 1. 加载输入 + 一致性校验
    // ------------------------------------------------------------------
    auto input = loadLaserInput(inDir);
    if (!input) {
        spdlog::error("load laser input failed");
        return 1;
    }
    const auto& cfg = input->config;
    const auto& h = input->handoff;

    std::string why;
    if (!validateHandoffConsistency(cfg, h, why)) {
        spdlog::error("handoff inconsistent: {}", why);
        return 1;
    }

    spdlog::info("poses={}, imageSize={}x{}, referenceTemp={:.2f}",
                 input->poseFrames.size(),
                 h.imageSize.width, h.imageSize.height,
                 h.referenceTemp);

    // ------------------------------------------------------------------
    // 2. 构造 4-1/4-2/4-3 算子 (L 和 R 各一独立实例)
    //    参数: 当前用默认值或从 cfg 取 deviceId
    //    TODO 6.2-b: 从 config.json 扩展 mask threshold/erodeSize 等可配项
    // ------------------------------------------------------------------
    cv::cuda::Stream stream;

    MaskExtractParams maskParams;
    // maskParams.threshold / erodeSize / ... 用默认值 (Task 6.2-a)
    MaskExtractCUDA maskL(maskParams);
    MaskExtractCUDA maskR(maskParams);
    spdlog::info("4-1 MaskExtractCUDA x2 (L/R) constructed");

    RegionAnalyzerParams cclParams;
    cclParams.deviceId = cfg.deviceId;
    RegionAnalyzerCUDA cclL(cclParams);
    RegionAnalyzerCUDA cclR(cclParams);
    spdlog::info("4-2 RegionAnalyzerCUDA x2 (L/R) constructed");

    LaserLabelParams labelParams;
    labelParams.deviceId = cfg.deviceId;
    LaserLabelerCUDA labelL(labelParams);
    LaserLabelerCUDA labelR(labelParams);
    spdlog::info("4-3 LaserLabelerCUDA x2 (L/R) constructed");

    // ------------------------------------------------------------------
    // 3. 主循环: pose × tube, 跑 4-1 ~ 4-3
    //    6.2-a 只跑到 label, 后续 4-4+ 留给 6.2-b/c/d/e
    // ------------------------------------------------------------------
    size_t framesOk = 0;
    size_t framesSkip = 0;

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi) {
        const auto& tubes = input->poseFrames[pi];
        for (size_t ti = 0; ti < tubes.size(); ++ti) {
            const auto& f = tubes[ti];

            // ----- 4-1 mask_extract (L + R) -----
            // Execute(const cv::Mat& gray, Stream&) → MaskExtractResult{d_grayImage, d_cleanedMask, ...}
            auto maskResL = maskL.Execute(f.leftGray, stream);
            auto maskResR = maskR.Execute(f.rightGray, stream);
            if (!maskResL.success || !maskResR.success) {
                spdlog::warn("pose {} tube {}: 4-1 mask failed (L={}, R={}), skip",
                             pi, ti, maskResL.success, maskResR.success);
                ++framesSkip;
                continue;
            }
            if (!maskResL.d_cleanedMask || !maskResR.d_cleanedMask
                || maskResL.d_cleanedMask->empty() || maskResR.d_cleanedMask->empty()) {
                spdlog::warn("pose {} tube {}: 4-1 mask empty, skip", pi, ti);
                ++framesSkip;
                continue;
            }

            // ----- 4-2 region_analyze (L + R) -----
            // Execute(const shared_ptr<GpuMat>& d_mask, Stream&) → RegionAnalysisResult{d_labeledMask CV_32SC1, components}
            auto cclResL = cclL.Execute(maskResL.d_cleanedMask, stream);
            auto cclResR = cclR.Execute(maskResR.d_cleanedMask, stream);
            if (!cclResL.success || !cclResR.success) {
                spdlog::warn("pose {} tube {}: 4-2 ccl failed (L={}, R={}), skip",
                             pi, ti, cclResL.success, cclResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-3 laser_label (L + R) -----
            // Execute(const GpuMat& d_inputMask, Stream&) 输入 CV_32SC1 (来自 4-2)
            // → LaserLabelResult{d_labeledMask (重编号 CV_32SC1)}
            if (!cclResL.d_labeledMask || !cclResR.d_labeledMask) {
                spdlog::warn("pose {} tube {}: 4-2 d_labeledMask null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto labelResL = labelL.Execute(*cclResL.d_labeledMask, stream);
            auto labelResR = labelR.Execute(*cclResR.d_labeledMask, stream);
            if (!labelResL.success || !labelResR.success) {
                spdlog::warn("pose {} tube {}: 4-3 label failed (L={}, R={}), skip",
                             pi, ti, labelResL.success, labelResR.success);
                ++framesSkip;
                continue;
            }

            ++framesOk;
            spdlog::info("pose {} tube {}: OK (L ccl={}, R ccl={})",
                         pi, ti, cclResL.componentCount, cclResR.componentCount);
        }
    }

    spdlog::info("loop done: {} ok, {} skipped", framesOk, framesSkip);

    // ------------------------------------------------------------------
    // 4. 资源销毁 (算子规范要求析构前显式 Destroy)
    // ------------------------------------------------------------------
    maskL.Destroy(); maskR.Destroy();
    cclL.Destroy();  cclR.Destroy();
    labelL.Destroy(); labelR.Destroy();

    // ------------------------------------------------------------------
    // 5. 6.2-a 占位输出 (后续 6.2-e 替换为真实 laser_calib.json)
    // ------------------------------------------------------------------
    nlohmann::json j;
    j["schema"]  = "factory_calib.laser_calib.v1";
    j["build"]   = "6.2-a-partial";
    j["posesProcessed"] = input->poseFrames.size();
    j["framesOk"]       = framesOk;
    j["framesSkipped"]  = framesSkip;
    if (!writeJson(outPath, j)) {
        spdlog::error("cannot write output: {}", outPath);
        return 1;
    }
    spdlog::info("laser_calib (6.2-a partial) -> {}", outPath);
    return 0;
}

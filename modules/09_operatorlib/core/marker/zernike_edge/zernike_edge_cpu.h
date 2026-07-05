/**
 * @file zernike_edge_cpu.h
 * @brief Zernike椭圆边缘亚像素提取算�?- 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2④（共用�? * 平台：CPU
 *
 * 功能：对输入灰度子图执行 Canny 边缘检测，再利�?Zernike 矩算�? *       对边缘像素进行亚像素级精确定位，输出边缘点坐标、方向和幅值�? *
 * 精度容差档次：档次①（亚像素级，~0.1-0.2 pixel�? */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

struct ZernikeEdgeCPUParams {
    double cannyLowThreshold = 50.0;
    double cannyHighThreshold = 150.0;
    int gaussianKernelSize = 3;
    double gaussianSigma = 1.0;
    int templateSize = 5;
    double edgeStrengthThreshold = 20.0;
    int sobelApertureSize = 3;

    void validate() const {
        if (templateSize != 5 && templateSize != 7) {
            throw std::invalid_argument("templateSize must be 5 or 7");
        }
        if (cannyLowThreshold <= 0) {
            throw std::invalid_argument("cannyLowThreshold must be > 0");
        }
        if (cannyHighThreshold <= 0) {
            throw std::invalid_argument("cannyHighThreshold must be > 0");
        }
        if (cannyLowThreshold >= cannyHighThreshold) {
            throw std::invalid_argument("cannyLowThreshold must be < cannyHighThreshold");
        }
        if (gaussianKernelSize <= 0 || gaussianKernelSize % 2 == 0) {
            throw std::invalid_argument("gaussianKernelSize must be a positive odd number");
        }
        if (gaussianSigma <= 0) {
            throw std::invalid_argument("gaussianSigma must be > 0");
        }
        if (edgeStrengthThreshold < 0) {
            throw std::invalid_argument("edgeStrengthThreshold must be >= 0");
        }
        if (sobelApertureSize != 3 && sobelApertureSize != 5 && sobelApertureSize != 7) {
            throw std::invalid_argument("sobelApertureSize must be 3, 5 or 7");
        }
    }

    nlohmann::json toJson() const {
        return {
            {"cannyLowThreshold", cannyLowThreshold},
            {"cannyHighThreshold", cannyHighThreshold},
            {"gaussianKernelSize", gaussianKernelSize},
            {"gaussianSigma", gaussianSigma},
            {"templateSize", templateSize},
            {"edgeStrengthThreshold", edgeStrengthThreshold},
            {"sobelApertureSize", sobelApertureSize}
        };
    }

    static ZernikeEdgeCPUParams fromJson(const nlohmann::json& j) {
        ZernikeEdgeCPUParams p;
        if (j.contains("cannyLowThreshold")) p.cannyLowThreshold = j.at("cannyLowThreshold").get<double>();
        if (j.contains("cannyHighThreshold")) p.cannyHighThreshold = j.at("cannyHighThreshold").get<double>();
        if (j.contains("gaussianKernelSize")) p.gaussianKernelSize = j.at("gaussianKernelSize").get<int>();
        if (j.contains("gaussianSigma")) p.gaussianSigma = j.at("gaussianSigma").get<double>();
        if (j.contains("templateSize")) p.templateSize = j.at("templateSize").get<int>();
        if (j.contains("edgeStrengthThreshold")) p.edgeStrengthThreshold = j.at("edgeStrengthThreshold").get<double>();
        if (j.contains("sobelApertureSize")) p.sobelApertureSize = j.at("sobelApertureSize").get<int>();
        return p;
    }
};

struct ZernikeEdgeCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<EdgePoint> edgePoints;
    int edgeCount = 0;
    cv::Mat cannyEdgeImage;

    ZernikeEdgeCPUResult() = default;
    ~ZernikeEdgeCPUResult() = default;

    ZernikeEdgeCPUResult(ZernikeEdgeCPUResult&&) = default;
    ZernikeEdgeCPUResult& operator=(ZernikeEdgeCPUResult&&) = default;

    ZernikeEdgeCPUResult(const ZernikeEdgeCPUResult&) = delete;
    ZernikeEdgeCPUResult& operator=(const ZernikeEdgeCPUResult&) = delete;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的暂存缓冲；SetParams 缓存 Canny/Zernike 阈值等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API ZernikeEdgeCPU {
public:
    static constexpr const char* kLogTag = "04-ZernikeEdgeCPU";

    explicit ZernikeEdgeCPU(const ZernikeEdgeCPUParams& params = {});

    ~ZernikeEdgeCPU();

    ZernikeEdgeCPU(const ZernikeEdgeCPU&) = delete;
    ZernikeEdgeCPU& operator=(const ZernikeEdgeCPU&) = delete;

    ZernikeEdgeCPUResult Execute(const cv::Mat& srcImage);

    void Warmup(int rows, int cols);
    void Warmup(const WarmupConfig& config);

    void SetParams(const ZernikeEdgeCPUParams& params);
    const ZernikeEdgeCPUParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getZernikeEdgeCPUInfo();

} // namespace calib
/**
 * @file image_split_cpu.h
 * @brief 标记点图像分割算�?- 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2❼（共用�? * 平台：CPU
 *
 * 功能：根�?ROI 矩形列表从源灰度图像中裁剪出子图像（深拷贝）�? *       支持边界安全裁剪模式和无边界检查的高速模式�? *
 * 精度容差档次：档次②（整像素/几何类）
 * - 输出为源图像 ROI 区域的精确深拷贝，无浮点运算，零误差
 */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

/**
 * @brief 图像分割参数
 *
 * 异常行为�? * - fromJson() 字段缺失或类型不匹配 �?抛出 std::invalid_argument
 * - fromJson() 值超出合法范�?�?抛出 std::invalid_argument（与 validate() 一致）
 * - fromJson() 未知字段 �?忽略（不抛异常），保证前向兼�? */
struct ImageSplitCPUParams {
    bool enableBoundaryCheck = false;

    void validate() const {}

    nlohmann::json toJson() const {
        return {
            {"enableBoundaryCheck", enableBoundaryCheck}
        };
    }

    static ImageSplitCPUParams fromJson(const nlohmann::json& j) {
        ImageSplitCPUParams p;
        if (j.contains("enableBoundaryCheck")) p.enableBoundaryCheck = j.at("enableBoundaryCheck").get<bool>();
        return p;
    }
};

/**
 * @brief 图像分割结果
 *
 * - success: 处理是否成功
 * - message: 错误或辅助信息（success=false 时保证非空）
 * - qualityFlag: 质量标记（success=true 时有效）
 *   - Normal:   正常分割
 *   - Degraded: 部分 ROI 越界被裁剪（�?enableBoundaryCheck=true 时）
 *   - Warning:  所�?ROI 均无效，输出为空
 * - splitImages: 分割后的子图像列表（深拷贝，�?Mat 独立�? * - splitCount: 实际分割成功的子图像数量
 */
struct ImageSplitCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<cv::Mat> splitImages;
    int splitCount = 0;

    ImageSplitCPUResult() = default;
    ~ImageSplitCPUResult() = default;

    ImageSplitCPUResult(ImageSplitCPUResult&&) = default;
    ImageSplitCPUResult& operator=(ImageSplitCPUResult&&) = default;

    ImageSplitCPUResult(const ImageSplitCPUResult&) = delete;
    ImageSplitCPUResult& operator=(const ImageSplitCPUResult&) = delete;
};

/**
 * @brief 标记点图像分割算子（CPU 实现�? *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 split() �?setParams()
 * - 单线程流水线中应在帧间隙调用 setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 * - Debug 模式下维�?inProcess_ 原子变量进行断言检�? * - Release 模式下并发调用为未定义行为（数据竞争�? */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的暂存缓冲；SetParams 仅缓存边界检查开关，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API ImageSplitCPU {
public:
    static constexpr const char* kLogTag = "08-ImageSplitCPU";

    explicit ImageSplitCPU(const ImageSplitCPUParams& params = {});

    ~ImageSplitCPU();

    ImageSplitCPU(const ImageSplitCPU&) = delete;
    ImageSplitCPU& operator=(const ImageSplitCPU&) = delete;

    /**
     * @brief 图像分割（核心方法）
     *
     * 输入灰度图像（CV_8UC1）和 ROI 矩形列表�?     * 对每�?ROI 执行深拷贝裁剪，返回子图像列表�?     *
    * @param srcImage 源灰度图像（CV_8UC1�? 
    * @param roiRects ROI 矩形列表
     * @return ImageSplitCPUResult 包含分割结果子图�?     */
    ImageSplitCPUResult Execute(const cv::Mat& srcImage,
                              const std::vector<cv::Rect>& roiRects);

    /**
     * @brief 预热 CPU 资源
     * @param rows 图像行数
     * @param cols 图像列数
     *
     * 预分配内部缓冲区，首�?split() 无冷启动延迟�?     */
    void Warmup(int rows, int cols);

    /**
     * @brief 使用统一配置预热
     * @param config 预热配置（来�?common/calib_warmup_config.h�?     */
    void Warmup(const WarmupConfig& config);

    /**
    * @brief 动态更新参�? 
    * @param params 新参数
    *
     * @warning 不得�?split() 并发调用
     */
    void SetParams(const ImageSplitCPUParams& params);

    /**
     * @brief 获取当前参数
     */
    const ImageSplitCPUParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getImageSplitCPUInfo();

} // namespace calib
/**
 * @file image_merge_cpu.h
 * @brief 标记点图像合并算�?- 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2⑨（共用�? * 平台：CPU
 *
 * 功能：根据分割小图的 ROI 坐标，将每个小图的亚像素边缘�? *       从局部坐标变换回大图坐标系，合并为统一的边缘点列表�? *
 * 精度容差档次：档次②（坐标偏移为整数加法，零浮点误差�? */

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
 * @brief 图像合并参数
 *
 * 无特殊参数，保留 JSON 框架以保持算子接口一致性�? *
 * 异常行为�? * - fromJson() 未知字段 �?忽略（不抛异常），保证前向兼�? */
struct ImageMergeCPUParams {
    void validate() const {}

    nlohmann::json toJson() const {
        return {};
    }

    static ImageMergeCPUParams fromJson(const nlohmann::json& /*j*/) {
        return ImageMergeCPUParams{};
    }
};

/**
 * @brief 图像合并结果
 *
 * - success: 处理是否成功
 * - message: 错误或辅助信息（success=false 时保证非空）
 * - qualityFlag: 质量标记（success=true 时有效）
 *   - Normal:   正常合并
 *   - Warning:  无子图数据可合并，输出为�? * - mergedEdgePoints: 大图坐标系的边缘点列�? * - mergedEdgeCount: 合并后的总边缘点�? * - groupIds: 每个边缘点所属的分组编号，与 mergedEdgePoints 一一对应
 *   groupIds[i] 表示 mergedEdgePoints[i] 来自第几组子图（0-indexed�? *   调用方可�?groupIds 拆分后分别调用下游算子（�?ellipse_fit_cpu�? * - groupCount: 总分组数（等于输入子图数量）
 */
struct ImageMergeCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<EdgePoint> mergedEdgePoints;
    int mergedEdgeCount = 0;
    std::vector<int> groupIds;
    int groupCount = 0;

    ImageMergeCPUResult() = default;
    ~ImageMergeCPUResult() = default;

    ImageMergeCPUResult(ImageMergeCPUResult&&) = default;
    ImageMergeCPUResult& operator=(ImageMergeCPUResult&&) = default;

    std::vector<std::vector<EdgePoint>> splitByGroup() const {
        std::vector<std::vector<EdgePoint>> groups(groupCount);
        for (size_t i = 0; i < mergedEdgePoints.size(); ++i) {
            groups[groupIds[i]].push_back(mergedEdgePoints[i]);
        }
        return groups;
    }

    ImageMergeCPUResult(const ImageMergeCPUResult&) = delete;
    ImageMergeCPUResult& operator=(const ImageMergeCPUResult&) = delete;
};

/**
 * @brief 标记点图像合并算子（CPU 实现�? *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 Execute() �?setParams()
 * - 单线程流水线中应在帧间隙调用 setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 * - Debug 模式下维�?inProcess_ 原子变量进行断言检�? * - Release 模式下并发调用为未定义行为（数据竞争�? */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的暂存缓冲；无实质参数，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API ImageMergeCPU {
public:
    static constexpr const char* kLogTag = "09-ImageMergeCPU";

    explicit ImageMergeCPU(const ImageMergeCPUParams& params = {});

    ~ImageMergeCPU();

    ImageMergeCPU(const ImageMergeCPU&) = delete;
    ImageMergeCPU& operator=(const ImageMergeCPU&) = delete;

    /**
     * @brief 图像合并（核心方法）
     *
     * 输入各子图的边缘点列表和对应�?ROI 矩形列表�?     * 将每个子图的边缘点坐标偏移到大图坐标系，合并输出�?     *
     * @param edgePointsPerSubImage 各子图的边缘点列表（局部坐标）
    * @param roiRects 各子图在大图中的 ROI 矩形（与 edgePointsPerSubImage 一一对应�? 
    * @return ImageMergeCPUResult 包含大图坐标系的合并边缘�?     */
    ImageMergeCPUResult Execute(const std::vector<std::vector<EdgePoint>>& edgePointsPerSubImage,
                               const std::vector<cv::Rect>& roiRects);

    /**
     * @brief 预热 CPU 资源
     * @param rows 大图行数
     * @param cols 大图列数
     *
     * 预分配内部缓冲区，首�?Execute() 无冷启动延迟�?     */
    void Warmup(int rows, int cols);

    /**
     * @brief 使用统一配置预热
     * @param config 预热配置（来�?common/calib_warmup_config.h�?     */
    void Warmup(const WarmupConfig& config);

    /**
    * @brief 动态更新参�? 
    * @param params 新参数
    *
     * @warning 不得与 Execute() 并发调用
     */
    void SetParams(const ImageMergeCPUParams& params);

    /**
     * @brief 获取当前参数
     */
    const ImageMergeCPUParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getImageMergeCPUInfo();

} // namespace calib
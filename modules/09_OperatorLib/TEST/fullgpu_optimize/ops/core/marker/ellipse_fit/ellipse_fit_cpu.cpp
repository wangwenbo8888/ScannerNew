/**
 * @file ellipse_fit_cpu.cpp
 * @brief 椭圆拟合和中心提取算子 - 实现文件
 *
 * 算法流程：
 *   1. 输入亚像素边缘点集
 *   2. RANSAC 采样拟合椭圆，用代数距离筛选内点
 *   3. 对内点集执行 fitEllipseAMS（或 fitEllipse 降级）进行最终拟合
 *   4. 输出椭圆中心和几何参数
 */

#include "ellipse_fit_cpu.h"
#include "ellipse_fit_simd.h"   // 【E2】内点扫描 SSE2
#include "ellipse_fit_vendor.h" // 【E3】vendor fitEllipseDirect（5 点直连）
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <cassert>
#include <cmath>

using namespace calib;

OperatorInfo getEllipseFitCPUInfo() {
    return OperatorInfo{"EllipseFitCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(07, EllipseFitCPU);

struct EllipseFitCPU::Impl {
    EllipseFitCPUParams params_;
    bool warmed_up_ = false;
    int warmup_maxPoints_ = 0;

    // 【E1】Impl 暂存缓冲（每调用 clear 复用——落实头文件 §4 既有声明）：
    // 采样/内点/掩码全复用，消 RANSAC 热路径逐迭代堆分配。线程契约＝每 lane
    // 独立实例（跨线程并行须独立实例），成员无跨实例共享。
    std::vector<cv::Point2f> samplePts_;
    std::vector<cv::Point2f> finalInliers_;
    std::vector<cv::Point2f> nearPts_;
    std::vector<uint8_t> iterMask_, bestMask_;   // 【E4】最优内点掩码（O(1) 交换缓存）
    std::vector<uint8_t> nearMask_;              // 【E2】nearPts 扫描掩码
    const bool e2Legacy_ = (getenv("JMW_E2_LEGACY") != nullptr
                            && getenv("JMW_E2_LEGACY")[0] == '1');   // 【E2】SIMD 退路
    const bool e3Legacy_ = (getenv("JMW_E3_LEGACY") != nullptr
                            && getenv("JMW_E3_LEGACY")[0] == '1');   // 【E3】vendor 退路

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const EllipseFitCPUParams& params)
        : params_(params)
    {
        params_.validate();
    }

    ~Impl() = default;

    EllipseFitCPUResult ExecuteImpl(const std::vector<cv::Point2f>& subPixPoints) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "Concurrent call detected - NOT thread-safe!");
        struct ScopedFlag {
            std::atomic<bool>* flag;
            ScopedFlag(std::atomic<bool>* f) : flag(f) {}
            ~ScopedFlag() { flag->store(false); }
        };
        ScopedFlag guard(&inProcess_);
        inProcess_.store(true);
#endif

        EllipseFitCPUResult result;
        result.totalPointCount = static_cast<int>(subPixPoints.size());

        if (subPixPoints.size() < 5) {
            result.success = false;
            result.message = "Too few points for ellipse fitting (need >= 5)";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        cv::RotatedRect bestModel;
        int maxInliers = 0;
        cv::RNG rng(12345);

        for (int i = 0; i < params_.ransacIterations; ++i) {
            // 【E1】成员缓冲复用（原逐迭代 vector 构造/析构）；push 语义与 RNG
            // 调用序逐字不变（:74-76 原序）
            samplePts_.clear();
            for (int j = 0; j < 5; ++j) {
                samplePts_.push_back(subPixPoints[rng.uniform(0, static_cast<int>(subPixPoints.size()))]);
            }

            // 【E1】手写 5 点包围盒替 cv::boundingRect——OpenCV float 路径同为
            // cvFloor+min/max（min/max 精确可交换、floor 单调 → Rect 逐位同），
            // 免 InputArray/Mat 头适配分发开销
            int bxmin = cvFloor(samplePts_[0].x), bxmax = bxmin;
            int bymin = cvFloor(samplePts_[0].y), bymax = bymin;
            for (int j = 1; j < 5; ++j) {
                const int px = cvFloor(samplePts_[j].x), py = cvFloor(samplePts_[j].y);
                bxmin = std::min(bxmin, px); bxmax = std::max(bxmax, px);
                bymin = std::min(bymin, py); bymax = std::max(bymax, py);
            }
            if ((bxmax - bxmin + 1) * (bymax - bymin + 1) < 2)
                continue;

            cv::RotatedRect ell;
            try {
                ell = e3Legacy_
                    ? cv::fitEllipse(samplePts_)
                    : calib::fitEllipse5Direct(samplePts_.data());
            } catch (...) {
                continue;
            }

            if (ell.size.width < params_.minEllipseAxis || ell.size.height < params_.minEllipseAxis)
                continue;

            float ratio = ell.size.width / ell.size.height;
            if (ratio > params_.maxAxisRatio || ratio < 1.0 / params_.maxAxisRatio)
                continue;

            float a = ell.size.width / 2.0f;
            float b = ell.size.height / 2.0f;
            float a2 = a * a;
            float b2 = b * b;
            float rad = ell.angle * static_cast<float>(CV_PI) / 180.0f;
            float cos_a = std::cos(rad);
            float sin_a = std::sin(rad);

            // 【E1】阈值外提（同一次转换，比较语义不变）
            const float thr = static_cast<float>(params_.ransacThreshold);
            if (iterMask_.size() != subPixPoints.size()) {
                iterMask_.assign(subPixPoints.size(), 0);
                bestMask_.assign(subPixPoints.size(), 0);
            }
            int inliersCount;
            if (!e2Legacy_) {
                // 【E2】SSE2 扫描（lane＝点，divps 真除、表达式序与标量逐句对应
                // → 判定逐位同）；掩码随扫随写（E4 消费）
                inliersCount = countInliersSimd(
                    subPixPoints.data(), static_cast<int>(subPixPoints.size()),
                    ell.center.x, ell.center.y, cos_a, sin_a, a2, b2, thr,
                    iterMask_.data());
            } else {
                inliersCount = 0;
                for (size_t k = 0; k < subPixPoints.size(); ++k) {
                    const auto& pt = subPixPoints[k];
                    float dx = pt.x - ell.center.x;
                    float dy = pt.y - ell.center.y;
                    float x_rot = dx * cos_a + dy * sin_a;
                    float y_rot = -dx * sin_a + dy * cos_a;
                    float dist = std::abs((x_rot * x_rot) / a2
                                          + (y_rot * y_rot) / b2 - 1.0f);
                    const bool inlier = dist < thr;
                    iterMask_[k] = inlier ? 1 : 0;
                    if (inlier)
                        inliersCount++;
                }
            }

            if (inliersCount > maxInliers) {
                maxInliers = inliersCount;
                bestModel = ell;
                iterMask_.swap(bestMask_);   // 【E4】O(1) 换掩码（iterMask 下轮全量重写）
                if (maxInliers > static_cast<int>(subPixPoints.size() * params_.earlyStopRatio))
                    break;
                // 【精确早停】全点命中——严格 > 判定下后续迭代不可能再胜，
                // 未消费的 RNG 抽样只影响本就被丢弃的后续候选（rng 局部实例）＝等变
                if (maxInliers == static_cast<int>(subPixPoints.size()))
                    break;
            }
        }

        if (maxInliers < params_.minInliers) {
            result.success = false;
            result.message = "Insufficient inliers after RANSAC";
            result.qualityFlag = calib::QualityFlag::Degraded;
            return result;
        }

        // 【E4】最优内点集直接按 bestMask 收集——与原「重扫一遍」恒等：
        // 重扫处 cos/sin 输入＝bestModel.angle（获胜迭代时整体拷贝的同位值），
        // libm 确定性 → 逐点判定与获胜迭代的掩码逐位同（终审 PASS 论证）
        finalInliers_.clear();
        finalInliers_.reserve(maxInliers);
        for (size_t k = 0; k < subPixPoints.size(); ++k)
            if (bestMask_[k])
                finalInliers_.push_back(subPixPoints[k]);

        if (finalInliers_.size() < 5) {
            result.success = false;
            result.message = "Too few inliers for final fitting";
            result.qualityFlag = calib::QualityFlag::Degraded;
            return result;
        }

        cv::RotatedRect finalResult;
        try {
            if (params_.useAMS) {
                finalResult = cv::fitEllipseAMS(finalInliers_);
            } else {
                finalResult = cv::fitEllipse(finalInliers_);
            }
        } catch (...) {
            try {
                finalResult = cv::fitEllipse(finalInliers_);
            } catch (...) {
                result.success = false;
                result.message = "Ellipse fitting failed";
                result.qualityFlag = calib::QualityFlag::Degraded;
                return result;
            }
        }

        // 终拟合复检：AMS 结果可能比 RANSAC 候选拉长（内点集几何偏斜时），
        // 用与候选一致的质量闸把关——不合格判失败，避免输出错误中心
        if (finalResult.size.width < params_.minEllipseAxis ||
            finalResult.size.height < params_.minEllipseAxis) {
            result.success = false;
            result.message = "Final ellipse below minEllipseAxis";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }
        const double finalRatio = static_cast<double>(finalResult.size.width) /
                                  static_cast<double>(finalResult.size.height);
        if (finalRatio > params_.maxAxisRatio ||
            finalRatio < 1.0 / params_.maxAxisRatio) {
            result.success = false;
            result.message = "Final ellipse exceeds maxAxisRatio";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        // 形状闸（可选，0=不启用）：长轴=max(宽,高)，椭圆度=短轴/长轴
        const double longAxis = std::max(finalResult.size.width, finalResult.size.height);
        const double shortAxis = std::min(finalResult.size.width, finalResult.size.height);
        if (params_.maxLongAxis > 0.0 && longAxis > params_.maxLongAxis) {
            result.success = false;
            result.message = "Final ellipse long axis above maxLongAxis";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }
        if (params_.minLongAxis > 0.0 && longAxis < params_.minLongAxis) {
            result.success = false;
            result.message = "Final ellipse long axis below minLongAxis";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }
        if (params_.minRoundness > 0.0 && shortAxis / longAxis < params_.minRoundness) {
            result.success = false;
            result.message = "Final ellipse roundness below minRoundness";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        // 内点率复检：贴近最终椭圆的点 / 该组总点数 < minInlierRatio → 丢弃
        // （防少数点凑出的假共识，如粘连弧线只罩住零星点）
        nearPts_.clear();                     // 【E1】成员缓冲复用
        nearPts_.reserve(subPixPoints.size());
        {
            const float fa = finalResult.size.width / 2.0f;
            const float fb = finalResult.size.height / 2.0f;
            const float fa2 = fa * fa, fb2 = fb * fb;
            const float frad = finalResult.angle * static_cast<float>(CV_PI) / 180.0f;
            const float fcos = std::cos(frad), fsin = std::sin(frad);
            const float fthr = static_cast<float>(params_.ransacThreshold);
            if (!e2Legacy_) {
                // 【E2】SSE2 扫描＋按掩码原序收集（判定/次序与标量逐位同）
                if (nearMask_.size() != subPixPoints.size())
                    nearMask_.assign(subPixPoints.size(), 0);
                countInliersSimd(
                    subPixPoints.data(), static_cast<int>(subPixPoints.size()),
                    finalResult.center.x, finalResult.center.y,
                    fcos, fsin, fa2, fb2, fthr, nearMask_.data());
                for (size_t k = 0; k < subPixPoints.size(); ++k)
                    if (nearMask_[k])
                        nearPts_.push_back(subPixPoints[k]);
            } else {
                for (const auto& pt : subPixPoints) {
                    const float dx = pt.x - finalResult.center.x;
                    const float dy = pt.y - finalResult.center.y;
                    const float xr = dx * fcos + dy * fsin;
                    const float yr = -dx * fsin + dy * fcos;
                    if (std::abs((xr * xr) / fa2 + (yr * yr) / fb2 - 1.0f) < fthr)
                        nearPts_.push_back(pt);
                }
            }
            const double nearRatio =
                subPixPoints.empty()
                    ? 0.0
                    : static_cast<double>(nearPts_.size()) /
                          static_cast<double>(subPixPoints.size());
            if (nearRatio < params_.minInlierRatio) {
                result.success = false;
                result.message = "Inlier ratio below minInlierRatio";
                result.qualityFlag = calib::QualityFlag::Warning;
                return result;
            }
        }

        result.success = true;
        result.message = "Ellipse fitting successful";
        result.qualityFlag = calib::QualityFlag::Normal;
        result.centerX = finalResult.center.x;
        result.centerY = finalResult.center.y;
        result.majorAxis = finalResult.size.width;
        result.minorAxis = finalResult.size.height;
        result.angle = finalResult.angle;
        if (params_.outputInliersOnly) {
            // 可选输出过滤：只交付贴最终椭圆的亚像素点（带外点全删）
            // 【E1】成员缓冲不掏空——assign 拷贝输出（~1KB/组，可忽略）
            result.inlierCount = static_cast<int>(nearPts_.size());
            result.inlierPoints.assign(nearPts_.begin(), nearPts_.end());
        } else {
            result.inlierCount = static_cast<int>(finalInliers_.size());
            result.inlierPoints.assign(finalInliers_.begin(), finalInliers_.end());
        }
        return result;
    }

    void Warmup(int maxPointCount) {
        warmup_maxPoints_ = maxPointCount;
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: maxPointCount={}", maxPointCount);
    }

    void SetParams(const EllipseFitCPUParams& params) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() while processing - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        warmed_up_ = false;
    }

    const EllipseFitCPUParams& GetParams() const { return params_; }
};

// ============================================================
// 构造 / 析构
// ============================================================
EllipseFitCPU::EllipseFitCPU(const EllipseFitCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("EllipseFitCPU initialized (ransacIter={}, threshold={})",
                   params.ransacIterations, params.ransacThreshold);
}

EllipseFitCPU::~EllipseFitCPU() = default;

void EllipseFitCPU::Destroy() { }

// ============================================================
// fit(EdgePoint)
// ============================================================
EllipseFitCPUResult EllipseFitCPU::Execute(const std::vector<EdgePoint>& edgePoints) {
    CALIB_LOG_DEBUG("fit(EdgePoint) called: pointCount={}", edgePoints.size());
    std::vector<cv::Point2f> pts;
    pts.reserve(edgePoints.size());
    for (const auto& ep : edgePoints)
        pts.emplace_back(static_cast<float>(ep.x), static_cast<float>(ep.y));
    return pImpl_->ExecuteImpl(pts);
}

// ============================================================
// fit(Point2f)
// ============================================================
EllipseFitCPUResult EllipseFitCPU::Execute(const std::vector<cv::Point2f>& points) {
    CALIB_LOG_DEBUG("fit(Point2f) called: pointCount={}", points.size());
    return pImpl_->ExecuteImpl(points);
}

// ============================================================
// fit(Point2d)
// ============================================================
EllipseFitCPUResult EllipseFitCPU::Execute(const std::vector<cv::Point2d>& points) {
    CALIB_LOG_DEBUG("fit(Point2d) called: pointCount={}", points.size());
    std::vector<cv::Point2f> pts;
    pts.reserve(points.size());
    for (const auto& pt : points)
        pts.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    return pImpl_->ExecuteImpl(pts);
}

// ============================================================
// warmup
// ============================================================
void EllipseFitCPU::Warmup(int maxPointCount) {
    CALIB_LOG_INFO("warmup() called: maxPointCount={}", maxPointCount);
    pImpl_->Warmup(maxPointCount);
}

void EllipseFitCPU::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    Warmup(config.maxPointCount);
}

// ============================================================
// setParams / getParams
// ============================================================
void EllipseFitCPU::SetParams(const EllipseFitCPUParams& params) {
    CALIB_LOG_INFO("setParams() called");
    pImpl_->SetParams(params);
}

const EllipseFitCPUParams& EllipseFitCPU::GetParams() const {
    return pImpl_->GetParams();
}

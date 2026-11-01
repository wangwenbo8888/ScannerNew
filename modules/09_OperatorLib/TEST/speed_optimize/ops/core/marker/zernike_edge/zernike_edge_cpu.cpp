/**
 * @file zernike_edge_cpu.cpp
 * @brief Zernike椭圆边缘亚像素提取算子 - 实现文件
 *
 * CPU 算子无需 pImpl 隔离（无 CUDA 类型），但为与框架风格统一仍保留 Impl 结构。
 * 算法流水线：GaussianBlur → Canny → 边缘像素遍历 → Zernike 矩卷积 → 亚像素定位
 */

#include "zernike_edge_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <cassert>
#include <array>
#include <stdexcept>
#include <cmath>

using namespace calib;

OperatorInfo getZernikeEdgeCPUInfo() {
    return OperatorInfo{"ZernikeEdgeCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(04, ZernikeEdgeCPU);

namespace {

struct ZernikeTemplates {
    int size;
    std::vector<double> T00;
    std::vector<double> T11_re;
    std::vector<double> T11_im;
    std::vector<double> T20;

    static ZernikeTemplates create(int templateSize) {
        ZernikeTemplates t;
        t.size = templateSize;
        int N = templateSize;
        int half = N / 2;
        int count = N * N;

        t.T00.resize(count);
        t.T11_re.resize(count);
        t.T11_im.resize(count);
        t.T20.resize(count);

        double normRadius = static_cast<double>(half);

        for (int dy = -half; dy <= half; ++dy) {
            for (int dx = -half; dx <= half; ++dx) {
                int idx = (dy + half) * N + (dx + half);
                double x = static_cast<double>(dx) / normRadius;
                double y = static_cast<double>(dy) / normRadius;
                double r2 = x * x + y * y;

                if (r2 > 1.0) {
                    t.T00[idx] = 0.0;
                    t.T11_re[idx] = 0.0;
                    t.T11_im[idx] = 0.0;
                    t.T20[idx] = 0.0;
                    continue;
                }

                double r = std::sqrt(r2);
                double theta = std::atan2(y, x);
                double w = (r2 <= 1.0) ? 1.0 : 0.0;

                double pi = 3.14159265358979323846;

                t.T00[idx] = w * 1.0 / pi;

                t.T11_re[idx] = w * (2.0 / pi) * r * std::cos(theta);

                t.T11_im[idx] = w * (2.0 / pi) * r * std::sin(theta);

                t.T20[idx] = w * (std::sqrt(6.0) / pi) * (2.0 * r2 - 1.0);
            }
        }

        return t;
    }
};

double computeZernikeMoment(const cv::Mat& blurred, int cx, int cy,
                            const std::vector<double>& templateData, int templateSize)
{
    int half = templateSize / 2;
    double moment = 0.0;
    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            int iy = cy + dy;
            int ix = cx + dx;
            double pixel = 0.0;
            if (iy >= 0 && iy < blurred.rows && ix >= 0 && ix < blurred.cols) {
                pixel = static_cast<double>(blurred.at<uchar>(iy, ix));
            }
            moment += pixel * templateData[(dy + half) * templateSize + (dx + half)];
        }
    }
    return moment;
}

// 【zernike 优化 2026-10-19】三矩融合：原同一窗口跑 3 遍 computeZernikeMoment，
// 后融合为 Pair＋Z20 两函数；【Z2·本版】再并为下方 computeZernikeTriple 单趟
// （三矩独立累加器、dy 外 dx 内序不变 → 逐位一致）。
// computeZernikeMoment（上方）保留作模板数学参考。

#ifdef _MSC_VER
#include <intrin.h>
#include <immintrin.h>
#endif

// 【Z3】AVX2 运行时探测（一次性）。MSVC 编译 intrinsic 不受 /arch 门控，
// 旧机无 AVX2 会非法指令——必须运行时守卫；JMW_Z3_LEGACY=1 显式回退标量。
static bool zernikeAvx2Available()
{
#ifdef _MSC_VER
    static const bool ok = [] {
        int c[4];
        __cpuid(c, 0);
        if (c[0] < 7) return false;
        __cpuidex(c, 7, 0);
        return (c[1] & (1 << 5)) != 0;   // leaf7 EBX[5]＝AVX2
    }();
    return ok;
#else
    return false;
#endif
}

// 【Z3】点间 SIMD：4 点/向量（lane＝点），每点内部保持 25-tap 串行累加链
// → 每点结果与标量快路逐位一致（vmulpd/vaddpd 逐条 IEEE 舍入、禁 FMA；
// uchar→int32→double 两级转换均精确）。只用于 4 点全 interior 的组；
// 贴边点/尾点走 computeZernikeTriple 标量（慢路口径不变）。
inline void computeZernikeTripleSimd4(
    const cv::Mat& b, const int* xs, const int* ys,
    const double* tre, const double* tim, const double* tz2, int N, int half,
    double* out_re, double* out_im, double* out_z2)
{
#ifdef _MSC_VER
    __m256d are = _mm256_setzero_pd();
    __m256d aim = _mm256_setzero_pd();
    __m256d az2 = _mm256_setzero_pd();
    for (int dy = -half; dy <= half; ++dy) {
        const uchar* r0 = b.ptr<uchar>(ys[0] + dy);
        const uchar* r1 = b.ptr<uchar>(ys[1] + dy);
        const uchar* r2 = b.ptr<uchar>(ys[2] + dy);
        const uchar* r3 = b.ptr<uchar>(ys[3] + dy);
        const double* qre = tre + (size_t)(dy + half) * N + half;
        const double* qim = tim + (size_t)(dy + half) * N + half;
        const double* qz2 = tz2 + (size_t)(dy + half) * N + half;
        for (int dx = -half; dx <= half; ++dx) {
            // 每 tap 4 点字节现场组装（gather 仅 dword/qword 且延迟高——否决）
            __m128i px = _mm_set_epi32(static_cast<int>(r3[xs[3] + dx]),
                                        static_cast<int>(r2[xs[2] + dx]),
                                        static_cast<int>(r1[xs[1] + dx]),
                                        static_cast<int>(r0[xs[0] + dx]));
            const __m256d v = _mm256_cvtepi32_pd(px);   // uchar→double 精确
            are = _mm256_add_pd(are, _mm256_mul_pd(v, _mm256_set1_pd(qre[dx])));
            aim = _mm256_add_pd(aim, _mm256_mul_pd(v, _mm256_set1_pd(qim[dx])));
            az2 = _mm256_add_pd(az2, _mm256_mul_pd(v, _mm256_set1_pd(qz2[dx])));
        }
    }
    _mm256_storeu_pd(out_re, are);
    _mm256_storeu_pd(out_im, aim);
    _mm256_storeu_pd(out_z2, az2);
#else
    (void)b; (void)xs; (void)ys; (void)tre; (void)tim; (void)tz2;
    (void)N; (void)half; (void)out_re; (void)out_im; (void)out_z2;
#endif
}

// 【Z2·三矩同趟】z20 并入 z11 同一窗口行走：一次像素读同时累加三矩，
// 各矩独立累加器、dy 外 dx 内序与原 Pair/Z20 两函数严格一致 → 逐位一致。
// 被强度门滤除的点现在也会算 z20（多付寄存器内 1 mul+add/tap），但省掉
// 幸存点整趟第二窗口读＋边界判断——幸存率高时净赚（thr=20 下实测占大头）；
// 被滤点的 z20 从不进任何输出（原口径它们本不算 z20——多算不改结果）。
inline void computeZernikeTriple(const cv::Mat& blurred, int cx, int cy,
                                 const double* tre, const double* tim,
                                 const double* tz2, int N, int half,
                                 double& z11_re, double& z11_im, double& z20)
{
    z11_re = 0.0;
    z11_im = 0.0;
    z20 = 0.0;
    if (cx >= half && cx + half < blurred.cols
        && cy >= half && cy + half < blurred.rows) {
        for (int dy = -half; dy <= half; ++dy) {
            const uchar* row = blurred.ptr<uchar>(cy + dy);
            const double* pre = tre + (size_t)(dy + half) * N + half;
            const double* pim = tim + (size_t)(dy + half) * N + half;
            const double* pz2 = tz2 + (size_t)(dy + half) * N + half;
            for (int dx = -half; dx <= half; ++dx) {
                const double pixel = static_cast<double>(row[cx + dx]);
                z11_re += pixel * pre[dx];
                z11_im += pixel * pim[dx];
                z20 += pixel * pz2[dx];
            }
        }
        return;
    }
    for (int dy = -half; dy <= half; ++dy) {
        const int iy = cy + dy;
        const bool rowOk = (iy >= 0 && iy < blurred.rows);
        const uchar* row = rowOk ? blurred.ptr<uchar>(iy) : nullptr;
        const double* pre = tre + (size_t)(dy + half) * N + half;
        const double* pim = tim + (size_t)(dy + half) * N + half;
        const double* pz2 = tz2 + (size_t)(dy + half) * N + half;
        for (int dx = -half; dx <= half; ++dx) {
            const int ix = cx + dx;
            const double pixel = (row && ix >= 0 && ix < blurred.cols)
                ? static_cast<double>(row[ix]) : 0.0;
            z11_re += pixel * pre[dx];
            z11_im += pixel * pim[dx];
            z20 += pixel * pz2[dx];
        }
    }
}

} // anonymous namespace

struct ZernikeEdgeCPU::Impl {
    ZernikeEdgeCPUParams params_;
    ZernikeTemplates templates_;
    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;

    // 【Z4】Impl 持久暂存（每调用复用；头文件 §4 状态模型既有口径落实）：
    // 尺寸不变时零重配。线程契约＝每 lane 独立实例。
    cv::Mat blurredBuf_;
    cv::Mat cannyBuf_;
    std::vector<cv::Point> edgePixels_;
    std::vector<double> zre_, zim_, zz2_;   // 【Z3】点级矩值缓存（两趟结构中间量）

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const ZernikeEdgeCPUParams& params)
        : params_(params)
        , templates_(ZernikeTemplates::create(params.templateSize))
    {
        params_.validate();
    }

    ~Impl() = default;

    ZernikeEdgeCPUResult Execute(const cv::Mat& srcImage)
    {
#ifndef NDEBUG
        assert(!inProcess_.load() && "Concurrent Execute() calls detected - NOT thread-safe!");
        struct ScopedFlag {
            std::atomic<bool>* flag;
            ScopedFlag(std::atomic<bool>* f) : flag(f) {}
            ~ScopedFlag() { flag->store(false); }
        };
        ScopedFlag guard(&inProcess_);
        inProcess_.store(true);
#endif

        ZernikeEdgeCPUResult result;

        if (srcImage.empty()) {
            result.success = true;
            result.message = "Input image is empty";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        if (srcImage.type() != CV_8UC1) {
            result.success = false;
            result.message = "Input must be CV_8UC1 grayscale image";
            return result;
        }

        int minDim = std::min(srcImage.rows, srcImage.cols);
        if (minDim < params_.templateSize) {
            result.success = false;
            result.message = "Image too small for template size";
            return result;
        }

        // 【Z4】持久暂存复用（尺寸同则 OpenCV 免重配）
        cv::GaussianBlur(srcImage, blurredBuf_,
                          cv::Size(params_.gaussianKernelSize, params_.gaussianKernelSize),
                          params_.gaussianSigma);

        cv::Canny(blurredBuf_, cannyBuf_,
                  params_.cannyLowThreshold, params_.cannyHighThreshold,
                  params_.sobelApertureSize);

        // 【N1·2026-10-23】cannyEdgeImage 改开关产出（默认跳过）：活链路（image_merge
        // 只吃 edgePoints）无消费者——第三轮审核 N1 项，免 ~20 次/帧 clone＋
        // 0.3-0.8MB/帧拷贝。测试/诊断：JMW_Z6_CANNYIMG=1 恢复原语义（每 Result
        // 独占深拷贝——防同 lane 连续 Result 浅引用串写，见下原注释）
        // 原【Z4】注：暂存复用后浅引用会让同 lane 连续 Result 互相串写
        if (std::getenv("JMW_Z6_CANNYIMG"))
            result.cannyEdgeImage = cannyBuf_.clone();

        // 【Z4】手扫替 findNonZero：一趟、免 countNonZero 预趟＋逐调用 vector 分配；
        // 行主序（y 升 x 升）＝ cv::findNonZero 产出序（OpenCV 4.13 源码核实）——
        // 399 对拍＋M3 dump 逐位门把关
        edgePixels_.clear();
        for (int y = 0; y < cannyBuf_.rows; ++y) {
            const uchar* row = cannyBuf_.ptr<uchar>(y);
            for (int x = 0; x < cannyBuf_.cols; ++x)
                if (row[x]) edgePixels_.emplace_back(x, y);
        }
        const std::vector<cv::Point>& edgePixels = edgePixels_;

        if (edgePixels.empty()) {
            result.success = true;
            result.message = "No edge pixels detected by Canny";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        result.edgePoints.reserve(edgePixels.size());
        int filteredCount = 0;

        // 【zernike 优化】三矩融合单趟（快）＋滤除点免算 z20（结果逐位不变）
        const int N = templates_.size;
        const int half = N / 2;
        const double* tre = templates_.T11_re.data();
        const double* tim = templates_.T11_im.data();
        const double* tz2 = templates_.T20.data();

        // 【Z3】两趟结构：先全点算矩（SIMD/标量混合），后原序 emit——
        // 输出序＝输入序不动（emit 红线）；JMW_Z3_LEGACY=1 全标量回退
        const size_t nPts = edgePixels.size();
        zre_.resize(nPts);
        zim_.resize(nPts);
        zz2_.resize(nPts);
        const bool useSimd =
            zernikeAvx2Available()
            && !(getenv("JMW_Z3_LEGACY") != nullptr
                 && getenv("JMW_Z3_LEGACY")[0] == '1');
        auto interiorPt = [&](int x, int y) {
            return x >= half && x + half < blurredBuf_.cols
                && y >= half && y + half < blurredBuf_.rows;
        };
        size_t iP = 0;
        for (; useSimd && iP + 4 <= nPts; iP += 4) {
            if (interiorPt(edgePixels[iP].x, edgePixels[iP].y)
                && interiorPt(edgePixels[iP + 1].x, edgePixels[iP + 1].y)
                && interiorPt(edgePixels[iP + 2].x, edgePixels[iP + 2].y)
                && interiorPt(edgePixels[iP + 3].x, edgePixels[iP + 3].y)) {
                int xs4[4] = { edgePixels[iP].x, edgePixels[iP + 1].x,
                               edgePixels[iP + 2].x, edgePixels[iP + 3].x };
                int ys4[4] = { edgePixels[iP].y, edgePixels[iP + 1].y,
                               edgePixels[iP + 2].y, edgePixels[iP + 3].y };
                computeZernikeTripleSimd4(blurredBuf_, xs4, ys4, tre, tim, tz2,
                                          N, half, &zre_[iP], &zim_[iP], &zz2_[iP]);
            } else {
                for (int k = 0; k < 4; ++k)
                    computeZernikeTriple(blurredBuf_, edgePixels[iP + k].x,
                                         edgePixels[iP + k].y, tre, tim, tz2,
                                         N, half, zre_[iP + k], zim_[iP + k],
                                         zz2_[iP + k]);
            }
        }
        for (; iP < nPts; ++iP)
            computeZernikeTriple(blurredBuf_, edgePixels[iP].x, edgePixels[iP].y,
                                 tre, tim, tz2, N, half, zre_[iP], zim_[iP],
                                 zz2_[iP]);

        for (size_t k = 0; k < nPts; ++k) {
            const cv::Point& pt = edgePixels[k];
            const double z11_re = zre_[k];
            const double z11_im = zim_[k];
            const double z20 = zz2_[k];

            double mag = std::sqrt(z11_re * z11_re + z11_im * z11_im);

            if (mag < params_.edgeStrengthThreshold) {
                ++filteredCount;
                continue;
            }

            double angle = std::atan2(z11_im, z11_re);
            double l = 0.0;
            if (mag > 1e-10) {
                l = z20 / mag;
            }

            // 【E6·三角函数消元】l·cos(θ)=z20·re/mag²、l·sin(θ)=z20·im/mag²（θ=atan2(im,re)
            // 的恒等代入——双精度下与原式同簇 ~ulp 级，免去每点 cos/sin 两次超越函数；
            // angle 字段保留 atan2 写回（image_merge 逐字拷贝——下游值语义不变）
            double subX = static_cast<double>(pt.x);
            double subY = static_cast<double>(pt.y);
            const double mag2 = z11_re * z11_re + z11_im * z11_im;
            if (mag2 > 1e-20) {   // 【R3-低4】mag≈0 守卫（threshold=0 合法配置下防 inf/NaN）
                const double lk = z20 / mag2;   // = l/mag
                subX += lk * z11_re;
                subY += lk * z11_im;
            }

            EdgePoint ep;
            ep.x = subX;
            ep.y = subY;
            ep.angle = angle;
            ep.amplitude = mag;
            ep.pixelX = pt.x;
            ep.pixelY = pt.y;

            result.edgePoints.push_back(ep);
        }

        result.edgeCount = static_cast<int>(result.edgePoints.size());
        result.success = true;

        if (result.edgeCount == 0) {
            result.message = "All edge points filtered by strength threshold";
            result.qualityFlag = calib::QualityFlag::Warning;
        } else if (filteredCount > 0) {
            result.message = "Some edge points filtered by strength threshold";
            result.qualityFlag = calib::QualityFlag::Degraded;
        } else {
            result.message = "Sub-pixel edge extraction successful";
            result.qualityFlag = calib::QualityFlag::Normal;
        }

        CALIB_LOG_DEBUG("Execute() done: edgeCount={}, filtered={}",
                        result.edgeCount, filteredCount);

        return result;
    }

    void Warmup(int rows, int cols)
    {
        cv::Mat dummy = cv::Mat::zeros(rows, cols, CV_8UC1);
        cv::Mat blurred;
        cv::GaussianBlur(dummy, blurred, cv::Size(3, 3), 1.0);
        cv::Mat cannyOut;
        cv::Canny(blurred, cannyOut, 50, 150, 3);
        (void)cannyOut;

        warmup_rows_ = rows;
        warmup_cols_ = cols;
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: rows={}, cols={}", rows, cols);
    }

    void SetParams(const ZernikeEdgeCPUParams& params)
    {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() called while Execute() is running - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        templates_ = ZernikeTemplates::create(params.templateSize);
        warmed_up_ = false;
    }

    const ZernikeEdgeCPUParams& GetParams() const { return params_; }
};

ZernikeEdgeCPU::ZernikeEdgeCPU(const ZernikeEdgeCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("ZernikeEdgeCPU initialized: templateSize={}, canny=[{:.1f},{:.1f}]",
                   params.templateSize, params.cannyLowThreshold, params.cannyHighThreshold);
}

ZernikeEdgeCPU::~ZernikeEdgeCPU() = default;

void ZernikeEdgeCPU::Destroy() { }

ZernikeEdgeCPUResult ZernikeEdgeCPU::Execute(const cv::Mat& srcImage)
{
    CALIB_LOG_DEBUG("Execute() called: image size={}x{}", srcImage.cols, srcImage.rows);

    if (srcImage.empty()) {
        CALIB_LOG_ERROR("Execute() failed: input image is empty");
        ZernikeEdgeCPUResult result;
        result.success = false;
        result.message = "Input image is empty";
        return result;
    }

    if (srcImage.type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute() failed: input must be CV_8UC1, got type {}", srcImage.type());
        ZernikeEdgeCPUResult result;
        result.success = false;
        result.message = "Input must be CV_8UC1 grayscale image";
        return result;
    }

    return pImpl_->Execute(srcImage);
}

void ZernikeEdgeCPU::Warmup(int rows, int cols)
{
    CALIB_LOG_INFO("warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->Warmup(rows, cols);
}

void ZernikeEdgeCPU::Warmup(const calib::WarmupConfig& config)
{
    CALIB_LOG_INFO("warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

void ZernikeEdgeCPU::SetParams(const ZernikeEdgeCPUParams& params)
{
    CALIB_LOG_INFO("setParams() called: templateSize={}", params.templateSize);
    pImpl_->SetParams(params);
}

const ZernikeEdgeCPUParams& ZernikeEdgeCPU::GetParams() const
{
    return pImpl_->GetParams();
}

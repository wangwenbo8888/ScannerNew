#include "laser_markingpoint_mask_separation_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/cudaimgproc.hpp>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(LaserMarkingSep, LaserMarkingSeparationCUDA);

constexpr int BW = 16;
constexpr int BH = 16;

// ============================================================
// Kernel: Fused Gaussian 5x5 + Threshold (Step 1)
// ============================================================
__global__ void fusedGaussianThresholdKernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int width, int height, int threshold)
{
    __shared__ uint8_t smem[BH + 4][BW + 4];
    int tx = threadIdx.x, ty = threadIdx.y;

    for (int j = ty; j < BH + 4; j += BH) {
        for (int i = tx; i < BW + 4; i += BW) {
            int gx = min(max(blockIdx.x * BW + i - 2, 0), width - 1);
            int gy = min(max(blockIdx.y * BH + j - 2, 0), height - 1);
            smem[j][i] = src[gy * width + gx];
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;

    const int g[5] = {19, 66, 100, 66, 19};
    int sum = 0;
    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy)
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx)
            sum += g[dy+2] * g[dx+2] * smem[ty+2+dy][tx+2+dx];
    dst[y * width + x] = (sum > threshold * 72900) ? 255 : 0;
}

// ============================================================
// Kernel: Binary Erode / Dilate with shared memory
// ============================================================
template<int KHALF, bool IS_ERODE>
__global__ void binaryMorphKernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int width, int height)
{
    constexpr int SW = BW + 2*KHALF, SH = BH + 2*KHALF;
    __shared__ uint8_t smem[SH][SW];
    int tx = threadIdx.x, ty = threadIdx.y;

    for (int j = ty; j < SH; j += BH)
        for (int i = tx; i < SW; i += BW) {
            int gx = blockIdx.x*BW + i - KHALF;
            int gy = blockIdx.y*BH + j - KHALF;
            smem[j][i] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                         ? src[gy*width + gx] : (IS_ERODE ? 255 : 0);
        }
    __syncthreads();

    int x = blockIdx.x*BW + tx, y = blockIdx.y*BH + ty;
    if (x >= width || y >= height) return;

    uint8_t result = IS_ERODE ? 255 : 0;
    for (int dy = -KHALF; dy <= KHALF; ++dy)
        for (int dx = -KHALF; dx <= KHALF; ++dx) {
            uint8_t v = smem[ty+KHALF+dy][tx+KHALF+dx];
            if constexpr (IS_ERODE) { if (v == 0) { dst[y*width+x]=0; return; } }
            else                     { if (v != 0) { dst[y*width+x]=255; return; } }
        }
    dst[y*width+x] = result;
}

// ============================================================
// Kernel: Fused Dilate + Subtract (Step 3)
// ============================================================
template<int KHALF>
__global__ void fusedDilateSubtractKernel(
    const uint8_t* __restrict__ src, const uint8_t* __restrict__ subtrahend,
    uint8_t* __restrict__ dst, int width, int height)
{
    constexpr int SW = BW+2*KHALF, SH = BH+2*KHALF;
    __shared__ uint8_t smem[SH][SW];
    int tx = threadIdx.x, ty = threadIdx.y;

    for (int j = ty; j < SH; j += BH)
        for (int i = tx; i < SW; i += BW) {
            int gx = blockIdx.x*BW+i-KHALF, gy = blockIdx.y*BH+j-KHALF;
            smem[j][i] = (gx>=0 && gx<width && gy>=0 && gy<height)
                         ? src[gy*width+gx] : 0;
        }
    __syncthreads();

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;
    if (x >= width || y >= height) return;

    uint8_t dilated = 0;
    for (int dy = -KHALF; dy <= KHALF; ++dy)
        for (int dx = -KHALF; dx <= KHALF; ++dx)
            if (smem[ty+KHALF+dy][tx+KHALF+dx] != 0) { dilated=255; goto done; }
    done:
    dst[y*width+x] = (subtrahend[y*width+x]==255 && dilated==0) ? 255 : 0;
}

// ============================================================
// Kernel: Fused Dilate + Dual Output (Step 4+5: laser + marking)
// ============================================================
template<int KHALF>
__global__ void fusedDilateDualOutputKernel(
    const uint8_t* __restrict__ src, const uint8_t* __restrict__ combined,
    uint8_t* __restrict__ laser, uint8_t* __restrict__ marking,
    int width, int height)
{
    constexpr int SW = BW+2*KHALF, SH = BH+2*KHALF;
    __shared__ uint8_t smem[SH][SW];
    int tx = threadIdx.x, ty = threadIdx.y;

    for (int j = ty; j < SH; j += BH)
        for (int i = tx; i < SW; i += BW) {
            int gx = blockIdx.x*BW+i-KHALF, gy = blockIdx.y*BH+j-KHALF;
            smem[j][i] = (gx>=0 && gx<width && gy>=0 && gy<height)
                         ? src[gy*width+gx] : 0;
        }
    __syncthreads();

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;
    if (x >= width || y >= height) return;

    uint8_t dilated = 0;
    for (int dy = -KHALF; dy <= KHALF; ++dy)
        for (int dx = -KHALF; dx <= KHALF; ++dx)
            if (smem[ty+KHALF+dy][tx+KHALF+dx] != 0) { dilated=255; goto done2; }
    done2:
    laser[y*width+x] = dilated;
    marking[y*width+x] = (combined[y*width+x]==255 && dilated==0) ? 255 : 0;
}

// ============================================================
// Launch Helpers
// ============================================================
static dim3 gridSz(int w, int h) { return dim3((w+BW-1)/BW, (h+BH-1)/BH); }

static void launchErode(const uint8_t* s, uint8_t* d, int w, int h, int ks, cudaStream_t st) {
    int kh=(ks-1)/2; dim3 b(BW,BH), g=gridSz(w,h);
    switch(kh){
        case 1: binaryMorphKernel<1,true><<<g,b,0,st>>>(s,d,w,h);break;
        case 2: binaryMorphKernel<2,true><<<g,b,0,st>>>(s,d,w,h);break;
        case 3: binaryMorphKernel<3,true><<<g,b,0,st>>>(s,d,w,h);break;
        case 4: binaryMorphKernel<4,true><<<g,b,0,st>>>(s,d,w,h);break;
        case 5: binaryMorphKernel<5,true><<<g,b,0,st>>>(s,d,w,h);break;
        default: throw std::runtime_error("Bad erode khalf");
    }
}
static void launchDilate(const uint8_t* s, uint8_t* d, int w, int h, int ks, cudaStream_t st) {
    int kh=(ks-1)/2; dim3 b(BW,BH), g=gridSz(w,h);
    switch(kh){
        case 1: binaryMorphKernel<1,false><<<g,b,0,st>>>(s,d,w,h);break;
        case 2: binaryMorphKernel<2,false><<<g,b,0,st>>>(s,d,w,h);break;
        case 3: binaryMorphKernel<3,false><<<g,b,0,st>>>(s,d,w,h);break;
        case 4: binaryMorphKernel<4,false><<<g,b,0,st>>>(s,d,w,h);break;
        case 5: binaryMorphKernel<5,false><<<g,b,0,st>>>(s,d,w,h);break;
        default: throw std::runtime_error("Bad dilate khalf");
    }
}
static void launchFusedDilateSub(const uint8_t* s, const uint8_t* sub,
                                 uint8_t* d, int w, int h, int ks, cudaStream_t st) {
    int kh=(ks-1)/2; dim3 b(BW,BH), g=gridSz(w,h);
    switch(kh){
        case 1: fusedDilateSubtractKernel<1><<<g,b,0,st>>>(s,sub,d,w,h);break;
        case 2: fusedDilateSubtractKernel<2><<<g,b,0,st>>>(s,sub,d,w,h);break;
        case 3: fusedDilateSubtractKernel<3><<<g,b,0,st>>>(s,sub,d,w,h);break;
        case 4: fusedDilateSubtractKernel<4><<<g,b,0,st>>>(s,sub,d,w,h);break;
        case 5: fusedDilateSubtractKernel<5><<<g,b,0,st>>>(s,sub,d,w,h);break;
        default: throw std::runtime_error("Bad fused sub khalf");
    }
}
static void launchFusedDual(const uint8_t* s, const uint8_t* c,
                            uint8_t* l, uint8_t* m, int w, int h, int ks, cudaStream_t st) {
    int kh=(ks-1)/2; dim3 b(BW,BH), g=gridSz(w,h);
    switch(kh){
        case 1: fusedDilateDualOutputKernel<1><<<g,b,0,st>>>(s,c,l,m,w,h);break;
        case 2: fusedDilateDualOutputKernel<2><<<g,b,0,st>>>(s,c,l,m,w,h);break;
        case 3: fusedDilateDualOutputKernel<3><<<g,b,0,st>>>(s,c,l,m,w,h);break;
        case 4: fusedDilateDualOutputKernel<4><<<g,b,0,st>>>(s,c,l,m,w,h);break;
        case 5: fusedDilateDualOutputKernel<5><<<g,b,0,st>>>(s,c,l,m,w,h);break;
        default: throw std::runtime_error("Bad fused dual khalf");
    }
}

static const uint8_t* gp(const cv::cuda::GpuMat& m) { return m.ptr<uint8_t>(); }
static uint8_t* gp(cv::cuda::GpuMat& m) { return m.ptr<uint8_t>(); }

// ============================================================
// Impl
// ============================================================
LaserMarkingSeparationCUDA::Impl::Impl(const LaserMarkingSeparationParams& p) : params_(p) {
    params_.validate();
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        throw std::runtime_error("No CUDA GPU");
    createEvents();
}
LaserMarkingSeparationCUDA::Impl::~Impl() { destroyEvents(); }

void LaserMarkingSeparationCUDA::Impl::createEvents() {
    if (events_created_) return;
    cudaEventCreate(&event_start_);
    cudaEventCreate(&event_upload_done_);
    cudaEventCreate(&event_step1_done_);
    cudaEventCreate(&event_step2_done_);
    cudaEventCreate(&event_step3_done_);
    cudaEventCreate(&event_step4_done_);
    cudaEventCreate(&event_step5_done_);
    cudaEventCreate(&event_step6_done_);
    events_created_ = true;
}
void LaserMarkingSeparationCUDA::Impl::destroyEvents() {
    if (!events_created_) return;
    cudaEventDestroy(event_start_);   cudaEventDestroy(event_upload_done_);
    cudaEventDestroy(event_step1_done_); cudaEventDestroy(event_step2_done_);
    cudaEventDestroy(event_step3_done_); cudaEventDestroy(event_step4_done_);
    cudaEventDestroy(event_step5_done_); cudaEventDestroy(event_step6_done_);
    events_created_ = false;
}

// ============================================================
// executePipeline
// ============================================================
void LaserMarkingSeparationCUDA::Impl::executePipeline(
    cv::cuda::Stream& stream, MaskSeparationTimings& timings)
{
    auto cs = cv::cuda::StreamAccessor::getStream(stream);
    int w = d_inputBuffer.cols, h = d_inputBuffer.rows;

    cudaEventRecord(event_start_, cs);

    // Step 1: Fused Gaussian + Threshold
    fusedGaussianThresholdKernel<<<gridSz(w,h),dim3(BW,BH),0,cs>>>(
        gp(d_inputBuffer), gp(d_binary), w, h, params_.threshold);
    cudaEventRecord(event_step1_done_, cs);

    // Step 2: erode(k2) + dilate(k2)
    launchErode(gp(d_binary), gp(d_temp), w, h, params_.step2_erodeSize, cs);
    launchDilate(gp(d_temp), gp(d_step2_mask), w, h, params_.step2_dilateSize, cs);
    cudaEventRecord(event_step2_done_, cs);

    // Step 3: erode(k3) + fused dilate(k3d)+subtract
    launchErode(gp(d_step2_mask), gp(d_temp), w, h, params_.step3_erodeSize, cs);
    launchFusedDilateSub(gp(d_temp), gp(d_step2_mask), gp(d_combined),
                         w, h, params_.step3_dilateSize, cs);
    cudaEventRecord(event_step3_done_, cs);

    // Step 4+5: erode(k4) + fused dilate(k4d)+dual(laser,marking)
    launchErode(gp(d_combined), gp(d_temp), w, h, params_.step4_erodeSize, cs);
    launchFusedDual(gp(d_temp), gp(d_combined), gp(d_laser_mask), gp(d_marking_raw),
                    w, h, params_.step4_dilateSize, cs);
    cudaEventRecord(event_step4_done_, cs);
    cudaEventRecord(event_step5_done_, cs);

    // Step 6: dilate(k6) marking
    launchDilate(gp(d_marking_raw), gp(d_marking_final), w, h, params_.step6_dilateSize, cs);
    cudaEventRecord(event_step6_done_, cs);

    cudaEventSynchronize(event_step6_done_);
    float ms;
    cudaEventElapsedTime(&ms, event_start_, event_upload_done_);          timings.upload_ms = ms;
    cudaEventElapsedTime(&ms, event_upload_done_, event_step1_done_);     timings.step1_gaussian_threshold_ms = ms;
    cudaEventElapsedTime(&ms, event_step1_done_, event_step2_done_);      timings.step2_remove_small_noise_ms = ms;
    cudaEventElapsedTime(&ms, event_step2_done_, event_step3_done_);      timings.step3_remove_large_noise_ms = ms;
    cudaEventElapsedTime(&ms, event_step3_done_, event_step4_done_);      timings.step4_extract_laser_ms = ms;
    cudaEventElapsedTime(&ms, event_step4_done_, event_step5_done_);      timings.step5_extract_marking_ms = ms;
    cudaEventElapsedTime(&ms, event_step5_done_, event_step6_done_);      timings.step6_dilate_marking_ms = ms;
    cudaEventElapsedTime(&ms, event_start_, event_step6_done_);           timings.total_pipeline_ms = ms;
}

void LaserMarkingSeparationCUDA::Impl::warmup(int rows, int cols) {
    auto a = [&](cv::cuda::GpuMat& b){ cv::cuda::createContinuous(rows,cols,CV_8UC1,b); };
    a(d_inputBuffer); a(d_binary); a(d_temp); a(d_step2_mask);
    a(d_combined); a(d_laser_mask); a(d_marking_raw); a(d_marking_final);
    warmup_rows_=rows; warmup_cols_=cols;

    cv::Mat z = cv::Mat::zeros(rows,cols,CV_8UC1);
    d_inputBuffer.upload(z);
    cv::cuda::Stream s;
    MaskSeparationTimings t;
    cudaEventRecord(event_start_, cv::cuda::StreamAccessor::getStream(s));
    d_inputBuffer.upload(z, s);
    executePipeline(s, t);
#ifndef NDEBUG
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("warmup failed: ")+cudaGetErrorString(err));
#endif
    warmed_up_ = true;
}

void LaserMarkingSeparationCUDA::Impl::setParams(const LaserMarkingSeparationParams& p) {
#ifndef NDEBUG
    assert(!inProcess_.load());
#endif
    params_ = p; params_.validate(); warmed_up_ = false;
}

LaserMarkingSeparationResult LaserMarkingSeparationCUDA::Impl::separate(
    const cv::Mat& grayImage, cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    assert(!inProcess_.load());
    struct SF { std::atomic<bool>*f; SF(std::atomic<bool>*f_):f(f_){} ~SF(){f->store(false);} };
    SF guard(&inProcess_); inProcess_.store(true);
#endif

    LaserMarkingSeparationResult result;
    try {
        int r=grayImage.rows, c=grayImage.cols;
        if (!warmed_up_||warmup_rows_!=r||warmup_cols_!=c) {
            auto a = [&](cv::cuda::GpuMat& b){ cv::cuda::createContinuous(r,c,CV_8UC1,b); };
            a(d_inputBuffer); a(d_binary); a(d_temp); a(d_step2_mask);
            a(d_combined); a(d_laser_mask); a(d_marking_raw); a(d_marking_final);
            warmup_rows_=r; warmup_cols_=c; warmed_up_=true;
        }
        auto cs = cv::cuda::StreamAccessor::getStream(stream);
        cudaEventRecord(event_start_, cs);
        d_inputBuffer.upload(grayImage, stream);
        cudaEventRecord(event_upload_done_, cs);
        executePipeline(stream, result.timings);

        result.success = true;
        result.message = "Separation successful";
        result.qualityFlag = calib::QualityFlag::Normal;
        result.d_laserMask = std::make_shared<cv::cuda::GpuMat>(d_laser_mask.clone());
        result.d_markingPointMask = std::make_shared<cv::cuda::GpuMat>(d_marking_final.clone());
        result.d_combinedMask = std::make_shared<cv::cuda::GpuMat>(d_combined.clone());

        CALIB_LOG_INFO("Pipeline timings (ms): upload={:.3f} step1={:.3f} "
            "step2={:.3f} step3={:.3f} step4={:.3f} step5={:.3f} "
            "step6={:.3f} total={:.3f}",
            result.timings.upload_ms, result.timings.step1_gaussian_threshold_ms,
            result.timings.step2_remove_small_noise_ms,
            result.timings.step3_remove_large_noise_ms,
            result.timings.step4_extract_laser_ms,
            result.timings.step5_extract_marking_ms,
            result.timings.step6_dilate_marking_ms,
            result.timings.total_pipeline_ms);
    } catch (const cv::Exception& e) {
        result.success=false; result.message=std::string("OpenCV: ")+e.what();
        result.qualityFlag=calib::QualityFlag::Degraded;
    } catch (const std::exception& e) {
        result.success=false; result.message=std::string("Err: ")+e.what();
        result.qualityFlag=calib::QualityFlag::Degraded;
    }
    return result;
}

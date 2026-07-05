#include "laser_cloud_fuse_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <chrono>
#include <stdexcept>

using namespace calib;

OperatorInfo getLaserCloudFuseCudaInfo() {
    return OperatorInfo{"LaserCloudFuseCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(02C, LaserCloudFuseCuda);

// ============================================================
// Params
// ============================================================

void LaserCloudFuseCUDAParams::validate() const {
    if (voxelSize <= 0.0f)
        throw std::invalid_argument("voxelSize must be > 0");
    if (saturationThreshold < 1)
        throw std::invalid_argument("saturationThreshold must be >= 1");
    if (reserveVoxelCount < 64)
        throw std::invalid_argument("reserveVoxelCount must be >= 64");
}

nlohmann::json LaserCloudFuseCUDAParams::toJson() const {
    nlohmann::json j;
    j["voxelSize"] = voxelSize;
    j["saturationThreshold"] = saturationThreshold;
    j["reserveVoxelCount"] = reserveVoxelCount;
    j["collectStatistics"] = collectStatistics;
    return j;
}

LaserCloudFuseCUDAParams LaserCloudFuseCUDAParams::fromJson(const nlohmann::json& j) {
    LaserCloudFuseCUDAParams p;
    if (j.contains("voxelSize")) p.voxelSize = j.at("voxelSize").get<float>();
    if (j.contains("saturationThreshold")) p.saturationThreshold = j.at("saturationThreshold").get<int>();
    if (j.contains("reserveVoxelCount")) p.reserveVoxelCount = j.at("reserveVoxelCount").get<size_t>();
    if (j.contains("collectStatistics")) p.collectStatistics = j.at("collectStatistics").get<bool>();
    return p;
}

// ============================================================
// BUILD_CUDA 瀹堝崼
// ============================================================

#if BUILD_CUDA

#include "laser_cloud_fuse_cuda_pimpl.h"

// ============================================================
// Constructor / Destructor
// ============================================================

LaserCloudFuseCuda::LaserCloudFuseCuda(const LaserCloudFuseCUDAParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserCloudFuseCuda initialized (voxelSize={}, threshold={}, reserve={})",
                   params.voxelSize, params.saturationThreshold, params.reserveVoxelCount);
}

LaserCloudFuseCuda::~LaserCloudFuseCuda() = default;

// ============================================================
// Execute
// ============================================================

LaserCloudFuseCudaResult LaserCloudFuseCuda::Execute(const cv::cuda::GpuMat& d_points3d,
                                 cv::cuda::Stream& stream) {
    return pImpl_->fuseImpl(d_points3d, cv::Matx33d::eye(), cv::Vec3d(0,0,0), stream);
}

LaserCloudFuseCudaResult LaserCloudFuseCuda::Execute(const cv::cuda::GpuMat& d_points3d) {
    cv::cuda::Stream stream;
    return Execute(d_points3d, stream);
}

LaserCloudFuseCudaResult LaserCloudFuseCuda::Execute(const cv::cuda::GpuMat& d_points3d,
                                 const cv::Matx33d& R, const cv::Vec3d& T,
                                 cv::cuda::Stream& stream) {
    return pImpl_->fuseImpl(d_points3d, R, T, stream);
}

LaserCloudFuseCudaResult LaserCloudFuseCuda::Execute(const cv::cuda::GpuMat& d_points3d,
                                 const cv::Matx33d& R, const cv::Vec3d& T) {
    cv::cuda::Stream stream;
    return Execute(d_points3d, R, T, stream);
}

// ============================================================
// Destroy
// ============================================================

void LaserCloudFuseCuda::Destroy() {
    pImpl_.reset();
}

// ============================================================
// clear / reserve / Warmup
// ============================================================

void LaserCloudFuseCuda::Clear() noexcept {
    pImpl_->clearImpl(0);
}

void LaserCloudFuseCuda::Reserve(size_t voxelCount) {
    pImpl_->allocateHash(voxelCount);
}

void LaserCloudFuseCuda::Warmup(int maxPointCount) {
    pImpl_->warmed_up_ = true;
    pImpl_->ensureTempBuffers(static_cast<size_t>(maxPointCount));
    CALIB_LOG_INFO("Warmup: maxPointCount={}", maxPointCount);
}

void LaserCloudFuseCuda::Warmup(const calib::WarmupConfig& config) {
    Warmup(config.maxPointCount);
}

// ============================================================
// accessors
// ============================================================

size_t LaserCloudFuseCuda::GetVoxelCount() const noexcept {
    return pImpl_->h_voxelCount_;
}

size_t LaserCloudFuseCuda::GetFusedPointCount() const noexcept {
    return pImpl_->h_fusedPointCount_;
}

LaserCloudFuseDeviceContext LaserCloudFuseCuda::GetDeviceContext() const {
    LaserCloudFuseDeviceContext ctx;
    ctx.d_keys        = pImpl_->d_keys_;
    ctx.d_fusedIdx    = pImpl_->d_fusedIdx_;
    ctx.d_fusedXyz    = pImpl_->d_fusedXyz_;
    ctx.d_fusedNormal = pImpl_->d_fusedNormal_;
    ctx.mask          = pImpl_->mask_;
    ctx.voxelSize     = pImpl_->params_.voxelSize;
    ctx.invVoxelSize  = 1.0f / pImpl_->params_.voxelSize;
    ctx.fusedPointCount = pImpl_->h_fusedPointCount_;
    return ctx;
}

void LaserCloudFuseCuda::SetParams(const LaserCloudFuseCUDAParams& params) {
    params.validate();
    pImpl_->params_ = params;
    pImpl_->warmed_up_ = false;
}

const LaserCloudFuseCUDAParams& LaserCloudFuseCuda::GetParams() const {
    return pImpl_->params_;
}

#else // !BUILD_CUDA

// ============================================================
// CUDA 涓嶅彲鐢ㄦ椂鐨勬々
// ============================================================

LaserCloudFuseCuda::LaserCloudFuseCuda(const LaserCloudFuseCUDAParams&) {
    throw std::runtime_error("LaserCloudFuseCuda: BUILD_CUDA is OFF");
}
LaserCloudFuseCuda::~LaserCloudFuseCuda() = default;

LaserCloudFuseCudaResult LaserCloudFuseCuda::Execute(const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    LaserCloudFuseCudaResult r; r.success = false; r.message = "BUILD_CUDA is OFF"; return r;
}
LaserCloudFuseCudaResult LaserCloudFuseCuda::Execute(const cv::cuda::GpuMat&) {
    LaserCloudFuseCudaResult r; r.success = false; r.message = "BUILD_CUDA is OFF"; return r;
}
LaserCloudFuseCudaResult LaserCloudFuseCuda::Execute(const cv::cuda::GpuMat&, const cv::Matx33d&, const cv::Vec3d&, cv::cuda::Stream&) {
    LaserCloudFuseCudaResult r; r.success = false; r.message = "BUILD_CUDA is OFF"; return r;
}
LaserCloudFuseCudaResult LaserCloudFuseCuda::Execute(const cv::cuda::GpuMat&, const cv::Matx33d&, const cv::Vec3d&) {
    LaserCloudFuseCudaResult r; r.success = false; r.message = "BUILD_CUDA is OFF"; return r;
}
void LaserCloudFuseCuda::Destroy() {}
void LaserCloudFuseCuda::Clear() noexcept {}
void LaserCloudFuseCuda::Reserve(size_t) {}
void LaserCloudFuseCuda::Warmup(int) {}
void LaserCloudFuseCuda::Warmup(const calib::WarmupConfig&) {}
size_t LaserCloudFuseCuda::GetVoxelCount() const noexcept { return 0; }
size_t LaserCloudFuseCuda::GetFusedPointCount() const noexcept { return 0; }
LaserCloudFuseDeviceContext LaserCloudFuseCuda::GetDeviceContext() const { return {}; }
void LaserCloudFuseCuda::SetParams(const LaserCloudFuseCUDAParams& p) { p.validate(); }
const LaserCloudFuseCUDAParams& LaserCloudFuseCuda::GetParams() const {
    static LaserCloudFuseCUDAParams p; return p;
}

#endif // BUILD_CUDA
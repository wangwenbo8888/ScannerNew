#include "laser_cloud_normal_cuda.h"
#include "common/calib_logging.h"

#include <chrono>
#include <stdexcept>

using namespace calib;

OperatorInfo getLaserCloudNormalCudaInfo() {
    return OperatorInfo{"LaserCloudNormalCuda", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(03C, LaserCloudNormalCuda);

// ============================================================
// Params
// ============================================================

void LaserCloudNormalCUDAParams::validate() const {
    if (minNeighbors < 3)
        throw std::invalid_argument("minNeighbors must be >= 3");
}

nlohmann::json LaserCloudNormalCUDAParams::toJson() const {
    nlohmann::json j;
    j["minNeighbors"] = minNeighbors;
    j["fallbackNx"] = fallbackNx;
    j["fallbackNy"] = fallbackNy;
    j["fallbackNz"] = fallbackNz;
    return j;
}

LaserCloudNormalCUDAParams LaserCloudNormalCUDAParams::fromJson(const nlohmann::json& j) {
    LaserCloudNormalCUDAParams p;
    if (j.contains("minNeighbors")) p.minNeighbors = j.at("minNeighbors").get<int>();
    if (j.contains("fallbackNx")) p.fallbackNx = j.at("fallbackNx").get<float>();
    if (j.contains("fallbackNy")) p.fallbackNy = j.at("fallbackNy").get<float>();
    if (j.contains("fallbackNz")) p.fallbackNz = j.at("fallbackNz").get<float>();
    return p;
}

// ============================================================
// BUILD_CUDA
// ============================================================

#if BUILD_CUDA

#include "laser_cloud_normal_cuda_pimpl.h"

LaserCloudNormalCuda::LaserCloudNormalCuda(const LaserCloudNormalCUDAParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserCloudNormalCuda initialized (minNeighbors={})",
                   params.minNeighbors);
}

LaserCloudNormalCuda::~LaserCloudNormalCuda() = default;

LaserCloudNormalCudaResult LaserCloudNormalCuda::Execute(const LaserCloudFuseDeviceContext& ctx,
                                    size_t beginIdx, size_t endIdx,
                                    cv::cuda::Stream& stream) {
    return pImpl_->computeImpl(ctx, beginIdx, endIdx, stream);
}

LaserCloudNormalCudaResult LaserCloudNormalCuda::Execute(const LaserCloudFuseDeviceContext& ctx,
                                    size_t beginIdx, size_t endIdx) {
    cv::cuda::Stream stream;
    return Execute(ctx, beginIdx, endIdx, stream);
}

LaserCloudNormalCudaResult LaserCloudNormalCuda::Execute(LaserCloudFuseCuda& fuse,
                                    const LaserCloudFuseCudaResult& fuseResult,
                                    cv::cuda::Stream& stream) {
    auto ctx = fuse.GetDeviceContext();
    size_t total = ctx.fusedPointCount;
    size_t begin = total - fuseResult.newVoxelCount;
    return pImpl_->computeImpl(ctx, begin, total, stream);
}

LaserCloudNormalCudaResult LaserCloudNormalCuda::Execute(LaserCloudFuseCuda& fuse,
                                    const LaserCloudFuseCudaResult& fuseResult) {
    cv::cuda::Stream stream;
    return Execute(fuse, fuseResult, stream);
}

void LaserCloudNormalCuda::Destroy() {
    pImpl_.reset();
}

void LaserCloudNormalCuda::SetParams(const LaserCloudNormalCUDAParams& params) {
    params.validate();
    pImpl_->params_ = params;
}

const LaserCloudNormalCUDAParams& LaserCloudNormalCuda::GetParams() const {
    return pImpl_->params_;
}

#else

LaserCloudNormalCuda::LaserCloudNormalCuda(const LaserCloudNormalCUDAParams&) {
    throw std::runtime_error("LaserCloudNormalCuda: BUILD_CUDA is OFF");
}
LaserCloudNormalCuda::~LaserCloudNormalCuda() = default;
LaserCloudNormalCudaResult LaserCloudNormalCuda::Execute(const LaserCloudFuseDeviceContext&, size_t, size_t, cv::cuda::Stream&) {
    LaserCloudNormalCudaResult r; r.success = false; r.message = "BUILD_CUDA is OFF"; return r;
}
LaserCloudNormalCudaResult LaserCloudNormalCuda::Execute(const LaserCloudFuseDeviceContext&, size_t, size_t) {
    LaserCloudNormalCudaResult r; r.success = false; r.message = "BUILD_CUDA is OFF"; return r;
}
LaserCloudNormalCudaResult LaserCloudNormalCuda::Execute(LaserCloudFuseCuda&, const LaserCloudFuseCudaResult&, cv::cuda::Stream&) {
    LaserCloudNormalCudaResult r; r.success = false; r.message = "BUILD_CUDA is OFF"; return r;
}
LaserCloudNormalCudaResult LaserCloudNormalCuda::Execute(LaserCloudFuseCuda&, const LaserCloudFuseCudaResult&) {
    LaserCloudNormalCudaResult r; r.success = false; r.message = "BUILD_CUDA is OFF"; return r;
}
void LaserCloudNormalCuda::Destroy() {}
void LaserCloudNormalCuda::SetParams(const LaserCloudNormalCUDAParams& p) { p.validate(); }
const LaserCloudNormalCUDAParams& LaserCloudNormalCuda::GetParams() const {
    static LaserCloudNormalCUDAParams p; return p;
}

#endif
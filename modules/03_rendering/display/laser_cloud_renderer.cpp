#include "laser_cloud_renderer.h"
#include "point_expand_kernel.h"
#include "common/calib_logging.h"
#include "laser_cloud_fuse_cuda.h"

#include <osg/RenderInfo>
#include <osg/StateSet>
#include <osg/BoundingBox>
#include <osg/BoundingSphere>
#include <osg/GL>
#include <osg/GLExtensions>

// Windows GL/gl.h 仅 OpenGL 1.1，需补充 GL 1.5+ 常量
#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER         0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_DYNAMIC_DRAW         0x88E8
#define GL_STATIC_DRAW          0x88E4
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cmath>
#include <cstring>
#include <chrono>

using namespace calib;


// ============================================================
// LaserCloudDrawable — 自定义 osg::Drawable
//
// 在 drawImplementation 中：
//   1. 首帧创建 GL VBO + 注册 CUDA interop
//   2. map VBO → kernel 展点 → unmap VBO
//   3. 绑定 VBO + glDrawElements
// 全程 GPU 零拷贝
// ============================================================

class LaserCloudDrawable : public osg::Drawable {
public:
    LaserCloudDrawable(size_t maxPoints)
        : maxPoints_(maxPoints)
    {
        setUseDisplayList(false);
        setUseVertexBufferObjects(false);

        osg::StateSet* ss = getOrCreateStateSet();
        ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    }

    // OSG 克隆接口（最小实现）
    LaserCloudDrawable() = default;
    LaserCloudDrawable(const LaserCloudDrawable&, const osg::CopyOp&) {}
    osg::Object* cloneType() const override { return new LaserCloudDrawable(); }
    osg::Object* clone(const osg::CopyOp& op) const override { return new LaserCloudDrawable(*this, op); }
    bool isSameKindAs(const osg::Object* obj) const override
        { return dynamic_cast<const LaserCloudDrawable*>(obj) != nullptr; }
    const char* className()    const override { return "LaserCloudDrawable"; }
    const char* libraryName()  const override { return "scanner_display"; }

    // 设置本帧数据（仅存指针）
    void setCloudData(const float* d_xyz, const float* d_normal,
                      float voxelSize, size_t count)
    {
        d_xyz_       = d_xyz;
        d_normal_    = d_normal;
        halfSize_    = voxelSize * 0.5f;
        activeCount_ = count;
        dirtyBound();
    }

    size_t activeCount() const { return activeCount_; }
    size_t maxPoints()   const { return maxPoints_; }

protected:
    ~LaserCloudDrawable() override {
        releaseGLObjects(nullptr);
    }

    void releaseGLObjects(osg::State* state) const override {
        if (cudaRes_[0]) { cudaGraphicsUnregisterResource(cudaRes_[0]); cudaRes_[0] = nullptr; }
        if (cudaRes_[1]) { cudaGraphicsUnregisterResource(cudaRes_[1]); cudaRes_[1] = nullptr; }
        if (cudaRes_[2]) { cudaGraphicsUnregisterResource(cudaRes_[2]); cudaRes_[2] = nullptr; }
        if (d_visibleCount_) { cudaFree(d_visibleCount_); d_visibleCount_ = nullptr; }
    }

    osg::BoundingBox computeBoundingBox() const override {
        // 宽泛包围盒（实际点云范围由融合决定）
        float r = 1e4f;
        return osg::BoundingBox(-r, -r, -r, r, r, r);
    }

    void drawImplementation(osg::RenderInfo& renderInfo) const override {
        if (activeCount_ == 0 || !d_xyz_) return;

        osg::State& state = *renderInfo.getState();
        auto* ext = state.get<osg::GLExtensions>();

        // ---- 首帧：创建 GL buffer + CUDA interop ----
        if (!initialized_) {
            initGLObjects(ext);
        }

        // ---- 1. Map VBO → CUDA ----
        cudaError_t err = cudaGraphicsMapResources(3, cudaRes_);
        if (err != cudaSuccess) return;

        float* ptrPos = nullptr;
        float* ptrUv  = nullptr;
        float* ptrNrm = nullptr;
        size_t sz;
        cudaGraphicsResourceGetMappedPointer((void**)&ptrPos, &sz, cudaRes_[0]);
        cudaGraphicsResourceGetMappedPointer((void**)&ptrUv,  &sz, cudaRes_[1]);
        cudaGraphicsResourceGetMappedPointer((void**)&ptrNrm, &sz, cudaRes_[2]);

        // ---- 2. 展点 kernel（GPU→GPU 零拷贝，无剔除）----
        if (ptrPos && ptrUv && ptrNrm) {
            launchExpandLaserPoints(ptrPos, ptrUv, ptrNrm,
                                    d_xyz_, d_normal_,
                                    halfSize_, activeCount_, nullptr);
        }
        cudaStreamSynchronize(0);

        // ---- 3. Unmap（CUDA 写入对 GL 可见）----
        cudaGraphicsUnmapResources(3, cudaRes_);

        // ---- 4. 绑定 VBO + 绘制 ----
        GLsizei idxCount = (GLsizei)(activeCount_ * 6);

        ext->glEnableVertexAttribArray(0);
        ext->glBindBuffer(GL_ARRAY_BUFFER, vboPos_);
        ext->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

        ext->glEnableVertexAttribArray(1);
        ext->glBindBuffer(GL_ARRAY_BUFFER, vboUv_);
        ext->glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

        ext->glEnableVertexAttribArray(2);
        ext->glBindBuffer(GL_ARRAY_BUFFER, vboNormal_);
        ext->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

        ext->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
        glDrawElements(GL_TRIANGLES, idxCount, GL_UNSIGNED_INT, nullptr);

        ext->glDisableVertexAttribArray(0);
        ext->glDisableVertexAttribArray(1);
        ext->glDisableVertexAttribArray(2);
    }

private:
    void initGLObjects(osg::GLExtensions* ext) const {
        size_t vertCount = maxPoints_ * 4;
        size_t posBytes = vertCount * 3 * sizeof(float);
        size_t uvBytes  = vertCount * 2 * sizeof(float);

        // 创建 3 个 VBO
        ext->glGenBuffers(1, &vboPos_);
        ext->glBindBuffer(GL_ARRAY_BUFFER, vboPos_);
        ext->glBufferData(GL_ARRAY_BUFFER, posBytes, nullptr, GL_DYNAMIC_DRAW);

        ext->glGenBuffers(1, &vboUv_);
        ext->glBindBuffer(GL_ARRAY_BUFFER, vboUv_);
        ext->glBufferData(GL_ARRAY_BUFFER, uvBytes, nullptr, GL_DYNAMIC_DRAW);

        ext->glGenBuffers(1, &vboNormal_);
        ext->glBindBuffer(GL_ARRAY_BUFFER, vboNormal_);
        ext->glBufferData(GL_ARRAY_BUFFER, posBytes, nullptr, GL_DYNAMIC_DRAW);

        ext->glBindBuffer(GL_ARRAY_BUFFER, 0);

        // 静态索引缓冲
        ext->glGenBuffers(1, &ibo_);
        ext->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
        std::vector<unsigned> indices;
        indices.reserve(maxPoints_ * 6);
        for (size_t i = 0; i < maxPoints_; ++i) {
            unsigned b = (unsigned)(i * 4);
            indices.push_back(b+0); indices.push_back(b+1); indices.push_back(b+2);
            indices.push_back(b+0); indices.push_back(b+2); indices.push_back(b+3);
        }
        ext->glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned),
                          indices.data(), GL_STATIC_DRAW);
        ext->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // 注册 CUDA interop
        cudaGraphicsGLRegisterBuffer(&cudaRes_[0], vboPos_,    cudaGraphicsMapFlagsWriteDiscard);
        cudaGraphicsGLRegisterBuffer(&cudaRes_[1], vboUv_,     cudaGraphicsMapFlagsWriteDiscard);
        cudaGraphicsGLRegisterBuffer(&cudaRes_[2], vboNormal_, cudaGraphicsMapFlagsWriteDiscard);

        initialized_ = true;
        cudaMalloc(&d_visibleCount_, sizeof(unsigned int));
        printf("[LaserCloudDrawable] GL VBO + CUDA interop 初始化完成 (maxPoints=%zu)\n",
               maxPoints_);
    }

    // 配置
    size_t maxPoints_ = 0;

    // 本帧数据指针（GPU 显存，零拷贝直读）
    const float* d_xyz_       = nullptr;
    const float* d_normal_    = nullptr;
    float        halfSize_    = 0.25f;
    size_t       activeCount_ = 0;
    mutable unsigned int visibleCount_ = 0;

    // GPU 可见计数器（cull kernel 输出）
    mutable unsigned int* d_visibleCount_ = nullptr;

    // GL + CUDA 资源（mutable：drawImplementation 是 const）
    mutable bool                   initialized_ = false;
    mutable GLuint                 vboPos_ = 0, vboUv_ = 0, vboNormal_ = 0, ibo_ = 0;
    mutable cudaGraphicsResource*  cudaRes_[3] = {nullptr, nullptr, nullptr};
};

// ============================================================
// LaserCloudRenderer::Impl
// ============================================================

struct LaserCloudRenderer::Impl {
    osg::ref_ptr<LaserCloudDrawable> drawable_;
    size_t activePoints_ = 0;

    explicit Impl(size_t maxPoints) {
        drawable_ = new LaserCloudDrawable(maxPoints);
    }
};

// ============================================================
// 公开接口
// ============================================================

LaserCloudRenderer::LaserCloudRenderer(size_t maxPoints)
    : pImpl_(std::make_unique<Impl>(maxPoints)) {}

LaserCloudRenderer::~LaserCloudRenderer() = default;

void LaserCloudRenderer::update(const LaserCloudFuseDeviceContext& ctx)
{
    if (!ctx.d_fusedXyz || !ctx.d_fusedNormal || ctx.fusedPointCount == 0) {
        pImpl_->drawable_->setCloudData(nullptr, nullptr, 0, 0);
        pImpl_->activePoints_ = 0;
        return;
    }

    size_t count = ctx.fusedPointCount;
    if (count > pImpl_->drawable_->maxPoints()) {
        CALIB_LOG_WARN("点数 {} 超过 maxPoints {}，截断", count, pImpl_->drawable_->maxPoints());
        count = pImpl_->drawable_->maxPoints();
    }

    pImpl_->drawable_->setCloudData(ctx.d_fusedXyz, ctx.d_fusedNormal,
                                    ctx.voxelSize, count);
    pImpl_->activePoints_ = count;
}

osg::Drawable* LaserCloudRenderer::getDrawable() const { return pImpl_->drawable_; }

void LaserCloudRenderer::clear() {
    pImpl_->drawable_->setCloudData(nullptr, nullptr, 0, 0);
    pImpl_->activePoints_ = 0;
}

size_t LaserCloudRenderer::maxPoints() const    { return pImpl_->drawable_->maxPoints(); }
size_t LaserCloudRenderer::activePoints() const { return pImpl_->activePoints_; }

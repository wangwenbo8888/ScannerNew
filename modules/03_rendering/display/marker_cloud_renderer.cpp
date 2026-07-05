#include "marker_cloud_renderer.h"
#include "point_expand_math.h"
#include "common/calib_logging.h"

#include <osg/Array>
#include <osg/PrimitiveSet>
#include <osg/StateSet>

#include <cmath>
#include <algorithm>

using namespace calib;


// ============================================================
// Impl
// ============================================================

struct MarkerCloudRenderer::Impl {
    size_t maxMarkers_ = 0;

    // 外部数据源引用（由 update() 设置，flush() 消费）
    const std::vector<MarkerCloudPoint>* srcPoints_ = nullptr;
    size_t srcCount_ = 0;

    // OSG geometry
    osg::ref_ptr<osg::Geometry>         geometry_;
    osg::ref_ptr<osg::Vec3Array>        positions_;
    osg::ref_ptr<osg::Vec2Array>        uvs_;
    osg::ref_ptr<osg::Vec3Array>        normals_;
    osg::ref_ptr<osg::DrawElementsUInt> indices_;

    explicit Impl(size_t maxMarkers) : maxMarkers_(maxMarkers) {
        size_t vertCount = maxMarkers * 4;

        positions_ = new osg::Vec3Array;
        positions_->resize(vertCount);

        uvs_ = new osg::Vec2Array;
        uvs_->resize(vertCount);

        normals_ = new osg::Vec3Array;
        normals_->resize(vertCount);

        indices_ = new osg::DrawElementsUInt(GL_TRIANGLES);
        indices_->reserve(maxMarkers * 6);
        for (size_t i = 0; i < maxMarkers; ++i) {
            unsigned b = (unsigned)(i * 4);
            indices_->push_back(b + 0);
            indices_->push_back(b + 1);
            indices_->push_back(b + 2);
            indices_->push_back(b + 0);
            indices_->push_back(b + 2);
            indices_->push_back(b + 3);
        }

        geometry_ = new osg::Geometry;
        geometry_->setUseDisplayList(false);
        geometry_->setVertexArray(positions_);
        geometry_->setNormalArray(normals_);
        geometry_->setTexCoordArray(0, uvs_);
        geometry_->addPrimitiveSet(indices_);

        osg::StateSet* ss = geometry_->getOrCreateStateSet();
        ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    }
};

// ============================================================
// 公开接口
// ============================================================

MarkerCloudRenderer::MarkerCloudRenderer(size_t maxMarkers)
    : pImpl_(std::make_unique<Impl>(maxMarkers)) {}

MarkerCloudRenderer::~MarkerCloudRenderer() = default;

void MarkerCloudRenderer::update(const std::vector<MarkerCloudPoint>& fusedPoints)
{
    pImpl_->srcPoints_ = &fusedPoints;
    pImpl_->srcCount_  = fusedPoints.size();
}

void MarkerCloudRenderer::flush()
{
    const auto* src = pImpl_->srcPoints_;
    size_t count = (src && src->size() > 0) ? std::min(src->size(), pImpl_->maxMarkers_) : 0;

    if (count == 0) {
        pImpl_->indices_->resize(0);
        pImpl_->indices_->dirty();
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        const auto& e = (*src)[i];
        float outerRadius = e.whiteRadius * RING_RATIO;

        ExpandVertex v[4];
        expandMarkerPointCPU(v, e.x, e.y, e.z, e.nx, e.ny, e.nz, outerRadius);

        size_t base = i * 4;
        for (int j = 0; j < 4; ++j) {
            (*pImpl_->positions_)[base + j].set(v[j].x, v[j].y, v[j].z);
            (*pImpl_->uvs_)[base + j].set(v[j].u, v[j].v);
            (*pImpl_->normals_)[base + j].set(v[j].nx, v[j].ny, v[j].nz);
        }
    }

    pImpl_->indices_->resize((unsigned)(count * 6));
    pImpl_->positions_->dirty();
    pImpl_->uvs_->dirty();
    pImpl_->normals_->dirty();
    pImpl_->indices_->dirty();
}

osg::Geometry* MarkerCloudRenderer::getGeometry() const { return pImpl_->geometry_; }

void MarkerCloudRenderer::clear()
{
    pImpl_->srcPoints_ = nullptr;
    pImpl_->srcCount_  = 0;
    pImpl_->indices_->resize(0);
    pImpl_->indices_->dirty();
}

size_t MarkerCloudRenderer::pointCount() const { return pImpl_->srcCount_; }

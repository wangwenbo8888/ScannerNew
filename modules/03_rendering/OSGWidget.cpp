#include "OSGWidget.h"
#include "file_io.h"
#include "RenderSanity.h"

#include <chrono>
#include <cstring>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Point>
#include <osg/PointSprite>
#include <osg/Material>
#include <osg/LightModel>
#include <osgDB/ReadFile>
#include <osg/LineWidth>
#include <osg/BlendFunc>
#include <osgText/Text>
#include <osg/ComputeBoundsVisitor>

#include <QMouseEvent>
#include <QWheelEvent>
#include <QApplication>
#include <QMessageBox>
#include <QImage>
#include <QPainter>
#include <QFont>

#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Eigen/Dense>

#include <osgUtil/CullVisitor>
#include <osgUtil/Optimizer>



OSGWidget::OSGWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_timer(new QTimer(this))
    , m_streamTimer(new QTimer(this))
    , m_lastMouseX(0)
    , m_lastMouseY(0)
    , m_firstMouse(true)
    , m_pcaSum(0, 0, 0)
    , m_pcaSumXX(0), m_pcaSumXY(0), m_pcaSumXZ(0)
    , m_pcaSumYY(0), m_pcaSumYZ(0), m_pcaSumZZ(0)
    , m_pcaCount(0), m_pcaTicks(0)
{
    m_root = new osg::Group();
    m_root->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    connect(m_timer, &QTimer::timeout, this, [this]() { update(); });
    m_timer->start(33);                 // 渲染 30fps（原 16ms≈60，降半省资源）

    connect(m_streamTimer, &QTimer::timeout, this, &OSGWidget::streamNextBatch);
}

OSGWidget::~OSGWidget()
{
    m_timer->stop();
    m_streamTimer->stop();
    if (m_streamFile.is_open())
        m_streamFile.close();
}

void OSGWidget::setSceneData(osg::Node *node)
{
    m_root->removeChildren(0, m_root->getNumChildren());
    if (node)
        m_root->addChild(node);
}

void OSGWidget::setCenterOverlayVisible(bool visible)
{
    m_centerOverlayVisible = visible;
    if (!m_centerOverlayCamera.valid()) return;

    if (visible) {
        m_centerOverlayCamera->setNodeMask(0xffffffff);
        // 确保 camera 在 sceneRoot 中
        osg::Group* sceneRoot = m_viewer.valid() ? m_viewer->getSceneData()->asGroup() : nullptr;
        if (sceneRoot && m_centerOverlayCamera->getNumParents() == 0)
            sceneRoot->addChild(m_centerOverlayCamera.get());
    } else {
        m_centerOverlayCamera->setNodeMask(0);
        // 从场景图中彻底移除
        while (m_centerOverlayCamera->getNumParents() > 0)
            m_centerOverlayCamera->getParent(0)->removeChild(m_centerOverlayCamera.get());
    }
    printf("  setCenterOverlayVisible(%d): parents=%d\n",
           visible ? 1 : 0, m_centerOverlayCamera->getNumParents());
}

void OSGWidget::clearScene()
{
    // 诊断（2026-08-31"停止后点消失"）：清场动作留痕——谁在扫描停止链上清场景
    JMW_LOG_INFO("03-OSGWidget", "[OSGWidget] clearScene 执行（场景子节点={} 被清）",
                 m_root->getNumChildren());
    m_root->removeChildren(0, m_root->getNumChildren());
    m_viewLocked = false;  // 解除视图锁定
    m_lastFitRadius = -1.0;   // 视角跟随基准复位（新会话重新首取景）
    m_lastFitTime = {};
    m_streamTimer->stop();
    if (m_streamFile.is_open())
        m_streamFile.close();
    m_streamLoaded = 0;
    m_streamDone = true;
    m_streamBBoxValid = false;
    m_highlights.clear();
    m_deleteHistory.clear();
    m_selectedPolylines.clear();
}

void OSGWidget::setCameraManipulator(osgGA::CameraManipulator *manipulator)
{
    if (m_viewer.valid())
        m_viewer->setCameraManipulator(manipulator);
}

void OSGWidget::home()
{
    if (m_viewer.valid() && m_viewer->getCameraManipulator())
        m_viewer->getCameraManipulator()->home(0);
}

void OSGWidget::createAxesIndicator()
{
    m_axesCamera = new osg::Camera();
    m_axesCamera->setRenderOrder(osg::Camera::POST_RENDER);
    m_axesCamera->setClearMask(GL_DEPTH_BUFFER_BIT);
    m_axesCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    m_axesCamera->setViewMatrix(osg::Matrix::identity());
    m_axesCamera->setAllowEventFocus(false);
    m_axesCamera->setGraphicsContext(m_gw.get());

    int s = devicePixelRatio();
    int axesSize = 200;
    int margin = 10;
    m_axesCamera->setViewport(margin, margin, axesSize * s, axesSize * s);
    m_axesCamera->setProjectionMatrixAsOrtho(-2.5f, 2.5f, -2.5f, 2.5f, -2.5f, 2.5f);

    m_axesGroup = new osg::Group();
    m_axesGroup->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    m_axesCamera->addChild(m_axesGroup.get());

    float shaftLen = 1.2f;
    float coneR = 0.1f;
    float coneH = 0.3f;

    auto makeAxis = [&](const osg::Vec3& dir, const osg::Vec4& color, const osg::Vec3& conePos)
    {
        osg::ref_ptr<osg::Geode> geode = new osg::Geode();

        osg::ref_ptr<osg::Geometry> line = new osg::Geometry();
        osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array();
        verts->push_back(osg::Vec3(0, 0, 0));
        verts->push_back(dir * shaftLen);
        line->setVertexArray(verts);
        line->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 2));

        osg::ref_ptr<osg::Vec4Array> lc = new osg::Vec4Array();
        lc->push_back(color);
        line->setColorArray(lc, osg::Array::BIND_OVERALL);
        line->getOrCreateStateSet()->setAttribute(new osg::LineWidth(3.0f));
        geode->addDrawable(line);

        m_axesGroup->addChild(geode);

        // Build cone tip as triangle fan: tip at conePos + dir*coneH, base at conePos
        osg::ref_ptr<osg::Geometry> coneGeom = new osg::Geometry();
        osg::ref_ptr<osg::Vec3Array> cv = new osg::Vec3Array();
        osg::Vec3 tip = conePos + dir * coneH;
        cv->push_back(tip);
        int segs = 16;
        for (int i = 0; i <= segs; ++i) {
            float a = (float)i / segs * 2.0f * osg::PI;
            osg::Vec3 sideDir(dir.y() * cos(a) + dir.z() * sin(a),
                              dir.z() * cos(a) + dir.x() * sin(a),
                              dir.x() * cos(a) + dir.y() * sin(a));
            // Project sideDir onto the plane perpendicular to dir
            sideDir = sideDir - dir * (sideDir * dir);
            float len = sideDir.length();
            if (len > 1e-6f) sideDir *= coneR / len; else sideDir = osg::Vec3(0, 0, 0);
            cv->push_back(conePos + sideDir);
        }
        coneGeom->setVertexArray(cv);
        coneGeom->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLE_FAN, 0, segs + 2));
        osg::ref_ptr<osg::Vec4Array> cc = new osg::Vec4Array();
        cc->push_back(color);
        coneGeom->setColorArray(cc, osg::Array::BIND_OVERALL);
        osg::ref_ptr<osg::Geode> coneGeode = new osg::Geode();
        coneGeode->addDrawable(coneGeom);
        m_axesGroup->addChild(coneGeode);
    };

    // X: red cone at (shaftLen, 0, 0), pointing +X
    makeAxis(osg::Vec3(1, 0, 0), osg::Vec4(1, 0, 0, 1), osg::Vec3(shaftLen, 0, 0));
    // Y: green cone at (0, shaftLen, 0), pointing +Y
    makeAxis(osg::Vec3(0, 1, 0), osg::Vec4(0, 1, 0, 1), osg::Vec3(0, shaftLen, 0));
    // Z: blue cone at (0, 0, shaftLen), pointing +Z
    makeAxis(osg::Vec3(0, 0, 1), osg::Vec4(0, 0, 1, 1), osg::Vec3(0, 0, shaftLen));

    // Axis labels
    auto makeLabel = [&](const osg::Vec3& pos, const std::string& text, const osg::Vec4& color)
    {
        osg::ref_ptr<osgText::Text> label = new osgText::Text();
        label->setText(text);
        label->setCharacterSize(0.25f);
        label->setColor(color);
        label->setPosition(pos);
        label->setAlignment(osgText::Text::CENTER_CENTER);
        label->setAxisAlignment(osgText::Text::SCREEN);
        osg::ref_ptr<osg::Geode> labelGeode = new osg::Geode();
        labelGeode->addDrawable(label);
        m_axesGroup->addChild(labelGeode);
    };
    makeLabel(osg::Vec3(shaftLen + coneH + 0.2f, -0.2f, 0), "X", osg::Vec4(1, 1, 1, 1));
    makeLabel(osg::Vec3(-0.2f, shaftLen + coneH + 0.2f, 0), "Y", osg::Vec4(1, 1, 1, 1));
    makeLabel(osg::Vec3(-0.2f, 0, shaftLen + coneH + 0.2f), "Z", osg::Vec4(1, 1, 1, 1));

    osg::ref_ptr<osg::Geode> originGeode = new osg::Geode();
    osg::ref_ptr<osg::Sphere> sphere = new osg::Sphere(osg::Vec3(0, 0, 0), 0.03f);
    osg::ref_ptr<osg::ShapeDrawable> sphereDraw = new osg::ShapeDrawable(sphere);
    sphereDraw->setColor(osg::Vec4(0.2f, 0.2f, 0.2f, 1));
    originGeode->addDrawable(sphereDraw);
    m_axesGroup->addChild(originGeode);

    // Add axes camera to the scene root (which already contains m_root)
    osg::Group* sceneRoot = m_viewer->getSceneData()->asGroup();
    if (sceneRoot)
        sceneRoot->addChild(m_axesCamera.get());
}

void OSGWidget::updateAxesView()
{
    if (!m_axesCamera.valid() || !m_viewer.valid())
        return;

    osg::Matrix vm = m_viewer->getCamera()->getViewMatrix();
    vm.setTrans(0, 0, 0);
    // 诊断（2026-08-31 HUD Y 轴上下跳）：view 的 Z 轴行（画面 up 方向）跳变
    // 时留痕——两态数值直接暴露视角源打架的真实来源
    static osg::Matrix lastVm;
    static bool hasLast = false;
    static auto lastLog = std::chrono::steady_clock::now() - std::chrono::seconds(2);
    if (hasLast) {
        const bool flipped = ((vm(2, 0) > 0) != (lastVm(2, 0) > 0)) ||
                             ((vm(2, 1) > 0) != (lastVm(2, 1) > 0));
        if (flipped) {
            const auto now = std::chrono::steady_clock::now();
            if (now - lastLog > std::chrono::seconds(1)) {
                lastLog = now;
                JMW_LOG_INFO("03-OSGWidget",
                             "[OSGWidget] view Z行跳变: 前=({:.2f},{:.2f},{:.2f}) 后=({:.2f},{:.2f},{:.2f})",
                             lastVm(2, 0), lastVm(2, 1), lastVm(2, 2),
                             vm(2, 0), vm(2, 1), vm(2, 2));
            }
        }
    }
    lastVm = vm;
    hasLast = true;
    m_axesCamera->setViewMatrix(vm);
}

osg::ref_ptr<osg::Geode> OSGWidget::buildCloudGeode(const std::vector<osg::Vec3>& points,
                                                    const std::vector<osg::Vec4ub>* colors)
{
    // LeadScan 风格：Vec4Array(float 颜色) + VBO + DYNAMIC
    // 分配集中于此——调用方在挂场景前构建（异常可弃，旧画面不动）
    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array();
    v->reserve(points.size());
    c->reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        v->push_back(points[i]);
        if (colors && i < colors->size())
            c->push_back(osg::Vec4((*colors)[i].r() / 255.0f, (*colors)[i].g() / 255.0f,
                                   (*colors)[i].b() / 255.0f, (*colors)[i].a() / 255.0f));
        else
            c->push_back(osg::Vec4(0.529f, 0.808f, 0.980f, 1.0f));  // LeadScan 蓝
    }

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setUseVertexBufferObjects(true);
    geom->setUseDisplayList(false);
    geom->setDataVariance(osg::Object::DYNAMIC);

    geom->setVertexArray(v.get());
    geom->setColorArray(c.get());
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, v->size()));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom.get());

    // Point size 3.0 (LeadScan 风格)
    osg::ref_ptr<osg::StateSet> ss = geode->getOrCreateStateSet();
    osg::ref_ptr<osg::Point> pointSize = new osg::Point;
    pointSize->setSize(3.0f);
    ss->setAttribute(pointSize);
    return geode;
}

void OSGWidget::fitCameraToRoot()
{
    // 相机定位到数据，并保留 manipulator 供旋转/缩放操作（原 loadPointCloud 尾块）
    osg::BoundingSphere bs = m_root->getBound();
    if (bs.valid() && bs.radius() > 0 && m_viewer.valid()) {
        double r = bs.radius();
        osg::Vec3d ctr(bs.center());
        osg::Vec3d eye(ctr.x(), ctr.y() - r * 3.0, ctr.z() + r * 0.5);
        osg::Vec3d up(0, 0, 1);

        double zNear = r * 0.1;
        double zFar = r * 100.0;
        const osg::GraphicsContext::Traits* traits = m_gw->getTraits();
        double aspect = static_cast<double>(traits->width) / static_cast<double>(traits->height);
        m_viewer->getCamera()->setProjectionMatrixAsPerspective(30.0, aspect, zNear, zFar);
        m_userProjection = m_viewer->getCamera()->getProjectionMatrix();

        m_viewLocked = false;
        osg::ref_ptr<osgGA::TrackballManipulator> manip = new osgGA::TrackballManipulator;
        manip->setAllowThrow(false);
        manip->setHomePosition(eye, ctr, up, false);
        m_viewer->setCameraManipulator(manip);
        manip->home(0);
    }
}

void OSGWidget::loadPointCloud(const std::vector<osg::Vec3>& points)
{
    if (points.empty()) return;

    osg::ref_ptr<osg::Geode> geode = buildCloudGeode(points, nullptr);
    m_root->addChild(geode.get());
    fitCameraToRoot();
}

void OSGWidget::loadPointCloud(const std::vector<osg::Vec3>& points,
                                const std::vector<osg::Vec4ub>& colors)
{
    if (points.empty()) return;

    osg::ref_ptr<osg::Geode> geode = buildCloudGeode(points, &colors);
    m_root->addChild(geode.get());
}

// ============================================================================
// P1 摄入加固：四道闸（RenderSanity）→ 先建后换（原子替换）→ 版本记账
// （解 PointCloudBuffer 耦合 2026-09-01：快照拉取归调用方——渲染只吃
// 值类型，不再 include 06 实现类；跨层 include 挂账清账）
// ============================================================================
void OSGWidget::loadCloudSnapshot(uint64_t version,
                                  const std::vector<cv::Point3f>& points,
                                  const std::vector<cv::Vec3b>& colors) {
    // 原实现 pcb->getSnapshot 由调用方完成；sanitize 就地压缩（NaN 滤除/
    // 抽稀）——值语义下拷局部（渲染摄入 500ms 一次，拷贝可接受）
    if (points.empty()) return;
    std::vector<cv::Point3f> pts = points;
    std::vector<cv::Vec3b> cols = colors;

    // ① 净化与判定（版本短路/空/NaN/尺寸不配/包围盒/超预算抽稀）
    const auto d = Scanner::render::sanitizeSnapshot(
        pts, cols, version, m_lastIngestVersion,
        m_maxIngestPoints, m_maxIngestExtentMm);
    if (!d.accept) {
        if (std::strcmp(d.reason, "unchanged") != 0)   // 版本未变=正常静默路径
            reportFault(static_cast<int>(Scanner::render::RenderEvent::SnapshotRejected),
                        std::string("快照拒绝: ") + d.reason);
        return;                                        // 拒绝→保留现画面
    }

    // ② 先建后换：构建完整才触碰场景（构建期异常→旧画面保留，0x0302）
    osg::ref_ptr<osg::Geode> geode;
    try {
        std::vector<osg::Vec3> osgPoints;
        osgPoints.reserve(pts.size());
        for (const auto& p : pts)
            osgPoints.emplace_back(p.x, p.y, p.z);

        if (!cols.empty()) {
            std::vector<osg::Vec4ub> osgColors;
            osgColors.reserve(cols.size());
            for (const auto& c : cols)
                osgColors.emplace_back(c[0], c[1], c[2], 255);
            geode = buildCloudGeode(osgPoints, &osgColors);
        } else {
            geode = buildCloudGeode(osgPoints, nullptr);
        }
    } catch (const std::bad_alloc&) {
        reportFault(static_cast<int>(Scanner::render::RenderEvent::BuildDegraded),
                    "场景构建内存不足，保留旧画面");
        return;
    } catch (const std::exception& e) {
        reportFault(static_cast<int>(Scanner::render::RenderEvent::BuildDegraded),
                    std::string("场景构建异常，保留旧画面: ") + e.what());
        return;
    } catch (...) {
        reportFault(static_cast<int>(Scanner::render::RenderEvent::BuildDegraded),
                    "场景构建未知异常，保留旧画面");
        return;
    }

    // ③ 原子替换：仅替换点云 geode——不再 clearScene（曾把标志点几何一并
    // 清掉，与标志点信号形成"清-建-清-建"循环=标志点闪烁+双视角源（此处
    // fitCameraToRoot z-up vs setLeftCameraView -y-up）交替=HUD Y 轴上下跳
    // 的共同根因 2026-08-31）；定时器直读路径不动视角
    if (m_pcbGeode && m_pcbGeode->getNumParents() > 0)
        m_root->removeChild(m_pcbGeode.get());
    m_pcbGeode = geode;
    m_root->addChild(geode.get());

    m_lastIngestVersion = version;
    ++m_ingestCount;
    if (d.truncated) {
        reportFault(static_cast<int>(Scanner::render::RenderEvent::DegradeChanged),
                    "摄入点数超预算，已均匀抽稀至 " + std::to_string(d.keptCount) + " 点");
    }
}

void OSGWidget::setIngestBudget(size_t maxPoints, double maxExtentMm) {
    m_maxIngestPoints = maxPoints > 0 ? maxPoints : 1;
    m_maxIngestExtentMm = maxExtentMm > 0.0 ? maxExtentMm : 1.0e6;
    m_degradeLevel = 0;                    // 外部显式设预算＝退出自动降级阶梯
    m_overBudgetStreak = 0;
}

bool OSGWidget::loadTestData(int numPoints)
{
    clearScene();

    double vol = 200.0 * 100.0 * 50.0;
    double step = std::cbrt(vol / numPoints * 1.2);
    int nx = int(200.0 / step);
    int ny = int(100.0 / step);
    int nz = int(50.0 / step);
    int estimate = int(nx * ny * nz * 0.85);

    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4ubArray> c = new osg::Vec4ubArray();
    v->reserve(estimate);
    c->reserve(estimate);

    int written = 0;
    for (int iz = 0; iz < nz; ++iz)
    {
        float z = (iz + 0.5f) * step;
        if (z < 0.0f || z > 50.0f) continue;

        for (int iy = 0; iy < ny; ++iy)
        {
            float y = -50.0f + (iy + 0.5f) * step;
            if (y < -50.0f || y > 50.0f) continue;

            for (int ix = 0; ix < nx; ++ix)
            {
                float x = -100.0f + (ix + 0.5f) * step;
                if (x < -100.0f || x > 100.0f) continue;

                if (y > 25.0f && z > 35.0f) continue;

                float hx1 = x - 50.0f, hy1 = y - 25.0f;
                float hx2 = x + 50.0f, hy2 = y - 25.0f;
                float hx3 = x - 50.0f, hy3 = y + 25.0f;
                float hx4 = x + 50.0f, hy4 = y + 25.0f;
                float d1 = hx1 * hx1 + hy1 * hy1;
                float d2 = hx2 * hx2 + hy2 * hy2;
                float d3 = hx3 * hx3 + hy3 * hy3;
                float d4 = hx4 * hx4 + hy4 * hy4;
                if (d1 < 49.0f || d2 < 49.0f || d3 < 49.0f || d4 < 49.0f) continue;

                v->push_back(osg::Vec3(x, y, z));

                float minD = std::min({ d1, d2, d3, d4 });
                osg::Vec4ub color;
                if (minD < 81.0f)
                    color.set(220, 60, 60, 255);
                else if (y > 20.0f && z > 30.0f)
                    color.set(60, 120, 220, 255);
                else if (z < 5.0f)
                    color.set(140, 180, 140, 255);
                else
                    color.set(180, 180, 190, 255);
                c->push_back(color);

                ++written;
            }
        }
    }

    if (v->empty()) return false;

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setVertexArray(v.get());
    geom->setColorArray(c.get());
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, v->size()));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom.get());
    geode->getOrCreateStateSet()->setAttribute(new osg::Point(1.0f));

    m_root->addChild(geode.get());

    osg::ComputeBoundsVisitor cbv;
    geode->accept(cbv);
    osg::BoundingBox bb = cbv.getBoundingBox();
    double diag = (bb._max - bb._min).length();
    double znear = diag * 0.001;
    double zfar = diag * 10.0;

    if (m_viewer.valid())
    {
        osg::Camera* cam = m_viewer->getCamera();
        cam->setProjectionMatrixAsPerspective(45.0,
            static_cast<double>(width() * devicePixelRatio()) /
            static_cast<double>(height() * devicePixelRatio()), znear, zfar);
        m_userProjection = cam->getProjectionMatrix();
        if (m_viewer->getCameraManipulator())
            m_viewer->getCameraManipulator()->home(0);
    }

    return true;
}

bool OSGWidget::loadTestDataFromPLY(const std::string& filepath, int numPoints)
{
    clearScene();
    m_streamTimer->stop();
    if (m_streamFile.is_open())
        m_streamFile.close();
    m_streamLoaded = 0;
    m_streamDone = false;

    m_streamFile.open(filepath, std::ios::binary);
    if (!m_streamFile.is_open())
        return false;
    m_streamFileBuf = std::vector<char>(1 << 20, '\0');          // 1MB 大缓冲：
    m_streamFile.rdbuf()->pubsetbuf(m_streamFileBuf.data(),      // 整批 read 少走 OS 页
        static_cast<std::streamsize>(m_streamFileBuf.size()));   // 调用（方案 B）

    std::string line;
    int vertexCount = 0;
    bool binaryFormat = false;
    bool headerEnd = false;

    while (std::getline(m_streamFile, line))
    {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line == "end_header") { headerEnd = true; break; }
        if (line.compare(0, 14, "element vertex") == 0)
            vertexCount = std::stoi(line.substr(15));
        if (line.find("binary_little_endian") != std::string::npos)
            binaryFormat = true;
    }
    if (!headerEnd || vertexCount <= 0 || !binaryFormat)
    {
        m_streamFile.close();
        return false;
    }

    m_streamTotal = (numPoints > 0 && numPoints < vertexCount) ? numPoints : vertexCount;
    m_streamPath = filepath;
    m_streamBatch = 50000;
    m_streamStart = std::chrono::steady_clock::now();   // 耗时打点起

    m_streamTimer->start(8);

    return true;
}

void OSGWidget::updateStreamCameraView()
{
    if (!m_viewer.valid() || !m_viewer->getCameraManipulator())
        return;

    float diag = 1.0f;
    if (m_streamBBoxValid)
        diag = (m_streamBBoxMax - m_streamBBoxMin).length();
    if (diag < 1.0f) diag = 1.0f;

    double aspect = static_cast<double>(width() * devicePixelRatio()) /
                    static_cast<double>(height() * devicePixelRatio());
    double znear = diag * 0.001;
    double zfar = diag * 10.0;
    m_viewer->getCamera()->setProjectionMatrixAsPerspective(45.0, aspect, znear, zfar);
    m_userProjection = m_viewer->getCamera()->getProjectionMatrix();

    // Let OSG compute the ideal view (rotation & center) from the scene's bounding sphere
    m_viewer->getCameraManipulator()->home(0);

    // Override distance: workpiece fills ~85% of viewport height
    // formula: distance = (diag/2) / tan(0.85 * vfov * 0.5)
    //   = diag / (2 * tan(19.125°)) = diag * 1.44
    osgGA::TrackballManipulator* tb =
        dynamic_cast<osgGA::TrackballManipulator*>(m_viewer->getCameraManipulator());
    if (tb)
    {
        tb->setDistance(diag * 1.44);
    }
}

void OSGWidget::adjustViewToMaxProjection()
{
    if (m_pcaCount < 10 || !m_viewer.valid() || !m_viewer->getCameraManipulator())
        return;

    double n = static_cast<double>(m_pcaCount);
    osg::Vec3 mean = m_pcaSum / static_cast<float>(n);

    double cxx = m_pcaSumXX / n - mean.x() * mean.x();
    double cxy = m_pcaSumXY / n - mean.x() * mean.y();
    double cxz = m_pcaSumXZ / n - mean.x() * mean.z();
    double cyy = m_pcaSumYY / n - mean.y() * mean.y();
    double cyz = m_pcaSumYZ / n - mean.y() * mean.z();
    double czz = m_pcaSumZZ / n - mean.z() * mean.z();

    Eigen::Matrix3d cov;
    cov << cxx, cxy, cxz,
           cxy, cyy, cyz,
           cxz, cyz, czz;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d ev = solver.eigenvectors().col(0);
    osg::Vec3 normal(ev(0), ev(1), ev(2));
    normal.normalize();

    osgGA::TrackballManipulator* tb =
        dynamic_cast<osgGA::TrackballManipulator*>(m_viewer->getCameraManipulator());
    if (tb)
    {
        osg::Quat rot;
        rot.makeRotate(osg::Vec3(0, 0, -1), normal);
        tb->setRotation(rot);
    }
}

void OSGWidget::streamNextBatch()
{
    if (m_streamDone || !m_streamFile.is_open())
    {
        m_streamTimer->stop();
        return;
    }

    int remaining = m_streamTotal - m_streamLoaded;
    int batch = std::min(m_streamBatch, remaining);

    if (batch <= 0)
    {
        m_streamFile.close();
        m_streamDone = true;
        m_streamTimer->stop();
        return;
    }

    // 批量整块读：一点 15B（3×f32 坐标+3×u8 颜色）——一次 read 拉整批到内存再解析，
    // 消除原逐字节 6 次/点的流调用（2000 万点=1.2 亿次 ifstream 调用，加载主瓶颈）
    static constexpr size_t kBytesPerPoint = 15;
    m_streamReadBuf.assign(batch * kBytesPerPoint, '\0');
    m_streamFile.read(m_streamReadBuf.data(), m_streamReadBuf.size());
    const size_t got = static_cast<size_t>(m_streamFile.gcount());
    const bool fileOk = m_streamFile.good() || got == m_streamReadBuf.size();

    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4ubArray> c = new osg::Vec4ubArray();
    v->reserve(batch);
    c->reserve(batch);

    osg::Vec3 batchMin(1e30f, 1e30f, 1e30f);
    osg::Vec3 batchMax(-1e30f, -1e30f, -1e30f);
    int batchCount = 0;

    const size_t nAvail = got / kBytesPerPoint;
    for (size_t i = 0; i < nAvail; ++i)
    {
        const char* p = m_streamReadBuf.data() + i * kBytesPerPoint;
        float x, y, z;
        std::memcpy(&x, p, 4);
        std::memcpy(&y, p + 4, 4);
        std::memcpy(&z, p + 8, 4);

        if (fileOk && std::isfinite(x) && std::isfinite(y) && std::isfinite(z)
            && std::fabs(x) < 1e5f && std::fabs(y) < 1e5f && std::fabs(z) < 1e5f)
        {
            v->push_back(osg::Vec3(x, y, z));
            c->push_back(osg::Vec4ub(135, 206, 250, 255));  // light blue
            if (x < batchMin.x()) batchMin.x() = x;
            if (y < batchMin.y()) batchMin.y() = y;
            if (z < batchMin.z()) batchMin.z() = z;
            if (x > batchMax.x()) batchMax.x() = x;
            if (y > batchMax.y()) batchMax.y() = y;
            if (z > batchMax.z()) batchMax.z() = z;
            batchCount++;

            m_pcaSum.x() += x;
            m_pcaSum.y() += y;
            m_pcaSum.z() += z;
            m_pcaSumXX += x * x;
            m_pcaSumXY += x * y;
            m_pcaSumXZ += x * z;
            m_pcaSumYY += y * y;
            m_pcaSumYZ += y * z;
            m_pcaSumZZ += z * z;
            m_pcaCount++;
        }
    }

    if (v->empty())
    {
        m_streamFile.close();
        m_streamDone = true;
        m_streamTimer->stop();
        return;
    }

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setVertexArray(v.get());
    geom->setColorArray(c.get());
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, v->size()));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom.get());
    geode->getOrCreateStateSet()->setAttribute(new osg::Point(1.0f));

    m_root->addChild(geode.get());

    m_streamLoaded += batch;



    if (m_streamBBoxValid)
    {
        if (batchMin.x() < m_streamBBoxMin.x()) m_streamBBoxMin.x() = batchMin.x();
        if (batchMin.y() < m_streamBBoxMin.y()) m_streamBBoxMin.y() = batchMin.y();
        if (batchMin.z() < m_streamBBoxMin.z()) m_streamBBoxMin.z() = batchMin.z();
        if (batchMax.x() > m_streamBBoxMax.x()) m_streamBBoxMax.x() = batchMax.x();
        if (batchMax.y() > m_streamBBoxMax.y()) m_streamBBoxMax.y() = batchMax.y();
        if (batchMax.z() > m_streamBBoxMax.z()) m_streamBBoxMax.z() = batchMax.z();
    }
    else
    {
        m_streamBBoxMin = batchMin;
        m_streamBBoxMax = batchMax;
        m_streamBBoxValid = true;
    }

    updateStreamCameraView();

    m_pcaTicks++;
    //if (m_pcaTicks >= 25 || m_streamDone)
    //{
    //    adjustViewToMaxProjection();
    //    m_pcaTicks = 0;
    //    m_pcaCount = 0;
    //    m_pcaSum.set(0, 0, 0);
    //    m_pcaSumXX = m_pcaSumXY = m_pcaSumXZ = 0;
    //    m_pcaSumYY = m_pcaSumYZ = m_pcaSumZZ = 0;
    //}

    emit streamProgress(m_streamLoaded, m_streamTotal);

    if (m_streamLoaded >= m_streamTotal)
    {
        m_streamFile.close();
        m_streamDone = true;
        m_streamTimer->stop();
        // 加载耗时打点（性能口径：批读优化后 2000 万点目标 <10s）
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - m_streamStart).count();
        JMW_LOG_INFO("03-OSGWidget", "[OSGWidget] PLY 流式加载完成: {} 点 / {} ms（{} 点/秒）",
                     m_streamLoaded, ms,
                     ms > 0 ? static_cast<long long>(m_streamLoaded) * 1000 / ms : 0);
    }
}

void OSGWidget::initializeGL()
{
    m_viewer = new osgViewer::Viewer();
    m_viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);

    int w = width() * devicePixelRatio();
    int h = height() * devicePixelRatio();

    osg::ref_ptr<osg::Camera> camera = new osg::Camera();
    camera->setViewport(0, 0, w, h);
    camera->setProjectionMatrixAsPerspective(45.0,
        static_cast<double>(w) / static_cast<double>(h), 1.0, 10000.0);
    m_userProjection = camera->getProjectionMatrix();
    camera->setClearColor(osg::Vec4(0.412f, 0.412f, 0.412f, 1.0f));

    m_gw = new osgViewer::GraphicsWindowEmbedded(0, 0, w, h);
    camera->setGraphicsContext(m_gw.get());

    m_viewer->setCamera(camera.get());

    osg::ref_ptr<osg::Group> sceneRoot = new osg::Group();
    sceneRoot->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    sceneRoot->addChild(m_root.get());
    m_viewer->setSceneData(sceneRoot.get());

    osgGA::TrackballManipulator* manip = new osgGA::TrackballManipulator();
    manip->setAllowThrow(false);
    m_viewer->setCameraManipulator(manip);

    createAxesIndicator();
    // createCenterOverlay();  // 暂时禁用 center.png 背景叠加

    // CullCallback captures the EXACT rendering matrices for hit-testing
    m_hitTestCallback = new HitTestCullCallback(
        &m_capturedMVP, &m_camProjAfterFrame, &m_camViewAfterFrame);
    m_viewer->getCamera()->addCullCallback(m_hitTestCallback.get());
}

void OSGWidget::resizeGL(int w, int h)
{
    int s = devicePixelRatio();
    qDebug("3D viewport size: %d x %d", w * s, h * s);
    m_gw->resized(0, 0, w * s, h * s);
    m_viewer->getCamera()->setViewport(0, 0, w * s, h * s);

    // 保持当前投影的 fov/near/far，仅更新 aspect（避免旋转操作时 resize 重置视角）
    {
        double fov, aspect, zNear, zFar;
        m_userProjection.getPerspective(fov, aspect, zNear, zFar);
        aspect = static_cast<double>(w * s) / static_cast<double>(h * s);
        m_viewer->getCamera()->setProjectionMatrixAsPerspective(fov, aspect, zNear, zFar);
        m_userProjection = m_viewer->getCamera()->getProjectionMatrix();
    }

    if (m_axesCamera.valid())
    {
        int axesSize = 200;
        int margin = 10;
        m_axesCamera->setViewport(margin, margin, axesSize * s, axesSize * s);
        m_axesCamera->setProjectionMatrixAsOrtho(-2.5f, 2.5f, -2.5f, 2.5f, -2.5f, 2.5f);
    }

    // createCenterOverlay();
}

void OSGWidget::paintGL()
{
    if (!m_viewer.valid())
        return;
    if (m_renderSuspended)
        return;                          // 挂起态：不画，保留末帧（timer 已停）

    if (m_gw.valid())
        m_gw->makeCurrent();

    // 恢复投影矩阵（OSG 内部可能在 frame() 中修改）
    m_viewer->getCamera()->setProjectionMatrix(m_userProjection);

    // 恢复锁定的视图矩阵
    if (m_viewLocked)
        m_viewer->getCamera()->setViewMatrix(m_userView);

    updateAxesView();

    // P1 渲染护栏：渲染无权弄死 App——frame() 异常→挂起循环＋上报，Qt 事件循环存活
    // P2.2 帧耗时探针：连续 5 帧 >50ms → 预算降一级自动抽稀（0x0311）
    const auto t0 = std::chrono::steady_clock::now();
    try {
        m_viewer->frame();
        ++m_framesDrawn;

        // Capture matrices that OSG actually used for this frame
        m_camViewAfterFrame = m_viewer->getCamera()->getViewMatrix();
        m_camProjAfterFrame = m_viewer->getCamera()->getProjectionMatrix();
    } catch (const std::exception& e) {
        suspendRender(std::string("渲染帧异常: ") + e.what());
    } catch (...) {
        suspendRender("渲染帧未知异常");
    }
    m_lastFrameMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
    if (m_lastFrameMs > 50.0) {
        if (++m_overBudgetStreak >= 5 && m_degradeLevel < 3) {
            ++m_degradeLevel;
            m_overBudgetStreak = 0;
            m_maxIngestPoints = kDegradeBudgets[m_degradeLevel];
            reportFault(static_cast<int>(Scanner::render::RenderEvent::FrameOverbudget),
                        "渲染连续超 50ms，摄入预算降级至 level " +
                            std::to_string(m_degradeLevel));
        }
    } else {
        m_overBudgetStreak = 0;
    }
}

// ============================================================================
// P1 渲染加固实现件
// ============================================================================
const size_t OSGWidget::kDegradeBudgets[4] = {
    size_t{8} << 20,   // level 0：8M（默认）
    size_t{4} << 20,   // level 1：4M
    size_t{1} << 20,   // level 2：1M
    size_t{256} << 10, // level 3：256K
};

OSGWidget::RenderStats OSGWidget::renderStats() const {
    RenderStats s;
    s.framesDrawn = m_framesDrawn;
    s.ingestCount = m_ingestCount;
    s.lastFrameMs = m_lastFrameMs;
    s.degradeLevel = m_degradeLevel;
    s.suspended = m_renderSuspended;
    return s;
}

void OSGWidget::reportFault(int code, const std::string& msg)
{
    if (m_faultSink)
        m_faultSink(code, msg);          // app 装配桥接（EventBus/日志）
    else
        qWarning("[OSGWidget] render event 0x%04X: %s", code, msg.c_str());
}

void OSGWidget::suspendRender(const std::string& why)
{
    if (m_renderSuspended) return;       // 幂等（只报首个异常）
    m_renderSuspended = true;
    m_timer->stop();
    reportFault(static_cast<int>(Scanner::render::RenderEvent::RenderSuspended),
                why + "（循环已挂起，UI 存活；tryResumeRender 可恢复）");
}

bool OSGWidget::tryResumeRender()
{
    if (!m_renderSuspended) return true;
    m_renderSuspended = false;
    m_timer->start(33);                 // 渲染 30fps（原 16ms≈60，降半省资源）                  // 与 ctor 同节拍
    reportFault(static_cast<int>(Scanner::render::RenderEvent::RenderResumed),
                "渲染循环恢复");
    return true;
}

void OSGWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (m_lassoMode)
        return;
    if (!m_gw.valid()) return;
    float x = event->pos().x();
    float y = event->pos().y();

    if (m_firstMouse)
    {
        m_lastMouseX = x;
        m_lastMouseY = y;
        m_firstMouse = false;
    }

    m_lastMouseX = x;
    m_lastMouseY = y;

    m_gw->getEventQueue()->mouseMotion(x, y);
}

void OSGWidget::mousePressEvent(QMouseEvent *event)
{
    m_userInteracting = true;              // 视角跟随避让（Release 复位）
    if (m_lassoMode)
    {
        if (event->button() == Qt::LeftButton)
            addLassoPoint(event->pos().x(), event->pos().y());
        else if (event->button() == Qt::RightButton)
            closeLasso();
        return;
    }
    if (!m_gw.valid()) return;
    m_firstMouse = true;
    unsigned int button = Qt::LeftButton;
    if (event->button() == Qt::MidButton)     button = osgGA::GUIEventAdapter::MIDDLE_MOUSE_BUTTON;
    else if (event->button() == Qt::RightButton) button = osgGA::GUIEventAdapter::RIGHT_MOUSE_BUTTON;
    else                                          button = osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON;

    m_gw->getEventQueue()->mouseButtonPress(event->pos().x(), event->pos().y(), button);
}

void OSGWidget::mouseReleaseEvent(QMouseEvent *event)
{
    m_userInteracting = false;             // 视角跟随恢复（Press 置位处）
    if (m_lassoMode)
        return;
    if (!m_gw.valid()) return;
    unsigned int button = Qt::LeftButton;
    if (event->button() == Qt::MidButton)     button = osgGA::GUIEventAdapter::MIDDLE_MOUSE_BUTTON;
    else if (event->button() == Qt::RightButton) button = osgGA::GUIEventAdapter::RIGHT_MOUSE_BUTTON;
    else                                          button = osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON;

    m_gw->getEventQueue()->mouseButtonRelease(event->pos().x(), event->pos().y(), button);
}

void OSGWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (m_lassoMode)
        return;
    if (!m_gw.valid()) return;
    m_gw->getEventQueue()->mouseDoubleButtonPress(event->pos().x(), event->pos().y(),
        osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON);
}

void OSGWidget::wheelEvent(QWheelEvent *event)
{
    if (m_lassoMode)
        return;
    if (!m_gw.valid()) return;
    int delta = event->angleDelta().y();
    m_gw->getEventQueue()->mouseScroll(
        delta > 0 ? osgGA::GUIEventAdapter::SCROLL_UP : osgGA::GUIEventAdapter::SCROLL_DOWN);
}

// ---- Lasso / Polyline selection ----

void OSGWidget::enterLassoDeleteMode()
{
    m_lassoDeleteMode = true;
    enterLassoMode();
}

void OSGWidget::enterLassoMode()
{
    clearHighlight();
    m_lassoMode = true;
    m_lassoPoints = new osg::Vec2Array();
    m_selectedPolylines.clear();

    m_lassoCamera = new osg::Camera();
    m_lassoCamera->setRenderOrder(osg::Camera::POST_RENDER, 5);
    m_lassoCamera->setClearMask(GL_DEPTH_BUFFER_BIT);
    m_lassoCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    m_lassoCamera->setViewMatrix(osg::Matrix::identity());
    m_lassoCamera->setAllowEventFocus(false);
    m_lassoCamera->setGraphicsContext(m_gw.get());

    int s = devicePixelRatio();
    int vpw = width() * s;
    int vph = height() * s;
    m_lassoCamera->setViewport(0, 0, vpw, vph);
    m_lassoCamera->setProjectionMatrixAsOrtho2D(-vpw * 0.5f, vpw * 0.5f, -vph * 0.5f, vph * 0.5f);
    m_lassoCamera->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    m_lassoVerts = new osg::Vec3Array();
    m_lassoGeom = new osg::Geometry();
    m_lassoGeom->setVertexArray(m_lassoVerts);

    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
    colors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    m_lassoGeom->setColorArray(colors, osg::Array::BIND_OVERALL);

    m_lassoDrawArrays = new osg::DrawArrays(GL_LINE_STRIP, 0, 0);
    m_lassoGeom->addPrimitiveSet(m_lassoDrawArrays);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(m_lassoGeom);
    geode->getOrCreateStateSet()->setAttribute(new osg::LineWidth(3.0f));
    m_lassoCamera->addChild(geode);

    osg::Group* sceneRoot = m_viewer->getSceneData()->asGroup();
    if (sceneRoot)
        sceneRoot->addChild(m_lassoCamera.get());
}

void OSGWidget::deleteSelectedPoints()
{
    if (m_highlights.empty())
        return;

    for (auto& entry : m_highlights)
    {
        pushDeleteUndo(entry.geom, entry.indices, entry.originalColors);

        osg::Vec4ubArray* colors = dynamic_cast<osg::Vec4ubArray*>(
            entry.geom->getColorArray());
        if (!colors) continue;

        for (unsigned int vi : entry.indices)
        {
            if (vi < colors->size())
                (*colors)[vi] = osg::Vec4ub(0, 0, 0, 0);
        }
        colors->dirty();
        entry.geom->setColorArray(colors);

        osg::StateSet* ss = entry.geom->getOrCreateStateSet();
        ss->setMode(GL_BLEND, osg::StateAttribute::ON);
        osg::ref_ptr<osg::BlendFunc> bf = new osg::BlendFunc();
        bf->setFunction(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        ss->setAttribute(bf);
    }

    clearHighlight();
    m_selectedPolylines.clear();
}

void OSGWidget::exitLassoMode()
{
    m_lassoMode = false;
    if (m_lassoPoints.valid())
        m_lassoPoints->clear();

    if (m_lassoCamera.valid())
    {
        osg::Group* sceneRoot = m_viewer->getSceneData()->asGroup();
        if (sceneRoot)
            sceneRoot->removeChild(m_lassoCamera.get());
        m_lassoCamera = nullptr;
    }
    m_lassoGeom = nullptr;
    m_lassoVerts = nullptr;
    m_lassoDrawArrays = nullptr;
}

void OSGWidget::addLassoPoint(float mx, float my)
{
    int s = devicePixelRatio();
    int vpw = width() * s;
    int vph = height() * s;

    float sx = mx * s - vpw * 0.5f;
    float sy = vph * 0.5f - my * s;

    m_lassoPoints->push_back(osg::Vec2(sx, sy));
    updateLassoGeometry();

    if (m_lassoPoints->size() > 2)
    {
        const osg::Vec2& first = m_lassoPoints->front();
        float dx = sx - first.x();
        float dy = sy - first.y();
        if (dx * dx + dy * dy < 100.0f)
        {
            closeLasso();
            return;
        }
    }
}

void OSGWidget::debugValidateProjection()
{
    if (!m_viewer.valid()) return;

    osg::Matrix P = m_userProjection;
    osg::Matrix V = m_viewer->getCamera()->getViewMatrix();
    osg::Matrix iV = osg::Matrix::inverse(V);
    osg::Matrix captured = m_capturedMVP;

    int vpw = m_hitTestVpw > 0 ? m_hitTestVpw : (width() * devicePixelRatio());
    int vph = m_hitTestVph > 0 ? m_hitTestVph : (height() * devicePixelRatio());

    // Compare V with manipulator matrices
    osgGA::TrackballManipulator* tb =
        dynamic_cast<osgGA::TrackballManipulator*>(m_viewer->getCameraManipulator());
    osg::Matrix manipMat, manipInv;
    if (tb) { manipMat = tb->getMatrix(); manipInv = tb->getInverseMatrix(); }

    double diffInv = 0, diffMat = 0;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
        diffInv += fabs(V(r,c) - manipInv(r,c));
        diffMat += fabs(V(r,c) - manipMat(r,c));
    }

    // Find a sample point
    osg::Vec3 sample(0,0,0);
    bool found = false;
    for (unsigned int ci = 0; ci < m_root->getNumChildren() && !found; ++ci) {
        osg::Geode* g = m_root->getChild(ci)->asGeode();
        if (!g) continue;
        for (unsigned int di = 0; di < g->getNumDrawables() && !found; ++di) {
            osg::Geometry* geom = g->getDrawable(di)->asGeometry();
            if (!geom) continue;
            osg::Vec3Array* v = dynamic_cast<osg::Vec3Array*>(geom->getVertexArray());
            if (v && !v->empty()) { sample = (*v)[0]; found = true; }
        }
    }

    auto toScreen = [vpw, vph](const osg::Matrix& mvp, const osg::Vec3& w, float& sx, float& sy) -> bool {
        osg::Vec4 clip = mvp * osg::Vec4(w.x(), w.y(), w.z(), 1.0);
        if (fabs(clip.w()) < 1e-10) return false;
        if (clip.w() < 0) { clip.x()=-clip.x(); clip.y()=-clip.y(); clip.z()=-clip.z(); clip.w()=-clip.w(); }
        float ndx = clip.x()/clip.w(), ndy = clip.y()/clip.w();
        if (ndx<-1||ndx>1||ndy<-1||ndy>1) return false;
        sx = ndx*vpw*0.5f; sy = ndy*vph*0.5f;
        return true;
    };

    float s1x, s1y, s2x, s2y, s3x, s3y;
    bool ok1 = toScreen(P * V, sample, s1x, s1y);
    bool ok2 = toScreen(P * iV, sample, s2x, s2y);
    bool ok3 = toScreen(captured, sample, s3x, s3y);

    auto matStr = [](const osg::Matrix& m) {
        QString s;
        for (int r = 0; r < 4; ++r)
            s += QString("  [%1 %2 %3 %4]\n")
                .arg(m(r,0),8,'f',3).arg(m(r,1),8,'f',3).arg(m(r,2),8,'f',3).arg(m(r,3),8,'f',3);
        return s;
    };

    QString msg;
    msg += "=== Projection Validation ===\n\n";
    msg += QString("getViewMatrix() vs manip.getInverseMatrix() L1: %1\n").arg(diffInv,0,'f',4);
    msg += QString("getViewMatrix() vs manip.getMatrix()        L1: %2\n").arg(diffMat,0,'f',4);
    msg += (diffInv < diffMat) ? "→ getViewMatrix() IS the view matrix (world→eye)\n" : "→ getViewMatrix() IS the placement matrix (eye→world)\n";

    if (found) {
        msg += QString("\nSample point: (%1, %2, %3)\n").arg(sample.x(),0,'f',1).arg(sample.y(),0,'f',1).arg(sample.z(),0,'f',1);
        msg += QString("P * V  (correct Proj*View):   ");
        msg += ok1 ? QString("(%1, %2)\n").arg(s1x,0,'f',0).arg(s1y,0,'f',0) : "OFF SCREEN\n";
        msg += QString("P * iV (wrong, for ref):      ");
        msg += ok2 ? QString("(%1, %2)\n").arg(s2x,0,'f',0).arg(s2y,0,'f',0) : "OFF SCREEN\n";
        msg += QString("captured MVP (mv*proj, wrong):");
        msg += ok3 ? QString("(%1, %2)\n").arg(s3x,0,'f',0).arg(s3y,0,'f',0) : "OFF SCREEN\n";
    }

    // Compare raw vs after-frame matrices
    osg::Matrix selV = m_hitTestView;
    osg::Matrix selP = m_hitTestProj;
    osg::Matrix afterV = m_camViewAfterFrame;
    osg::Matrix afterP = m_camProjAfterFrame;
    double dV = 0, dP = 0;
    for (int r=0;r<4;++r) for(int c=0;c<4;++c) {
        dV += fabs(selV(r,c)-afterV(r,c));
        dP += fabs(selP(r,c)-afterP(r,c));
    }
    msg += QString("\n--- Selection vs frame-after matrices ---\n");
    msg += QString("View diff: %1  Proj diff: %2\n").arg(dV,0,'f',4).arg(dP,0,'f',4);

    msg += "\nSelection V (m_hitTestView):\n" + matStr(selV);
    msg += "\nSelection P (m_hitTestProj):\n" + matStr(selP);
    msg += "\nAfter-frame V:\n" + matStr(afterV);
    msg += "\nAfter-frame P:\n" + matStr(afterP);

    msg += "\n--- Raw camera matrices ---\n";
    msg += "\nV matrix (getViewMatrix):\n" + matStr(V);
    msg += "\niV matrix:\n" + matStr(iV);
    msg += "\ncapturedMVP (V*P from cull):\n" + matStr(captured);

    QMessageBox::information(this, "Projection Debug", msg);
}

void OSGWidget::closeLasso()
{
    if (!m_lassoPoints.valid() || m_lassoPoints->size() < 3)
    {
        exitLassoMode();
        emit lassoCompleted();
        return;
    }

    m_selectedPolylines.push_back(new osg::Vec2Array(*m_lassoPoints));

    // Show the fully closed polygon FIRST
    m_lassoVerts->clear();
    for (unsigned int i = 0; i < m_lassoPoints->size(); ++i)
    {
        const osg::Vec2& pt = (*m_lassoPoints)[i];
        m_lassoVerts->push_back(osg::Vec3(pt.x(), pt.y(), 0.0f));
    }
    m_lassoVerts->dirty();
    m_lassoGeom->removePrimitiveSet(0, m_lassoGeom->getNumPrimitiveSets());
    m_lassoDrawArrays = new osg::DrawArrays(GL_LINE_LOOP, 0, m_lassoVerts->size());
    m_lassoGeom->addPrimitiveSet(m_lassoDrawArrays);
    update();
    QApplication::processEvents();

    // Use the projection and view matrices captured by CullCallback during the last frame().
    // computeRenderingMVP() transposes them to match OpenGL's column-major convention
    // (P_GL * V_GL), which produces the correct clip.w = -z_eye.
    m_hitTestView = m_camViewAfterFrame;
    m_hitTestProj = m_camProjAfterFrame;


    m_hitTestVpw = width() * devicePixelRatio();
    m_hitTestVph = height() * devicePixelRatio();

    if (m_lassoDeleteMode)
    {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            QStringLiteral("确认删除"),
            QStringLiteral("确定要删除多段线圈定的区域内的点吗？"),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No);

        if (reply == QMessageBox::Yes)
            deletePointsInPolyline();

        m_lassoDeleteMode = false;
        exitLassoMode();
        emit lassoCompleted();
    }
    else
    {
        highlightSelectedPoints();
        exitLassoMode();
        update();
        emit lassoCompleted();
    }
}

void OSGWidget::updateLassoGeometry()
{
    if (!m_lassoVerts.valid() || !m_lassoGeom.valid())
        return;

    m_lassoVerts->clear();
    for (unsigned int i = 0; i < m_lassoPoints->size(); ++i)
    {
        const osg::Vec2& pt = (*m_lassoPoints)[i];
        m_lassoVerts->push_back(osg::Vec3(pt.x(), pt.y(), 0.0f));
    }
    m_lassoVerts->dirty();

    m_lassoGeom->removePrimitiveSet(0, m_lassoGeom->getNumPrimitiveSets());
    m_lassoDrawArrays = new osg::DrawArrays(GL_LINE_STRIP, 0, m_lassoVerts->size());
    m_lassoGeom->addPrimitiveSet(m_lassoDrawArrays);
}

bool OSGWidget::isPointInPolygon2D(const osg::Vec2d& point, const osg::Vec2Array& polygon)
{
    bool inside = false;
    int n = polygon.size();
    for (int i = 0, j = n - 1; i < n; j = i++)
    {
        if (((polygon[i].y() > point.y()) != (polygon[j].y() > point.y())) &&
            (point.x() < (polygon[j].x() - polygon[i].x()) * (point.y() - polygon[i].y())
                / (polygon[j].y() - polygon[i].y()) + polygon[i].x()))
        {
            inside = !inside;
        }
    }
    return inside;
}

void OSGWidget::highlightSelectedPoints()
{
    if (m_selectedPolylines.empty() || !m_viewer.valid())
        return;

    osg::Vec2Array* poly = m_selectedPolylines.back().get();
    if (!poly || poly->size() < 3)
        return;

    osg::Matrix mvp = computeRenderingMVP(m_hitTestProj, m_hitTestView);
    const int vpw = m_hitTestVpw;
    const int vph = m_hitTestVph;

    // 命中预筛加速（同 deletePointsInPolyline：包围盒预筛+预取边数组）
    double minX = 1e30, minY = 1e30, maxX = -1e30, maxY = -1e30;
    const size_t nEdges = poly->size();
    std::vector<double> px(nEdges), py(nEdges);
    for (size_t i = 0; i < nEdges; ++i) {
        const double x = (*poly)[i].x(), y = (*poly)[i].y();
        px[i] = x; py[i] = y;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
    }

    for (unsigned int ci = 0; ci < m_root->getNumChildren(); ++ci)
    {
        osg::Geode* geode = m_root->getChild(ci)->asGeode();
        if (!geode) continue;

        for (unsigned int di = 0; di < geode->getNumDrawables(); ++di)
        {
            osg::Geometry* geom = geode->getDrawable(di)->asGeometry();
            if (!geom) continue;

            osg::Vec3Array* verts = dynamic_cast<osg::Vec3Array*>(geom->getVertexArray());
            osg::Vec4ubArray* colors = dynamic_cast<osg::Vec4ubArray*>(geom->getColorArray());
            if (!verts || !colors || verts->size() != colors->size())
                continue;

            bool alreadyHighlighted = false;
            for (auto& h : m_highlights)
            {
                if (h.geom == geom) { alreadyHighlighted = true; break; }
            }
            if (alreadyHighlighted) continue;

            HighlightEntry entry;
            entry.geom = geom;
            entry.indices.reserve(verts->size() / 16 + 16);
            entry.originalColors.reserve(verts->size() / 16 + 16);

            const unsigned int nVerts = static_cast<unsigned int>(verts->size());
            for (unsigned int vi = 0; vi < nVerts; ++vi)
            {
                const osg::Vec3& vp = (*verts)[vi];
                float cx = mvp(0,0)*vp.x() + mvp(0,1)*vp.y() + mvp(0,2)*vp.z() + mvp(0,3);
                float cy = mvp(1,0)*vp.x() + mvp(1,1)*vp.y() + mvp(1,2)*vp.z() + mvp(1,3);
                float cw = mvp(3,0)*vp.x() + mvp(3,1)*vp.y() + mvp(3,2)*vp.z() + mvp(3,3);
                if (cw < 1e-10f && cw > -1e-10f) continue;
                if (cw < 0.0f) { cx = -cx; cy = -cy; cw = -cw; }

                const float sx = (cx / cw) * vpw * 0.5f;
                const float sy = (cy / cw) * vph * 0.5f;

                if (sx < minX || sx > maxX || sy < minY || sy > maxY)
                    continue;                      // 包围盒外：一次淘汰

                bool inside = false;
                for (size_t i = 0, j = nEdges - 1; i < nEdges; j = i++) {
                    if ((py[i] > sy) != (py[j] > sy) &&
                        (sx < (px[j] - px[i]) * (sy - py[i]) / (py[j] - py[i]) + px[i]))
                        inside = !inside;
                }
                if (inside) {
                    entry.indices.push_back(vi);
                    entry.originalColors.push_back((*colors)[vi]);
                }
            }

            if (!entry.indices.empty())
            {
                osg::ref_ptr<osg::Vec4ubArray> newColors = new osg::Vec4ubArray(*colors);
                for (unsigned int vi : entry.indices)
                    (*newColors)[vi] = osg::Vec4ub(0, 0, 255, 255);
                newColors->dirty();
                geom->setColorArray(newColors.get(), osg::Array::BIND_PER_VERTEX);
                geom->dirtyDisplayList();
                m_highlights.push_back(std::move(entry));
            }
        }
    }
}

void OSGWidget::clearHighlight()
{
    for (auto& entry : m_highlights)
    {
        osg::Vec4ubArray* colors = dynamic_cast<osg::Vec4ubArray*>(
            entry.geom->getColorArray());
        if (!colors) continue;

        for (size_t i = 0; i < entry.indices.size(); ++i)
        {
            unsigned int vi = entry.indices[i];
            if (vi < colors->size())
                (*colors)[vi] = entry.originalColors[i];
        }
        colors->dirty();
        entry.geom->setColorArray(colors);
    }
    m_highlights.clear();
}

void OSGWidget::pushDeleteUndo(osg::Geometry* geom,
    const std::vector<unsigned int>& indices,
    const std::vector<osg::Vec4ub>& origColors)
{
    if (indices.empty()) return;
    DeleteEntry entry;
    entry.geom = geom;
    entry.indices = indices;
    entry.originalColors = origColors;
    m_deleteHistory.push_back(std::move(entry));
}

void OSGWidget::undoDelete()
{
    if (m_deleteHistory.empty())
        return;

    const DeleteEntry& entry = m_deleteHistory.back();
    osg::Vec4ubArray* colors = dynamic_cast<osg::Vec4ubArray*>(
        entry.geom->getColorArray());
    if (colors)
    {
        for (size_t i = 0; i < entry.indices.size(); ++i)
        {
            unsigned int vi = entry.indices[i];
            if (vi < colors->size())
                (*colors)[vi] = entry.originalColors[i];
        }
        colors->dirty();
        entry.geom->setColorArray(colors);
    }
    m_deleteHistory.pop_back();
}
void OSGWidget::deletePointsInPolyline()
{
    if (!m_viewer.valid() || m_selectedPolylines.empty())
        return;

    osg::Vec2Array* poly = m_selectedPolylines.back().get();
    if (!poly || poly->size() < 3)
        return;

    osg::Matrix mvp = computeRenderingMVP(m_hitTestProj, m_hitTestView);
    const int vpw = m_hitTestVpw;
    const int vph = m_hitTestVph;

    // 命中预筛加速：①多边形屏幕包围盒（域外点一次比较淘汰——大点云场景淘汰
    // 95%+ 顶点，免进 O(边数) 射线法）②预拷贝边数组到连续 vector<double>（消除
    // ref_ptr 寻址与 float/double 混转）③命中容器 reserve（免反复扩容拷贝）
    double minX = 1e30, minY = 1e30, maxX = -1e30, maxY = -1e30;
    const size_t nEdges = poly->size();
    std::vector<double> px(nEdges), py(nEdges);
    for (size_t i = 0; i < nEdges; ++i) {
        const double x = (*poly)[i].x(), y = (*poly)[i].y();
        px[i] = x; py[i] = y;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
    }

    for (unsigned int ci = 0; ci < m_root->getNumChildren(); ++ci)
    {
        osg::Geode* geode = m_root->getChild(ci)->asGeode();
        if (!geode) continue;

        for (unsigned int di = 0; di < geode->getNumDrawables(); ++di)
        {
            osg::Geometry* geom = geode->getDrawable(di)->asGeometry();
            if (!geom) continue;

            osg::Vec3Array* verts = dynamic_cast<osg::Vec3Array*>(geom->getVertexArray());
            osg::Vec4ubArray* colors = dynamic_cast<osg::Vec4ubArray*>(geom->getColorArray());
            if (!verts || !colors || verts->size() != colors->size())
                continue;

            std::vector<unsigned int> delIndices;
            std::vector<osg::Vec4ub> delOrigColors;
            delIndices.reserve(verts->size() / 16 + 16);
            delOrigColors.reserve(verts->size() / 16 + 16);

            const unsigned int nVerts = static_cast<unsigned int>(verts->size());
            for (unsigned int vi = 0; vi < nVerts; ++vi)
            {
                const osg::Vec3& vp = (*verts)[vi];
                float cx = mvp(0,0)*vp.x() + mvp(0,1)*vp.y() + mvp(0,2)*vp.z() + mvp(0,3);
                float cy = mvp(1,0)*vp.x() + mvp(1,1)*vp.y() + mvp(1,2)*vp.z() + mvp(1,3);
                float cw = mvp(3,0)*vp.x() + mvp(3,1)*vp.y() + mvp(3,2)*vp.z() + mvp(3,3);
                if (cw < 1e-10f && cw > -1e-10f) continue;
                if (cw < 0.0f) { cx = -cx; cy = -cy; cw = -cw; }

                const float sx = (cx / cw) * vpw * 0.5f;
                const float sy = (cy / cw) * vph * 0.5f;

                if (sx < minX || sx > maxX || sy < minY || sy > maxY)
                    continue;                      // 包围盒外：一次淘汰（原逐边射线法的主开销）

                // 内联射线法（原 isPointInPolygon2D——语义不变，去函数调用/寻址开销）
                bool inside = false;
                for (size_t i = 0, j = nEdges - 1; i < nEdges; j = i++) {
                    if ((py[i] > sy) != (py[j] > sy) &&
                        (sx < (px[j] - px[i]) * (sy - py[i]) / (py[j] - py[i]) + px[i]))
                        inside = !inside;
                }
                if (inside) {
                    delIndices.push_back(vi);
                    delOrigColors.push_back((*colors)[vi]);
                }
            }
            if (!delIndices.empty())
            {
                pushDeleteUndo(geom, delIndices, delOrigColors);
                osg::ref_ptr<osg::Vec4ubArray> newColors = new osg::Vec4ubArray(*colors);
                for (unsigned int vi : delIndices)
                    (*newColors)[vi] = osg::Vec4ub(0, 0, 0, 0);
                newColors->dirty();
                geom->setColorArray(newColors.get(), osg::Array::BIND_PER_VERTEX);
                geom->dirtyDisplayList();
                osg::StateSet* ss = geom->getOrCreateStateSet();
                ss->setMode(GL_BLEND, osg::StateAttribute::ON);
                osg::ref_ptr<osg::BlendFunc> bf = new osg::BlendFunc();
                bf->setFunction(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                ss->setAttribute(bf);
            }
        }
    }

    update();
    m_selectedPolylines.clear();
}

void OSGWidget::createCenterOverlay()
{
    if (!m_viewer.valid())
        return;

    if (m_centerOverlayCamera.valid())
    {
        osg::Group* sceneRoot = m_viewer->getSceneData()->asGroup();
        if (sceneRoot)
            sceneRoot->removeChild(m_centerOverlayCamera.get());
        m_centerOverlayCamera = nullptr;
    }

    m_centerOverlayCamera = new osg::Camera();
    m_centerOverlayCamera->setRenderOrder(osg::Camera::POST_RENDER);
    m_centerOverlayCamera->setClearMask(GL_DEPTH_BUFFER_BIT);
    m_centerOverlayCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    m_centerOverlayCamera->setViewMatrix(osg::Matrix::identity());
    m_centerOverlayCamera->setAllowEventFocus(false);
    m_centerOverlayCamera->setGraphicsContext(m_gw.get());

    int s = devicePixelRatio();
    int vpw = width() * s;
    int vph = height() * s;
    m_centerOverlayCamera->setViewport(0, 0, vpw, vph);
    m_centerOverlayCamera->setProjectionMatrixAsOrtho2D(-vpw * 0.5f, vpw * 0.5f, -vph * 0.5f, vph * 0.5f);
    m_centerOverlayCamera->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();

    // Load center.png as textured quad at viewport center
    QImage qimg(QStringLiteral("E:/workfold/20260509intergrate/center.png"));
    if (!qimg.isNull())
    {
        qimg = qimg.convertToFormat(QImage::Format_RGBA8888);
        int imgW = qimg.width(), imgH = qimg.height();

        unsigned char* data = new unsigned char[imgW * imgH * 4];
        for (int y = 0; y < imgH; ++y)
            memcpy(data + y * imgW * 4, qimg.constScanLine(imgH - 1 - y), imgW * 4);

        osg::ref_ptr<osg::Image> osgImg = new osg::Image();
        osgImg->setImage(imgW, imgH, 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
            data, osg::Image::USE_NEW_DELETE);

        osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D();
        tex->setImage(osgImg);
        tex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        tex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);

        osg::ref_ptr<osg::Geometry> tgeom = new osg::Geometry();
        osg::ref_ptr<osg::Vec3Array> tverts = new osg::Vec3Array();
        float imgW2 = vpw * 0.5f;
        float imgH2 = vph * 0.5f;
        tverts->push_back(osg::Vec3(-imgW2, -imgH2, 0.0f));
        tverts->push_back(osg::Vec3( imgW2, -imgH2, 0.0f));
        tverts->push_back(osg::Vec3( imgW2,  imgH2, 0.0f));
        tverts->push_back(osg::Vec3(-imgW2,  imgH2, 0.0f));
        tgeom->setVertexArray(tverts);
        tgeom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

        osg::ref_ptr<osg::Vec2Array> ttc = new osg::Vec2Array();
        ttc->push_back(osg::Vec2(0.0f, 0.0f));
        ttc->push_back(osg::Vec2(1.0f, 0.0f));
        ttc->push_back(osg::Vec2(1.0f, 1.0f));
        ttc->push_back(osg::Vec2(0.0f, 1.0f));
        tgeom->setTexCoordArray(0, ttc);

        tgeom->getOrCreateStateSet()->setTextureAttributeAndModes(0, tex);
        tgeom->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
        tgeom->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

        geode->addDrawable(tgeom);
    }

    geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
    m_centerOverlayCamera->addChild(geode);
    m_centerOverlayCamera->setNodeMask(m_centerOverlayVisible ? 0xffffffff : 0);

    osg::Group* sceneRoot = m_viewer->getSceneData()->asGroup();
    if (sceneRoot)
        sceneRoot->addChild(m_centerOverlayCamera.get());
}

// ============================================================================
// LeadScan 移植：带法线的动态点云（VBO + DYNAMIC）
// ============================================================================
void OSGWidget::loadPointCloudWithNormals(const std::vector<osg::Vec3>& points,
                                          const std::vector<osg::Vec3>& normals,
                                          const osg::Vec4& color)
{
    if (points.empty()) return;

    // 首次创建
    if (!m_cloudRoot) {
        m_cloudRoot = new osg::Group;
        m_cloudGeode = new osg::Geode;
        m_cloudGeom = new osg::Geometry;
        m_cloudGeom->setUseVertexBufferObjects(true);
        m_cloudGeom->setUseDisplayList(false);
        m_cloudGeom->setDataVariance(osg::Object::DYNAMIC);
        m_cloudGeode->addDrawable(m_cloudGeom);
        m_cloudRoot->addChild(m_cloudGeode);
        m_root->addChild(m_cloudRoot);

        // 点大小
        osg::ref_ptr<osg::Point> pointSize = new osg::Point;
        pointSize->setSize(3.0f);
        m_cloudRoot->getOrCreateStateSet()->setAttribute(pointSize);

        m_cloudCoords = new osg::Vec3Array;
        m_cloudNormals = new osg::Vec3Array;
        m_cloudColors = new osg::Vec4Array;
    }

    // 设置数据
    m_cloudCoords->assign(points.begin(), points.end());
    m_cloudColors->resize(points.size(), color);

    m_cloudGeom->setVertexArray(m_cloudCoords);
    m_cloudGeom->setColorArray(m_cloudColors);
    m_cloudGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    if (normals.size() == points.size()) {
        m_cloudNormals->assign(normals.begin(), normals.end());
        m_cloudGeom->setNormalArray(m_cloudNormals);
        m_cloudGeom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    m_cloudGeom->setPrimitiveSet(0, new osg::DrawArrays(osg::DrawArrays::POINTS, 0, (int)points.size()));
    m_cloudGeom->setInitialBound(osg::BoundingBox(
        osg::Vec3(-100, -100, -100), osg::Vec3(100, 100, 100)));

    // 自动相机
    autoFitCamera();
}

// ============================================================================
// LeadScan 移植：加载网格文件（STL/OBJ，带光照材质）
// ============================================================================
bool OSGWidget::loadMesh(const QString& filepath)
{
    QByteArray ba = filepath.toUtf8();
    const char* cpath = ba.constData();

    FILE* lf = fopen("E:/workfold/framework/build/mesh_debug.log", "a");
    if (lf) fprintf(lf, "loadMesh: %s\n", cpath);

    // 用 file_io 解析
    Scanner::data::fileio::MeshData mesh;
    std::string spath(cpath);
    if (!Scanner::data::fileio::importMesh(spath, mesh) || mesh.vertices.empty()) {
        if (lf) { fprintf(lf, "  importMesh failed\n"); fclose(lf); }
        return false;
    }

    if (lf) { fprintf(lf, "  verts=%zu indices=%zu\n", mesh.vertices.size(), mesh.indices.size()); fclose(lf); }

    // file_io 点类型为 cv::Point3f——就地转 osg::Vec3 供渲染
    std::vector<osg::Vec3> mv(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i)
        mv[i].set(mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z);
    std::vector<osg::Vec3> mn(mesh.normals.size());
    for (size_t i = 0; i < mesh.normals.size(); ++i)
        mn[i].set(mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z);

    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> norms = new osg::Vec3Array;
    verts->reserve(mv.size());
    norms->reserve(mv.size());

    if (!mesh.indices.empty()) {
        // 平面法线：用 STL 原始面法线（参照 K2），无平滑
        bool hasNormals = (mn.size() == mv.size());
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            const auto& p0 = mv[mesh.indices[i]];
            const auto& p1 = mv[mesh.indices[i+1]];
            const auto& p2 = mv[mesh.indices[i+2]];
            verts->push_back(p0); verts->push_back(p1); verts->push_back(p2);
            if (hasNormals) {
                norms->push_back(mn[mesh.indices[i]]);
                norms->push_back(mn[mesh.indices[i+1]]);
                norms->push_back(mn[mesh.indices[i+2]]);
            } else {
                osg::Vec3 nm = (p1 - p0) ^ (p2 - p0); nm.normalize();
                norms->push_back(nm); norms->push_back(nm); norms->push_back(nm);
            }
        }
    } else {
        for (const auto& p : mv) verts->push_back(p);
    }

    if (verts->empty()) return false;

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry;
    geom->setUseVertexBufferObjects(true);
    geom->setVertexArray(verts);
    geom->setNormalArray(norms, osg::Array::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLES, 0, (int)verts->size()));

    // 光照材质（参照 LEADSCAN K2 loadModel：GL_LIGHT1 + 双面光照 + LeadScan 蓝材质）
    osg::ref_ptr<osg::StateSet> ss = geom->getOrCreateStateSet();
    ss->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
    ss->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    ss->setMode(GL_LIGHT1, osg::StateAttribute::ON);
    osg::ref_ptr<osg::LightModel> lm = new osg::LightModel;
    lm->setTwoSided(true);
    ss->setAttributeAndModes(lm);
    osg::ref_ptr<osg::Material> mat = new osg::Material;
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.529f, 0.808f, 0.980f, 1.0f));
    mat->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.529f, 0.808f, 0.980f, 1.0f));
    mat->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.3f, 0.3f, 0.3f, 1.0f));
    mat->setShininess(osg::Material::FRONT_AND_BACK, 100.0f);
    ss->setAttributeAndModes(mat);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable(geom);
    osgUtil::Optimizer optimizer;
    optimizer.optimize(geom);
    optimizer.reset();
    m_root->addChild(geode);

    // 相机定位并锁定视图
    osg::BoundingSphere bs = m_root->getBound();
    if (bs.valid() && bs.radius() > 0 && m_viewer.valid()) {
        double r = bs.radius();
        osg::Vec3d c(bs.center());
        osg::Vec3d eye(c.x(), c.y() - r * 3.0, c.z() + r * 0.5);
        osg::Vec3d up(0, 0, 1);

        double zNear = r * 0.1;
        double zFar = r * 100.0;
        const osg::GraphicsContext::Traits* traits = m_gw->getTraits();
        double aspect = static_cast<double>(traits->width) / static_cast<double>(traits->height);
        m_viewer->getCamera()->setProjectionMatrixAsPerspective(30.0, aspect, zNear, zFar);
        m_userProjection = m_viewer->getCamera()->getProjectionMatrix();

        m_viewLocked = false;
        osg::ref_ptr<osgGA::TrackballManipulator> manip = new osgGA::TrackballManipulator;
        manip->setAllowThrow(false);
        manip->setHomePosition(eye, c, up, false);
        m_viewer->setCameraManipulator(manip);
        manip->home(0);
    }

    return true;
}

// ============================================================================
// LeadScan 移植：加载标志点
// ============================================================================
void OSGWidget::loadMarkerPoints(const std::vector<osg::Vec3>& markers,
                                 const osg::Vec4& color)
{
    (void)color;   // 颜色参数退役：同心圆贴图样式（用户口径 2026-08-31）
    if (markers.empty()) return;

    if (!m_markerRoot || m_markerRoot->getNumParents() == 0) {
        // clearScene() 会 removeChildren 但不置空延迟指针——重建判定按"是否
        // 仍在场景树"（getNumParents==0=已被 clear，须重建重挂），否则二次
        // 扫描标志点永不显示（指针非空跳过创建，几何悬空在场景外）
        m_markerRoot = new osg::Group;
        m_markerGeode = new osg::Geode;
        m_markerGeom = new osg::Geometry;
        m_markerGeom->setUseVertexBufferObjects(true);
        m_markerGeom->setUseDisplayList(false);
        m_markerGeom->setDataVariance(osg::Object::DYNAMIC);
        m_markerGeode->addDrawable(m_markerGeom);
        m_markerRoot->addChild(m_markerGeode);
        m_root->addChild(m_markerRoot);

        // 同心圆样式（POINTS+点精灵路径——billboard 管线实测不可见弃用）：
        // 纹理 64×64 外黑环+内白圆（内6外10 比例），点尺寸 24px
        const int S = 64;
        QImage q(S, S, QImage::Format_RGBA8888);
        q.fill(Qt::transparent);
        {
            QPainter p(&q);
            p.setRenderHint(QPainter::Antialiasing);
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(15, 15, 15, 255));
            p.drawEllipse(2, 2, S - 4, S - 4);
            const int inner = S * 36 / 60;   // 内径 = 外径 × 0.6（6/10）
            p.setBrush(QColor(250, 250, 250, 255));
            p.drawEllipse((S - inner) / 2, (S - inner) / 2, inner, inner);
        }
        m_markerTexture = new osg::Texture2D;
        osg::ref_ptr<osg::Image> img = new osg::Image;
        auto* px = new unsigned char[S * S * 4];
        std::memcpy(px, q.constBits(), static_cast<size_t>(S * S * 4));
        img->setImage(S, S, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, px, osg::Image::USE_NEW_DELETE);
        m_markerTexture->setImage(img.get());
        // 过滤器必须显式 LINEAR：默认 MIN_FILTER=LINEAR_MIPMAP_LINEAR 而 setImage
        // 不自动生成 mipmap——采样全透明=点不可见（本次"无显示"根因）
        m_markerTexture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
        m_markerTexture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);

        osg::StateSet* ss = m_markerRoot->getOrCreateStateSet();
        osg::ref_ptr<osg::Point> pointSize = new osg::Point;
        pointSize->setSize(24.0f);
        ss->setAttribute(pointSize);
        ss->setTextureAttributeAndModes(0, m_markerTexture.get(), osg::StateAttribute::ON);
        ss->setMode(GL_BLEND, osg::StateAttribute::ON);
        ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        // 点精灵（纹理贴到点光栅上）
        osg::ref_ptr<osg::PointSprite> sprite = new osg::PointSprite;
        ss->setAttributeAndModes(sprite.get(), osg::StateAttribute::ON);
        ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

        m_markerCoords = new osg::Vec3Array;
        m_markerColors = new osg::Vec4Array;
        m_leftCamViewApplied = false;    // 新会话首点——左相机视角重新设置
    }

    m_markerCoords->assign(markers.begin(), markers.end());
    m_markerColors->resize(markers.size(), osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));  // 白（贴图调制）
    m_markerGeom->setVertexArray(m_markerCoords);
    m_markerGeom->setColorArray(m_markerColors);
    m_markerGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    // 首次须 addPrimitiveSet：空图元列表上 setPrimitiveSet(0,...) 被 OSG 静默
    // 忽略（i<size 检查）——标志点从未渲染的根因（2026-08-31）
    if (m_markerGeom->getNumPrimitiveSets() == 0)
        m_markerGeom->addPrimitiveSet(new osg::DrawArrays(osg::DrawArrays::POINTS, 0, (int)markers.size()));
    else
        m_markerGeom->setPrimitiveSet(0, new osg::DrawArrays(osg::DrawArrays::POINTS, 0, (int)markers.size()));

    // 扫描视图跟框（生长感知+节流+交互避让；取景=左相机视角）——每次标志点
    // 到达都检查：半径带外（±20%）重设视角。此前"首点一次制"在场景含残留
    // 大物体时取景过远，点过小"看不见"，须导入触发 fit 才可见（2026-08-31）
    maybeAutoFrame();
}

// ============================================================================
// 左相机视角——重建坐标系即左相机系（原点=左相机光心，+z 朝场景，y 向下）。
// eye 置于点云中心正后方（沿 -z），看向中心，up=(0,-1,0)——画面方向与左
// 相机原图一致（用户口径 2026-08-31）
// ============================================================================
void OSGWidget::loadLaserPoints(const std::vector<osg::Vec3>& laser)
{
    if (laser.empty()) return;
    if (!m_laserRoot || m_laserRoot->getNumParents() == 0) {
        // 重建判定同标志点（clearScene 摘除后指针不置空的坑）
        m_laserRoot = new osg::Group;
        m_laserGeode = new osg::Geode;
        m_laserGeom = new osg::Geometry;
        m_laserGeom->setUseVertexBufferObjects(true);
        m_laserGeom->setUseDisplayList(false);
        m_laserGeom->setDataVariance(osg::Object::DYNAMIC);
        m_laserGeode->addDrawable(m_laserGeom);
        m_laserRoot->addChild(m_laserGeode);
        m_root->addChild(m_laserRoot);

        osg::ref_ptr<osg::Point> pointSize = new osg::Point;
        pointSize->setSize(2.0f);
        m_laserRoot->getOrCreateStateSet()->setAttribute(pointSize);
        m_laserRoot->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        // 颜色=导入点云默认色（LeadScan 浅蓝 0.529/0.808/0.980——用户口径同源）
        auto* lc = new osg::Vec4Array(1);
        (*lc)[0] = osg::Vec4(0.529f, 0.808f, 0.980f, 1.0f);
        m_laserGeom->setColorArray(lc, osg::Array::BIND_OVERALL);
        m_laserCoords = new osg::Vec3Array;
    }
    m_laserCoords->assign(laser.begin(), laser.end());
    m_laserGeom->setVertexArray(m_laserCoords);
    if (m_laserGeom->getNumPrimitiveSets() == 0)
        m_laserGeom->addPrimitiveSet(new osg::DrawArrays(osg::DrawArrays::POINTS, 0, (int)laser.size()));
    else
        m_laserGeom->setPrimitiveSet(0, new osg::DrawArrays(osg::DrawArrays::POINTS, 0, (int)laser.size()));
}

// ============================================================================
// 左相机视角（实现）——eye 沿 -z 看向点云中心、up=-y（OpenCV 系 y 向下）
// ============================================================================
void OSGWidget::setLeftCameraView()
{
    if (!m_viewer || !m_root) return;
    // 视角基准=标志点自身包围球（与 maybeAutoFrame 同源——全场景 bound 会被
    // 残留点云污染）
    const osg::BoundingSphere bs =
        (m_markerRoot && m_markerRoot->getNumParents() > 0)
            ? m_markerRoot->getBound()
            : m_root->getBound();
    if (!bs.valid() || bs.radius() <= 0) return;

    const osg::Vec3d c(bs.center());
    const double dist = bs.radius() * 1.2;   // 取景距离 2.0r→1.2r（点占屏更大）
    const osg::Vec3d eye(c.x(), c.y(), c.z() - dist);
    const osg::Vec3d up(0.0, -1.0, 0.0);

    const double r = bs.radius();
    const osg::GraphicsContext::Traits* traits = m_gw->getTraits();
    const double aspect = static_cast<double>(traits->width) / static_cast<double>(traits->height);
    m_viewer->getCamera()->setProjectionMatrixAsPerspective(30.0, aspect, r * 0.01, r * 100.0);
    m_userProjection = m_viewer->getCamera()->getProjectionMatrix();

    osg::ref_ptr<osgGA::TrackballManipulator> manip = new osgGA::TrackballManipulator;
    manip->setAllowThrow(false);
    manip->setHomePosition(eye, c, up, false);
    m_viewer->setCameraManipulator(manip);
    manip->home(0);
}

// ============================================================================
// 扫描视角跟随——marker 云生长感知自动取景（P2 配套；UI 线程）
//   触发：首次有效更新，或场景半径较上次取景增长 >25%（新区域显著出画）
//   避让：用户拖拽中不抢视角；1s 节流防镜头抖动
// ============================================================================
void OSGWidget::maybeAutoFrame()
{
    if (!m_autoViewFit || m_userInteracting) return;
    const auto now = std::chrono::steady_clock::now();
    if (m_lastFitRadius >= 0.0 && now - m_lastFitTime < std::chrono::seconds(1)) return;

    // 取景基准=标志点自身包围球（非全场景——残留点云 geode 会污染半径/中心，
    // 视角过远点过小"看不见"，须导入触发 fit 才可见的根因 2026-09-01）
    const osg::BoundingSphere bs =
        (m_markerRoot && m_markerRoot->getNumParents() > 0)
            ? m_markerRoot->getBound()
            : osg::BoundingSphere();
    if (!bs.valid() || bs.radius() <= 0.0) return;
    // 已取过景且未显著变化（0.8×~1.25× 带内）——视角不动（防每 5 帧一跳）；
    // 带外（显著增大或缩小）都重取：缩小覆盖"停止扫描/二次会话"场景——
    // 此前只防增大，停止后一次跳走的视角永不回来，点云"消失"假象
    if (m_lastFitRadius >= 0.0 &&
        bs.radius() < m_lastFitRadius * 1.25 &&
        bs.radius() > m_lastFitRadius * 0.8) return;

    // 扫描标志点路径：取景用左相机视角（用户口径）——通用 fitCameraToRoot
    // 斜视角仅导入路径用
    setLeftCameraView();
    m_lastFitRadius = bs.radius();
    m_lastFitTime = now;
}

// ============================================================================
// LeadScan 移植：自动相机定位
// ============================================================================
void OSGWidget::autoFitCamera()
{
    if (!m_viewer || !m_root) return;

    osg::BoundingSphere bs = m_root->getBound();
    if (!bs.valid() || bs.radius() <= 0) return;

    // LeadScan 风格定位
    double radius = bs.radius();
    double viewDistance = radius;  // LeadScan 用 radius，不是 2.5*radius
    osg::Vec3d up(0.0, 0.0, 1.0);
    osg::Vec3d viewDirection(0.0, -1.0, 0.5);
    viewDirection.normalize();
    osg::Vec3d center = bs.center();
    center.y() -= radius;  // LeadScan 风格 Y 偏移
    osg::Vec3d eye = center + viewDirection * viewDistance;

    // 更新投影 near/far
    double zNear = radius * 0.01;
    double zFar = radius * 100.0;
    if (zNear < 0.001) zNear = 0.001;

    const osg::GraphicsContext::Traits* traits = m_gw->getTraits();
    double aspect = static_cast<double>(traits->width) / static_cast<double>(traits->height);
    m_viewer->getCamera()->setProjectionMatrixAsPerspective(30.0, aspect, zNear, zFar);
    m_userProjection = m_viewer->getCamera()->getProjectionMatrix();

    // 设置 manipulator
    osgGA::CameraManipulator* manip = m_viewer->getCameraManipulator();
    if (manip) {
        manip->setNode(m_root.get());
        manip->setHomePosition(eye, center, up);
        manip->home(0.0);
    }

    // 直接设置相机
    m_viewer->getCamera()->setViewMatrixAsLookAt(eye, center, up);
    m_viewer->frame();
    update();
}

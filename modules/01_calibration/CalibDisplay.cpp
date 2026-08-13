// CalibDisplay.cpp
// 标定显示场景构建：标定板 + 标志点 + STL 姿态模型 + 偏差 HUD
// 移植自 calib_display_demo.cpp（去掉 main/相机/坐标轴，坐标轴复用 OSGWidget 现有 overlay）
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include "CalibDisplay.h"

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/PrimitiveSet>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/LineWidth>
#include <osg/MatrixTransform>
#include <osg/Camera>
#include <osg/BoundingBox>
#include <osg/BoundingSphere>
#include <osg/CopyOp>
#include <osg/LightModel>
#include <osgText/Text>
#include <osgDB/ReadFile>
#include <osgDB/Registry>

#include <QPainter>
#include <QPainterPath>

#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

namespace calib_display {

static const float kBoardW = 180.0f;
static const float kBoardH = 120.0f;
static const float kGridSpacing = 30.0f;
static const float kMarkerRadius = 3.0f;
static const int   kMarkerCols = 7;
static const int   kMarkerRows = 6;

// 标定板白色底板
static osg::Geometry* createBoardPlane()
{
    float hw = kBoardW * 0.5f;
    float hh = kBoardH * 0.5f;

    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array;
    v->push_back(osg::Vec3(-hw, -hh, 0));
    v->push_back(osg::Vec3(hw, -hh, 0));
    v->push_back(osg::Vec3(hw, hh, 0));
    v->push_back(osg::Vec3(-hw, hh, 0));

    osg::ref_ptr<osg::Vec3Array> n = new osg::Vec3Array;
    n->push_back(osg::Vec3(0, 0, 1));

    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array;
    c->push_back(osg::Vec4(0.95f, 0.95f, 0.95f, 1.0f));

    osg::ref_ptr<osg::Geometry> g = new osg::Geometry;
    g->setUseDisplayList(false);
    g->setVertexArray(v);
    g->setNormalArray(n);
    g->setNormalBinding(osg::Geometry::BIND_OVERALL);
    g->setColorArray(c);
    g->setColorBinding(osg::Geometry::BIND_OVERALL);
    g->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));
    return g.release();
}

// 网格线
static osg::Geometry* createGridLines()
{
    float hw = kBoardW * 0.5f;
    float hh = kBoardH * 0.5f;

    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array;
    for (float x = -hw; x <= hw + 0.01f; x += kGridSpacing) {
        v->push_back(osg::Vec3(x, -hh, 0.01f));
        v->push_back(osg::Vec3(x, hh, 0.01f));
    }
    for (float y = -hh; y <= hh + 0.01f; y += kGridSpacing) {
        v->push_back(osg::Vec3(-hw, y, 0.01f));
        v->push_back(osg::Vec3(hw, y, 0.01f));
    }

    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array;
    c->push_back(osg::Vec4(0.5f, 0.5f, 0.5f, 1.0f));

    osg::ref_ptr<osg::Geometry> g = new osg::Geometry;
    g->setUseDisplayList(false);
    g->setVertexArray(v);
    g->setColorArray(c);
    g->setColorBinding(osg::Geometry::BIND_OVERALL);
    g->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, v->size()));

    osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(1.0f);
    g->getOrCreateStateSet()->setAttribute(lw);
    g->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    return g.release();
}

// 标志点（圆点 + 编号），绿=已扫 红=未扫
static osg::Group* createMarkers()
{
    osg::ref_ptr<osg::Group> grp = new osg::Group;

    int totalMarkers = kMarkerCols * kMarkerRows;
    std::vector<osg::Vec3> positions;
    float xMargin = 15.0f;
    float yMargin = 15.0f;
    float xStep = (kBoardW - 2 * xMargin) / (kMarkerCols - 1);
    float yStep = (kBoardH - 2 * yMargin) / (kMarkerRows - 1);
    for (int row = 0; row < kMarkerRows; ++row) {
        for (int col = 0; col < kMarkerCols; ++col) {
            float x = -kBoardW * 0.5f + xMargin + col * xStep;
            float y = -kBoardH * 0.5f + yMargin + row * yStep;
            positions.push_back(osg::Vec3(x, y, 0.1f));
        }
    }

    std::vector<bool> scanned(totalMarkers, false);
    for (int i = 0; i < 15 && i < totalMarkers; ++i) scanned[i] = true;

    const int segs = 32;
    for (size_t i = 0; i < positions.size(); ++i) {
        float cx = positions[i].x();
        float cy = positions[i].y();
        float cz = positions[i].z();

        osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array;
        for (int s = 0; s < segs; ++s) {
            float a = (float)s / segs * 2.0f * 3.14159265f;
            v->push_back(osg::Vec3(cx + kMarkerRadius * cosf(a),
                                   cy + kMarkerRadius * sinf(a), cz));
        }
        osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array;
        c->push_back(scanned[i] ? osg::Vec4(0, 1, 0, 1) : osg::Vec4(1, 0.3f, 0.3f, 1));

        osg::ref_ptr<osg::Geometry> circle = new osg::Geometry;
        circle->setUseDisplayList(false);
        circle->setVertexArray(v);
        circle->setColorArray(c);
        circle->setColorBinding(osg::Geometry::BIND_OVERALL);
        circle->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, segs));
        circle->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

        osg::ref_ptr<osg::Geode> geode = new osg::Geode;
        geode->addDrawable(circle);

        osg::ref_ptr<osgText::Text> label = new osgText::Text;
        label->setText(std::to_string(i));
        label->setPosition(osg::Vec3(cx, cy, cz + 0.1f));
        label->setCharacterSize(12.0f);
        label->setAlignment(osgText::Text::CENTER_CENTER);
        label->setColor(osg::Vec4(1, 1, 1, 1));
        label->setAxisAlignment(osgText::Text::SCREEN);
        geode->addDrawable(label);

        grp->addChild(geode);
    }
    return grp.release();
}

// 递归计算节点包围盒
static osg::BoundingBox computeBoundingBox(osg::Node* node)
{
    osg::BoundingBox bb;
    if (auto* geode = dynamic_cast<osg::Geode*>(node)) {
        for (unsigned i = 0; i < geode->getNumDrawables(); ++i)
            bb.expandBy(geode->getDrawable(i)->getBoundingBox());
    }
    if (auto* group = dynamic_cast<osg::Group*>(node)) {
        for (unsigned i = 0; i < group->getNumChildren(); ++i)
            bb.expandBy(computeBoundingBox(group->getChild(i)));
    }
    return bb;
}

// 递归给节点所有 Geometry 设置统一颜色（关光照下也能显示）
static void applyColorRecursive(osg::Node* node, const osg::Vec4& color)
{
    if (auto* geode = dynamic_cast<osg::Geode*>(node)) {
        osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array;
        c->push_back(color);
        for (unsigned i = 0; i < geode->getNumDrawables(); ++i) {
            osg::Geometry* geom = geode->getDrawable(i)->asGeometry();
            if (geom) {
                geom->setColorArray(c);
                geom->setColorBinding(osg::Geometry::BIND_OVERALL);
            }
        }
    }
    if (auto* group = dynamic_cast<osg::Group*>(node)) {
        for (unsigned i = 0; i < group->getNumChildren(); ++i)
            applyColorRecursive(group->getChild(i), color);
    }
}

// 手动解析 binary STL（绕过 osgDB 插件加载问题）
static osg::Geode* loadStlManual(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return nullptr;

    char header[80];
    if (fread(header, 1, 80, f) != 80) { fclose(f); return nullptr; }

    unsigned int numTris = 0;
    if (fread(&numTris, 4, 1, f) != 1) { fclose(f); return nullptr; }

    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> norms = new osg::Vec3Array;
    verts->reserve(numTris * 3);
    norms->reserve(numTris * 3);

    for (unsigned int i = 0; i < numTris; ++i) {
        float n[3], v[9];
        unsigned short attr;
        if (fread(n, 4, 3, f) != 3) break;
        if (fread(v, 4, 9, f) != 9) break;
        if (fread(&attr, 2, 1, f) != 1) break;

        verts->push_back(osg::Vec3(v[0], v[1], v[2]));
        verts->push_back(osg::Vec3(v[6], v[7], v[8]));  // p2 在前（交换 p1/p2 翻转绕序 CW→CCW）
        verts->push_back(osg::Vec3(v[3], v[4], v[5]));
        osg::Vec3 p0(v[0],v[1],v[2]), p1(v[3],v[4],v[5]), p2(v[6],v[7],v[8]);
        osg::Vec3 nm = (p2 - p0) ^ (p1 - p0);   // 与新绕序一致的叉积法线
        nm.normalize();
        for (int k = 0; k < 3; ++k) norms->push_back(nm);
    }
    fclose(f);

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry;
    geom->setUseDisplayList(false);
    geom->setVertexArray(verts);
    geom->setNormalArray(norms);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, (int)verts->size()));

    // 第1项：LeadScan 式光照（叉积法线 + GL_LIGHTING + GL_LIGHT0 + twoSided + Material）
    osg::ref_ptr<osg::StateSet> ss = geom->getOrCreateStateSet();
    ss->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    ss->setMode(GL_LIGHT0, osg::StateAttribute::ON);
    osg::ref_ptr<osg::LightModel> lightModel = new osg::LightModel;
    lightModel->setTwoSided(true);  // 双面光照：翻转绕序后内表面法线朝里，需要twoSided才亮
    ss->setAttributeAndModes(lightModel.get());
    osg::ref_ptr<osg::Material> mat = new osg::Material;
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.75f, 0.75f, 0.75f, 1.0f));
    mat->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.75f, 0.75f, 0.75f, 1.0f));
    mat->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.3f, 0.3f, 0.3f, 1.0f));
    mat->setShininess(osg::Material::FRONT_AND_BACK, 100.0f);
    ss->setAttributeAndModes(mat.get(), osg::StateAttribute::ON);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable(geom);
    printf("  loadStlManual: %u 三角面, %zu 顶点\n", numTris, verts->size());
    return geode.release();
}

// STL 模型（使用 STL 原始坐标，不做居中/缩放/平移，颜色数组着色）
static osg::MatrixTransform* createPoseModel(const std::string& stlPath,
                                             float r, float g, float b)
{
    osg::ref_ptr<osg::Node> geo = loadStlManual(stlPath);
    {
        FILE* f = fopen("E:/workfold/20260509intergrate/calib_debug.log", "a");
        if (f) { fprintf(f, "createPoseModel stl=%s geo=%p\n", stlPath.c_str(), (void*)geo.get()); fclose(f); }
    }
    if (!geo) {
        printf("  STL 加载失败: %s\n", stlPath.c_str());
        return nullptr;
    }

    osg::BoundingBox bb = computeBoundingBox(geo);
    printf("  STL 包围盒: X[%.1f, %.1f] Y[%.1f, %.1f] Z[%.1f, %.1f]\n",
           bb.xMin(), bb.xMax(), bb.yMin(), bb.yMax(), bb.zMin(), bb.zMax());

    auto* xform = new osg::MatrixTransform;
    xform->setMatrix(osg::Matrix::identity());

    auto* cloned = dynamic_cast<osg::Node*>(geo->clone(osg::CopyOp::DEEP_COPY_NODES));
    osg::Node* child = cloned ? cloned : geo.get();
    applyColorRecursive(child, osg::Vec4(r, g, b, 1.0f));
    xform->addChild(child);

    osg::StateSet* ss = xform->getOrCreateStateSet();
    ss->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);     // 不剔除，靠深度缓冲遮挡
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);     // 显式开启深度测试
    ss->setMode(GL_BLEND, osg::StateAttribute::OFF);        // 不透明
    ss->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    return xform;
}

// 偏差 HUD
static osg::Camera* createHud(int w, int h)
{
    auto* hud = new osg::Camera;
    hud->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    hud->setViewMatrix(osg::Matrix::identity());
    hud->setProjectionMatrix(osg::Matrix::ortho2D(0, w, 0, h));
    hud->setClearMask(GL_DEPTH_BUFFER_BIT);
    hud->setRenderOrder(osg::Camera::POST_RENDER, 10);

    auto* geode = new osg::Geode;
    hud->addChild(geode);

    osg::StateSet* ss = hud->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);

    struct Item { const char* label; float val; const char* unit; };
    Item items[] = {
        {"\xE5\x89\x8D\xE5\x90\x8E", 12.3f, "mm"},
        {"\xE5\xB7\xA6\xE5\x8F\xB3", -5.1f, "mm"},
        {"\xE8\xBF\x9C\xE8\xBF\x91",  8.7f, "mm"},
        {"\xE4\xBF\xAF\xE4\xBB\xB0",  2.1f, "\xC2\xB0"},
        {"\xE5\x81\x8F\xE8\x88\xAA", -1.3f, "\xC2\xB0"},
        {"\xE7\xBF\xBB\xE6\xBB\x9A",  0.5f, "\xC2\xB0"},
    };

    float panelX = (float)w - 260.0f;
    float panelH = 30.0f;
    float startY = (float)h - 40.0f;

    for (int i = 0; i < 6; ++i) {
        float y = startY - i * panelH;

        auto* label = new osgText::Text;
        label->setText(items[i].label);
        label->setPosition(osg::Vec3(panelX, y, 0));
        label->setCharacterSize(16.0f);
        label->setColor(osg::Vec4(1, 1, 1, 1));
        geode->addDrawable(label);

        char buf[32];
        snprintf(buf, sizeof(buf), "%+.1f%s", items[i].val, items[i].unit);
        auto* val = new osgText::Text;
        val->setText(buf);
        val->setPosition(osg::Vec3(panelX + 60, y, 0));
        val->setCharacterSize(16.0f);
        float level = fabsf(items[i].val) / (i < 3 ? 30.0f : 10.0f);
        level = std::min(level, 1.0f);
        val->setColor(osg::Vec4(level, 1 - level, 0, 1));
        geode->addDrawable(val);

        for (int j = 0; j < 5; ++j) {
            float segLevel = j / 4.0f;
            float alpha = (segLevel <= level) ? 1.0f : 0.2f;
            float bx = panelX + 150 + j * 18;
            float by = y - 2;

            osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array;
            v->push_back(osg::Vec3(bx, by, 0));
            v->push_back(osg::Vec3(bx + 16, by, 0));
            v->push_back(osg::Vec3(bx + 16, by + 14, 0));
            v->push_back(osg::Vec3(bx, by + 14, 0));

            osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array;
            c->push_back(osg::Vec4(segLevel, 1 - segLevel, 0, alpha));

            auto* seg = new osg::Geometry;
            seg->setUseDisplayList(false);
            seg->setVertexArray(v);
            seg->setColorArray(c);
            seg->setColorBinding(osg::Geometry::BIND_OVERALL);
            seg->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));
            geode->addDrawable(seg);
        }
    }
    return hud;
}

osg::Group* buildCalibScene(const std::string& stlPath)
{
    osg::ref_ptr<osg::Group> root = new osg::Group;

    // 仅扫描仪：3D 场景
    auto* scanGroup = new osg::MatrixTransform;
    scanGroup->setMatrix(osg::Matrix::translate(0.0f, 250.0f, 0.0f) *
                         osg::Matrix::rotate(osg::DegreesToRadians(90.0f), osg::Vec3(0, 0, 1)) *
                         osg::Matrix::scale(0.7f, 0.7f, 0.7f));
    {
        auto* target = createPoseModel(stlPath, 0, 1, 0);
        if (target)  scanGroup->addChild(target);
    }
    root->addChild(scanGroup);

    return root.release();
}

// ============================================================================
// CalibBoard2D — Qt 2D 标定板控件
// ============================================================================
CalibBoard2D::CalibBoard2D(QWidget* parent) : QWidget(parent)
{
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, Qt::white);
    setPalette(pal);
}

void CalibBoard2D::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int w = width();
    int h = height();

    // 白色底板
    p.fillRect(rect(), QColor(245, 245, 245));

    // 标定板区域（居中，留边距）
    int margin = 40;
    int boardX = margin;
    int boardY = margin;
    int boardW = w - 2 * margin;
    int boardH = h - 2 * margin;

    // 白色板面
    p.fillRect(boardX, boardY, boardW, boardH, Qt::white);

    // 网格线
    int cols = 7, rows = 6;
    p.setPen(QPen(QColor(200, 200, 200), 1));
    for (int i = 0; i <= cols; ++i) {
        int x = boardX + i * boardW / cols;
        p.drawLine(x, boardY, x, boardY + boardH);
    }
    for (int i = 0; i <= rows; ++i) {
        int y = boardY + i * boardH / rows;
        p.drawLine(boardX, y, boardX + boardW, y);
    }

    // 标志点（红色圆点 + 编号）
    int markerCols = 6, markerRows = 7;
    float xMargin = boardW * 0.08f;
    float yMargin = boardH * 0.08f;
    float xStep = (boardW - 2 * xMargin) / (markerCols - 1);
    float yStep = (boardH - 2 * yMargin) / (markerRows - 1);
    float radius = 8.0f;

    p.setFont(QFont("Arial", 7));
    int idx = 0;
    for (int c = 0; c < markerCols; ++c) {
        for (int r = 0; r < markerRows; ++r) {
            float cx = boardX + xMargin + c * xStep;
            float cy = boardY + yMargin + r * yStep;

            // 圆点
            p.setBrush(QBrush(QColor(200, 50, 50)));
            p.setPen(Qt::NoPen);
            p.drawEllipse(QPointF(cx, cy), radius, radius);

            // 编号
            p.setPen(QPen(Qt::white));
            p.drawText(QRectF(cx - 10, cy - 8, 20, 16), Qt::AlignCenter, QString::number(idx++));
        }
    }

    // 边框
    p.setPen(QPen(QColor(100, 100, 100), 2));
    p.setBrush(Qt::NoBrush);
    p.drawRect(boardX, boardY, boardW, boardH);
}

} // namespace calib_display

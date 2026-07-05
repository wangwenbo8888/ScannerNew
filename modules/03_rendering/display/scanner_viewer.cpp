#include "scanner_viewer.h"
#include "common/calib_logging.h"

#include <osgViewer/Viewer>
#include <osg/Group>
#include <osg/Geode>
#include <osg/StateSet>
#include <osg/Program>
#include <osg/Shader>
#include <osgGA/TrackballManipulator>

using namespace calib;


// ============================================================
// 内联 Shader 源码
// ============================================================

namespace {

// ---- 通用顶点着色器（激光 + 标记点共用）----
constexpr const char* kVertShader = R"(
#version 330 core

in vec3 a_position;
in vec2 a_uv;
in vec3 a_normal;

uniform mat4 osg_ModelViewProjectionMatrix;

out vec2 v_uv;
out vec3 v_normal;

void main() {
    v_uv     = a_uv;
    v_normal = a_normal;
    gl_Position = osg_ModelViewProjectionMatrix * vec4(a_position, 1.0);
}
)";

// ---- 激光片元着色器（实心方形面 + 正反面区分 + 法线光照）----
constexpr const char* kLaserFragShader = R"(
#version 330 core

in vec2 v_uv;
in vec3 v_normal;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(vec3(0.4, 0.6, 0.8));
    float diff = max(dot(N, L), 0.0);
    float ambient = 0.35;

    if (gl_FrontFacing) {
        // 正面：蓝灰色 + 法线光照
        vec3 baseColor = vec3(0.25, 0.55, 0.85);
        fragColor = vec4(baseColor * (ambient + diff * 0.65), 1.0);
    } else {
        // 背面：暗灰色（区分内外表面）
        vec3 backColor = vec3(0.35, 0.35, 0.38);
        fragColor = vec4(backColor * (ambient + diff * 0.3), 1.0);
    }
}
)";

// ---- 标记点片元着色器（白圆+黑环 SDF，正反面区分）----
constexpr const char* kMarkerFragShader = R"(
#version 330 core

in vec2 v_uv;    // [-1, 1]
in vec3 v_normal;

out vec4 fragColor;

void main() {
    float dist = length(v_uv);

    // 白圆半径 / 外环半径 = 6/10 = 0.6
    const float INNER_RATIO = 0.6;

    if (gl_FrontFacing) {
        // 正面：白圆 + 黑环
        if (dist <= INNER_RATIO) {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);   // 白色标记点
        } else if (dist <= 1.0) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);   // 黑色圆环
        } else {
            discard;                                  // 四边形角落透明
        }
    } else {
        // 背面：暗灰色圆（粘胶面效果）
        if (dist <= 1.0) {
            fragColor = vec4(0.3, 0.3, 0.3, 1.0);
        } else {
            discard;
        }
    }
}
)";

} // anonymous namespace

// ============================================================
// Impl
// ============================================================

struct ScannerViewer::Impl {
    std::unique_ptr<LaserCloudRenderer>  laserRenderer_;
    std::unique_ptr<MarkerCloudRenderer> markerRenderer_;

    osg::ref_ptr<osgViewer::Viewer> viewer_;
    osg::ref_ptr<osg::Group>       root_;
    osg::ref_ptr<osg::Geode>       laserGeode_;
    osg::ref_ptr<osg::Geode>       markerGeode_;
    osg::ref_ptr<osg::Program>     laserProgram_;
    osg::ref_ptr<osg::Program>     markerProgram_;
};

// ============================================================
// 公开接口
// ============================================================

ScannerViewer::ScannerViewer(size_t maxLaserPoints, size_t maxMarkers)
    : pImpl_(std::make_unique<Impl>())
{
    pImpl_->laserRenderer_  = std::make_unique<LaserCloudRenderer>(maxLaserPoints);
    pImpl_->markerRenderer_ = std::make_unique<MarkerCloudRenderer>(maxMarkers);
}

ScannerViewer::~ScannerViewer() = default;

void ScannerViewer::init(int width, int height)
{
    // 创建 Viewer
    pImpl_->viewer_ = new osgViewer::Viewer;
    pImpl_->viewer_->setUpViewInWindow(50, 50, width, height);
    pImpl_->viewer_->setThreadingModel(osgViewer::Viewer::SingleThreaded);

    // 场景图
    pImpl_->root_ = new osg::Group;

    pImpl_->laserGeode_ = new osg::Geode;
    pImpl_->laserGeode_->addDrawable(pImpl_->laserRenderer_->getDrawable());
    pImpl_->root_->addChild(pImpl_->laserGeode_);

    pImpl_->markerGeode_ = new osg::Geode;
    pImpl_->markerGeode_->addDrawable(pImpl_->markerRenderer_->getGeometry());
    // markerGeode 初始含 0 顶点的空 Geometry，不产生 draw call，天然不可见
    // flush() 填充顶点后自然显示，无需 NodeMask 控制
    pImpl_->root_->addChild(pImpl_->markerGeode_);

    // Shader 程序 — 激光
    pImpl_->laserProgram_ = new osg::Program;
    pImpl_->laserProgram_->addShader(new osg::Shader(osg::Shader::VERTEX, kVertShader));
    pImpl_->laserProgram_->addShader(new osg::Shader(osg::Shader::FRAGMENT, kLaserFragShader));
    pImpl_->laserProgram_->addBindAttribLocation("a_position", 0);
    pImpl_->laserProgram_->addBindAttribLocation("a_uv", 1);
    pImpl_->laserProgram_->addBindAttribLocation("a_normal", 2);
    pImpl_->laserGeode_->getOrCreateStateSet()->setAttribute(pImpl_->laserProgram_);

    // Shader 程序 — 标记点
    pImpl_->markerProgram_ = new osg::Program;
    pImpl_->markerProgram_->addShader(new osg::Shader(osg::Shader::VERTEX, kVertShader));
    pImpl_->markerProgram_->addShader(new osg::Shader(osg::Shader::FRAGMENT, kMarkerFragShader));
    pImpl_->markerProgram_->addBindAttribLocation("a_position", 0);
    pImpl_->markerProgram_->addBindAttribLocation("a_uv", 1);
    pImpl_->markerProgram_->addBindAttribLocation("a_normal", 2);
    pImpl_->markerGeode_->getOrCreateStateSet()->setAttribute(pImpl_->markerProgram_);

    // 深度测试 + 背景色
    pImpl_->root_->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
    pImpl_->viewer_->getCamera()->setClearColor(osg::Vec4(0.15f, 0.15f, 0.18f, 1.0f));

    pImpl_->viewer_->setSceneData(pImpl_->root_);

    // 相机交互 + 自动定位到场景
    pImpl_->viewer_->setCameraManipulator(new osgGA::TrackballManipulator);

    // 首帧 realize（创建 GL context + 编译 VBO）
    pImpl_->viewer_->realize();

    // 渲染一帧让 OSG 创建/编译所有 GL 对象 + 计算包围盒
    pImpl_->viewer_->frame();
    pImpl_->viewer_->getCameraManipulator()->home(0.0);

    CALIB_LOG_INFO("{}", "ScannerViewer: 初始化完成");
}

bool ScannerViewer::frame()
{
    if (pImpl_->viewer_->done()) return false;
    pImpl_->viewer_->frame();
    return true;
}

bool ScannerViewer::done() const
{
    return pImpl_->viewer_ ? pImpl_->viewer_->done() : true;
}

LaserCloudRenderer* ScannerViewer::laserRenderer()
{
    return pImpl_->laserRenderer_.get();
}

MarkerCloudRenderer* ScannerViewer::markerRenderer()
{
    return pImpl_->markerRenderer_.get();
}

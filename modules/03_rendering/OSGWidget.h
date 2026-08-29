#pragma once

#include <QOpenGLWidget>
#include <QTimer>
#include <QLabel>

#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osg/Group>
#include <osgGA/TrackballManipulator>

#include <osg/MatrixTransform>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Camera>

#include <chrono>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <cstdint>

namespace Scanner::data { class PointCloudBuffer; }

class OSGWidget : public QOpenGLWidget
{
    Q_OBJECT
public:
    explicit OSGWidget(QWidget *parent = nullptr);
    ~OSGWidget();

    void setSceneData(osg::Node *node);
    void setCenterOverlayVisible(bool visible);
    osgViewer::Viewer* viewer() { return m_viewer.get(); }
    osg::Group* root() { return m_root.get(); }

    void loadPointCloud(const std::vector<osg::Vec3>& points);
    void loadPointCloud(const std::vector<osg::Vec3>& points,
                        const std::vector<osg::Vec4ub>& colors);

    // LeadScan 移植：带法线的动态点云（VBO + DYNAMIC）
    void loadPointCloudWithNormals(const std::vector<osg::Vec3>& points,
                                  const std::vector<osg::Vec3>& normals,
                                  const osg::Vec4& color = osg::Vec4(0.0f, 0.604f, 0.804f, 1.0f));

    // LeadScan 移植：加载网格文件（STL/OBJ，带光照材质）
    bool loadMesh(const QString& filepath);

    // LeadScan 移植：加载标志点
    void loadMarkerPoints(const std::vector<osg::Vec3>& markers,
                          const osg::Vec4& color = osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    // LeadScan 移植：自动相机定位到场景包围球
    void autoFitCamera();
    bool loadTestData(int numPoints);
    bool loadTestDataFromPLY(const std::string& filepath, int numPoints);

    // 从 PointCloudBuffer 直读快照（ADR7.8 只读窄接口）
    void loadFromPointCloudBuffer(class Scanner::data::PointCloudBuffer* pcb);
    void clearScene();

    // —— P1 渲染加固（docs/plans/2026-08-25-渲染模块加固计划.md）——
    // 轻量事件出口：03 不直接依赖 base/EventBus，app 装配桥接（码表 RenderSanity.h RenderEvent）
    using RenderFaultSink = std::function<void(int code, const std::string& msg)>;
    void setFaultSink(RenderFaultSink sink) { m_faultSink = std::move(sink); }
    // 摄入预算：点数上限（超限均匀抽稀＋0x0312 上报）与包围盒半径上限（mm，越界拒绝）
    void setIngestBudget(size_t maxPoints, double maxExtentMm);
    bool isRenderSuspended() const { return m_renderSuspended; }
    bool tryResumeRender();                  // 挂起后恢复（外部按钮/定时重试）
    uint64_t lastIngestedVersion() const { return m_lastIngestVersion; }

    // 渲染观测（P3）：只读快照——app 装配喂 HardwareMonitor（processFps 等）
    struct RenderStats {
        uint64_t framesDrawn = 0;    // 成功执行的 frame() 次数
        uint64_t ingestCount = 0;    // 已摄入的快照份数（过闸计入）
        double   lastFrameMs = 0.0;  // 最近一帧渲染耗时
        int      degradeLevel = 0;   // 自动降级级别（0=未降级；帧超时阶梯加一）
        bool     suspended = false;  // 渲染循环挂起标志
    };
    RenderStats renderStats() const;

    void setCameraManipulator(osgGA::CameraManipulator* manipulator);
    void home();

    // Lasso / polyline selection
    void enterLassoMode();
    void exitLassoMode();
    bool isLassoMode() const { return m_lassoMode; }
    void deleteSelectedPoints();
    void undoDelete();
    void enterLassoDeleteMode();

    // Debug: validate projection matrix math
    void debugValidateProjection();

    // Center overlay: two circles + text
    void createCenterOverlay();

private:
    void updateStreamCameraView();
    void createAxesIndicator();
    void updateAxesView();
    void adjustViewToMaxProjection();
    void addLassoPoint(float mx, float my);
    void closeLasso();
    void updateLassoGeometry();
    void deletePointsInPolyline();
    bool isPointInPolygon2D(const osg::Vec2d& point, const osg::Vec2Array& polygon);
    void highlightSelectedPoints();
    void clearHighlight();
    void pushDeleteUndo(osg::Geometry* geom, const std::vector<unsigned int>& indices,
                        const std::vector<osg::Vec4ub>& origColors);

    // —— P1 渲染加固实现件 ——
    void reportFault(int code, const std::string& msg);   // m_faultSink 出口（无 sink 仅 qWarning）
    void suspendRender(const std::string& why);           // 挂起渲染循环（保 UI 存活）
    // 构建点云 Geode（VBO+DYNAMIC；分配集中于此——异常可弃，场景未动）
    osg::ref_ptr<osg::Geode> buildCloudGeode(const std::vector<osg::Vec3>& points,
                                             const std::vector<osg::Vec4ub>* colors);
    void fitCameraToRoot();                               // 相机定位到场景包围球（原 loadPointCloud 尾块）


protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mouseMoveEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

signals:
    void streamProgress(int loaded, int total);
    void lassoCompleted();

private slots:
    void streamNextBatch();

private:
    osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_gw;
    osg::ref_ptr<osgViewer::Viewer> m_viewer;
    osg::ref_ptr<osg::Group> m_root;
    QTimer *m_timer;
    QTimer *m_streamTimer;

    float m_lastMouseX;
    float m_lastMouseY;
    bool m_firstMouse;

    std::ifstream m_streamFile;
    std::vector<char> m_streamFileBuf;   // 流文件 1MB 缓冲（pubsetbuf——须存活于读全程）
    std::string m_streamPath;
    std::chrono::steady_clock::time_point m_streamStart;   // 加载耗时打点
    std::vector<char> m_streamReadBuf;   // 批量整块读缓冲（批×15B，成员复用免每批分配）
    int m_streamTotal = 0;
    int m_streamLoaded = 0;
    int m_streamBatch = 500000;
    bool m_streamDone = false;

    osg::Vec3 m_streamBBoxMin;
    osg::Vec3 m_streamBBoxMax;
    bool m_streamBBoxValid = false;

    osg::ref_ptr<osg::Camera> m_axesCamera;
    osg::ref_ptr<osg::Group> m_axesGroup;

    // PCA accumulators for 200ms batch view adjustment
    osg::Vec3 m_pcaSum;
    double m_pcaSumXX, m_pcaSumXY, m_pcaSumXZ;
    double m_pcaSumYY, m_pcaSumYZ, m_pcaSumZZ;
    int m_pcaCount;
    int m_pcaTicks;

    // 2D screen lasso (selection mode)
    bool m_lassoMode = false;
    bool m_lassoDeleteMode = false;
    osg::ref_ptr<osg::Camera> m_lassoCamera;
    osg::ref_ptr<osg::Geometry> m_lassoGeom;
    osg::ref_ptr<osg::Vec3Array> m_lassoVerts;
    osg::ref_ptr<osg::DrawArrays> m_lassoDrawArrays;
    osg::ref_ptr<osg::Vec2Array> m_lassoPoints;
    std::vector<osg::ref_ptr<osg::Vec2Array>> m_selectedPolylines;



    // Highlight tracking
    struct HighlightEntry {
        osg::Geometry* geom;
        std::vector<unsigned int> indices;
        std::vector<osg::Vec4ub> originalColors;
    };
    std::vector<HighlightEntry> m_highlights;

    // Our own copy of the projection matrix — OSG internals can corrupt the camera's
    // projection matrix during frame(), so we restore from this copy before each frame.
    osg::Matrix m_userProjection;
    osg::Matrix m_userView;        // 锁定的视图矩阵（loadPointCloud 时设置）
    bool m_viewLocked = false;     // 是否锁定视图

    // Camera state captured right after frame() — the actual matrices used for rendering.
    osg::Matrix m_camViewAfterFrame;
    osg::Matrix m_camProjAfterFrame;
    int m_camViewportW = 0;
    int m_camViewportH = 0;



    // CullCallback that captures the projection*view matrix from OSG's CullVisitor (used for culling, not rendering).
    struct HitTestCullCallback : public osg::NodeCallback
    {
        osg::Matrix* capturedMVP;
        osg::Matrix* capturedProj;
        osg::Matrix* capturedView;
        HitTestCullCallback(osg::Matrix* mvp, osg::Matrix* proj, osg::Matrix* view)
            : capturedMVP(mvp), capturedProj(proj), capturedView(view) {}
        void operator()(osg::Node* node, osg::NodeVisitor* nv) override
        {
            osgUtil::CullVisitor* cv = dynamic_cast<osgUtil::CullVisitor*>(nv);
            if (cv)
            {
                osg::Matrix mv = *cv->getModelViewMatrix();
                osg::Matrix proj = *cv->getProjectionMatrix();
                if (capturedMVP)
                    *capturedMVP = proj * mv;
                if (capturedProj)
                    *capturedProj = proj;
                if (capturedView)
                    *capturedView = mv;
            }
            traverse(node, nv);
        }
    };
    osg::ref_ptr<HitTestCullCallback> m_hitTestCallback;
    osg::Matrix m_capturedMVP;

    // Compute the actual MVP that OpenGL uses for rendering.
    // OSG stores matrices in row-major layout. When uploaded to OpenGL via glLoadMatrix,
    // the _v array is read as column-major, so OpenGL effectively gets the transposed matrix.
    // OpenGL rendering: clip = P_GL * V_GL * v  where P_GL = transpose(P_osg), V_GL = transpose(V_osg)
    inline osg::Matrix computeRenderingMVP(const osg::Matrix& proj, const osg::Matrix& view) const
    {
        osg::Matrix pT, vT;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
            {
                pT(r, c) = proj(c, r);
                vT(r, c) = view(c, r);
            }
        return pT * vT;
    }

    // Camera state captured in closeLasso() BEFORE processEvents() — used ONLY for hit testing.
    // These are NOT overwritten by paintGL(), unlike m_camViewAfterFrame/m_camProjAfterFrame
    // which get overwritten when processEvents() triggers a paint event during closeLasso().
    osg::Matrix m_hitTestView;
    osg::Matrix m_hitTestProj;
    int m_hitTestVpw = 0;
    int m_hitTestVph = 0;

    // Undo history
    struct DeleteEntry {
        osg::Geometry* geom;
        std::vector<unsigned int> indices;
        std::vector<osg::Vec4ub> originalColors;
    };
    std::vector<DeleteEntry> m_deleteHistory;

    // Center overlay (two circles + text)
    osg::ref_ptr<osg::Camera> m_centerOverlayCamera;
    bool m_centerOverlayVisible = true;

    // LeadScan 移植：动态点云几何
    osg::ref_ptr<osg::Group> m_cloudRoot;       // 点云根节点
    osg::ref_ptr<osg::Group> m_markerRoot;      // 标志点根节点
    osg::ref_ptr<osg::Geode> m_cloudGeode;      // 点云叶节点
    osg::ref_ptr<osg::Geode> m_markerGeode;     // 标志点叶节点
    osg::ref_ptr<osg::Geometry> m_cloudGeom;    // 点云几何（DYNAMIC）
    osg::ref_ptr<osg::Geometry> m_markerGeom;   // 标志点几何
    osg::ref_ptr<osg::Vec3Array> m_cloudCoords;
    osg::ref_ptr<osg::Vec3Array> m_cloudNormals;
    osg::ref_ptr<osg::Vec4Array> m_cloudColors;
    osg::ref_ptr<osg::Vec3Array> m_markerCoords;
    osg::ref_ptr<osg::Vec4Array> m_markerColors;

    // 坐标轴
    osg::ref_ptr<osg::Geode> m_axesGeode;

    // —— P1 渲染加固状态（UI 线程属主）——
    RenderFaultSink m_faultSink;                    // 可空（app 装配期注入）
    uint64_t m_lastIngestVersion = 0;               // 版本短路记账（上次已摄入版本）
    size_t m_maxIngestPoints = (size_t{8} << 20);   // 摄入点数预算（默认 8M）
    double m_maxIngestExtentMm = 1.0e6;             // 包围盒半径上限（mm）
    bool m_renderSuspended = false;                 // 渲染循环挂起标志（paintGL 短路）

    // —— P2.2 帧耗时探针＋自动降级（UI 线程属主）——
    uint64_t m_framesDrawn = 0;                     // 成功 frame() 计数
    uint64_t m_ingestCount = 0;                     // 摄入计数
    double m_lastFrameMs = 0.0;                     // 最近帧耗时
    int m_overBudgetStreak = 0;                     // 连续超阈帧计数（5 触发降级）
    int m_degradeLevel = 0;                         // 降级级别（0..3——预算阶梯索引）
    static const size_t kDegradeBudgets[4];         // 8M/4M/1M/256K（cpp 定义）
};

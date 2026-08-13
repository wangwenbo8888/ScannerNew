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

#include <fstream>

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
    std::string m_streamPath;
    int m_streamTotal = 0;
    int m_streamLoaded = 0;
    int m_streamBatch = 50000;
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
};

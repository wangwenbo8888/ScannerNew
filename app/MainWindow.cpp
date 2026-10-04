#include "MainWindow.h"
#include "AppContext.h"
#include "SceneFeedAdapter.h"
#include "DeviceStateCache.h"
#include "PointCloudBuffer.h"
#include "CalibDialog.h"
#include "CalibDisplay.h"
#include "IntegrateTestDialog.h"
#include "ScannerWindow.h"
#include "stubs/LEADSCANSeries.h"
#include "stubs/CameraControl.h"
#include "stubs/camera_calib_workflow.h"  // calibration::CameraCalibWorkflow（相机标定联调口）
#include "stubs/laser_calib_workflow.h"   // calibration::LaserCalibInput（激光标定联调口）
#include "ScanWorkflow.h"       // 02（编辑物理化访问链 markerFuse/obsAccumulator——P4b）
#include "PostProcessWorkflow.h" // 04（后处理进度回调——P1-1 UI 入口）
#include "pipelines/scan/FuseConsumer.h"       // 07 IMarkerFuse（removePoints 契约）
#include "pipelines/scan/FrameObsAccumulator.h" // 07 obs（excludeMarkerObs 契约）
#include "file_io.h"
#include "PerfMonitor.h"   // A-T17：updateInfoSection 内 perfMonitor()->poll() 需完整类型
#include "base/EventBus.h" // P2 渲染事件桥：faultSink lambda 需完整类型（publish）
#include "IState.h"        // P1-2 UI 状态图标：SystemState 7 态完整枚举（switch/映射）
#include "StateMachine.h"  // P1-2: getCurrentState() 需完整类型
#include <spdlog/spdlog.h>
#include "jmw_logging.h"
#include "modules/08_devicemgmt/ParamStore.h"      // ParamEntry::Source
#include "modules/08_devicemgmt/DeviceManager.h"   // setParam 三路联动（完整类型）

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>
#include <QSignalBlocker>   // 面片预设：旋钮同步阻断耦合反灌（260912）
#include <opencv2/imgproc.hpp>   // 相机监视预览降采样（cv::resize INTER_AREA）
#include <osg/Vec3>
#include <osg/Matrix>
#include <osgGA/TrackballManipulator>
#include <QPainter>
#include <QPainterPath>
#include <QApplication>
#include <QHeaderView>
#include <QScreen>
#include <QResizeEvent>
#include <QTimer>
#include <QScrollArea>
#include <QButtonGroup>
#include <QCursor>
#include <QShortcut>
#include <QToolTip>
#include <QMessageBox>
#include <QMenu>
#include <QFileDialog>
#include <QStatusBar>
#include <QProgressDialog>
#include <QGridLayout>     // 虚拟按键表盘（261002 临时测试机）
#include <cstdio>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <pdh.h>
#pragma comment(lib, "pdh.lib")

#pragma execution_character_set("utf-8")

QPixmap MainWindow::renderSvg(const QString &svgPath, int size)
{
    return renderSvg(svgPath, size, size);
}

QPixmap MainWindow::renderSvg(const QString &svgPath, int w, int h)
{
    QSvgRenderer renderer(svgPath);
    qreal dpr = qApp->devicePixelRatio();
    QPixmap pix(QSize(int(w * dpr), int(h * dpr)));
    pix.fill(Qt::transparent);
    pix.setDevicePixelRatio(dpr);
    QPainter painter(&pix);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    renderer.render(&painter);
    return pix;
}

ArrowSlider::ArrowSlider(Qt::Orientation orientation, QWidget *parent)
    : QSlider(orientation, parent)
{
}

void ArrowSlider::setGroovePixmap(const QPixmap &pixmap)
{
    m_groovePixmap = pixmap;
    update();
}

void ArrowSlider::paintEvent(QPaintEvent *event)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    QStyleOptionSlider opt;
    initStyleOption(&opt);
    QRect handleRect = style()->subControlRect(QStyle::CC_Slider, &opt, QStyle::SC_SliderHandle, this);

    if (orientation() == Qt::Vertical) {
        // 竖排：渐变槽竖向，箭头指向左
        if (!m_groovePixmap.isNull()) {
            QPixmap scaled = m_groovePixmap.scaled(24, height() - 20, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
            int gx = (width() - scaled.width()) / 2;
            p.drawPixmap(gx, 10, scaled);
        }
        int cy = handleRect.center().y();
        int grooveX = (width() - 24) / 2;
        int cx = grooveX + 24;
        p.setBrush(QBrush(Qt::white));
        p.setPen(QPen(QColor(80, 80, 80), 1));
        QPolygon arrow;
        arrow << QPoint(cx, cy - 4)
              << QPoint(cx - 24, cy)
              << QPoint(cx, cy + 4);
        p.drawPolygon(arrow);
    } else {
        // 横排：渐变槽横向(旋转90度)，圆头，箭头指向下
        if (!m_groovePixmap.isNull()) {
            QTransform rotate;
            rotate.rotate(90);
            QPixmap rotated = m_groovePixmap.transformed(rotate, Qt::SmoothTransformation);
            int gw = width() - 20;
            int gh = 24;
            QPixmap scaled = rotated.scaled(gw, gh, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
            int gy = (height() - gh) / 2;
            int radius = gh / 2;
            QPainterPath clip;
            clip.addRoundedRect(QRectF(10, gy, gw, gh), radius, radius);
            p.setClipPath(clip);
            p.drawPixmap(10, gy, scaled);
            p.setClipping(false);
        }
        int cx = handleRect.center().x();
        int grooveY = (height() - 24) / 2;
        int cy = grooveY;
        p.setBrush(QBrush(Qt::white));
        p.setPen(QPen(QColor(80, 80, 80), 1));
        QPolygon arrow;
        arrow << QPoint(cx - 4, cy)
              << QPoint(cx, cy + 24)
              << QPoint(cx + 4, cy);
        p.drawPolygon(arrow);
    }
}

MainWindow::MainWindow(AppContext* appCtx, QWidget *parent) : QMainWindow(parent), m_appCtx(appCtx)
{
    setObjectName("mainWindow");
    setWindowTitle(QStringLiteral("LeadScan K2"));
    setWindowFlags(Qt::FramelessWindowHint);

    QWidget *central = new QWidget();
    central->setObjectName("centralWidget");
    QVBoxLayout *mainLayout = new QVBoxLayout(central);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    mainLayout->addWidget(createTitleBar());
    mainLayout->addWidget(createNavBar());
    mainLayout->addWidget(createToolBar());

    QHBoxLayout *contentLayout = new QHBoxLayout();
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(0);
    contentLayout->addWidget(createLeftPanel(), 2);
    contentLayout->addWidget(create3DViewArea(), 5);
    mainLayout->addLayout(contentLayout, 1);

    setCentralWidget(central);

    createFloatingToolbar();

    // —— P2 渲染加固接线 ——
    // ⓪ 扫描会话终止回调（后台装配失败——AppContext 后台线程调，queued 切 UI）：
    //    复原扫描按钮态＋状态栏提示（幂等；正常停止路径的复原在点击 lambda 内）
    if (m_appCtx) {
        m_appCtx->setScanSessionEndedHandler([this](bool ok) {
            QMetaObject::invokeMethod(this, [this, ok]() {
                if (!ok) {
                    if (m_activeScanToolIdx >= 0) setScanButtonVisual(m_activeScanToolIdx, false);
                    m_activeScanToolIdx = -1;
                    statusBar()->showMessage(QStringLiteral("扫描启动失败（后台装配）——详见日志"), 5000);
                }
                // 终局遍弹窗兜底关闭（正常完成路径由 100% 关；此为安全网）
                if (m_finalBADlg) { m_finalBADlg->close(); m_finalBADlg = nullptr; }
            }, Qt::QueuedConnection);
        });
        // 终局遍进度弹窗（260920：finish 后台 GBA 分钟级——用户须见"优化中"；
        // 回调在后台线程 → queued 切 UI 更新/关闭）
        // 261003 竞态修复：此处不创建弹窗——GBA 瞬时降级（激光缓存=0→3ms 返回）
        // 时 sessionEnded 先关弹窗置 null，迟到的进度回调到 UI 发现 null 会重建
        // 新弹窗卡在最后进度（用户实测卡 20% 根因）。弹窗仅在关会话成功时创建
        m_appCtx->setFinalBAProgress([this](int percent, const std::string& stage) {
            QMetaObject::invokeMethod(this, [this, percent, stage]() {
                if (!m_finalBADlg) return;   // 已关（sessionEnded 先到）——不重建
                m_finalBADlg->setValue(std::clamp(percent, 0, 100));
                m_finalBADlg->setLabelText(QString(QStringLiteral("全局优化中（%1%）——%2"))
                                               .arg(percent)
                                               .arg(QString::fromStdString(stage)));
                if (percent >= 100) {               // 完成（点云入库后）——延迟一拍关
                    QMetaObject::invokeMethod(this, [this]() {
                        if (m_finalBADlg) { m_finalBADlg->close(); m_finalBADlg = nullptr; }
                    }, Qt::QueuedConnection);
                }
            }, Qt::QueuedConnection);
        });
    }
    // ① 流水线→渲染：SceneFeedAdapter（queued）→ OSGWidget 标志点更新（UI 线程）
    if (m_appCtx && m_3dView) {
        if (auto* feed = m_appCtx->sceneFeed()) {
            connect(feed, &SceneFeedAdapter::markerCloudUpdated, this,
                    [this](const std::vector<cv::Point3f>& pts, const std::vector<cv::Vec3f>& normals) {
                        // 显示链终点观测（节流 1/30）：信号到 UI 的点数；同步导出
                        // PLY 快照（exe 目录 markers_snapshot.ply——数据正确性人工
                        // 核对用，覆盖写）
                        static std::atomic<uint64_t> s_uiRecv{0};
                        if (const auto k = s_uiRecv.fetch_add(1); k % 30 == 0) {
                            JMW_LOG_INFO("app-MainWindow", "[MainWindow] 标志点信号到达: {} 点（第 {} 次）",
                                         pts.size(), k);
                            if (QFile f(QStringLiteral("markers_snapshot.ply"));
                                f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                                QByteArray out;
                                out += "ply\nformat ascii 1.0\n";
                                out += QString("element vertex %1\n").arg(pts.size()).toUtf8();
                                out += "property float x\nproperty float y\nproperty float z\nend_header\n";
                                for (const auto& p : pts) {
                                    out += QString("%1 %2 %3\n")
                                              .arg(p.x, 0, 'f', 3)
                                              .arg(p.y, 0, 'f', 3)
                                              .arg(p.z, 0, 'f', 3)
                                              .toUtf8();
                                }
                                f.write(out);
                            }
                        }
                        // 工程树实时计数（节流 1/30 信号）：当前扫描会话节点
                        // 「标记点 00N (M)」＝重建标志点个数（融合云瞬时点数）；
                        // 历史会话节点保留各自最终值；点云数据由落库后 cloudTimer 更新
                        static std::atomic<uint64_t> s_treeTicks{0};
                        if (s_treeTicks.fetch_add(1) % 30 == 0) {
                            if (m_markerCurrentItem)
                                m_markerCurrentItem->setText(
                                    0, QStringLiteral("标记点 %1 (%2)")
                                              .arg(m_markerScanSeq, 3, 10, QChar('0'))
                                              .arg(pts.size()));
                        }
                        if (!m_3dView || pts.empty()) return;
                        // 排障插桩（260927 标志点不显示）：信号到达＋首点坐标量级留痕
                        {
                            static std::atomic<uint64_t> s_mkLog{0};
                            if (s_mkLog.fetch_add(1) % 60 == 0) {
                                JMW_LOG_INFO("app-MainWindow",
                                    "[标志点排障] markersUpdated：n={} 首点=({:.1f},{:.1f},{:.1f})"
                                    " 3D可见开关={}",
                                    pts.size(), pts[0].x, pts[0].y, pts[0].z,
                                    m_3dView ? m_3dView->markerPointsVisible() : false);
                            }
                        }
                        // —— 260912 导出补链：标志点快照入 PointCloudBuffer——
                        //    setMarkers 原零调用者（导出标志点恒空，同激光病）。
                        //    快照语义（整表替换）与融合云瞬时快照口径一致
                        if (auto* pcbMk = m_appCtx ? m_appCtx->pointCloudBuffer() : nullptr) {
                            std::vector<Scanner::data::MarkerRecord> recs;
                            recs.reserve(pts.size());
                            for (size_t i = 0; i < pts.size(); ++i) {
                                Scanner::data::MarkerRecord r;
                                r.globalId = static_cast<uint32_t>(i);
                                r.pos = pts[i];
                                if (normals.size() == pts.size())
                                    r.normal = normals[i];
                                recs.push_back(r);
                            }
                            pcbMk->setMarkers(recs);
                        }
                        std::vector<osg::Vec3> markers;
                        markers.reserve(pts.size());
                        for (const auto& p : pts) markers.emplace_back(p.x, p.y, p.z);
                        std::vector<osg::Vec3> norms;
                        if (normals.size() == pts.size()) {
                            norms.reserve(normals.size());
                            for (const auto& nn : normals)
                                norms.emplace_back(nn[0], nn[1], nn[2]);
                        }
                        // 定向圆盘（法线世界系固定朝向——2026-09-05 口径；
                        // 法线缺失时圆盘回退朝上）
                        m_3dView->loadMarkerPoints(markers, norms);
                    });
            // 激光点云（host 块直推——FuseConsumer 下载的当帧激光块）
            connect(feed, &SceneFeedAdapter::laserCloudUpdated, this,
                    [this](const std::vector<cv::Point3f>& pts) {
                // 显示/导出减负（260919：全域扫描后融合云至数百万点——全量 VBO
                // 重传＋数千万字节文本 PLY 每 30 拍写盘拖死 UI；>150 万点按步长
                // 抽样至 ~150 万（仓库/日志计数仍为全量真值）
                const size_t kDispMax = 1500000;
                const size_t stride = pts.size() > kDispMax ? (pts.size() + kDispMax - 1) / kDispMax : 1;
                // 节流导出（exe 目录 laser_snapshot.ply——人工导入核对用）
                static std::atomic<uint64_t> s_lRecv{0};
                if (const auto k = s_lRecv.fetch_add(1); k % 30 == 0 && !pts.empty()) {
                    if (QFile f(QStringLiteral("laser_snapshot.ply"));
                        f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                        QByteArray out;
                        out += "ply\nformat ascii 1.0\n";
                        out += QString("element vertex %1\n")
                                   .arg((pts.size() + stride - 1) / stride)
                                   .toUtf8();
                        out += "property float x\nproperty float y\nproperty float z\nend_header\n";
                        for (size_t i = 0; i < pts.size(); i += stride) {
                            const auto& p = pts[i];
                            out += QString("%1 %2 %3\n")
                                       .arg(p.x, 0, 'f', 3)
                                       .arg(p.y, 0, 'f', 3)
                                       .arg(p.z, 0, 'f', 3)
                                       .toUtf8();
                        }
                        f.write(out);
                    }
                }
                        if (!m_3dView) return;
                        // —— 仓库写入已退役（260919 正式写口上线）：07 FuseConsumer
                        //    融合线程经 ICloudWarehouse 直写 06 PointCloudBuffer
                        //    （会话两层：start 折叠基线＋节流推会话层替换）——
                        //    本 UI 处理器只余显示/导出，不再承担数据通道
                         // 3D 激光显示（累积渲染；>150 万点抽样——见头部减负注）
                         std::vector<osg::Vec3> laser;
                         laser.reserve((pts.size() + stride - 1) / stride);
                         for (size_t i = 0; i < pts.size(); i += stride)
                             laser.emplace_back(pts[i].x, pts[i].y, pts[i].z);
                         m_3dView->loadLaserPoints(laser);
                     });
            connect(feed, &SceneFeedAdapter::freezeChanged, this,
                    [this](bool frozen) {
                        // D 批处理期冻结：画面保持末帧（ingest 已由 adapter 丢弃）——状态条提示可后续挂
                        JMW_LOG_DEBUG("app-MainWindow", "[MainWindow] sceneFeed freeze={}", frozen);
                    });
        }
        // ② 03 渲染事件 → EventBus（P1.4 桥：码表见 RenderSanity.h RenderEvent；
        //    FaultOccurred 统一契约 param1=severity param2=码，sourceId=0x03；
        //    Degraded→Warning（对齐 07 EventBusEventSink 映射），文本走日志）
        if (auto* bus = m_appCtx->eventBus()) {
            m_3dView->setFaultSink([bus](int code, const std::string& msg) {
                JMW_LOG_WARN("app-MainWindow", "[render] 0x{:04X} {}", code, msg);
                Scanner::Event ev;
                ev.type = Scanner::EventType::FaultOccurred;
                ev.sourceId = 0x03;
                ev.param1 = static_cast<int64_t>(Scanner::FaultSeverity::Warning);
                ev.param2 = code;
                bus->publish(ev);
            });
        }
    }

    m_integrateTestDialog = nullptr;
    m_calibDialog = nullptr;

    // —— P1-2 UI 状态图标：状态栏右下角常驻态指示，订阅 10 状态机 StateChanged ——
    m_stateIndicator = new QLabel(QStringLiteral("● 初始化"), this);
    m_stateIndicator->setToolTip(QStringLiteral("系统状态机（10 可观测性）"));
    statusBar()->addPermanentWidget(m_stateIndicator);
    if (m_appCtx) {
        if (auto* sm = m_appCtx->stateMachine())
            updateStateIndicator(sm->getCurrentState());   // 初始态立即上屏（不等首发事件）
        if (auto* bus = m_appCtx->eventBus()) {
            // 锁内轻判红区：publish 持总线锁——handler 只拷贝 param2(新态)，
            // 绝不 transition/publish/直写 UI；Queued 投递主线程后刷指示
            m_stateChangedSubId_ = bus->subscribe(Scanner::EventType::StateChanged,
                [this](const Scanner::Event& ev) {
                    const int64_t newState = ev.param2;
                    QMetaObject::invokeMethod(this, [this, newState]() {
                        updateStateIndicator(static_cast<Scanner::service::SystemState>(newState));
                        updateVirtualKeypadStates();      // P-键盘态同步：灰化/恢复表盘
                    }, Qt::QueuedConnection);
                });
            // —— 按键管理闭环（260927 接线；261002 按键域定稿事件段迁移）：菜单
            //    事件族 p1=101-105（=100+游标：102 就绪/103 扫描完成/104 后处理/
            //    105 重置）、菜单变化=110、档位变化=111-114；参数改账=1000+idx。
            //    总线锁纪律同上：锁内只拷贝，Queued 切 UI 线程再动手
            m_userDefinedSubId_ = bus->subscribe(Scanner::EventType::UserDefined,
                [this](const Scanner::Event& ev) {
                    const int64_t p1 = ev.param1, p2 = ev.param2;
                    QMetaObject::invokeMethod(this, [this, p1, p2]() {
                        if (p1 == 103) {           // 菜单③「扫描完成」＝关闭会话＋GBA
                            if (!m_appCtx || !m_appCtx->isScanSessionActive()) return;
                            if (m_activeScanToolIdx >= 0) setScanButtonVisual(m_activeScanToolIdx, false);
                            m_activeScanToolIdx = -1;
                            auto r = m_appCtx->stopScanSession();
                            if (r.success && !m_finalBADlg) {   // 统一弹「全局优化中」
                                m_finalBADlg = new QProgressDialog(
                                    QStringLiteral("全局优化中，请稍候……"), QString(), 0, 100, this);
                                m_finalBADlg->setWindowTitle(QStringLiteral("全局优化"));
                                m_finalBADlg->setWindowModality(Qt::ApplicationModal);
                                m_finalBADlg->setMinimumDuration(0);
                                m_finalBADlg->setAutoClose(false);
                                m_finalBADlg->show();
                            }
                            statusBar()->showMessage(r.success
                                ? QStringLiteral("按键完成扫描——全局优化后台执行中")
                                : QString::fromStdString("完成被拒: " + r.message));
                        } else if (p1 == 104) {    // 菜单④「后处理」＝STL 导出批算
                            if (!m_appCtx) return;
                            auto r = m_appCtx->startPostProcessSession("scan_output.stl", 0);
                            statusBar()->showMessage(r.success
                                ? QStringLiteral("按键启动后处理——STL 导出执行中")
                                : QString::fromStdString("后处理被拒: " + r.message));
                        } else if (p1 == 102) {    // 菜单②「进入就绪」（261002 新增消费：
                            // 面片就绪流程——同屏幕模式键 armScanSession 路径）
                            if (!m_appCtx) return;
                            if (m_appCtx->deviceManager())
                                m_appCtx->deviceManager()->setCalibCaptureArmed(false);
                            auto r = m_appCtx->armScanSession(Scanner::ScanMode::MarkerPlusLaser);
                            statusBar()->showMessage(r.success
                                ? QStringLiteral("按键就绪——按设备 M 键开始扫描")
                                : QString::fromStdString("就绪被拒: " + r.message));
                        } else if (p1 == 105) {    // 菜单⑤「重置」（261002：停采＋撕会话
                            // 不落库＋回 S2——毁灭性操作已二次确认过，此处直接执行）
                            if (!m_appCtx) return;
                            if (m_appCtx->deviceManager() &&
                                m_appCtx->deviceManager()->isCapturing())
                                m_appCtx->deviceManager()->stopCapture();
                            if (m_appCtx->isScanSessionActive()) {
                                auto r = m_appCtx->stopScanSession();
                                statusBar()->showMessage(r.success
                                    ? QStringLiteral("会话已重置（数据丢弃不落库）")
                                    : QString::fromStdString("重置被拒: " + r.message));
                            }
                            if (m_activeScanToolIdx >= 0) setScanButtonVisual(m_activeScanToolIdx, false);
                            m_activeScanToolIdx = -1;
                        } else if (p1 == 100) {    // View 视点缩放（260927 旧口径保留）
                            if (m_3dView) m_3dView->zoomView(p2 > 0 ? 0.8 : 1.25);
                        } else if (p1 == 112) {    // P-2 显示远近档（261002：5 档距离
                            // 预设——只改预览观看远近；快照兜底见启动对齐）
                            if (m_3dView) m_3dView->setViewDistanceLadder(static_cast<int>(p2));
                        } else if (p1 == 111) {    // P-3 档位横幅（按键来源专属——UI 不来）
                            // ＋P-5 回显：参数面板档位滑条同步（阻断防环路）＋三参只读行刷新
                            showBanner(QStringLiteral("亮度 ▸ 第%1/%2档")
                                       .arg(p2).arg(m_appCtx->deviceManager()
                                                        ? m_appCtx->deviceManager()->presetLadderSize()
                                                        : 20));
                            if (m_param1Slider) {
                                const QSignalBlocker blocker(m_param1Slider);
                                m_param1Slider->setValue(static_cast<int>(p2));
                            }
                            if (m_paramROLabel) {
                                auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
                                if (dm) {
                                    const auto e = dm->getParam("exposure").value;
                                    const auto l = dm->getParam("laserLevel").value;
                                    const auto b = dm->getParam("bgLight").value;
                                    m_paramROLabel->setText(QStringLiteral(
                                        "三参（随档只读）：曝光 %1 ms · 激光 %2 · 补光 %3")
                                        .arg(e, 0, 'f', 1).arg(l, 0, 'f', 0).arg(b, 0, 'f', 0));
                                }
                            }
                        } else if (p1 == 113) {    // 体素密度档（①子态常驻行内刷新
                            refreshBannerPersistent();
                            refreshMenuDialog();       // P-菜单弹窗①子态值更新
                            // P-分辨率：滑条回显——不用 QSignalBlocker（valueChanged
                            // 只刷标签不回调 setVoxelLadderIndex，无环路）
                            if (m_voxelSlider) {
                                auto* dm113 = m_appCtx ? m_appCtx->deviceManager() : nullptr;
                                if (dm113) m_voxelSlider->setValue(
                                    static_cast<int>(p2) > 0 ? static_cast<int>(p2)
                                                             : dm113->voxelLadderIndex());
                            }
                        } else if (p1 == 114) {     // 景深直切
                            showBanner(p2 == 1 ? QStringLiteral("景深 ▸ 远")
                                               : QStringLiteral("景深 ▸ 近"));
                        } else if (p1 == 110) {     // 菜单变化（含子态进出）
                            refreshBannerPersistent();
                            refreshMenuDialog();       // P-菜单弹窗
                        } else if (p1 == 115) {     // 换调节对象 → 亮度（P-键盘补）
                            showBanner(QStringLiteral("调节 ▸ 亮度"));
                        } else if (p1 == 116) {     // 换调节对象 → 显示远近（P-键盘补）
                            showBanner(QStringLiteral("调节 ▸ 显示远近"));
                        } else if (p1 == 117) {     // 切扫描模式（P-键盘补：param2=modeCursor）
                            showBanner(QStringLiteral("模式 ▸ %1").arg(
                                p2 == 1 ? QStringLiteral("精细扫描")
                              : p2 == 2 ? QStringLiteral("深孔扫描")
                                        : QStringLiteral("面片扫描")));
                        } else if (p1 == 120) {     // 执行结果/拒因（P-3：红底横幅）
                            // p2 拒因码：1=已急停 2=标点会话隔离 3=扫描中改密度防呆
                            if (m_menuDlg) m_menuDlg->hide();   // 急停/拒因关菜单弹窗
                            showBanner(p2 == 1 ? QStringLiteral("已急停")
                                             : p2 == 2 ? QStringLiteral("标点会话·模式锁定")
                                             : p2 == 3 ? QStringLiteral("扫描中·密度锁定")
                                                       : QStringLiteral("操作被拒"),
                                       true);
                        }
                    }, Qt::QueuedConnection);
                });
        }
    }

    startInfoTimer();
    // 261002 临时测试机：无实体按键——软件启动即弹虚拟键盘（自检期灰化，
    // 自检完自动恢复 S2 可用态；StateChanged 事件驱动灰化/恢复已接线）
    QTimer::singleShot(100, this, [this]() { showVirtualKeypad(); });
}

MainWindow::~MainWindow() {
    // 退订生命周期：window 先于 appCtx 析构（main.cpp 声明顺序）——退订时 bus 必存活
    if (m_stateChangedSubId_ && m_appCtx) {
        if (auto* bus = m_appCtx->eventBus())
            bus->unsubscribe(m_stateChangedSubId_);
        m_stateChangedSubId_ = 0;
    }
    if (m_userDefinedSubId_ && m_appCtx) {
        if (auto* bus = m_appCtx->eventBus())
            bus->unsubscribe(m_userDefinedSubId_);
        m_userDefinedSubId_ = 0;
    }
}

// —— P1-2 7 态文案/配色映射（10 状态机 SystemState；落点：状态栏右下角常驻件）——
QString MainWindow::stateText(Scanner::service::SystemState s)
{
    switch (s)
    {
        case Scanner::service::SystemState::Init:            return QStringLiteral("初始化");
        case Scanner::service::SystemState::Standby:         return QStringLiteral("待机");
        case Scanner::service::SystemState::Calibrating:     return QStringLiteral("标定中");
        case Scanner::service::SystemState::ScanMarker:      return QStringLiteral("扫描·标点");
        case Scanner::service::SystemState::ScanMarkerLaser: return QStringLiteral("扫描·标志+激光");
        case Scanner::service::SystemState::PostProcessing:  return QStringLiteral("后处理中");
        case Scanner::service::SystemState::FaultSelfCheck:  return QStringLiteral("故障自检");
    }
    return QStringLiteral("未知");
}

QString MainWindow::stateColor(Scanner::service::SystemState s)
{
    switch (s)
    {
        case Scanner::service::SystemState::Init:            return QStringLiteral("#8E8E8E");
        case Scanner::service::SystemState::Standby:         return QStringLiteral("#00AA00");
        case Scanner::service::SystemState::Calibrating:     return QStringLiteral("#0066FF");
        case Scanner::service::SystemState::ScanMarker:      return QStringLiteral("#00B0C4");
        case Scanner::service::SystemState::ScanMarkerLaser: return QStringLiteral("#6C4FD6");
        case Scanner::service::SystemState::PostProcessing:  return QStringLiteral("#E69112");
        case Scanner::service::SystemState::FaultSelfCheck:  return QStringLiteral("#C0392B");
    }
    return QStringLiteral("#8E8E8E");
}

void MainWindow::updateStateIndicator(Scanner::service::SystemState s)
{
    if (!m_stateIndicator) return;
    m_stateIndicator->setText(QStringLiteral("● ") + stateText(s));
    m_stateIndicator->setStyleSheet(QStringLiteral("color:%1;").arg(stateColor(s)));
}

void MainWindow::onIntegrateTestClicked()
{
    if (!m_integrateTestDialog) {
        auto* dlg = new ScannerWindow(m_appCtx, this);
        m_integrateTestDialog = dlg;
    }
    m_integrateTestDialog->show();
    m_integrateTestDialog->setWindowState(Qt::WindowActive);
    m_integrateTestDialog->raise();
    m_integrateTestDialog->activateWindow();
}

void MainWindow::onReloadPointCloud()
{
    m_3dView->clearScene();
    m_importedCloudCount = 0;           // 3D 重载=回到仓库快照（导入云计数随之清）
    if (m_cloudItem001)
        m_cloudItem001->setText(0, QStringLiteral("点云数据 001 (0)"));
    // 3000 万点截断（100M.ply 全量 ~2 亿？实为 2000 万；30M 上限=测试口径：真实
    // 扫描单工件点量级远低于此，全量加载浪费时间）
    m_3dView->loadTestDataFromPLY("D:/pointcloud_100M.ply", 30000000);
}

void MainWindow::onCalibDeviceClicked()
{
    JMW_LOG_INFO("app-MainWindow", "[calib] 校准设备点击（v3 透明验证：材质 alpha=0.15＋双级混合）");
    {
        FILE* f = fopen("E:/workfold/20260509intergrate/calib_debug.log", "a");
        if (f) { fprintf(f, "[%s] onCalibDeviceClicked ENTER m_3dView=%p\n", __TIME__, (void*)m_3dView); fclose(f); }
    }
    if (!m_calibDialog) {
        m_calibDialog = new CalibDialog(this);
        connect(m_calibDialog, &CalibDialog::cameraCalibClicked, this, [this]() {
            statusBar()->showMessage(QStringLiteral("相机标定：正在打开相机..."));

            auto* series = static_cast<LEADSCANSeries*>(m_integrateTestDialog);
            if (!series) {
                if (!m_integrateTestDialog) {
                    m_integrateTestDialog = new LEADSCANSeries();
                }
                series = static_cast<LEADSCANSeries*>(m_integrateTestDialog);
            }
            auto* cam = series->getCameraControl();
            if (!cam || !cam->isScannerCameraOpen()) {
                QMessageBox::warning(this, QStringLiteral("相机标定"), QStringLiteral("请先在集成测试中打开扫描相机"));
                return;
            }

            const int numFrames = 15;
            // 打光（261002 补）：相机标定＝棋盘格场景只开补光（激光线会毁角点
            // 检测）；采完全灭。经门面灯控＝N10 即时生效，无需启采集
            if (m_appCtx && m_appCtx->deviceManager())
                m_appCtx->deviceManager()->lightsBgOnly();
            calibration::CameraCalibInput input;
            input.imageWidth = 2048;
            input.imageHeight = 1536;

            for (int i = 0; i < numFrames; ++i) {
                statusBar()->showMessage(QStringLiteral("采集标定图像 %1/%2").arg(i + 1).arg(numFrames));
                QApplication::processEvents();

                cv::Mat left, right;
                cam->GetScannerImages(left, right, 10000);
                if (!left.empty() && !right.empty()) {
                    input.leftImages.push_back(left.clone());
                    input.rightImages.push_back(right.clone());
                }
            }
            if (m_appCtx && m_appCtx->deviceManager())
                m_appCtx->deviceManager()->lightsAllOff();

            statusBar()->showMessage(QStringLiteral("正在执行相机标定算法..."));
            QApplication::processEvents();

            calibration::CameraCalibWorkflow workflow;
            workflow.setProgressCallback([this](int pct, const std::string& step) {
                statusBar()->showMessage(QString::fromStdString(step) + " (" + QString::number(pct) + "%)");
                QApplication::processEvents();
            });

            auto result = workflow.run(input);
            if (result.success) {
                QMessageBox::information(this, QStringLiteral("相机标定"),
                    QStringLiteral("标定完成!\n左重投影误差: %1\n右重投影误差: %2\n立体重投影误差: %3")
                    .arg(result.reprojError, 0, 'f', 4));
            } else {
                QMessageBox::warning(this, QStringLiteral("相机标定"), QString::fromStdString(result.message));
            }
        });
        connect(m_calibDialog, &CalibDialog::laserCalibClicked, this, [this]() {
            statusBar()->showMessage(QStringLiteral("激光线标定：正在打开相机..."));

            auto* series = static_cast<LEADSCANSeries*>(m_integrateTestDialog);
            if (!series) {
                if (!m_integrateTestDialog) {
                    m_integrateTestDialog = new LEADSCANSeries();
                }
                series = static_cast<LEADSCANSeries*>(m_integrateTestDialog);
            }
            auto* cam = series->getCameraControl();
            if (!cam || !cam->isScannerCameraOpen()) {
                QMessageBox::warning(this, QStringLiteral("激光线标定"), QStringLiteral("请先在集成测试中打开扫描相机"));
                return;
            }

            statusBar()->showMessage(QStringLiteral("采集激光线图像..."));
            QApplication::processEvents();

            // 打光（261002 补）：激光线标定必须点亮激光——左右斜交叉 V1C1
            // （261002 临时测试机管语义；经门面灯控＝N10 即时生效，无需启采集），
            // 采完全灭
            if (m_appCtx && m_appCtx->deviceManager())
                m_appCtx->deviceManager()->lightsBgAndCrossLaser();

            cv::Mat left, right;
            cam->GetScannerImages(left, right, 10000);
            if (left.empty() || right.empty()) {
                if (m_appCtx && m_appCtx->deviceManager())
                    m_appCtx->deviceManager()->lightsAllOff();
                QMessageBox::warning(this, QStringLiteral("激光线标定"), QStringLiteral("图像采集失败"));
                return;
            }
            if (m_appCtx && m_appCtx->deviceManager())
                m_appCtx->deviceManager()->lightsAllOff();

            statusBar()->showMessage(QStringLiteral("正在执行激光线标定算法..."));
            QApplication::processEvents();

            calibration::LaserCalibWorkflow workflow;
            workflow.setProgressCallback([this](int pct, const std::string& step) {
                statusBar()->showMessage(QString::fromStdString(step) + " (" + QString::number(pct) + "%)");
                QApplication::processEvents();
            });

            calibration::LaserCalibInput input;
            input.leftImage = left;
            input.rightImage = right;

            auto result = workflow.run(input);
            if (result.success) {
                QMessageBox::information(this, QStringLiteral("激光线标定"),
                    QStringLiteral("标定完成!\n激光线数: %1\n端点数: %2")
                    .arg(result.lineCount).arg(result.totalEndpoints));
            } else {
                QMessageBox::warning(this, QStringLiteral("激光线标定"), QString::fromStdString(result.message));
            }
        });
    }
    // 分屏：左 3D 扫描仪 + 右 2D 标定板
    if (m_3dView && m_3dViewArea) {
        // 隐藏悬浮工具条
        if (m_floatingToolbar) {
            m_floatingToolbar->setVisible(false);
            m_floatingToolbar->hide();
            m_floatingToolbar->move(-10000, -10000);
        }

        // 加载扫描仪 STL 到 3D 视图
        std::string stlTarget = "E:/workfold/framework/build/JEAMMSCAN.stl";
        osg::ref_ptr<osg::Group> scene = calib_display::buildCalibScene(stlTarget);
        m_3dView->setSceneData(scene);
        m_3dView->setCenterOverlayVisible(false);
        // 设置视角：X横、Y竖、Z朝外（正前视图）
        auto* manip = new osgGA::TrackballManipulator();
        osg::BoundingSphere bs = scene->getBound();
        double dist = 500.0;
        manip->setHomePosition(
            osg::Vec3(bs.center().x(), bs.center().y(), bs.center().z() + dist),
            bs.center(),
            osg::Vec3(0.0, 1.0, 0.0)
        );
        m_3dView->setCameraManipulator(manip);
        manip->home(0);
        // 固定模型：锁定视角（禁止旋转和缩放）
        m_3dView->viewer()->frame();
        osg::Matrix lockedView = m_3dView->viewer()->getCamera()->getViewMatrix();
        m_3dView->setCameraManipulator(nullptr);
        m_3dView->viewer()->getCamera()->setViewMatrix(lockedView);

        // 创建 2D 标定板 + 彩条（ArrowSlider，和远近一样的风格）
        if (!m_calibBoard2D) {
            m_calibBoard2D = new calib_display::CalibBoard2D();
            auto* viewContainer = m_3dView->parentWidget();
            auto* hLayout = qobject_cast<QHBoxLayout*>(viewContainer->layout());
            auto* viewArea = viewContainer->parentWidget();
            auto* vLayout = qobject_cast<QVBoxLayout*>(viewArea->layout());

            // 上侧横排：左右（ArrowSlider 水平 + 标签），两侧留 10% 空白使总长 80%
            if (vLayout) {
                auto* lrWidget = new QWidget();
                lrWidget->setObjectName("calibLrBar");
                lrWidget->setFixedHeight(60);
                auto* lrLayout = new QHBoxLayout(lrWidget);
                lrLayout->setContentsMargins(0, 10, 0, 10);
                lrLayout->setSpacing(4);
                lrLayout->addStretch(1);  // 左侧 10% 空白
                auto* labelL = new QLabel(QStringLiteral("\xe5\xb7\xa6"));  // 左
                labelL->setAlignment(Qt::AlignCenter);
                labelL->setStyleSheet("color: #0066FF; font-size: 14px; font-weight: bold;");
                auto* lrSlider = new ArrowSlider(Qt::Horizontal);
                lrSlider->setRange(0, 100);
                lrSlider->setValue(50);
                lrSlider->setFixedHeight(48);
                lrSlider->setMinimumWidth(200);
                lrSlider->setGroovePixmap(renderSvg(":/icons/resources/icons/div.color-gradient-bar.svg", 800, 24));
                auto* labelR = new QLabel(QStringLiteral("\xe5\x8f\xb3"));  // 右
                labelR->setAlignment(Qt::AlignCenter);
                labelR->setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;");
                lrLayout->addWidget(labelL);
                lrLayout->addWidget(lrSlider, 8);  // 80% 比例
                lrLayout->addWidget(labelR);
                lrLayout->addStretch(1);  // 右侧 10% 空白
                vLayout->insertWidget(0, lrWidget);
            }

            // 右侧竖排：前后（ArrowSlider 垂直 + 标签，和远近一样）
            if (hLayout) {
                hLayout->addWidget(m_calibBoard2D, 1);
                // 前后容器（和远近完全一样的结构）
                auto* fbWidget = new QWidget();
                fbWidget->setObjectName("calibFbBar");
                auto* fbLayout = new QVBoxLayout(fbWidget);
                fbLayout->setContentsMargins(6, 10, 6, 10);
                auto* labelF = new QLabel(QStringLiteral("\xe5\x89\x8d"));  // 前
                labelF->setAlignment(Qt::AlignCenter);
                labelF->setStyleSheet("color: #0066FF; font-size: 14px; font-weight: bold;");
                fbLayout->addWidget(labelF);
                auto* fbSlider = new ArrowSlider(Qt::Vertical);
                fbSlider->setObjectName("fbSlider");
                fbSlider->setRange(0, 100);
                fbSlider->setValue(50);
                fbSlider->setFixedWidth(48);
                fbSlider->setMinimumHeight(200);
                fbSlider->setGroovePixmap(renderSvg(":/icons/resources/icons/div.color-gradient-bar.svg", 24, 800));
                fbSlider->setStyleSheet(
                    "QSlider#fbSlider::groove:vertical { background: transparent; }"
                    "QSlider#fbSlider::handle:vertical { background: transparent; width: 20px; height: 16px; }"
                    "QSlider#fbSlider::sub-page:vertical { background: transparent; }"
                    "QSlider#fbSlider::add-page:vertical { background: transparent; }"
                );
                fbLayout->addWidget(fbSlider, 1);
                auto* labelB = new QLabel(QStringLiteral("\xe5\x90\x8e"));  // 后
                labelB->setAlignment(Qt::AlignCenter);
                labelB->setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;");
                fbLayout->addWidget(labelB);
                hLayout->addWidget(fbWidget);
            }
        }
        // 每次进入标定都显示（第二次以后也能显示）
        m_calibBoard2D->show();
        m_calibBoard2D->update();
        {
            auto* va = m_3dView ? m_3dView->parentWidget()->parentWidget() : nullptr;
            if (va) { auto* b = va->findChild<QWidget*>("calibLrBar"); if (b) b->show(); }
            auto* vc = m_3dView ? m_3dView->parentWidget() : nullptr;
            if (vc) { auto* b = vc->findChild<QWidget*>("calibFbBar"); if (b) b->show(); }
        }

        statusBar()->showMessage(QStringLiteral("标定显示模式"));
    }
    // 261002 用户标定五态循环采集（用户口径：点校准设备→点 M 开拍→再点 M 停）：
    // 布防＝帧流接线＋标定布防（AppContext::armCalibCapture——不接帧流则相机
    // 不开流零帧，灯序停在补光态不循环）；M 键启采分流进五态灯序自动循环
    if (m_appCtx) {
        const auto r = m_appCtx->armCalibCapture();
        statusBar()->showMessage(r.success
            ? QStringLiteral("校准设备：按 M 键（或虚拟按键中键单击）开始五态循环采集"
                             "——补光→左斜→右斜→精细→深孔；再按 M 停止")
            : QString::fromStdString("校准采集布防失败: " + r.message), 8000);
    }
    showVirtualKeypad();
}

// P0-3 编辑门禁唯一事实源：经 AppContext::canEnterEditSession()（SM==S2 或暂停
// 就绪态）且有点云。不满足→拒并状态栏提示（工具栏编辑工具统一走此口）
bool MainWindow::ensureEditAllowed() {
    if (m_appCtx && m_appCtx->canEnterEditSession()) return true;
    // 拒绝留痕（260927 排障插桩：套索点击无反应时定位三条件哪个不满足）
    if (m_appCtx) {
        const size_t mk = m_appCtx->sceneFeed() ? m_appCtx->sceneFeed()->latestMarkers().size() : 0;
        const auto pc = m_appCtx->pointCloudBuffer() ? m_appCtx->pointCloudBuffer()->getTotalPointCount() : 0;
        JMW_LOG_WARN("app-MainWindow",
            "[编辑门禁] 拒绝——标志点={} 点云={}（状态/暂停详情见 canEnterEditSession 口径）",
            mk, pc);
    }
    statusBar()->showMessage(
        QStringLiteral("编辑不可用：需处于待机/扫描就绪态且有标志点或激光点云"), 3000);
    return false;
}

void MainWindow::onScanClicked()
{
    // 切换回默认界面：隐藏标定板和彩条，恢复相机
    if (m_calibBoard2D) m_calibBoard2D->hide();
    // 隐藏左右、前后彩条
    auto* viewArea = m_3dView ? m_3dView->parentWidget()->parentWidget() : nullptr;
    if (viewArea) {
        auto* lrBar = viewArea->findChild<QWidget*>("calibLrBar");
        if (lrBar) lrBar->hide();
    }
    auto* viewContainer = m_3dView ? m_3dView->parentWidget() : nullptr;
    if (viewContainer) {
        auto* fbBar = viewContainer->findChild<QWidget*>("calibFbBar");
        if (fbBar) fbBar->hide();
    }
    // 清空 3D 场景，恢复初始状态
    if (m_3dView) {
        m_3dView->clearScene();
        m_3dView->setCenterOverlayVisible(true);
        m_3dView->viewer()->getCamera()->setClearColor(osg::Vec4(0.412f, 0.412f, 0.412f, 1.0f));
        auto* manip = new osgGA::TrackballManipulator();
        m_3dView->setCameraManipulator(manip);
        manip->home(0);
    }
    // 显示悬浮工具条
    if (m_floatingToolbar) {
        m_floatingToolbar->setVisible(true);
        m_floatingToolbar->show();
    }

    auto* series = static_cast<LEADSCANSeries*>(m_integrateTestDialog);
    (void)series;   // 旧 stub 路径退役：下方改走真链（AppContext 统一点火口）
    if (!m_appCtx) return;

    // 真链启停切换：navBar"扫描"——活跃则停（复原活跃键）、空闲则面片扫描（B）
    if (m_appCtx->isScanSessionActive()) {
        auto r = m_appCtx->stopScanSession();
        if (m_activeScanToolIdx >= 0) setScanButtonVisual(m_activeScanToolIdx, false);
        m_activeScanToolIdx = -1;
        statusBar()->showMessage(r.success ? QStringLiteral("扫描停止中——会话合账")
                                           : QString::fromStdString("停止被拒: " + r.message));
        return;
    }
    applyMeshPreset();       // 导航"扫描"＝面片：推荐预设＋旋钮同步（260912）
    m_laserSessionLatched = false;   // 新会话：激光仓库基线待重锁（260912c）
    if (m_appCtx->deviceManager())
        m_appCtx->deviceManager()->setCalibCaptureArmed(false);   // 扫描会话撤标定布防
    auto r = m_appCtx->startScanSession(Scanner::ScanMode::MarkerPlusLaser);
    if (!r.success) {
        QMessageBox::warning(this, QStringLiteral("扫描"),
            QString::fromStdString("扫描启动被拒:\n" + r.message));
    } else {
        setScanButtonVisual(3, true);           // navBar 起的面片——只亮面片键
        m_activeScanToolIdx = 3;
        showCameraMonitor();                    // 调试：弹相机左右图监视
        statusBar()->showMessage(QString::fromStdString(r.message) + QStringLiteral("——再点停止"));
    }
}

// ============================================================================
// 相机预览监视弹窗（调试）：扫描启动时弹出，实时显示左右相机灰度图——观察
// 灯帧交替（补光帧↔激光帧）/标记点可见性。数据走 AppContext 调试帧分路
// （相机 SDK 线程直调，此处 invokeMethod queued 切 UI 线程；每 3 帧刷新）
// ============================================================================
namespace {
QImage camMatToImage(const cv::Mat& m) {
    if (m.empty()) return {};
    return QImage(m.data, m.cols, m.rows, static_cast<qsizetype>(m.step),
                  QImage::Format_Grayscale8).copy();
}

// pushParam1Scaled（260912 三参比例旋钮）随 P-5 收敛退役——三参唯一合法写入口
// 是亮度梯换档（禁绕梯直改三参，设计 §3.3.2 一本账纪律）
} // namespace

void MainWindow::showCameraMonitor() {
    if (!m_appCtx) return;
    if (!m_camDlg) {
        m_camDlg = new QDialog(this);
        m_camDlg->setWindowTitle(QStringLiteral("相机预览监视（左 / 右）——调试"));
        m_camDlg->setModal(false);
        m_camDlg->resize(1040, 420);
        m_camLeft = new QLabel(m_camDlg);
        m_camRight = new QLabel(m_camDlg);
        for (QLabel* l : {m_camLeft, m_camRight}) {
            l->setMinimumSize(500, 380);
            l->setAlignment(Qt::AlignCenter);
            l->setStyleSheet("background:#202225;");
        }
        m_camFrameLabel = new QLabel(m_camDlg);
        m_camFrameLabel->setStyleSheet(
            "color:#3c7ad9; font-family:Consolas; font-size:12px; padding:2px 8px;");
        // 单一 QVBoxLayout：上层图片行（QHBoxLayout）＋底部帧号行
        auto* vLay = new QVBoxLayout(m_camDlg);
        vLay->setContentsMargins(4, 4, 4, 2);
        vLay->setSpacing(2);
        auto* imgLay = new QHBoxLayout();
        imgLay->setContentsMargins(0, 0, 0, 0);
        imgLay->addWidget(m_camLeft);
        imgLay->addWidget(m_camRight);
        vLay->addLayout(imgLay);
        vLay->addWidget(m_camFrameLabel);
        // 关闭即摘分路（帧回调不再进 UI）——260912：记日志（tap 空＝监视窗死因
        // 定位锚点：关闭时刻 vs 后续画面停时刻的相对关系直接读日志）
        connect(m_camDlg, &QDialog::finished, this, [this]() {
            if (m_appCtx) m_appCtx->setDebugFrameTap(nullptr);
            JMW_LOG_INFO("app-MainWindow", "[预览] 监视窗关闭——tap 摘除");
        });
    }
    m_appCtx->setDebugFrameTap([this](const Scanner::hal::StereoFrame& f) {
        // —— 260912 崩溃隔离（提交内卡死根因手术）：本 tap 可能从左右两个相机
        //    SDK 线程交替进入，体内任一异常会被 CameraControl 回调最外层
        //    catch(...) 整体吞掉——此后每帧同点重抛＝tap「永久死」而扫描链（同
        //    回调的②）照跑（真机日志实证：心跳停在第 1201 帧，管线跑到会话尾）
        //    。整体兜异常＋限频记日志，单帧故障不再杀死预览链 ——
        try {
        // 接收帧率计数（节流前每帧计数——给帧号行用；双 SDK 线程交替进入——
        // s_lastRxTick 非原子 static 须互斥护，撕裂值可触发异常路径）
        static std::mutex s_fpsMtx;
        static std::atomic<uint64_t> s_rxCnt{0};
        static std::atomic<uint64_t> s_rxFps{0};
        ++s_rxCnt;
        {
            std::lock_guard<std::mutex> lock(s_fpsMtx);
            static auto s_lastRxTick = std::chrono::steady_clock::now();
            const auto now = std::chrono::steady_clock::now();
            if (now - s_lastRxTick >= std::chrono::seconds(1)) {
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now - s_lastRxTick).count();
                s_rxFps.store(ms > 0 ? s_rxCnt.exchange(0) * 1000 / ms : 0);
                s_lastRxTick = now;
            }
        }
        // 时间基准节流（2026-09-05）：帧率上限＝config/camera.json previewFps
        //（原帧计数法绑定 30fps 流假设——不同帧率下预览 fps 漂移）
        using clock = std::chrono::steady_clock;
        const auto now = clock::now();
        thread_local auto lastPreview = clock::time_point{};
        const auto minInterval = std::chrono::microseconds(
            1000000 / std::max(1, m_appCtx ? m_appCtx->cameraPreviewFps() : 10));
        if (now - lastPreview < minInterval) return;
        lastPreview = now;
        // —— 260911 预览减负（显示 10fps 不变）：原全分辨率 clone（6MB/帧）＋
        //    UI 侧全尺寸 QImage 拷贝/QPixmap 缩放改为回调线程一次降采样 1/4
        //    （INTER_AREA 质量更优）——送 UI 载荷 6MB→~200KB，UI 线程像素工作
        //    量降两个量级，拖动主窗/对话框不再与预览互卡
        cv::Mat lSmall, rSmall;
        if (!f.leftGray.empty())
            cv::resize(f.leftGray, lSmall, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        if (!f.rightGray.empty())
            cv::resize(f.rightGray, rSmall, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        const auto fidL = f.frameIdLeft;
        const auto fidR = f.frameIdRight;
        // T/V 激光组判定（260927 时间戳奇偶法，CameraControl 随帧）：预览监视显示
        const bool tvKnown = f.tvKnown;
        const bool tvLeft = f.tvLeftSkew;

        // —— 激光线宽度测量（2026-09-06）：260912 诊断隔离（用户指令）——整块
        //    暂停调用：验证预览卡死是否与其相关（tap 内异常被相机回调外层
        //    catch(...) 吞掉＝预览链猝死的嫌疑源之一）。定位后按结论恢复或
        //    重写。标签显示沿用 0 ——
        // 【原实现见 post_0336317_回退备份.patch／git 历史：采样列×全高扫描
        //  img.at 逐像素找亮段取中位数，1Hz 节流，f.leftGray 原图测量】
        static std::atomic<int> s_laserWidth{0};
        // static std::atomic<uint64_t> s_previewCnt{0};
        // if (s_previewCnt.fetch_add(1) % 10 == 0) {
        //     int lw = 0;
        //     const cv::Mat& img = f.leftGray;   // 左图（原图测量——缩放会改线宽）
        //     if (!img.empty()) {
        //         std::vector<int> runs;
        //         const int cols = img.cols;
        //         const int step = std::max(1, cols / 100);
        //         for (int x = 0; x < cols; x += step) {
        //             int maxRun = 0, curRun = 0;
        //             for (int y = 0; y < img.rows; ++y) {
        //                 if (img.at<uint8_t>(y, x) > 150) {
        //                     ++curRun;
        //                     if (curRun > maxRun) maxRun = curRun;
        //                 } else {
        //                     curRun = 0;
        //                 }
        //             }
        //             if (maxRun > 2 && maxRun < img.rows / 10)
        //                 runs.push_back(maxRun);
        //         }
        //         if (!runs.empty()) {
        //             std::sort(runs.begin(), runs.end());
        //             lw = runs[runs.size() / 2];
        //         }
        //     }
        //     s_laserWidth.store(lw);
        // }
        const int laserWidth = s_laserWidth.load();

        QMetaObject::invokeMethod(this, [this, lSmall, rSmall, fidL, fidR, laserWidth,
                                         tvKnown, tvLeft]() {
            // 260912：isVisible 闸移除——帧流/窗开/定时器活三者俱证时预览仍停，
            // 该闸为残余嫌疑（隐藏窗 setPixmap 无害且 10fps 开销可忽略）；空判保留
            if (m_camDlg) {
                // UI 侧终审计数（每 30 次≈3s 一条）：lambda 是否真在 UI 线程跑——
                // 有此日志而画面停＝窗口绘制层（ghost/合成器）；无＝投递层。
                // 260912b：带帧号——卡死期间帧号仍在走＝数据新鲜而画面停（渲染层）；
                // 帧号也停＝上游给了重复/旧帧（数据层）
                static std::atomic<uint64_t> s_uiRefresh{0};
                if (s_uiRefresh.fetch_add(1) % 30 == 0)
                    JMW_LOG_INFO("app-MainWindow", "[预览] UI 刷新：累计 {} 次（帧 L{} R{}）",
                                 s_uiRefresh.load(), fidL, fidR);
                // 标题栏活体指示：帧号跳动＝链路活；整窗冻结（含标题）＝渲染层/ghost
                if (m_camDlg)
                    m_camDlg->setWindowTitle(
                        QStringLiteral("相机预览监视（左 / 右）——帧 L%1 R%2  激光组 %3  刷新%4")
                            .arg(static_cast<qulonglong>(fidL))
                            .arg(static_cast<qulonglong>(fidR))
                            .arg(tvKnown ? (tvLeft ? QStringLiteral("T 左斜")
                                                   : QStringLiteral("V 右斜"))
                                         : QStringLiteral("—"))
                            .arg(s_uiRefresh.load()));
                m_camLeft->setPixmap(QPixmap::fromImage(camMatToImage(lSmall))
                                           .scaled(m_camLeft->size(), Qt::KeepAspectRatio));
                m_camRight->setPixmap(QPixmap::fromImage(camMatToImage(rSmall))
                                            .scaled(m_camRight->size(), Qt::KeepAspectRatio));
                // 流水线消费帧率（1s 计算一次——帧计数差分）
                static uint64_t lastPipelineCnt = 0;
                static auto lastCalc = std::chrono::steady_clock::now();
                static double pipelineFps = 0.0;
                {
                    const auto now2 = std::chrono::steady_clock::now();
                    if (now2 - lastCalc >= std::chrono::seconds(1)) {
                        if (m_appCtx && m_appCtx->scanWorkflow()) {
                            const uint64_t cur = m_appCtx->scanWorkflow()->processedFrameCount();
                            const double sec = std::chrono::duration<double>(now2 - lastCalc).count();
                            pipelineFps = sec > 0 ? (cur - lastPipelineCnt) / sec : 0;
                            lastPipelineCnt = cur;
                        }
                        lastCalc = now2;
                    }
                }
                if (m_camFrameLabel)
                    m_camFrameLabel->setText(
                        QStringLiteral("激光组: %1    激光线宽: %2 px    接收: %3 fps    流水线: %4 fps    左帧号: %5    右帧号: %6    偏移: %7")
                            .arg(tvKnown ? (tvLeft ? QStringLiteral("T 左斜")
                                                   : QStringLiteral("V 右斜"))
                                         : QStringLiteral("未知(帧号奇偶)"))
                            .arg(laserWidth)
                            .arg(s_rxFps.load())
                            .arg(pipelineFps, 0, 'f', 1)
                            .arg(static_cast<qulonglong>(fidL))
                            .arg(static_cast<qulonglong>(fidR))
                            .arg(static_cast<qint64>(fidL) - static_cast<qint64>(fidR)));  // 有符号——右超前时显示负偏移（原 qulonglong 相减下溢出巨数）
            }
        }, Qt::QueuedConnection);
        } catch (const std::exception& e) {
            // 限频记错（首错即记；累计计数）——凶手自报名字，预览链不死
            static std::atomic<uint64_t> s_tapErrLogged{0};
            if (s_tapErrLogged.fetch_add(1) % 100 == 0)
                JMW_LOG_ERROR("app-MainWindow",
                    "[相机监视] tap 帧处理异常（已兜住——预览链不死；累计 {}）: {}",
                    s_tapErrLogged.load(), e.what());
        } catch (...) {
            JMW_LOG_ERROR("app-MainWindow", "[相机监视] tap 帧处理未知异常（已兜住）");
        }
    });
    m_camDlg->show();
    m_camDlg->raise();
    m_camDlg->activateWindow();
}

// 扫描就绪窗口（260927 就绪流程·用户口径）：UI 模式键→会话备好（模式/帧流/
// 工作流，不启采）→弹此窗——「按设备 M 键开始扫描」；200ms 轮询 isCapturing
// （M 键 captureToggle 启采）自动关闭；取消＝终止会话（stopScanSession＋键复原）
void MainWindow::showScanReadyPrompt(const QString& modeTitle, int btnIdx) {
    if (!m_scanReadyDlg) {
        m_scanReadyDlg = new QDialog(this);
        m_scanReadyDlg->setWindowTitle(QStringLiteral("扫描就绪"));
        m_scanReadyDlg->setModal(false);
        m_scanReadyDlg->setFixedWidth(360);
        auto* lay = new QVBoxLayout(m_scanReadyDlg);
        m_scanReadyLabel = new QLabel(m_scanReadyDlg);
        m_scanReadyLabel->setAlignment(Qt::AlignCenter);
        m_scanReadyLabel->setStyleSheet(
            "font-size:14px; padding:12px 8px; color:#202225;");
        lay->addWidget(m_scanReadyLabel);
        auto* cancel = new QPushButton(QStringLiteral("取消（终止会话）"), m_scanReadyDlg);
        lay->addWidget(cancel);
        connect(cancel, &QPushButton::clicked, this, [this]() {
            if (m_appCtx) {
                const auto r = m_appCtx->stopScanSession();
                if (!r.success)
                    statusBar()->showMessage(QString::fromStdString("终止被拒: " + r.message));
            }
            if (m_scanReadyBtnIdx >= 0) setScanButtonVisual(m_scanReadyBtnIdx, false);
            m_activeScanToolIdx = -1;
            if (m_scanReadyPoll) m_scanReadyPoll->stop();
            m_scanReadyDlg->close();
        });
        // 轮询：M 键开扫（isCapturing 翻真）自动关；会话已亡（他路终止）也关
        m_scanReadyPoll = new QTimer(this);
        m_scanReadyPoll->setInterval(200);
        connect(m_scanReadyPoll, &QTimer::timeout, this, [this]() {
            if (!m_appCtx) { m_scanReadyPoll->stop(); return; }
            auto* dm = m_appCtx->deviceManager();
            if ((dm && dm->isCapturing()) || !m_appCtx->isScanSessionActive()) {
                m_scanReadyPoll->stop();
                if (m_scanReadyDlg->isVisible()) {
                    m_scanReadyDlg->close();
                    if (dm && dm->isCapturing())
                        statusBar()->showMessage(
                            QStringLiteral("扫描中——按设备 M 键停止"), 5000);
                }
            }
        });
    }
    m_scanReadyBtnIdx = btnIdx;
    m_scanReadyLabel->setText(
        QStringLiteral("%1 已就绪\n\n按设备【M 键】开始扫描（扫描中再按 M 停止）\n"
                       "临时测试机无实体键：点「⌨ 按键」表盘中键【单击】")
            .arg(modeTitle));
    m_scanReadyDlg->show();
    m_scanReadyDlg->raise();
    m_scanReadyDlg->activateWindow();
    m_scanReadyPoll->start();
    showVirtualKeypad();   // 261002 临时测试机：无实体键——表盘随就绪窗自动弹出
    statusBar()->showMessage(modeTitle + QStringLiteral(" 已就绪——按设备 M 键开始扫描（临时机用虚拟按键表盘）"));
}

// 261002 临时测试机：无实体按键——虚拟按键表盘（弹窗）模拟扫描仪面板五键。
// U/L/R/M 四键经 DeviceManager::testInjectTextLine 注入 G01 手势帧，与真机 rx
// 路径完全同链（文本行→手势环→KeySemantics→动作）——表盘点按即等价真机按键；
// 下键（第五键）260831 协议 G01 键位白名单无此键（McuFrame.cpp parseG01Payload
// 仅 U/L/M/R）——置灰标注，待协议方补充后启用。
// P-键盘态同步（§3.2.5 七态矩阵）：StateChanged 事件驱动灰化/恢复——
// S1/S3/S6/S7 全拦态＝除逃生类（M 长按/U 长按）外全灰；S2/S4/S5＝恢复可用
// （功能口拒〔未就绪/标点隔离/防呆〕的键仍可按——拒因经横幅 p1=120 可见，
// 「拒＝可按但有反馈」与「拦＝无效直接灰」区分，设计 §3.2.5 定版口径）
void MainWindow::showVirtualKeypad() {
    if (!m_vkeyPad) {
        m_vkeyPad = new QDialog(nullptr);   // 无父窗口（独立顶层——全屏主窗口不遮）
        m_vkeyPad->setWindowFlag(Qt::WindowStaysOnTopHint, true);
        m_vkeyPad->setWindowTitle(QStringLiteral("虚拟按键表盘"));
        m_vkeyPad->setModal(false);
        m_vkeyPad->setMinimumSize(560, 520);
        auto* lay = new QGridLayout(m_vkeyPad);
        lay->setContentsMargins(16, 12, 16, 12);
        lay->setHorizontalSpacing(10);
        lay->setVerticalSpacing(8);
        const QStringList gestures{
            QStringLiteral("单击"), QStringLiteral("双击"), QStringLiteral("长按")};
        for (int g = 0; g < 3; ++g) {
            auto* h = new QLabel(gestures[g], m_vkeyPad);
            h->setAlignment(Qt::AlignCenter);
            h->setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;");
            lay->addWidget(h, 0, g + 1);
        }
        // 行序＝U/D/L/R/M；按钮文本＝功能名（非手势名——列头已有手势，按钮要
        // 一眼看懂「按了干什么」）；tooltip＝完整描述（§3.2.2 手势总表对齐）
        struct RowDef {
            QString name; QChar key; bool inProto;
            QString btnText[3];    // 按钮文本（功能简称）
            QString tips[3];       // tooltip（完整描述）
        };
        const RowDef rows[] = {
            { QStringLiteral("上键 U"), QChar('U'), true,
              { QStringLiteral("菜单"), QStringLiteral("景深"), QStringLiteral("回主界面") },
              { QStringLiteral("单击＝进/退菜单"),
                QStringLiteral("双击＝景深直切 近↔远"),
                QStringLiteral("长按＝回主界面（全域免门禁）") } },
            { QStringLiteral("下键 D"), QChar('D'), false,
              { QStringLiteral("—"), QStringLiteral("—"), QStringLiteral("—") },
              { QStringLiteral("协议 G01 无此键"), QStringLiteral("—"), QStringLiteral("—") } },
            { QStringLiteral("左键 L"), QChar('L'), true,
              { QStringLiteral("调档↓"), QStringLiteral("换对象"), QStringLiteral("预留") },
              { QStringLiteral("单击＝调档 ↓（菜单态＝游标左移）"),
                QStringLiteral("双击＝换调节对象 亮度↔显示远近"),
                QStringLiteral("长按＝预留") } },
            { QStringLiteral("右键 R"), QChar('R'), true,
              { QStringLiteral("调档↑"), QStringLiteral("预留"), QStringLiteral("预留") },
              { QStringLiteral("单击＝调档 ↑（菜单态＝游标右移）"),
                QStringLiteral("双击＝预留"),
                QStringLiteral("长按＝预留") } },
            { QStringLiteral("中键 M"), QChar('M'), true,
              { QStringLiteral("启停"), QStringLiteral("切模式"), QStringLiteral("急停") },
              { QStringLiteral("单击＝启/停扫描（菜单态＝选中）"),
                QStringLiteral("双击＝切换扫描模式"),
                QStringLiteral("长按＝急停（全域免门禁）") } },
        };
        for (int r = 0; r < 5; ++r) {
            const auto& row = rows[r];
            auto* lbl = new QLabel(row.name, m_vkeyPad);
            lbl->setStyleSheet(row.inProto
                ? "font-weight: bold; font-size: 14px; padding: 4px;"
                : "color:#999; font-size: 14px; padding: 4px;");
            lay->addWidget(lbl, r + 1, 0);
            for (int g = 0; g < 3; ++g) {
                auto* btn = new QPushButton(row.btnText[g], m_vkeyPad);
                btn->setMinimumHeight(44);           // 手指可点的高度
                btn->setMinimumWidth(110);           // 文字不截断
                m_vkeyBtns[r][g] = btn;               // 存指针（态同步用）
                btn->setToolTip(row.tips[g]);
                if (row.inProto) {
                    const QString line = QStringLiteral("G01 %1%2").arg(row.key).arg(g + 1);
                    const QString tip = row.tips[g];
                    connect(btn, &QPushButton::clicked, this, [this, line, tip]() {
                        if (m_appCtx && m_appCtx->deviceManager())
                            m_appCtx->deviceManager()->testInjectTextLine(line.toStdString());
                        statusBar()->showMessage(
                            QStringLiteral("虚拟按键注入: %1（%2）").arg(line, tip), 3000);
                    });
                }
                // 不在此设 enabled——updateVirtualKeypadStates 统一按当前态定
                lay->addWidget(btn, r + 1, g + 1);  // ← 关键行：加入网格布局
            }
        }
        auto* note = new QLabel(
            QStringLiteral("261002 临时测试机：经 G01 注入测试缝下发，与真机按键同链路\n"
                           "蓝框＝可按 · 灰框＝当前态不可用（拦）或协议预留"),
            m_vkeyPad);
        note->setStyleSheet("color:#888; font-size: 12px; padding: 4px;");
        note->setAlignment(Qt::AlignCenter);
        lay->addWidget(note, 6, 0, 1, 4);
        // 当前全局态常驻标签（灰化/恢复是否与系统态同步——用户可验证）
        m_vkeyStateLbl = new QLabel(m_vkeyPad);
        m_vkeyStateLbl->setAlignment(Qt::AlignCenter);
        m_vkeyStateLbl->setStyleSheet("font-size: 13px; padding: 6px;");
        lay->addWidget(m_vkeyStateLbl, 7, 0, 1, 4);

        // 独立刷新定时器（500ms）——不依赖 info timer，排除连接问题
        auto* padTimer = new QTimer(m_vkeyPad);
        connect(padTimer, &QTimer::timeout, this, [this]() {
            updateVirtualKeypadStates();
        });
        padTimer->start(500);

        updateVirtualKeypadStates();                  // 首建即对齐当前态
    }
    m_vkeyPad->show();
    m_vkeyPad->raise();
    m_vkeyPad->activateWindow();
}

// ============================================================================
// P-键盘态同步（§3.2.5 七态矩阵 → 表盘按钮灰化/恢复）
//
// 规则（设计定版）：
//   拦（S1/S3/S6/S7 四个全拦态）＝按钮灰（用户不可按——按了也没意义）
//   拒（功能口拒：未就绪/标点隔离/防呆…）＝按钮亮（可按——拒因经横幅 p1=120
//       可见，「按了被拒」是有效反馈，P3 红底横幅「未就绪」等即此物）
//   逃生类（M 长按急停＋U 长按回主界面）＝任何态恒亮（§3.2.3 真免门禁）
//   预留（L 长按＋R 双击＋R 长按＋D 全行）＝恒灰（协议无/未分配）
//
// 行序＝[0]=U [1]=D [2]=L [3]=R [4]=M；列序＝[0]=单击 [1]=双击 [2]=长按
// ============================================================================
void MainWindow::updateVirtualKeypadStates() {
    if (!m_vkeyPad) return;
    using S = Scanner::service::SystemState;
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    const auto s = (m_appCtx && m_appCtx->stateMachine())
                       ? m_appCtx->stateMachine()->getCurrentState() : S::Init;
    // S1/S3/S6/S7＝全拦（除逃生类）；S2/S4/S5＝放行
    const bool blocking = (s == S::Init || s == S::Calibrating ||
                           s == S::PostProcessing || s == S::FaultSelfCheck);
    // 261002 校准布防子态：calibArmed 且未采集＝S2 的校准上下文——全局态仍
    // S2 但按键语义变了（M＝开拍五态灯序，菜单/调节不适用）→ 键盘须反映
    const bool inCalib = dm && dm->isCalibCaptureArmed();
    const bool capturing = dm && dm->isCapturing();

    // 状态标签：当前全局态＋校准布防上下文（用户验证灰化/恢复是否与系统态同步）
    static const char* kStateNames[] = { "", "S1 初始化", "S2 待机", "S3 标定",
                                         "S4 扫标点", "S5 标点+激光", "S6 后处理", "S7 故障" };
    if (m_vkeyStateLbl) {
        const int si = static_cast<int>(s);
        QString stateText = si >= 1 && si <= 7
            ? QString::fromUtf8(kStateNames[si]) : QStringLiteral("未知");
        QString hintText;
        if (inCalib && !capturing) {
            stateText += QStringLiteral(" · 校准布防");
            hintText = QStringLiteral("（按 M 开始五态循环采集）");
        } else if (capturing && inCalib) {
            stateText += QStringLiteral(" · 校准采集中");
            hintText = QStringLiteral("（采集不可中断——急停退出 或 跑完自动停）");
        } else if (blocking) {
            hintText = QStringLiteral("（除急停/回主界面外全灰）");
        } else {
            hintText = QStringLiteral("（可用）");
        }
        m_vkeyStateLbl->setText(
            QStringLiteral("当前状态：%1  %2").arg(stateText, hintText));
        m_vkeyStateLbl->setStyleSheet(blocking || inCalib
            ? "color: #C0392B; font-weight: bold; padding: 2px;"
            : "color: #27AE60; font-weight: bold; padding: 2px;");
    }

    for (int r = 0; r < 5; ++r) {
        for (int g = 0; g < 3; ++g) {
            auto* btn = m_vkeyBtns[r][g];
            if (!btn) continue;
            const bool escape = (r == 4 && g == 2) || (r == 0 && g == 2);
            const bool reserved = (r == 1) || (r == 2 && g == 2) ||
                                  (r == 3 && g >= 1);
            bool enabled;
            if (inCalib && !capturing) {
                // 校准布防态：M 短按（开拍）＋逃生类可用；菜单/调节/切模式灰
                enabled = (r == 4 && g == 0) || escape;   // M单击 + 逃生
            } else if (inCalib && capturing) {
                // 校准采集中：M 短按**失效**（校准不可中断——要么跑完要么急停
                // 退出，用户口径 261003）；仅逃生类可用
                enabled = escape;
            } else {
                enabled = escape || (!reserved && !blocking);
            }
            btn->setEnabled(enabled);
            // 显式样式：可用＝蓝底白字（校准主操作）或白底蓝框（常规）；
            // 不可用＝灰底半透明
            // isCalibMain 仅在 enabled 时才绿（采集中 M 灰——不可中断）
            const bool isCalibMain = enabled && inCalib && !capturing && (r == 4 && g == 0);
            btn->setStyleSheet(isCalibMain
                ? QStringLiteral(
                    "QPushButton { background-color: #27AE60; color: white;"
                    " border: 2px solid #1E8449; border-radius: 6px;"
                    " font-size: 15px; font-weight: bold; padding: 8px 12px; }"
                    "QPushButton:hover { background-color: #2ECC71; }")
                : enabled
                ? QStringLiteral(
                    "QPushButton { background-color: #E8F0FE; color: #1A5276;"
                    " border: 1px solid #2980B9; border-radius: 6px;"
                    " font-size: 14px; font-weight: bold; padding: 8px 12px; }"
                    "QPushButton:hover { background-color: #D4E6F1; }"
                    "QPushButton:pressed { background-color: #AED6F1; }")
                : QStringLiteral(
                    "QPushButton { background-color: #E8E8E8; color: #B0B0B0;"
                    " border: 1px solid #D0D0D0; border-radius: 6px;"
                    " font-size: 14px; padding: 8px 12px; }"));
        }
    }
}

// 单键扫描按钮态视觉：idx=2 标点/3 面片/4 精细/5 深孔——活跃＝红框＋红字"停止扫描"，
// 停止＝复原黑字原名。各键独立（点哪键哪键变，其他键不动）
void MainWindow::setScanButtonVisual(int idx, bool active){
    if (idx < 0 || idx >= m_toolButtons.size()) return;
    auto* btn = m_toolButtons[idx];
    auto* textLbl = btn->findChild<QLabel*>(QStringLiteral("toolButtonText"));
    if (!textLbl) return;
    if (active) {
        textLbl->setText(QStringLiteral("停止扫描"));
        textLbl->setStyleSheet("color: #C0392B; font-weight: bold;");
        btn->setStyleSheet(
            "QPushButton { background-color: rgba(192,57,43,0.10);"
            " border: 1px solid #C0392B; border-radius: 4px; }"
            "QPushButton:hover { background-color: rgba(192,57,43,0.22); }");
    } else {
        // 复原名对齐工具栏四模式键（协议批3：点云扫描已改名精细扫描＋新增深孔）。
        // 兜底锚定：现调用方仅传 2-5（工具栏四模式键＋navBar 面片=3），idx>4 静默
        // 落「深孔扫描」——加第 6 键须扩此处映射
        const QString name = (idx == 2) ? QStringLiteral("标点扫描")
                           : (idx == 3) ? QStringLiteral("面片扫描")
                           : (idx == 4) ? QStringLiteral("精细扫描")
                                        : QStringLiteral("深孔扫描");
        textLbl->setText(name);
        textLbl->setStyleSheet("");
        btn->setStyleSheet(
            "QPushButton { background-color: transparent; border: none; }"
            "QPushButton:hover { background-color: rgba(0,0,0,0.05); }");
    }
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (m_floatingToolbar) {
        m_floatingToolbar->close();
        delete m_floatingToolbar;
        m_floatingToolbar = nullptr;
    }
    if (m_integrateTestDialog) {
        m_integrateTestDialog->close();
        delete m_integrateTestDialog;
        m_integrateTestDialog = nullptr;
    }
    if (m_calibDialog) {
        m_calibDialog->close();
        delete m_calibDialog;
        m_calibDialog = nullptr;
    }
    QMainWindow::closeEvent(event);
}

void MainWindow::createFloatingToolbar()
{
    m_floatingToolbar = new QWidget();
    m_floatingToolbar->setObjectName("floatingToolbar");
    m_floatingToolbar->setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);
    m_floatingToolbar->setAttribute(Qt::WA_TranslucentBackground);
    m_floatingToolbar->setStyleSheet("background-color: white; border: 1px solid #e0e0e0; border-radius: 8px;");
    m_floatingToolbar->setMinimumWidth(400);
    m_floatingToolbar->setMaximumWidth(800);
    m_floatingToolbar->setFixedHeight(47);

    QHBoxLayout *toolbarLayout = new QHBoxLayout(m_floatingToolbar);
    toolbarLayout->setContentsMargins(16, 0, 16, 0);
    toolbarLayout->setSpacing(8);
    toolbarLayout->addStretch();

    toolbarLayout->addWidget(createBottomToolBar());
    toolbarLayout->addStretch();

    m_floatingToolbar->installEventFilter(this);
    m_floatingToolbar->adjustSize();
    m_floatingToolbar->show();
    QTimer::singleShot(100, this, [this]() { repositionFloatingToolbar(); });
}

void MainWindow::repositionFloatingToolbar()
{
    if (!m_floatingToolbar || !m_3dViewArea) return;
    if (!m_floatingToolbar->isVisible()) return;
    m_floatingToolbar->adjustSize();
    // 基于整个显示窗口(view3DArea)居中，而不是只基于 m_3dView
    QRect areaGeo = m_3dViewArea->geometry();
    QPoint areaBottomCenter = m_3dViewArea->mapToGlobal(QPoint(areaGeo.width() / 2, areaGeo.height()));
    int tbX = areaBottomCenter.x() - m_floatingToolbar->width() / 2;
    int tbY = areaBottomCenter.y() - m_floatingToolbar->height() - 10;
    m_floatingToolbar->move(tbX, tbY);
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    QTimer::singleShot(0, this, [this]() { repositionFloatingToolbar(); });
    if (m_banner) m_banner->setGeometry(0, 0, width(), 56);   // P-3 横幅全宽贴顶
}

// ============================================================================
// P-3 顶部大字横幅（261002 按键域 §3.4 远距可读）：顶部全宽条带、半透明底、
// 高对比大字（≤10 字短语）、1.8s 自动消失（新顶旧）；红＝失败/急停/确认，绿白
// ＝中性。菜单期常驻（p1=110 驱动）：当前游标项/①⑤子态；瞬态插入显示、结束
// 自动恢复常驻。P8 分源：本横幅只吃 08 事件（＝按键来源）；UI 控件操作走
// statusBar 本地提示不来此
// ============================================================================
void MainWindow::showBanner(const QString& text, bool danger)
{
    if (!m_banner) {
        m_banner = new QLabel(this);
        m_banner->setAttribute(Qt::WA_TransparentForMouseEvents);  // 点击穿透——不挡关闭按钮
        m_banner->setAlignment(Qt::AlignCenter);
        m_banner->setWordWrap(false);
        m_bannerTimer = new QTimer(this);
        m_bannerTimer->setSingleShot(true);
        m_bannerTimer->setInterval(10000);   // 261003 用户口径：停留 10 秒
        connect(m_bannerTimer, &QTimer::timeout, this, [this]() {
            if (m_bannerPersist.isEmpty()) m_banner->hide();  // 瞬态 10s 隐藏；常驻（菜单/子态）等确认/退出
        });
    }
    m_banner->setStyleSheet(danger
        ? QStringLiteral("QLabel { background-color: rgba(192,57,43,0.88); color: white;"
                         " font-size: 26px; font-weight: bold; border: none; }")
        : QStringLiteral("QLabel { background-color: rgba(39,174,96,0.85); color: white;"
                         " font-size: 26px; font-weight: bold; border: none; }"));
    m_banner->setText(text);
    m_banner->setGeometry(0, 0, width(), 56);
    m_banner->raise();
    m_banner->show();
    m_bannerTimer->start();
}

// ============================================================================
// P-菜单弹窗（独立置顶窗口）：进菜单弹出、退菜单关闭、游标实时高亮＋子态提示
// 随 p1=110 事件（菜单变化）驱动刷新
// ============================================================================
void MainWindow::refreshMenuDialog() {
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (!dm) return;
    const auto ms = dm->menuState();
    using Sub = Scanner::device::MenuState::Substate;

    // 主界面＝关闭弹窗
    if (ms.layer != 2) {
        if (m_menuDlg) m_menuDlg->hide();
        return;
    }

    // 懒建弹窗（首次进菜单时创建独立置顶窗）
    if (!m_menuDlg) {
        m_menuDlg = new QDialog(nullptr);
        m_menuDlg->setWindowFlag(Qt::WindowStaysOnTopHint, true);
        m_menuDlg->setWindowTitle(QStringLiteral("菜单"));
        m_menuDlg->setModal(false);
        m_menuDlg->setMinimumSize(320, 300);
        auto* lay = new QVBoxLayout(m_menuDlg);
        lay->setContentsMargins(20, 16, 20, 12);
        lay->setSpacing(4);
        static const char* kNames[5] = {
            "① 分辨率设置", "② 进入就绪", "③ 扫描完成", "④ 后处理", "⑤ 重置"
        };
        for (int i = 0; i < 5; ++i) {
            m_menuItems[i] = new QLabel(QString::fromUtf8(kNames[i]), m_menuDlg);
            m_menuItems[i]->setMinimumHeight(36);
            m_menuItems[i]->setAlignment(Qt::AlignCenter);
            lay->addWidget(m_menuItems[i]);
        }
        m_menuSubLbl = new QLabel(m_menuDlg);
        m_menuSubLbl->setAlignment(Qt::AlignCenter);
        m_menuSubLbl->setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;");
        lay->addWidget(m_menuSubLbl);
        auto* hint = new QLabel(
            QStringLiteral("L/R 移动游标 · M 选中 · U 退出"), m_menuDlg);
        hint->setAlignment(Qt::AlignCenter);
        hint->setStyleSheet("color: #888; font-size: 12px;");
        lay->addWidget(hint);
    }

    // 五项样式：当前游标高亮蓝底白字，其余白底黑字
    for (int i = 0; i < 5; ++i) {
        const bool cur = (ms.cursor == i + 1);
        m_menuItems[i]->setStyleSheet(cur
            ? QStringLiteral(
                "background-color: #2980B9; color: white; font-size: 18px;"
                " font-weight: bold; border-radius: 6px;")
            : QStringLiteral(
                "background-color: #F5F5F5; color: #333; font-size: 16px;"
                " border-radius: 6px;"));
    }

    // 子态提示
    if (ms.substate == Sub::AdjustVoxel) {
        m_menuSubLbl->setText(QStringLiteral("分辨率 %1mm（L/R 调 · M 确认）")
            .arg(dm->voxelLadderValue(), 0, 'f', 2));
        m_menuSubLbl->setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #27AE60; padding: 8px;");
    } else if (ms.substate == Sub::ConfirmReset) {
        m_menuSubLbl->setText(QStringLiteral("再按 M 确认重置（其他键取消）"));
        m_menuSubLbl->setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #C0392B; padding: 8px;");
    } else {
        m_menuSubLbl->setText(QString());
    }

    m_menuDlg->show();
    m_menuDlg->raise();
    m_menuDlg->activateWindow();
}

// 常驻行重建（菜单变化/子态/①档位变化后调）：菜单期显示游标项，①⑤子态显
// 子态文案（⑤红底）；主界面＝无常驻（瞬态自然消失）
void MainWindow::refreshBannerPersistent()
{
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (!dm) return;
    if (!m_banner) {
        // 懒创建补齐（菜单首次进入时可能尚未触发过任何横幅事件——m_banner 为 null
        // 导致菜单变化被静默吞掉，用户「点菜单没弹出」根因）
        showBanner(QStringLiteral("…"), false);
        if (!m_banner) return;
        if (!m_bannerTimer) return;   // showBanner 内已建 timer
    }
    const auto ms = dm->menuState();
    using Sub = Scanner::device::MenuState::Substate;
    if (ms.layer != 2) {
        m_bannerPersist.clear();
        if (m_banner) m_banner->hide();   // 退菜单/确认后隐藏
        return;
    }
    static const char* kItems[5] = {"① 分辨率设置", "② 进入就绪", "③ 扫描完成",
                                    "④ 后处理", "⑤ 重置"};
    if (ms.substate == Sub::AdjustVoxel) {
        m_bannerPersist = QStringLiteral("分辨率 %1mm（L/R 调 M 确认）")
                              .arg(dm->voxelLadderValue(), 0, 'f', 2);
    } else if (ms.substate == Sub::ConfirmReset) {
        m_bannerPersist = QStringLiteral("再按 M 键确认重置（其他键取消）");
    } else {
        m_bannerPersist = QStringLiteral("菜单 ▸ %1")
                              .arg(QString::fromUtf8(kItems[ms.cursor - 1]));
    }
    // 常驻即时上屏（菜单期游标动/切档常驻行内刷新，不发瞬态——设计 §3.4）
    const bool danger = (ms.substate == Sub::ConfirmReset);
    m_banner->setStyleSheet(danger
        ? QStringLiteral("QLabel { background-color: rgba(192,57,43,0.88); color: white;"
                         " font-size: 26px; font-weight: bold; border: none; }")
        : QStringLiteral("QLabel { background-color: rgba(68,108,179,0.85); color: white;"
                         " font-size: 26px; font-weight: bold; border: none; }"));
    m_banner->setText(m_bannerPersist);
    m_banner->setGeometry(0, 0, width(), 56);
    m_banner->raise();
    m_banner->show();
    m_bannerTimer->stop();                          // 常驻不过期
}

QWidget *MainWindow::createTitleBar()
{
    QWidget *bar = new QWidget();
    bar->setObjectName("titleBar");
    bar->setMinimumHeight(28);
    bar->setMaximumHeight(36);
    QHBoxLayout *layout = new QHBoxLayout(bar);
    layout->setContentsMargins(8, 0, 0, 0);
    layout->setSpacing(0);

    QPushButton *btnLogo = new QPushButton();
    btnLogo->setObjectName("btnLogo");
    btnLogo->setFixedSize(28, 28);
    btnLogo->setIcon(QIcon(renderSvg(":/icons/resources/icons/icon/firstandsecond/trace-black-11.svg", 18)));
    btnLogo->setIconSize(QSize(18, 18));
    layout->addWidget(btnLogo);

    QPushButton *btnPrev = new QPushButton();
    btnPrev->setObjectName("btnTitleAction");
    btnPrev->setFixedSize(28, 28);
    btnPrev->setIcon(QIcon(renderSvg(":/icons/resources/icons/icon/firstandsecond/save-red-13.svg", 14)));
    btnPrev->setIconSize(QSize(14, 14));
    layout->addWidget(btnPrev);

    QPushButton *btnLast = new QPushButton();
    btnLast->setObjectName("btnTitleAction");
    btnLast->setFixedSize(28, 28);
    btnLast->setIcon(QIcon(renderSvg(":/icons/resources/icons/icon/firstandsecond/last-red-13.svg", 14)));
    btnLast->setIconSize(QSize(14, 14));
    layout->addWidget(btnLast);

    QPushButton* btnNext = new QPushButton();
    btnNext->setObjectName("btnTitleAction");
    btnNext->setFixedSize(28, 28);
    btnNext->setIcon(QIcon(renderSvg(":/icons/resources/icons/icon/firstandsecond/next-red-13.svg", 14)));
    btnNext->setIconSize(QSize(14, 14));
    layout->addWidget(btnNext);

    layout->addStretch();

    m_projectName = new QLabel(QStringLiteral("工程001_Turbine_Blade - V2.0.4 PRO"));
    m_projectName->setObjectName("projectNameLabel");
    m_projectName->setAlignment(Qt::AlignCenter);
    layout->addWidget(m_projectName);

    layout->addStretch();

    QPushButton *btnSave = new QPushButton();
    btnSave->setObjectName("btnTitleAction");
    btnSave->setFixedSize(40, 36);
    btnSave->setIcon(QIcon(renderSvg(":/icons/resources/icons/保存-黑-13.svg", 14)));
    btnSave->setIconSize(QSize(14, 14));
    layout->addWidget(btnSave);

    QPushButton *btnMin = new QPushButton();
    btnMin->setObjectName("btnWindowControl");
    btnMin->setFixedSize(40, 36);
    btnMin->setText(QStringLiteral("--"));
    connect(btnMin, &QPushButton::clicked, this, &QWidget::showMinimized);
    layout->addWidget(btnMin);

    QPushButton *btnClose = new QPushButton();
    btnClose->setObjectName("btnWindowControl");
    btnClose->setFixedSize(40, 36);
    btnClose->setText("X");
    connect(btnClose, &QPushButton::clicked, this, &QWidget::close);
    layout->addWidget(btnClose);

    QString titleBtnStyle = "QPushButton { background-color: transparent; border: none; }"
                            "QPushButton:hover { background-color: rgba(0,0,0,0.05); }";
    for (int i = 0; i < layout->count(); ++i) {
        QPushButton *btn = qobject_cast<QPushButton*>(layout->itemAt(i)->widget());
        if (btn) btn->setStyleSheet(titleBtnStyle);
    }

    return bar;
}

QWidget *MainWindow::createNavBar()
{
    QWidget *bar = new QWidget();
    bar->setObjectName("navBar");
    bar->setMinimumHeight(32);
    bar->setMaximumHeight(42);
    bar->setStyleSheet("background-color: #8B1A2B; color: white;");
    QHBoxLayout *layout = new QHBoxLayout(bar);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    QWidget *leftGroup = new QWidget();
    leftGroup->setMinimumWidth(200);
    leftGroup->setStyleSheet("border: none;");
    QHBoxLayout *leftLayout = new QHBoxLayout(leftGroup);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(0);

    QPushButton *btnFile = new QPushButton(QStringLiteral("菜单"));
    btnFile->setObjectName("navButton");
    btnFile->setFixedHeight(42);
    btnFile->setFixedWidth(50);
    leftLayout->addWidget(btnFile);
    m_navLeftButtons.append(btnFile);

    QPushButton *btnScan = new QPushButton(QStringLiteral("扫描"));
    btnScan->setObjectName("navButtonActive");
    btnScan->setFixedHeight(42);
    btnScan->setFixedWidth(50);
    leftLayout->addWidget(btnScan);
    m_navLeftButtons.append(btnScan);
    connect(btnScan, &QPushButton::clicked, this, &MainWindow::onScanClicked);

    QPushButton *btnManage = new QPushButton(QStringLiteral("管理"));
    btnManage->setObjectName("navButton");
    btnManage->setFixedHeight(42);
    btnManage->setFixedWidth(50);
    leftLayout->addWidget(btnManage);
    m_navLeftButtons.append(btnManage);

    QPushButton *btnIntegrateTest = new QPushButton(QStringLiteral("集成测试"));
    btnIntegrateTest->setObjectName("navButton");
    btnIntegrateTest->setFixedHeight(42);
    btnIntegrateTest->setFixedWidth(75);
    leftLayout->addWidget(btnIntegrateTest);
    m_navLeftButtons.append(btnIntegrateTest);
    connect(btnIntegrateTest, &QPushButton::clicked, this, &MainWindow::onIntegrateTestClicked);

    QPushButton *btnReloadCloud = new QPushButton(QStringLiteral("加载点云"));
    btnReloadCloud->setObjectName("navButton");
    btnReloadCloud->setFixedHeight(42);
    btnReloadCloud->setFixedWidth(75);
    leftLayout->addWidget(btnReloadCloud);
    m_navLeftButtons.append(btnReloadCloud);
    connect(btnReloadCloud, &QPushButton::clicked, this, &MainWindow::onReloadPointCloud);

    // 虚拟按键表盘（261002 临时测试机·无实体键）：弹窗模拟扫描仪面板五键
    QPushButton *btnVKeys = new QPushButton(QStringLiteral("⌨ 按键"));
    btnVKeys->setObjectName("navButton");
    btnVKeys->setFixedHeight(42);
    btnVKeys->setFixedWidth(64);
    btnVKeys->setToolTip(QStringLiteral(
        "虚拟按键表盘（261002 临时测试机）：模拟扫描仪面板按键，G01 注入与真机同链路"));
    leftLayout->addWidget(btnVKeys);
    m_navLeftButtons.append(btnVKeys);
    connect(btnVKeys, &QPushButton::clicked, this, &MainWindow::showVirtualKeypad);

    layout->addWidget(leftGroup);
    layout->addStretch();

    QWidget *rightGroup = new QWidget();
    rightGroup->setMinimumWidth(300);
    rightGroup->setStyleSheet("border: none;");
    QHBoxLayout *rightLayout = new QHBoxLayout(rightGroup);
    rightLayout->setContentsMargins(-10, 0, 0, 0);
    rightLayout->setSpacing(0);

    QStringList scanModes = {
        QStringLiteral("手持扫描"), QStringLiteral("跟踪扫描"),
        QStringLiteral("摄影测量"), QStringLiteral("自动扫描"),
        QStringLiteral("检测分析")
    };

    QStringList scanModeIcons = {
        "handlescan-white-11", "trace-white-11",
        "photo-white-11", "auto-white-11",
        "analysis-white-11"
    };

    for (int i = 0; i < scanModes.size(); ++i) {
        QPushButton *btn = new QPushButton();
        btn->setObjectName("scanModeButton");
        btn->setFixedHeight(42);
        btn->setFixedWidth(84);

        QHBoxLayout *btnLayout = new QHBoxLayout(btn);
        btnLayout->setContentsMargins(6, 0, 6, 0);
        btnLayout->setSpacing(4);
        btnLayout->setAlignment(Qt::AlignCenter);

        QLabel *iconLbl = new QLabel();
        iconLbl->setStyleSheet("border: none; background: transparent;");
        int iconSize = 14;
        iconLbl->setPixmap(renderSvg(QString(":/icons/resources/icons/icon/firstandsecond/%1.svg").arg(scanModeIcons[i]), iconSize));
        iconLbl->setFixedSize(iconSize, iconSize);
        btnLayout->addWidget(iconLbl);

        QLabel *textLbl = new QLabel(scanModes[i]);
        textLbl->setObjectName("scanModeText");
        textLbl->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        btnLayout->addWidget(textLbl);

        rightLayout->addWidget(btn);
        m_navRightButtons.append(btn);
    }

    layout->addWidget(rightGroup);
    return bar;
}

QWidget *MainWindow::createToolBar()
{
    QWidget *bar = new QWidget();
    bar->setObjectName("toolBar");
    bar->setMinimumHeight(40);
    bar->setMaximumHeight(56);
    QHBoxLayout *layout = new QHBoxLayout(bar);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(32);

    struct ToolItem { QString icon; QString text; };
    // 扫描四模式（协议批3·D7；精细/深孔 2026-09-10 用户纠正对调）：标点(2)=
    // MarkerOnly / 面片(3)=MarkerPlusLaser / 精细(4)=FineScan（原「点云扫描」改名，
    // D 管）/ 深孔(5)=DeepHoleScan（新增，C 管）；深孔 icon 复用 cloudscan
    // （不新增美术资源，专属图标待设计）
    QList<ToolItem> items = {
        {"filemanager-black-14", QStringLiteral("文件管理")},
        {"equipcalib-black-14", QStringLiteral("校准设备")},
        {"markscan-black-14", QStringLiteral("标点扫描")},
        {"meshscan-black-14", QStringLiteral("面片扫描")},
        {"cloudscan-black-14", QStringLiteral("精细扫描")},
        {"cloudscan-black-14", QStringLiteral("深孔扫描")},
        {"both-black-14", QStringLiteral("正反扫描")},
        {"slicerscan-black-14", QStringLiteral("切面扫描")},
        {"projmanager-black-14", QStringLiteral("重置项目")}
    };

    for (int i = 0; i < items.size(); ++i) {
        QPushButton *btn = new QPushButton();
        btn->setObjectName("toolButton");
        btn->setFixedSize(64, 56);
        QVBoxLayout *btnLayout = new QVBoxLayout(btn);
        btnLayout->setContentsMargins(0, 4, 0, 4);
        btnLayout->setSpacing(2);
        btnLayout->setAlignment(Qt::AlignCenter);

        QLabel *iconLbl = new QLabel();
        int iconSize = 20;
        iconLbl->setPixmap(renderSvg(QString(":/icons/resources/icons/icon/third/%1.svg").arg(items[i].icon), iconSize));
        iconLbl->setFixedSize(iconSize, iconSize);
        btnLayout->addWidget(iconLbl, 0, Qt::AlignHCenter);

        QLabel *textLbl = new QLabel(items[i].text);
        textLbl->setObjectName("toolButtonText");
        textLbl->setAlignment(Qt::AlignCenter);
        textLbl->setMinimumWidth(btn->width());
        btnLayout->addWidget(textLbl);

        if (i == 0) btn->setProperty("active", true);
        btn->setStyleSheet("QPushButton { background-color: transparent; border: none; }"
                           "QPushButton:hover { background-color: rgba(0,0,0,0.05); }");
        layout->addWidget(btn);
        m_toolButtons.append(btn);

        // 精细/深孔键隐藏（260927 用户口径）：两模式由设备按键（M 双击循环）调节，
        // 软件界面不体现——钮与接线保留（同悬浮条操作组先例），仅 hide
        if (i == 4 || i == 5) btn->hide();

        if (i == 1) {
            connect(btn, &QPushButton::clicked, this, &MainWindow::onCalibDeviceClicked);
        }

        // 扫描四模式键（i==2 标点/3 面片/4 精细/5 深孔）：各自独立启停——点哪
        // 键哪键变红（其他键不动）；扫描中点另一键＝停旧启新（模式切换）
        if (i >= 2 && i <= 5) {
            const int myIdx = i;
            const QString title = items[i].text;
            const auto mode = (i == 2) ? Scanner::ScanMode::MarkerOnly
                          : (i == 3) ? Scanner::ScanMode::MarkerPlusLaser
                          : (i == 4) ? Scanner::ScanMode::FineScan
                                     : Scanner::ScanMode::DeepHoleScan;
            connect(btn, &QPushButton::clicked, this, [this, myIdx, title, mode]() {
                if (!m_appCtx) return;
                // —— 编辑成果物理化（P4b 2026-09-06）：就绪态续采/完成前，把悬浮
                //    工具栏的显示级删除（alpha=0）路由到真账本——融合云移除＋obs
                //    剔除，编辑从此真实有效（非显示级掩盖）——
                auto materializeEdits = [this]() {
                    if (!m_3dView) return;
                    const auto pending = m_3dView->pendingMarkerDeleteIndices();
                    if (!pending.empty()) {
                        auto* sw = m_appCtx ? m_appCtx->scanWorkflow() : nullptr;
                        auto* fuse = sw ? sw->markerFuse() : nullptr;
                        auto* obs = sw ? sw->obsAccumulator() : nullptr;
                        if (!fuse || !obs) return;
                        // 融合云物理移除（越界批原子——快照与云间下标漂移时整批不动）
                        const auto st = fuse->removePoints(pending);
                        if (!st.success) {
                            JMW_LOG_WARN("app-MainWindow", "[编辑物理化] 融合云移除失败: {}", st.message);
                            return;
                        }
                        // obs 剔除（下标＝globalId——快照构建时恒等）
                        obs->excludeMarkerObs(
                            std::vector<int>(pending.begin(), pending.end()), true);
                        m_3dView->clearPendingMarkerDeletes();
                        JMW_LOG_INFO("app-MainWindow", "[编辑物理化] {} 点真删完成（融合云+obs）",
                                     pending.size());
                    }
                    // —— 激光侧（05 P4b）：显示级删除 → 09 融合云物理移除 ——
                    const auto pendingCloud = m_3dView->pendingCloudDeleteIndices();
                    if (!pendingCloud.empty()) {
                        auto* sw = m_appCtx ? m_appCtx->scanWorkflow() : nullptr;
#ifdef JMW_BUILD_CUDA
                        auto* lfuse = sw ? sw->laserFuse() : nullptr;
#else
                        auto* lfuse = static_cast<Scanner::pipeline::ILaserFuse*>(nullptr);
#endif
                        if (lfuse) {
                            const auto stc = lfuse->removePoints(pendingCloud);
                            if (stc.success) {
                                m_3dView->clearPendingCloudDeletes();
                                JMW_LOG_INFO("app-MainWindow",
                                             "[编辑物理化·激光] {} 点真删完成（融合云）",
                                             pendingCloud.size());
                            } else {
                                JMW_LOG_WARN("app-MainWindow",
                                             "[编辑物理化·激光] 融合云移除失败: {}", stc.message);
                            }
                        }
                    }
                };
                if (m_appCtx->isScanSessionActive()) {
                    // —— 关闭会话（260927 用户口径）：点模式键＝完成当前会话——
                    //    finish_scan 后台链含 GBA 终局批（全局优化）＋合账落库；
                    //    编辑成果先物理化。本键＝纯关闭（面片扫描点面片键即触发
                    //    全局优化）；他键＝关闭旧会话后落下启动新模式。
                    //    （原「本键暂停/续采」路径退役——采集启停归设备 M 键：
                    //    arm 就绪→M 开扫→M 停采→点键关闭+GBA，闭环）
                    const bool wasSelf = (m_activeScanToolIdx == myIdx);
                    materializeEdits();
                    auto sr = m_appCtx->stopScanSession();
                    if (m_activeScanToolIdx >= 0) setScanButtonVisual(m_activeScanToolIdx, false);
                    m_activeScanToolIdx = -1;
                    if (sr.success) {
                        // 260927 弹窗与进度回调解耦：关闭即弹「全局优化中」——GBA 降级
                        // 秒完/无进度回调（真机迭代=0 终局遍 195ms）时用户亦有可见反馈；
                        // 完成关闭归 endedHandler 兜底/100% 路径
                        if (!m_finalBADlg) {
                            m_finalBADlg = new QProgressDialog(
                                QStringLiteral("全局优化中，请稍候……"), QString(), 0, 100, this);
                            m_finalBADlg->setWindowTitle(QStringLiteral("全局优化"));
                            m_finalBADlg->setWindowModality(Qt::ApplicationModal);
                            m_finalBADlg->setMinimumDuration(0);
                            m_finalBADlg->setAutoClose(false);
                            m_finalBADlg->show();
                        }
                        statusBar()->showMessage(
                            QStringLiteral("会话已关闭——全局优化（GBA）后台执行中"));
                    } else {
                        statusBar()->showMessage(
                            QString::fromStdString("关闭被拒: " + sr.message));
                    }
                    if (wasSelf || !sr.success) return;   // 本键＝纯关闭；他键落下启新
                }
                if (mode == Scanner::ScanMode::MarkerOnly)
                    applyMarkerPreset();       // 标点：推荐预设＋旋钮同步（260912）
                else if (mode == Scanner::ScanMode::MarkerPlusLaser)
                    applyMeshPreset();         // 面片：推荐预设＋旋钮同步（260912）
                // 精细/深孔：P-5 收敛后无独立预设（保持当前亮度档——待真机标定预设）
                m_laserSessionLatched = false; // 新会话：激光仓库基线待重锁（260912c）
                // 260927 就绪流程（用户口径）：UI 模式键只备会话（模式/帧流/工作流），
                // 不启采——正式开扫由设备 M 键触发（captureToggle→N10 四管掩码）
                if (m_appCtx->deviceManager())
                    m_appCtx->deviceManager()->setCalibCaptureArmed(false);   // 扫描会话撤标定布防
                const auto r = m_appCtx->armScanSession(mode);
                if (!r.success) {
                    QMessageBox::warning(this, title,
                        QString::fromStdString("扫描就绪被拒:\n" + r.message));
                } else {
                    setScanButtonVisual(myIdx, true);          // 只有本键变红
                    m_activeScanToolIdx = myIdx;
                    m_sessionSimExtract = m_appCtx->simExtract();   // 会话开关基线（续采变体检测）
                    // 工程树：首个扫描会话建「标记点 001」，后续会话（标点/面片/
                    // 精细/深孔）复用同一节点不新建（有对应类型即可——用户口径 2026-09-05）；
                    // 计数实时刷新；激光点数据由「点云数据 001」承载（合账落库
                    // 后 cloudTimer 更新计数）
                    if (!m_markerCurrentItem && m_markerRootItem) {
                        m_markerScanSeq = 1;
                        auto* item = new QTreeWidgetItem(
                            m_markerRootItem,
                            QStringList() << QStringLiteral("标记点 %1")
                                                 .arg(m_markerScanSeq, 3, 10, QChar('0')));
                        item->setIcon(0, QIcon(renderSvg(
                            ":/icons/resources/icons/icon/left/marklist-black-11.svg", 11)));
                        m_markerCurrentItem = item;
                        m_markerRootItem->setExpanded(true);
                    }
                    showCameraMonitor();                       // 调试：弹相机左右图监视
                    showScanReadyPrompt(title, myIdx);         // 就绪窗口：等设备 M 键
                }
            });
        }

        // 文件管理：弹出导入/导出菜单
        if (i == 0) {
            connect(btn, &QPushButton::clicked, this, [this, btn]() {
                QMenu menu(btn);
                menu.setStyleSheet("QMenu { background: white; border: 1px solid #d0d0d0; }"
                                   "QMenu::item { padding: 6px 24px; }"
                                   "QMenu::item:selected { background: #e0e0e0; }");

                menu.addAction(QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe6\xa0\x87\xe5\xbf\x97\xe7\x82\xb9"), [this]() {
                    QString path = QFileDialog::getOpenFileName(this, QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe6\xa0\x87\xe5\xbf\x97\xe7\x82\xb9"), "",
                        "Marker/PLY (*.json *.txt *.ply);;All Files (*.*)");
                    if (path.isEmpty()) return;
                    std::string spath = path.toStdString();
                    std::vector<cv::Point3f> markers;
                    if (Scanner::data::fileio::importMarkers(spath, markers)) {
                        statusBar()->showMessage(QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe6\xa0\x87\xe5\xbf\x97\xe7\x82\xb9 %1 \xe4\xb8\xaa").arg(markers.size()));
                        // 上屏：定向圆盘（世界系固定朝向——文件无法线，圆盘默认
                        // 朝上 +Z，不随视角转）＋最优取景（PCA 面视+视场填充）
                        if (m_3dView && !markers.empty()) {
                            std::vector<osg::Vec3> pts;
                            pts.reserve(markers.size());
                            for (const auto& p : markers) pts.emplace_back(p.x, p.y, p.z);
                            m_3dView->loadMarkerPoints(pts, std::vector<osg::Vec3>());
                            m_3dView->fitCloudOptimal(pts);
                        }
                    }
                    else
                        statusBar()->showMessage(QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe5\xa4\xb1\xe8\xb4\xa5"));
                });
                menu.addAction(QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe7\x82\xb9\xe4\xba\x91"), [this]() {
                    QString path = QFileDialog::getOpenFileName(this, QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe7\x82\xb9\xe4\xba\x91"), "", "Point Cloud (*.ply *.pcd *.xyz *.txt);;All Files (*.*)");
                    if (path.isEmpty()) return;
                    QByteArray ba = path.toUtf8();
                    std::string spath(ba.constData(), ba.size());

                    QProgressDialog progress(QStringLiteral("Importing point cloud..."), QString(), 0, 0, this);
                    progress.setWindowModality(Qt::WindowModal);
                    progress.setMinimumDuration(0);
                    progress.setCancelButton(nullptr);
                    progress.setRange(0, 0);
                    progress.setAutoClose(false);
                    progress.setAutoReset(false);
                    progress.show();
                    QApplication::processEvents();

                    std::vector<cv::Point3f> rawPts;
                    bool ok = Scanner::data::fileio::importPointCloud(spath, rawPts);
                    // 模拟提取 stash（「模拟数据」开关的激光数据源——2026-09-13
                    // 模拟调通；普通导入路径无副作用，仅值拷贝一份）
                    if (m_appCtx && ok && !rawPts.empty())
                        m_appCtx->setLastImportedCloud(rawPts);
                    std::vector<osg::Vec3> points;   // UI 容器仍用 osg::Vec3，就地转换
                    points.reserve(rawPts.size());
                    for (const auto& p : rawPts) points.emplace_back(p.x, p.y, p.z);

                    // 诊断日志写文件
                    FILE* logf = fopen("E:/workfold/framework/build/import_debug.log", "a");
                    if (logf) {
                        fprintf(logf, "=== import ===\n");
                        fprintf(logf, "file: %s\n", spath.c_str());
                        fprintf(logf, "ok=%d points=%zu\n", ok?1:0, points.size());
                        if (!points.empty()) {
                            float minx=1e30,miny=1e30,minz=1e30,maxx=-1e30,maxy=-1e30,maxz=-1e30;
                            for (const auto& p : points) {
                                if (p.x()<minx) minx=p.x(); if (p.x()>maxx) maxx=p.x();
                                if (p.y()<miny) miny=p.y(); if (p.y()>maxy) maxy=p.y();
                                if (p.z()<minz) minz=p.z(); if (p.z()>maxz) maxz=p.z();
                            }
                            fprintf(logf, "bbox: x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]\n",
                                    minx,maxx,miny,maxy,minz,maxz);
                            fprintf(logf, "first5: (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f)\n",
                                    points[0].x(),points[0].y(),points[0].z(),
                                    points[1].x(),points[1].y(),points[1].z(),
                                    points[2].x(),points[2].y(),points[2].z(),
                                    points[3].x(),points[3].y(),points[3].z(),
                                    points[4].x(),points[4].y(),points[4].z());
                        }
                        fprintf(logf, "m_3dView=%p\n", (void*)m_3dView);
                        fclose(logf);
                    }

                    if (ok && !points.empty()) {
                        statusBar()->showMessage(QStringLiteral("Imported %1 points").arg(points.size()));
                        // 260927 导入计数上树：导入云不入扫描仓库（pcb），树节点原只显
                        // 仓库计数＝恒 0；cloudTimer 仅在仓库计数变化时改写——此处直接
                        // 设文本可存活到下次扫描出点
                        m_importedCloudCount = points.size();
                        if (m_cloudItem001)
                            m_cloudItem001->setText(0, QStringLiteral("点云数据 001 (%1)")
                                                        .arg(points.size()));
                        if (m_3dView) {
                            m_3dView->setCenterOverlayVisible(false);
                            m_3dView->loadPointCloud(points);
                            m_3dView->update();
                        }
                    } else {
                        statusBar()->showMessage("Import failed");
                        QMessageBox::warning(this, "Import", "Failed to read point cloud. Check file format.");
                    }
                    progress.close();
                });
                menu.addAction(QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe7\xbd\x91\xe6\xa0\xbc"), [this]() {
                    QString path = QFileDialog::getOpenFileName(this, QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe7\xbd\x91\xe6\xa0\xbc"), "", "Mesh (*.stl *.obj);;All Files (*.*)");
                    if (path.isEmpty()) return;
                    if (m_3dView) {
                        QProgressDialog progress(QStringLiteral("Importing mesh..."), QString(), 0, 0, this);
                        progress.setWindowModality(Qt::WindowModal);
                        progress.setMinimumDuration(0);
                        progress.setCancelButton(nullptr);
                        progress.setRange(0, 0);
                        progress.setAutoClose(false);
                        progress.setAutoReset(false);
                        progress.show();
                        QApplication::processEvents();

                        m_3dView->clearScene();
                        if (m_3dView->loadMesh(path))
                            statusBar()->showMessage(QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe7\xbd\x91\xe6\xa0\xbc\xe6\x88\x90\xe5\x8a\x9f: ") + path);
                        else {
                            statusBar()->showMessage(QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe7\xbd\x91\xe6\xa0\xbc\xe5\xa4\xb1\xe8\xb4\xa5"));
                            QMessageBox::warning(this, "Import", "Failed to read mesh.");
                        }
                        m_3dView->update();
                        progress.close();
                    }
                });
                menu.addAction(QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe5\xb7\xa5\xe7\xa8\x8b\xe6\x96\x87\xe4\xbb\xb6"), [this]() {
                    QString path = QFileDialog::getOpenFileName(this, QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5\xe5\xb7\xa5\xe7\xa8\x8b\xe6\x96\x87\xe4\xbb\xb6"), "", "Project (*.leadscan)");
                    if (!path.isEmpty()) statusBar()->showMessage(QStringLiteral("\xe5\xaf\xbc\xe5\x85\xa5: ") + path);
                });

                menu.addSeparator();

                menu.addAction(QStringLiteral("\xe5\xaf\xbc\xe5\x87\xba\xe6\xa0\x87\xe5\xbf\x97\xe7\x82\xb9"), [this]() {
                    QString path = QFileDialog::getSaveFileName(this, QStringLiteral("\xe5\xaf\xbc\xe5\x87\xba\xe6\xa0\x87\xe5\xbf\x97\xe7\x82\xb9"), "markers.json", "Marker Files (*.json *.txt)");
                    if (path.isEmpty()) return;
                    std::string spath = path.toStdString();
                    auto* pcb = m_appCtx ? m_appCtx->pointCloudBuffer() : nullptr;   // 仓库内存直导（02-D5 唯一出口）
                    if (pcb && pcb->exportMarkers(spath))
                        statusBar()->showMessage(QStringLiteral("\xe5\xaf\xbc\xe5\x87\xba\xe6\xa0\x87\xe5\xbf\x97\xe7\x82\xb9\xe6\x88\x90\xe5\x8a\x9f"));
                    else
                        statusBar()->showMessage(QStringLiteral("\xe5\xaf\xbc\xe5\x87\xba\xe5\xa4\xb1\xe8\xb4\xa5"));
                });
                menu.addAction(QStringLiteral("\xe5\xaf\xbc\xe5\x87\xba\xe7\x82\xb9\xe4\xba\x91"), [this]() {
                    QString path = QFileDialog::getSaveFileName(this, QStringLiteral("\xe5\xaf\xbc\xe5\x87\xba\xe7\x82\xb9\xe4\xba\x91"), "pointcloud.ply", "Point Cloud (*.ply *.pcd *.xyz)");
                    if (path.isEmpty()) return;
                    std::string spath = path.toStdString();
                    auto* pcb = m_appCtx ? m_appCtx->pointCloudBuffer() : nullptr;   // 仓库内存直导（02-D5 唯一出口）
                    // 260912 数量回显：导出成败带点数——0 点=喂入侧问题（看扫描中
                    // 「点云数据 001」是否增长）；N>0 而文件空/找不到=路径编码问题
                    const int nPts = pcb ? pcb->getTotalPointCount() : 0;
                    const bool ok = pcb && pcb->exportCloud(spath);
                    JMW_LOG_INFO("app-MainWindow", "[导出] 点云：{} 点 ok={} → {}",
                                 nPts, ok ? 1 : 0, spath);
                    if (ok)
                        statusBar()->showMessage(
                            QStringLiteral("导出点云成功（%1 点）").arg(nPts), 5000);
                    else
                        statusBar()->showMessage(
                            QStringLiteral("导出点云失败（仓库 %1 点）").arg(nPts), 5000);
                });
                menu.addAction(QStringLiteral("\xe5\x90\x8e\xe5\xa4\x84\xe7\x90\x86\xe5\xaf\xbc\xe5\x87\xba STL"), [this]() {
                    QString path = QFileDialog::getSaveFileName(this, QStringLiteral("\xe5\x90\x8e\xe5\xa4\x84\xe7\x90\x86\xe5\xaf\xbc\xe5\x87\xba\xe7\xbd\x91\xe6\xa0\xbc"), "mesh.stl", "Mesh (*.stl *.obj)");
                    if (path.isEmpty()) return;
                    // P1-1（260912 诚实化的兑现）：04/07-E 后处理真正点火——S2→S6
                    // 批算（法线重算→封装→补洞→光顺→边界）→ STL 写出 → 合账回 S2。
                    // 阶段 0 法线真算；网格四族 09 待建→当前产物为法线化点云（有棱无面）
                    if (!m_appCtx) return;
                    auto r = m_appCtx->startPostProcessSession(path.toStdString());
                    if (!r.success) {
                        QMessageBox::warning(this, QStringLiteral("后处理"),
                            QStringLiteral("后处理无法启动：%1").arg(QString::fromStdString(r.message)));
                        return;
                    }
                    statusBar()->showMessage(
                        QStringLiteral("后处理已启动：法线重算→封装→补洞→光顺→边界→STL 写出……"), 6000);
                    // 进度透传（阶段名+百分位；04 postThread_ 回调→队列内直更状态栏）
                    auto* pw = m_appCtx->postWorkflow();
                    if (pw) pw->setProgressCallback([this](const auto& p) {
                        const int pct = static_cast<int>(p.progress * 100.0f);
                        if (pct >= 100)
                            statusBar()->showMessage(QStringLiteral("后处理完成，STL 已导出。"), 6000);
                        else if (pct > 0)
                            statusBar()->showMessage(
                                QStringLiteral("后处理中[%1]：%2%……")
                                    .arg(QString::fromStdString(p.stageName)).arg(pct), 1500);
                    });
                });
                menu.addAction(QStringLiteral("\xe5\xaf\xbc\xe5\x87\xba\xe5\xb7\xa5\xe7\xa8\x8b\xe6\x96\x87\xe4\xbb\xb6"), [this]() {
                    // 260912 诚实化：原仅状态栏假提示；工程序列化格式规划中
                    QMessageBox::information(this, QStringLiteral("导出工程文件"),
                        QStringLiteral("工程文件导出未实现（序列化格式规划中）。"));
                });

                QPoint pos = btn->mapToGlobal(QPoint(btn->width() + 4, 0));
                menu.exec(pos);
            });
        }

        // 标点扫描/面片扫描/精细扫描/深孔扫描/正反扫描/切面扫描：切换回默认界面
        //（键位随深孔新增右移：扫描四模式 2-5，正反=6；正反保留切回行为）
        // 注意：i==0 是"文件管理/导入"按钮，只开菜单，不能在这里 clearScene（否则导入后被清空）
        if (i == 2 || i == 3 || i == 4 || i == 5 || i == 6) {
            connect(btn, &QPushButton::clicked, this, [this]() {
                // 仅从标定分屏切回时恢复（clearScene 会清扫描标志点——恢复
                // lambda 与扫描启停 lambda 同键先后无条件执行，曾致"停止后点
                // 消失/扫描中清建循环闪烁/HUD Y 轴上下跳（z-up home 与左相机
                // -y up 视角打架）"三症并发，2026-08-31）
                if (!m_calibBoard2D || !m_calibBoard2D->isVisible()) return;
                if (m_calibBoard2D) m_calibBoard2D->hide();
                // 隐藏左右、前后彩条
                auto* va = m_3dView ? m_3dView->parentWidget()->parentWidget() : nullptr;
                if (va) { auto* b = va->findChild<QWidget*>("calibLrBar"); if (b) b->hide(); }
                auto* vc = m_3dView ? m_3dView->parentWidget() : nullptr;
                if (vc) { auto* b = vc->findChild<QWidget*>("calibFbBar"); if (b) b->hide(); }
                if (m_3dView) {
                    m_3dView->clearScene();
                    m_3dView->setCenterOverlayVisible(true);
                    m_3dView->viewer()->getCamera()->setClearColor(osg::Vec4(0.412f, 0.412f, 0.412f, 1.0f));
                    auto* manip = new osgGA::TrackballManipulator();
                    m_3dView->setCameraManipulator(manip);
                    manip->home(0);
                }
                if (m_floatingToolbar) {
                    m_floatingToolbar->setVisible(true);
                    m_floatingToolbar->show();
                    repositionFloatingToolbar();
                }
            });
        }

        if (i == 0) {
            layout->addSpacing(16);
            QFrame *separator = new QFrame();
            separator->setFixedWidth(1);
            separator->setFixedHeight(36);
            separator->setStyleSheet("background-color: #C0C0C0; border: none;");
            layout->addWidget(separator);
            layout->addSpacing(16);
        }
        if (i == 5) {
            QFrame *separator = new QFrame();
            separator->setFixedWidth(1);
            separator->setFixedHeight(36);
            separator->setStyleSheet("background-color: #C0C0C0; border: none;");
            layout->addWidget(separator);
        }
    }

    // —— 「模拟数据」开关（调试件）：置位后下个扫描会话启用中段模拟提取——
    //    真机前端照常采集，每帧标志点/激光提取结果由模拟观测替换，配准（标志点
    //    vs 全局锚 R/T）→点云变换→体素融合→3D 显示全走生产代码（260917）
    QPushButton *simToggle = new QPushButton();
    simToggle->setObjectName("toolButton");
    simToggle->setFixedSize(64, 56);
    simToggle->setCheckable(true);
    simToggle->setToolTip(QStringLiteral("模拟数据：开启后扫描会话的中段提取（标志点/激光 3D）"
                                         "由模拟数据替换——配准/融合/显示走真实流水线"));
    QVBoxLayout *simLay = new QVBoxLayout(simToggle);
    simLay->setContentsMargins(0, 4, 0, 4);
    simLay->setSpacing(2);
    simLay->setAlignment(Qt::AlignCenter);
    QLabel *simIcon = new QLabel();
    simIcon->setPixmap(renderSvg(
        QStringLiteral(":/icons/resources/icons/icon/third/both-black-14.svg"), 20));
    simIcon->setFixedSize(20, 20);
    simLay->addWidget(simIcon, 0, Qt::AlignHCenter);
    QLabel *simText = new QLabel(QStringLiteral("模拟数据"));
    simText->setObjectName("toolButtonText");
    simText->setAlignment(Qt::AlignCenter);
    simText->setMinimumWidth(simToggle->width());
    simLay->addWidget(simText);
    simToggle->setStyleSheet(
        "QPushButton { background-color: transparent; border: none; }"
        "QPushButton:hover { background-color: rgba(0,0,0,0.05); }"
        "QPushButton:checked { background-color: rgba(230,145,18,0.18);"
        " border: 1px solid #E69112; border-radius: 4px; }");
    connect(simToggle, &QPushButton::toggled, this, [this](bool on) {
        if (!m_appCtx) return;
        m_appCtx->setSimExtract(on);
        JMW_LOG_INFO("app-MainWindow", "[模拟数据] 开关: {}（下个扫描会话生效）",
                     on ? "开" : "关");
        statusBar()->showMessage(on
            ? QStringLiteral("模拟数据：开——下个扫描会话启用中段模拟提取")
            : QStringLiteral("模拟数据：关——纯真机链"));
    });
    layout->addWidget(simToggle);

    layout->addStretch();
    return bar;
}

QWidget *MainWindow::createLeftPanel()
{
    QWidget *innerPanel = new QWidget();
    innerPanel->setObjectName("leftPanelInner");
    innerPanel->setMinimumWidth(160);
    QVBoxLayout *innerLayout = new QVBoxLayout(innerPanel);
    innerLayout->setContentsMargins(0, 0, 0, 0);
    innerLayout->setSpacing(0);

    innerLayout->addWidget(createProjectSection(), 0);
    innerLayout->addWidget(createParamSection(), 1);
    innerLayout->addWidget(createInfoSection(), 0);

    QScrollArea *scrollArea = new QScrollArea();
    scrollArea->setObjectName("leftPanelScroll");
    scrollArea->setWidget(innerPanel);
    scrollArea->setWidgetResizable(true);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea->setMinimumWidth(160);
    scrollArea->setStyleSheet(
        "QScrollArea#leftPanelScroll { border: none; background: transparent; }"
        "QScrollBar:vertical { width: 4px; background: transparent; }"
        "QScrollBar::handle:vertical { background: #888888; border-radius: 2px; min-height: 20px; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }"
    );
    return scrollArea;
}

QWidget *MainWindow::createProjectSection()
{
    QWidget *section = new QWidget();
    section->setObjectName("projectSection");
    QVBoxLayout *layout = new QVBoxLayout(section);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    QWidget *header = new QWidget();
    header->setFixedHeight(28);
    header->setStyleSheet("background-color: #E1E1E1; border-bottom: 1px solid #C0C0C0; margin: 0px; padding: 0px;");
    QHBoxLayout *headerLayout = new QHBoxLayout(header);
    headerLayout->setContentsMargins(8, 0, 8, 0);

    QLabel *iconLbl = new QLabel();
    iconLbl->setPixmap(renderSvg(":/icons/resources/icons/项目管理-黑-14.svg", 11));
    headerLayout->addWidget(iconLbl);

    QLabel *title = new QLabel(QStringLiteral("项目树形结构"));
    title->setObjectName("sectionTitle");
    headerLayout->addWidget(title);
    headerLayout->addStretch();
    layout->addWidget(header);

    m_projectTree = new QTreeWidget();
    m_projectTree->setObjectName("projectTree");
    m_projectTree->setHeaderHidden(false);
    m_projectTree->setHeaderLabel(QStringLiteral("当前工程"));
    QFont headerFont = m_projectTree->header()->font();
    headerFont.setBold(true);
    m_projectTree->header()->setFont(headerFont);
    m_projectTree->setMinimumHeight(60);
    m_projectTree->setIndentation(16);
    m_projectTree->setStyleSheet("QTreeWidget { border: none; margin: 0px; padding: 0px; } QTreeWidget::item { padding: 0px; margin: 0px; }");

    QTreeWidgetItem *markerRoot = new QTreeWidgetItem(QStringList() << QStringLiteral("标记点列表"));
    markerRoot->setIcon(0, QIcon(renderSvg(":/icons/resources/icons/icon/left/marklist-black-11.svg", 14)));
    m_projectTree->addTopLevelItem(markerRoot);
    m_markerRootItem = markerRoot;   // 扫描会话节点按实际启动动态创建（见扫描启动 lambda）

    QTreeWidgetItem *cloudRoot = new QTreeWidgetItem(QStringList() << QStringLiteral("点云/三角面列表"));
    cloudRoot->setIcon(0, QIcon(renderSvg(":/icons/resources/icons/icon/left/cloudlist-black-11.svg", 14)));
    m_cloudItem001 = new QTreeWidgetItem(cloudRoot, QStringList() << QStringLiteral("点云数据 001"));
    m_cloudItem001->setIcon(0, QIcon(renderSvg(":/icons/resources/icons/icon/left/cloudlist-black-11.svg", 11)));
    QTreeWidgetItem *c2 = new QTreeWidgetItem(cloudRoot, QStringList() << QStringLiteral("三角面 001"));
    c2->setIcon(0, QIcon(renderSvg(":/icons/resources/icons/icon/left/cloudlist-black-11.svg", 11)));
    m_projectTree->addTopLevelItem(cloudRoot);

    QTreeWidgetItem* lineRoot = new QTreeWidgetItem(QStringList() << QStringLiteral("特征线列表"));
    lineRoot->setIcon(0, QIcon(renderSvg(":/icons/resources/icons/icon/left/marklist-black-11.svg", 14)));
    QTreeWidgetItem *l1 = new QTreeWidgetItem(lineRoot, QStringList() << QStringLiteral("特征线 001"));
    l1->setIcon(0, QIcon(renderSvg(":/icons/resources/icons/icon/left/marklist-black-11.svg", 11)));
    QTreeWidgetItem *l2 = new QTreeWidgetItem(lineRoot, QStringList() << QStringLiteral("特征线 002"));
    l2->setIcon(0, QIcon(renderSvg(":/icons/resources/icons/icon/left/marklist-black-11.svg", 11)));
    m_projectTree->addTopLevelItem(lineRoot);

    layout->addWidget(m_projectTree);
    return section;
}

QWidget *MainWindow::createParamSection()
{
    QWidget *section = new QWidget();
    section->setObjectName("paramSection");
    QVBoxLayout *layout = new QVBoxLayout(section);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    QWidget *header = new QWidget();
    header->setFixedHeight(28);
    header->setStyleSheet("background-color: #E1E1E1; border-bottom: 1px solid #C0C0C0;");
    QHBoxLayout *headerLayout = new QHBoxLayout(header);
    headerLayout->setContentsMargins(8, 0, 8, 0);
    QLabel *iconLbl = new QLabel();
    iconLbl->setPixmap(renderSvg(":/icons/resources/icons/icon/left/parapanel-black-11.svg", 11));
    headerLayout->addWidget(iconLbl);
    QLabel *title = new QLabel(QStringLiteral("参数面板"));
    title->setObjectName("sectionTitle");
    headerLayout->addWidget(title);
    headerLayout->addStretch();
    layout->addWidget(header);

    QWidget *tabBar = new QWidget();
    tabBar->setFixedHeight(28);
    QHBoxLayout *tabLayout = new QHBoxLayout(tabBar);
    tabLayout->setContentsMargins(0, 0, 0, 0);
    tabLayout->setSpacing(0);
    // P-5 收敛（261002 按键域 §3.4/§6）：自由/推荐/自定义装饰页签退役——亮度在
    // UI 只有一根 20 档预设档滑条（与按键左右键同一把梯同一本账），三参只读
    {
        QLabel *hint = new QLabel(QStringLiteral("亮度档（与设备按键同梯同账）"));
        hint->setStyleSheet("background-color: #FFFFFF; color: #505050; border: none; padding-left: 8px;");
        tabLayout->addWidget(hint);
    }
    tabLayout->addStretch();
    layout->addWidget(tabBar);

    QWidget *slidersWidget = new QWidget();
    slidersWidget->setStyleSheet("background-color: #FFFFFF;");
    slidersWidget->setMinimumHeight(80);
    QVBoxLayout *slidersLayout = new QVBoxLayout(slidersWidget);
    slidersLayout->setContentsMargins(8, 4, 8, 4);
    slidersLayout->setSpacing(0);

    // P-5：五条装饰参数滑条（点云分辨率等未接线占位）退役——面板只留档位滑条
    // ＋三参只读行（换档提交走 sliderReleased 松手离散提交＝设计 P6）
    {
        QWidget *row = new QWidget();
        row->setMinimumHeight(40);
        QVBoxLayout *rowLayout = new QVBoxLayout(row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        rowLayout->setSpacing(0);
        QLabel *label = new QLabel(QStringLiteral("亮度档 1-20"));
        label->setObjectName("paramLabel");
        label->setMinimumHeight(12);
        label->setContentsMargins(0, 0, 0, 0);
        rowLayout->addWidget(label);
        QHBoxLayout *controlLayout = new QHBoxLayout();
        controlLayout->setSpacing(6);
        QSlider *slider = new QSlider(Qt::Horizontal);
        slider->setObjectName("paramSlider");
        slider->setRange(1, 20);
        slider->setValue(10);                     // 档10=面片推荐（档值表插值精确点）
        slider->setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #E1E1E1; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #900021; width: 12px; height: 12px; margin: -5px 0px; border-radius: 6px; border: none; }"
        );
        controlLayout->addWidget(slider, 1);
        QLabel *valueLbl = new QLabel(QStringLiteral("10/20"));
        valueLbl->setObjectName("paramValue");
        valueLbl->setFixedWidth(48);
        valueLbl->setFixedHeight(20);
        valueLbl->setAlignment(Qt::AlignCenter);
        valueLbl->setStyleSheet("border: 1px solid #C0C0C0; border-radius: 4px; background-color: #FFFFFF; color: #000000;");
        QObject::connect(slider, &QSlider::valueChanged, valueLbl, [valueLbl](int val) {
            valueLbl->setText(QString::number(val) + QStringLiteral("/20"));
        });
        controlLayout->addWidget(valueLbl);
        m_param1Slider = slider;                 // 复用成员（语义＝亮度档，P-5 收敛）
        // 松手提交终值（P6 离散提交）：拖动中仅预览档号；按键侧改档经 111 事件
        // 回显同步（QSignalBlocker 防环路）
        QObject::connect(slider, &QSlider::sliderReleased, this, [this]() {
            auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
            if (dm && m_param1Slider) dm->setBrightnessLadderIndex(m_param1Slider->value());
        });
        rowLayout->addLayout(controlLayout);
        slidersLayout->addWidget(row);

        // P-分辨率滑条（261002：体素密度 4 档——菜单①同梯同账；仅待机可调，
        // 扫描中防呆锁住＝与按键侧同一功能口拒口径）
        {
            auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
            QWidget* voxRow = new QWidget();
            voxRow->setMinimumHeight(40);
            QVBoxLayout* voxRowLay = new QVBoxLayout(voxRow);
            voxRowLay->setContentsMargins(0, 0, 0, 0);
            voxRowLay->setSpacing(0);
            QLabel* voxLabel = new QLabel(QStringLiteral("分辨率"));
            voxLabel->setObjectName("paramLabel");
            voxLabel->setMinimumHeight(12);
            voxLabel->setContentsMargins(0, 0, 0, 0);
            voxRowLay->addWidget(voxLabel);
            QHBoxLayout* voxCtl = new QHBoxLayout();
            voxCtl->setSpacing(6);
            QSlider* voxSlider = new QSlider(Qt::Horizontal);
            voxSlider->setObjectName("paramSlider");
            // 27 档三段步长（0.01~5mm）——从梯取实际步值表，不依赖 DM 快照
            const auto voxSteps = Scanner::device::VoxelDensityLadder().steps();
            const int voxSize = static_cast<int>(voxSteps.size());
            voxSlider->setRange(1, voxSize);
            const int voxInit = dm ? dm->voxelLadderIndex() : 1;
            voxSlider->setValue(std::clamp(voxInit, 1, voxSize));
            voxSlider->setStyleSheet(
                "QSlider::groove:horizontal { height: 4px; background: #E1E1E1; border-radius: 2px; }"
                "QSlider::handle:horizontal { background: #900021; width: 12px; height: 12px; margin: -5px 0px; border-radius: 6px; border: none; }"
            );
            voxCtl->addWidget(voxSlider, 1);
            // 值标签显示实际 mm 值——connect 到 this（MainWindow）非 voxVal，
            // 排除 context 生命周期问题；每次 valueChanged 现取梯值
            QLabel* voxVal = new QLabel(slidersWidget);
            m_voxelValLbl = voxVal;                 // 成员存（113 事件回显直更）
            voxVal->setObjectName("paramValue");
            voxVal->setFixedWidth(70);
            voxVal->setFixedHeight(20);
            voxVal->setAlignment(Qt::AlignCenter);
            voxVal->setStyleSheet("border: 1px solid #C0C0C0; border-radius: 4px; background-color: #FFFFFF; color: #000000; font-weight: bold;");
            {
                const auto s0 = Scanner::device::VoxelDensityLadder().steps();
                const int i0 = voxSlider->value();
                voxVal->setText(
                    i0 >= 1 && i0 <= static_cast<int>(s0.size())
                        ? QString::number(s0[static_cast<size_t>(i0 - 1)], 'f', 2) + "mm"
                        : QStringLiteral("--"));
            }
            QObject::connect(voxSlider, &QSlider::valueChanged, this,
                             [this, voxVal](int val) {
                const auto s = Scanner::device::VoxelDensityLadder().steps();
                const QString txt =
                    val >= 1 && val <= static_cast<int>(s.size())
                        ? QString::number(s[static_cast<size_t>(val - 1)], 'f', 2) + "mm"
                        : QStringLiteral("--");
                voxVal->setText(txt);
                statusBar()->showMessage(
                    QStringLiteral("分辨率滑条→ %1（档 %2）").arg(txt).arg(val), 2000);
            });
            voxCtl->addWidget(voxVal);
            m_voxelSlider = voxSlider;              // 成员存（113 事件回显＋扫描锁）
            QObject::connect(voxSlider, &QSlider::sliderReleased, this, [this]() {
                auto* d = m_appCtx ? m_appCtx->deviceManager() : nullptr;
                if (d && m_voxelSlider) d->setVoxelLadderIndex(m_voxelSlider->value());
            });
            voxRowLay->addLayout(voxCtl);
            slidersLayout->addWidget(voxRow);
        }

        m_paramROLabel = new QLabel(QStringLiteral("三参（随档只读）：曝光 -- ms · 激光 -- · 补光 --"));
        m_paramROLabel->setObjectName("paramRO");
        m_paramROLabel->setStyleSheet("color: #707070; padding: 2px;");
        m_paramROLabel->setWordWrap(true);
        slidersLayout->addWidget(m_paramROLabel);
    }

    layout->addWidget(slidersWidget, 1);
    return section;
}

// 面片扫描推荐预设（P-5 收敛 261002）：B=10/L=40/曝光 3ms＝亮度梯档 10 精确点
// （档值表插值锚——260912 口径随档入梯）。换档走 setBrightnessLadderIndex（与
// 按键/UI 滑条同一把梯同一本账）；滑条回显经 111 事件回刷（此处只同步预置位）
void MainWindow::applyMeshPreset() {
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (!dm) return;
    dm->setBrightnessLadderIndex(10);          // 档10=中位（均分：≈2.9ms/47/47）
    statusBar()->showMessage(
        QStringLiteral("面片扫描推荐参数已套用（亮度档 10/20）"), 3000);
}

// 标点扫描推荐预设（P-5 收敛 261002）：B=40＝亮度梯档 10 同档（L 在标点模式
// 实发恒 0 由 effectiveN10 强制，账本值占位）；仅新启会话套用
void MainWindow::applyMarkerPreset() {
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (!dm) return;
    dm->setBrightnessLadderIndex(10);          // 档10 中位（标点检测从此档起步调）
    statusBar()->showMessage(QStringLiteral("标点扫描推荐参数已套用（亮度档 10/20·B=40）"), 3000);
}

QWidget *MainWindow::createInfoSection()
{
    QWidget *section = new QWidget();
    section->setObjectName("infoSection");
    section->setStyleSheet("background-color: #FFFFFF;");
    QVBoxLayout *layout = new QVBoxLayout(section);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    QWidget *header = new QWidget();
    header->setFixedHeight(28);
    header->setStyleSheet("background-color: #E1E1E1; border-bottom: 1px solid #C0C0C0;");
    QHBoxLayout *headerLayout = new QHBoxLayout(header);
    headerLayout->setContentsMargins(8, 0, 8, 0);
    QLabel *iconLbl = new QLabel();
    iconLbl->setPixmap(renderSvg(":/icons/resources/icons/icon/left/infopanel-black-11.svg", 11));
    headerLayout->addWidget(iconLbl);
    QLabel *title = new QLabel(QStringLiteral("系统信息"));
    title->setObjectName("sectionTitle");
    headerLayout->addWidget(title);
    headerLayout->addStretch();
    layout->addWidget(header);

    QGridLayout *gridLayout = new QGridLayout();
    gridLayout->setContentsMargins(6, 6, 6, 6);
    gridLayout->setSpacing(6);

    struct InfoItem { QString icon; QString label; };
    QList<InfoItem> infos = {
        {"WIFIstate-black-11",   QStringLiteral("连接状态")},
        {"cloudnumber-black-11", QStringLiteral("点云数量")},
        {"infopanel-black-11",   QStringLiteral("帧率 (FPS)")},
        {"temprature-black-11",  QStringLiteral("MCU温度")},
        {"cloudlist-black-11",   QStringLiteral("CPU占用率")},
        {"memory-black-11",      QStringLiteral("内存状态")},
        {"icon/left/marklist-black-11", QStringLiteral("键控状态")}
    };

    QLabel** labelPtrs[] = {
        &m_infoConnLabel, &m_infoPointCloudLabel, &m_infoFpsLabel,
        &m_infoTempLabel, &m_infoCpuLabel, &m_infoMemLabel, &m_infoKeyLabel
    };

    for (int i = 0; i < infos.size(); ++i) {
        QWidget *card = new QWidget();
        card->setObjectName("infoCard");
        card->setStyleSheet("QWidget#infoCard { background-color: #FFFFFF; border: 1px solid #C0C0C0; border-radius: 10px; }");
        QVBoxLayout *cardLayout = new QVBoxLayout(card);
        cardLayout->setContentsMargins(9, 12, 9, 12);
        cardLayout->setSpacing(4);

        QHBoxLayout *labelRow = new QHBoxLayout();
        QLabel *icon = new QLabel();
        icon->setPixmap(renderSvg(QString(":/icons/resources/icons/icon/left/%1.svg").arg(infos[i].icon), 11));
        labelRow->addWidget(icon);
        QLabel *lbl = new QLabel(infos[i].label);
        lbl->setObjectName("infoCardLabel");
        labelRow->addWidget(lbl);
        labelRow->addStretch();
        cardLayout->addLayout(labelRow);

        QLabel *val = new QLabel("--");
        val->setObjectName("infoCardValue");
        val->setFixedHeight(30);
        QFont valFont = val->font();
        valFont.setBold(true);
        val->setFont(valFont);
        cardLayout->addWidget(val);

        *labelPtrs[i] = val;
        gridLayout->addWidget(card, i / 2, i % 2);
    }

    layout->addLayout(gridLayout, 1);
    return section;
}

QWidget *MainWindow::create3DViewArea()
{
    QWidget *area = new QWidget();
    area->setObjectName("view3DArea");
    QVBoxLayout *layout = new QVBoxLayout(area);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    QWidget *viewContainer = new QWidget();
    QHBoxLayout *viewLayout = new QHBoxLayout(viewContainer);
    viewLayout->setContentsMargins(0, 0, 0, 0);
    viewLayout->setSpacing(0);

    QVBoxLayout *gradientLayout = new QVBoxLayout();
    gradientLayout->setContentsMargins(6, 10, 6, 10);

    QLabel *labelFar = new QLabel(QStringLiteral("远"));
    labelFar->setAlignment(Qt::AlignCenter);
    labelFar->setStyleSheet("color: #0066FF; font-size: 14px; font-weight: bold;");
    gradientLayout->addWidget(labelFar);

    ArrowSlider *rangeSlider = new ArrowSlider(Qt::Vertical);
    rangeSlider->setObjectName("rangeSlider");
    rangeSlider->setRange(0, 100);
    rangeSlider->setValue(50);
    rangeSlider->setFixedWidth(48);
    rangeSlider->setMinimumHeight(200);
    rangeSlider->setGroovePixmap(renderSvg(":/icons/resources/icons/div.color-gradient-bar.svg", 24, 800));
    rangeSlider->setStyleSheet(
        "QSlider#rangeSlider::groove:vertical { background: transparent; }"
        "QSlider#rangeSlider::handle:vertical { background: transparent; width: 20px; height: 16px; }"
        "QSlider#rangeSlider::sub-page:vertical { background: transparent; }"
        "QSlider#rangeSlider::add-page:vertical { background: transparent; }"
    );
    gradientLayout->addWidget(rangeSlider, 1);

    QLabel *labelNear = new QLabel(QStringLiteral("近"));
    labelNear->setAlignment(Qt::AlignCenter);
    labelNear->setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;");
    gradientLayout->addWidget(labelNear);

    viewLayout->addLayout(gradientLayout);

    m_3dView = new OSGWidget();
    connect(m_3dView, &OSGWidget::streamProgress, this,
        [this](int loaded, int total)
        {
            if (m_cloudItem001)
                m_cloudItem001->setText(0, QStringLiteral("点云数据 001 (%1)").arg(loaded));
        });
    viewLayout->addWidget(m_3dView, 1);

    layout->addWidget(viewContainer, 1);

    m_3dViewArea = area;
    return area;
}

QWidget *MainWindow::createBottomToolBar()
{
    QWidget *bar = new QWidget();
    bar->setObjectName("bottomToolBar");
    bar->setFixedHeight(47);
    QHBoxLayout *layout = new QHBoxLayout(bar);
    layout->setContentsMargins(10, 0, 10, 0);
    layout->setSpacing(0);
    layout->addStretch();

    QWidget *container = new QWidget();
    container->setObjectName("selectionToolBar");
    container->setStyleSheet("border: none;");
    QHBoxLayout *containerLayout = new QHBoxLayout(container);
    containerLayout->setContentsMargins(0, 0, 0, 0);
    containerLayout->setSpacing(2);
    containerLayout->setAlignment(Qt::AlignVCenter);

    // —— 三栏选择模型（05 定稿 D3）三组互斥按钮＋操作组（用户口径 2026-09-05）——
    struct SelBtn { QString iconBlack; QString tip; };
    QList<SelBtn> selButtons = {
        // 组1 对象类型（栏3，三选一；2026-09-05 裁定：激光点改称点云、三角面
        // 不可编辑——三钮口径对齐 D3）
        {":/icons/resources/icons/点云和三角面选择 (1).svg", QStringLiteral("点云：仅圈选激光点云（点云）点")},
        {":/icons/resources/icons/标记点选择 (1).svg", QStringLiteral("标志点：仅圈选标志点")},
        {":/icons/resources/icons/标记点和点云三角面选择 (1).svg", QStringLiteral("标志点加点云：同时圈选标志点与点云")},
        // 组2 选择类型（栏2）：贯穿 / 只取表面
        {":/icons/resources/icons/贯穿选择 (1).svg", QStringLiteral("贯穿：曲线内任意深度的点全部选中")},
        {":/icons/resources/icons/表面选择 (1).svg", QStringLiteral("只取表面：仅选中视线方向最近的表面层")},
        // 组3 工具类型（栏1）：贯穿 / 套索 / 多段线（图标仿套索三态新制 2026-09-05）
        {":/icons/resources/icons/icon/bottom/lasso (1).svg", QStringLiteral("套索：按住左键自由手绘闭合曲线圈选")},
        {":/icons/resources/icons/icon/bottom/polyline (1).svg", QStringLiteral("多段线：逐点落子画折线，双击/回车闭合圈选")},
        // 操作组：全选 / 反选 / 撤销选择 / 抛弃 / 删除
        {":/icons/resources/icons/全选 (1).svg", QStringLiteral("全选：选中全部可选对象")},
        {":/icons/resources/icons/反选 (1).svg", QStringLiteral("反选：反转当前选中集")},
        {":/icons/resources/icons/撤销选择 (1).svg", QStringLiteral("取消选择：清空当前选中集")},
        {":/icons/resources/icons/icon/bottom/throwselect (1).svg", QStringLiteral("抛弃所选：丢弃所选对象")},
        {":/icons/resources/icons/icon/bottom/delete (3).svg", QStringLiteral("删除：圈选并删除所选点（Ctrl+Z 撤销）")},
    };

    // 三栏互斥（QButtonGroup 各组独占）；操作组（索引 7-11）非 checkable
    QButtonGroup* objTypeGroup = new QButtonGroup(container);
    QButtonGroup* toolGroup = new QButtonGroup(container);
    QButtonGroup* depthGroup = new QButtonGroup(container);
    m_objBtnGroup = objTypeGroup;      // 圈选结束复位用（临时解除互斥再清选中）
    m_toolBtnGroup = toolGroup;
    m_depthBtnGroup = depthGroup;

    for (int i = 0; i < selButtons.size(); ++i) {
        QPushButton *btn = new QPushButton();
        btn->setObjectName("selectionButton");
        btn->setFixedSize(40, 40);
        btn->setToolTip(selButtons[i].tip);
        // 选中态视觉：底色淡红高亮＋圆角（配合 toggled 换 (3) 号图标双通道）
        btn->setStyleSheet(
            "QPushButton { border: none; margin: 0px; padding: 0px; background: transparent; }"
            "QPushButton:checked { background: rgba(144,0,33,0.15); border-radius: 6px; }"
            "QPushButton:hover { background: rgba(0,0,0,0.06); border-radius: 6px; }");

        QPixmap pix = renderSvg(selButtons[i].iconBlack, 28);
        btn->setIcon(QIcon(pix));
        btn->setIconSize(QSize(28, 28));
        btn->setContentsMargins(0, 0, 0, 0);
        if (i <= 6) {                                 // 三栏＝可选中互斥（组分辖）
            btn->setCheckable(true);
            if (i <= 2) objTypeGroup->addButton(btn, i);
            else if (i <= 4) depthGroup->addButton(btn, i);
            else toolGroup->addButton(btn, i);
            // 选中态视觉反馈：checked 切 (3) 号图标（暗红激活态），弹回换常态图
            QString iconOn = selButtons[i].iconBlack;
            iconOn.replace(" (1)", " (3)");
            QPixmap pixOn = renderSvg(iconOn, 28);
            QIcon iconNormal = QIcon(pix);
            QIcon iconActive = QIcon(pixOn);
            btn->setIcon(iconNormal);
            connect(btn, &QPushButton::toggled, this, [btn, iconNormal, iconActive](bool on) {
                btn->setIcon(on ? iconActive : iconNormal);
            });
        }
        // 初始不选（用户口径 2026-09-05：启动悬浮条无选中态；三栏选择仅圈选
        // 流程中生效——圈选结束回工具默认套索，见 lassoCompleted 接线）
        // 操作组（索引 7-11）暂隐藏（用户口径 2026-09-05：三栏先行；P4 编辑
        // 会话接线时随功能恢复显示——钮与接线保留，仅 hide）
        if (i >= 7) btn->hide();
        // tip 强制显示（2026-09-05）：无边框半透明悬浮窗下 QToolTip 自动机制
        // 不触发——开 WA_Hover，经 MainWindow::eventFilter 于 HoverEnter 立即
        // showText（见 eventFilter "selectionButton" 分支）
        btn->setAttribute(Qt::WA_Hover, true);
        btn->installEventFilter(this);
        containerLayout->addWidget(btn);
        m_selectionButtons.append(btn);

        if (i == 2 || i == 4) {                       // 组间分隔（组1|组2|组3）
            QFrame *separator = new QFrame();
            separator->setFixedWidth(1);
            separator->setFixedHeight(28);
            separator->setStyleSheet("background-color: #C0C0C0; border: none;");
            containerLayout->addWidget(separator);
        }
    }

    layout->addWidget(container);
    layout->addStretch();

    // 对象类型组（索引 0-2）→ 圈选目标掩码（D3 栏3：0 点云＝Clouds，
    // 1 标志点＝Markers，2 标志点加点云＝两者）
    if (m_3dView && m_selectionButtons.size() > 2)
    {
        connect(objTypeGroup, &QButtonGroup::idToggled, this, [this](int id, bool on) {
            if (!on || !m_3dView) return;
            using LT = OSGWidget::LassoTarget;
            const int mask = (id == 1) ? LT::LassoMarkers
                          : (id == 2) ? (LT::LassoMarkers | LT::LassoClouds)
                                      : LT::LassoClouds;      // 0 点云
            m_3dView->setLassoTargets(mask);
        });
        // 选择类型组（索引 3 贯穿 / 4 只取表面）→ 深度模式（D3 栏2）
        connect(depthGroup, &QButtonGroup::idToggled, this, [this](int id, bool on) {
            if (!on || !m_3dView) return;
            m_3dView->setLassoFirstLayer(id == 4);
        });
    }

    // 套索（索引 5）：按住左键拖拽自由手绘，松开自动闭合
    // 多段线（索引 6）：逐点落子，右键/双击/回车闭合（260920 工具类型分发）
    if (m_selectionButtons.size() > 5)
    {
        connect(m_selectionButtons[5], &QPushButton::clicked, this, [this]()
        {
            if (!ensureEditAllowed() || !m_3dView) return;
            m_3dView->enterLassoDeleteMode(OSGWidget::LassoTool::Lasso);
        });
    }
    if (m_selectionButtons.size() > 6)
    {
        connect(m_selectionButtons[6], &QPushButton::clicked, this, [this]()
        {
            if (!ensureEditAllowed() || !m_3dView) return;
            m_3dView->enterLassoDeleteMode(OSGWidget::LassoTool::Polyline);
        });
    }

    // 圈选流程结束（确认删除/取消均发 lassoCompleted）→ 工具组回无选中初始态
    //＋左侧统计栏刷新（可见标志点数——软删后 alpha>0 折算，2026-09-06）
    //——用户口径 2026-09-05（启动与流程结束均无选中）；singleShot(0) 延后一拍
    // 避开模态弹窗销毁期的状态覆盖
    if (m_3dView && m_selectionButtons.size() > 7)
    {
        connect(m_3dView, &OSGWidget::lassoCompleted, this, [this]()
        {
            JMW_LOG_INFO("app-MainWindow", "[lassoCompleted] 收到——三栏复位＋统计刷新（流程节点留痕）");
            QTimer::singleShot(0, this, [this]() {
                // 互斥组内 setChecked(false) 会被独占语义挡下（实测日志：复位后仍
                // checked=true）——三组各自临时解除互斥→清全部选中→恢复互斥
                for (QButtonGroup* g : {m_objBtnGroup, m_toolBtnGroup, m_depthBtnGroup})
                    if (g) g->setExclusive(false);
                for (int i = 0; i <= 6; ++i)
                    if (m_selectionButtons.size() > i) m_selectionButtons[i]->setChecked(false);
                for (QButtonGroup* g : {m_objBtnGroup, m_toolBtnGroup, m_depthBtnGroup})
                    if (g) g->setExclusive(true);
                // 统计栏同步可见数（删除已生效——软删 alpha=0 不计入）
                if (m_markerCurrentItem && m_3dView) {
                    const size_t vis = m_3dView->visibleMarkerCount();
                    m_markerCurrentItem->setText(
                        0, QStringLiteral("标记点 %1 (%2)")
                                  .arg(m_markerScanSeq, 3, 10, QChar('0'))
                                  .arg(vis));
                }
                // 点云计数（260912 收口）：仓库计数为准（软删不改仓库——导出
                // 口径与计数一致；渲染侧软删仅影响画面）
                if (auto* pcbCnt = m_appCtx ? m_appCtx->pointCloudBuffer() : nullptr;
                    m_cloudItem001 && pcbCnt) {
                    m_cloudItem001->setText(0, QStringLiteral("点云数据 001 (%1)")
                                                .arg(pcbCnt->getTotalPointCount()));
                }
                JMW_LOG_INFO("app-MainWindow", "[lassoCompleted] 三栏复位完成（0-6 全清）");
            });
        });
    }

    // 删除（索引 11）— lasso-to-delete mode
    if (m_selectionButtons.size() > 11)
    {
        connect(m_selectionButtons[11], &QPushButton::clicked, this, [this]()
        {
            if (!ensureEditAllowed() || !m_3dView) return;
            m_3dView->enterLassoDeleteMode();
        });
    }

    // Ctrl+Z undo（统计栏同步——恢复点后可见数变化，2026-09-06）
    QShortcut* undoShortcut = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Z), this);
    connect(undoShortcut, &QShortcut::activated, this, [this]()
    {
        m_3dView->undoDelete();
        if (m_markerCurrentItem && m_3dView) {
            const size_t vis = m_3dView->visibleMarkerCount();
            m_markerCurrentItem->setText(
                0, QStringLiteral("标记点 %1 (%2)")
                          .arg(m_markerScanSeq, 3, 10, QChar('0'))
                          .arg(vis));
        }
    });

    return bar;
}

QPushButton *MainWindow::createNavButton(const QString &text, const QString &)
{
    QPushButton *btn = new QPushButton(text);
    btn->setObjectName("navButton");
    btn->setFixedHeight(38);
    return btn;
}

QPushButton *MainWindow::createToolButton(const QString &iconBlack, const QString &iconRed,
                                            const QString &iconGray, const QString &text)
{
    QPushButton *btn = new QPushButton();
    btn->setObjectName("toolButton");
    btn->setFixedSize(64, 56);
    QVBoxLayout *btnLayout = new QVBoxLayout(btn);
    btnLayout->setContentsMargins(0, 4, 0, 4);
    btnLayout->setSpacing(2);
    btnLayout->setAlignment(Qt::AlignCenter);

    QLabel *iconLbl = new QLabel();
    iconLbl->setPixmap(renderSvg(QString(":/icons/resources/icons/%1.svg").arg(iconBlack), 20));
    iconLbl->setAlignment(Qt::AlignCenter);
    btnLayout->addWidget(iconLbl);

    QLabel *textLbl = new QLabel(text);
    textLbl->setObjectName("toolButtonText");
    textLbl->setAlignment(Qt::AlignCenter);
    textLbl->setFixedWidth(60);
    btnLayout->addWidget(textLbl);

    return btn;
}

QPushButton *MainWindow::createSelectionButton(const QString &iconFile1, const QString &iconFile2,
                                                 const QString &iconFile3)
{
    QPushButton *btn = new QPushButton();
    btn->setObjectName("selectionButton");
    btn->setFixedSize(44, 32);
    btn->setIcon(QIcon(renderSvg(iconFile1, 14)));
    btn->setIconSize(QSize(14, 14));
    return btn;
}

void MainWindow::setButtonGroupExclusive(QList<QPushButton*> buttons)
{
    for (QPushButton *btn : buttons) {
        connect(btn, &QPushButton::clicked, this, [this, btn, buttons]() {
            setActiveButton(btn, buttons);
        });
    }
}

void MainWindow::setActiveButton(QPushButton *btn, QList<QPushButton*> group)
{
    for (QPushButton *b : group) {
        b->setProperty("active", false);
        b->style()->unpolish(b);
        b->style()->polish(b);
    }
    btn->setProperty("active", true);
    btn->style()->unpolish(btn);
    btn->style()->polish(btn);
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    // 悬浮条选择钮：HoverEnter 强制弹 tip（QToolTip 自动机制在无边框半透明
    // 悬浮窗下不触发——用户实测 2026-09-05）
    if (obj->objectName() == "selectionButton") {
        QWidget* w = qobject_cast<QWidget*>(obj);
        if (w) {
            if (event->type() == QEvent::HoverEnter && !w->toolTip().isEmpty()) {
                QToolTip::showText(QCursor::pos(), w->toolTip(), w);
            } else if (event->type() == QEvent::HoverLeave) {
                QToolTip::hideText();
            }
        }
        return QMainWindow::eventFilter(obj, event);
    }

    if (obj->objectName() == "floatingToolbar") {
        if (event->type() == QEvent::MouseButtonPress) {
            QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton) {
                // 获取当前窗口的位置
                QPoint globalPos = mouseEvent->globalPos();
                // 获取窗口的位置
                QWidget *widget = qobject_cast<QWidget*>(obj);
                if (widget) {
                    QPoint windowPos = widget->window()->pos();
                    // 计算鼠标在窗口内的相对位置
                    QPoint relativePos = globalPos - windowPos;
                    
                    // 保存拖动状态和位置
                    bool dragging = true;
                    QPoint dragPosition = relativePos;
                    
                    // 设置拖动标志和位置
                    obj->setProperty("dragging", QVariant(dragging));
                    obj->setProperty("dragPosition", QVariant(dragPosition));
                    
                    return true;
                }
            }
        } else if (event->type() == QEvent::MouseMove) {
            QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->buttons() & Qt::LeftButton) {
                bool dragging = obj->property("dragging").toBool();
                QPoint dragPosition = obj->property("dragPosition").toPoint();
                
                if (dragging) {
                    // 计算新的窗口位置
                    QPoint globalPos = mouseEvent->globalPos();
                    QPoint newPos = globalPos - dragPosition;
                    
                    // 移动窗口
                    QWidget *widget = qobject_cast<QWidget*>(obj);
                    if (widget) {
                        widget->window()->move(newPos);
                    }
                    
                    return true;
                }
            }
        } else if (event->type() == QEvent::MouseButtonRelease) {
            QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton) {
                obj->setProperty("dragging", QVariant());
                obj->setProperty("dragPosition", QVariant());
                
                return true;
            }
        }
    }
    
    return QMainWindow::eventFilter(obj, event);
}

// ============================================================================
// 系统信息面板 — 定时采集真实数据
// ============================================================================
void MainWindow::startInfoTimer()
{
    // CPU 采集走 PDH（updateInfoSection 头部双计数器）——GetSystemTimes 时间戳
    // 基线随旧差分块一并移除（2026-09-06）
    m_infoTimer = new QTimer(this);
    connect(m_infoTimer, &QTimer::timeout, this, [this]() {
        updateInfoSection();
        updateVirtualKeypadStates();   // P-键盘：每秒刷新——响应 SystemState
                                        // ＋calibArmed/采集等非 StateChanged 驱动态
        // P-分辨率：滑条与 DM 同步（非拖动时——初始时序偏差自动收敛）
        if (m_voxelSlider && !m_voxelSlider->isSliderDown()) {
            auto* dmVox = m_appCtx ? m_appCtx->deviceManager() : nullptr;
            if (dmVox && m_voxelSlider->value() != dmVox->voxelLadderIndex())
                m_voxelSlider->setValue(dmVox->voxelLadderIndex());
        }
    });
    m_infoTimer->start(1000);

    // 3D 视图定时刷新（从 PointCloudBuffer 拉快照→值传渲染——快照拉取归
    // 调用方，03 不再依赖 06 实现类 2026-09-01）
    QTimer* cloudTimer = new QTimer(this);
    connect(cloudTimer, &QTimer::timeout, this, [this]() {
        if (!m_appCtx || !m_3dView) return;
        auto* pcb = m_appCtx->pointCloudBuffer();
        if (!pcb) return;
        static int lastCount = 0;
        int curCount = pcb->getTotalPointCount();
        // 260912b 标签独立于 3D 重载：计数始终刷新（真值=仓库快照）
        if (m_cloudItem001 && curCount != lastCount)
            m_cloudItem001->setText(0, QStringLiteral("点云数据 001 (%1)")
                                        .arg(curCount));
        // 3D 快照重载仅非扫描期（扫描期激光由 laserCloudUpdated 直渲——
        // 此处再拉同份点＝双份顶点/内存；会话结束后（含导入云）才走此路）
        if (curCount == 0 || curCount == lastCount) return;
        if (m_appCtx->isScanSessionActive()) {
            lastCount = curCount;                  // 会话期只跟计数不重载
            return;
        }
        uint64_t version = 0;
        std::vector<cv::Point3f> points;
        std::vector<cv::Vec3b> colors;
        pcb->getSnapshot(version, points, colors);
        m_3dView->loadCloudSnapshot(version, points, colors);
        lastCount = curCount;
    });
    cloudTimer->start(500);
}

void MainWindow::updateInfoSection()
{
    static int updateCount = 0;
    ++updateCount;
    // 诊断（2026-09-01 状态不更新）：函数心跳+poll 耗时（perfMonitor::poll
    // 阻塞嫌疑——卡住则横幅/CPU/连接全不更新）
    const bool infoVerbose = (updateCount % 30 == 0);
    auto pollMs = 0;

    // A-T17：PerfMonitor 1s 拉取驱动（10 设计 P3——挂既有 m_infoTimer；
    // AppContext 无 Qt 依赖不持定时器，poll=provider 快照→阈值判定→双级告警）
    if (m_appCtx && m_appCtx->perfMonitor()) {
        const auto t0 = std::chrono::steady_clock::now();
        m_appCtx->perfMonitor()->poll();
        pollMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - t0).count());
    }
    if (infoVerbose) {
        JMW_LOG_INFO("app-MainWindow", "[MainWindow] infoTimer 心跳 #{}（perfPoll={}ms）",
                     updateCount, pollMs);
    }

    // 启动自检横幅（1s 轮询快照；完成后继续显示终态 10s 再撤，常态零占用）
    if (m_appCtx) {
        static int selfCheckHoldSecs = 0;
        const auto items = m_appCtx->selfCheckSnapshot();
        if (!items.empty()) {
            const bool done = items.back().second;   // "完成" 项
            if (done) ++selfCheckHoldSecs;
            if (done && selfCheckHoldSecs > 10) {
                statusBar()->showMessage(QString());
            } else {
                QString msg = done ? QStringLiteral("自检完成 ") : QStringLiteral("自检中… ");
                for (const auto& it : items) {
                    const QString name = QString::fromStdString(it.first);
                    if (name == QStringLiteral("完成")) continue;
                    msg += (it.second ? QStringLiteral("✓%1 ").arg(name)
                                      : QStringLiteral("✗%1 ").arg(name));
                }
                statusBar()->showMessage(msg.trimmed());
            }
        }
    }

    // === 先更新 CPU 和内存（纯 Windows API，不依赖任何框架组件）===
    if (m_infoCpuLabel) {
        // 双计数器（2026-09-05 实测定口径）：显示＝% Processor Utility 钳位
        // [0,100]——实测任务管理器（Win10/11）＝效用口径封顶 100（TM=100 时
        // Utility=121.6/Time=75.4，三方对齐判定）；Time 留诊断日志比对。
        //（原 GetSystemTimes 差分实测恒 0——多核语义陷阱，2026-09-01 弃）
        static PDH_HQUERY hQuery = nullptr;
        static PDH_HCOUNTER hCpuTime = nullptr;     // % Processor Time（日志比对）
        static PDH_HCOUNTER hCpuUtil = nullptr;     // % Processor Utility（显示）
        static bool pdhFailed = false;
        if (!hQuery && !pdhFailed) {
            if (PdhOpenQueryW(nullptr, 0, &hQuery) == ERROR_SUCCESS &&
                PdhAddEnglishCounterW(hQuery,
                    L"\\Processor Information(_Total)\\% Processor Time",
                    0, &hCpuTime) == ERROR_SUCCESS &&
                PdhAddEnglishCounterW(hQuery,
                    L"\\Processor Information(_Total)\\% Processor Utility",
                    0, &hCpuUtil) == ERROR_SUCCESS) {
                // 首次收集建立基线
                PdhCollectQueryData(hQuery);
            } else {
                pdhFailed = true;
                hQuery = nullptr;
            }
        }
        double busy = 0.0, utility = 0.0;
        if (hQuery && PdhCollectQueryData(hQuery) == ERROR_SUCCESS) {
            PDH_FMT_COUNTERVALUE val;
            if (hCpuTime &&
                PdhGetFormattedCounterValue(hCpuTime, PDH_FMT_DOUBLE, nullptr, &val) ==
                    ERROR_SUCCESS)
                busy = val.doubleValue;
            if (hCpuUtil &&
                PdhGetFormattedCounterValue(hCpuUtil, PDH_FMT_DOUBLE, nullptr, &val) ==
                    ERROR_SUCCESS)
                utility = val.doubleValue;
        }
        double usage = utility;                    // TM 口径＝Utility 封顶 100
        if (usage < 0) usage = 0; if (usage > 100) usage = 100;
        m_infoCpuLabel->setText(QString::number(usage, 'f', 1) + " %");
        if (infoVerbose) {
            JMW_LOG_INFO("app-MainWindow", "[CPU口径] Time={:.1f}% Utility={:.1f}% 显示={:.1f}%",
                         busy, utility, usage);
        }
    }

    if (m_infoMemLabel) {
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        if (GlobalMemoryStatusEx(&mem)) {
            double totalGB = static_cast<double>(mem.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
            double usedGB = totalGB - static_cast<double>(mem.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
            m_infoMemLabel->setText(QString("%1 / %2 GB (#%3)")
                .arg(usedGB, 0, 'f', 1).arg(totalGB, 0, 'f', 1).arg(updateCount));
        }
    }

    // === 再更新设备相关状态 ===
    Scanner::data::DeviceStateCache* dsc = nullptr;
    Scanner::data::PointCloudBuffer* pcb = nullptr;
    if (m_appCtx) {
        dsc = m_appCtx->deviceStateCache();
        pcb = m_appCtx->pointCloudBuffer();
    }

    // 1. 连接状态 — 从 DeviceStateCache 读
    if (m_infoConnLabel) {
        bool camConnected = false;
        if (dsc) {
            auto camState = dsc->getState("Camera");
            camConnected = (camState.state == Scanner::DeviceState::Connected ||
                            camState.state == Scanner::DeviceState::Streaming);
            if (infoVerbose) {
                JMW_LOG_INFO("app-MainWindow",
                             "[MainWindow] 状态诊断: Camera.state={} camConn={} dsc={} poll={}ms",
                             static_cast<int>(camState.state), camConnected, dsc ? 1 : 0, pollMs);
            }
        }
        m_infoConnLabel->setText(camConnected ? "已连接" : "未连接");
        m_infoConnLabel->setStyleSheet(camConnected ? "color: #00AA00;" : "color: #CC0000;");
    }

    // 2. 点云数量 — 从 PointCloudBuffer 读（260927：仓库空时回落显示导入云计数）
    if (m_infoPointCloudLabel) {
        int count = pcb ? pcb->getTotalPointCount() : 0;
        if (count == 0) count = static_cast<int>(m_importedCloudCount);
        m_infoPointCloudLabel->setText(QString::number(count));
    }

    // 3. 帧率 — 从 DeviceStateCache 或 FrameBuffer 水位
    if (m_infoFpsLabel) {
        // 实测接收帧率（2026-09-06 改：原 DeviceStateCache.getFps 为硬件设定值
        // 恒 60 不反映实际——改为配对交付差分实测）
        const int fps = m_appCtx ? m_appCtx->cameraMeasuredFps() : 0;
        if (fps > 0) {
            m_infoFpsLabel->setText(QString::number(fps) + " fps");
        } else {
            m_infoFpsLabel->setText("-- fps");
        }
    }

    // 4. 下位机温度——G02 四路取最高一路显示（260912 用户口径；ts=0=未收帧）
    if (m_infoTempLabel) {
        const auto t = (m_appCtx && m_appCtx->deviceManager())
                           ? m_appCtx->deviceManager()->getLastTemperatures()
                           : Scanner::device::serial::TempFrame{};
        if (t.ts > 0) {
            const double mx = std::max({t.celsius[0], t.celsius[1],
                                        t.celsius[2], t.celsius[3]});
            m_infoTempLabel->setText(QString::number(mx, 'f', 1) + QStringLiteral(" ℃"));
            m_infoTempLabel->setStyleSheet(mx > 50.0 ? "color: #DDAA00;" : "");
        } else {
            m_infoTempLabel->setText(QStringLiteral("-- ℃"));
            m_infoTempLabel->setStyleSheet("");
        }
    }

    // 5. CPU 占用率——唯一写入点在函数头部（PDH 双计数器 Utility 钳位口径）。
    //    原 GetSystemTimes 差分块已删（2026-09-06：双重写入每拍覆盖 PDH 结果，
    //    面板恒显示忙时口径——「与任务管理器不一致」真凶）。

    // 6. 内存状态
    if (m_infoMemLabel) {
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        if (GlobalMemoryStatusEx(&mem)) {
            double totalGB = static_cast<double>(mem.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
            double usedGB = totalGB - static_cast<double>(mem.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
            m_infoMemLabel->setText(QString("%1 / %2 GB (#%3)")
                .arg(usedGB, 0, 'f', 1)
                .arg(totalGB, 0, 'f', 1)
                .arg(updateCount));
        } else {
            m_infoMemLabel->setText("-- / -- GB");
        }
    }

    // 7. 键控状态回显（260927 按设计实现；261002 按键域定稿刷新）：状态栏常驻
    //    指示菜单/子态/调节对象/四梯档位（档位可见——解决「按了半天不知道调到哪」）
    if (m_infoKeyLabel) {
        auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
        if (dm) {
            const auto ms = dm->menuState();
            QString s;
            if (ms.layer == 2) {
                using Sub = Scanner::device::MenuState::Substate;
                if (ms.substate == Sub::AdjustVoxel) {
                    s = QStringLiteral("分辨率 %1mm")
                        .arg(dm->voxelLadderValue(), 0, 'f', 2);
                } else if (ms.substate == Sub::ConfirmReset) {
                    s = QStringLiteral("⑤重置：再按 M 确认（其他键取消）");
                } else {
                    QString curs;
                    for (int i = 1; i <= 5; ++i)
                        curs += (ms.cursor == i) ? QStringLiteral("◆") : QStringLiteral("·");
                    s = QStringLiteral("菜单 ") + curs + QStringLiteral("（M 选中/U 退）");
                }
            } else {
                if (ms.adjustCtx == Scanner::device::MenuState::AdjustCtx::Brightness) {
                    s = QStringLiteral("调节:亮度 第%1/%2档（L 降/R 升）·景深:%3")
                        .arg(dm->presetLadderIndex())
                        .arg(dm->presetLadderSize())
                        .arg(dm->depthOfField() == 0 ? QStringLiteral("近") : QStringLiteral("远"));
                } else {
                    s = QStringLiteral("调节:显示远近 第%1/%2档（L 近/R 远）")
                        .arg(dm->distanceLadderIndex()).arg(dm->distanceLadderSize());
                }
            }
            m_infoKeyLabel->setText(s);
        }
    }
}

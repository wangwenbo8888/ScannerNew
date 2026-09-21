#pragma once

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QSlider>
#include <QSpinBox>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QStackedWidget>
#include <QPaintEvent>
#include <QSlider>
#include <QSvgRenderer>

#include "OSGWidget.h"
#include "base/EventBus.h"   // SubscriberId（P1-2 UI 状态图标订阅句柄）

class ArrowSlider : public QSlider
{
    Q_OBJECT
public:
    explicit ArrowSlider(Qt::Orientation orientation, QWidget *parent = nullptr);
    void setGroovePixmap(const QPixmap &pixmap);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QPixmap m_groovePixmap;
};

class CalibDialog;
class CameraControl;
class LEADSCANSeries;
class AppContext;
namespace calib_display { class CalibBoard2D; }
namespace Scanner::service { enum class SystemState : uint8_t; }   // P1-2 前向声明（值传递签名）

class QProgressDialog;
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(AppContext* appCtx = nullptr, QWidget *parent = nullptr);
    ~MainWindow();

    AppContext* appCtx() { return m_appCtx; }

private:
    QWidget *createTitleBar();
    QWidget *createNavBar();
    QWidget *createToolBar();
    QWidget *createLeftPanel();
    QWidget *createProjectSection();
    QWidget *createParamSection();
    QWidget *createInfoSection();
    QWidget *create3DViewArea();
    QWidget *createBottomToolBar();
    QWidget *createCoordOverlay();

    QPushButton *createIconButton(const QString &iconBlack, const QString &iconRed, const QString &iconGray,
                                   const QString &text = QString(), int iconSize = 14, bool vertical = false);
    QPushButton *createNavButton(const QString &text, const QString &iconWhite);
    QPushButton *createToolButton(const QString &iconBlack, const QString &iconRed, const QString &iconGray,
                                   const QString &text);
    QPushButton *createSelectionButton(const QString &iconFile1, const QString &iconFile2, const QString &iconFile3);

    static QPixmap renderSvg(const QString &svgPath, int size);
    static QPixmap renderSvg(const QString &svgPath, int w, int h);
    void setupUILayout();
    void repositionFloatingToolbar();
    void setButtonGroupExclusive(QList<QPushButton*> buttons);
    void setActiveButton(QPushButton *btn, QList<QPushButton*> group);
    void createFloatingToolbar();
    bool ensureEditAllowed();                 // P0-3 编辑门禁唯一事实源（canEnterEditSession）：不满足则拒并提示

private slots:
    void onIntegrateTestClicked();
    void onReloadPointCloud();
    void onCalibDeviceClicked();
    void onScanClicked();
    /// 单键扫描按钮态视觉：idx=2 标点/3 面片/4 精细/5 深孔（四键 2-5）；active=
    /// 红框"停止扫描"、false=复原。各键独立显示自己的会话态（活跃键记录
    /// m_activeScanToolIdx）
    void setScanButtonVisual(int idx, bool active);
    /// 相机预览监视弹窗（调试）：扫描启动时弹出，实时显示左右相机灰度图
    /// （观察灯帧交替/标记点可见性）。数据走 AppContext 调试帧分路
    void showCameraMonitor();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;
    void closeEvent(QCloseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

    int m_activeScanToolIdx = -1;     // 当前活跃扫描键（2-5 四模式键；-1=无——停止复原用）
    bool m_sessionSimExtract = false; // 会话启动时「模拟数据」开关基线（260917：
                                      //   就绪态续采检测开关翻转——变则完整重启装配）
    QDialog *m_camDlg = nullptr;      // 相机预览监视弹窗（调试，懒建）
    QLabel *m_camLeft = nullptr;      // 左相机图
    QLabel *m_camRight = nullptr;
    QLabel *m_camFrameLabel = nullptr; // 左右帧号＋偏移显示（调试）     // 右相机图
    int m_camFrameSkip = 0;           // （已退役 2026-09-05：预览节流改时间基准 previewFps——成员待清）
    QList<QPushButton*> m_navLeftButtons;
    QList<QPushButton*> m_navRightButtons;
    QList<QPushButton*> m_toolButtons;
    QList<QPushButton*> m_selectionButtons;
    class QSlider *m_param1Slider = nullptr;  // 参数1（曝光/亮度三路联动旋钮——B 口径：
                                              //   启动扫描前把三参压成旋钮值，恒一致）
    // 激光点仓库跨会话累计（260912c）：融合云是会话私有的（新会话从零）——仓库
    // 直接整包替换会在新会话首推把累计清掉（真机「点云数据 001 清零」实证）。
    // 新会话首推锁存基线（=此前全部累积），此后每次替换=基线+当期融合云——
    // 计数/导出跨会话只增不减（正反扫描两趟合导出的数据基础）
    std::vector<cv::Point3f> m_laserBasePts;
    bool m_laserSessionLatched = false;
    void applyParam1ToLedger();               // 三参（exposure/bgLight/laserLevel）压账本
    void applyMarkerPreset();                 // 标点扫描推荐预设（260912：B=40 成功基线
                                              //   ＋H=60＋旋钮同步；续采不套用——保留现值）
    void applyMeshPreset();                   // 面片扫描推荐预设（260912：B=10 成功基线
                                              //   ＋L=40＋H=60＋曝光3ms；B≠L 非比例曲线，
                                              //   直写账本＋旋钮同步激光位）

    OSGWidget *m_3dView;
    QWidget *m_3dViewArea;
    QLabel *m_projectName;
    QTreeWidget *m_projectTree;
    QTreeWidgetItem *m_cloudItem001;
    QTreeWidgetItem *m_markerRootItem = nullptr;     // 标记点列表根（动态挂扫描会话节点）
    QTreeWidgetItem *m_markerCurrentItem = nullptr;  // 当前扫描会话节点（实时计数落点）
    int m_markerScanSeq = 0;                         // 会话序号（标记点 001、002…）
    QButtonGroup *m_toolBtnGroup = nullptr;          // 三栏互斥组（圈选结束复位需临时解除互斥）
    QButtonGroup *m_objBtnGroup = nullptr;           // 对象类型组（同上）
    QButtonGroup *m_depthBtnGroup = nullptr;         // 选择类型组（同上）
    QWidget *m_floatingToolbar;

    AppContext *m_appCtx = nullptr;

    // 标定分屏
    QWidget* m_calibSplitWidget = nullptr;
    calib_display::CalibBoard2D* m_calibBoard2D = nullptr;

    QWidget *m_integrateTestDialog;
    CalibDialog *m_calibDialog;
    LEADSCANSeries *m_series = nullptr;

    // 系统信息面板
    QTimer *m_infoTimer = nullptr;
    QLabel *m_infoConnLabel = nullptr;
    QLabel *m_infoPointCloudLabel = nullptr;
    QLabel *m_infoFpsLabel = nullptr;
    QLabel *m_infoTempLabel = nullptr;
    QLabel *m_infoCpuLabel = nullptr;
    QLabel *m_infoMemLabel = nullptr;

    // —— UI 状态图标（P1-2）：订阅 10 状态机 StateChanged（EventBus 主通道），
    //    主线程刷 7 态指示。EventBus 同步分发持总线锁——锁内只拷贝 param2 再
    //    QMetaObject::invokeMethod queued 投递主线程（对齐 P0-2 故障桥红线）。
    // ——
    Scanner::infra::SubscriberId m_stateChangedSubId_ = 0;
    QLabel *m_stateIndicator = nullptr;                 // 状态栏右下角常驻态指示（色点+文案）
    void updateStateIndicator(Scanner::service::SystemState s);   // UI 线程槽（仅设文案/配色）
    static QString stateText(Scanner::service::SystemState s);    // 7 态文案映射
    static QString stateColor(Scanner::service::SystemState s);   // 7 态配色映射

    void startInfoTimer();
    void updateInfoSection();
    QProgressDialog* m_finalBADlg = nullptr;   // final-BA progress dialog (260920 finish background GBA)

};
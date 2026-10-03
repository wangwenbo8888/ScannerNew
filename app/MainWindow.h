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
    /// 扫描就绪窗口（260927 就绪流程）：UI 模式键→armScanSession 备好会话后弹出
    /// ——「按设备 M 键开始扫描」；轮询 isCapturing 自动关闭，取消＝终止会话
    void showScanReadyPrompt(const QString& modeTitle, int btnIdx);
    /// 虚拟按键表盘（261002 临时测试机·无实体键）：弹窗模拟扫描仪面板五键——
    /// U/L/R/M 经 G01 注入测试缝走与真机完全相同的链路（rx 文本行→手势环→
    /// KeySemantics→动作）；下键 D 不在 260831 协议 G01 键位表（置灰标注）
    void showVirtualKeypad();

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
    QLabel *m_camFrameLabel = nullptr; // 左右帧号＋偏移显示（调试）
    QDialog *m_scanReadyDlg = nullptr;      // 扫描就绪窗口（260927 就绪流程，懒建）
    QLabel *m_scanReadyLabel = nullptr;     // 就绪提示文案（模式名注入）
    class QTimer *m_scanReadyPoll = nullptr; // 就绪轮询（200ms 查 isCapturing→自动关）
    int m_scanReadyBtnIdx = -1;             // 就绪对应的模式键（取消时复原视觉）
    QDialog *m_vkeyPad = nullptr;           // 虚拟按键表盘（261002 临时测试机，懒建）
    QPushButton *m_vkeyBtns[5][3] = {};     // P-键盘态同步：[行][手势] 按钮指针
                                            //   行序 U/D/L/R/M×单击/双击/长按（懒建后填）
    class QLabel *m_vkeyStateLbl = nullptr; // P-键盘态同步：当前全局态常驻标签
    /// P-键盘态同步（§3.2.5 七态矩阵）：按当前全局态灰化/恢复表盘按钮——
    /// S1/S3/S6/S7 全拦态＝除逃生类（M 长按急停/U 长按回主界面）外全灰；
    /// S2/S4/S5＝恢复（功能口拒的键仍可按——拒因在横幅可见）
    void updateVirtualKeypadStates();
    QList<QPushButton*> m_navLeftButtons;
    QList<QPushButton*> m_navRightButtons;
    QList<QPushButton*> m_toolButtons;
    QList<QPushButton*> m_selectionButtons;
    class QSlider *m_param1Slider = nullptr;  // P-5 收敛：亮度档滑条（1-20，与按键
                                              //   同梯同账——松手提交；三参唯一写入口）
    class QSlider *m_voxelSlider = nullptr;   // P-分辨率：体素密度滑条（1-4，菜单①
                                              //   同梯同账；仅待机可调，扫描中锁住）
    class QLabel *m_paramROLabel = nullptr;   // P-5：三参只读行（随档显示，UI 无三参控件）
    // 激光点仓库跨会话累计（260912c）：融合云是会话私有的（新会话从零）——仓库
    // 直接整包替换会在新会话首推把累计清掉（真机「点云数据 001 清零」实证）。
    // 新会话首推锁存基线（=此前全部累积），此后每次替换=基线+当期融合云——
    // 计数/导出跨会话只增不减（正反扫描两趟合导出的数据基础）
    std::vector<cv::Point3f> m_laserBasePts;
    bool m_laserSessionLatched = false;
    void applyMarkerPreset();                 // 标点扫描推荐预设（261002 P-5：档10
                                              //   B=40 成功基线；续采不套用）
    void applyMeshPreset();                   // 面片扫描推荐预设（261002 P-5：档10
                                              //   {3ms,40,10} 梯上精确点）

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
    QLabel *m_infoKeyLabel = nullptr;   // 键控状态回显（260927：层/调节上下文/档位）

    // —— P-3 顶部大字横幅（261002 按键域 §3.4：远距可读——操作者距屏数米）——
    /// 瞬态提示（1.8s 自动消失、新顶旧；danger=红底）；菜单期常驻内容由
    /// refreshBannerPersistent 依据 menuState 快照重建（瞬态结束自动恢复）
    void showBanner(const QString& text, bool danger = false);
    void refreshBannerPersistent();     // p1=110/113 菜单/子态/档位变化后刷新常驻行
    QLabel *m_banner = nullptr;         // 横幅条（懒建，overlay 顶部不占布局）
    class QTimer *m_bannerTimer = nullptr;   // 瞬态 1.8s 计时（到时恢复常驻/隐藏）
    QString m_bannerPersist;            // 菜单期常驻文案（空=无常驻）
    uint64_t m_importedCloudCount = 0; // 导入点云计数（260927 上树/信息卡——不入扫描仓库）

    // —— UI 状态图标（P1-2）：订阅 10 状态机 StateChanged（EventBus 主通道），
    //    主线程刷 7 态指示。EventBus 同步分发持总线锁——锁内只拷贝 param2 再
    //    QMetaObject::invokeMethod queued 投递主线程（对齐 P0-2 故障桥红线）。
    // ——
    Scanner::infra::SubscriberId m_stateChangedSubId_ = 0;
    Scanner::infra::SubscriberId m_userDefinedSubId_ = 0;   // 按键③④/视点缩放消费（260927）
    QLabel *m_stateIndicator = nullptr;                 // 状态栏右下角常驻态指示（色点+文案）
    void updateStateIndicator(Scanner::service::SystemState s);   // UI 线程槽（仅设文案/配色）
    static QString stateText(Scanner::service::SystemState s);    // 7 态文案映射
    static QString stateColor(Scanner::service::SystemState s);   // 7 态配色映射

    void startInfoTimer();
    void updateInfoSection();
    QProgressDialog* m_finalBADlg = nullptr;   // final-BA progress dialog (260920 finish background GBA)

};
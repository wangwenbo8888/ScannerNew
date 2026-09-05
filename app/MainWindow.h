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

private slots:
    void onIntegrateTestClicked();
    void onReloadPointCloud();
    void onCalibDeviceClicked();
    void onScanClicked();
    /// 单键扫描按钮态视觉：idx=2 标点/3 面片；active=红框"停止扫描"、false=复原。
    /// 各键独立显示自己的会话态（活跃键记录 m_activeScanToolIdx）
    void setScanButtonVisual(int idx, bool active);
    /// 相机预览监视弹窗（调试）：扫描启动时弹出，实时显示左右相机灰度图
    /// （观察灯帧交替/标记点可见性）。数据走 AppContext 调试帧分路
    void showCameraMonitor();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;
    void closeEvent(QCloseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

    int m_activeScanToolIdx = -1;     // 当前活跃扫描键（2/3；-1=无——停止复原用）
    QDialog *m_camDlg = nullptr;      // 相机预览监视弹窗（调试，懒建）
    QLabel *m_camLeft = nullptr;      // 左相机图
    QLabel *m_camRight = nullptr;     // 右相机图
    int m_camFrameSkip = 0;           // 帧节流计数（每 3 帧刷新 ≈10fps@30fps 流）
    QList<QPushButton*> m_navLeftButtons;
    QList<QPushButton*> m_navRightButtons;
    QList<QPushButton*> m_toolButtons;
    QList<QPushButton*> m_selectionButtons;

    OSGWidget *m_3dView;
    QWidget *m_3dViewArea;
    QLabel *m_projectName;
    QTreeWidget *m_projectTree;
    QTreeWidgetItem *m_cloudItem001;
    QTreeWidgetItem *m_markerRootItem = nullptr;     // 标记点列表根（动态挂扫描会话节点）
    QTreeWidgetItem *m_markerCurrentItem = nullptr;  // 当前扫描会话节点（实时计数落点）
    int m_markerScanSeq = 0;                         // 会话序号（标记点 001、002…）
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
    double m_prevCpuIdle = 0;
    double m_prevCpuKernel = 0;
    double m_prevCpuUser = 0;

    void startInfoTimer();
    void updateInfoSection();
};

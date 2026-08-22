#include "MainWindow.h"
#include "AppContext.h"
#include "ObsLogger.h"
#include "CrashHandler.h"
#include "jmw_logging.h"
#include <QApplication>
#include <QDialog>
#include <QFile>
#include <QLabel>
#include <QProgressBar>
#include <QScreen>
#include <QSurfaceFormat>
#include <QTimer>
#include <QVBoxLayout>
#include <string>

int main(int argc, char *argv[])
{
    Scanner::service::obsLoggerInit({});
    Scanner::service::crash::install("dumps");
    if (std::string residual; Scanner::service::crash::detectResidualDump(residual)) {
        JMW_LOG_WARN("10-Crash", "检测到崩溃残留: {}", residual);
        Scanner::service::obsExportDiagnosticsPackage(residual);
    }
    JMW_LOG_INFO("app", "=== ScannerFramework 启动 ===");

    _putenv_s("OSG_PLUGIN_PATH", "F:/osg3.6.5/install/bin/osgPlugins-3.6.5");
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

    // 8x MSAA 抗锯齿（参照 LEADSCAN K2 的 setNumMultiSamples(8)），消除网格鱼鳞/闪烁
    QSurfaceFormat fmt;
    fmt.setDepthBufferSize(24);
    fmt.setSamples(8);
    QSurfaceFormat::setDefaultFormat(fmt);

    QApplication app(argc, argv);

    QFile styleFile(":/icons/dark.qss");
    if (styleFile.open(QFile::ReadOnly | QFile::Text)) {
        app.setStyleSheet(styleFile.readAll());
        styleFile.close();
    }

    // 装配全部框架组件
    AppContext appCtx;
    appCtx.initialize();

    MainWindow window(&appCtx);
    QScreen *screen = app.primaryScreen();
    QRect avail = screen->availableGeometry();
    window.setGeometry(avail);
    window.show();                        // 主界面照常显示——弹窗置前挡操作

    // 初始化弹窗（模态置顶）：主界面可见但被"初始化中......"挡住不可操作；
    // 自检完成自动消失（20s 兜底——失败也放行，失败项由状态栏横幅持续显示）
    QDialog initDlg(&window);
    initDlg.setWindowTitle(QStringLiteral("初始化"));
    initDlg.setModal(true);
    initDlg.setWindowFlags(initDlg.windowFlags() | Qt::WindowStaysOnTopHint);
    {
        auto* dlgLayout = new QVBoxLayout(&initDlg);
        auto* dlgMsg = new QLabel(QStringLiteral("初始化中......（设备连接 / 灯路检测 / 相机检测）"), &initDlg);
        dlgMsg->setAlignment(Qt::AlignCenter);
        auto* dlgBar = new QProgressBar(&initDlg);
        dlgBar->setRange(0, 0);           // 不确定进度（忙碌指示）
        dlgBar->setFixedWidth(320);
        dlgLayout->addWidget(dlgMsg);
        dlgLayout->addWidget(dlgBar, 0, Qt::AlignHCenter);
        initDlg.setLayout(dlgLayout);
    }

    // 设备链路（相机枚举+自动搜口）后台起
    appCtx.startDevicesAsync();

    QTimer initTimer;
    int initTicks = 0;
    QObject::connect(&initTimer, &QTimer::timeout, &app, [&]() {
        ++initTicks;
        if (appCtx.selfCheckDone() || initTicks >= 100) {   // 200ms×100=20s 兜底
            initTimer.stop();
            initDlg.accept();            // 自检完成/超时——弹窗自动消失，主界面可操作
        }
    });
    initTimer.start(200);
    initDlg.exec();                      // 模态事件循环（initTimer 驱动 accept 退出）

    int ret = app.exec();

    appCtx.shutdown();
    JMW_LOG_INFO("app", "=== ScannerFramework 退出 ===");
    return ret;
}

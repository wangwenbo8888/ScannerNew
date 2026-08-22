#include "MainWindow.h"
#include "AppContext.h"
#include "ObsLogger.h"
#include "CrashHandler.h"
#include "jmw_logging.h"
#include <QApplication>
#include <QFile>
#include <QLabel>
#include <QScreen>
#include <QSplashScreen>
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
    window.setGeometry(avail);           // 不 show——初始化完成由 splashTimer 放行显示

    // 设备初始化弹窗（完成才进主界面）：状态栏横幅此前只是"进界面后可见"，
    // 需求是初始化期不进主界面——QSplashScreen 挡主窗，自检完成/超时再放行
    QSplashScreen splash;
    splash.setWindowTitle(QStringLiteral("初始化"));
    {
        auto* splLayout = new QVBoxLayout(&splash);
        auto* splMsg = new QLabel(QStringLiteral("初始化中……（设备连接 / 灯路检测 / 相机检测）"), &splash);
        splMsg->setAlignment(Qt::AlignCenter);
        splLayout->addWidget(splMsg);
        splash.setLayout(splLayout);
        splash.show();
        app.processEvents();
    }

    // 设备链路（相机枚举+自动搜口）后台起；弹窗轮询自检完成（上限 20s 兜底放行——
    // 失败也进主界面，失败项由状态栏横幅持续显示，避免死锁在弹窗）
    appCtx.startDevicesAsync();

    QTimer splashTimer;
    int splashSecs = 0;
    QObject::connect(&splashTimer, &QTimer::timeout, &app, [&]() {
        ++splashSecs;
        if (appCtx.selfCheckDone() || splashSecs >= 100) {   // 200ms×100=20s 兜底
            splashTimer.stop();
            window.show();       // 先显主窗再关弹窗——反序会瞬态零可见窗口触发
            splash.close();      // lastWindowClosed → 事件循环退出（实测自退 bug）
        }
        app.processEvents();
    });
    splashTimer.start(200);

    int ret = app.exec();

    appCtx.shutdown();
    JMW_LOG_INFO("app", "=== ScannerFramework 退出 ===");
    return ret;
}

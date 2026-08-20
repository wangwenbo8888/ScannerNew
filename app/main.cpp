#include "MainWindow.h"
#include "AppContext.h"
#include "ObsLogger.h"
#include "CrashHandler.h"
#include "jmw_logging.h"
#include <QApplication>
#include <QFile>
#include <QScreen>
#include <QSurfaceFormat>
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
    window.show();

    int ret = app.exec();

    appCtx.shutdown();
    JMW_LOG_INFO("app", "=== ScannerFramework 退出 ===");
    return ret;
}

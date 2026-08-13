#include "MainWindow.h"
#include "AppContext.h"
#include <QApplication>
#include <QFile>
#include <QScreen>
#include <QSurfaceFormat>
#include <spdlog/spdlog.h>

int main(int argc, char *argv[])
{
    _putenv_s("OSG_PLUGIN_PATH", "F:/osg3.6.5/install/bin/osgPlugins-3.6.5");
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

    spdlog::set_level(spdlog::level::info);
    spdlog::info("=== ScannerFramework 启动 ===");

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
    spdlog::info("=== ScannerFramework 退出 ===");
    return ret;
}

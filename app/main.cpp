#include "MainWindow.h"
#include "AppContext.h"
#include "modules/08_devicemgmt/DeviceManager.h"   // setWireTap（串口监视弹窗）
#include "ObsLogger.h"
#include "CrashHandler.h"
#include "jmw_logging.h"
#include <QApplication>
#include <QColor>
#include <QDateTime>
#include <QDialog>
#include <QFile>
#include <QFont>
#include <QLabel>
#include <QProgressBar>
#include <QScreen>
#include <QSurfaceFormat>
#include <QTextCursor>
#include <QTextEdit>
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

    // —— 串口通信监视弹窗（调试）：上位机 TX / 下位机 RX 双向实时显示 ——
    //（open 前挂 wireTap，探测/自检/扫描全链路帧都进窗口；回调来自 rx/写线程，
    //  invokeMethod queued 投递到 UI 线程追加；窗口关闭后投递自动丢弃）
    QDialog tapDlg(&window);
    tapDlg.setWindowTitle(QStringLiteral("串口通信监视（下位机调试）"));
    tapDlg.resize(760, 480);
    QTextEdit tapEdit;
    tapEdit.setReadOnly(true);
    tapEdit.setFont(QFont("Consolas", 9));
    QVBoxLayout tapLay(&tapDlg);
    tapLay.setContentsMargins(4, 4, 4, 4);
    tapLay.addWidget(&tapEdit);
    appCtx.deviceManager()->setWireTap([&tapEdit](bool tx, const std::string& data) {
        QMetaObject::invokeMethod(&tapEdit, [&tapEdit, tx, s = QString::fromStdString(data)]() {
            const QString ts = QDateTime::currentDateTime().toString("HH:mm:ss.zzz");
            const QColor c = tx ? QColor(38, 122, 231) : QColor(21, 138, 70);
            // 整行先拼纯文本再统一转义——"RX<" 的 < 若不转义会被 QTextEdit 当
            // HTML 标签吞掉整行（RX 行不显示的根因）
            const QString line = QString("[%1] %2 %3")
                                     .arg(ts, tx ? QStringLiteral("TX>") : QStringLiteral("RX<"), s);
            tapEdit.append(QString("<span style='color:%1'>%2</span>")
                               .arg(c.name(), line.toHtmlEscaped()));
            if (tapEdit.document()->blockCount() > 2000) {   // 防爆：滚出前 1000 行
                QTextCursor cur(tapEdit.document());
                cur.movePosition(QTextCursor::Start);
                cur.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor, 1000);
                cur.removeSelectedText();
            }
        }, Qt::QueuedConnection);
    });
    // tapDlg.show();   // 通信监视弹窗默认隐藏（调试需要时恢复此行）

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

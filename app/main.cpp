#include "MainWindow.h"
#include "AppContext.h"
#include "modules/08_devicemgmt/DeviceManager.h"   // setWireTap（串口监视弹窗）
#include "ObsLogger.h"
#include "CrashHandler.h"
#include "jmw_logging.h"
#include <QApplication>
#include <QColor>
#include <QDateTime>
#include <QDir>
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
    // 261004 排障开关：JMW_NO_CRASH_HANDLER=1 时跳过内置崩溃转储——堆损坏场景
    // 内置 MiniDumpWriteDump 亦崩（0 字节 dump）且吞掉 WER；跳过后由 Windows
    // LocalDumps（已配）写完整转储供离线符号化
    if (!std::getenv("JMW_NO_CRASH_HANDLER"))
        Scanner::service::crash::install("dumps");
    if (std::string residual; !std::getenv("JMW_NO_CRASH_HANDLER") &&
                              Scanner::service::crash::detectResidualDump(residual)) {
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

    // 锚定 CWD 到 exe 目录（260705 根治）：config//logs/dumps 全走相对路径，
    // 启动器工作目录各异（VS 默认工程目录/资源管理器/快捷方式/真机自启）曾是
    // 「标定参数未就绪→扫描键不激活」根因——此后任何方式启动都从 exe 旁读配置
    QDir::setCurrent(QCoreApplication::applicationDirPath());

    // 全局弹窗设计语言（260705）：QMessageBox/QProgressDialog 家族统一白卡风
    //（白底＋圆角＋#2980B9 交互色；作用域选择器只影响弹窗内控件，主窗按钮不受扰；
    // QFileDialog 走系统原生不换肤。原 :/icons/dark.qss 未登记 qrc 恒空载——退役）
    app.setStyleSheet(QStringLiteral(
        "QMessageBox { background-color: white; }"
        "QMessageBox QLabel { color: #333; font-size: 14px; }"
        "QMessageBox QPushButton, QProgressDialog QPushButton {"
        " background-color: #E8F0FE; color: #1A5276; border: 1px solid #2980B9;"
        " border-radius: 6px; font-size: 14px; font-weight: bold;"
        " padding: 6px 18px; min-width: 64px; }"
        "QMessageBox QPushButton:hover, QProgressDialog QPushButton:hover"
        " { background-color: #D4E6F1; }"
        "QMessageBox QPushButton:pressed, QProgressDialog QPushButton:pressed"
        " { background-color: #AED6F1; }"
        "QProgressDialog { background-color: white; }"
        "QProgressDialog QLabel { color: #444; font-size: 14px; }"
        "QProgressBar { border: none; border-radius: 4px; background-color: #F0F2F5;"
        " max-height: 10px; text-align: center; color: transparent; }"
        "QProgressBar::chunk { border-radius: 4px; background-color: #2980B9; }"));

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
    // 261005 用户口径：弹窗隐藏（保留实现——kShowCommMonitor 改 true 即恢复；
    // 不挂 wireTap，通信监视零开销）
    static constexpr bool kShowCommMonitor = false;
    if (kShowCommMonitor) {
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
    tapDlg.show();   // 通信监视弹窗（2026-09-06 用户调试：暂停/续采串口消息排查）
    }

    // 初始化弹窗（模态置顶）：主界面可见但被"初始化中......"挡住不可操作；
    // 自检完成自动消失（20s 兜底——失败也放行，失败项由状态栏横幅持续显示）
    // 260705 卡片化（与其余弹窗同风）：白底圆角无边框＋酒红标题＋蓝色忙碌条
    QDialog initDlg(&window);
    initDlg.setWindowTitle(QStringLiteral("初始化"));
    initDlg.setModal(true);
    initDlg.setWindowFlags(initDlg.windowFlags() | Qt::WindowStaysOnTopHint);
    initDlg.setWindowFlag(Qt::FramelessWindowHint, true);
    initDlg.setAttribute(Qt::WA_TranslucentBackground);
    initDlg.setFixedSize(360, 180);
    initDlg.setStyleSheet(
        "QDialog { background-color: white; border: 1px solid #e0e0e0;"
        " border-radius: 8px; }");
    {
        auto* dlgLayout = new QVBoxLayout(&initDlg);
        dlgLayout->setContentsMargins(20, 16, 20, 16);
        auto* dlgTitle = new QLabel(QStringLiteral("系统初始化"), &initDlg);
        dlgTitle->setAlignment(Qt::AlignCenter);
        dlgTitle->setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #8B1A2B; padding: 0 0 8px 0;");
        dlgLayout->addWidget(dlgTitle);
        auto* dlgMsg = new QLabel(QStringLiteral("初始化中......（设备连接 / 灯路检测 / 相机检测）"), &initDlg);
        dlgMsg->setAlignment(Qt::AlignCenter);
        dlgMsg->setStyleSheet("color:#444; font-size:13px;");
        dlgLayout->addWidget(dlgMsg);
        auto* dlgBar = new QProgressBar(&initDlg);
        dlgBar->setRange(0, 0);           // 不确定进度（忙碌指示）
        dlgBar->setFixedWidth(320);
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

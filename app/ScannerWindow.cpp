#include "ScannerWindow.h"
#include "AppContext.h"
#include "IWorkflow.h"
#include "ScanWorkflow.h"
#include "CalibrationWorkflow.h"
#include "modules/08_devicemgmt/DeviceManager.h"
#include "modules/08_devicemgmt/HardwareMonitor.h"
#include "StateMachine.h"
#include "CommandGate.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>

// ============================================================================
// Mat → QImage 转换
// ============================================================================
static QImage matToQImage(cv::Mat& mat)
{
    if (mat.type() == CV_8UC1) {
        QImage img(mat.cols, mat.rows, QImage::Format_Indexed8);
        img.setColorCount(256);
        for (int i = 0; i < 256; i++)
            img.setColor(i, qRgb(i, i, i));
        uchar* pSrc = mat.data;
        for (int row = 0; row < mat.rows; row++) {
            uchar* pDest = img.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return img;
    } else if (mat.type() == CV_8UC3) {
        QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    return QImage();
}

// ============================================================================
// 构造 / 析构
// ============================================================================
void ScannerWindow::pushFrameToBuffer(const Scanner::hal::StereoFrame& frame)
{
    ++m_frameCount;
    Scanner::data::FrameData fd;
    fd.frameId = frame.frameId;
    fd.timestamp = frame.timestamp;
    fd.leftGray = frame.leftGray;
    fd.rightGray = frame.rightGray;
    m_frameBuffer->pushFrame(fd);
}

ScannerWindow::ScannerWindow(AppContext* appCtx, QWidget *parent)
    : QMainWindow(parent), m_appCtx(appCtx)
{
    ui.setupUi(this);

    // 从 AppContext 获取已装配的组件（设备经 DeviceManager 门面——A-T17 收口）
    if (m_appCtx) {
        m_frameBuffer = m_appCtx->frameBuffer();
    } else {
        // 无 AppContext 时自建（兼容旧模式；设备门面不可用——设备操作将提示不可用）
        m_frameBuffer = new Scanner::data::FrameBuffer(60);
    }

    // 连接按钮
    connect(ui.pushButton_OpenScannerCamera, &QPushButton::clicked, this, &ScannerWindow::onOpenScannerCamera);
    connect(ui.pushButton_CloseScannerCamera, &QPushButton::clicked, this, &ScannerWindow::onCloseScannerCamera);
    connect(ui.pushButton_Start_Scanner, &QPushButton::clicked, this, &ScannerWindow::onStartScanner);
    connect(ui.pushButton_Stop_Scanner, &QPushButton::clicked, this, &ScannerWindow::onStopScanner);

    // 连接滑块
    connect(ui.horizontalSlider_Freq, &QSlider::valueChanged, this, &ScannerWindow::onSliderFreqChanged);
    connect(ui.horizontalSlider_Background_Lighting, &QSlider::valueChanged, this, &ScannerWindow::onSliderBackgroundChanged);
    connect(ui.horizontalSlider_Laser_Lighting, &QSlider::valueChanged, this, &ScannerWindow::onSliderLaserChanged);
    connect(ui.horizontalSlider_ExposeTime, &QSlider::valueChanged, this, &ScannerWindow::onSliderExposeChanged);

    // FPS 定时器
    m_fpsTimer = new QTimer(this);
    connect(m_fpsTimer, &QTimer::timeout, this, &ScannerWindow::onTimeout);
    m_fpsTimer->start(1000);

    // 显示定时器 — 固定频率从 FrameBuffer 取最新帧显示
    m_consumerTimer = new QTimer(this);
    connect(m_consumerTimer, &QTimer::timeout, this, [this]() {
        if (!m_frameBuffer) return;
        // 取出所有积压帧，只显示最新的一帧
        std::optional<Scanner::data::FrameData> latest;
        while (auto f = m_frameBuffer->popFrame(std::chrono::milliseconds(0))) {
            latest = std::move(f);
        }
        if (!latest || latest->leftGray.empty()) return;

        cv::Mat leftReduced, rightReduced;
        cv::resize(latest->leftGray, leftReduced, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        cv::resize(latest->rightGray, rightReduced, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        emit updateImages(matToQImage(leftReduced), matToQImage(rightReduced));
    });
    m_consumerTimer->start(100);  // 显示固定 10fps，不影响采集帧率

    // 图像更新信号
    connect(this, &ScannerWindow::updateImages, this, [this](QImage left, QImage right) {
        ui.label_LeftImage->setPixmap(QPixmap::fromImage(left).scaled(
            ui.label_LeftImage->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        ui.label_RightImage->setPixmap(QPixmap::fromImage(right).scaled(
            ui.label_RightImage->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    });

    // 分辨率选择下拉框
    m_resCombo = new QComboBox(this);
    m_resCombo->addItem("2048x1536", QSize(2048, 1536));
    m_resCombo->addItem("1280x1024", QSize(1280, 1024));
    m_resCombo->addItem("1024x768",  QSize(1024, 768));
    m_resCombo->addItem("640x480",   QSize(640, 480));
    m_resCombo->setFixedWidth(150);
    m_resCombo->move(80, 490);
    m_resCombo->show();
    connect(m_resCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ScannerWindow::onResolutionChanged);

    // ScanWorkflow 进度回调
    if (m_appCtx && m_appCtx->scanWorkflow()) {
        m_appCtx->scanWorkflow()->setProgressCallback(
            [this](const Scanner::workflow::WorkflowProgress& p) {
                QMetaObject::invokeMethod(this, [this, p]() {
                    ui.textEdit_Info->append(QString("[%1] %2%")
                        .arg(QString::fromStdString(p.stageName))
                        .arg(static_cast<int>(p.progress * 100)));
                });
            });
    }

    ui.textEdit_Info->append("ScannerWindow 已就绪");
}

ScannerWindow::~ScannerWindow()
{
    if (m_consumerTimer) m_consumerTimer->stop();
    // AppContext 拥有的组件不在此销毁（设备归 DeviceManager 门面——A-T17 收口）
    if (!m_appCtx) {
        delete m_frameBuffer;
    }
}

// ============================================================================
// 设备操作（经 DeviceManager 门面——A-T17 串口旁路收口，相机+下位机统一走 08）
// ============================================================================
void ScannerWindow::onOpenScannerCamera()
{
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (!dm) {
        ui.textEdit_Info->append("设备门面不可用（无 AppContext）");
        return;
    }
    // open 一条龙：相机→MCU→参数→N12Z1→逻辑线程（幂等：已开直返 ok）
    auto r = dm->open();
    ui.textEdit_Info->append(r.success ? "设备已打开（相机+下位机）" :
        QString("打开失败: %1").arg(QString::fromStdString(r.message)));

    // 自检清单回填（open 含相机步与串口步——两项同 r.success，AppContext 同口径）
    if (m_appCtx) {
        m_appCtx->notifySelfCheckItem("camera", r.success);
        m_appCtx->notifySelfCheckItem("serialPort", r.success);
    }
}

void ScannerWindow::onCloseScannerCamera()
{
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (!dm) return;
    auto r = dm->close();
    ui.textEdit_Info->append(r.success ? "设备已关闭（相机+下位机）" :
        QString("关闭失败: %1").arg(QString::fromStdString(r.message)));

    // 保留 transition 直连：断连→S1 是状态机矩阵合法边，属设备事件桥非命令通道写口（设计 §9）
    if (r.success && m_appCtx && m_appCtx->stateMachine()) {
        m_appCtx->stateMachine()->transition(Scanner::EventType::DeviceDisconnected);
        // close 倒序关相机与串口——两项自检同时复位，重开后重新过自检
        m_appCtx->notifySelfCheckItem("camera", false);
        m_appCtx->notifySelfCheckItem("serialPort", false);
    }
}

void ScannerWindow::onStartScanner()
{
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (!dm || !dm->isCameraOpen()) {
        ui.textEdit_Info->append("请先打开相机");
        return;
    }

    // A-T17 串口旁路收口：原 N10/N11 手拼串口命令删除——曝光经参数账本
    // （setParam=相机直设+记账；N10 组参自账本组帧，门面 enterScan/采集链下发）
    const int expose = ui.horizontalSlider_ExposeTime->value();
    dm->setParam("exposure", expose, Scanner::device::ParamEntry::Source::Ui);

    // 帧出口接线 + 采集启动（门面 startCapture 内含 N11 H1 与相机开流；
    // 帧回调只做计数+入队，最轻量）
    dm->startFrameStream([this](const Scanner::hal::StereoFrame& frame) {
        pushFrameToBuffer(frame);
    });
    dm->startCapture();

    // P5-T15 ①：经统一命令通道点火扫描（门禁 S2→S4/S5；payload=ScanMode 0/1
    // 只喂状态机 S4/S5 判别——handler 无参拿不到，模式经 setScanMode 先设进
    // 工作流）。TODO(UI 接入期)：模式选择控件落地后改读控件值；现状固定 A 模式
    // （纯标记点——激光温度表空=A 模式正常配置）。拒绝时 gate 已发
    // CommandRejected 事件，此处沿用信息面板轻提示（同 T14 口径）
    if (m_appCtx && m_appCtx->scanWorkflow()) {
        const auto mode = Scanner::ScanMode::MarkerOnly;
        m_appCtx->scanWorkflow()->setScanMode(mode);
        auto gr = m_appCtx->commandGate()->submit("start_scan",
                                                  static_cast<int64_t>(mode));
        if (!gr.success)
            ui.textEdit_Info->append(QString("扫描启动被拒: %1")
                .arg(QString::fromStdString(gr.message)));
    }

    // 启动 HardwareMonitor（周期采集温度/帧率）
    if (m_appCtx && m_appCtx->hwMonitor()) {
        m_appCtx->hwMonitor()->setFrameCounter([this]() {
            return static_cast<int>(m_currentFps);
        });
        // TODO(07 接入期): process/drop 计数器自 07 流水线侧注入（A-T17 暂不接）
        m_appCtx->hwMonitor()->start(1000);
    }

    ui.textEdit_Info->append(QString("采集已启动 (曝光 %1 ms)").arg(expose));
}

void ScannerWindow::onStopScanner()
{
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;

    // P5-T15 ⑦：工作流收尾经 gate 触发（finish_scan：S4/S5 内合法，handler 点火
    // stop() 合账 → ⑩ onFinished_ 回报 notifyCompleted 切 S2——切态不在 ⑦）。
    // 门禁拒绝且无活跃会话属预期（如未开扫即点停）；但断连已切 S1 等角落路径下
    // 工作流可能仍活——兜底直停保证回收（§3.3 不许悬死）
    if (m_appCtx && m_appCtx->scanWorkflow()) {
        auto gr = m_appCtx->commandGate()->submit("finish_scan");
        const auto ws = m_appCtx->scanWorkflow()->getState();
        if (gr.success) {
            ui.textEdit_Info->append("扫描管线已停止");
        } else if (ws == Scanner::workflow::WorkflowState::Running ||
                   ws == Scanner::workflow::WorkflowState::Paused) {
            m_appCtx->scanWorkflow()->stop();
            ui.textEdit_Info->append("扫描管线已停止（兜底直停——门禁已拒绝）");
        }
    }

    // 硬件停采（②③⑥ 采集启停语义=S4/S5 子态不经 gate）：监视器直停；
    // 下位机 N11 H0 + 相机停流一律经门面（原串口手拼+相机直调段收口）
    if (m_appCtx && m_appCtx->hwMonitor()) {
        m_appCtx->hwMonitor()->stop();
    }
    if (dm) {
        dm->stopCapture();
        ui.textEdit_Info->append("采集已停止");
    }
}

// ============================================================================
// 标定
// ============================================================================
void ScannerWindow::onCalibrateClicked()
{
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (!m_appCtx || !m_appCtx->calibWorkflow()) {
        ui.textEdit_Info->append("标定工作流不可用");
        return;
    }
    if (!dm || !dm->isCameraOpen()) {
        ui.textEdit_Info->append("请先打开相机");
        return;
    }

    ui.textEdit_Info->append("开始标定流程...");
    auto* calib = m_appCtx->calibWorkflow();
    calib->setProgressCallback([this](const Scanner::workflow::WorkflowProgress& p) {
        QMetaObject::invokeMethod(this, [this, p]() {
            ui.textEdit_Info->append(QString("[标定] %1 (%2%)")
                .arg(QString::fromStdString(p.stageName))
                .arg(static_cast<int>(p.progress * 100)));
        });
    });

    // 启动采集供标定使用（经门面：帧出口接线 + N11 H1 + 开流）
    if (!dm->isCapturing()) {
        dm->startFrameStream([this](const Scanner::hal::StereoFrame& frame) {
            pushFrameToBuffer(frame);
        });
        dm->startCapture();
    }

    // P5-T14：经统一命令通道点火（门禁 S2→S3）——initialize+start 移入 gate
    // handler（毫秒级点火，B 批算在专属线程）；拒绝时 gate 已发 CommandRejected
    // 事件，UI 统一反馈后续接（此处仅沿用信息面板轻提示）
    auto r = m_appCtx->commandGate()->submit("start_calibration");
    if (!r.success)
        ui.textEdit_Info->append(QString("标定启动被拒: %1")
            .arg(QString::fromStdString(r.message)));
}

// ============================================================================
// 滑块
// ============================================================================
void ScannerWindow::onSliderFreqChanged(int v)
{
    ui.label_Freq_Value->setText(QString::number(v));
}

void ScannerWindow::onSliderBackgroundChanged(int v)
{
    ui.label_Background_Value->setText(QString::number(v));
}

void ScannerWindow::onSliderLaserChanged(int v)
{
    ui.label_Laser_Lighting_Value->setText(QString::number(v));
}

void ScannerWindow::onSliderExposeChanged(int v)
{
    ui.label_ExposeTime_Value->setText(QString::number(v));
    // A-T17：曝光经参数账本（setParam=相机直设+记账——08 ParamStore 唯一真相源；
    // 相机未开时纯记账，开门面后账本值生效）
    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (dm) {
        dm->setParam("exposure", static_cast<double>(v),
                     Scanner::device::ParamEntry::Source::Ui);
        if (dm->isCameraOpen())
            ui.textEdit_Info->append(QString("曝光设置: %1 ms").arg(v));
    }
}

void ScannerWindow::onResolutionChanged(int index)
{
    QSize res = m_resCombo->itemData(index).toSize();
    m_pendingWidth = res.width();
    m_pendingHeight = res.height();

    auto* dm = m_appCtx ? m_appCtx->deviceManager() : nullptr;
    if (dm && dm->isCameraOpen()) {
        ui.textEdit_Info->append(QString("分辨率将在下次打开相机时生效: %1x%2")
            .arg(res.width()).arg(res.height()));
        ui.textEdit_Info->append("请先关闭相机，再重新打开");
    } else {
        ui.textEdit_Info->append(QString("分辨率预设: %1x%2").arg(res.width()).arg(res.height()));
    }
}

// ============================================================================
// FPS 定时器
// ============================================================================
void ScannerWindow::onTimeout()
{
    uint64_t fps = m_frameCount - m_prevFrameCount;
    m_prevFrameCount = m_frameCount;
    m_currentFps = fps;
    ui.label_FrameRate_Value_Left->setText(QString::number(fps));
    ui.label_FrameRate_Value_Right->setText(QString::number(fps));
}

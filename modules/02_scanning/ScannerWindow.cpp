#include "ScannerWindow.h"
#include "AppContext.h"
#include "workflow/IWorkflow.h"
#include "ScanWorkflow.h"
#include "CalibrationWorkflow.h"
#include "modules/08_devicemgmt/HardwareMonitor.h"
#include "service/StateMachine.h"

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
Scanner::device::CameraControl* ScannerWindow::getCameraControl() {
    return m_cam;
}

ScannerWindow::ScannerWindow(AppContext* appCtx, QWidget *parent)
    : QMainWindow(parent), m_appCtx(appCtx)
{
    ui.setupUi(this);

    // 从 AppContext 获取已装配的组件
    if (m_appCtx) {
        m_cam = m_appCtx->camera();
        m_frameBuffer = m_appCtx->frameBuffer();
    } else {
        // 无 AppContext 时自建（兼容旧模式）
        Scanner::device::StereoPairConfig cfg;
        cfg.deviceIndexLeft = 0;
        cfg.deviceIndexRight = 1;
        cfg.rotateRight180 = true;
        cfg.defaultExposureMs = ui.horizontalSlider_ExposeTime->value();
        m_cam = new Scanner::device::CameraControl(cfg);
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

    // 打开串口
    openPorts();

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
    // AppContext 拥有的组件不在此销毁
    if (!m_appCtx) {
        if (m_cam) {
            m_cam->stopAsyncCapture();
            m_cam->close();
            delete m_cam;
        }
        delete m_frameBuffer;
    }
    if (m_serialPort1 && m_serialPort1->isOpen()) m_serialPort1->close();
    if (m_serialPort2 && m_serialPort2->isOpen()) m_serialPort2->close();
    delete m_serialPort1;
    delete m_serialPort2;
}

// ============================================================================
// 串口
// ============================================================================
void ScannerWindow::openPorts()
{
    // 枚举可用串口
    foreach (const QSerialPortInfo& info, QSerialPortInfo::availablePorts()) {
        m_portNameList << info.portName();
        ui.textEdit_Info->append(QString("发现串口: %1").arg(info.portName()));
    }

    if (m_portNameList.empty()) {
        ui.textEdit_Info->append("未发现串口");
        return;
    }

    // 打开串口1
    m_serialPort1 = new QSerialPort(this);
    m_serialPort1->setPortName(m_portNameList[0]);
    if (m_serialPort1->open(QIODevice::ReadWrite)) {
        m_serialPort1->setBaudRate(QSerialPort::Baud115200, QSerialPort::AllDirections);
        m_serialPort1->setDataBits(QSerialPort::Data8);
        m_serialPort1->setFlowControl(QSerialPort::NoFlowControl);
        m_serialPort1->setParity(QSerialPort::NoParity);
        m_serialPort1->setStopBits(QSerialPort::OneStop);
        connect(m_serialPort1, &QSerialPort::readyRead, this, [this]() {
            QByteArray data = m_serialPort1->readAll();
            ui.textEdit_Info->append(QString("串口1 收到: %1").arg(QString(data)));
        });
        ui.textEdit_Info->append(QString("串口1 已打开: %1").arg(m_portNameList[0]));
    } else {
        ui.textEdit_Info->append(QString("串口1 打开失败: %1").arg(m_serialPort1->errorString()));
    }

    // 打开串口2（如果有）
    if (m_portNameList.size() >= 2) {
        m_serialPort2 = new QSerialPort(this);
        m_serialPort2->setPortName(m_portNameList[1]);
        if (m_serialPort2->open(QIODevice::ReadWrite)) {
            m_serialPort2->setBaudRate(QSerialPort::Baud115200, QSerialPort::AllDirections);
            m_serialPort2->setDataBits(QSerialPort::Data8);
            m_serialPort2->setFlowControl(QSerialPort::NoFlowControl);
            m_serialPort2->setParity(QSerialPort::NoParity);
            m_serialPort2->setStopBits(QSerialPort::OneStop);
            connect(m_serialPort2, &QSerialPort::readyRead, this, [this]() {
                QByteArray data = m_serialPort2->readAll();
                ui.textEdit_Info->append(QString("串口2 收到: %1").arg(QString(data)));
            });
            ui.textEdit_Info->append(QString("串口2 已打开: %1").arg(m_portNameList[1]));
        } else {
            ui.textEdit_Info->append(QString("串口2 打开失败: %1").arg(m_serialPort2->errorString()));
        }
    }
}

void ScannerWindow::sendData(QSerialPort* port, const QString& data)
{
    if (!port || !port->isOpen()) {
        ui.textEdit_Info->append("串口未打开，无法发送");
        return;
    }
    QByteArray bytes = data.toUtf8();
    qint64 written = port->write(bytes);
    if (written == -1) {
        ui.textEdit_Info->append(QString("发送失败: %1").arg(port->errorString()));
    } else {
        ui.textEdit_Info->append(QString("发送 %1 字节: %2").arg(written).arg(data));
    }
}

QString ScannerWindow::buildStartCommand()
{
    // 协议: N10 [timestamp] [interval] H{freq} B{background} T1 V2 L{laser};
    int freq = ui.horizontalSlider_Freq->value();
    int bg = ui.horizontalSlider_Background_Lighting->value();
    int laser = ui.horizontalSlider_Laser_Lighting->value();

    QString cmd("N10 ");

    if (ui.checkBox_SetTimeStamp->isChecked()) {
        // 获取当前时间戳（微秒级）
        auto now = std::chrono::system_clock::now().time_since_epoch();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
        cmd.append(QString::number(us));
        cmd.append(" ");
        cmd.append(QString::number(ui.spinBox_Scanner->value()));
        cmd.append(" ");
    }

    cmd.append("H").append(QString::number(freq));
    cmd.append(" B").append(QString::number(bg));
    cmd.append(" T1");
    cmd.append(" V2");
    cmd.append(" L").append(QString::number(laser));
    cmd.append(";");

    return cmd;
}

QString ScannerWindow::buildStopCommand()
{
    return "N11 H0;";
}

// ============================================================================
// 相机操作
// ============================================================================
void ScannerWindow::onOpenScannerCamera()
{
    if (!m_cam) return;
    auto r = m_cam->open();
    ui.textEdit_Info->append(r.success ? "扫描仪相机已打开" :
        QString("打开失败: %1").arg(QString::fromStdString(r.message)));

    // 状态机: Init → DeviceReady
    if (r.success && m_appCtx && m_appCtx->stateMachine()) {
        m_appCtx->stateMachine()->transition(Scanner::EventType::DeviceConnected);
    }
}

void ScannerWindow::onCloseScannerCamera()
{
    if (!m_cam) return;
    m_cam->stopAsyncCapture();
    auto r = m_cam->close();
    ui.textEdit_Info->append(r.success ? "扫描仪相机已关闭" :
        QString("关闭失败: %1").arg(QString::fromStdString(r.message)));

    // 状态机: → Init
    if (r.success && m_appCtx && m_appCtx->stateMachine()) {
        m_appCtx->stateMachine()->transition(Scanner::EventType::DeviceDisconnected);
    }
}

void ScannerWindow::onStartScanner()
{
    if (!m_cam || !m_cam->isOpen()) {
        ui.textEdit_Info->append("请先打开相机");
        return;
    }

    int expose = ui.horizontalSlider_ExposeTime->value();
    m_cam->setExposure(expose);

    // 先发送串口命令给下位机
    QString startCmd = buildStartCommand();
    ui.textEdit_Info->append(QString("发送下位机命令: %1").arg(startCmd));
    sendData(m_serialPort1, startCmd);
    sendData(m_serialPort2, startCmd);

    // 相机采集 — 回调只做计数+入队，最轻量
    auto r = m_cam->startAsyncCapture([this](const Scanner::hal::StereoFrame& frame) {
        ++m_frameCount;
        Scanner::data::FrameData fd;
        fd.frameId = frame.frameId;
        fd.timestamp = frame.timestamp;
        fd.leftGray = frame.leftGray;
        fd.rightGray = frame.rightGray;
        m_frameBuffer->pushFrame(fd);
    });

    if (!r.success) {
        ui.textEdit_Info->append(QString("启动失败: %1").arg(QString::fromStdString(r.message)));
        return;
    }

    // 启动 ScanWorkflow（暂时禁用，定位崩溃原因）
    // if (m_appCtx && m_appCtx->scanWorkflow()) {
    //     m_appCtx->scanWorkflow()->initialize();
    //     auto wfR = m_appCtx->scanWorkflow()->start();
    // }

    // 启动 HardwareMonitor（周期采集温度/帧率）
    if (m_appCtx && m_appCtx->hwMonitor()) {
        m_appCtx->hwMonitor()->setFrameCounter([this]() {
            return static_cast<int>(m_currentFps);
        });
        m_appCtx->hwMonitor()->start(1000);
    }

    ui.textEdit_Info->append(QString("采集已启动 (曝光 %1 ms)").arg(expose));
}

void ScannerWindow::onStopScanner()
{
    if (!m_cam) return;

    // 先停止 ScanWorkflow
    if (m_appCtx && m_appCtx->scanWorkflow()) {
        m_appCtx->scanWorkflow()->stop();
        ui.textEdit_Info->append("扫描管线已停止");
    }

    // 停止 HardwareMonitor
    if (m_appCtx && m_appCtx->hwMonitor()) {
        m_appCtx->hwMonitor()->stop();
    }

    // 发送停止命令给下位机
    QString stopCmd = buildStopCommand();
    ui.textEdit_Info->append(QString("发送下位机命令: %1").arg(stopCmd));
    sendData(m_serialPort1, stopCmd);
    sendData(m_serialPort2, stopCmd);

    // 再停止相机采集
    auto r = m_cam->stopAsyncCapture();
    ui.textEdit_Info->append(r.success ? "采集已停止" :
        QString("停止失败: %1").arg(QString::fromStdString(r.message)));
}

// ============================================================================
// 标定
// ============================================================================
void ScannerWindow::onCalibrateClicked()
{
    if (!m_appCtx || !m_appCtx->calibWorkflow()) {
        ui.textEdit_Info->append("标定工作流不可用");
        return;
    }
    if (!m_cam || !m_cam->isOpen()) {
        ui.textEdit_Info->append("请先打开相机");
        return;
    }

    ui.textEdit_Info->append("开始标定流程...");
    auto* calib = m_appCtx->calibWorkflow();
    calib->initialize();
    calib->setProgressCallback([this](const Scanner::workflow::WorkflowProgress& p) {
        QMetaObject::invokeMethod(this, [this, p]() {
            ui.textEdit_Info->append(QString("[标定] %1 (%2%)")
                .arg(QString::fromStdString(p.stageName))
                .arg(static_cast<int>(p.progress * 100)));
        });
    });

    // 启动相机采集供标定使用
    if (!m_cam->isCapturing()) {
        m_cam->startAsyncCapture([this](const Scanner::hal::StereoFrame& frame) {
            Scanner::data::FrameData fd;
            fd.frameId = frame.frameId;
            fd.timestamp = frame.timestamp;
            fd.leftGray = frame.leftGray;
            fd.rightGray = frame.rightGray;
            m_frameBuffer->pushFrame(fd);
        });
    }

    calib->start();
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
    if (m_cam && m_cam->isOpen()) {
        m_cam->setExposure(v);
        ui.textEdit_Info->append(QString("曝光设置: %1 ms").arg(v));
    }
}

void ScannerWindow::onResolutionChanged(int index)
{
    QSize res = m_resCombo->itemData(index).toSize();
    m_pendingWidth = res.width();
    m_pendingHeight = res.height();

    if (m_cam && m_cam->isOpen()) {
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

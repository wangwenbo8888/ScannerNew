#include "IntegrateTestDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFont>
#include <spdlog/spdlog.h>

// ============================================================================
// 构造 / 析构
// ============================================================================
IntegrateTestDialog::IntegrateTestDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("集成测试 - Scanner Framework"));
    resize(1400, 900);
    setMinimumSize(1200, 800);

    setupUILayout();

    // FPS 定时器（1 秒刷新）
    m_fpsTimer = new QTimer(this);
    connect(m_fpsTimer, &QTimer::timeout, this, [this]() {
        m_labelScannerLeftFps->setText(QString::number(m_leftFrameCount));
        m_labelScannerRightFps->setText(QString::number(m_rightFrameCount));
        m_leftFrameCount = 0;
        m_rightFrameCount = 0;
    });
    m_fpsTimer->start(1000);
}

IntegrateTestDialog::~IntegrateTestDialog()
{
    if (m_fpsTimer) m_fpsTimer->stop();
}

// ============================================================================
// 设置 CameraControl
// ============================================================================
void IntegrateTestDialog::setCameraControl(Scanner::device::CameraControl* cam)
{
    m_cam = cam;

    if (m_cam) {
        m_textEditInfo->append(QString("相机已绑定: %1").arg(
            QString::fromStdString(m_cam->getDeviceName())));

        // 注册异步回调
        m_cam->startAsyncCapture([this](const Scanner::hal::StereoFrame& frame) {
            QMetaObject::invokeMethod(this, [this, frame]() {
                onFrameReady();
            });
        });
    }
}

// ============================================================================
// 帧回调（从 CameraControl 异步线程触发，通过 invokeMethod 回到主线程）
// ============================================================================
void IntegrateTestDialog::onFrameReady()
{
    if (!m_cam) return;

    // 获取最新帧（同步方式从缓冲读取）
    Scanner::hal::StereoFrame frame;
    auto r = m_cam->grabFrame(frame, 100);
    if (!r.success) return;

    ++m_successCount;
    m_labelSuccessCount->setText(QString::number(m_successCount));

    // 显示左图
    if (!frame.leftGray.empty()) {
        QImage qimg(frame.leftGray.data, frame.leftGray.cols, frame.leftGray.rows,
                    frame.leftGray.step, QImage::Format_Grayscale8);
        QPixmap pix = QPixmap::fromImage(qimg).scaled(
            m_labelLeftImage->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        m_labelLeftImage->setPixmap(pix);
        ++m_leftFrameCount;
    }

    // 显示右图
    if (!frame.rightGray.empty()) {
        QImage qimg(frame.rightGray.data, frame.rightGray.cols, frame.rightGray.rows,
                    frame.rightGray.step, QImage::Format_Grayscale8);
        QPixmap pix = QPixmap::fromImage(qimg).scaled(
            m_labelRightImage->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        m_labelRightImage->setPixmap(pix);
        ++m_rightFrameCount;
    }
}

// ============================================================================
// 按钮槽
// ============================================================================
void IntegrateTestDialog::onOpenCamera()
{
    if (!m_cam) {
        m_textEditInfo->append("[错误] 未绑定相机");
        return;
    }
    auto r = m_cam->open();
    m_textEditInfo->append(r.success ? "相机已打开" :
        QString("打开失败: %1").arg(QString::fromStdString(r.message)));
}

void IntegrateTestDialog::onCloseCamera()
{
    if (!m_cam) return;
    m_cam->stopAsyncCapture();
    auto r = m_cam->close();
    m_textEditInfo->append(r.success ? "相机已关闭" :
        QString("关闭失败: %1").arg(QString::fromStdString(r.message)));
}

void IntegrateTestDialog::onStartCapture()
{
    if (!m_cam) return;

    // 应用曝光参数
    double exp = m_sliderExposeTime->value();
    m_cam->setExposure(exp);

    auto r = m_cam->startCapture();
    m_textEditInfo->append(r.success ?
        QString("采集已启动 (曝光 %1 ms)").arg(exp) :
        QString("启动失败: %1").arg(QString::fromStdString(r.message)));
}

void IntegrateTestDialog::onStopCapture()
{
    if (!m_cam) return;
    auto r = m_cam->stopCapture();
    m_textEditInfo->append(r.success ? "采集已停止" :
        QString("停止失败: %1").arg(QString::fromStdString(r.message)));
}

// ============================================================================
// UI 布局
// ============================================================================
void IntegrateTestDialog::setupUILayout()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(8, 8, 8, 8);
    mainLayout->setSpacing(8);

    QHBoxLayout *contentLayout = new QHBoxLayout();
    contentLayout->setSpacing(8);

    QVBoxLayout *leftLayout = new QVBoxLayout();
    leftLayout->setSpacing(4);
    leftLayout->addWidget(createControlPanel(), 0);
    leftLayout->addWidget(createParameterPanel(), 0);
    leftLayout->addWidget(createCalibPanel(), 0);
    leftLayout->addStretch(1);

    contentLayout->addLayout(leftLayout, 0);
    contentLayout->addWidget(createImagePanel(), 1);

    mainLayout->addLayout(contentLayout, 1);
}

QWidget* IntegrateTestDialog::createControlPanel()
{
    QGroupBox *group = new QGroupBox(QStringLiteral("设备控制"));
    group->setMaximumHeight(200);

    QVBoxLayout *layout = new QVBoxLayout(group);

    m_textEditInfo = new QTextEdit();
    m_textEditInfo->setReadOnly(true);
    m_textEditInfo->setMaximumHeight(100);
    layout->addWidget(m_textEditInfo);

    QHBoxLayout *btnRow = new QHBoxLayout();
    m_btnOpen  = new QPushButton(QStringLiteral("打开相机"));
    m_btnClose = new QPushButton(QStringLiteral("关闭相机"));
    m_btnStart = new QPushButton(QStringLiteral("开始采集"));
    m_btnStop  = new QPushButton(QStringLiteral("停止采集"));
    btnRow->addWidget(m_btnOpen);
    btnRow->addWidget(m_btnClose);
    btnRow->addWidget(m_btnStart);
    btnRow->addWidget(m_btnStop);
    layout->addLayout(btnRow);

    connect(m_btnOpen,  &QPushButton::clicked, this, &IntegrateTestDialog::onOpenCamera);
    connect(m_btnClose, &QPushButton::clicked, this, &IntegrateTestDialog::onCloseCamera);
    connect(m_btnStart, &QPushButton::clicked, this, &IntegrateTestDialog::onStartCapture);
    connect(m_btnStop,  &QPushButton::clicked, this, &IntegrateTestDialog::onStopCapture);

    return group;
}

QWidget* IntegrateTestDialog::createParameterPanel()
{
    QGroupBox *group = new QGroupBox(QStringLiteral("参数设置"));
    group->setMaximumHeight(200);

    QGridLayout *layout = new QGridLayout(group);

    // 采集频率
    layout->addWidget(new QLabel(QStringLiteral("采集频率")), 0, 0);
    m_sliderFreq = new QSlider(Qt::Horizontal);
    m_sliderFreq->setRange(20, 100);
    m_sliderFreq->setSingleStep(10);
    m_sliderFreq->setValue(50);
    layout->addWidget(m_sliderFreq, 0, 1);
    m_labelFreqValue = new QLabel("50");
    m_labelFreqValue->setFixedWidth(40);
    layout->addWidget(m_labelFreqValue, 0, 2);
    layout->addWidget(new QLabel("HZ"), 0, 3);

    // 补光灯亮度
    layout->addWidget(new QLabel(QStringLiteral("补光灯亮度")), 1, 0);
    m_sliderBackground = new QSlider(Qt::Horizontal);
    m_sliderBackground->setRange(1, 100);
    m_sliderBackground->setSingleStep(10);
    m_sliderBackground->setValue(40);
    layout->addWidget(m_sliderBackground, 1, 1);
    m_labelBackgroundValue = new QLabel("40");
    m_labelBackgroundValue->setFixedWidth(40);
    layout->addWidget(m_labelBackgroundValue, 1, 2);

    // 激光亮度
    layout->addWidget(new QLabel(QStringLiteral("激光亮度")), 2, 0);
    m_sliderLaserLighting = new QSlider(Qt::Horizontal);
    m_sliderLaserLighting->setRange(1, 100);
    m_sliderLaserLighting->setSingleStep(5);
    m_sliderLaserLighting->setValue(60);
    layout->addWidget(m_sliderLaserLighting, 2, 1);
    m_labelLaserLightingValue = new QLabel("60");
    m_labelLaserLightingValue->setFixedWidth(40);
    layout->addWidget(m_labelLaserLightingValue, 2, 2);

    // 曝光时间
    layout->addWidget(new QLabel(QStringLiteral("曝光时间")), 3, 0);
    m_sliderExposeTime = new QSlider(Qt::Horizontal);
    m_sliderExposeTime->setRange(1, 100);
    m_sliderExposeTime->setValue(5);
    layout->addWidget(m_sliderExposeTime, 3, 1);
    m_labelExposeTimeValue = new QLabel("5");
    m_labelExposeTimeValue->setFixedWidth(40);
    layout->addWidget(m_labelExposeTimeValue, 3, 2);
    layout->addWidget(new QLabel("MS"), 3, 3);

    connect(m_sliderExposeTime, &QSlider::valueChanged, this, [this](int v) {
        m_labelExposeTimeValue->setText(QString::number(v));
        if (m_cam && m_cam->isOpen()) {
            m_cam->setExposure(v);
        }
    });
    connect(m_sliderFreq, &QSlider::valueChanged, this, [this](int v) {
        m_labelFreqValue->setText(QString::number(v));
    });
    connect(m_sliderBackground, &QSlider::valueChanged, this, [this](int v) {
        m_labelBackgroundValue->setText(QString::number(v));
    });
    connect(m_sliderLaserLighting, &QSlider::valueChanged, this, [this](int v) {
        m_labelLaserLightingValue->setText(QString::number(v));
    });

    // FPS
    QGridLayout *fpsLayout = new QGridLayout();
    fpsLayout->addWidget(new QLabel(QStringLiteral("扫描仪左相机帧率")), 0, 0);
    m_labelScannerLeftFps = new QLabel("0");
    fpsLayout->addWidget(m_labelScannerLeftFps, 0, 1);
    fpsLayout->addWidget(new QLabel("HZ"), 0, 2);
    fpsLayout->addWidget(new QLabel(QStringLiteral("扫描仪右相机帧率")), 1, 0);
    m_labelScannerRightFps = new QLabel("0");
    fpsLayout->addWidget(m_labelScannerRightFps, 1, 1);
    fpsLayout->addWidget(new QLabel("HZ"), 1, 2);

    QHBoxLayout *paramAndFps = new QHBoxLayout();
    paramAndFps->addLayout(layout);
    paramAndFps->addLayout(fpsLayout);

    QVBoxLayout *outerLayout = new QVBoxLayout(group);
    outerLayout->addLayout(paramAndFps);

    return group;
}

QWidget* IntegrateTestDialog::createImagePanel()
{
    QGroupBox *group = new QGroupBox(QStringLiteral("图像显示"));
    QVBoxLayout *layout = new QVBoxLayout(group);

    QHBoxLayout *imageLayout = new QHBoxLayout();

    QVBoxLayout *leftImageLayout = new QVBoxLayout();
    leftImageLayout->addWidget(new QLabel(QStringLiteral("左相机图像")));
    m_labelLeftImage = new QLabel();
    m_labelLeftImage->setMinimumSize(400, 300);
    m_labelLeftImage->setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;");
    m_labelLeftImage->setAlignment(Qt::AlignCenter);
    m_labelLeftImage->setText(QStringLiteral("Left Image"));
    leftImageLayout->addWidget(m_labelLeftImage);
    imageLayout->addLayout(leftImageLayout);

    QVBoxLayout *rightImageLayout = new QVBoxLayout();
    rightImageLayout->addWidget(new QLabel(QStringLiteral("右相机图像")));
    m_labelRightImage = new QLabel();
    m_labelRightImage->setMinimumSize(400, 300);
    m_labelRightImage->setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;");
    m_labelRightImage->setAlignment(Qt::AlignCenter);
    m_labelRightImage->setText(QStringLiteral("Right Image"));
    rightImageLayout->addWidget(m_labelRightImage);
    imageLayout->addLayout(rightImageLayout);

    layout->addLayout(imageLayout, 1);

    QHBoxLayout *successRow = new QHBoxLayout();
    successRow->addWidget(new QLabel(QStringLiteral("成功计数")));
    m_labelSuccessCount = new QLabel("0");
    QFont font = m_labelSuccessCount->font();
    font.setPointSize(24);
    font.setBold(true);
    m_labelSuccessCount->setFont(font);
    successRow->addWidget(m_labelSuccessCount);
    successRow->addStretch();
    layout->addLayout(successRow);

    return group;
}

QWidget* IntegrateTestDialog::createCalibPanel()
{
    QGroupBox *group = new QGroupBox(QStringLiteral("标定操作"));
    group->setMaximumHeight(80);

    QHBoxLayout *layout = new QHBoxLayout(group);
    QPushButton *btnTakePhoto = new QPushButton(QStringLiteral("采集相机标定图像"));
    QPushButton *btnCalib = new QPushButton(QStringLiteral("计算相机标定参数"));
    QPushButton *btnTakeLaserPhoto = new QPushButton(QStringLiteral("采集激光线标定图像"));
    QPushButton *btnCalibLaser = new QPushButton(QStringLiteral("计算激光线标定参数"));

    layout->addWidget(btnTakePhoto);
    layout->addWidget(btnCalib);
    layout->addWidget(btnTakeLaserPhoto);
    layout->addWidget(btnCalibLaser);

    // TODO: 接入标定工作流
    connect(btnTakePhoto, &QPushButton::clicked, this, [this]() {
        m_textEditInfo->append("TODO: 采集相机标定图像");
    });
    connect(btnCalib, &QPushButton::clicked, this, [this]() {
        m_textEditInfo->append("TODO: 计算相机标定参数");
    });
    connect(btnTakeLaserPhoto, &QPushButton::clicked, this, [this]() {
        m_textEditInfo->append("TODO: 采集激光线标定图像");
    });
    connect(btnCalibLaser, &QPushButton::clicked, this, [this]() {
        m_textEditInfo->append("TODO: 计算激光线标定参数");
    });

    return group;
}
